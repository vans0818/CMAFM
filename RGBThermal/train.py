"""Training script for Multispectral Object Detection."""

import argparse
import multiprocessing
import os
import sys
import time
from pathlib import Path

# Must be set before any child processes are spawned
os.environ["PYTHONUTF8"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from config import Config
from dataset import build_datasets, collate_fn
from evaluate import evaluate_model
from model import build_model


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """Linear warmup scheduler."""
    def f(x):
        if x >= warmup_iters:
            return 1.0
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_one_epoch(model, optimizer, data_loader, device, epoch, cfg, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Warmup scheduler for first epoch
    lr_scheduler_warmup = None
    if epoch == 0 and cfg.train.warmup_epochs > 0:
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler_warmup = warmup_lr_scheduler(optimizer, warmup_iters, cfg.train.warmup_factor)

    t0 = time.time()
    for i, (rgb, thermal, targets) in enumerate(data_loader):
        rgb = rgb.to(device)
        thermal = thermal.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Skip batches with no valid targets
        valid = all(t["boxes"].shape[0] > 0 for t in targets)
        if not valid:
            continue

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(rgb, thermal, targets)
            losses = sum(loss for loss in loss_dict.values())

        if not torch.isfinite(losses):
            print(f"WARNING: Non-finite loss {losses.item()}, skipping batch {i}")
            continue

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        if lr_scheduler_warmup is not None:
            lr_scheduler_warmup.step()

        total_loss += losses.item()
        num_batches += 1

        if (i + 1) % cfg.train.print_freq == 0:
            avg_loss = total_loss / num_batches
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            print(f"  [{i+1}/{len(data_loader)}] "
                  f"loss={avg_loss:.4f}  lr={lr:.6f}  "
                  f"({elapsed:.1f}s)", flush=True)

    avg_loss = total_loss / max(num_batches, 1)
    elapsed = time.time() - t0
    print(f"Epoch {epoch} done — avg_loss={avg_loss:.4f}  time={elapsed:.1f}s")
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train Multispectral Detector")
    parser.add_argument("--data-root", type=str, default=None, help="Override data root path")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--backbone", type=str, default=None, choices=["resnet50", "resnet101"])
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    args = parser.parse_args()

    # Build config
    cfg = Config()
    if args.data_root:
        cfg.data.root = args.data_root
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.lr:
        cfg.train.lr = args.lr
    if args.backbone:
        cfg.model.backbone = args.backbone
    if args.resume:
        cfg.train.resume = args.resume
    if args.no_amp:
        cfg.train.amp = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: backbone={cfg.model.backbone}, epochs={cfg.train.epochs}, "
          f"batch_size={cfg.train.batch_size}, lr={cfg.train.lr}, amp={cfg.train.amp}")

    # Dataset
    train_dataset, val_dataset = build_datasets(cfg.data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Model
    model = build_model(cfg.model, num_classes=cfg.data.num_classes)
    model.to(device)

    # Optimizer: different LR for pretrained backbone vs fusion/head
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "rgb_stages" in name or "thermal_stages" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.SGD([
        {"params": backbone_params, "lr": cfg.train.lr * 0.1},  # lower LR for pretrained
        {"params": other_params, "lr": cfg.train.lr},
    ], momentum=cfg.train.momentum, weight_decay=cfg.train.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.train.lr_step_size, gamma=cfg.train.lr_gamma
    )

    scaler = torch.cuda.amp.GradScaler() if cfg.train.amp else None

    start_epoch = 0
    best_map = 0.0

    # Resume
    if cfg.train.resume:
        print(f"Resuming from {cfg.train.resume}")
        ckpt = torch.load(cfg.train.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_map = ckpt.get("best_map", 0.0)
        if scaler and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])

    # Eval only mode
    if args.eval_only:
        metrics = evaluate_model(model, val_loader, device, cfg.data)
        print(f"\nmAP@0.5: {metrics['mAP50']:.4f}  mAP@[.5:.95]: {metrics['mAP50_95']:.4f}")
        return

    # Create save directory
    save_dir = Path(cfg.train.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training: {cfg.train.epochs} epochs")
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, cfg.train.epochs):
        print(f"\n--- Epoch {epoch}/{cfg.train.epochs - 1} ---")
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, cfg, scaler)
        lr_scheduler.step()

        # Evaluate
        metrics = evaluate_model(model, val_loader, device, cfg.data)
        mAP50 = metrics["mAP50"]
        mAP50_95 = metrics["mAP50_95"]
        print(f"  Val mAP@0.5={mAP50:.4f}  mAP@[.5:.95]={mAP50_95:.4f}")

        # Save checkpoint
        is_best = mAP50 > best_map
        if is_best:
            best_map = mAP50

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "best_map": best_map,
            "config": cfg,
        }
        if scaler:
            ckpt["scaler"] = scaler.state_dict()

        if is_best:
            torch.save(ckpt, save_dir / "best.pth")
            print(f"  ★ New best mAP@0.5: {best_map:.4f}")

        if (epoch + 1) % cfg.train.save_every == 0 or epoch == cfg.train.epochs - 1:
            torch.save(ckpt, save_dir / f"epoch_{epoch:03d}.pth")

    print(f"\nTraining complete. Best mAP@0.5: {best_map:.4f}")
    print(f"Checkpoints saved to: {save_dir}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
