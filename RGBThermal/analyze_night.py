"""야간/주간 조건별 성능 분석 스크립트."""

import os
os.environ["PYTHONUTF8"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from dataset import build_datasets, collate_fn
from evaluate import evaluate_model
from model import build_model

CHECKPOINT = str(Path(__file__).parent / "runs" / "best.pth")
BRIGHTNESS_THRESHOLD = 60  # RGB 평균 밝기 기준 (< 60 = 야간)
BATCH_SIZE = 2


def classify_by_brightness(dataset, vis_dir: Path, threshold: float):
    """데이터셋 인덱스를 야간/주간으로 분류."""
    night_idx, day_idx = [], []
    for i, img_id in enumerate(dataset.samples):
        img_path = vis_dir / f"{img_id}.png"
        arr = np.array(Image.open(img_path).convert("L"))
        brightness = arr.mean()
        if brightness < threshold:
            night_idx.append(i)
        else:
            day_idx.append(i)
    return night_idx, day_idx


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT}\n")

    # 데이터셋 로드 (val set만 사용)
    _, val_dataset = build_datasets(cfg.data)
    vis_dir = Path(__file__).parent / "data" / "M3FD" / "Vis"

    print("이미지 밝기 분석 중...")
    night_idx, day_idx = classify_by_brightness(val_dataset, vis_dir, BRIGHTNESS_THRESHOLD)
    print(f"  Val 전체: {len(val_dataset)}장")
    print(f"  야간 (밝기 < {BRIGHTNESS_THRESHOLD}): {len(night_idx)}장")
    print(f"  주간 (밝기 >= {BRIGHTNESS_THRESHOLD}): {len(day_idx)}장\n")

    # 모델 로드
    model = build_model(cfg.model, num_classes=cfg.data.num_classes)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    print("모델 로드 완료\n")

    results = {}

    # 전체 val 평가
    print("=" * 50)
    print("[1/3] 전체 Val 세트 평가")
    print("=" * 50)
    all_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)
    results["all"] = evaluate_model(model, all_loader, device, cfg.data)
    r = results["all"]
    print(f"  mAP@0.5:      {r['mAP50']:.4f}")
    print(f"  mAP@[.5:.95]: {r['mAP50_95']:.4f}")
    print(f"  Precision:    {r['precision']:.4f}")
    print(f"  Recall:       {r['recall']:.4f}")
    print(f"  F1-Score:     {r['f1']:.4f}")
    print(f"  Miss Rate:    {r['miss_rate']:.4f}")
    print(f"  FPS:          {r['fps']:.1f}\n")

    # 야간 평가
    if len(night_idx) > 0:
        print("=" * 50)
        print(f"[2/3] 야간 세트 평가 ({len(night_idx)}장)")
        print("=" * 50)
        night_loader = DataLoader(Subset(val_dataset, night_idx), batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=0, collate_fn=collate_fn)
        results["night"] = evaluate_model(model, night_loader, device, cfg.data)
        r = results["night"]
        print(f"  mAP@0.5:      {r['mAP50']:.4f}")
        print(f"  mAP@[.5:.95]: {r['mAP50_95']:.4f}")
        print(f"  Precision:    {r['precision']:.4f}")
        print(f"  Recall:       {r['recall']:.4f}")
        print(f"  F1-Score:     {r['f1']:.4f}")
        print(f"  Miss Rate:    {r['miss_rate']:.4f}\n")

    # 주간 평가
    if len(day_idx) > 0:
        print("=" * 50)
        print(f"[3/3] 주간 세트 평가 ({len(day_idx)}장)")
        print("=" * 50)
        day_loader = DataLoader(Subset(val_dataset, day_idx), batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=0, collate_fn=collate_fn)
        results["day"] = evaluate_model(model, day_loader, device, cfg.data)
        r = results["day"]
        print(f"  mAP@0.5:      {r['mAP50']:.4f}")
        print(f"  mAP@[.5:.95]: {r['mAP50_95']:.4f}")
        print(f"  Precision:    {r['precision']:.4f}")
        print(f"  Recall:       {r['recall']:.4f}")
        print(f"  F1-Score:     {r['f1']:.4f}")
        print(f"  Miss Rate:    {r['miss_rate']:.4f}\n")

    # 최종 요약
    print("\n" + "=" * 70)
    print("최종 결과 요약")
    print("=" * 70)
    classes = ["People", "Car", "Bus", "Motorcycle", "Lamp", "Truck"]
    print(f"{'조건':<10} {'mAP@0.5':>9} {'mAP@.5:.95':>12} {'Precision':>11} {'Recall':>8} {'F1':>8} {'MissRate':>10}")
    print("-" * 70)
    for cond, label in [("all", "전체"), ("night", "야간"), ("day", "주간")]:
        if cond in results:
            r = results[cond]
            print(f"{label:<10} {r['mAP50']:>9.4f} {r['mAP50_95']:>12.4f} "
                  f"{r['precision']:>11.4f} {r['recall']:>8.4f} "
                  f"{r['f1']:>8.4f} {r['miss_rate']:>10.4f}")

    print("\n클래스별 AP@0.5 / Recall / Miss Rate (야간 vs 주간)")
    print("-" * 75)
    print(f"{'클래스':<14}", end="")
    for cond, label in [("all", "전체"), ("night", "야간"), ("day", "주간")]:
        if cond in results:
            print(f"{label+' AP':>10}{label+' Rec':>10}{label+' MR':>9}", end="")
    print()
    for cls in classes:
        print(f"{cls:<14}", end="")
        for cond in ["all", "night", "day"]:
            if cond in results:
                ap  = results[cond].get(f"AP50_{cls}", -1)
                rec = results[cond].get(f"recall_{cls}", -1)
                mr  = results[cond].get(f"miss_rate_{cls}", -1)
                ap_s  = f"{ap:.4f}"  if ap  >= 0 else "  —  "
                rec_s = f"{rec:.4f}" if rec >= 0 else "  —  "
                mr_s  = f"{mr:.4f}"  if mr  >= 0 else "  —  "
                print(f"{ap_s:>10}{rec_s:>10}{mr_s:>9}", end="")
        print()

    # 결과 저장
    out_path = Path(__file__).parent / "runs" / "night_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "brightness_threshold": BRIGHTNESS_THRESHOLD,
            "night_count": len(night_idx),
            "day_count": len(day_idx),
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
