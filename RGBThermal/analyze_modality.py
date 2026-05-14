"""
RGB-only / Thermal-only / Fused(RGB+Thermal) 모달리티별 정량적 평가 스크립트.
Precision, Recall, F1, Miss Rate, FPS, mAP 전체 지표 산출.
"""

import os
os.environ["PYTHONUTF8"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from PIL import Image

from config import Config
from dataset import build_datasets, collate_fn
from evaluate import evaluate_model
from model import build_model
from ablation_models import build_ablation_model

ABLATION_DIR = Path(__file__).parent / "runs" / "ablation"
FULL_CKPT    = Path(__file__).parent / "runs" / "best.pth"
OUT_PATH     = Path(__file__).parent / "runs" / "modality_analysis.json"
BATCH_SIZE   = 2
BRIGHTNESS_THRESH = 60


def classify_night_day(val_dataset, vis_dir):
    night_idx, day_idx = [], []
    for i, img_id in enumerate(val_dataset.samples):
        arr = np.array(Image.open(vis_dir / f"{img_id}.png").convert("L"))
        if arr.mean() < BRIGHTNESS_THRESH:
            night_idx.append(i)
        else:
            day_idx.append(i)
    return night_idx, day_idx


def run_eval(model, dataset, indices, device, cfg, label):
    subset = Subset(dataset, indices) if indices is not None else dataset
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)
    print(f"    → {label} ({len(subset)}장) 평가 중...", flush=True)
    return evaluate_model(model, loader, device, cfg.data)


def print_metrics(r, label):
    print(f"  {'지표':<14} {label}")
    print(f"  {'mAP@0.5':<14} {r['mAP50']:.4f}")
    print(f"  {'mAP@[.5:.95]':<14} {r['mAP50_95']:.4f}")
    print(f"  {'Precision':<14} {r['precision']:.4f}")
    print(f"  {'Recall':<14} {r['recall']:.4f}")
    print(f"  {'F1-Score':<14} {r['f1']:.4f}")
    print(f"  {'Miss Rate':<14} {r['miss_rate']:.4f}")
    if 'fps' in r:
        print(f"  {'FPS':<14} {r['fps']:.1f}")
    print()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    cfg = Config()
    _, val_dataset = build_datasets(cfg.data)
    vis_dir = Path(__file__).parent / "data" / "M3FD" / "Vis"

    print("야간/주간 분류 중...")
    night_idx, day_idx = classify_night_day(val_dataset, vis_dir)
    all_idx = list(range(len(val_dataset)))
    print(f"  전체: {len(all_idx)}장 / 야간: {len(night_idx)}장 / 주간: {len(day_idx)}장\n")

    # ── 평가할 모델 정의 ───────────────────────────────────────────────────────
    variants = [
        ("RGB-only",        "rgb_only",     ABLATION_DIR / "rgb_only_best.pth",     "rgb_only"),
        ("Thermal-only",    "thermal_only", ABLATION_DIR / "thermal_only_best.pth", "thermal_only"),
        ("Early Fusion",    "early_fusion", ABLATION_DIR / "early_fusion_best.pth", "early_fusion"),
        ("Dual+Concat",     "dual_no_attn", ABLATION_DIR / "dual_no_attn_best.pth", "dual_no_attn"),
        ("Fused (Ours)",    "full",         FULL_CKPT,                               None),
    ]

    all_results = {}

    for display_name, key, ckpt_path, variant_name in variants:
        print("=" * 60)
        print(f"[{display_name}]  체크포인트: {ckpt_path.name}")
        print("=" * 60)

        # 모델 로드
        if variant_name is None:
            model = build_model(cfg.model, num_classes=cfg.data.num_classes)
        else:
            model = build_ablation_model(variant_name, cfg.model,
                                         num_classes=cfg.data.num_classes)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # ablation 체크포인트는 state_dict 직접 저장 (dict 래핑 없음)
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        cond_results = {}

        # 전체
        cond_results["all"]   = run_eval(model, val_dataset, all_idx,   device, cfg, "전체")
        print_metrics(cond_results["all"], "전체")

        # 야간
        cond_results["night"] = run_eval(model, val_dataset, night_idx, device, cfg, "야간")
        print_metrics(cond_results["night"], "야간")

        # 주간
        cond_results["day"]   = run_eval(model, val_dataset, day_idx,   device, cfg, "주간")
        print_metrics(cond_results["day"], "주간")

        all_results[key] = {
            "display_name": display_name,
            "results": cond_results,
        }

        del model
        torch.cuda.empty_cache()

    # 저장
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {OUT_PATH}")

    # 최종 요약 출력
    print("\n" + "=" * 90)
    print("최종 요약 — 모달리티별 전체/야간/주간 성능")
    print("=" * 90)
    metrics = [("mAP@0.5","mAP50"), ("Recall","recall"), ("F1","f1"), ("MissRate","miss_rate")]
    conds = [("전체","all"), ("야간","night"), ("주간","day")]

    header = f"{'모델':<18}"
    for cond_name, _ in conds:
        for m_name, _ in metrics:
            header += f" {cond_name+'/'+m_name:>13}"
    print(header)
    print("-" * 90)

    for key, data in all_results.items():
        row = f"{data['display_name']:<18}"
        for _, cond_key in conds:
            r = data["results"][cond_key]
            for _, m_key in metrics:
                row += f" {r.get(m_key, 0):>13.4f}"
        print(row)


if __name__ == "__main__":
    main()
