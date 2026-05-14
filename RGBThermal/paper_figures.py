"""
논문용 Figure 생성 스크립트.
야간/주간별 RGB · Thermal · Fused feature · Detection 결과를 비교 시각화.
"""

import os
os.environ["PYTHONUTF8"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정 (Windows 맑은 고딕)
_font_candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic", "DejaVu Sans"]
_available = {f.name for f in fm.fontManager.ttflist}
_korean_font = next((f for f in _font_candidates if f in _available), None)
if _korean_font:
    matplotlib.rc("font", family=_korean_font)
matplotlib.rc("axes", unicode_minus=False)
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from PIL import Image

from config import Config
from dataset import build_datasets
from model import build_model

# ── 설정 ─────────────────────────────────────────────────────────────────────
CHECKPOINT   = Path(__file__).parent / "runs" / "best.pth"
OUT_DIR      = Path(__file__).parent / "runs" / "paper_figures"
IMG_SIZE     = (640, 640)
SCORE_THRESH = 0.45
BRIGHTNESS_THRESH = 60   # < 60 → 야간

CLASS_NAMES  = ["", "People", "Car", "Bus", "Motorcycle", "Lamp", "Truck"]
CLASS_COLORS = {          # RGB 0-1
    1: (0.2, 0.8, 0.2),   # People  - green
    2: (0.2, 0.4, 1.0),   # Car     - blue
    3: (1.0, 0.4, 0.0),   # Bus     - orange
    4: (0.8, 0.0, 0.8),   # Motorcycle - magenta
    5: (1.0, 1.0, 0.0),   # Lamp    - yellow
    6: (1.0, 0.2, 0.2),   # Truck   - red
}

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 유틸 ─────────────────────────────────────────────────────────────────────
def load_img(path: Path, mode="RGB"):
    return np.array(Image.open(path).convert(mode))

def to_tensor(arr, repeat3=False):
    t = torch.from_numpy(arr).float() / 255.0
    if arr.ndim == 2:
        t = t.unsqueeze(0)
        if repeat3:
            t = t.repeat(3, 1, 1)
    else:
        t = t.permute(2, 0, 1)
    return t.unsqueeze(0)

def preprocess(rgb_arr, th_arr):
    h, w = rgb_arr.shape[:2]
    rgb_r  = np.array(Image.fromarray(rgb_arr).resize((IMG_SIZE[1], IMG_SIZE[0])))
    th_r   = np.array(Image.fromarray(th_arr).resize((IMG_SIZE[1], IMG_SIZE[0])))
    rgb_t  = to_tensor(rgb_r)
    th_t   = to_tensor(th_r, repeat3=True)
    return rgb_t, th_t, h, w

def draw_boxes(ax, boxes, scores, labels, orig_h, orig_w, thresh=SCORE_THRESH):
    sh, sw = IMG_SIZE
    for box, sc, lb in zip(boxes, scores, labels):
        if sc < thresh:
            continue
        x1, y1, x2, y2 = box
        x1 = x1 / sw * orig_w;  x2 = x2 / sw * orig_w
        y1 = y1 / sh * orig_h;  y2 = y2 / sh * orig_h
        color = CLASS_COLORS.get(int(lb), (1,1,1))
        rect = mpatches.FancyBboxPatch(
            (x1, y1), x2-x1, y2-y1,
            boxstyle="square,pad=0", linewidth=2,
            edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, max(y1-4, 0), f"{CLASS_NAMES[int(lb)]} {sc:.2f}",
                fontsize=7, color="white",
                bbox=dict(facecolor=color, alpha=0.7, pad=1, edgecolor="none"))

def feat_to_heatmap(feat: torch.Tensor) -> np.ndarray:
    """(1,C,H,W) → H×W heatmap [0,1]."""
    f = feat[0].mean(0)
    f = (f - f.min()) / (f.max() - f.min() + 1e-8)
    return f.cpu().numpy()

# ── Hook 등록 ────────────────────────────────────────────────────────────────
class FeatureHook:
    def __init__(self):
        self.feats = {}

    def register(self, model):
        # RGB C4 (layer3 = ResNet stage3)
        model.backbone.rgb_stages.layer3.register_forward_hook(
            lambda m,i,o: self.feats.update({"rgb_c4": o}))
        # Thermal C4
        model.backbone.thermal_stages.layer3.register_forward_hook(
            lambda m,i,o: self.feats.update({"th_c4": o}))
        # Fused C4 (after fusion_c4 CMAFM)
        model.backbone.fusion_c4.register_forward_hook(
            lambda m,i,o: self.feats.update({"fused_c4": o[0] if isinstance(o,tuple) else o}))

# ── 샘플 선택 ────────────────────────────────────────────────────────────────
def pick_samples(val_dataset, vis_dir, n_night=3, n_day=3):
    night, day = [], []
    for i, img_id in enumerate(val_dataset.samples):
        arr = np.array(Image.open(vis_dir / f"{img_id}.png").convert("L"))
        br  = arr.mean()
        if br < BRIGHTNESS_THRESH and len(night) < n_night:
            night.append((i, img_id, br))
        elif br >= BRIGHTNESS_THRESH and len(day) < n_day:
            day.append((i, img_id, br))
        if len(night) >= n_night and len(day) >= n_day:
            break
    return night, day

# ── RGB 단독 추론 ─────────────────────────────────────────────────────────────
def run_rgb_only(model, hook, device, rgb_t, th_t):
    """Thermal 입력을 0으로 마스킹하여 RGB 단독 검출 결과를 얻는다."""
    th_zero = torch.zeros_like(th_t)
    hook.feats.clear()
    with torch.no_grad():
        outputs = model(rgb_t, th_zero)
    return outputs[0]


# ── Thermal 단독 추론 ─────────────────────────────────────────────────────────
def run_thermal_only(model, hook, device, rgb_t, th_t):
    """RGB 입력을 0으로 마스킹하여 Thermal 단독 검출 결과를 얻는다."""
    rgb_zero = torch.zeros_like(rgb_t)
    hook.feats.clear()
    with torch.no_grad():
        outputs = model(rgb_zero, th_t)
    return outputs[0]


# ── Figure 1: 야간/주간 단일 비교 패널 ────────────────────────────────────────
def make_comparison_figure(model, hook, device, samples, condition, vis_dir, ir_dir, cfg):
    """
    각 샘플에 대해 7열 패널 생성:
    (a) RGB | (b) Thermal | (c) RGB Feature(C4) | (d) Thermal Feature(C4)
    | (e) Fused Feature(C4) | (f) RGB 단독 검출[최초] | (g) RGB+Thermal 융합 검출[최종]
    """
    n = len(samples)
    fig, axes = plt.subplots(n, 8, figsize=(32, 4.5*n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        "(a) RGB 입력",
        "(b) Thermal 입력",
        "(c) RGB Backbone\nFeature (C4)",
        "(d) Thermal Backbone\nFeature (C4)",
        "(e) CMAFM 융합\nFeature (C4)",
        "(f) RGB 단독 검출",
        "(g) Thermal 단독 검출",
        "(h) RGB+Thermal 융합 검출",
    ]
    for ci, t in enumerate(col_titles):
        axes[0, ci].set_title(t, fontsize=10, fontweight="bold", pad=8)

    # (f)/(g)/(h) 열 색상 강조
    for ci, color in [(5, "#D9534F"), (6, "#F0A500"), (7, "#5CB85C")]:
        axes[0, ci].set_title(col_titles[ci], fontsize=10, fontweight="bold",
                              pad=8, color=color)

    for row, (idx, img_id, br) in enumerate(samples):
        rgb_arr = load_img(vis_dir / f"{img_id}.png", "RGB")
        th_arr  = load_img(ir_dir  / f"{img_id}.png", "L")
        orig_h, orig_w = rgb_arr.shape[:2]

        rgb_t, th_t, h, w = preprocess(rgb_arr, th_arr)
        rgb_t = rgb_t.to(device)
        th_t  = th_t.to(device)

        # RGB 단독 검출 — Thermal 마스킹
        det_rgb_only = run_rgb_only(model, hook, device, rgb_t, th_t)

        # Thermal 단독 검출 — RGB 마스킹
        det_th_only = run_thermal_only(model, hook, device, rgb_t, th_t)

        # 융합 검출 [최종]
        hook.feats.clear()
        with torch.no_grad():
            outputs = model(rgb_t, th_t)
        det_fused = outputs[0]

        # (a) RGB 입력
        axes[row, 0].imshow(rgb_arr)
        axes[row, 0].set_ylabel(
            f"{'야간' if condition=='night' else '주간'} #{row+1}\n(id:{img_id})",
            fontsize=9, rotation=0, labelpad=60, va="center")

        # (b) Thermal 입력
        axes[row, 1].imshow(th_arr, cmap="inferno")

        # (c) RGB Backbone Feature
        if "rgb_c4" in hook.feats:
            axes[row, 2].imshow(feat_to_heatmap(hook.feats["rgb_c4"]), cmap="jet")
        else:
            axes[row, 2].text(0.5, 0.5, "N/A", ha="center", va="center",
                              transform=axes[row, 2].transAxes)

        # (d) Thermal Backbone Feature
        if "th_c4" in hook.feats:
            axes[row, 3].imshow(feat_to_heatmap(hook.feats["th_c4"]), cmap="jet")
        else:
            axes[row, 3].text(0.5, 0.5, "N/A", ha="center", va="center",
                              transform=axes[row, 3].transAxes)

        # (e) CMAFM 융합 Feature
        if "fused_c4" in hook.feats:
            axes[row, 4].imshow(feat_to_heatmap(hook.feats["fused_c4"]), cmap="jet")
        else:
            axes[row, 4].text(0.5, 0.5, "N/A", ha="center", va="center",
                              transform=axes[row, 4].transAxes)

        # (f) RGB 단독 검출 — 빨간 테두리
        axes[row, 5].imshow(rgb_arr)
        draw_boxes(axes[row, 5],
                   det_rgb_only["boxes"].cpu().numpy(),
                   det_rgb_only["scores"].cpu().numpy(),
                   det_rgb_only["labels"].cpu().numpy(),
                   orig_h, orig_w)
        n_rgb = (det_rgb_only["scores"].cpu().numpy() >= SCORE_THRESH).sum()
        axes[row, 5].text(5, 20, f"Det: {n_rgb}", fontsize=8,
                          color="white", bbox=dict(facecolor="#D9534F", alpha=0.8))
        for spine in axes[row, 5].spines.values():
            spine.set_edgecolor("#D9534F")
            spine.set_linewidth(3)

        # (g) Thermal 단독 검출 — Thermal 영상 위에 표시, 주황 테두리
        axes[row, 6].imshow(th_arr, cmap="inferno")
        draw_boxes(axes[row, 6],
                   det_th_only["boxes"].cpu().numpy(),
                   det_th_only["scores"].cpu().numpy(),
                   det_th_only["labels"].cpu().numpy(),
                   orig_h, orig_w)
        n_th = (det_th_only["scores"].cpu().numpy() >= SCORE_THRESH).sum()
        axes[row, 6].text(5, 20, f"Det: {n_th}", fontsize=8,
                          color="white", bbox=dict(facecolor="#F0A500", alpha=0.8))
        for spine in axes[row, 6].spines.values():
            spine.set_edgecolor("#F0A500")
            spine.set_linewidth(3)

        # (h) RGB+Thermal 융합 검출 — 초록 테두리
        axes[row, 7].imshow(rgb_arr)
        draw_boxes(axes[row, 7],
                   det_fused["boxes"].cpu().numpy(),
                   det_fused["scores"].cpu().numpy(),
                   det_fused["labels"].cpu().numpy(),
                   orig_h, orig_w)
        n_fused = (det_fused["scores"].cpu().numpy() >= SCORE_THRESH).sum()
        axes[row, 7].text(5, 20, f"Det: {n_fused}", fontsize=8,
                          color="white", bbox=dict(facecolor="#5CB85C", alpha=0.8))
        for spine in axes[row, 7].spines.values():
            spine.set_edgecolor("#5CB85C")
            spine.set_linewidth(3)

        # (f)→(h) 변화량 표시
        delta = n_fused - n_rgb
        sign = "+" if delta >= 0 else ""
        axes[row, 7].text(5, orig_h - 25, f"Δ {sign}{delta} vs RGB-only",
                          fontsize=8, color="white",
                          bbox=dict(facecolor="#333333", alpha=0.7))

        for ax in axes[row]:
            ax.axis("off")

    # 범례
    handles = [mpatches.Patch(color=c, label=CLASS_NAMES[i])
               for i, c in CLASS_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=6,
               fontsize=9, frameon=True, title="Classes",
               bbox_to_anchor=(0.5, -0.01))

    cond_kr = "야간 (Night)" if condition == "night" else "주간 (Day)"
    fig.suptitle(
        f"그림. {cond_kr} 장면에서의 모델 단계별 처리 결과 — "
        f"(f) RGB 단독 vs (g) Thermal 단독 vs (h) RGB+Thermal 융합",
        fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = OUT_DIR / f"fig_{condition}_comparison.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {out_path}")
    return out_path


# ── Figure 2: 야간 vs 주간 나란히 비교 (논문 삽입용 compact) ─────────────────
def make_night_vs_day_figure(model, hook, device, night_sample, day_sample,
                              vis_dir, ir_dir, cfg):
    """
    2행 × 5열: 위=야간, 아래=주간
    열: RGB | Thermal | Fused Feature | Detection | Thermal Contribution
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    row_labels = ["야간\n(Night)", "주간\n(Day)"]
    col_titles = ["(a) RGB 영상", "(b) Thermal 영상",
                  "(c) Fused Feature", "(d) 검출 결과",
                  "(e) Thermal 기여도\n(Fused−RGB Feature)"]

    for ci, t in enumerate(col_titles):
        axes[0, ci].set_title(t, fontsize=10, fontweight="bold", pad=8)

    for row, (idx, img_id, br) in enumerate([night_sample, day_sample]):
        rgb_arr = load_img(vis_dir / f"{img_id}.png", "RGB")
        th_arr  = load_img(ir_dir  / f"{img_id}.png", "L")
        orig_h, orig_w = rgb_arr.shape[:2]

        rgb_t, th_t, h, w = preprocess(rgb_arr, th_arr)
        rgb_t = rgb_t.to(device)
        th_t  = th_t.to(device)

        hook.feats.clear()
        with torch.no_grad():
            outputs = model(rgb_t, th_t)
        det = outputs[0]

        axes[row,0].imshow(rgb_arr)
        axes[row,0].set_ylabel(row_labels[row], fontsize=12, fontweight="bold",
                               rotation=0, labelpad=55, va="center")
        axes[row,1].imshow(th_arr, cmap="inferno")

        # Fused feature
        if "fused_c4" in hook.feats:
            hm_fused = feat_to_heatmap(hook.feats["fused_c4"])
            axes[row,2].imshow(hm_fused, cmap="jet")
        else:
            axes[row,2].text(0.5,0.5,"N/A",ha="center",va="center",
                             transform=axes[row,2].transAxes)

        # Detection
        axes[row,3].imshow(rgb_arr)
        draw_boxes(axes[row,3],
                   det["boxes"].cpu().numpy(),
                   det["scores"].cpu().numpy(),
                   det["labels"].cpu().numpy(),
                   orig_h, orig_w)
        n_det = (det["scores"].cpu().numpy() >= SCORE_THRESH).sum()
        axes[row,3].text(5, 20, f"Det: {n_det}", fontsize=8,
                         color="white", bbox=dict(facecolor="black", alpha=0.6))

        # Thermal 기여도 = Fused - RGB feature (절댓값)
        if "fused_c4" in hook.feats and "rgb_c4" in hook.feats:
            fused_up = F.interpolate(hook.feats["fused_c4"],
                                     size=hook.feats["rgb_c4"].shape[-2:],
                                     mode="bilinear", align_corners=False)
            diff = (fused_up - hook.feats["rgb_c4"]).abs()
            hm_diff = feat_to_heatmap(diff)
            im = axes[row,4].imshow(hm_diff, cmap="hot")
            plt.colorbar(im, ax=axes[row,4], fraction=0.046)
        else:
            axes[row,4].text(0.5,0.5,"N/A",ha="center",va="center",
                             transform=axes[row,4].transAxes)

        for ax in axes[row]:
            ax.axis("off")

    # 범례
    handles = [mpatches.Patch(color=c, label=CLASS_NAMES[i])
               for i, c in CLASS_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=6,
               fontsize=9, frameon=True, title="Detection Classes",
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("그림. 야간 vs 주간 장면에서의 RGB-Thermal 융합 검출 비교",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = OUT_DIR / "fig_night_vs_day.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {out_path}")
    return out_path


# ── Figure 3: 클래스별 성능 바 차트 ──────────────────────────────────────────
def make_classwise_bar(json_path: Path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    classes = ["People", "Car", "Bus", "Motorcycle", "Lamp", "Truck"]
    all_v   = [data["results"]["all"].get(f"AP50_{c}", 0)   for c in classes]
    night_v = [data["results"]["night"].get(f"AP50_{c}", 0) for c in classes]
    day_v   = [data["results"]["day"].get(f"AP50_{c}", 0)   for c in classes]

    # Truck 야간 샘플 없음(-1) → 0으로
    night_v = [max(v, 0) for v in night_v]

    x = np.arange(len(classes))
    w = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    bars_all   = ax.bar(x - w,   all_v,   w, label="전체 (Overall)", color="#4472C4", alpha=0.9)
    bars_night = ax.bar(x,       night_v, w, label="야간 (Night)",   color="#ED7D31", alpha=0.9)
    bars_day   = ax.bar(x + w,   day_v,   w, label="주간 (Day)",     color="#70AD47", alpha=0.9)

    def label_bars(bars):
        for b in bars:
            h = b.get_height()
            if h > 0.01:
                ax.text(b.get_x() + b.get_width()/2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    label_bars(bars_all)
    label_bars(bars_night)
    label_bars(bars_day)

    # Truck 야간 주석
    ax.annotate("샘플\n없음", xy=(x[-1], 0.01), ha="center", fontsize=7,
                color="gray", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylabel("AP@IoU=0.5", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_title("그림. 클래스별 AP@0.5 — 전체/야간/주간 비교", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_path = OUT_DIR / "fig_classwise_bar.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {out_path}")
    return out_path


# ── Figure 4: 전체/야간/주간 mAP 요약 테이블 Figure ─────────────────────────
def make_summary_table(json_path: Path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    rows = [
        ["조건", "이미지 수", "mAP@0.5", "mAP@[.5:.95]"],
        ["전체 (Overall)", "840",
         f"{data['results']['all']['mAP50']:.4f}",
         f"{data['results']['all']['mAP50_95']:.4f}"],
        ["야간 (Night)",
         str(data["night_count"]),
         f"{data['results']['night']['mAP50']:.4f}",
         f"{data['results']['night']['mAP50_95']:.4f}"],
        ["주간 (Day)",
         str(data["day_count"]),
         f"{data['results']['day']['mAP50']:.4f}",
         f"{data['results']['day']['mAP50_95']:.4f}"],
    ]

    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.4, 2.0)

    # 헤더 색
    for j in range(4):
        tbl[0, j].set_facecolor("#4472C4")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # 야간 행 강조
    for j in range(4):
        tbl[2, j].set_facecolor("#FFF2CC")

    ax.set_title("표. 야간/주간 조건별 검출 성능 (M3FD Val set)",
                 fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    out_path = OUT_DIR / "fig_summary_table.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {out_path}")
    return out_path


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = Config()
    _, val_dataset = build_datasets(cfg.data)

    base = Path(__file__).parent / "data" / "M3FD"
    vis_dir = base / "Vis"
    ir_dir  = base / "Ir"

    # 모델 & hook
    model = build_model(cfg.model, num_classes=cfg.data.num_classes)
    ckpt  = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    hook = FeatureHook()
    hook.register(model)

    print("\n샘플 선택 중...")
    night_samples, day_samples = pick_samples(val_dataset, vis_dir, n_night=3, n_day=3)
    print(f"  야간 {len(night_samples)}장, 주간 {len(day_samples)}장")

    print("\n[Figure 1] 야간 비교 패널 생성...")
    make_comparison_figure(model, hook, device, night_samples, "night", vis_dir, ir_dir, cfg)

    print("\n[Figure 2] 주간 비교 패널 생성...")
    make_comparison_figure(model, hook, device, day_samples, "day", vis_dir, ir_dir, cfg)

    print("\n[Figure 3] 야간 vs 주간 나란히 비교...")
    make_night_vs_day_figure(model, hook, device,
                              night_samples[0], day_samples[0],
                              vis_dir, ir_dir, cfg)

    json_path = Path(__file__).parent / "runs" / "night_analysis.json"
    print("\n[Figure 4] 클래스별 바 차트...")
    make_classwise_bar(json_path)

    print("\n[Figure 5] 요약 테이블...")
    make_summary_table(json_path)

    print(f"\n완료! 모든 Figure → {OUT_DIR}")


if __name__ == "__main__":
    main()
