"""Step-by-step pipeline visualization for Multispectral Detection."""

import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from dataset import build_datasets, get_val_transforms, M3FDDataset
from model import build_model

# Class names and colors (BGR for cv2)
CLASS_NAMES = {1: "People", 2: "Car", 3: "Bus", 4: "Motorcycle", 5: "Lamp", 6: "Truck"}
CLASS_COLORS = {
    1: (0, 220, 0),      # green
    2: (220, 0, 0),      # blue
    3: (0, 165, 255),    # orange
    4: (0, 255, 255),    # yellow
    5: (255, 0, 255),    # magenta
    6: (255, 100, 0),    # cyan
}


def load_model_and_data(checkpoint_path, num_samples=4):
    """Load model and select diverse validation samples."""
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg.model, num_classes=cfg.data.num_classes)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    _, val_ds = build_datasets(cfg.data)

    # Pick samples with diverse content
    indices = np.linspace(0, len(val_ds) - 1, num_samples, dtype=int)
    return model, val_ds, indices, device, cfg


def get_raw_images(val_ds, idx):
    """Load raw (un-normalized) images for display."""
    rgb_path, thermal_path, ann_path = val_ds.samples[idx]
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    thermal = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
    return rgb, thermal, ann_path


def extract_features(model, rgb_tensor, thermal_tensor, device):
    """Extract intermediate features for visualization."""
    rgb_t = rgb_tensor.unsqueeze(0).to(device)
    thermal_t = thermal_tensor.unsqueeze(0).to(device)

    # Manually run through backbone stages
    backbone = model.backbone

    # FasterRCNN transform first
    combined = torch.cat([rgb_t, thermal_t], dim=1)
    images = [combined[0]]
    images_transformed = model.detector.transform(images)
    x = images_transformed[0].tensors

    # Split back
    rgb_in = x[:, :3]
    thermal_in = x[:, 3:]

    features = {}

    with torch.no_grad():
        # RGB backbone stages
        r = backbone.rgb_stages["stem"](rgb_in)
        r_c2 = backbone.rgb_stages["layer1"](r)
        r_c3 = backbone.rgb_stages["layer2"](r_c2)
        r_c4 = backbone.rgb_stages["layer3"](r_c3)
        r_c5 = backbone.rgb_stages["layer4"](r_c4)

        # Thermal backbone stages
        t = backbone.thermal_stages["stem"](thermal_in)
        t_c2 = backbone.thermal_stages["layer1"](t)
        t_c3 = backbone.thermal_stages["layer2"](t_c2)
        t_c4 = backbone.thermal_stages["layer3"](t_c3)
        t_c5 = backbone.thermal_stages["layer4"](t_c4)

        # Fused features
        fused_c3 = backbone.fusion_c3(r_c3, t_c3)
        fused_c4 = backbone.fusion_c4(r_c4, t_c4)
        fused_c5 = backbone.fusion_c5(r_c5, t_c5)

    features["rgb_c4"] = r_c4
    features["thermal_c4"] = t_c4
    features["fused_c4"] = fused_c4
    features["rgb_c3"] = r_c3
    features["thermal_c3"] = t_c3
    features["fused_c3"] = fused_c3

    return features


def feat_to_heatmap(feat, size=(640, 640)):
    """Convert feature map to displayable heatmap."""
    # Channel-wise mean, then normalize
    with torch.no_grad():
        hm = feat[0].mean(dim=0).cpu().numpy()  # H, W
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    hm = (hm * 255).astype(np.uint8)
    hm = cv2.resize(hm, (size[1], size[0]))
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    return hm_color


def draw_boxes(image, boxes, labels, scores=None, score_thresh=0.5):
    """Draw bounding boxes on image."""
    vis = image.copy()
    for i in range(len(boxes)):
        if scores is not None and scores[i] < score_thresh:
            continue
        x1, y1, x2, y2 = [int(v) for v in boxes[i]]
        label = int(labels[i])
        color = CLASS_COLORS.get(label, (255, 255, 255))
        name = CLASS_NAMES.get(label, f"cls{label}")

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        if scores is not None:
            text = f"{name} {scores[i]:.2f}"
        else:
            text = name
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(vis, text, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return vis


def draw_gt_boxes(image, ann_path, class_to_id, img_size):
    """Draw ground truth boxes from XML annotation."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(ann_path)
    root = tree.getroot()

    h, w = image.shape[:2]
    orig_w = int(root.find("size/width").text)
    orig_h = int(root.find("size/height").text)
    sx, sy = w / orig_w, h / orig_h

    vis = image.copy()
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in class_to_id:
            continue
        label = class_to_id[name]
        bbox = obj.find("bndbox")
        x1 = int(float(bbox.find("xmin").text) * sx)
        y1 = int(float(bbox.find("ymin").text) * sy)
        x2 = int(float(bbox.find("xmax").text) * sx)
        y2 = int(float(bbox.find("ymax").text) * sy)

        color = CLASS_COLORS.get(label, (255, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(vis, name, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return vis


def visualize_single(model, val_ds, idx, device, save_dir, sample_num):
    """Create step-by-step visualization for one sample."""
    # Step 1: Load raw images
    rgb_raw, thermal_raw, ann_path = get_raw_images(val_ds, idx)
    img_size = (640, 640)

    rgb_disp = cv2.resize(rgb_raw, (img_size[1], img_size[0]))
    thermal_disp = cv2.resize(thermal_raw, (img_size[1], img_size[0]))
    thermal_color = cv2.applyColorMap(thermal_disp, cv2.COLORMAP_INFERNO)

    # Step 2: Get model tensors
    rgb_tensor, thermal_tensor, target = val_ds[idx]

    # Step 3: Extract intermediate features
    features = extract_features(model, rgb_tensor, thermal_tensor, device)

    # Step 4: Run full detection
    model.eval()
    with torch.no_grad():
        rgb_t = rgb_tensor.unsqueeze(0).to(device)
        thermal_t = thermal_tensor.unsqueeze(0).to(device)
        outputs = model(rgb_t, thermal_t)

    det = outputs[0]
    boxes = det["boxes"].cpu().numpy()
    labels = det["labels"].cpu().numpy()
    scores = det["scores"].cpu().numpy()

    # Rescale boxes to display size
    # FasterRCNN internally resizes, need to map back
    scale_x = img_size[1] / img_size[1]  # already 640
    scale_y = img_size[0] / img_size[0]

    # --- Create figure ---
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle(f"Multispectral Detection Pipeline — Sample {sample_num}",
                 fontsize=16, fontweight="bold", y=0.98)

    class_to_id = {name: i + 1 for i, name in enumerate(val_ds.cfg.target_classes)}

    # Row 1: Input images
    axes[0, 0].imshow(cv2.cvtColor(rgb_disp, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Step 1: RGB Input", fontsize=13, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cv2.cvtColor(thermal_color, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Step 2: Thermal Input (Inferno)", fontsize=13, fontweight="bold")
    axes[0, 1].axis("off")

    # Ground truth
    gt_vis = draw_gt_boxes(rgb_disp, ann_path, class_to_id, img_size)
    axes[0, 2].imshow(cv2.cvtColor(gt_vis, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("Ground Truth", fontsize=13, fontweight="bold")
    axes[0, 2].axis("off")

    # Row 2: Feature maps (C4 level)
    rgb_hm = feat_to_heatmap(features["rgb_c4"], img_size)
    axes[1, 0].imshow(cv2.cvtColor(rgb_hm, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Step 3: RGB Features (C4)", fontsize=13, fontweight="bold")
    axes[1, 0].axis("off")

    thermal_hm = feat_to_heatmap(features["thermal_c4"], img_size)
    axes[1, 1].imshow(cv2.cvtColor(thermal_hm, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Step 4: Thermal Features (C4)", fontsize=13, fontweight="bold")
    axes[1, 1].axis("off")

    fused_hm = feat_to_heatmap(features["fused_c4"], img_size)
    axes[1, 2].imshow(cv2.cvtColor(fused_hm, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("Step 5: Fused Features (C4)", fontsize=13, fontweight="bold")
    axes[1, 2].axis("off")

    # Row 3: Detection results
    det_rgb = draw_boxes(rgb_disp, boxes, labels, scores, score_thresh=0.5)
    axes[2, 0].imshow(cv2.cvtColor(det_rgb, cv2.COLOR_BGR2RGB))
    axes[2, 0].set_title("Step 6: Detection on RGB", fontsize=13, fontweight="bold")
    axes[2, 0].axis("off")

    det_thermal = draw_boxes(thermal_color, boxes, labels, scores, score_thresh=0.5)
    axes[2, 1].imshow(cv2.cvtColor(det_thermal, cv2.COLOR_BGR2RGB))
    axes[2, 1].set_title("Step 7: Detection on Thermal", fontsize=13, fontweight="bold")
    axes[2, 1].axis("off")

    # Overlay: blend RGB + Thermal with detections
    blend = cv2.addWeighted(rgb_disp, 0.5, thermal_color, 0.5, 0)
    det_blend = draw_boxes(blend, boxes, labels, scores, score_thresh=0.5)
    axes[2, 2].imshow(cv2.cvtColor(det_blend, cv2.COLOR_BGR2RGB))
    axes[2, 2].set_title("Step 8: Fused Overlay + Detection", fontsize=13, fontweight="bold")
    axes[2, 2].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = save_dir / f"pipeline_sample_{sample_num}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")

    # Also save high-level feature comparison (C3 level)
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle(f"Feature Fusion Detail (C3 level) — Sample {sample_num}",
                  fontsize=14, fontweight="bold")

    rgb_hm3 = feat_to_heatmap(features["rgb_c3"], img_size)
    axes2[0].imshow(cv2.cvtColor(rgb_hm3, cv2.COLOR_BGR2RGB))
    axes2[0].set_title("RGB Features (C3 — high res)", fontsize=12)
    axes2[0].axis("off")

    thermal_hm3 = feat_to_heatmap(features["thermal_c3"], img_size)
    axes2[1].imshow(cv2.cvtColor(thermal_hm3, cv2.COLOR_BGR2RGB))
    axes2[1].set_title("Thermal Features (C3 — high res)", fontsize=12)
    axes2[1].axis("off")

    fused_hm3 = feat_to_heatmap(features["fused_c3"], img_size)
    axes2[2].imshow(cv2.cvtColor(fused_hm3, cv2.COLOR_BGR2RGB))
    axes2[2].set_title("Fused Features (C3 — after Cross-Modal Attention)", fontsize=12)
    axes2[2].axis("off")

    plt.tight_layout()
    out_path2 = save_dir / f"fusion_detail_sample_{sample_num}.png"
    plt.savefig(out_path2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path2}")


def main():
    checkpoint = "/workspace/military/runs/best.pth"
    save_dir = Path("/workspace/military/runs/visualizations")
    save_dir.mkdir(parents=True, exist_ok=True)

    num_samples = 6
    print(f"Loading model from {checkpoint}...")
    model, val_ds, indices, device, cfg = load_model_and_data(checkpoint, num_samples)
    print(f"Generating {num_samples} visualizations...\n")

    for i, idx in enumerate(indices):
        print(f"--- Sample {i+1}/{num_samples} (idx={idx}) ---")
        visualize_single(model, val_ds, int(idx), device, save_dir, i + 1)
        print()

    print(f"All visualizations saved to: {save_dir}")


if __name__ == "__main__":
    main()
