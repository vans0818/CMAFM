"""Inference script: run detection on paired RGB + Thermal images."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from config import Config
from model import build_model


# Class names and colors
CLASS_NAMES = {1: "person", 2: "car", 3: "bicycle"}
CLASS_COLORS = {1: (0, 255, 0), 2: (255, 0, 0), 3: (0, 255, 255)}


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    cfg = Config()
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "config" in ckpt:
        cfg = ckpt["config"]

    model = build_model(cfg.model, num_classes=cfg.data.num_classes)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, cfg


def preprocess(rgb_path: str, thermal_path: str, img_size=(640, 640)):
    """Load and preprocess an image pair."""
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    thermal = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)

    orig_h, orig_w = rgb.shape[:2]

    # Resize
    rgb_resized = cv2.resize(rgb, (img_size[1], img_size[0]))
    thermal_resized = cv2.resize(thermal, (img_size[1], img_size[0]))

    # To tensor [0,1]
    rgb_tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1).float() / 255.0
    thermal_tensor = torch.from_numpy(thermal_resized).unsqueeze(0).float() / 255.0
    thermal_tensor = thermal_tensor.repeat(3, 1, 1)

    return (rgb_tensor.unsqueeze(0), thermal_tensor.unsqueeze(0),
            rgb, orig_h, orig_w)


def draw_detections(image, detections, orig_h, orig_w, img_size, score_thresh=0.5):
    """Draw bounding boxes on image."""
    vis = image.copy()
    if vis.dtype != np.uint8:
        vis = (vis * 255).astype(np.uint8)

    scale_x = orig_w / img_size[1]
    scale_y = orig_h / img_size[0]

    boxes = detections["boxes"].cpu().numpy()
    scores = detections["scores"].cpu().numpy()
    labels = detections["labels"].cpu().numpy()

    count = 0
    for box, score, label in zip(boxes, scores, labels):
        if score < score_thresh:
            continue

        x1 = int(box[0] * scale_x)
        y1 = int(box[1] * scale_y)
        x2 = int(box[2] * scale_x)
        y2 = int(box[3] * scale_y)

        color = CLASS_COLORS.get(label, (255, 255, 255))
        name = CLASS_NAMES.get(label, f"cls{label}")

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        text = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(vis, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 1, cv2.LINE_AA)
        count += 1

    return vis, count


def main():
    parser = argparse.ArgumentParser(description="Multispectral Detection Inference")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--rgb", required=True, help="Path to RGB image")
    parser.add_argument("--thermal", required=True, help="Path to thermal image")
    parser.add_argument("--output", default="detection_result.jpg", help="Output path")
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--img-size", type=int, nargs=2, default=[640, 640])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg = load_model(args.checkpoint, device)
    print(f"Model loaded from {args.checkpoint}")

    rgb_tensor, thermal_tensor, orig_rgb, orig_h, orig_w = preprocess(
        args.rgb, args.thermal, tuple(args.img_size))

    rgb_tensor = rgb_tensor.to(device)
    thermal_tensor = thermal_tensor.to(device)

    with torch.no_grad():
        outputs = model(rgb_tensor, thermal_tensor)

    detections = outputs[0]
    vis, count = draw_detections(
        cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR),
        detections, orig_h, orig_w,
        tuple(args.img_size), args.score_thresh)

    cv2.imwrite(args.output, vis)
    print(f"Detected {count} objects (score > {args.score_thresh})")
    print(f"Result saved to {args.output}")


if __name__ == "__main__":
    main()
