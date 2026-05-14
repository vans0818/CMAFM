"""COCO-style evaluation for Multispectral Object Detection."""

import json
import os
import tempfile
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

from config import DataConfig


def _compute_precision_recall_f1(coco_eval: COCOeval, iou_thr: float = 0.5) -> Dict[str, float]:
    """
    COCOeval 결과에서 Precision, Recall, F1을 추출한다.
    pycocotools precision shape: [T, R, K, A, M]
      T=IoU thresholds, R=recall thresholds, K=categories, A=area, M=maxDets
    """
    # IoU threshold index (0.5:0.05:0.95 → index 0 = 0.50)
    iou_idx = int(round((iou_thr - 0.5) / 0.05))

    p = coco_eval.eval["precision"]   # (T, R, K, A, M)
    r = coco_eval.eval["recall"]      # (T, K, A, M)

    # area=all(0), maxDets=last(-1)
    prec = p[iou_idx, :, :, 0, -1]   # (R, K)
    rec  = r[iou_idx, :, 0, -1]      # (K,)

    # mean precision over recall thresholds (ignore -1 = no GT)
    valid_prec = prec[prec > -1]
    mean_prec  = float(valid_prec.mean()) if len(valid_prec) > 0 else 0.0

    valid_rec  = rec[rec > -1]
    mean_rec   = float(valid_rec.mean()) if len(valid_rec) > 0 else 0.0

    f1 = (2 * mean_prec * mean_rec / (mean_prec + mean_rec)
          if (mean_prec + mean_rec) > 0 else 0.0)
    miss_rate = 1.0 - mean_rec

    return {
        "precision": mean_prec,
        "recall":    mean_rec,
        "f1":        f1,
        "miss_rate": miss_rate,
    }


@torch.no_grad()
def evaluate_model(model, data_loader: DataLoader, device: torch.device,
                   data_cfg: DataConfig) -> Dict[str, float]:
    """
    Evaluate model using COCO mAP metrics + Precision / Recall / F1 / Miss Rate / FPS.

    Returns dict with:
        mAP50, mAP50_95, mAP75,
        precision, recall, f1, miss_rate,
        fps,
        AP50_{class} per class
    """
    model.eval()

    all_predictions = []
    all_gt = []
    img_id_counter = 0
    total_time = 0.0
    total_images = 0

    for rgb, thermal, targets in data_loader:
        rgb = rgb.to(device)
        thermal = thermal.to(device)

        t0 = time.perf_counter()
        outputs = model(rgb, thermal)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - t0
        total_images += len(targets)

        for i, (output, target) in enumerate(zip(outputs, targets)):
            img_id = img_id_counter + i

            boxes  = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            for j in range(len(boxes)):
                x1, y1, x2, y2 = boxes[j]
                all_predictions.append({
                    "image_id":   img_id,
                    "category_id": int(labels[j]),
                    "bbox":  [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    "score": float(scores[j]),
                })

            gt_boxes  = target["boxes"].cpu().numpy()
            gt_labels = target["labels"].cpu().numpy()
            for j in range(len(gt_boxes)):
                x1, y1, x2, y2 = gt_boxes[j]
                all_gt.append({
                    "image_id":   img_id,
                    "id":          len(all_gt),
                    "category_id": int(gt_labels[j]),
                    "bbox":  [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    "area":  float((x2-x1)*(y2-y1)),
                    "iscrowd": 0,
                })

        img_id_counter += len(targets)

    fps = total_images / total_time if total_time > 0 else 0.0

    if len(all_predictions) == 0 or len(all_gt) == 0:
        return {"mAP50": 0.0, "mAP50_95": 0.0, "precision": 0.0,
                "recall": 0.0, "f1": 0.0, "miss_rate": 1.0, "fps": fps}

    categories = [{"id": i+1, "name": name}
                  for i, name in enumerate(data_cfg.target_classes)]
    images = [{"id": i} for i in range(img_id_counter)]
    gt_coco_dict = {"images": images, "annotations": all_gt, "categories": categories}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(gt_coco_dict, f);  gt_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(all_predictions, f);  pred_path = f.name

    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(pred_path)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    pr_metrics = _compute_precision_recall_f1(coco_eval, iou_thr=0.5)

    results = {
        "mAP50_95":  float(coco_eval.stats[0]),
        "mAP50":     float(coco_eval.stats[1]),
        "mAP75":     float(coco_eval.stats[2]),
        "precision": pr_metrics["precision"],
        "recall":    pr_metrics["recall"],
        "f1":        pr_metrics["f1"],
        "miss_rate": pr_metrics["miss_rate"],
        "fps":       fps,
    }

    # Per-class AP50 + Recall
    for cat_id, cat_name in enumerate(data_cfg.target_classes, start=1):
        ce = COCOeval(coco_gt, coco_dt, "bbox")
        ce.params.catIds = [cat_id]
        ce.evaluate(); ce.accumulate(); ce.summarize()
        results[f"AP50_{cat_name}"] = float(ce.stats[1])
        pr_cls = _compute_precision_recall_f1(ce, iou_thr=0.5)
        results[f"recall_{cat_name}"]    = pr_cls["recall"]
        results[f"miss_rate_{cat_name}"] = pr_cls["miss_rate"]

    model.train()
    os.unlink(gt_path)
    os.unlink(pred_path)

    return results
