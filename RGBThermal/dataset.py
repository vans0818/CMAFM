"""M3FD Multispectral Dataset — paired RGB (Vis) + Thermal (Ir) with VOC XML annotations."""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import DataConfig


class M3FDDataset(Dataset):
    """
    M3FD Dataset: 4200 aligned RGB-Thermal pairs with VOC XML annotations.

    Structure:
        {root}/
        ├── Vis/         # RGB images (1024x768 PNG)
        ├── Ir/          # Thermal images (1024x768 PNG)
        └── Annotation/  # VOC XML annotations
    """

    def __init__(
        self,
        cfg: DataConfig,
        image_ids: List[str],
        transforms: Optional[A.Compose] = None,
    ):
        self.transforms = transforms
        # Derive base dirs from this file's location so workers get correct Unicode paths
        _base = Path(__file__).parent / "data" / "M3FD"
        self._rgb_dir = _base / cfg.rgb_dir
        self._thermal_dir = _base / cfg.thermal_dir
        self._ann_dir = _base / cfg.ann_dir

        # Build class name → contiguous ID mapping (1-based, 0 = background)
        self.class_to_id: Dict[str, int] = {
            name: i + 1 for i, name in enumerate(cfg.target_classes)
        }

        # Store only img_ids; paths are rebuilt in __getitem__ via __file__
        self.samples: List[str] = []
        for img_id in image_ids:
            if (
                (self._rgb_dir / f"{img_id}.png").exists()
                and (self._thermal_dir / f"{img_id}.png").exists()
                and (self._ann_dir / f"{img_id}.xml").exists()
            ):
                self.samples.append(img_id)

        print(f"  Loaded {len(self.samples)} paired samples "
              f"({len(cfg.target_classes)} classes: {cfg.target_classes})")

    def _parse_voc_xml(self, xml_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Parse VOC XML annotation → boxes (xyxy) + labels."""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in self.class_to_id:
                continue

            bbox = obj.find("bndbox")
            x1 = float(bbox.find("xmin").text)
            y1 = float(bbox.find("ymin").text)
            x2 = float(bbox.find("xmax").text)
            y2 = float(bbox.find("ymax").text)

            if x2 - x1 < 1 or y2 - y1 < 1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_to_id[name])

        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        labels = np.array(labels, dtype=np.int64)
        return boxes, labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        img_id = self.samples[idx]
        # Rebuild paths from __file__ so workers get correct Unicode paths on Windows
        _base = Path(__file__).parent / "data" / "M3FD"
        rgb_path = _base / self._rgb_dir.name / f"{img_id}.png"
        thermal_path = _base / self._thermal_dir.name / f"{img_id}.png"
        ann_path = _base / self._ann_dir.name / f"{img_id}.xml"

        # Load images via PIL
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        thermal = np.array(Image.open(thermal_path).convert("L"))

        # Parse annotations
        boxes, labels = self._parse_voc_xml(ann_path)

        # Resize thermal to match RGB if needed
        h, w = rgb.shape[:2]
        if thermal.shape[:2] != (h, w):
            thermal = cv2.resize(thermal, (w, h), interpolation=cv2.INTER_LINEAR)

        # Apply augmentations (same transform to both images + boxes)
        if self.transforms is not None and len(boxes) > 0:
            transformed = self.transforms(
                image=rgb,
                image2=thermal,
                bboxes=boxes.tolist(),
                labels=labels.tolist(),
            )
            rgb = transformed["image"]
            thermal = transformed["image2"]
            boxes = np.array(transformed["bboxes"], dtype=np.float32).reshape(-1, 4)
            labels = np.array(transformed["labels"], dtype=np.int64)
        elif self.transforms is not None:
            # No boxes — just resize
            transformed = self.transforms(
                image=rgb, image2=thermal, bboxes=[], labels=[],
            )
            rgb = transformed["image"]
            thermal = transformed["image2"]

        # To tensors — [0, 1] range (normalization handled by FasterRCNN transform)
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        thermal_tensor = torch.from_numpy(thermal).unsqueeze(0).float() / 255.0
        # Repeat thermal to 3 channels for pretrained backbone
        thermal_tensor = thermal_tensor.repeat(3, 1, 1)

        target = {
            "boxes": torch.from_numpy(boxes) if len(boxes) > 0
                     else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.from_numpy(labels) if len(labels) > 0
                      else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }

        return rgb_tensor, thermal_tensor, target


def get_train_transforms(img_size: Tuple[int, int]) -> A.Compose:
    """Training augmentations applied identically to RGB + Thermal."""
    return A.Compose(
        [
            A.Resize(height=img_size[0], width=img_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3, brightness_limit=0.2, contrast_limit=0.2),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",  # xyxy
            label_fields=["labels"],
            min_visibility=0.3,
        ),
        additional_targets={"image2": "image"},
    )


def get_val_transforms(img_size: Tuple[int, int]) -> A.Compose:
    """Validation: resize only."""
    return A.Compose(
        [
            A.Resize(height=img_size[0], width=img_size[1]),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.3,
        ),
        additional_targets={"image2": "image"},
    )


def build_datasets(cfg: DataConfig):
    """Build train/val datasets with random split."""
    ann_dir = Path(cfg.root) / cfg.ann_dir
    all_ids = sorted([f.stem for f in ann_dir.glob("*.xml")])

    # Reproducible split
    rng = np.random.RandomState(cfg.split_seed)
    rng.shuffle(all_ids)

    n_val = int(len(all_ids) * cfg.val_ratio)
    val_ids = all_ids[:n_val]
    train_ids = all_ids[n_val:]

    print(f"Dataset split: {len(train_ids)} train / {len(val_ids)} val")

    train_ds = M3FDDataset(cfg, train_ids, get_train_transforms(cfg.img_size))
    val_ds = M3FDDataset(cfg, val_ids, get_val_transforms(cfg.img_size))

    return train_ds, val_ds


def collate_fn(batch):
    """Custom collate for variable-size targets."""
    rgb_imgs, thermal_imgs, targets = zip(*batch)
    rgb_imgs = torch.stack(rgb_imgs)
    thermal_imgs = torch.stack(thermal_imgs)
    return rgb_imgs, thermal_imgs, list(targets)
