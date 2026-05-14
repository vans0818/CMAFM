"""
FLIR ADAS v2 Dataset Download Guide & Preparation Script.

The FLIR ADAS v2 dataset requires a free Teledyne FLIR account to download.

Steps:
  1. Visit: https://www.flir.com/oem/adas/adas-dataset-form/
  2. Register / sign in and download "FLIR_ADAS_v2" (aligned version)
  3. Extract to: /workspace/data/FLIR_ADAS_v2/
  4. Run this script to verify and prepare the data

Expected structure after extraction:
    /workspace/data/FLIR_ADAS_v2/
    ├── images_rgb_train/data/*.jpg
    ├── images_thermal_train/data/*.jpg
    ├── images_rgb_val/data/*.jpg
    ├── images_thermal_val/data/*.jpg
    └── coco.json  (or separate train/val JSONs)

This script reorganizes into the format expected by dataset.py:
    /workspace/data/FLIR_ADAS_v2/
    ├── train/
    │   ├── rgb/*.jpg
    │   ├── thermal_8_bit/*.jpg
    │   └── train.json
    └── val/
        ├── rgb/*.jpg
        ├── thermal_8_bit/*.jpg
        └── val.json
"""

import argparse
import json
import os
import shutil
from pathlib import Path


def reorganize_flir_adas(root: str):
    """Reorganize FLIR ADAS v2 into train/val splits."""
    root = Path(root)

    # Detect layout
    # Layout A: images_rgb_train/data/, images_thermal_train/data/, ...
    # Layout B: train/rgb/, train/thermal_8_bit/, ... (already organized)

    if (root / "train" / "rgb").exists():
        print("Dataset already in expected format.")
        return

    for split in ["train", "val"]:
        split_dir = root / split
        rgb_dir = split_dir / "rgb"
        thermal_dir = split_dir / "thermal_8_bit"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        thermal_dir.mkdir(parents=True, exist_ok=True)

        # Try Layout A
        src_rgb = root / f"images_rgb_{split}" / "data"
        src_thermal = root / f"images_thermal_{split}" / "data"

        if src_rgb.exists():
            print(f"Copying {split} RGB images from {src_rgb}...")
            for f in src_rgb.glob("*"):
                shutil.copy2(f, rgb_dir / f.name)

        if src_thermal.exists():
            print(f"Copying {split} thermal images from {src_thermal}...")
            for f in src_thermal.glob("*"):
                shutil.copy2(f, thermal_dir / f.name)

    # Handle annotations
    # FLIR ADAS v2 may have a single coco.json or split JSONs
    for ann_name in ["coco.json", "annotations.json"]:
        ann_file = root / ann_name
        if ann_file.exists():
            split_annotations(ann_file, root)
            return

    # Check for already split annotations
    for split in ["train", "val"]:
        for name in [f"{split}.json", f"index_{split}.json",
                     f"FLIR_ADAS_v2_{split}.json"]:
            src = root / name
            if src.exists() and not (root / split / f"{split}.json").exists():
                shutil.copy2(src, root / split / f"{split}.json")
                print(f"Copied {src} → {root / split / f'{split}.json'}")

    print("Done. Verify annotation files exist in train/ and val/.")


def split_annotations(ann_file: Path, root: Path):
    """Split a single COCO JSON into train/val based on image paths."""
    print(f"Splitting annotations from {ann_file}...")
    with open(ann_file) as f:
        coco = json.load(f)

    train_imgs, val_imgs = [], []
    train_img_ids, val_img_ids = set(), set()

    for img in coco["images"]:
        fname = img["file_name"].lower()
        if "train" in fname:
            train_imgs.append(img)
            train_img_ids.add(img["id"])
        elif "val" in fname:
            val_imgs.append(img)
            val_img_ids.add(img["id"])

    for split, imgs, ids in [("train", train_imgs, train_img_ids),
                              ("val", val_imgs, val_img_ids)]:
        anns = [a for a in coco["annotations"] if a["image_id"] in ids]
        split_coco = {
            "images": imgs,
            "annotations": anns,
            "categories": coco["categories"],
        }
        out_path = root / split / f"{split}.json"
        with open(out_path, "w") as f:
            json.dump(split_coco, f)
        print(f"  {split}: {len(imgs)} images, {len(anns)} annotations → {out_path}")


def verify_dataset(root: str):
    """Quick verification of dataset structure."""
    root = Path(root)
    ok = True

    for split in ["train", "val"]:
        rgb_dir = root / split / "rgb"
        thermal_dir = root / split / "thermal_8_bit"
        ann_file = root / split / f"{split}.json"

        rgb_count = len(list(rgb_dir.glob("*"))) if rgb_dir.exists() else 0
        thermal_count = len(list(thermal_dir.glob("*"))) if thermal_dir.exists() else 0
        has_ann = ann_file.exists()

        status = "OK" if (rgb_count > 0 and thermal_count > 0 and has_ann) else "MISSING"
        if status == "MISSING":
            ok = False

        print(f"[{status}] {split}: {rgb_count} RGB, {thermal_count} thermal, "
              f"annotations={'yes' if has_ann else 'NO'}")

    if ok:
        print("\nDataset ready for training!")
    else:
        print("\nSome files are missing. Check the download and run again.")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/data/FLIR_ADAS_v2",
                        help="Path to extracted FLIR ADAS v2 dataset")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        verify_dataset(args.root)
    else:
        reorganize_flir_adas(args.root)
        verify_dataset(args.root)
