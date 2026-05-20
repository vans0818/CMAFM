"""
Step 1: M3FD VOC XML → YOLO 변환 + train/val split
Step 2: FLIR ADAS COCO JSON → YOLO 변환
Step 3: M3FD + FLIR 통합 txt 및 yaml 생성

실행:
  python prepare_all.py
"""

import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

random.seed(42)

# ── 경로 ──────────────────────────────────────────────────────────────────
BASE         = Path("d:/★RGB-LWIR(멘토ver-최종)")
M3FD_ROOT    = BASE / "RGBThermal/data/M3FD"
FLIR_ROOT    = BASE / "FLIR_ADAS"
OUT_M3FD     = BASE / "RGBThermal/data/M3FD_yolo"
OUT_FLIR     = BASE / "RGBThermal/data/FLIR_yolo"
YAML_OUT     = BASE / "CFT_repo/data/multispectral/M3FD_FLIR.yaml"

# M3FD 클래스
M3FD_CLASSES = ['People', 'Car', 'Bus', 'Motorcycle', 'Lamp', 'Truck']
M3FD_CLS_MAP = {c: i for i, c in enumerate(M3FD_CLASSES)}

# FLIR category_id → M3FD class_id
FLIR_CAT_MAP = {0: 1, 1: 0}  # car→Car(1), person→People(0)

VAL_RATIO = 0.2


# ══════════════════════════════════════════════════════════════════════════
# STEP 1: M3FD 변환
# ══════════════════════════════════════════════════════════════════════════
def convert_m3fd():
    print("── [1/3] M3FD VOC → YOLO 변환 ──")
    ann_dir = M3FD_ROOT / "Annotation"
    vis_dir = M3FD_ROOT / "Vis"
    ir_dir  = M3FD_ROOT / "Ir"

    all_stems = sorted([p.stem for p in ann_dir.glob("*.xml")])
    random.shuffle(all_stems)
    n_val = int(len(all_stems) * VAL_RATIO)
    val_stems  = set(all_stems[:n_val])
    train_stems = set(all_stems[n_val:])
    print(f"  train: {len(train_stems)}, val: {len(val_stems)}")

    txt_files = {"train_rgb": [], "train_ir": [], "val_rgb": [], "val_ir": []}

    for split, stems in [("train", train_stems), ("val", val_stems)]:
        lbl_out = OUT_M3FD / split / "labels"
        rgb_out = OUT_M3FD / split / "rgb"
        ir_out  = OUT_M3FD / split / "ir"
        for d in [lbl_out, rgb_out, ir_out]:
            d.mkdir(parents=True, exist_ok=True)

        for stem in stems:
            xml_path = ann_dir / f"{stem}.xml"
            vis_path = vis_dir  / f"{stem}.png"
            ir_path  = ir_dir   / f"{stem}.png"

            if not vis_path.exists() or not ir_path.exists():
                continue

            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find("size")
            img_w = int(size.find("width").text)
            img_h = int(size.find("height").text)

            lines = []
            for obj in root.findall("object"):
                name = obj.find("name").text.strip()
                if name not in M3FD_CLS_MAP:
                    continue
                cls = M3FD_CLS_MAP[name]
                bb = obj.find("bndbox")
                xmin = float(bb.find("xmin").text)
                ymin = float(bb.find("ymin").text)
                xmax = float(bb.find("xmax").text)
                ymax = float(bb.find("ymax").text)
                cx = ((xmin + xmax) / 2) / img_w
                cy = ((ymin + ymax) / 2) / img_h
                w  = (xmax - xmin) / img_w
                h  = (ymax - ymin) / img_h
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            if not lines:
                continue

            shutil.copy2(vis_path, rgb_out / f"{stem}.png")
            shutil.copy2(ir_path,  ir_out  / f"{stem}.png")
            (lbl_out / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")

            txt_files[f"{split}_rgb"].append(str(rgb_out / f"{stem}.png").replace("\\", "/"))
            txt_files[f"{split}_ir"].append(str(ir_out   / f"{stem}.png").replace("\\", "/"))

    for key, lines in txt_files.items():
        (OUT_M3FD / f"{key}.txt").write_text("\n".join(lines), encoding="utf-8")
        print(f"  {key}.txt: {len(lines)}개")

    return txt_files


# ══════════════════════════════════════════════════════════════════════════
# STEP 2: FLIR 변환
# ══════════════════════════════════════════════════════════════════════════
def convert_flir():
    print("\n── [2/3] FLIR ADAS → YOLO 변환 ──")
    split_map = {"train": "train", "test": "val"}
    txt_files = {"train_rgb": [], "train_ir": [], "val_rgb": [], "val_ir": []}

    for flir_split, out_split in split_map.items():
        ann_path   = FLIR_ROOT / "coco_annotations" / f"{flir_split}.json"
        rgb_in_dir = FLIR_ROOT / "visible"  / flir_split
        ir_in_dir  = FLIR_ROOT / "thermal"  / flir_split

        with open(ann_path) as f:
            coco = json.load(f)

        img_info = {img["id"]: img for img in coco["images"]}
        ann_map  = defaultdict(list)
        for ann in coco["annotations"]:
            ann_map[ann["image_id"]].append(ann)

        rgb_out = OUT_FLIR / out_split / "rgb"
        ir_out  = OUT_FLIR / out_split / "ir"
        lbl_out = OUT_FLIR / out_split / "labels"
        for d in [rgb_out, ir_out, lbl_out]:
            d.mkdir(parents=True, exist_ok=True)

        skipped = 0
        for img_id, info in img_info.items():
            fname  = info["file_name"]
            img_w  = info["width"]
            img_h  = info["height"]
            stem   = Path(fname).stem

            rgb_path = rgb_in_dir / fname
            ir_path  = ir_in_dir  / fname
            if not rgb_path.exists() or not ir_path.exists():
                skipped += 1
                continue

            lines = []
            for ann in ann_map[img_id]:
                cat = ann["category_id"]
                if cat not in FLIR_CAT_MAP:
                    continue
                cls = FLIR_CAT_MAP[cat]
                x, y, w, h = ann["bbox"]
                cx = min(1.0, max(0.0, (x + w/2) / img_w))
                cy = min(1.0, max(0.0, (y + h/2) / img_h))
                nw = min(1.0, max(0.0, w / img_w))
                nh = min(1.0, max(0.0, h / img_h))
                if nw > 0 and nh > 0:
                    lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            if not lines:
                skipped += 1
                continue

            shutil.copy2(rgb_path, rgb_out / fname)
            shutil.copy2(ir_path,  ir_out  / fname)
            (lbl_out / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")

            txt_files[f"{out_split}_rgb"].append(str(rgb_out / fname).replace("\\", "/"))
            txt_files[f"{out_split}_ir"].append(str(ir_out   / fname).replace("\\", "/"))

        print(f"  [{flir_split}] 완료: {len(txt_files[f'{out_split}_rgb'])}개, 건너뜀: {skipped}개")

    return txt_files


# ══════════════════════════════════════════════════════════════════════════
# STEP 3: 통합 txt + yaml
# ══════════════════════════════════════════════════════════════════════════
def merge_and_write_yaml(m3fd_txt, flir_txt):
    print("\n── [3/3] 통합 txt 및 yaml 생성 ──")
    merged_dir = BASE / "RGBThermal/data/merged_txt"
    merged_dir.mkdir(parents=True, exist_ok=True)

    for key in ["train_rgb", "train_ir", "val_rgb", "val_ir"]:
        merged = m3fd_txt.get(key, []) + flir_txt.get(key, [])
        random.shuffle(merged)
        out_path = merged_dir / f"{key}.txt"
        out_path.write_text("\n".join(merged), encoding="utf-8")
        print(f"  {key}.txt: M3FD {len(m3fd_txt.get(key,[]))} + FLIR {len(flir_txt.get(key,[]))} = {len(merged)}개")

    mp = merged_dir.as_posix()
    yaml_content = f"""# M3FD + FLIR ADAS Aligned 통합 데이터셋
# M3FD 6클래스 + FLIR People/Car 보강 (train/val 각각 통합)

train_rgb: {mp}/train_rgb.txt
val_rgb:   {mp}/val_rgb.txt
train_ir:  {mp}/train_ir.txt
val_ir:    {mp}/val_ir.txt

nc: 6
names: ['People', 'Car', 'Bus', 'Motorcycle', 'Lamp', 'Truck']
"""
    YAML_OUT.parent.mkdir(parents=True, exist_ok=True)
    YAML_OUT.write_text(yaml_content, encoding="utf-8")
    print(f"\nyaml 저장: {YAML_OUT}")


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    m3fd_txt = convert_m3fd()
    flir_txt = convert_flir()
    merge_and_write_yaml(m3fd_txt, flir_txt)
    print("\n모든 작업 완료!")
    print(f"학습 시: --data CFT_repo/data/multispectral/M3FD_FLIR.yaml")
