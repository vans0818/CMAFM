import json
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET

BASE = Path("d:/★RGB-LWIR(멘토ver-최종)")

# ── FLIR 분석 ──────────────────────────────────
flir_root = BASE / "FLIR_ADAS"
print("=" * 50)
print("FLIR ADAS Aligned")
print("=" * 50)
total_flir = {"images": 0, "labels": Counter(), "boxes": 0}
for split in ["train", "test"]:
    with open(flir_root / "coco_annotations" / f"{split}.json") as f:
        d = json.load(f)
    cats = {c["id"]: c["name"] for c in d["categories"]}
    cnt = Counter(cats[a["category_id"]] for a in d["annotations"])
    w, h = d["images"][0]["width"], d["images"][0]["height"]
    print(f"\n  [{split}]")
    print(f"    이미지 수  : {len(d['images'])}")
    print(f"    해상도     : {w} x {h}")
    print(f"    총 박스 수 : {len(d['annotations'])}")
    for k, v in sorted(cnt.items(), key=lambda x: -x[1]):
        print(f"      {k:12s}: {v}")
    total_flir["images"] += len(d["images"])
    total_flir["labels"].update(cnt)
    total_flir["boxes"] += len(d["annotations"])

print(f"\n  [합계]")
print(f"    이미지 수  : {total_flir['images']}")
print(f"    총 박스 수 : {total_flir['boxes']}")
for k, v in sorted(total_flir["labels"].items(), key=lambda x: -x[1]):
    print(f"      {k:12s}: {v}")

# ── M3FD 분석 ──────────────────────────────────
m3fd_root = BASE / "RGBThermal/data/M3FD/Annotation"
print("\n" + "=" * 50)
print("M3FD")
print("=" * 50)
cnt = Counter()
sizes = Counter()
xml_files = list(m3fd_root.glob("*.xml"))
total_boxes = 0
for xf in xml_files:
    tree = ET.parse(xf)
    root = tree.getroot()
    sz = root.find("size")
    sizes[f"{sz.find('width').text}x{sz.find('height').text}"] += 1
    objs = root.findall("object")
    total_boxes += len(objs)
    for obj in objs:
        cnt[obj.find("name").text] += 1

print(f"\n  이미지 수  : {len(xml_files)} (RGB + IR 쌍)")
print(f"  해상도     : {dict(sizes)}")
print(f"  총 박스 수 : {total_boxes}")
print(f"  클래스 수  : {len(cnt)}")
for k, v in sorted(cnt.items(), key=lambda x: -x[1]):
    print(f"    {k:12s}: {v}")

# ── 비교 요약 ───────────────────────────────────
print("\n" + "=" * 50)
print("비교 요약")
print("=" * 50)
print(f"{'항목':<20} {'M3FD':>12} {'FLIR ADAS':>12}")
print("-" * 46)
print(f"{'이미지 쌍':<20} {len(xml_files):>12,} {total_flir['images']:>12,}")
print(f"{'총 박스':<20} {total_boxes:>12,} {total_flir['boxes']:>12,}")
print(f"{'클래스 수':<20} {len(cnt):>12} {'4':>12}")
print(f"{'해상도':<20} {'1024x768':>12} {'640x512':>12}")
print(f"{'촬영 환경':<20} {'주/야간 혼합':>12} {'주/야간 혼합':>12}")
print(f"{'RGB+IR 정렬':<20} {'완전 정렬':>12} {'완전 정렬':>12}")
