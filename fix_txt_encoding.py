"""txt 파일 BOM 제거 + 경로 슬래시 통일 (Python으로 안전하게)"""
from pathlib import Path

merged_dir = Path("d:/★RGB-LWIR(멘토ver-최종)/RGBThermal/data/merged_txt")

for fname in ["train_rgb.txt", "train_ir.txt", "val_rgb.txt", "val_ir.txt"]:
    fpath = merged_dir / fname
    # utf-8-sig: BOM 자동 제거
    lines = fpath.read_text(encoding="utf-8-sig").strip().splitlines()
    # 슬래시 통일
    lines = [l.replace("\\", "/") for l in lines if l.strip()]
    # BOM 없는 utf-8로 저장
    fpath.write_text("\n".join(lines), encoding="utf-8")
    print(f"완료: {fname} ({len(lines)}줄) 첫줄={lines[0]}")
