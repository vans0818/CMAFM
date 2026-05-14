"""RTDOD 데이터셋에서 RGB + Thermal 샘플 3쌍 추출."""

import shutil
from pathlib import Path

# 경로 설정 — 실제 RTDOD 다운로드 경로로 변경하세요
RTDOD_ROOT = Path(r"C:\Users\CAU\Downloads")
RGB_DIR = RTDOD_ROOT / "rgb"
THERMAL_DIR = RTDOD_ROOT / "thermal"

OUTPUT_DIR = Path(__file__).parent / "data" / "RTDOD_samples"
OUTPUT_RGB = OUTPUT_DIR / "rgb"
OUTPUT_THERMAL = OUTPUT_DIR / "thermal"

OUTPUT_RGB.mkdir(parents=True, exist_ok=True)
OUTPUT_THERMAL.mkdir(parents=True, exist_ok=True)

# RGB 파일 기준으로 정렬 후 3개 선택
rgb_files = sorted(RGB_DIR.glob("*.png"))[:3]

for rgb_path in rgb_files:
    thermal_path = THERMAL_DIR / rgb_path.name

    if not thermal_path.exists():
        print(f"[SKIP] Thermal 없음: {rgb_path.name}")
        continue

    shutil.copy(rgb_path, OUTPUT_RGB / rgb_path.name)
    shutil.copy(thermal_path, OUTPUT_THERMAL / rgb_path.name)
    print(f"[OK] {rgb_path.name}")

print(f"\n완료: {OUTPUT_DIR}")
