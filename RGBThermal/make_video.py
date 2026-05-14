"""RTDOD PNG 프레임 시퀀스를 RGB + Thermal 나란히 배치한 mp4로 변환."""

import cv2
import numpy as np
from pathlib import Path

# 경로 설정
RGB_DIR     = Path(r"C:\Users\CAU\Downloads\RTDOD\rgb\rgb")
THERMAL_DIR = Path(r"C:\Users\CAU\Downloads\RTDOD\trm")
OUTPUT_DIR  = Path(r"C:\Users\CAU\Downloads\RTDOD\videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 20  # RTDOD 원본 촬영 fps

# 시퀀스별로 그룹화 (D_0000_*, D_0001_* 등)
rgb_files = sorted(RGB_DIR.glob("*.png"))
sequences = {}
for f in rgb_files:
    seq_id = f.name.split("_")[1]  # "0000"
    sequences.setdefault(seq_id, []).append(f)

print(f"총 {len(sequences)}개 시퀀스 발견: {list(sequences.keys())}")

for seq_id, frames in sequences.items():
    frames = sorted(frames)
    out_path = OUTPUT_DIR / f"seq_{seq_id}_rgb_thermal.mp4"

    # 첫 프레임으로 크기 파악
    sample_rgb = cv2.imread(str(frames[0]))
    h, w = sample_rgb.shape[:2]
    # RGB + Thermal 좌우 배치 → 가로 2배
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (w * 2, h),
    )

    for f in frames:
        thermal_path = THERMAL_DIR / f.name
        rgb = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if rgb is None:
            continue
        if rgb.ndim == 2:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)

        if thermal_path.exists():
            thermal = cv2.imread(str(thermal_path), cv2.IMREAD_COLOR)
            if thermal is None:
                thermal = np.zeros_like(rgb)
            else:
                if thermal.ndim == 2:
                    thermal = cv2.cvtColor(thermal, cv2.COLOR_GRAY2BGR)
                if thermal.shape[:2] != (h, w):
                    thermal = cv2.resize(thermal, (w, h))
        else:
            thermal = np.zeros_like(rgb)

        # 좌: RGB / 우: Thermal 나란히
        frame = np.hstack([rgb, thermal])

        # 레이블 표시
        cv2.putText(frame, "RGB",     (10, 30),    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, "Thermal", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        writer.write(frame)

    writer.release()
    print(f"[OK] {out_path.name}  ({len(frames)} 프레임)")

print(f"\n완료: {OUTPUT_DIR}")
