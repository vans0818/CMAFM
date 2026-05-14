"""M3FD 00865~00969 PNG 프레임을 RGB+Thermal 나란히 배치한 mp4로 변환."""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image


def read_img(path: Path) -> np.ndarray:
    """한글 경로 대응 이미지 읽기 (PIL 경유)."""
    img = Image.open(path).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


VIS_DIR    = Path(r"D:\★RGB-LWIR(멘토ver-최종)\RGBThermal\data\M3FD\Vis")
IR_DIR     = Path(r"D:\★RGB-LWIR(멘토ver-최종)\RGBThermal\data\M3FD\Ir")
Path(r"D:\★RGB-LWIR(멘토ver-최종)\RGBThermal\data\M3FD").mkdir(parents=True, exist_ok=True)

START, END = 865, 969
FPS = 10

frames = [f"{i:05d}" for i in range(START, END + 1)]
frames = [f for f in frames if (VIS_DIR / f"{f}.png").exists()]

# 첫 프레임으로 크기 파악
sample = read_img(VIS_DIR / f"{frames[0]}.png")
h, w = sample.shape[:2]

OUTPUT_RGB     = Path(r"D:\★RGB-LWIR(멘토ver-최종)\RGBThermal\data\M3FD\m3fd_00865_00969_rgb.mp4")
OUTPUT_THERMAL = Path(r"D:\★RGB-LWIR(멘토ver-최종)\RGBThermal\data\M3FD\m3fd_00865_00969_thermal.mp4")

writer_rgb = cv2.VideoWriter(str(OUTPUT_RGB),     cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
writer_ir  = cv2.VideoWriter(str(OUTPUT_THERMAL), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))

for fid in frames:
    rgb = read_img(VIS_DIR / f"{fid}.png")
    ir_path = IR_DIR / f"{fid}.png"
    ir = read_img(ir_path) if ir_path.exists() else np.zeros_like(rgb)

    if ir.shape[:2] != (h, w):
        ir = cv2.resize(ir, (w, h))

    cv2.putText(rgb, f"RGB {fid}",     (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0),   2)
    cv2.putText(ir,  f"Thermal {fid}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

    writer_rgb.write(rgb)
    writer_ir.write(ir)

writer_rgb.release()
writer_ir.release()
print(f"완료: {OUTPUT_RGB.name}  ({len(frames)} 프레임)")
print(f"완료: {OUTPUT_THERMAL.name}  ({len(frames)} 프레임)")
