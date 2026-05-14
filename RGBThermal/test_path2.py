# -*- coding: utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import numpy as np
import cv2

# Use raw string with actual path
data_path = Path("d:/멘토님 수정ver/RGBThermal/data/M3FD")
test_img = data_path / "Vis" / "00000.png"

print("path:", repr(str(test_img)))
print("exists:", test_img.exists())
print("size:", test_img.stat().st_size if test_img.exists() else "N/A")

# Try reading with io
import io
try:
    with io.open(str(test_img), 'rb') as f:
        data = f.read()
    print("io.open len:", len(data))
    if len(data) > 0:
        buf = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        print("img shape:", img.shape if img is not None else None)
except Exception as e:
    print("io.open error:", e)

# Try PIL
try:
    from PIL import Image
    img_pil = Image.open(str(test_img))
    print("PIL shape:", np.array(img_pil).shape)
except Exception as e:
    print("PIL error:", e)
