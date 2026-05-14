import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import numpy as np
import cv2

# Check what path the dataset uses
from config import DataConfig
cfg = DataConfig()
print("root:", repr(cfg.root))

p = Path(cfg.root) / "Vis" / "00000.png"
print("path:", repr(str(p)))
print("exists:", p.exists())

# Try reading
try:
    data = p.read_bytes()
    print("read_bytes len:", len(data))
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    print("imdecode shape:", img.shape if img is not None else None)
except Exception as e:
    print("Error:", e)

# Try os.fsencode workaround
try:
    with open(str(p).encode('utf-8'), 'rb') as f:
        data2 = f.read()
    print("utf-8 open len:", len(data2))
except Exception as e:
    print("utf-8 open error:", e)
