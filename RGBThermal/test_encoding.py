import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
print("fs encoding:", sys.getfilesystemencoding())
print("stdout encoding:", sys.stdout.encoding)
print("locale:", __import__('locale').getpreferredencoding())

from pathlib import Path
import numpy as np, cv2

# Try with the actual path from __file__
here = Path(__file__).parent
vis_path = here / "data" / "M3FD" / "Vis" / "00000.png"
print("vis_path:", vis_path)
print("exists:", vis_path.exists())
print("stat size:", vis_path.stat().st_size)

data = vis_path.read_bytes()
print("read_bytes len:", len(data))
buf = np.frombuffer(data, dtype=np.uint8)
img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
print("img:", img.shape if img is not None else "None")
