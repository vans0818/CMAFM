import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.insert(0, 'd:/멘토님 수정ver/RGBThermal')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

from config import Config
from dataset import build_datasets
cfg = Config()
print('Data root:', cfg.data.root)
train_ds, val_ds = build_datasets(cfg.data)
rgb, thermal, target = train_ds[0]
print(f'RGB shape: {rgb.shape}')
print(f'Thermal shape: {thermal.shape}')
print(f'Boxes: {target["boxes"].shape}')
print('데이터 로딩 성공!')
