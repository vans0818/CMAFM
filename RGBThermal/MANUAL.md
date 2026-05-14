# RGB-Thermal 융합 객체 검출 (Multispectral Object Detection)

## 완전 매뉴얼: 환경 세팅부터 학습, 평가, 시각화까지

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [환경 세팅](#2-환경-세팅)
3. [데이터셋 준비 (M3FD)](#3-데이터셋-준비-m3fd)
4. [프로젝트 구조](#4-프로젝트-구조)
5. [코드 설명](#5-코드-설명)
6. [학습 실행](#6-학습-실행)
7. [평가 및 추론](#7-평가-및-추론)
8. [시각화](#8-시각화)
9. [Ablation Study](#9-ablation-study)
10. [트러블슈팅](#10-트러블슈팅)

---

## 1. 프로젝트 개요

### 무엇을 하는 프로젝트인가?

RGB(가시광) 카메라와 Thermal(적외선) 카메라 이미지를 **동시에 활용**하여 보행자, 차량 등을 검출하는 딥러닝 모델입니다.

### 왜 두 개를 합치나?

| 상황 | RGB | Thermal | 융합 |
|------|-----|---------|------|
| 밝은 낮 | 잘 보임 | 보통 | 잘 보임 |
| 밤/저조도 | 안 보임 | 잘 보임 | **잘 보임** |
| 안개/연기 | 안 보임 | 잘 보임 | **잘 보임** |

### 모델 구조 요약

```
RGB 이미지  → [ResNet-50 #1] → 특징 추출 ─┐
                                           ├→ Cross-Modal Attention Fusion → FPN → Faster R-CNN → 검출 결과
Thermal 이미지 → [ResNet-50 #2] → 특징 추출 ─┘
```

### 성능 (M3FD 데이터셋, 6클래스)

| 지표 | 값 |
|------|-----|
| mAP@0.5 | **73.7%** |
| mAP@[.5:.95] | **40.6%** |

---

## 2. 환경 세팅

### 2-1. 필수 요구사항

- **OS**: Ubuntu 20.04 이상 (Linux 권장, Windows WSL2도 가능)
- **GPU**: NVIDIA GPU (VRAM 8GB 이상, 권장 16GB+)
- **CUDA**: 12.1 이상
- **Python**: 3.10

### 2-2. Conda 환경 생성

```bash
# 1. Miniconda 설치 (이미 있으면 건너뛰기)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# 설치 후 터미널 재시작

# 2. 새 환경 만들기
conda create -n rgbt python=3.10 -y
conda activate rgbt
```

### 2-3. PyTorch 설치 (GPU 버전)

```bash
# CUDA 12.1 기준 (본인의 CUDA 버전에 맞게 수정)
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
```

> **CUDA 버전 확인**: `nvidia-smi` 명령어로 확인 가능

> **다른 CUDA 버전**: https://pytorch.org/get-started/locally/ 에서 본인 환경에 맞는 명령어 확인

### 2-4. 추가 라이브러리 설치

```bash
pip install albumentations pycocotools matplotlib opencv-python tqdm gdown
```

### 2-5. 설치 확인

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA 사용 가능: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
import torchvision, albumentations, cv2
print('모든 라이브러리 정상!')
"
```

**정상 출력 예시:**
```
PyTorch: 2.4.0+cu121
CUDA 사용 가능: True
GPU: NVIDIA GeForce RTX 3090
VRAM: 24.0 GB
모든 라이브러리 정상!
```

> **CUDA 사용 가능이 False인 경우**: PyTorch GPU 버전이 올바르게 설치되지 않았습니다.
> `pip uninstall torch torchvision` 후 2-3단계를 다시 수행하세요.

---

## 3. 데이터셋 준비 (M3FD)

### 3-1. M3FD 데이터셋이란?

- **4,200장**의 aligned RGB-Thermal 이미지 쌍
- **6개 클래스**: People, Car, Bus, Motorcycle, Lamp, Truck
- **34,407개** 객체 어노테이션
- 주간/야간/흐림/역광 등 다양한 조건 포함
- 해상도: 1024x768

### 3-2. 다운로드

**방법 A: gdown 사용 (권장)**

```bash
cd /path/to/your/project    # 프로젝트 폴더로 이동
mkdir -p data
cd data

# Google Drive에서 다운로드 (약 5.7GB)
gdown --folder "https://drive.google.com/drive/folders/1H-oO7bgRuVFYDcMGvxstT1nmy0WF_Y_6?usp=sharing"

# M3FD_Detection.zip만 필요 (나머지는 삭제 가능)
unzip M3FD_Detection.zip
```

> **gdown 오류 시**: Google Drive 다운로드 제한에 걸린 경우입니다.
> 브라우저에서 직접 다운로드하세요:
> https://drive.google.com/drive/folders/1H-oO7bgRuVFYDcMGvxstT1nmy0WF_Y_6

**방법 B: 직접 다운로드**

1. 위 Google Drive 링크를 브라우저에서 열기
2. `M3FD_Detection.zip` (5.7GB) 다운로드
3. 프로젝트의 `data/` 폴더에 압축 해제

### 3-3. 데이터 정리

압축 해제 후, 아래 구조가 되도록 정리합니다:

```
data/
└── M3FD/
    ├── Vis/          # RGB 이미지 (4,200개 PNG)
    │   ├── 00000.png
    │   ├── 00001.png
    │   └── ...
    ├── Ir/           # Thermal 이미지 (4,200개 PNG)
    │   ├── 00000.png
    │   ├── 00001.png
    │   └── ...
    └── Annotation/   # VOC XML 어노테이션 (4,200개)
        ├── 00000.xml
        ├── 00001.xml
        └── ...
```

> **중요**: `Ir/`, `Vis/`, `Annotation/` 폴더가 `M3FD/` 안에 있어야 합니다.
> 압축 해제 시 최상위 폴더 없이 풀리면 직접 `M3FD/` 폴더를 만들어 옮겨주세요.

### 3-4. 데이터 경로 설정

`config.py`의 `DataConfig.root`를 본인 경로에 맞게 수정합니다:

```python
# config.py 10번째 줄
root: str = "/path/to/your/project/data/M3FD"  # ← 본인 경로로 수정
```

### 3-5. 데이터 검증

```bash
python -c "
from config import Config
from dataset import build_datasets
cfg = Config()
train_ds, val_ds = build_datasets(cfg.data)
rgb, thermal, target = train_ds[0]
print(f'RGB shape: {rgb.shape}')        # torch.Size([3, 640, 640])
print(f'Thermal shape: {thermal.shape}') # torch.Size([3, 640, 640])
print(f'Boxes: {target[\"boxes\"].shape}')
print(f'Labels: {target[\"labels\"]}')
print('데이터 로딩 성공!')
"
```

---

## 4. 프로젝트 구조

```
military/
│
│  # ====== 핵심 코드 ======
├── config.py              # 모든 설정 (데이터 경로, 모델 구조, 학습 하이퍼파라미터)
├── dataset.py             # M3FD 데이터 로더 (RGB+Thermal 쌍 로딩, augmentation)
├── model.py               # 메인 모델: Dual-Backbone + Cross-Modal Attention + Faster R-CNN
├── train.py               # 학습 스크립트
├── evaluate.py            # COCO mAP 평가
│
│  # ====== 추론/시각화 ======
├── inference.py            # 단일 이미지 추론 + bbox 그리기
├── visualize_pipeline.py   # 파이프라인 단계별 시각화
│
│  # ====== Ablation Study ======
├── ablation_models.py      # 비교 모델들 (RGB-only, Thermal-only, Early Fusion 등)
├── run_ablation.py         # Ablation 전체 실행 스크립트
│
│  # ====== 기타 ======
├── download_data.py        # FLIR ADAS 데이터 준비 (미사용)
├── MANUAL.md               # 이 문서
│
│  # ====== 데이터/결과 ======
├── data/
│   └── M3FD/               # 데이터셋 (별도 다운로드 필요)
└── runs/
    ├── best.pth             # Best 모델 체크포인트
    ├── epoch_029.pth        # 마지막 epoch 체크포인트
    ├── ablation/            # Ablation study 결과
    └── visualizations/      # 시각화 이미지
```

---

## 5. 코드 설명

### 5-1. config.py — 설정

모든 하이퍼파라미터가 여기 있습니다. 바꿀 일이 가장 많은 파일입니다.

```python
@dataclass
class DataConfig:
    root: str = ".../M3FD"           # 데이터 경로 ← 반드시 수정
    num_classes: int = 6             # 클래스 수
    img_size: Tuple = (640, 640)     # 입력 이미지 크기

@dataclass
class TrainConfig:
    epochs: int = 30                 # 학습 반복 횟수
    batch_size: int = 8              # 배치 크기 (VRAM 부족 시 줄이기)
    lr: float = 0.005                # 학습률
```

> **GPU VRAM이 부족한 경우**: `batch_size`를 4 또는 2로 줄이세요.

### 5-2. model.py — 모델 구조

핵심 모듈 3개:

1. **DualBackboneWithFPN**: 두 개의 ResNet-50으로 각각 RGB/Thermal feature 추출 → FPN
2. **CrossModalAttentionFusion**: Channel cross-attention + Spatial cross-gating으로 두 modality 융합
3. **MultispectralDetector**: 위 모듈들 + Faster R-CNN을 결합한 전체 모델

### 5-3. dataset.py — 데이터 로더

- M3FD의 VOC XML 어노테이션을 파싱
- RGB + Thermal을 쌍으로 로딩
- Albumentations로 동일한 augmentation 적용 (HFlip, Brightness 등)
- Train/Val 자동 분할 (8:2)

### 5-4. train.py — 학습

- Warmup LR scheduler (첫 epoch)
- Differential LR: backbone 0.1x, 나머지 1x
- Mixed Precision (AMP)로 속도 향상
- 매 epoch mAP 평가 + best model 저장

---

## 6. 학습 실행

### 6-1. 기본 학습

```bash
cd /path/to/your/project    # 프로젝트 폴더로 이동
python train.py
```

이것만 실행하면 기본 설정(30 epoch, batch=8)으로 학습됩니다.

### 6-2. 옵션 변경

```bash
# epoch 수 변경
python train.py --epochs 50

# 배치 크기 변경 (VRAM 부족 시)
python train.py --batch-size 4

# 학습률 변경
python train.py --lr 0.01

# backbone 변경 (더 큰 모델)
python train.py --backbone resnet101

# Mixed Precision 비활성화 (정밀도 문제 시)
python train.py --no-amp

# 전부 조합
python train.py --epochs 50 --batch-size 4 --lr 0.003
```

### 6-3. 학습 중단 후 재개

```bash
# 체크포인트에서 이어서 학습
python train.py --resume runs/epoch_014.pth
```

### 6-4. 학습 출력 예시

```
Device: cuda
Config: backbone=resnet50, epochs=30, batch_size=8, lr=0.005, amp=True
Dataset split: 3360 train / 840 val
Model built: 191.0M params (191.0M trainable)

--- Epoch 0/29 ---
  [50/420] loss=1.5662  lr=0.000060  (24.8s)
  [100/420] loss=1.0874  lr=0.000120  (48.5s)
  ...
Epoch 0 done — avg_loss=0.6455  time=214.7s
  Val mAP@0.5=0.1402  mAP@[.5:.95]=0.0482
  ★ New best mAP@0.5: 0.1402
```

### 6-5. GPU 메모리별 권장 설정

| GPU VRAM | batch_size | img_size | 예상 학습 시간 (30ep) |
|----------|-----------|----------|-------------------|
| 8GB | 2 | (512, 512) | ~3시간 |
| 12GB | 4 | (640, 640) | ~2.5시간 |
| 16GB | 6 | (640, 640) | ~2시간 |
| 24GB+ | 8 | (640, 640) | ~1.5시간 |

> **img_size 변경**: `config.py`의 `DataConfig.img_size`를 `(512, 512)`로 수정

---

## 7. 평가 및 추론

### 7-1. 학습된 모델 평가

```bash
# Best 모델로 validation set 평가
python train.py --eval-only --resume runs/best.pth
```

출력:
```
mAP@0.5: 0.7371  mAP@[.5:.95]: 0.4058
```

### 7-2. 단일 이미지 추론

```bash
python inference.py \
    --checkpoint runs/best.pth \
    --rgb /path/to/rgb_image.jpg \
    --thermal /path/to/thermal_image.jpg \
    --output result.jpg \
    --score-thresh 0.5
```

> `--score-thresh`: 검출 신뢰도 임계값 (0.3~0.7 범위 조절)

### 7-3. M3FD 이미지로 테스트

```bash
# 데이터셋의 이미지로 빠르게 테스트
python inference.py \
    --checkpoint runs/best.pth \
    --rgb data/M3FD/Vis/00100.png \
    --thermal data/M3FD/Ir/00100.png \
    --output test_result.jpg
```

---

## 8. 시각화

### 8-1. 파이프라인 단계별 시각화

```bash
python visualize_pipeline.py
```

`runs/visualizations/` 폴더에 6개 샘플의 시각화가 생성됩니다:

- `pipeline_sample_N.png`: 9칸 그리드 (입력 → feature → 검출)
- `fusion_detail_sample_N.png`: C3 레벨 feature fusion 상세

### 8-2. 시각화 내용

```
┌─────────────────┬──────────────────┬────────────────┐
│ Step 1: RGB     │ Step 2: Thermal  │ Ground Truth   │
│ (원본 이미지)    │ (적외선 이미지)   │ (정답 bbox)    │
├─────────────────┼──────────────────┼────────────────┤
│ Step 3: RGB     │ Step 4: Thermal  │ Step 5: Fused  │
│ Features (C4)   │ Features (C4)    │ Features (C4)  │
│ (RGB 특징맵)    │ (Thermal 특징맵)  │ (융합 특징맵)   │
├─────────────────┼──────────────────┼────────────────┤
│ Step 6: Det     │ Step 7: Det on   │ Step 8: Fused  │
│ on RGB          │ Thermal          │ Overlay + Det  │
│ (RGB에 검출)    │ (Thermal에 검출)  │ (합성+검출)    │
└─────────────────┴──────────────────┴────────────────┘
```

---

## 9. Ablation Study

### 9-1. 실행

```bash
# 전체 ablation (4개 변형 + full 모델, 약 2~3시간)
python run_ablation.py --epochs 20

# 특정 변형만 실행
python run_ablation.py --variants rgb_only thermal_only --epochs 20

# GPU 메모리 적은 경우
python run_ablation.py --epochs 20 --batch-size 4
```

### 9-2. 비교 대상

| 변형 | 설명 | 목적 |
|------|------|------|
| `rgb_only` | RGB 단일 모달리티 | Thermal 추가 효과 측정 |
| `thermal_only` | Thermal 단일 모달리티 | RGB 추가 효과 측정 |
| `early_fusion` | 6ch 입력 concat, 단일 backbone | 간단한 융합 baseline |
| `dual_no_attn` | 두 backbone + concat (attention 없음) | Cross-Modal Attention 효과 측정 |
| `full` | 두 backbone + Cross-Modal Attention | **제안 모델** |

### 9-3. 우리 실험 결과

```
Variant              mAP@0.5    mAP@[.5:.95]
────────────────────────────────────────────
thermal_only          0.528       0.273
rgb_only              0.632       0.322
early_fusion          0.650       0.346
dual_no_attn          0.700       0.378
full (ours)           0.737       0.406
```

### 9-4. 결과 해석

- 융합 효과: RGB-only → Full = **+10.5%p**
- Dual backbone 효과: Early Fusion → Dual+Concat = **+5.0%p**
- Cross-Modal Attention 효과: Dual+Concat → Full = **+3.7%p**

---

## 10. 트러블슈팅

### Q: `CUDA out of memory` 에러

**원인**: GPU 메모리 부족

**해결**:
```bash
# 배치 크기 줄이기
python train.py --batch-size 4   # 또는 2

# 이미지 크기 줄이기 (config.py 수정)
# img_size: Tuple[int, int] = (512, 512)  # 640 → 512
```

### Q: `ModuleNotFoundError: No module named 'xxx'`

**원인**: 라이브러리 미설치

**해결**:
```bash
pip install albumentations pycocotools matplotlib opencv-python
```

### Q: 데이터 로딩 시 `FileNotFoundError`

**원인**: 데이터 경로 불일치

**해결**:
1. `config.py`의 `DataConfig.root` 경로 확인
2. 해당 경로에 `Vis/`, `Ir/`, `Annotation/` 폴더가 있는지 확인
```bash
ls /your/path/M3FD/
# 출력: Annotation  Ir  Vis
```

### Q: 학습은 되는데 mAP가 0

**가능한 원인**: 어노테이션의 클래스명이 config와 불일치

**확인**:
```bash
python -c "
import xml.etree.ElementTree as ET
tree = ET.parse('data/M3FD/Annotation/00000.xml')
for obj in tree.getroot().findall('object'):
    print(obj.find('name').text)
"
```
출력된 클래스명이 `config.py`의 `target_classes`와 정확히 일치해야 합니다.

### Q: Windows에서 실행하고 싶어요

1. **WSL2 사용 권장** (Windows Subsystem for Linux)
2. WSL2 설치 후 Ubuntu를 설치하면 Linux와 동일하게 사용 가능
3. NVIDIA GPU 드라이버가 WSL2를 지원해야 함

### Q: 학습을 이어서 하고 싶어요

```bash
python train.py --resume runs/epoch_014.pth --epochs 50
```

### Q: 내 데이터셋에 적용하고 싶어요

1. M3FD와 같은 구조로 데이터 준비:
   - `Vis/` (RGB), `Ir/` (Thermal), `Annotation/` (VOC XML)
   - RGB와 Thermal 파일명이 동일해야 함
2. `config.py`의 `target_classes`를 내 클래스명으로 수정
3. `num_classes`를 클래스 수에 맞게 수정
4. 학습 실행

---

## 부록: 빠른 시작 요약

```bash
# 1. 환경 설치
conda create -n rgbt python=3.10 -y && conda activate rgbt
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install albumentations pycocotools matplotlib opencv-python gdown

# 2. 코드 가져오기
git clone <repository_url>
cd military

# 3. 데이터 다운로드
mkdir -p data && cd data
gdown --folder "https://drive.google.com/drive/folders/1H-oO7bgRuVFYDcMGvxstT1nmy0WF_Y_6"
unzip M3FD_Detection.zip && mkdir M3FD && mv Ir Vis Annotation M3FD/
cd ..

# 4. config.py에서 데이터 경로 수정
# root: str = "/your/absolute/path/data/M3FD"

# 5. 학습
python train.py --epochs 30 --batch-size 8

# 6. 평가
python train.py --eval-only --resume runs/best.pth

# 7. 시각화
python visualize_pipeline.py

# 8. Ablation
python run_ablation.py --epochs 20
```
