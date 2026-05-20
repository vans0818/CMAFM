"""
RGB / Thermal / CMAFM-YOLO / CMAFM(Faster R-CNN) 4열 비교 이미지 생성
"""
import sys, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CFT_DIR  = os.path.join(BASE_DIR, "CFT_repo")
sys.path.insert(0, CFT_DIR)
os.chdir(CFT_DIR)

import cv2
import torch
import numpy as np
from pathlib import Path

# ── 경로 설정 ────────────────────────────────────────────────────────────────
BASE     = Path(BASE_DIR)
WEIGHTS  = BASE / "CFT_repo/runs/train/cmafm_m3fd_flir/weights/best.pt"
RGB_DIR  = BASE / "RGBThermal/data/M3FD_yolo/val/rgb"
IR_DIR   = BASE / "RGBThermal/data/M3FD_yolo/val/ir"
FRCNN_DIR= BASE / "RGBThermal/runs/visualizations"
OUT_DIR  = BASE / "RGBThermal/runs/paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES  = ['People','Car','Bus','Motorcycle','Lamp','Truck']
COLORS   = [(0,255,0),(0,128,255),(255,0,0),(0,0,255),(255,255,0),(255,0,255)]
IMG_SIZE = 640
CONF_THR = 0.35
IOU_THR  = 0.45
ROW_H    = 280

# ── 모델 로드 ────────────────────────────────────────────────────────────────
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

device = select_device('0')
model  = attempt_load(str(WEIGHTS), map_location=device)
model.half().eval()
print(f"Model loaded")

# ── 유틸 ─────────────────────────────────────────────────────────────────────
def read_img(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)

def letterbox_img(img, size=640):
    h, w   = img.shape[:2]
    r      = size / max(h, w)
    nw, nh = int(w*r), int(h*r)
    img    = cv2.resize(img, (nw, nh))
    dw     = (size - nw) // 2
    dh     = (size - nh) // 2
    img    = cv2.copyMakeBorder(img, dh, size-nh-dh, dw, size-nw-dw,
                                 cv2.BORDER_CONSTANT, value=(114,114,114))
    return img, r, dw, dh

def resize_h(img, H=ROW_H):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w*H/h), H))

def add_title(img, text, color=(255,255,255)):
    out = img.copy()
    cv2.rectangle(out, (0,0), (img.shape[1], 26), (30,30,30), -1)
    cv2.putText(out, text, (6,19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
    return out

# ── YOLO 추론 ─────────────────────────────────────────────────────────────────
def yolo_detect(rgb_path, ir_path):
    img0_rgb = read_img(rgb_path)
    img0_ir  = read_img(ir_path)
    h0, w0   = img0_rgb.shape[:2]
    lb_rgb, r, dw, dh = letterbox_img(img0_rgb, IMG_SIZE)
    lb_ir, *_          = letterbox_img(img0_ir,  IMG_SIZE)
    t_rgb = torch.from_numpy(lb_rgb[:,:,::-1].copy()).permute(2,0,1).unsqueeze(0).half().to(device)/255.
    t_ir  = torch.from_numpy(lb_ir [:,:,::-1].copy()).permute(2,0,1).unsqueeze(0).half().to(device)/255.
    with torch.no_grad():
        pred = model(t_rgb, t_ir)[0]
    dets = non_max_suppression(pred, CONF_THR, IOU_THR)[0]
    result = img0_rgb.copy()
    if dets is not None and len(dets):
        dets[:,:4] = scale_coords((IMG_SIZE,IMG_SIZE), dets[:,:4],
                                   (h0,w0), ratio_pad=((r,r),(dh,dw))).round()
        for *xyxy, conf, cls in dets:
            x1,y1,x2,y2 = map(int, xyxy)
            c   = int(cls)
            col = COLORS[c % len(COLORS)]
            cv2.rectangle(result, (x1,y1), (x2,y2), col, 2)
            cv2.putText(result, f"{CLASSES[c]} {conf:.2f}",
                        (x1, max(y1-4,12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
    return img0_rgb, img0_ir, result

# ── Faster R-CNN 결과 추출 ─────────────────────────────────────────────────────
def get_frcnn_det(pipeline_img_path):
    """
    pipeline_sample 이미지: 3행 x 3열 그리드
    Row0: RGB Input / Thermal Input / Ground Truth
    Row1: RGB Features / Thermal Features / Fused Features
    Row2: Detection on RGB / Detection on Thermal / Fused Overlay+Detection  ← 사용
    마지막 셀(row2, col2) = Faster R-CNN Fused Detection 결과
    """
    img = read_img(pipeline_img_path)
    if img is None:
        return None
    h, w  = img.shape[:2]
    # 상단 타이틀 바 제거 (약 5%)
    top   = int(h * 0.05)
    body  = img[top:, :]
    bh, bw = body.shape[:2]
    row_h = bh // 3
    col_w = bw // 3
    # 3행 3열의 마지막 셀
    cell  = body[row_h*2:row_h*3, col_w*2:col_w*3]
    return cell

# ── 샘플 선별 ─────────────────────────────────────────────────────────────────
def get_samples(n=3, night=True):
    out = []
    for f in sorted(RGB_DIR.iterdir()):
        if not (IR_DIR/f.name).exists():
            continue
        img = read_img(f)
        if img is None:
            continue
        b = img.mean()
        cond = (b < 55) if night else (75 <= b <= 145)
        if cond:
            out.append(f.name)
        if len(out) >= n:
            break
    return out

# ── 비교 패널 생성 ─────────────────────────────────────────────────────────────
def make_panel(samples, scene_label, out_path, pipeline_indices):
    rows = []
    for i, fname in enumerate(samples):
        rgb_img, ir_img, yolo_img = yolo_detect(RGB_DIR/fname, IR_DIR/fname)

        # Faster R-CNN 결과 (pipeline_sample 이미지에서 추출)
        pidx = pipeline_indices[i]
        frcnn_path = FRCNN_DIR / f"pipeline_sample_{pidx}.png"
        frcnn_img  = get_frcnn_det(frcnn_path)

        rgb_r  = add_title(resize_h(rgb_img),  "(a) RGB Input")
        ir_r   = add_title(resize_h(ir_img),   "(b) Thermal Input")
        yolo_r = add_title(resize_h(yolo_img), "(c) CMAFM-YOLO", (80,255,80))

        if frcnn_img is not None:
            fr_r  = add_title(resize_h(frcnn_img), "(d) CMAFM (Faster R-CNN)", (80,180,255))
            row   = np.hstack([rgb_r, ir_r, yolo_r, fr_r])
        else:
            row   = np.hstack([rgb_r, ir_r, yolo_r])

        rows.append(row)

    max_w = max(r.shape[1] for r in rows)
    uniform = [cv2.copyMakeBorder(r,0,0,0,max_w-r.shape[1],
                cv2.BORDER_CONSTANT,value=(20,20,20)) for r in rows]
    sep   = np.full((3, max_w, 3), 60, dtype=np.uint8)
    final = uniform[0]
    for r in uniform[1:]:
        final = np.vstack([final, sep, r])

    title = np.full((44, max_w, 3), 20, dtype=np.uint8)
    cv2.putText(title, scene_label, (12,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220,220,220), 2, cv2.LINE_AA)
    final = np.vstack([title, final])

    cv2.imencode('.png', final)[1].tofile(str(out_path))
    print(f"Saved: {out_path.name}")

# ── 실행 ─────────────────────────────────────────────────────────────────────
print("Selecting samples...")
night_s = get_samples(3, night=True)
day_s   = get_samples(3, night=False)
print(f"Night: {night_s}")
print(f"Day:   {day_s}")

# pipeline_sample 1~3: 야간, 4~6: 주간 (기존 생성 순서 기준)
make_panel(night_s, "Night Scene | RGB / Thermal / CMAFM-YOLO / CMAFM(Faster R-CNN)",
           OUT_DIR/"comparison_night_yolo_frcnn.png", [4,5,6])

make_panel(day_s, "Day Scene   | RGB / Thermal / CMAFM-YOLO / CMAFM(Faster R-CNN)",
           OUT_DIR/"comparison_day_yolo_frcnn.png", [1,2,3])

print("Done!")
