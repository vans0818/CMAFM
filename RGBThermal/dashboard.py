"""
CMAFM Dashboard — RGB+Thermal Multispectral Object Detection
실시간 영상/이미지 업로드 → 객체 탐지 결과 시각화
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CMAFM Tactical Detection System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Military Dark Navy Theme ──────────────────────────────────────────────────
st.markdown("""
<style>
/* ── 전체 배경 & 기본 텍스트 ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="block-container"],
.main, .block-container,
section[data-testid="stSidebarContent"],
div[data-testid="stVerticalBlock"] {
    background-color: #1e1e1e !important;
    color: #d4d4d4 !important;
    font-weight: bold !important;
}
[data-testid="stHeader"] {
    background-color: #1e1e1e !important;
}

/* ── 사이드바 ── */
[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
    background-color: #252526 !important;
    border-right: 2px solid #6b8f5e !important;
}
[data-testid="stSidebar"] *,
[data-testid="stSidebarContent"] * {
    color: #d4d4d4 !important;
    font-weight: bold !important;
}

/* ── 메인 텍스트 전체 bold ── */
*, p, label, span, div, li {
    font-weight: bold !important;
}

/* ── 제목 ── */
h1, h2, h3, h4, h5, h6 {
    color: #6b8f5e !important;
    font-family: 'Courier New', monospace !important;
    font-weight: 900 !important;
    letter-spacing: 1px;
}

/* ── 캡션 ── */
.stCaption, [data-testid="stCaptionContainer"] {
    color: #b0c4d8 !important;
    font-weight: bold !important;
}

/* ── 탭 전체 컨테이너 배경 ── */
[data-testid="stTabs"],
[data-testid="stTabs"] > div,
[data-testid="stTabs"] > div > div,
[data-testid="stTabs"] > div > div > div,
div[role="tablist"],
div[role="tablist"]::before,
div[role="tablist"]::after {
    background-color: #1e1e1e !important;
    border-bottom: 1px solid #6b8f5e !important;
}

/* ── 탭 버튼: 흰색 배경 + 검정 굵은 글자 ── */
[data-testid="stTabs"] button,
div[role="tablist"] button {
    background-color: #d4d4d4 !important;
    color: #000000 !important;
    border: 1px solid #c0c0c0 !important;
    border-radius: 4px 4px 0 0 !important;
    font-family: 'Courier New', monospace !important;
    font-weight: 900 !important;
    letter-spacing: 1px;
}
/* 탭 버튼 내부 텍스트 강제 검정 */
[data-testid="stTabs"] button p,
[data-testid="stTabs"] button span,
[data-testid="stTabs"] button div,
div[role="tablist"] button p,
div[role="tablist"] button span,
div[role="tablist"] button div {
    color: #000000 !important;
    font-weight: 900 !important;
}
/* ── 선택된 탭: 골드 배경 + 검정 글자 ── */
[data-testid="stTabs"] button[aria-selected="true"],
div[role="tablist"] button[aria-selected="true"] {
    background-color: #6b8f5e !important;
    color: #000000 !important;
    border-bottom: 3px solid #4a6741 !important;
    font-weight: 900 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] p,
[data-testid="stTabs"] button[aria-selected="true"] span,
[data-testid="stTabs"] button[aria-selected="true"] div,
div[role="tablist"] button[aria-selected="true"] p,
div[role="tablist"] button[aria-selected="true"] span,
div[role="tablist"] button[aria-selected="true"] div {
    color: #000000 !important;
    font-weight: 900 !important;
}

/* ── 탭 콘텐츠 영역 배경 ── */
[data-testid="stTabsTabPanel"],
div[role="tabpanel"] {
    background-color: #1e1e1e !important;
}

/* ── 기본 버튼 ── */
[data-testid="stButton"] button {
    background-color: #252526 !important;
    color: #d4d4d4 !important;
    border: 1px solid #37373d !important;
    font-weight: bold !important;
}
/* ── Primary 버튼 ── */
[data-testid="stButton"] button[kind="primary"] {
    background-color: #37373d !important;
    color: #6b8f5e !important;
    border: 2px solid #6b8f5e !important;
    font-family: 'Courier New', monospace !important;
    font-weight: 900 !important;
    letter-spacing: 1px;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    background-color: #6b8f5e !important;
    color: #1e1e1e !important;
}
/* ── 다운로드 버튼 ── */
[data-testid="stDownloadButton"] button {
    background-color: #252526 !important;
    color: #6b8f5e !important;
    border: 1px solid #6b8f5e !important;
    font-family: 'Courier New', monospace !important;
    font-weight: bold !important;
}

/* ── 메트릭 카드 ── */
[data-testid="stMetric"] {
    background-color: #252526 !important;
    border: 1px solid #37373d !important;
    border-left: 4px solid #6b8f5e !important;
    border-radius: 4px !important;
    padding: 8px 12px !important;
}
[data-testid="stMetricLabel"] {
    color: #b0c4d8 !important;
    font-family: 'Courier New', monospace !important;
    font-size: 0.75rem !important;
    font-weight: bold !important;
}
[data-testid="stMetricValue"] {
    color: #6b8f5e !important;
    font-family: 'Courier New', monospace !important;
    font-weight: 900 !important;
}

/* ── 입력 필드 ── */
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] select,
textarea {
    background-color: #252526 !important;
    color: #d4d4d4 !important;
    border: 1px solid #37373d !important;
    font-weight: bold !important;
}
[data-testid="stSlider"] { accent-color: #6b8f5e; }

/* ── 데이터프레임 ── */
[data-testid="stDataFrame"] { border: 1px solid #37373d !important; }
.stDataFrame thead th {
    background-color: #37373d !important;
    color: #6b8f5e !important;
    font-weight: 900 !important;
}
.stDataFrame tbody td {
    background-color: #252526 !important;
    color: #d4d4d4 !important;
    font-weight: bold !important;
}

/* ── 알림 박스 ── */
[data-testid="stAlert"] {
    background-color: #252526 !important;
    border-left: 4px solid #6b8f5e !important;
    color: #d4d4d4 !important;
    font-weight: bold !important;
}

/* ── 구분선 ── */
hr { border-color: #37373d !important; }

/* ── 체크박스 / 슬라이더 ── */
[data-testid="stCheckbox"] { accent-color: #6b8f5e; }

/* ── 프로그레스바 ── */
[data-testid="stProgressBar"] > div > div {
    background-color: #6b8f5e !important;
}

/* ── 파일 업로더 ── */
[data-testid="stFileUploader"] {
    background-color: #252526 !important;
    border: 2px dashed #37373d !important;
    border-radius: 4px !important;
}

/* ── 라디오 버튼 ── */
[data-testid="stRadio"] label,
[data-testid="stRadio"] span,
[data-testid="stRadio"] p {
    color: #d4d4d4 !important;
    font-weight: bold !important;
}

/* ── 체크박스 텍스트 ── */
[data-testid="stCheckbox"] label,
[data-testid="stCheckbox"] span,
[data-testid="stCheckbox"] p {
    color: #d4d4d4 !important;
    font-weight: bold !important;
}

/* ── selectbox 텍스트 ── */
[data-testid="stSelectbox"] label,
[data-testid="stSelectbox"] span,
[data-testid="stSelectbox"] div[data-baseweb="select"] span {
    color: #d4d4d4 !important;
    font-weight: bold !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background-color: #252526 !important;
    border: 1px solid #37373d !important;
}

/* ── 슬라이더 레이블 & 수치 ── */
[data-testid="stSlider"] label,
[data-testid="stSlider"] span,
[data-testid="stSlider"] p {
    color: #d4d4d4 !important;
    font-weight: bold !important;
}

/* ── number input ── */
[data-testid="stNumberInput"] input {
    background-color: #252526 !important;
    color: #d4d4d4 !important;
    font-weight: bold !important;
    border: 1px solid #37373d !important;
}
[data-testid="stNumberInput"] label,
[data-testid="stNumberInput"] span {
    color: #d4d4d4 !important;
    font-weight: bold !important;
}

/* ── 파일 업로더 텍스트 ── */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small {
    color: #d4d4d4 !important;
    font-weight: bold !important;
}
[data-testid="stFileUploader"] section {
    background-color: #252526 !important;
    border: 2px dashed #37373d !important;
}

/* ── 텍스트 입력 레이블 ── */
[data-testid="stTextInput"] label,
[data-testid="stTextInput"] span {
    color: #d4d4d4 !important;
    font-weight: bold !important;
}

/* ── 탭 패널 내부 모든 텍스트 ── */
[data-testid="stTabsTabPanel"] p,
[data-testid="stTabsTabPanel"] span,
[data-testid="stTabsTabPanel"] label,
[data-testid="stTabsTabPanel"] div {
    color: #d4d4d4 !important;
    font-weight: bold !important;
}

/* ── 마크다운 텍스트 ── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    color: #d4d4d4 !important;
    font-weight: bold !important;
}

/* ── success / warning / error / info 박스 내부 텍스트 ── */
[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div {
    color: #d4d4d4 !important;
    font-weight: bold !important;
}

/* ── spinner 텍스트 ── */
[data-testid="stSpinner"] p,
[data-testid="stSpinner"] span {
    color: #6b8f5e !important;
    font-weight: bold !important;
}

/* ── 프로그레스바 텍스트 ── */
[data-testid="stProgressBar"] + div,
[data-testid="stProgressBar"] ~ p {
    color: #6b8f5e !important;
    font-weight: bold !important;
}

/* ── 빈 상태 / 안내 문구 ── */
[data-testid="stEmpty"] p {
    color: #b0c4d8 !important;
    font-weight: bold !important;
}

/* ── 바 차트 레이블 ── */
[data-testid="stVegaLiteChart"] text {
    fill: #d4d4d4 !important;
    font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES  = {1: "People", 2: "Car", 3: "Bus", 4: "Motorcycle", 5: "Lamp", 6: "Truck"}
CLASS_COLORS = {
    1: (0,   255,  80),   # green
    2: (255, 80,   0),    # orange-red
    3: (0,   160, 255),   # blue
    4: (255, 220,  0),    # yellow
    5: (200,   0, 255),   # purple
    6: (0,   220, 200),   # teal
}
IMG_SIZE = (640, 640)

DEFAULT_CKPT = str(Path(__file__).parent / "runs" / "best.pth")

# ── Session state ─────────────────────────────────────────────────────────────
if "model" not in st.session_state:
    st.session_state.model = None
if "device" not in st.session_state:
    st.session_state.device = None
if "cfg" not in st.session_state:
    st.session_state.cfg = None
if "rgb_only_model" not in st.session_state:
    st.session_state.rgb_only_model = None
if "thermal_only_model" not in st.session_state:
    st.session_state.thermal_only_model = None


# ══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="모델 로딩 중…")
def load_model_cached(ckpt_path: str, device_str: str):
    from config import Config
    from model import build_model

    device = torch.device(device_str)
    cfg = Config()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "config" in ckpt:
        cfg = ckpt["config"]

    model = build_model(cfg.model, num_classes=cfg.data.num_classes)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, cfg, device


@st.cache_resource(show_spinner="단일 모달 모델 로딩 중…")
def load_single_modal_models(ckpt_path: str, device_str: str):
    """RGB 단독 / Thermal 단독 ablation 체크포인트 로드."""
    from config import Config
    from ablation_models import SingleModalDetector

    device = torch.device(device_str)

    # 융합 모델 체크포인트에서 config 추출
    cfg = Config()
    fusion_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "config" in fusion_ckpt:
        cfg = fusion_ckpt["config"]

    ablation_dir = Path(ckpt_path).parent / "ablation"
    rgb_ckpt_path = ablation_dir / "rgb_only_best.pth"
    th_ckpt_path  = ablation_dir / "thermal_only_best.pth"

    def _load(modality, ckpt_p):
        m = SingleModalDetector(cfg.model, num_classes=cfg.data.num_classes, modality=modality)
        if ckpt_p.exists():
            ck = torch.load(str(ckpt_p), map_location=device, weights_only=False)
            state = ck.get("model", ck)
            m.load_state_dict(state)
        m.to(device).eval()
        return m

    rgb_model = _load("rgb",     rgb_ckpt_path)
    th_model  = _load("thermal", th_ckpt_path)
    return rgb_model, th_model, device


@torch.no_grad()
def run_single_inference(model, rgb_t, th_t, device):
    rgb_t = rgb_t.to(device)
    th_t  = th_t.to(device)
    outputs = model(rgb_t, th_t)
    return outputs[0]


def preprocess_pair(rgb_np: np.ndarray, thermal_np: np.ndarray):
    """numpy RGB (H,W,3) + thermal (H,W) or (H,W,3) → tensors."""
    orig_h, orig_w = rgb_np.shape[:2]

    rgb_r  = cv2.resize(rgb_np,      (IMG_SIZE[1], IMG_SIZE[0]))
    if thermal_np.ndim == 3:
        thermal_gray = cv2.cvtColor(thermal_np, cv2.COLOR_RGB2GRAY)
    else:
        thermal_gray = thermal_np
    th_r   = cv2.resize(thermal_gray, (IMG_SIZE[1], IMG_SIZE[0]))

    rgb_t = torch.from_numpy(rgb_r).permute(2, 0, 1).float() / 255.0
    th_t  = torch.from_numpy(th_r).unsqueeze(0).float() / 255.0
    th_t  = th_t.repeat(3, 1, 1)

    return rgb_t.unsqueeze(0), th_t.unsqueeze(0), orig_h, orig_w


@torch.no_grad()
def run_inference(model, rgb_t, th_t, device):
    rgb_t = rgb_t.to(device)
    th_t  = th_t.to(device)
    outputs = model(rgb_t, th_t)
    return outputs[0]


def draw_detections(rgb_np, detections, orig_h, orig_w, score_thresh=0.5):
    """Returns annotated BGR image + list of detection dicts."""
    vis = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
    scale_x = orig_w / IMG_SIZE[1]
    scale_y = orig_h / IMG_SIZE[0]

    boxes  = detections["boxes"].cpu().numpy()
    scores = detections["scores"].cpu().numpy()
    labels = detections["labels"].cpu().numpy()

    results = []
    for box, score, label in zip(boxes, scores, labels):
        if score < score_thresh:
            continue
        x1 = int(box[0] * scale_x)
        y1 = int(box[1] * scale_y)
        x2 = int(box[2] * scale_x)
        y2 = int(box[3] * scale_y)

        color = CLASS_COLORS.get(int(label), (255, 255, 255))
        name  = CLASS_NAMES.get(int(label), f"cls{label}")

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        text = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        results.append({"class": name, "score": float(score),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2})

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), results


def frame_to_np(uploaded_file):
    """Convert uploaded image file → RGB numpy array."""
    data = np.frombuffer(uploaded_file.read(), np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def thermal_to_np(uploaded_file):
    """Convert uploaded thermal file → grayscale numpy array."""
    data = np.frombuffer(uploaded_file.read(), np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    return img


def detection_summary(results):
    """Count detections per class."""
    from collections import Counter
    counts = Counter(r["class"] for r in results)
    return dict(counts)


def run_three_way_detection(rgb_np, th_np, score_thresh, thermal_source=""):
    """RGB단독 / Thermal단독 / 융합 3가지 탐지 결과를 3열로 표시."""
    import pandas as pd

    device = st.session_state.device
    rgb_t, th_t, orig_h, orig_w = preprocess_pair(rgb_np, th_np)

    t0 = time.perf_counter()
    dets_rgb = run_single_inference(st.session_state.rgb_only_model, rgb_t, th_t, device)
    elapsed_rgb = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    dets_th = run_single_inference(st.session_state.thermal_only_model, rgb_t, th_t, device)
    elapsed_th = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    dets_fusion = run_inference(st.session_state.model, rgb_t, th_t, device)
    elapsed_fusion = (time.perf_counter() - t0) * 1000

    vis_rgb,    results_rgb    = draw_detections(rgb_np, dets_rgb,    orig_h, orig_w, score_thresh)
    vis_fusion, results_fusion = draw_detections(rgb_np, dets_fusion, orig_h, orig_w, score_thresh)

    th_display = cv2.cvtColor(cv2.cvtColor(th_np, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
    th_display = cv2.resize(th_display, (orig_w, orig_h))
    vis_th, results_th = draw_detections(th_display, dets_th, orig_h, orig_w, score_thresh)

    st.markdown("---")
    if thermal_source:
        st.caption(f"Thermal 소스: {thermal_source}")

    # ── 결과 이미지 (상단) ──
    col_r, col_t, col_f = st.columns(3)
    with col_r:
        st.markdown("##### RGB 단독")
        st.image(vis_rgb, use_container_width=True)
    with col_t:
        st.markdown("##### Thermal 단독")
        st.image(vis_th, use_container_width=True)
    with col_f:
        st.markdown("##### RGB+Thermal 융합 (CMAFM)")
        st.image(vis_fusion, use_container_width=True)

    # ── 정량 수치 (하단) ──
    st.markdown("---")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("RGB 탐지 수", len(results_rgb))
    m2.metric("RGB 추론", f"{elapsed_rgb:.1f} ms")
    m3.metric("Thermal 탐지 수", len(results_th))
    m4.metric("Thermal 추론", f"{elapsed_th:.1f} ms")
    m5.metric("융합 탐지 수", len(results_fusion))
    m6.metric("융합 추론", f"{elapsed_fusion:.1f} ms")

    # 클래스별 분포 그래프
    if results_fusion:
        import plotly.graph_objects as go
        summary = detection_summary(results_fusion)
        fig = go.Figure(go.Bar(
            x=list(summary.keys()), y=list(summary.values()),
            marker_color="#6b8f5e", marker_line_width=0,
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#d4d4d4", size=11),
            margin=dict(l=40, r=20, t=20, b=40), height=220,
            xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#aaa")),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False, tickfont=dict(color="#888")),
            showlegend=False,
        )
        st.caption("클래스별 탐지 수 (융합 기준)")
        st.plotly_chart(fig, use_container_width=True)

    # 융합 상세 테이블
    st.subheader("융합 모델 상세 결과")
    if results_fusion:
        df = pd.DataFrame(results_fusion)
        df.index += 1
        df.columns = ["클래스", "신뢰도", "X1", "Y1", "X2", "Y2"]
        df["신뢰도"] = df["신뢰도"].apply(lambda x: f"{x:.3f}")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning(f"임계값 {score_thresh:.2f} 이상의 탐지 결과가 없습니다.")

    # 융합 결과 이미지 다운로드
    result_bgr = cv2.cvtColor(vis_fusion, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    st.download_button("💾 융합 결과 이미지 다운로드", data=buf.tobytes(),
                       file_name="detection_fusion.jpg", mime="image/jpeg")


# ══════════════════════════════════════════════════════════════════════════════
# UI — Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:10px 0;'>
        <div style='font-size:2.5rem;'>🎯</div>
        <div style='color:#6b8f5e; font-family:Courier New; font-size:1.1rem; font-weight:bold; letter-spacing:2px;'>
            TACTICAL SYSTEM
        </div>
        <div style='color:#a0b4c8; font-family:Courier New; font-size:0.7rem; letter-spacing:1px;'>
            CMAFM · RGB+LWIR FUSION
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # ── Model ──
    st.subheader("🔧 전투 모델 탑재")
    use_default_ckpt = st.checkbox("기본 경로 사용 (runs/best.pth)",
                                   value=Path(DEFAULT_CKPT).exists())
    if use_default_ckpt:
        ckpt_path = DEFAULT_CKPT
        if Path(ckpt_path).exists():
            st.success(f"✅ 모델 탑재 확인: `best.pth`")
        else:
            st.error("❌ runs/best.pth 없음 — 경로 직접 입력")
            ckpt_path = st.text_input("체크포인트 경로", value="")
    else:
        ckpt_path = st.text_input("체크포인트 경로", value=DEFAULT_CKPT)

    # ── Device (자동 선택) ──
    st.subheader("⚡ 연산 장치")
    cuda_avail = torch.cuda.is_available()
    device_str = "cuda" if cuda_avail else "cpu"
    if cuda_avail:
        gpu_name = torch.cuda.get_device_name(0)
        st.success(f"CUDA ✔  {gpu_name}")
    else:
        st.info("CPU 모드 (CUDA 없음)")

    # ── Load model button ──
    if st.button("🚀 시스템 가동", type="primary", use_container_width=True):
        if not ckpt_path or not Path(ckpt_path).exists():
            st.error("체크포인트 파일을 찾을 수 없습니다.")
        else:
            with st.spinner("전투 모델 로딩 중…"):
                model, cfg, device = load_model_cached(ckpt_path, device_str)
                st.session_state.model  = model
                st.session_state.device = device
                st.session_state.cfg    = cfg
                rgb_m, th_m, _ = load_single_modal_models(ckpt_path, device_str)
                st.session_state.rgb_only_model     = rgb_m
                st.session_state.thermal_only_model = th_m
            st.success("✅ 시스템 가동 완료 (융합 + RGB단독 + Thermal단독)")

    st.markdown("---")

    # ── Inference params ──
    st.subheader("🎚️ 탐지 감도 설정")
    score_thresh = st.slider("신뢰도 임계값 (Score Threshold)", 0.1, 0.95, 0.5, 0.05)

    st.markdown("---")
    st.subheader("🏷️ 표적 클래스")
    for cid, cname in CLASS_NAMES.items():
        r, g, b = CLASS_COLORS[cid]
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        st.markdown(
            f'<span style="background:{hex_color};border-radius:3px;'
            f'padding:3px 12px;color:#000;font-weight:bold;'
            f'font-family:Courier New;letter-spacing:1px;">{cname}</span>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# UI — Main Area
# ══════════════════════════════════════════════════════════════════════════════

_photo_path = Path(__file__).parent / "ltc_bansungwoo.png"
_photo_b64 = ""
if _photo_path.exists():
    import base64 as _b64
    _photo_b64 = _b64.b64encode(_photo_path.read_bytes()).decode()

_photo_html = (
    f"<img src='data:image/png;base64,{_photo_b64}' "
    f"style='width:90px; height:110px; object-fit:cover; object-position:top; "
    f"border-radius:4px; opacity:0.85; filter:sepia(30%);'/>"
    if _photo_b64 else ""
)

st.markdown(f"""
<div style='border:2px solid #6b8f5e; border-radius:6px; padding:16px 24px 10px 24px; margin-bottom:8px;
            background:linear-gradient(90deg,#252526 0%,#0a1628 100%);
            display:flex; align-items:center; justify-content:space-between;'>
    <div>
        <div style='font-size:2.6rem; font-weight:900; color:#6b8f5e;
                    font-family:Courier New; letter-spacing:4px;'>
            🎯 CMAFM TACTICAL DETECTION SYSTEM
        </div>
        <div style='color:#a0b4c8; font-family:Courier New; font-size:0.85rem; margin-top:4px; letter-spacing:1px;'>
            ◆ CROSS-MODAL ATTENTION FUSION &nbsp;|&nbsp; RGB + LWIR MULTISPECTRAL &nbsp;|&nbsp; M3FD DATASET
        </div>
        <div style='color:#556677; font-family:Courier New; font-size:0.7rem; margin-top:8px; letter-spacing:1px;'>
            by. LTC Bansungwoo
        </div>
    </div>
    <div style='margin-left:24px; flex-shrink:0;'>
        {_photo_html}
    </div>
</div>
""", unsafe_allow_html=True)

model_ready = st.session_state.model is not None

if not model_ready:
    st.warning("⚠️ 시스템 대기 중 — 좌측 패널에서 **시스템 가동** 버튼을 눌러 전투 모드로 진입하세요.")

# ── Mode selection ──
tab_image, tab_video, tab_webcam = st.tabs(["📡 이미지 탐지", "📹 영상 탐지", "🎖️ 샘플 테스트"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Image Detection
# ══════════════════════════════════════════════════════════════════════════════

with tab_image:
    st.subheader("📡 정지 영상 표적 탐지")

    st.caption("RGB 이미지와 Thermal 이미지를 각각 업로드하세요.")
    col_upload_rgb, col_upload_th = st.columns(2)
    with col_upload_rgb:
        rgb_file = st.file_uploader("📷 RGB 이미지",
                                     type=["jpg", "jpeg", "png", "bmp"],
                                     key="img_rgb")
    with col_upload_th:
        th_file  = st.file_uploader("🌡️ Thermal 이미지",
                                     type=["jpg", "jpeg", "png", "bmp"],
                                     key="img_th")

    if rgb_file and th_file:
        col_prev_r, col_prev_t = st.columns(2)
        rgb_file.seek(0); th_file.seek(0)
        with col_prev_r:
            st.image(rgb_file, caption="RGB 입력", use_container_width=True)
        with col_prev_t:
            st.image(th_file,  caption="Thermal 입력", use_container_width=True)

    run_img = st.button("🔍 탐지 실행", type="primary",
                         disabled=(not model_ready or rgb_file is None or th_file is None),
                         key="btn_img")

    if run_img and rgb_file and th_file:
        rgb_file.seek(0); th_file.seek(0)
        rgb_np = frame_to_np(rgb_file)
        th_np  = thermal_to_np(th_file)
        run_three_way_detection(rgb_np, th_np, score_thresh, "직접 업로드")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Video Detection
# ══════════════════════════════════════════════════════════════════════════════

with tab_video:
    st.subheader("📹 동적 표적 추적 탐지")

    st.markdown("> ⚠️ **두 영상의 프레임 수와 해상도가 동일**해야 합니다.")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        rgb_vid = st.file_uploader("📹 RGB 영상", type=["mp4", "avi", "mov", "mkv"],
                                    key="vid_rgb")
    with col_v2:
        th_vid  = st.file_uploader("🌡️ Thermal 영상", type=["mp4", "avi", "mov", "mkv"],
                                    key="vid_th")

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        max_frames = st.number_input("최대 처리 프레임 수 (0 = 전체)",
                                      min_value=0, max_value=10000, value=100, step=10)
    with col_opt2:
        frame_skip = st.number_input("프레임 스킵 (N프레임마다 1개 처리)",
                                      min_value=1, max_value=30, value=1, step=1)

    vid_ready = model_ready and rgb_vid is not None and th_vid is not None
    run_vid = st.button("🎬 영상 탐지 시작", type="primary",
                         disabled=not vid_ready,
                         key="btn_vid")

    if run_vid and rgb_vid and th_vid:
        # Save RGB/Thermal to temp files
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as rf:
            rf.write(rgb_vid.read()); rgb_tmp = rf.name
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            tf.write(th_vid.read()); th_tmp = tf.name

        cap_r = cv2.VideoCapture(rgb_tmp)
        cap_t = cv2.VideoCapture(th_tmp) if th_tmp else None

        total_frames = int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_in       = cap_r.get(cv2.CAP_PROP_FPS) or 25
        width        = int(cap_r.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames_to_process = total_frames if max_frames == 0 else min(total_frames, max_frames * frame_skip)

        # Output video
        # 3개 영상 출력 파일 (mp4v로 먼저 기록 후 ffmpeg로 H.264 재인코딩)
        raw_rgb_tmp    = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        raw_th_tmp     = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        raw_fusion_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        out_rgb_tmp    = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        out_th_tmp     = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        out_fusion_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = fps_in / frame_skip
        writer_rgb    = cv2.VideoWriter(raw_rgb_tmp,    fourcc, out_fps, (width, height))
        writer_th     = cv2.VideoWriter(raw_th_tmp,     fourcc, out_fps, (width, height))
        writer_fusion = cv2.VideoWriter(raw_fusion_tmp, fourcc, out_fps, (width, height))

        st.markdown("---")
        prog_bar  = st.progress(0, text="처리 중…")
        # 라이브 프리뷰 3열
        prev_cols   = st.columns(3)
        prev_rgb    = prev_cols[0].empty()
        prev_th     = prev_cols[1].empty()
        prev_fusion = prev_cols[2].empty()
        prev_cols[0].caption("RGB 단독")
        prev_cols[1].caption("Thermal 단독")
        prev_cols[2].caption("융합 (CMAFM)")

        frame_idx    = 0
        proc_count   = 0
        total_dets   = 0
        total_time   = 0.0
        all_results  = []
        device       = st.session_state.device
        # 그래프용 시계열 데이터
        log_frames, log_dets, log_ms = [], [], []

        while cap_r.isOpened():
            ret_r, frm_r = cap_r.read()
            if not ret_r or frame_idx >= frames_to_process:
                break

            if cap_t is not None and cap_t.isOpened():
                ret_t, frm_t = cap_t.read()
                th_np = cv2.cvtColor(frm_t, cv2.COLOR_BGR2GRAY) if ret_t else cv2.cvtColor(frm_r, cv2.COLOR_BGR2GRAY)
            else:
                th_np = cv2.cvtColor(frm_r, cv2.COLOR_BGR2GRAY)

            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            rgb_np = cv2.cvtColor(frm_r, cv2.COLOR_BGR2RGB)
            rgb_t, th_t, orig_h, orig_w = preprocess_pair(rgb_np, th_np)

            t0 = time.perf_counter()
            dets_rgb    = run_single_inference(st.session_state.rgb_only_model,     rgb_t, th_t, device)
            dets_th     = run_single_inference(st.session_state.thermal_only_model, rgb_t, th_t, device)
            dets_fusion = run_inference(st.session_state.model,                     rgb_t, th_t, device)
            elapsed = (time.perf_counter() - t0) * 1000
            total_time += elapsed

            vis_rgb,    results_rgb    = draw_detections(rgb_np, dets_rgb,    orig_h, orig_w, score_thresh)
            vis_fusion, results_fusion = draw_detections(rgb_np, dets_fusion, orig_h, orig_w, score_thresh)

            th_display = cv2.cvtColor(cv2.cvtColor(th_np, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
            th_display = cv2.resize(th_display, (orig_w, orig_h))
            vis_th, results_th = draw_detections(th_display, dets_th, orig_h, orig_w, score_thresh)

            writer_rgb.write(cv2.cvtColor(vis_rgb,    cv2.COLOR_RGB2BGR))
            writer_th.write(cv2.cvtColor(vis_th,     cv2.COLOR_RGB2BGR))
            writer_fusion.write(cv2.cvtColor(vis_fusion, cv2.COLOR_RGB2BGR))

            all_results.extend(results_fusion)
            total_dets += len(results_fusion)
            proc_count += 1

            # 시계열 로그
            log_frames.append(frame_idx)
            log_dets.append(len(results_fusion))
            log_ms.append(round(elapsed, 1))

            # 라이브 프리뷰 5프레임마다
            if proc_count % 5 == 1:
                prev_rgb.image(vis_rgb,    caption=f"Frame {frame_idx} | {len(results_rgb)}개",    use_container_width=True)
                prev_th.image(vis_th,     caption=f"Frame {frame_idx} | {len(results_th)}개",     use_container_width=True)
                prev_fusion.image(vis_fusion, caption=f"Frame {frame_idx} | {len(results_fusion)}개", use_container_width=True)

            avg_ms = total_time / proc_count
            prog_bar.progress(
                min(frame_idx / max(frames_to_process - 1, 1), 1.0),
                text=f"Frame {frame_idx}/{frames_to_process} | 평균 {avg_ms:.1f} ms | 융합 탐지 {total_dets}개"
            )
            frame_idx += 1

        cap_r.release()
        if cap_t is not None:
            cap_t.release()
        writer_rgb.release()
        writer_th.release()
        writer_fusion.release()

        # mp4v → H.264 재인코딩 (브라우저 재생 호환)
        import shutil as _shutil
        _ffmpeg = (
            _shutil.which("ffmpeg")
            or r"C:\Users\CAU\anaconda3\Library\bin\ffmpeg.exe"
        )
        _has_ffmpeg = _ffmpeg is not None and Path(_ffmpeg).exists()

        prog_bar.progress(1.0, text="H.264 인코딩 중…" if _has_ffmpeg else "완료!")

        def _reencode(src, dst):
            import subprocess
            subprocess.run(
                [_ffmpeg, "-y", "-i", src,
                 "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", dst],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

        if _has_ffmpeg:
            for src, dst in [(raw_rgb_tmp, out_rgb_tmp),
                              (raw_th_tmp,  out_th_tmp),
                              (raw_fusion_tmp, out_fusion_tmp)]:
                _reencode(src, dst)
        else:
            # ffmpeg 없으면 raw 파일을 그대로 사용
            import shutil as _sh
            for src, dst in [(raw_rgb_tmp, out_rgb_tmp),
                              (raw_th_tmp,  out_th_tmp),
                              (raw_fusion_tmp, out_fusion_tmp)]:
                _sh.copy(src, dst)

        prog_bar.progress(1.0, text="완료!")
        st.success(f"✅ 영상 처리 완료 — {proc_count}프레임, 융합 탐지 {total_dets}개")

        # 결과 영상 재생
        st.subheader("결과 영상 재생")

        # 상단 2열: RGB 단독 / Thermal 단독
        col_r, col_t = st.columns(2)
        for col, path, label in [
            (col_r, out_rgb_tmp, "RGB 단독"),
            (col_t, out_th_tmp,  "Thermal 단독"),
        ]:
            with open(path, "rb") as f:
                vid_bytes = f.read()
            col.markdown(f"##### {label}")
            col.video(vid_bytes)
            col.download_button(f"💾 {label} 다운로드",
                                data=vid_bytes,
                                file_name=f"detection_{label}.mp4",
                                mime="video/mp4",
                                key=f"dl_{label}")

        # 하단 1열: 융합 (가운데 정렬 + 확대)
        st.markdown("---")
        _, col_f, _ = st.columns([1, 4, 1])
        with open(out_fusion_tmp, "rb") as f:
            fusion_bytes = f.read()
        col_f.markdown("##### RGB+Thermal 융합 (CMAFM)")
        col_f.video(fusion_bytes)
        col_f.download_button("💾 융합 (CMAFM) 다운로드",
                              data=fusion_bytes,
                              file_name="detection_융합(CMAFM).mp4",
                              mime="video/mp4",
                              key="dl_fusion")

        # ── 프레임별 탐지 수치 그래프 ─────────────────────────────────────────
        if log_frames:
            import pandas as pd
            import plotly.graph_objects as go
            st.markdown("---")
            st.subheader("📊 프레임별 탐지 수치")

            _chart_layout = dict(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#d4d4d4", size=11),
                margin=dict(l=40, r=20, t=20, b=40),
                height=220,
                xaxis=dict(
                    showgrid=True, gridcolor="rgba(255,255,255,0.08)",
                    zeroline=False, showline=False, tickfont=dict(color="#888"),
                ),
                yaxis=dict(
                    showgrid=True, gridcolor="rgba(255,255,255,0.08)",
                    zeroline=False, showline=False, tickfont=dict(color="#888"),
                ),
                showlegend=False,
            )

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.caption("프레임별 탐지 객체 수")
                fig1 = go.Figure(go.Scatter(
                    x=log_frames, y=log_dets, mode="lines",
                    line=dict(color="#6b8f5e", width=1.5),
                    fill=None,
                ))
                fig1.update_layout(**_chart_layout)
                st.plotly_chart(fig1, use_container_width=True)
            with col_g2:
                st.caption("프레임별 추론 시간 (ms)")
                fig2 = go.Figure(go.Scatter(
                    x=log_frames, y=log_ms, mode="lines",
                    line=dict(color="#5e7a8f", width=1.5),
                    fill=None,
                ))
                fig2.update_layout(**_chart_layout)
                st.plotly_chart(fig2, use_container_width=True)

        # 클래스 분포
        if all_results:
            import pandas as pd
            import plotly.graph_objects as go
            from collections import Counter
            counts = Counter(r["class"] for r in all_results)
            df_sum = pd.DataFrame(counts.items(), columns=["클래스", "탐지 수"]).sort_values("탐지 수", ascending=False)
            st.subheader("전체 클래스별 탐지 통계 (융합 기준)")
            fig3 = go.Figure(go.Bar(
                x=df_sum["클래스"], y=df_sum["탐지 수"],
                marker_color="#6b8f5e", marker_line_width=0,
            ))
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#d4d4d4", size=11),
                margin=dict(l=40, r=20, t=20, b=40),
                height=260,
                xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#aaa")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False, tickfont=dict(color="#888")),
                showlegend=False,
            )
            st.plotly_chart(fig3, use_container_width=True)

        # 임시 파일 정리
        for p in [rgb_tmp, raw_rgb_tmp, raw_th_tmp, raw_fusion_tmp,
                  out_rgb_tmp, out_th_tmp, out_fusion_tmp]:
            try:
                os.unlink(p)
            except Exception:
                pass
        if th_tmp:
            try:
                os.unlink(th_tmp)
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Sample Test (데이터셋에서 랜덤 샘플)
# ══════════════════════════════════════════════════════════════════════════════

with tab_webcam:
    st.subheader("🎖️ 전술 데이터셋 샘플 테스트")
    st.markdown("M3FD 데이터셋에서 샘플을 선택해 RGB / Thermal / 융합 탐지를 비교합니다.")

    data_root = Path(__file__).parent / "data" / "M3FD"
    vis_dir   = data_root / "Vis"
    ir_dir    = data_root / "Ir"

    has_data = vis_dir.exists() and ir_dir.exists()
    if not has_data:
        st.warning("⚠️ `data/M3FD/Vis` 또는 `data/M3FD/Ir` 폴더가 없습니다. "
                   "데이터셋을 먼저 배치해주세요.")
    else:
        rgb_files = sorted(vis_dir.glob("*.png")) + sorted(vis_dir.glob("*.jpg"))
        st.caption(f"데이터셋: {len(rgb_files)}개 샘플 발견")

        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            sample_idx = st.slider("샘플 인덱스", 0, max(0, len(rgb_files) - 1), 0)
        with col_s2:
            if st.button("🎲 랜덤 선택", use_container_width=True):
                sample_idx = int(np.random.randint(0, len(rgb_files)))
                st.rerun()

        if rgb_files:
            chosen_rgb = rgb_files[sample_idx]
            chosen_th  = ir_dir / chosen_rgb.name

            col_s_r, col_s_t = st.columns(2)
            with col_s_r:
                st.image(str(chosen_rgb), caption=f"RGB: {chosen_rgb.name}",
                          use_container_width=True)
            with col_s_t:
                if chosen_th.exists():
                    st.image(str(chosen_th), caption=f"Thermal: {chosen_th.name}",
                              use_container_width=True)
                else:
                    st.error("대응하는 Thermal 이미지가 없습니다.")

            run_sample = st.button("🔍 샘플 탐지 실행", type="primary",
                                    disabled=(not model_ready or not chosen_th.exists()),
                                    key="btn_sample")

            if run_sample:
                rgb_np = cv2.cvtColor(cv2.imread(str(chosen_rgb)), cv2.COLOR_BGR2RGB)
                th_np  = cv2.imread(str(chosen_th), cv2.IMREAD_GRAYSCALE)

                rgb_t, th_t, orig_h, orig_w = preprocess_pair(rgb_np, th_np)
                device = st.session_state.device

                # ── 세 모델 추론 ──────────────────────────────────────────
                t0 = time.perf_counter()
                dets_fusion = run_inference(st.session_state.model, rgb_t, th_t, device)
                elapsed_fusion = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                dets_rgb = run_single_inference(st.session_state.rgb_only_model, rgb_t, th_t, device)
                elapsed_rgb = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                dets_th = run_single_inference(st.session_state.thermal_only_model, rgb_t, th_t, device)
                elapsed_th = (time.perf_counter() - t0) * 1000

                vis_fusion, results_fusion = draw_detections(rgb_np, dets_fusion, orig_h, orig_w, score_thresh)
                vis_rgb,    results_rgb    = draw_detections(rgb_np, dets_rgb,    orig_h, orig_w, score_thresh)

                # Thermal 결과는 thermal 이미지를 배경으로 표시
                th_display = cv2.cvtColor(
                    cv2.cvtColor(th_np, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB
                )
                th_display_resized = cv2.resize(th_display, (orig_w, orig_h))
                vis_th, results_th = draw_detections(th_display_resized, dets_th, orig_h, orig_w, score_thresh)

                st.markdown("---")

                # ── 메트릭 행 ─────────────────────────────────────────────
                st.subheader("탐지 결과 비교")
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("RGB 단독 탐지", len(results_rgb))
                m2.metric("RGB 추론", f"{elapsed_rgb:.1f} ms")
                m3.metric("Thermal 단독 탐지", len(results_th))
                m4.metric("Thermal 추론", f"{elapsed_th:.1f} ms")
                m5.metric("융합 탐지", len(results_fusion))
                m6.metric("융합 추론", f"{elapsed_fusion:.1f} ms")

                # ── 3열 결과 이미지 ───────────────────────────────────────
                col_r, col_t, col_f = st.columns(3)
                with col_r:
                    st.markdown("##### RGB 단독")
                    st.image(vis_rgb, use_container_width=True)
                    st.caption(f"탐지: {len(results_rgb)}개")
                with col_t:
                    st.markdown("##### Thermal 단독")
                    st.image(vis_th, use_container_width=True)
                    st.caption(f"탐지: {len(results_th)}개")
                with col_f:
                    st.markdown("##### RGB+Thermal 융합 (CMAFM)")
                    st.image(vis_fusion, use_container_width=True)
                    st.caption(f"탐지: {len(results_fusion)}개")

                # ── 융합 결과 상세 테이블 ─────────────────────────────────
                st.markdown("---")
                st.subheader("융합 모델 상세 결과")
                if results_fusion:
                    import pandas as pd
                    df = pd.DataFrame(results_fusion)
                    df.index += 1
                    df.columns = ["클래스", "신뢰도", "X1", "Y1", "X2", "Y2"]
                    df["신뢰도"] = df["신뢰도"].apply(lambda x: f"{x:.3f}")
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning(f"임계값 {score_thresh:.2f} 이상의 탐지 결과가 없습니다.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#a0b4c8; font-family:Courier New; font-size:0.75rem; letter-spacing:1px;'>
    🎯 CMAFM &nbsp;·&nbsp; CROSS-MODAL ATTENTION FUSION MODEL &nbsp;·&nbsp; M3FD DATASET<br>
    ResNet-50 DUAL BACKBONE + FPN + Faster R-CNN &nbsp;·&nbsp; RGB + LWIR MULTISPECTRAL
</div>
""", unsafe_allow_html=True)
