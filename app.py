"""Streamlit demo: run with  streamlit run app.py"""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from helmet_utils import draw_detections

WEIGHTS = Path("models/helmet_best.pt")
st.set_page_config(page_title="Helmet Detection", page_icon="🪖", layout="wide")
st.title("🪖 Helmet Detection for Two-Wheeler Riders")
st.caption("YOLOv8n trained to detect riders with and without helmets.")

if not WEIGHTS.exists():
    st.error("Model not found at models/helmet_best.pt. Train it first (see README).")
    st.stop()


@st.cache_resource
def load_model():
    return YOLO(str(WEIGHTS))


model = load_model()
conf = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05)


def run(img_bgr):
    result = model(img_bgr, conf=conf, verbose=False)[0]
    return draw_detections(img_bgr.copy(), result)


def show(img_bgr):
    out, safe, viol = run(img_bgr)
    c1, c2 = st.columns(2)
    c1.metric("With helmet", safe)
    c2.metric("Without helmet", viol)
    if viol:
        st.warning(f"⚠️ {viol} rider(s) without a helmet detected")
    st.image(out, channels="BGR", use_container_width=True)


def to_bgr(file):
    return cv2.cvtColor(np.array(Image.open(file).convert("RGB")), cv2.COLOR_RGB2BGR)


tab_img, tab_cam, tab_vid = st.tabs(["📷 Image", "🎥 Camera", "🎞️ Video"])

with tab_img:
    f = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if f:
        show(to_bgr(f))

with tab_cam:
    snap = st.camera_input("Take a photo")
    if snap:
        show(to_bgr(snap))
    st.info("For smooth real-time video, run  python detect_webcam.py  instead.")

with tab_vid:
    v = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if v and st.button("Process video"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(v.name).suffix)
        tmp.write(v.read())
        tmp.close()
        cap = cv2.VideoCapture(tmp.name)
        frame_box, stats = st.empty(), st.empty()
        total_viol = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out, safe, viol = run(frame)
            total_viol = max(total_viol, viol)
            stats.write(f"With helmet: **{safe}** · Without helmet: **{viol}**")
            frame_box.image(out, channels="BGR", use_container_width=True)
        cap.release()
        st.success(f"Done. Max riders without helmet in one frame: {total_viol}")
