import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import cv2
import pytesseract
import pandas as pd
import numpy as np
import time
import os
import av

st.set_page_config(page_title="リアルタイムOCR＋音声再生（クラウド安定版）", layout="wide")

st.title("📷 リアルタイムOCR＋音声再生（クラウド安定版）")

# -----------------------------
# OCRと音声マッピング
# -----------------------------
@st.cache_data
def load_mapping():
    df = pd.read_csv("mapping.csv")
    return {row["text"]: row["audio"] for _, row in df.iterrows()}

mapping = load_mapping()

# -----------------------------
# 初期設定
# -----------------------------
st.sidebar.header("設定")
interval = st.sidebar.slider("OCRの更新間隔（秒）", 1.0, 5.0, 2.0, 0.5)

roi_x = st.sidebar.slider("ROI X位置", 0, 100, 25)
roi_y = st.sidebar.slider("ROI Y位置", 0, 100, 25)
roi_w = st.sidebar.slider("ROI 幅", 10, 100, 50)
roi_h = st.sidebar.slider("ROI 高さ", 10, 100, 50)

# -----------------------------
# OCR処理関数
# -----------------------------
def process_frame(frame, last_ocr_time, prev_text):
    img = frame.to_ndarray(format="bgr24")
    h, w, _ = img.shape

    # ROIを割合から算出
    x1 = int(w * roi_x / 100)
    y1 = int(h * roi_y / 100)
    x2 = int(w * (roi_x + roi_w) / 100)
    y2 = int(h * (roi_y + roi_h) / 100)
    roi = img[y1:y2, x1:x2]

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    now = time.time()
    text = prev_text

    # 一定間隔でOCR
    if now - last_ocr_time > interval:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        text = pytesseract.image_to_string(gray, lang="jpn+eng").strip()
        return img, text, now
    else:
        return img, prev_text, last_ocr_time


# -----------------------------
# ストリーム表示
# -----------------------------
st.write("🎥 カメラ映像が下に表示されます（ROI枠内を認識）")

webrtc_ctx = webrtc_streamer(
    key="ocr",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=None,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# -----------------------------
# メインループ（音声再生など）
# -----------------------------
if webrtc_ctx.video_receiver:
    placeholder = st.empty()
    prev_text = ""
    last_ocr_time = 0
    last_sound_time = 0

    while webrtc_ctx.state.playing:
        frame = webrtc_ctx.video_receiver.get_frame(timeout=1)
        if frame is None:
            continue

        img, text, last_ocr_time = process_frame(frame, last_ocr_time, prev_text)
        prev_text = text

        # 表示
        stframe = placeholder.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if text:
            st.write(f"🔍 認識結果: **{text}**")

            # CSV対応音声があれば再生
            if text in mapping and time.time() - last_sound_time > interval:
                audio_path = os.path.join("sounds", mapping[text])
                if os.path.exists(audio_path):
                    st.audio(audio_path, format="audio/mp3")
                    last_sound_time = time.time()

        time.sleep(0.1)
