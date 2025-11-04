import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import easyocr
import pandas as pd
import cv2
import numpy as np
import threading
import time
import os

st.set_page_config(page_title="リアルタイムOCR＋音声再生", layout="wide")

st.title("📷 リアルタイムOCR＋音声再生（クラウド対応）")

# -----------------------------
# CSV読み込み（文字→音声対応）
# -----------------------------
@st.cache_data
def load_mapping():
    df = pd.read_csv("mapping.csv")
    return {row["text"]: row["audio"] for _, row in df.iterrows()}

mapping = load_mapping()

# -----------------------------
# EasyOCRモデル読込
# -----------------------------
@st.cache_resource
def load_reader():
    return easyocr.Reader(['ja', 'en'])

reader = load_reader()

# -----------------------------
# OCR処理クラス
# -----------------------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.prev_text = None
        self.last_play_time = 0
        self.delay = 5  # 秒
        self.result_text = ""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # ROI指定（中央部分を使う）
        h, w, _ = img.shape
        roi = img[h//3:h*2//3, w//4:w*3//4]

        # OCR実行
        results = reader.readtext(roi, detail=0)

        # 枠を描画
        cv2.rectangle(img, (w//4, h//3), (w*3//4, h*2//3), (0, 255, 0), 2)

        if results:
            text = results[0]
            self.result_text = text

            if text != self.prev_text and text in mapping:
                self.prev_text = text
                # 一定時間後に音声再生（st.audioはメインスレッド側で）
                self.last_play_time = time.time()

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -----------------------------
# WebRTCストリーム開始
# -----------------------------
webrtc_ctx = webrtc_streamer(
    key="realtime-ocr",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# -----------------------------
# 音声再生部分（Streamlit側）
# -----------------------------
if webrtc_ctx and webrtc_ctx.video_processor:
    vp = webrtc_ctx.video_processor
    placeholder = st.empty()

    while webrtc_ctx.state.playing:
        if vp.result_text and vp.prev_text == vp.result_text:
            detected_text = vp.result_text
            st.write(f"🔍 認識文字：**{detected_text}**")

            # 一定時間後に音声を再生
            if time.time() - vp.last_play_time > vp.delay:
                if detected_text in mapping:
                    audio_path = os.path.join("sounds", mapping[detected_text])
                    if os.path.exists(audio_path):
                        placeholder.audio(audio_path, format="audio/mp3")
                    else:
                        st.warning("音声ファイルが見つかりません。")
                vp.last_play_time = time.time() + 999  # 二重再生防止

        time.sleep(0.5)
