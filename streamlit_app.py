import streamlit as st
import cv2
import numpy as np
import easyocr
import csv
import time
import tempfile
import threading
import pygame
from PIL import Image

st.set_page_config(page_title="OCR + 音声再生", layout="wide")

# ===== 音声再生関連 =====
pygame.mixer.init()

def play_audio(audio_file):
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()

def stop_audio():
    pygame.mixer.music.stop()

# ===== CSVの読み込み =====
@st.cache_data
def load_mapping(csv_path):
    mapping = {}
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    mapping[row[0]] = row[1]
    except Exception as e:
        st.error(f"CSV読み込みエラー: {e}")
    return mapping

# ===== OCR初期化 =====
@st.cache_resource
def get_reader():
    return easyocr.Reader(['ja', 'en'])

reader = get_reader()

# ===== サイドバー =====
st.sidebar.header("設定")

csv_path = st.sidebar.text_input("CSVファイルパス", "mapping.csv")
mapping = load_mapping(csv_path)

camera_indices = [0, 1, 2]
selected_camera = st.sidebar.selectbox("使用するカメラを選択", camera_indices, index=0)
st.sidebar.info("カメラが映らない場合は番号を変更してね！")

roi_selection = st.sidebar.checkbox("ROIを指定する（範囲トリミング）", False)
delay_time = st.sidebar.slider("認識後の再生までの待機時間（秒）", 0, 30, 20)

st.title("🔠 リアルタイムOCR + 音声再生")

run = st.checkbox("カメラ起動", value=False)

if run:
    cap = cv2.VideoCapture(selected_camera)
    if not cap.isOpened():
        st.error("カメラが開けませんでした…")
    else:
        roi = None
        prev_text = ""
        last_detect_time = 0
        placeholder = st.empty()

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("カメラの映像が取得できません。")
                break

            # ROI選択モード
            if roi_selection and roi is None:
                st.info("ROIを選択してください（ウィンドウに表示されます）")
                cv2.imshow("ROI選択", frame)
                roi = cv2.selectROI("ROI選択", frame, False)
                cv2.destroyWindow("ROI選択")

            # ROI適用
            if roi_selection and roi:
                x, y, w, h = map(int, roi)
                frame = frame[y:y+h, x:x+w]

            # OCR実行
            results = reader.readtext(frame)
            text_detected = ""
            for res in results:
                text_detected += res[1]

            # 認識文字に対応する音声を再生
            if text_detected and text_detected != prev_text:
                prev_text = text_detected
                stop_audio()

                if text_detected in mapping:
                    audio_file = mapping[text_detected]
                    last_detect_time = time.time()

                    # 遅延して音声再生
                    def delayed_play():
                        time.sleep(delay_time)
                        if prev_text == text_detected:
                            play_audio(audio_file)
                    threading.Thread(target=delayed_play, daemon=True).start()

            # 画面表示
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            placeholder.image(frame_rgb, channels="RGB")

        cap.release()
        cv2.destroyAllWindows()
