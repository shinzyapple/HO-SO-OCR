# streamlit_app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import threading, time, queue, os
import easyocr
import pandas as pd
import cv2
import numpy as np

st.set_page_config(page_title="リアルタイムOCR（非同期）", layout="wide")
st.title("リアルタイムOCR（非同期）＋ROI（スライダー）＋音声再生（Cloud）")

# ---------------------
# 設定（CSV と sounds フォルダは同リポジトリ）
# ---------------------
@st.cache_data
def load_mapping():
    if os.path.exists("mapping.csv"):
        df = pd.read_csv("mapping.csv")
        return {str(r['text']): str(r['audio']) for _, r in df.iterrows()}
    return {}

mapping = load_mapping()
st.sidebar.markdown("### 音声マッピング（mapping.csv）")
st.sidebar.write(mapping or "mapping.csv がありません")

# OCR 設定 UI
st.sidebar.markdown("### OCR / ROI 設定")
ocr_interval = st.sidebar.slider("OCR実行間隔（秒）", 0.2, 2.0, 0.8, 0.1)
resize_width = st.sidebar.number_input("OCR用リサイズ幅(px)", 200, 1280, 640, 16)
debounce_sec = st.sidebar.number_input("同語デバウンス秒", 0.1, 10.0, 1.0, 0.1)

# ROI をスライダーで指定（割合ベースにしてどの解像度でも使える）
st.sidebar.markdown("### ROI（割合で指定）")
x0 = st.sidebar.slider("左 (%)", 0, 90, 25)
y0 = st.sidebar.slider("上 (%)", 0, 90, 33)
w_pct = st.sidebar.slider("幅 (%)", 1, 100 - x0, 50)
h_pct = st.sidebar.slider("高さ (%)", 1, 100 - y0, 34)

# 再生遅延（もし必要なら）
play_delay = st.sidebar.number_input("再生遅延（秒、認識から）", min_value=0.0, value=0.0, step=0.5)

# ---------------------
# グローバル共有変数（VideoProcessor と OCR スレッドで共有）
# ---------------------
frame_q = queue.Queue(maxsize=1)   # 最新フレームのみ保持
detected_text_shared = {"text": None, "timestamp": 0.0}
detected_lock = threading.Lock()
stop_flag = {"stop": False}

# ---------------------
# OCRワーカー（バックグラウンドスレッド）
# ---------------------
@st.cache_resource
def get_reader():
    return easyocr.Reader(['ja','en'], gpu=False)

reader = get_reader()

def ocr_worker():
    """ フレームキューから最新フレームを取り OCR を行い、結果を共有変数に書く """
    last_detect = None
    last_detect_time = 0.0
    while not stop_flag["stop"]:
        try:
            frame_full = frame_q.get(timeout=0.5)
        except queue.Empty:
            continue

        # ROI を割合からピクセルに変換
        h, w = frame_full.shape[:2]
        x1 = int(w * (x0 / 100.0))
        y1 = int(h * (y0 / 100.0))
        x2 = int(x1 + w * (w_pct / 100.0))
        y2 = int(y1 + h * (h_pct / 100.0))
        # clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            roi = frame_full
        else:
            roi = frame_full[y1:y2, x1:x2]

        # リサイズして OCR に渡す
        fh, fw = roi.shape[:2]
        if fw > resize_width:
            scale = resize_width / fw
            small = cv2.resize(roi, (int(fw*scale), int(fh*scale)))
        else:
            small = roi

        # EasyOCR 実行（detail=0 で速くする）
        try:
            results = reader.readtext(small, detail=0)
        except Exception as e:
            # モデル読込みなどで例外のときは少し休んで続行
            time.sleep(0.2)
            continue

        # 最初に得られた非空テキストを採用（必要に応じてロジック変更）
        new_text = None
        for t in results:
            tt = str(t).strip()
            if tt:
                new_text = tt
                break

        now = time.time()
        with detected_lock:
            if new_text:
                # デバウンス処理：同じ語を短時間で何度も更新しない
                if new_text != last_detect or (now - last_detect_time) > debounce_sec:
                    last_detect = new_text
                    last_detect_time = now
                    detected_text_shared["text"] = new_text
                    detected_text_shared["timestamp"] = now
            # else: 未検出なら共有を変えない（再生継続目的の場合）

        # OCR間隔を守る
        time.sleep(ocr_interval)

# 非同期ワーカー起動（webrtc開始時に）
ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
ocr_thread.start()

# ---------------------
# VideoProcessor（非ブロッキング）
# ---------------------
import av
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        # 受信フレームはすぐ返す（表示が止まらないように）
        img = frame.to_ndarray(format="bgr24")
        # ROI枠を描画（割合を使って）
        h, w = img.shape[:2]
        x1 = int(w * (x0 / 100.0))
        y1 = int(h * (y0 / 100.0))
        x2 = int(x1 + w * (w_pct / 100.0))
        y2 = int(y1 + h * (h_pct / 100.0))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 最新フレームだけキューに入れる（古いのは捨てる）
        try:
            if frame_q.full():
                _ = frame_q.get_nowait()
        except queue.Empty:
            pass
        try:
            frame_q.put_nowait(img)
        except queue.Full:
            pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------------
# webrtc 起動
# ---------------------
webrtc_ctx = webrtc_streamer(
    key="async-ocr",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# ---------------------
# メインループ：検出共有を見て st.audio() で再生（クラウドでOK）
# ---------------------
audio_placeholder = st.empty()
status_ph = st.empty()
last_played = {"text": None, "time": 0.0}

if webrtc_ctx and webrtc_ctx.state.playing:
    status_ph.info("ストリーミング中。ROIはサイドバーで調整できます。")
else:
    status_ph.info("ストリーミングを開始してください。")

# メインスレッドで定期的に共有変数をチェック
def main_monitor():
    while webrtc_ctx and webrtc_ctx.state.playing:
        with detected_lock:
            text = detected_text_shared.get("text")
            ts = detected_text_shared.get("timestamp", 0.0)
        if text:
            # 遅延再生を考慮
            if time.time() - ts >= play_delay:
                # 別語なら再生（1回）
                if text != last_played["text"] or (time.time() - last_played["time"]) > 60:
                    # mapping に一致するファイルがあれば再生
                    audio_fn = mapping.get(text)
                    if audio_fn:
                        audio_path = os.path.join("sounds", audio_fn)
                        if os.path.exists(audio_path):
                            audio_placeholder.audio(audio_path)
                            last_played["text"] = text
                            last_played["time"] = time.time()
                        else:
                            st.warning(f"音声ファイルが見つかりません: {audio_path}")
                    else:
                        # 何もしない（マッピングがない）
                        pass
        time.sleep(0.5)

monitor_thread = threading.Thread(target=main_monitor, daemon=True)
monitor_thread.start()

# ---------------------
# 終了時クリーンアップ（ブラウザを閉じると stop_flag にする）
# ---------------------
def shutdown():
    stop_flag["stop"] = True

st.button("停止（ワーカーを終了）", on_click=shutdown)
