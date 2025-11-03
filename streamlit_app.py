# streamlit_app.py
import streamlit as st
import pandas as pd
import json
from pathlib import Path
from streamlit.components.v1 import html

st.set_page_config(layout="wide", page_title="OCR→遅延再生（ROI選択対応）")

st.title("ブラウザOCR → ROIドラッグ選択 + 20秒遅延再生")
st.markdown("カメラはブラウザで処理（Tesseract.js）。プレビュー上でドラッグしてROIを決められるよ。")

# --- CSV読み込み ---
csv_path = Path("mapping.csv")
if not csv_path.exists():
    st.error("mapping.csv が見つかりません。ルートに作成して `word,audio` の形式で保存してね。")
    st.stop()

df = pd.read_csv(csv_path)
st.write("読み込んだマッピング（左がキーワード、右が音声ファイルパス）:")
st.dataframe(df)

mapping = {str(r['word']): str(r['audio']) for _, r in df.iterrows()}
mapping_json = json.dumps(mapping)

# UI 設定（Streamlit側）
col1, col2 = st.columns([1, 2])
with col1:
    play_delay = st.number_input("再生遅延（秒）", min_value=0.0, value=20.0, step=1.0)
    ocr_interval = st.number_input("OCRチェック間隔（秒）", min_value=0.1, value=0.8, step=0.1)
    lang_choice = st.selectbox("OCR言語（Tesseract）", options=["jpn+eng", "eng", "jpn"], index=0)
    allow_list = st.text_input("検出対象をカンマ区切りで限定（空は全部）", value="")
    st.write("※ 音声ファイルは `audio/xxx.wav` のようにプロジェクトに置いてね。")

with col2:
    st.markdown("#### カメラプレビュー（ドラッグでROI選択 → ROI保存でその範囲のみOCR）")
    st.caption("ブラウザでカメラを許可してね。")

# --- Client-side HTML/JS component (Tesseract.js + ROI drag + scheduling + playback) ---
component_html_template = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://cdn.jsdelivr.net/npm/tesseract.js@4/dist/tesseract.min.js"></script>
  <style>
    body{font-family: Arial, Helvetica, sans-serif;}
    #video{border:1px solid #444; max-width: 100%; height: auto; display:block;}
    #canvas_overlay{position: absolute; left:0; top:0;}
    #container{position: relative; display:inline-block;}
    #controls{margin-top:8px;}
    button{margin-right:6px;}
    #detected{font-size:1.1em; margin-top:8px; color: #0a0;}
  </style>
</head>
<body>
  <div id="container">
    <video id="video" autoplay playsinline></video>
    <canvas id="canvas_overlay"></canvas>
  </div>
  <div id="controls">
    <button id="save_roi">ROI保存</button>
    <button id="reset_roi">ROIリセット</button>
    <button id="cancel_schedule">スケジュールキャンセル</button>
    <span id="roi_info">ROI: 全画面</span>
  </div>
  <div id="detected">認識: -</div>

  <script>
    const mapping = __MAPPING__;
    const PLAY_DELAY_SEC = __PLAY_DELAY__;
    const OCR_INTERVAL = __OCR_INTERVAL__ * 1000;
    const LANG = "__LANG__";
    const allowListRaw = "__ALLOW_LIST__";
    const allowList = allowListRaw ? allowListRaw.split(',').map(s=>s.trim()).filter(s=>s) : null;

    let scheduledTimer = null;
    let scheduledKey = null;
    let currentPlayingKey = null;

    // ROI state (in video pixel coords)
    let roi = null;         // saved ROI: {x,y,w,h} or null for full
    let tempRoi = null;     // during drag
    let isDragging = false;
    let dragStart = null;

    function cancelScheduled() {
      if(scheduledTimer) {
        clearTimeout(scheduledTimer);
        scheduledTimer = null;
        scheduledKey = null;
      }
    }

    // play audio in browser once
    function playOnce(key) {
      const path = mapping[key];
      if(!path) return;
      // stop existing
      const existing = document.getElementById('audio_player');
      if(existing) { existing.pause(); existing.remove(); }
      const audio = document.createElement('audio');
      audio.id = 'audio_player';
      audio.src = path;
      audio.autoplay = true;
      audio.onended = ()=>{ currentPlayingKey = null; };
      document.body.appendChild(audio);
      currentPlayingKey = key;
    }

    function schedulePlayFor(key) {
      // if another key playing, stop immediately
      if(currentPlayingKey && currentPlayingKey !== key) {
        const ex = document.getElementById('audio_player');
        if(ex) { ex.pause(); ex.remove(); }
        currentPlayingKey = null;
      }
      // cancel previous schedule
      cancelScheduled();
      scheduledKey = key;
      scheduledTimer = setTimeout(()=>{ 
        if(scheduledKey === key) {
          playOnce(key);
          scheduledTimer = null;
          scheduledKey = null;
        }
      }, PLAY_DELAY_SEC * 1000);
    }

    function init() {
      const video = document.getElementById('video');
      const canvas = document.getElementById('canvas_overlay');
      const ctx = canvas.getContext('2d');
      const saveBtn = document.getElementById('save_roi');
      const resetBtn = document.getElementById('reset_roi');
      const cancelBtn = document.getElementById('cancel_schedule');
      const roiInfo = document.getElementById('roi_info');
      // get camera
      navigator.mediaDevices.getUserMedia({video:true, audio:false})
        .then(stream => {
          video.srcObject = stream;
          video.onloadedmetadata = () => {
            // size canvas to video display size
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.style.width = video.videoWidth + 'px';
            canvas.style.height = video.videoHeight + 'px';
            canvas.style.position = 'absolute';
            canvas.style.left = video.offsetLeft + 'px';
            canvas.style.top = video.offsetTop + 'px';
            // start OCR worker after video ready
            startOCRLoop(video, canvas);
            drawLoop();
          };
        })
        .catch(err => {
          document.getElementById('detected').innerText = "カメラが使えません: " + err.message;
        });

      // mouse events for ROI on canvas (display coords = video pixel coords here)
      canvas.addEventListener('mousedown', (e) => {
        isDragging = true;
        const rect = canvas.getBoundingClientRect();
        dragStart = {x: e.clientX - rect.left, y: e.clientY - rect.top};
        tempRoi = null;
      });
      canvas.addEventListener('mousemove', (e) => {
        if(!isDragging) return;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const x1 = Math.min(dragStart.x, x), y1 = Math.min(dragStart.y, y);
        const x2 = Math.max(dragStart.x, x), y2 = Math.max(dragStart.y, y);
        tempRoi = {x: x1, y: y1, w: x2-x1, h: y2-y1};
      });
      canvas.addEventListener('mouseup', (e) => {
        isDragging = false;
        // leave tempRoi until saved
      });
      canvas.addEventListener('mouseleave', (e) => {
        isDragging = false;
      });

      saveBtn.addEventListener('click', () => {
        if(tempRoi) {
          roi = tempRoi;
          roiInfo.innerText = 'ROI: ' + Math.round(roi.x) + ',' + Math.round(roi.y) + ',' + Math.round(roi.w) + 'x' + Math.round(roi.h);
        } else {
          roi = null;
          roiInfo.innerText = 'ROI: 全画面';
        }
      });
      resetBtn.addEventListener('click', () => {
        roi = null;
        tempRoi = null;
        roiInfo.innerText = 'ROI: 全画面';
      });
      cancelBtn.addEventListener('click', () => {
        cancelScheduled();
      });

      // draw overlay loop
      function drawLoop() {
        ctx.clearRect(0,0,canvas.width, canvas.height);
        // draw saved ROI (blue)
        if(roi) {
          ctx.strokeStyle = 'rgba(0,0,255,0.9)';
          ctx.lineWidth = 3;
          ctx.strokeRect(roi.x, roi.y, roi.w, roi.h);
        }
        // draw temp ROI (red)
        if(tempRoi) {
          ctx.strokeStyle = 'rgba(255,0,0,0.9)';
          ctx.lineWidth = 2;
          ctx.strokeRect(tempRoi.x, tempRoi.y, tempRoi.w, tempRoi.h);
        }
        requestAnimationFrame(drawLoop);
      }
    }

    async function startOCRLoop(video, canvas) {
      const worker = Tesseract.createWorker({logger: m => { /* progress */ }});
      await worker.load();
      await worker.loadLanguage(LANG);
      await worker.initialize(LANG);

      const detectCanvas = document.createElement('canvas');
      const detectCtx = detectCanvas.getContext('2d');

      setInterval(async () => {
        if(video.readyState < 2) return;
        // pick source area: ROI if set, else whole video
        const vw = video.videoWidth, vh = video.videoHeight;
        let sx=0, sy=0, sw=vw, sh=vh;
        if(roi) {
          sx = Math.max(0, Math.floor(roi.x));
          sy = Math.max(0, Math.floor(roi.y));
          sw = Math.max(1, Math.floor(roi.w));
          sh = Math.max(1, Math.floor(roi.h));
        } else {
          sx=0; sy=0; sw=vw; sh=vh;
        }
        // resize for speed
        const targetW = Math.min(640, sw);
        const scale = targetW / sw;
        const targetH = Math.max(1, Math.floor(sh * scale));
        detectCanvas.width = targetW;
        detectCanvas.height = targetH;
        detectCtx.drawImage(video, sx, sy, sw, sh, 0, 0, targetW, targetH);

        try {
          const res = await worker.recognize(detectCanvas);
          const recognized = (res.data && res.data.text) ? res.data.text.trim() : '';
          if(recognized) {
            let tokens = recognized.split(/\\s+|\\n|\\.|,|、|。/).map(s=>s.trim()).filter(s=>s);
            if(allowList) tokens = tokens.filter(t => allowList.includes(t));
            let matched = null;
            for(const t of tokens) {
              for(const key of Object.keys(mapping)) {
                if (t.includes(key) || key.includes(t)) { matched = key; break; }
              }
              if(matched) break;
            }
            if(matched) {
              document.getElementById('detected').innerText = "認識: " + matched;
              schedulePlayFor(matched);
            } else {
              document.getElementById('detected').innerText = "認識: -";
            }
          } else {
            // nothing recognized this cycle
          }
        } catch(e) {
          console.log('OCR error', e);
        }
      }, OCR_INTERVAL);
    }

    // init after DOM loaded
    window.addEventListener('DOMContentLoaded', init);
  </script>
</body>
</html>
"""

# Replace placeholders safely
component_html = component_html_template.replace("__MAPPING__", mapping_json)
component_html = component_html.replace("__PLAY_DELAY__", str(play_delay))
component_html = component_html.replace("__OCR_INTERVAL__", str(ocr_interval))
component_html = component_html.replace("__LANG__", lang_choice)
component_html = component_html.replace("__ALLOW_LIST__", allow_list.replace('"', ''))

# embed the HTML
html(component_html, height=640, scrolling=True)
