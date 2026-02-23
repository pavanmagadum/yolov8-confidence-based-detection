

import os

# -------------------------
# IMPORTANT: set BEFORE importing cv2
# -------------------------
RTSP_TRANSPORT = "tcp"  # "tcp" or "udp"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    f"rtsp_transport;{RTSP_TRANSPORT}|"
    f"fflags;nobuffer|"
    f"flags;low_delay|"
    f"max_delay;0|"
    f"stimeout;5000000"
)

import cv2
import numpy as np
import threading
import time
from flask import Flask, Response, request, jsonify

app = Flask(__name__)

# =========================
# Crack detection engine
# =========================
class CrackEngine:
    def __init__(self):
        # ✅ IP camera only

        self.ip_url = "rtsp://admin:20241221-1@10.134.251.194:554/Streaming/Channels/101"

        # crack settings.
        self.threshold1 = 50
        self.threshold2 = 150
        self.min_area = 100

        # performance / streaming settings
        self.resize_width = 960         # reduce to 640/800/960 for speed
        self.process_every_n = 2        # crack detection on every Nth frame
        self.jpeg_quality = 75          # 60-80 recommended for MJPEG
        self.stream_fps_limit = 30      # MJPEG push limit (browser friendly)

        self.lock = threading.Lock()
        self.running = False

        self.cap = None
        self.reader_thread = None
        self.processor_thread = None

        self.latest_frame = None          # newest frame from RTSP (BGR)
        self.frame_original = None        # resized original for streaming
        self.frame_result = None          # resized processed for streaming
        self.crack_count = 0

        self._frame_id = 0
        self._last_read_ok = False

    # -------- camera open helpers --------
    def _open_capture(self):
        url = (self.ip_url or "").strip()
        if not url:
            return None

        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            return None

        # reduce buffering (may or may not work depending on build)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        return cap

    def start(self):
        with self.lock:
            if self.running:
                return True

            cap = self._open_capture()
            if cap is None:
                self.cap = None
                return False

            self.cap = cap
            self.running = True
            self.latest_frame = None
            self.frame_original = None
            self.frame_result = None
            self.crack_count = 0
            self._frame_id = 0
            self._last_read_ok = False

            self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.processor_thread = threading.Thread(target=self._processor_loop, daemon=True)
            self.reader_thread.start()
            self.processor_thread.start()

            return True

    def stop(self):
        with self.lock:
            self.running = False
            cap = self.cap
            self.cap = None

        if cap is not None:
            try:
                cap.release()
            except:
                pass

    def update_settings(self, ip_url=None, threshold1=None, threshold2=None, min_area=None):
        restart = False

        with self.lock:
            if ip_url is not None:
                ip_url = str(ip_url).strip()
                if ip_url != self.ip_url and ip_url:
                    self.ip_url = ip_url
                    restart = True

            if threshold1 is not None:
                self.threshold1 = int(threshold1)
            if threshold2 is not None:
                self.threshold2 = int(threshold2)
            if min_area is not None:
                self.min_area = int(min_area)

        if restart:
            self.stop()
            time.sleep(0.2)
            return self.start()

        return True

    def _resize_keep_aspect(self, frame_bgr, target_w):
        h, w = frame_bgr.shape[:2]
        if w <= target_w:
            return frame_bgr
        scale = target_w / float(w)
        nh = int(h * scale)
        return cv2.resize(frame_bgr, (target_w, nh), interpolation=cv2.INTER_AREA)

    def _detect_cracks(self, frame_bgr_small):
        gray = cv2.cvtColor(frame_bgr_small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.threshold1, self.threshold2)

        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        crack_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_area]

        result = frame_bgr_small.copy()
        cv2.drawContours(result, crack_contours, -1, (0, 0, 255), 2)

        return result, len(crack_contours)

    def _reader_loop(self):
        """
        FAST RTSP reader:
        - grab()/retrieve() flushes buffered frames (low latency)
        - auto reconnect on repeated failure
        """
        fail_count = 0

        while True:
            with self.lock:
                if not self.running:
                    break
                cap = self.cap
                url = self.ip_url

            if cap is None:
                time.sleep(0.05)
                continue

            # Flush some buffered frames to reduce lag
            for _ in range(3):
                try:
                    cap.grab()
                except:
                    break

            ok, frame = cap.retrieve()
            if not ok or frame is None:
                self._last_read_ok = False
                fail_count += 1
                time.sleep(0.05)

                # reconnect after repeated failures
                if fail_count >= 20:
                    fail_count = 0
                    try:
                        cap.release()
                    except:
                        pass

                    new_cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                    if new_cap.isOpened():
                        new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        with self.lock:
                            self.cap = new_cap
                    else:
                        try:
                            new_cap.release()
                        except:
                            pass
                        time.sleep(0.2)

                continue

            fail_count = 0
            self._last_read_ok = True

            with self.lock:
                self.latest_frame = frame
                self._frame_id += 1

    def _processor_loop(self):
        last_processed_id = -1
        local_counter = 0

        while True:
            with self.lock:
                if not self.running:
                    break
                frame = None if self.latest_frame is None else self.latest_frame.copy()
                current_id = self._frame_id

            if frame is None or current_id == last_processed_id:
                time.sleep(0.005)
                continue

            last_processed_id = current_id
            local_counter += 1

            # resize early for speed (also for smoother MJPEG)
            small = self._resize_keep_aspect(frame, self.resize_width)

            # always update original stream
            with self.lock:
                self.frame_original = small

            # process crack detection only every N frames
            if local_counter % self.process_every_n == 0:
                result, count = self._detect_cracks(small)
                with self.lock:
                    self.frame_result = result
                    self.crack_count = count

            time.sleep(0.001)

    def get_jpeg(self, which="original"):
        with self.lock:
            frame = self.frame_original if which == "original" else self.frame_result
            count = self.crack_count
            running = self.running
            ip_url = self.ip_url
            last_ok = self._last_read_ok

        if frame is None:
            return None, count, running, ip_url, last_ok

        encode_params = [
            int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality),
            int(cv2.IMWRITE_JPEG_OPTIMIZE), 1
        ]
        ok, buf = cv2.imencode(".jpg", frame, encode_params)
        if not ok:
            return None, count, running, ip_url, last_ok

        return buf.tobytes(), count, running, ip_url, last_ok


engine = CrackEngine()
engine.start()

# =========================
# MJPEG helpers
# =========================
def _error_frame(msg):
    img = np.zeros((240, 900, 3), dtype=np.uint8)
    cv2.putText(img, msg, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return buf.tobytes() if ok else b""

def _mjpeg_chunk(jpg_bytes):
    return (b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Cache-Control: no-cache\r\n\r\n" + jpg_bytes + b"\r\n")

# =========================
# MJPEG streaming (fixed errors)
# =========================
def mjpeg_stream(which):
    frame_interval = 1.0 / max(1, engine.stream_fps_limit)
    last_sent = 0.0

    try:
        while True:
            now = time.time()
            if now - last_sent < frame_interval:
                time.sleep(0.001)
                continue
            last_sent = now

            jpg, _, running, ip_url, last_ok = engine.get_jpeg(which)
            if not running:
                yield _mjpeg_chunk(_error_frame("Camera stopped"))
                time.sleep(0.2)
                continue

            if jpg is None:
                msg = "Waiting for frames..." if last_ok else f"RTSP issue, reconnecting... ({ip_url})"
                yield _mjpeg_chunk(_error_frame(msg))
                time.sleep(0.05)
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Cache-Control: no-cache\r\n\r\n" + jpg + b"\r\n")

    except GeneratorExit:
        # client closed connection (normal)
        return
    except Exception:
        # prevent stream thread from crashing Flask
        return

# =========================
# Routes
# =========================
@app.get("/video/original")
def video_original():
    return Response(mjpeg_stream("original"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.get("/video/result")
def video_result():
    return Response(mjpeg_stream("result"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/status")
def api_status():
    with engine.lock:
        return jsonify({
            "running": engine.running,
            "ip_url": engine.ip_url,
            "threshold1": engine.threshold1,
            "threshold2": engine.threshold2,
            "min_area": engine.min_area,
            "crack_count": engine.crack_count,
            "rtsp_transport": RTSP_TRANSPORT,
            "resize_width": engine.resize_width,
            "process_every_n": engine.process_every_n,
            "jpeg_quality": engine.jpeg_quality
        })

@app.post("/api/settings")
def api_settings():
    data = request.get_json(force=True, silent=True) or {}
    ok = engine.update_settings(
        ip_url=data.get("ip_url", None),
        threshold1=data.get("threshold1", None),
        threshold2=data.get("threshold2", None),
        min_area=data.get("min_area", None),
    )
    return jsonify({"ok": bool(ok)})

@app.get("/")
def index():
    # FULL dashboard kept (graphs + direction + speed controls + crack video feeds)
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Crack Detection Vehicle Dashboard</title>

  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <script src="https://www.gstatic.com/firebasejs/9.22.2/firebase-app-compat.js"></script>
  <script src="https://www.gstatic.com/firebasejs/9.22.2/firebase-database-compat.js"></script>

  <style>
    :root{
      --bg:#0b1220; --muted:#8ea0c6; --text:#e9efff; --border:rgba(255,255,255,.08);
      --blue:#60a5fa; --red:#fb7185; --teal:#2dd4bf;
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif;
      background: radial-gradient(1200px 600px at 20% 0%, rgba(96,165,250,.18), transparent 60%),
                  radial-gradient(1200px 600px at 80% 20%, rgba(45,212,191,.14), transparent 55%),
                  var(--bg);
      color:var(--text);
    }
    .wrap{max-width:1280px;margin:0 auto;padding:18px}
    .top{display:flex;justify-content:space-between;gap:12px;margin-bottom:14px}
    h1{margin:0;font-size:18px}
    .sub{margin:4px 0 0;color:var(--muted);font-size:13px}
    .grid{display:grid;grid-template-columns:380px 1fr;gap:14px}
    .card{
      background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.03));
      border:1px solid var(--border);
      border-radius:16px;
      padding:14px;
      box-shadow: 0 10px 30px rgba(0,0,0,.22);
    }
    .row{display:flex;gap:10px;flex-wrap:wrap}
    .pill{
      background: rgba(0,0,0,.25);
      border:1px solid var(--border);
      border-radius:12px;
      padding:10px 12px;
      min-width:110px;
    }
    .k{font-size:12px;color:var(--muted)}
    .v{font-size:16px;font-weight:700;margin-top:2px}

    .dpad{width:200px;height:200px;border-radius:50%;background:rgba(0,0,0,.25);border:1px solid var(--border);
      margin:16px auto;position:relative}
    .dbtn{
      position:absolute;width:55px;height:55px;border-radius:50%;
      border:1px solid var(--border);background:rgba(0,0,0,.45);color:var(--text);
      font-size:22px;font-weight:700;cursor:pointer;
    }
    .dbtn:hover{background:rgba(255,255,255,.10)}
    .up{top:10px;left:50%;transform:translateX(-50%)}
    .down{bottom:10px;left:50%;transform:translateX(-50%)}
    .left{left:10px;top:50%;transform:translateY(-50%)}
    .right{right:10px;top:50%;transform:translateY(-50%)}
    .center{top:50%;left:50%;transform:translate(-50%,-50%);width:50px;height:50px}
    .blue{border-color:rgba(96,165,250,.35)}
    .red{border-color:rgba(251,113,133,.35)}
    .teal{border-color:rgba(45,212,191,.35)}

    button.big{
      width:100%;
      border:1px solid var(--border);
      background: rgba(0,0,0,.25);
      color:var(--text);
      padding:10px 12px;
      border-radius:12px;
      cursor:pointer;
      font-weight:700;
      margin-top:8px;
    }
    button.big:hover{background:rgba(255,255,255,.10)}
    .rightCol{display:grid;grid-template-rows:auto auto;gap:14px}
    .videoGrid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
    .videoBox{background:rgba(0,0,0,.25);border:1px solid var(--border);border-radius:16px;padding:12px}
    .videoBox h3{margin:0 0 8px 0;font-size:13px;color:#dbe7ff}
    .videoBox img{width:100%;border-radius:12px;border:1px solid var(--border);background:#000;display:block}
    canvas{width:100%!important;height:180px!important;background:rgba(0,0,0,.20);border:1px solid var(--border);border-radius:14px;padding:10px}
    .small{font-size:12px;color:var(--muted)}
    input[type="text"]{
      width:100%;
      padding:10px;
      border-radius:10px;
      border:1px solid var(--border);
      background:rgba(0,0,0,.25);
      color:var(--text);
      margin-top:6px;
    }
    @media (max-width:1050px){.grid{grid-template-columns:1fr}.videoGrid{grid-template-columns:1fr}}
  </style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div>
      <h1>Crack Detection Dashboard</h1>
      <div class="sub">IP camera Crack detection</div>
    </div>
  </div>

  <div class="grid">
    <!-- LEFT -->
    <div class="card">
      <h2 style="margin:0 0 10px 0;font-size:14px;color:#dbe7ff">Vehicle Controls (RTDB)</h2>

      <div class="row">
        <div class="pill"><div class="k">Voltage</div><div class="v" id="vVolt">--</div></div>
        <div class="pill"><div class="k">Battery</div><div class="v" id="vBatt">--</div></div>
        <div class="pill"><div class="k">Current</div><div class="v" id="vCurr">--</div></div>
        <div class="pill"><div class="k">Speed</div><div class="v" id="vSpeed">--</div></div>
      </div>

      <div class="pill" style="margin-top:10px;min-width:100%;display:flex;justify-content:space-between;align-items:center">
        <div>
          <div class="k">Direction (RTDB)</div>
          <div class="v" id="vDir">--</div>
        </div>
        <div class="small">Path:direction</div>
      </div>

      <div class="dpad">
        <button class="dbtn up blue" onclick="setDirection('F')">▲</button>
        <button class="dbtn left blue" onclick="setDirection('L')">◄</button>
        <button class="dbtn center red" onclick="setDirection('S')">■</button>
        <button class="dbtn right blue" onclick="setDirection('R')">►</button>
        <button class="dbtn down blue" onclick="setDirection('B')">▼</button>
      </div>

      <div class="row">
        <button class="big teal" style="flex:1" onclick="speedUp()">▲ Speed Up</button>
        <button class="big teal" style="flex:1" onclick="speedDown()">▼ Speed Down</button>
      </div>
      <div class="small" style="margin-top:6px">Speed range: 1 to 10 </div>

      <hr style="border:none;border-top:1px solid var(--border);margin:14px 0;">

      <h2 style="margin:0 0 10px 0;font-size:14px;color:#dbe7ff">Crack Detection Settings</h2>

      <div class="small">Canny Lower Threshold</div>
      <input id="t1" type="range" min="0" max="255" value="50" style="width:100%">

      <div class="small" style="margin-top:10px">Canny Upper Threshold</div>
      <input id="t2" type="range" min="0" max="255" value="150" style="width:100%">

      <div class="small" style="margin-top:10px">Minimum Area</div>
      <input id="minArea" type="range" min="10" max="1000" value="100" style="width:100%">

      <button class="big" onclick="applyPySettings()">Apply / Restart Camera</button>
      <div class="small" style="margin-top:8px">Crack regions found: <b id="crackCount">--</b></div>
    </div>

    <!-- RIGHT -->
    <div class="rightCol">
      <div class="card">
        <h2 style="margin:0 0 10px 0;font-size:14px;color:#dbe7ff">Live Crack Detection</h2>
        <div class="videoGrid">
          <div class="videoBox">
            <h3>IP Feed (Original)</h3>
            <img src="/video/original" alt="original stream">
          </div>
          <div class="videoBox">
            <h3>Detected Cracks</h3>
            <img src="/video/result" alt="result stream">
          </div>
        </div>
      </div>

      <div class="card">
        <h2 style="margin:0 0 10px 0;font-size:14px;color:#dbe7ff">Live Graphs (Firebase RTDB)</h2>

        <div style="margin-bottom:12px">
          <div class="small">Voltage</div>
          <canvas id="chartVoltage"></canvas>
        </div>

        <div style="margin-bottom:12px">
          <div class="small">Battery</div>
          <canvas id="chartBattery"></canvas>
        </div>

        <div>
          <div class="small">Current</div>
          <canvas id="chartCurrent"></canvas>
        </div>

        <div class="small" style="margin-top:10px">Reading: Voltage, Battery, Current</div>
      </div>
    </div>
  </div>
</div>

<script>
  // ===== Firebase config (UNCHANGED) =====
  const firebaseConfig = {
    apiKey: "AIzaSyAXHnvNZkb00PXbG5JidbD4PbRgf7L6Lg8",
    authDomain: "v2v-communication-d46c6.firebaseapp.com",
    databaseURL: "https://v2v-communication-d46c6-default-rtdb.firebaseio.com",
    projectId: "v2v-communication-d46c6",
    storageBucket: "v2v-communication-d46c6.firebasestorage.app",
    messagingSenderId: "536888356116",
    appId: "1:536888356116:web:c6bbab9c6faae7c84e2601",
    measurementId: "G-FXLP4KQXWM"
  };

  firebase.initializeApp(firebaseConfig);
  const db = firebase.database();

  const CAR_PATH = "Car1";
  const refVolt  = db.ref(CAR_PATH + "/Voltage");
  const refBatt  = db.ref(CAR_PATH + "/Battery");
  const refCurr  = db.ref(CAR_PATH + "/Current");
  const refDir   = db.ref(CAR_PATH + "/direction");
  const refSpeed = db.ref(CAR_PATH + "/Speed");

  const vVolt = document.getElementById("vVolt");
  const vBatt = document.getElementById("vBatt");
  const vCurr = document.getElementById("vCurr");
  const vDir  = document.getElementById("vDir");
  const vSpeed= document.getElementById("vSpeed");

  function formatNum(x){
    const n = Number(x);
    if (!Number.isFinite(n)) return "--";
    if (Math.abs(n) < 10) return n.toFixed(3);
    return n.toFixed(2);
  }

  // ===== Direction/Speed Controls (UNCHANGED) =====
  function setDirection(dir){ refDir.set(dir).catch(console.error); }

  async function speedUp(){
    const snap = await refSpeed.get();
    let sp = Number(snap.val() ?? 1);
    if (!Number.isFinite(sp)) sp = 1;
    sp = Math.min(10, sp + 1);
    await refSpeed.set(sp);
  }

  async function speedDown(){
    const snap = await refSpeed.get();
    let sp = Number(snap.val() ?? 1);
    if (!Number.isFinite(sp)) sp = 1;
    sp = Math.max(1, sp - 1);
    await refSpeed.set(sp);
  }

  // ===== Python status + settings (IP ONLY) =====
  async function refreshPyStatus(){
    try{
      const res = await fetch("/api/status");
      const js = await res.json();
      document.getElementById("pyStatus").textContent =
        `Python: ${js.running ? "running" : "stopped"} | ip=${(js.ip_url||"").slice(0,28)}... | transport=${js.rtsp_transport}`;
      document.getElementById("crackCount").textContent = js.crack_count ?? "--";

      document.getElementById("ipUrl").value = js.ip_url ?? "";
    }catch(e){
      document.getElementById("pyStatus").textContent = "Python: error";
    }
  }

  async function applyPySettings(){
    const payload = {
      ip_url: document.getElementById("ipUrl").value,
      threshold1: Number(document.getElementById("t1").value),
      threshold2: Number(document.getElementById("t2").value),
      min_area: Number(document.getElementById("minArea").value),
    };
    await fetch("/api/settings", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(payload)
    });
    refreshPyStatus();
  }

  // ===== Charts (UNCHANGED) =====
  const MAX_POINTS = 60;
  const labelsV = [], labelsB = [], labelsC = [];
  const dataV = [], dataB = [], dataC = [];

  const chartVoltage = new Chart(document.getElementById("chartVoltage"), {
    type:"line",
    data:{ labels:labelsV, datasets:[{ label:"Voltage", data:dataV, tension:0.25, pointRadius:0 }]},
    options:{ responsive:true, maintainAspectRatio:false, animation:false, plugins:{legend:{display:false}},
      scales:{ x:{ticks:{color:"#8ea0c6"},grid:{color:"rgba(255,255,255,.06)"}},
              y:{ticks:{color:"#8ea0c6"},grid:{color:"rgba(255,255,255,.06)"}} } }
  });

  const chartBattery = new Chart(document.getElementById("chartBattery"), {
    type:"line",
    data:{ labels:labelsB, datasets:[{ label:"Battery", data:dataB, tension:0.25, pointRadius:0 }]},
    options:{ responsive:true, maintainAspectRatio:false, animation:false, plugins:{legend:{display:false}},
      scales:{ x:{ticks:{color:"#8ea0c6"},grid:{color:"rgba(255,255,255,.06)"}},
              y:{ticks:{color:"#8ea0c6"},grid:{color:"rgba(255,255,255,.06)"}} } }
  });

  const chartCurrent = new Chart(document.getElementById("chartCurrent"), {
    type:"line",
    data:{ labels:labelsC, datasets:[{ label:"Current", data:dataC, tension:0.25, pointRadius:0 }]},
    options:{ responsive:true, maintainAspectRatio:false, animation:false, plugins:{legend:{display:false}},
      scales:{ x:{ticks:{color:"#8ea0c6"},grid:{color:"rgba(255,255,255,.06)"}},
              y:{ticks:{color:"#8ea0c6"},grid:{color:"rgba(255,255,255,.06)"}} } }
  });

  let latest = {Voltage:0,Battery:0,Current:0};

  function pushPoint(){
    const label = new Date().toLocaleTimeString();
    labelsV.push(label); dataV.push(Number(latest.Voltage)||0);
    labelsB.push(label); dataB.push(Number(latest.Battery)||0);
    labelsC.push(label); dataC.push(Number(latest.Current)||0);

    if(labelsV.length>MAX_POINTS){labelsV.shift();dataV.shift();}
    if(labelsB.length>MAX_POINTS){labelsB.shift();dataB.shift();}
    if(labelsC.length>MAX_POINTS){labelsC.shift();dataC.shift();}

    chartVoltage.update();
    chartBattery.update();
    chartCurrent.update();
  }

  // ===== Firebase listeners (UNCHANGED) =====
  refVolt.on("value", s => { latest.Voltage = s.val() ?? 0; vVolt.textContent = formatNum(latest.Voltage); });
  refBatt.on("value", s => { latest.Battery = s.val() ?? 0; vBatt.textContent = formatNum(latest.Battery); });
  refCurr.on("value", s => { latest.Current = s.val() ?? 0; vCurr.textContent = formatNum(latest.Current); });
  refDir.on("value",  s => { vDir.textContent = (s.val() ?? "--"); });
  refSpeed.on("value",s => { vSpeed.textContent = (s.val() ?? "--"); });

  setInterval(pushPoint, 1000);
  refreshPyStatus();
  setInterval(refreshPyStatus, 2000);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    # debug=False avoids double-reload which can cause stream lag
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
