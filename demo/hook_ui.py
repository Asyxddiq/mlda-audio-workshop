"""Live-mic recorder UI for `demo/hook_demo.ipynb` -- Record/Stop buttons, a
timer, and an on-canvas AnalyserNode visualizer (scrolling waveform +
frequency bars) plus a silence meter ("pause > 1.5s = next speaker",
matching `transcriber.py`'s speaker-change gap).

Presentation only -- doesn't touch labelling, minutes generation, or
`prototype/`. The banner, progress strip, minutes cards, and the dark
waveform/spectrogram playback panel are in `demo/hook_theme.py`; this file
is the recorder only.

The mic-recording wire path (ffmpeg conversion, webm -> wav, stopping the
mic tracks) is the same as the original `record_skit()`; only the on-page
UI changed, from a text "press Enter" prompt to on-canvas Record/Stop
buttons with a live waveform, frequency bars, and a silence meter.

Facilitator-only -- bundled into `facilitator_files.zip` by
`notebook/make_bundle.py`, not imported by the participants' notebook.
"""
import subprocess
from base64 import b64decode

from IPython.display import HTML, Javascript, display

BG0 = "#0D0D12"
PURPLE = "#8B5CF6"
PURPLE2 = "#A855F7"
PINK = "#EC4899"
TEXT = "#F4F4F8"
MUTED = "#9CA3AF"

_RECORDER_CSS = """
<style>
.wsxr * { font-family: 'Poppins', 'Segoe UI', system-ui, sans-serif; box-sizing: border-box; }
.wsxr-card {
  background: #2E2E3C; border-radius: 16px; padding: 20px 24px; margin: 10px 0;
  border: 1px solid #ffffff14; box-shadow: 0 0 26px __PURPLE__22; color: __TEXT__;
}
.wsxr-card h3 { margin: 0 0 10px 0; font-size: 26px; font-weight: 700; color: __TEXT__; }
.wsxr-record-btn, .wsxr-stop-btn {
  border: none; border-radius: 999px; font-weight: 600; font-size: 18px;
  padding: 14px 28px; cursor: pointer; color: __TEXT__; margin-right: 10px;
}
.wsxr-record-btn {
  background: radial-gradient(circle at 30% 30%, __PURPLE2__, __PURPLE__);
  box-shadow: 0 0 22px __PURPLE__88;
}
.wsxr-record-btn:disabled { opacity: 0.35; cursor: default; box-shadow: none; }
.wsxr-stop-btn { background: #2E2E3C; border: 1px solid __PINK__; }
.wsxr-stop-btn:disabled { opacity: 0.35; cursor: default; }
.wsxr-timer { font-size: 28px; font-weight: 700; color: __PURPLE2__; margin-left: 6px; letter-spacing: 1px; }
.wsxr-status { font-size: 16px; color: __MUTED__; margin-top: 6px; min-height: 20px; }
.wsxr-canvas-wrap { margin-top: 14px; border-radius: 12px; overflow: hidden; background: __BG0__; }
.wsxr-silence-row { margin-top: 10px; }
.wsxr-silence-label { font-size: 16px; color: __MUTED__; margin-bottom: 4px; }
.wsxr-silence-track { background: #ffffff14; border-radius: 999px; height: 8px; overflow: hidden; }
.wsxr-silence-fill { background: linear-gradient(90deg, __PURPLE__, __PINK__); height: 100%; width: 0%; transition: width 0.08s linear; }
.wsxr-silence-flash { color: __PINK__; font-weight: 600; }
</style>
"""

# Silence threshold: 1.5s of quiet = transcriber.py's speaker-change gap.
SILENCE_MS = 1500

_RECORDER_JS = """
(function() {
  const root = document.getElementById("__ROOT_ID__");
  const recordBtn = root.querySelector(".wsxr-record-btn");
  const stopBtn = root.querySelector(".wsxr-stop-btn");
  const timerEl = root.querySelector(".wsxr-timer");
  const statusEl = root.querySelector(".wsxr-status");
  const silenceFill = root.querySelector(".wsxr-silence-fill");
  const silenceLabel = root.querySelector(".wsxr-silence-label");
  const canvas = root.querySelector(".wsxr-canvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;

  let stream, recorder, chunks = [], audioCtx, analyser, rafId, startTime;
  let lastLoudTime = 0;

  function fmtTime(ms) {
    const s = Math.floor(ms / 1000);
    return String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
  }

  function draw() {
    const timeData = new Uint8Array(analyser.fftSize);
    const freqData = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteTimeDomainData(timeData);
    analyser.getByteFrequencyData(freqData);

    ctx.fillStyle = "__BG0__";
    ctx.fillRect(0, 0, W, H);

    const barCount = 48;
    const step = Math.floor(freqData.length / barCount);
    const barW = W / barCount;
    for (let i = 0; i < barCount; i++) {
      const v = freqData[i * step] / 255;
      const barH = v * H * 0.85;
      const grad = ctx.createLinearGradient(0, H, 0, H - barH);
      grad.addColorStop(0, "__PURPLE__" + "44");
      grad.addColorStop(1, "__PINK__" + "99");
      ctx.fillStyle = grad;
      ctx.fillRect(i * barW + 1, H - barH, barW - 2, barH);
    }

    ctx.lineWidth = 2.5;
    ctx.strokeStyle = "__PURPLE2__";
    ctx.shadowBlur = 8;
    ctx.shadowColor = "__PURPLE__";
    ctx.beginPath();
    let maxDev = 0;
    for (let i = 0; i < timeData.length; i++) {
      const x = (i / timeData.length) * W;
      const v = timeData[i] / 128.0 - 1.0;
      maxDev = Math.max(maxDev, Math.abs(v));
      const y = H / 2 + v * H * 0.4;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    const now = performance.now();
    if (maxDev > 0.04) { lastLoudTime = now; }
    const quietMs = now - lastLoudTime;
    const pct = Math.min(quietMs / __SILENCE_MS__, 1) * 100;
    silenceFill.style.width = pct.toFixed(0) + "%";
    if (quietMs >= __SILENCE_MS__) {
      silenceLabel.innerHTML = '<span class="wsxr-silence-flash">pause &gt; 1.5s -- next speaker!</span>';
    } else {
      silenceLabel.textContent = "pause > 1.5s = next speaker";
    }

    timerEl.textContent = fmtTime(now - startTime);
    rafId = requestAnimationFrame(draw);
  }

  function b2text(blob) {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result);
      reader.readAsDataURL(blob);
    });
  }

  window["__RESULT_PROMISE__"] = new Promise((resolveOuter) => {
    recordBtn.onclick = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      } catch (e) {
        statusEl.textContent = "Mic permission denied, or no mic found.";
        resolveOuter(JSON.stringify({ ok: false, error: String(e) }));
        return;
      }
      recordBtn.disabled = true;
      stopBtn.disabled = false;
      statusEl.textContent = "Recording...";
      chunks = [];
      recorder = new MediaRecorder(stream);
      recorder.ondataavailable = (e) => chunks.push(e.data);
      recorder.start();

      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioCtx.createMediaStreamSource(stream);
      analyser = audioCtx.createAnalyser();
      analyser.fftSize = 1024;
      source.connect(analyser);

      startTime = performance.now();
      lastLoudTime = startTime;
      draw();

      stopBtn.onclick = async () => {
        stopBtn.disabled = true;
        statusEl.textContent = "Converting...";
        cancelAnimationFrame(rafId);
        recorder.onstop = async () => {
          stream.getTracks().forEach((t) => t.stop());  // release the mic, stop the recording indicator
          try { await audioCtx.close(); } catch (e) {}
          const blob = new Blob(chunks);
          const dataUrl = await b2text(blob);
          resolveOuter(JSON.stringify({ ok: true, data: dataUrl }));
        };
        recorder.stop();
      };
    };
  });
})();
"""

_RECORDER_HTML = """
<div id="__ROOT_ID__" class="wsxr wsxr-card">
  <h3>__TITLE__</h3>
  <button class="wsxr-record-btn">&#9679; Record</button>
  <button class="wsxr-stop-btn" disabled>&#9632; Stop</button>
  <span class="wsxr-timer">00:00</span>
  <div class="wsxr-status"></div>
  <div class="wsxr-canvas-wrap"><canvas class="wsxr-canvas" width="640" height="150"></canvas></div>
  <div class="wsxr-silence-row">
    <div class="wsxr-silence-label">pause > 1.5s = next speaker</div>
    <div class="wsxr-silence-track"><div class="wsxr-silence-fill"></div></div>
  </div>
</div>
"""


def _css():
    return (_RECORDER_CSS
            .replace("__BG0__", BG0).replace("__PURPLE2__", PURPLE2)
            .replace("__PURPLE__", PURPLE).replace("__PINK__", PINK)
            .replace("__TEXT__", TEXT).replace("__MUTED__", MUTED))


def _render_recorder(root_id: str, title: str) -> None:
    html = _RECORDER_HTML.replace("__ROOT_ID__", root_id).replace("__TITLE__", title)
    js = (_RECORDER_JS
          .replace("__ROOT_ID__", root_id)
          .replace("__RESULT_PROMISE__", root_id + "_result")
          .replace("__SILENCE_MS__", str(SILENCE_MS))
          .replace("__BG0__", BG0).replace("__PURPLE2__", PURPLE2)
          .replace("__PURPLE__", PURPLE).replace("__PINK__", PINK))
    display(HTML(_css() + html))
    display(Javascript(js))


def record_with_ui(out_path: str = "skit.wav", title: str = "Record the skit") -> str | None:
    """Render the Record/Stop UI with a live waveform, frequency bars, and a
    silence meter, then block (via `eval_js`) until Stop is clicked.

    Returns the saved WAV path, or `None` if the mic was denied/unavailable
    (prints a plain message pointing at the backup WAV -- cells 8, 10).

    The ffmpeg webm -> wav conversion and the mic-release step are unchanged
    from the original `record_skit()`.
    """
    from google.colab.output import eval_js

    root_id = "wsxrec" + str(abs(hash(out_path)) % 100000)
    _render_recorder(root_id, title)

    import json
    raw = eval_js(f"{root_id}_result")
    result = json.loads(raw)
    if not result.get("ok"):
        print(f"No mic found, or permission denied. Use the backup WAV instead (cells 8, 10). ({result.get('error', '')})")
        return None

    data_url = result["data"]
    _, encoded = data_url.split(",", 1)
    webm_path = out_path.replace(".wav", ".webm")
    with open(webm_path, "wb") as f:
        f.write(b64decode(encoded))
    proc = subprocess.run(["ffmpeg", "-y", "-i", webm_path, out_path], capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError(f"ffmpeg failed converting {webm_path} -> {out_path} (see stderr above)")
    print(f"Saved {out_path}")
    return out_path
