"""
Camera vision module for voice assistant.

Captures frames from a USB camera and encodes them as base64 JPEG
for use with vision-language models (VLMs) via Ollama.

Includes a live preview overlay (OpenCV window and/or MJPEG browser stream)
showing assistant state, VLM responses, and watch status.

Requires: opencv-python-headless >= 4.8.0 (or opencv-python for --show-vision)
    pip install jetson-assistant[vision]
"""

import asyncio
import json as _json
import logging
import queue
import struct
import time
import threading
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ── Live transcript HTML page (served at http://<host>:9090/) ──
_TRANSCRIPT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>jetson-assistant</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0f0f0f; color: #e0e0e0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; height: 100vh; display: flex; }
  .camera { width: 45%; background: #000; display: flex; align-items: center; justify-content: center; position: relative; }
  .camera img { width: 100%; height: 100%; object-fit: contain; }
  .camera-label { position: absolute; top: 16px; left: 16px; font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1.5px; }
  .chat-panel { width: 55%; display: flex; flex-direction: column; border-left: 1px solid #1a1a1a; }
  .chat-header { padding: 20px 24px; border-bottom: 1px solid #1a1a1a; position: relative; }
  .chat-header h1 { font-size: 16px; font-weight: 600; color: #fff; }
  .chat-header p { font-size: 12px; color: #666; margin-top: 4px; }
  .chat-messages { flex: 1; overflow-y: auto; padding: 20px 24px; display: flex; flex-direction: column; gap: 12px; }
  .msg { max-width: 85%; padding: 10px 14px; border-radius: 12px; font-size: 14px; line-height: 1.5; animation: fadeIn 0.2s ease; }
  .msg .ts { font-size: 10px; color: #555; margin-top: 4px; }
  .msg-user { align-self: flex-end; background: #2563eb; color: #fff; border-bottom-right-radius: 4px; }
  .msg-user .ts { color: rgba(255,255,255,0.5); }
  .msg-assistant { align-self: flex-start; background: #1e1e1e; color: #e0e0e0; border-bottom-left-radius: 4px; }
  .status { padding: 12px 24px; border-top: 1px solid #1a1a1a; display: flex; align-items: center; gap: 8px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: #22c55e; }
  .dot.listening { background: #eab308; animation: pulse 1s infinite; }
  .status span { font-size: 12px; color: #888; }
  .pipeline { padding: 8px 24px; border-top: 1px solid #1a1a1a; display: flex; gap: 16px; }
  .pipe-stage { font-size: 10px; color: #555; text-transform: uppercase; letter-spacing: 1px; }
  .pipe-stage.active { color: #22c55e; }
  .pipe-stage .label { display: block; }
  .pipe-stage .val { color: #888; font-variant-numeric: tabular-nums; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 10px; background: #1a2e1a; color: #22c55e; margin-left: 8px; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
</style>
</head>
<body>
<div class="camera">
  <span class="camera-label">Live Camera</span>
  <img src="/video" alt="camera">
</div>
<div class="chat-panel">
  <div class="chat-header">
    <h1>jetson-assistant</h1>
    <p>On-device voice AI &mdash; Jetson Thor</p>
    <button id="clear-btn" onclick="fetch('/clear',{method:'POST'}).then(()=>{document.getElementById('msgs').innerHTML='';document.getElementById('debug-panel').innerHTML='';})" style="position:absolute;right:24px;top:20px;background:#1e1e1e;border:1px solid #333;color:#888;padding:6px 14px;border-radius:6px;font-size:12px;cursor:pointer;">Clear</button>
  </div>
  <div class="chat-messages" id="msgs"></div>
  <div class="pipeline">
    <div class="pipe-stage" id="ps-stt"><span class="label">STT</span><span class="val" id="pv-stt">--</span></div>
    <div class="pipe-stage" id="ps-llm"><span class="label">LLM</span><span class="val" id="pv-llm">--</span></div>
    <div class="pipe-stage" id="ps-tts"><span class="label">TTS</span><span class="val" id="pv-tts">--</span></div>
    <span class="badge">STREAMING</span>
  </div>
  <div id="cert-help" style="display:none;background:#2a1a00;border:1px solid #f59e0b;color:#fbbf24;padding:10px 14px;margin:0 12px 8px;border-radius:8px;font-size:12px;line-height:1.5;">
    <strong>iOS/Mobile:</strong> Mic requires a trusted certificate.<br>
    1. Download <a href="/cert" style="color:#60a5fa;text-decoration:underline;">jetson-assistant.pem</a><br>
    2. Install: Settings &gt; General &gt; VPN &amp; Device Management<br>
    3. Trust: Settings &gt; General &gt; About &gt; Certificate Trust Settings<br>
    4. Reload this page and try again.
  </div>
  <div id="debug-panel" style="display:none;background:#0a0a0a;border-top:1px solid #333;max-height:200px;overflow-y:auto;font-family:monospace;font-size:11px;padding:6px 10px;"></div>
  <div class="status">
    <div class="dot" id="dot"></div>
    <span id="status">Connected</span>
    <button id="debug-btn" style="background:#1e1e1e;border:1px solid #333;color:#666;padding:4px 8px;border-radius:4px;font-size:10px;cursor:pointer;margin-left:8px;">DBG</button>
    <button id="mic-btn" style="margin-left:auto;background:#1e1e1e;border:1px solid #333;color:#e0e0e0;padding:6px 14px;border-radius:6px;font-size:12px;cursor:pointer;">&#127908; Connect Audio</button>
  </div>
</div>
<script>
const msgs = document.getElementById('msgs');
const dot = document.getElementById('dot');
const status = document.getElementById('status');
const es = new EventSource('/events');
let streamEl = null;
let streamText = '';
const debugPanel = document.getElementById('debug-panel');
const debugBtn = document.getElementById('debug-btn');
let debugVisible = false;
debugBtn.onclick = () => {
  debugVisible = !debugVisible;
  debugPanel.style.display = debugVisible ? 'block' : 'none';
  debugBtn.style.color = debugVisible ? '#22c55e' : '#666';
};
const stageColors = {intent:'#60a5fa',llm_raw:'#a78bfa',tool_call:'#f59e0b',tool_result:'#22c55e',vision_fallback:'#ef4444',summarize_input:'#818cf8',summarize_output:'#34d399',summarize_error:'#ef4444'};
es.onmessage = (e) => {
  const d = JSON.parse(e.data);
  if (d.type === 'debug') {
    const color = stageColors[d.stage] || '#888';
    const line = document.createElement('div');
    line.style.cssText = 'margin:2px 0;word-break:break-all;';
    line.innerHTML = '<span style="color:' + color + ';font-weight:bold;">[' + d.stage + ']</span> <span style="color:#ccc;">' + d.ts + '</span> ' + d.data.replace(/</g,'&lt;');
    debugPanel.appendChild(line);
    debugPanel.scrollTop = debugPanel.scrollHeight;
    return;
  }
  if (d.type === 'timing') {
    const el = document.getElementById('pv-' + d.stage);
    const ps = document.getElementById('ps-' + d.stage);
    if (el) { el.textContent = d.ms + 'ms'; }
    if (ps) { ps.className = 'pipe-stage active'; setTimeout(() => ps.className = 'pipe-stage', 2000); }
    return;
  }
  if (d.type === 'stream') {
    if (!streamEl) {
      streamEl = document.createElement('div');
      streamEl.className = 'msg msg-assistant';
      streamEl.innerHTML = '<span class="stream-text"></span><div class="ts">' + d.ts + '</div>';
      msgs.appendChild(streamEl);
      streamText = '';
    }
    streamText += (streamText ? ' ' : '') + d.text;
    streamEl.querySelector('.stream-text').textContent = streamText;
    msgs.scrollTop = msgs.scrollHeight;
    dot.className = 'dot listening'; status.textContent = 'Speaking...';
    return;
  }
  if (d.role === 'assistant' && streamEl) { streamEl = null; streamText = ''; return; }
  if (d.role === 'assistant') {
    const div = document.createElement('div');
    div.className = 'msg msg-assistant';
    div.innerHTML = d.text + '<div class="ts">' + d.ts + '</div>';
    msgs.appendChild(div);
  } else {
    streamEl = null; streamText = '';
    const div = document.createElement('div');
    div.className = 'msg msg-user';
    div.innerHTML = d.text + '<div class="ts">' + d.ts + '</div>';
    msgs.appendChild(div);
    dot.className = 'dot listening'; status.textContent = 'Processing...';
  }
  msgs.scrollTop = msgs.scrollHeight;
  if (d.role === 'assistant') { dot.className = 'dot'; status.textContent = 'Listening'; }
};
es.onerror = () => { dot.className = 'dot'; status.textContent = 'Reconnecting...'; };

/* ── Browser Audio (mic capture + TTS playback over WebSocket) ── */
let ws = null;
let audioCtx = null;
let micStream = null;
let nextPlayTime = 0;
let playCtx = null;

const micBtn = document.getElementById('mic-btn');

micBtn.onclick = async () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.close();
    if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
    if (audioCtx) { audioCtx.close(); audioCtx = null; }
    micBtn.textContent = '\\u{1F3A4} Connect Audio';
    micBtn.style.borderColor = '#333';
    status.textContent = 'Disconnected from browser audio';
    return;
  }

  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(proto + '//' + location.host + '/ws');
  ws.binaryType = 'arraybuffer';

  ws.onopen = async () => {
    micBtn.textContent = '\\u23F9 Disconnect';
    micBtn.style.borderColor = '#22c55e';
    status.textContent = 'Browser audio connected';

    try {
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
      });
      audioCtx = new AudioContext({ sampleRate: 16000 });
      const source = audioCtx.createMediaStreamSource(micStream);
      const processor = audioCtx.createScriptProcessor(2048, 1, 1);
      processor.onaudioprocess = (e) => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        const f32 = e.inputBuffer.getChannelData(0);
        const i16 = new Int16Array(f32.length);
        for (let i = 0; i < f32.length; i++) {
          i16[i] = Math.max(-32768, Math.min(32767, f32[i] * 32768));
        }
        ws.send(i16.buffer);
      };
      source.connect(processor);
      processor.connect(audioCtx.destination);
    } catch (err) {
      status.textContent = 'Mic error: ' + err.message;
      // Show cert install help for iOS/mobile (self-signed cert not trusted)
      if (err.name === 'NotAllowedError' || err.message.includes('not allowed')) {
        const help = document.getElementById('cert-help');
        if (help) help.style.display = 'block';
      }
      ws.close();
    }
  };

  ws.onmessage = (e) => {
    if (typeof e.data === 'string') {
      const msg = JSON.parse(e.data);
      if (msg.type === 'config') { /* future use */ }
      return;
    }
    /* Binary: 4-byte little-endian sample rate + int16 PCM */
    const view = new DataView(e.data);
    const sr = view.getUint32(0, true);
    const i16 = new Int16Array(e.data, 4);
    playAudio(i16, sr);
  };

  ws.onclose = () => {
    micBtn.textContent = '\\u{1F3A4} Connect Audio';
    micBtn.style.borderColor = '#333';
    if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
    if (audioCtx) { audioCtx.close(); audioCtx = null; }
  };

  ws.onerror = () => { status.textContent = 'WebSocket error'; };
};

function playAudio(i16, sr) {
  if (!playCtx || playCtx.sampleRate !== sr) {
    if (playCtx) playCtx.close();
    playCtx = new AudioContext({ sampleRate: sr });
    nextPlayTime = 0;
  }
  const f32 = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768.0;
  const buf = playCtx.createBuffer(1, f32.length, sr);
  buf.getChannelData(0).set(f32);
  const src = playCtx.createBufferSource();
  src.buffer = buf;
  src.connect(playCtx.destination);
  const now = playCtx.currentTime;
  if (nextPlayTime < now) nextPlayTime = now;
  src.start(nextPlayTime);
  nextPlayTime += buf.duration;
}
</script>
</body>
</html>"""

import numpy as np


@dataclass
class CameraConfig:
    """Configuration for camera capture."""

    device: int = 0
    width: int = 640
    height: int = 480
    jpeg_quality: int = 85
    warmup_frames: int = 5


class Camera:
    """
    USB camera capture for vision-language model input.

    Captures single frames on demand and returns them as base64-encoded
    JPEG strings (the format Ollama expects for image inputs).

    Usage:
        with Camera() as cam:
            frame_b64 = cam.capture_base64()
            if frame_b64:
                # Pass to VLM via Ollama
                response = ollama.chat(messages=[{
                    "role": "user",
                    "content": "What do you see?",
                    "images": [frame_b64],
                }])
    """

    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self._cap = None

    @property
    def is_open(self) -> bool:
        """Check if camera is currently open."""
        return self._cap is not None and self._cap.isOpened()

    def open(self) -> bool:
        """
        Open the camera device.

        Returns:
            True if camera opened successfully, False otherwise.
        """
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV not installed. Install with: pip install opencv-python-headless")
            return False

        self._cv2 = cv2

        try:
            self._cap = cv2.VideoCapture(self.config.device)
            if not self._cap.isOpened():
                logger.error("Failed to open camera device %s", self.config.device)
                self._cap = None
                return False

            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)

            # Discard warmup frames (auto-exposure settling)
            for _ in range(self.config.warmup_frames):
                self._cap.read()

            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info("Camera ready: device=%s (%dx%d)", self.config.device, actual_w, actual_h)
            return True

        except Exception as e:
            logger.error("Camera error: %s", e)
            self._cap = None
            return False

    def capture_frame(self, flush: bool = True) -> Optional[np.ndarray]:
        """
        Capture a single raw BGR frame from the camera.

        Args:
            flush: If True, flush stale V4L2 buffer frames first (needed for
                   infrequent captures like VLM queries). Set False for
                   high-frequency reads (e.g. preview at 10fps) where the
                   latest buffered frame is acceptable.

        Returns:
            numpy BGR array, or None if capture failed.
        """
        if not self.is_open:
            return None

        if flush:
            # Flush stale buffered frames from the V4L2 driver
            for _ in range(3):
                self._cap.grab()

        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None

        return frame

    def capture_base64(self) -> Optional[str]:
        """
        Capture a single frame and return as base64-encoded JPEG.

        Returns:
            Base64 string of JPEG image, or None if capture failed.
        """
        frame = self.capture_frame()
        if frame is None:
            return None

        import base64

        encode_params = [self._cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
        success, jpeg_buf = self._cv2.imencode(".jpg", frame, encode_params)
        if not success:
            return None

        return base64.b64encode(jpeg_buf.tobytes()).decode("utf-8")

    def close(self) -> None:
        """Release the camera device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


@dataclass
class WatchCondition:
    """A condition for the vision monitor to watch for."""

    description: str  # e.g., "an apple"
    prompt: str  # e.g., "Do you see an apple in this image? Answer only YES or NO."
    announce_template: str  # e.g., "I can see an apple now!"
    invert: bool = False  # If True, trigger when VLM says NO (absence detection)


class VisionMonitor:
    """
    Background monitor that polls the camera and checks a VLM condition.

    One-shot: stops after detection. User can re-issue the watch command.

    Usage:
        monitor = VisionMonitor(
            camera=camera,
            camera_lock=lock,
            check_fn=llm.check_condition,
            on_detected=callback,
            can_speak_fn=lambda: state == IDLE,
        )
        monitor.start_watching(condition)
        # ... later ...
        monitor.stop_watching()
    """

    def __init__(
        self,
        camera: "Camera",
        camera_lock: threading.Lock,
        check_fn: Callable[[str, str], bool],
        on_detected: Callable[["WatchCondition", str], None],
        can_speak_fn: Callable[[], bool],
        poll_interval: float = 10.0,
        confidence_threshold: int = 2,
        vote_window: int = 3,
    ):
        """
        Args:
            camera: Camera instance for frame capture.
            camera_lock: Lock shared with main processing thread.
            check_fn: VLM binary check function (prompt, image_b64) -> bool.
            on_detected: Callback when condition is detected (condition, frame_b64).
            can_speak_fn: Returns True when assistant is idle and can speak.
            poll_interval: Seconds between checks.
            confidence_threshold: Minimum positive votes needed to trigger detection.
            vote_window: Number of recent checks to keep in the sliding window.
        """
        self._camera = camera
        self._camera_lock = camera_lock
        self._check_fn = check_fn
        self._on_detected = on_detected
        self._can_speak_fn = can_speak_fn
        self._poll_interval = poll_interval
        self._confidence_threshold = confidence_threshold
        self._vote_window = vote_window

        self._condition: Optional[WatchCondition] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def is_watching(self) -> bool:
        """True if monitor is actively watching."""
        return self._thread is not None and self._thread.is_alive()

    def start_watching(self, condition: WatchCondition) -> None:
        """Start watching for a condition. Stops any existing watch first."""
        self.stop_watching()

        self._condition = condition
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="vision-monitor"
        )
        self._thread.start()
        logger.info("VisionMonitor: watching for '%s'", condition.description)

    def stop_watching(self) -> None:
        """Stop the monitor thread."""
        if self._thread is not None and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=5.0)
            logger.info("VisionMonitor: stopped")
        self._thread = None
        self._condition = None

    def _monitor_loop(self) -> None:
        """Main monitor loop: capture frame, check condition with confidence voting."""
        condition = self._condition
        if condition is None:
            return

        votes: list[bool] = []
        prev_frame_gray = None

        while not self._stop_event.is_set():
            # Capture frame under lock (single capture, encode separately)
            frame_b64 = None
            frame_raw = None
            with self._camera_lock:
                if self._camera.is_open:
                    frame_raw = self._camera.capture_frame()

            if frame_raw is not None:
                try:
                    import cv2
                    import base64 as b64mod
                    ok, jpeg_buf = cv2.imencode(".jpg", frame_raw, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ok:
                        frame_b64 = b64mod.b64encode(jpeg_buf.tobytes()).decode("utf-8")
                except Exception:
                    pass

            if frame_b64 is None or frame_raw is None:
                # Camera not available, wait and retry
                self._stop_event.wait(self._poll_interval)
                continue

            # Frame-diff gating: skip VLM if scene hasn't changed
            try:
                import cv2
                gray = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (160, 120))  # small for fast comparison
                if prev_frame_gray is not None:
                    diff = cv2.absdiff(prev_frame_gray, gray)
                    change_pct = float(np.count_nonzero(diff > 25)) / diff.size
                    if change_pct < 0.05:  # <5% pixels changed
                        logger.debug("VisionMonitor: scene unchanged (%.1f%%), skipping VLM", change_pct * 100)
                        prev_frame_gray = gray
                        self._stop_event.wait(self._poll_interval)
                        continue
                prev_frame_gray = gray
            except Exception:
                pass  # If frame diff fails, proceed with VLM check anyway

            # VLM inference (outside lock — this takes ~1-2s)
            try:
                detected = self._check_fn(condition.prompt, frame_b64)
            except Exception as e:
                logger.error("VisionMonitor: check error: %s", e)
                detected = False

            # Sliding window confidence voting
            votes.append(detected)
            if len(votes) > self._vote_window:
                votes = votes[-self._vote_window:]

            positive = sum(votes)
            logger.debug(
                "VisionMonitor: confidence %d/%d (need %d/%d)",
                positive, len(votes), self._confidence_threshold, self._vote_window,
            )

            if positive >= self._confidence_threshold and self._can_speak_fn():
                # Condition met with sufficient confidence — announce and stop
                self._on_detected(condition, frame_b64)
                return  # One-shot: exit loop after detection

            # Sleep (interruptible)
            self._stop_event.wait(self._poll_interval)


class VisionPreview:
    """
    Live camera preview with text overlay showing assistant state.

    Runs as a daemon thread, grabbing frames from the camera and rendering
    overlay text (state, VLM responses, watch status). Supports:
    - OpenCV window (show_window=True) — requires opencv-python (not headless)
    - MJPEG browser stream (stream_port>0) — accessible at http://<host>:<port>/

    Thread safety:
    - Camera access is protected by camera_lock (shared with processing threads).
    - Overlay text is protected by an internal lock.
    - MJPEG frame is swapped atomically (reference assignment).
    """

    def __init__(
        self,
        camera: Optional["Camera"],
        camera_lock: threading.Lock,
        fps: int = 10,
        show_window: bool = True,
        stream_port: int = 0,
    ):
        self._camera = camera
        self._camera_lock = camera_lock
        self._fps = fps
        self._show_window = show_window
        self._stream_port = stream_port

        self._overlay_lines: list[str] = []
        self._overlay_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._mjpeg_server = None
        self._jpeg_encode = None  # Initialized in start() if streaming

        # Shared MJPEG frame (atomic reference swap — no lock needed)
        self._latest_jpeg: Optional[bytes] = None

        # Live transcript for browser page (capped at 100 entries)
        self._transcript: list[dict] = []
        self._transcript_id = 0

        # Browser audio state (WebSocket bidirectional PCM)
        self._browser_ws = None  # active aiohttp WebSocketResponse
        self._audio_callback: Optional[Callable] = None  # core.py _on_audio_chunk
        self._tts_audio_queue: queue.Queue = queue.Queue()
        self._aio_loop: Optional[asyncio.AbstractEventLoop] = None
        self._cert_pem: Optional[bytes] = None  # self-signed cert for /cert download
        self._clear_callback: Optional[Callable] = None  # core.py clear_conversation
        self._browser_frame_jpeg: Optional[bytes] = None  # latest JPEG from browser webcam

    def add_transcript(self, role: str, text: str) -> None:
        """Append a transcript entry (thread-safe, used from core.py)."""
        self._transcript_id += 1
        entry = {"id": self._transcript_id, "type": "msg", "role": role,
                 "text": text, "ts": time.strftime("%H:%M:%S")}
        self._transcript.append(entry)
        if len(self._transcript) > 100:
            self._transcript = self._transcript[-100:]

    def add_transcript_stream(self, text: str) -> None:
        """Append/update a streaming assistant message (shows words appearing live)."""
        self._transcript_id += 1
        entry = {"id": self._transcript_id, "type": "stream",
                 "text": text, "ts": time.strftime("%H:%M:%S")}
        self._transcript.append(entry)
        if len(self._transcript) > 100:
            self._transcript = self._transcript[-100:]

    def add_timing(self, stage: str, ms: float) -> None:
        """Append a timing event (stt/llm/tts) for the pipeline display."""
        self._transcript_id += 1
        entry = {"id": self._transcript_id, "type": "timing",
                 "stage": stage, "ms": round(ms)}
        self._transcript.append(entry)
        if len(self._transcript) > 100:
            self._transcript = self._transcript[-100:]

    def add_debug(self, stage: str, data: str) -> None:
        """Append a debug event for the pipeline debug panel.

        Stages: 'stt', 'intent', 'llm_prompt', 'llm_raw', 'tool_call',
                'tool_result', 'vision_fallback', 'summary', 'final'.
        """
        self._transcript_id += 1
        entry = {"id": self._transcript_id, "type": "debug",
                 "stage": stage, "data": data, "ts": time.strftime("%H:%M:%S")}
        self._transcript.append(entry)
        if len(self._transcript) > 100:
            self._transcript = self._transcript[-100:]

    # ── Browser audio API ──

    def set_audio_callback(self, fn: Callable[[np.ndarray], None]) -> None:
        """Set callback for browser audio chunks (called from core.py)."""
        self._audio_callback = fn

    @property
    def has_browser_client(self) -> bool:
        """True when a browser WebSocket is connected for audio."""
        return self._browser_ws is not None

    def queue_tts_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        """Queue TTS audio for delivery to browser. Called from core.py speech thread."""
        if not self.has_browser_client:
            logger.debug("queue_tts_audio: no browser client, discarding %d samples", len(audio))
            return
        if audio.dtype != np.int16:
            if audio.dtype in (np.float32, np.float64):
                audio = (audio * 32767).astype(np.int16)
            else:
                audio = audio.astype(np.int16)
        # Pack: 4 bytes sample_rate (little-endian uint32) + raw int16 PCM
        header = struct.pack("<I", sample_rate)
        payload = header + audio.tobytes()
        self._tts_audio_queue.put(payload)
        logger.debug("queue_tts_audio: queued %d samples (%d bytes), qsize=%d",
                      len(audio), len(payload), self._tts_audio_queue.qsize())

    def set_overlay(self, lines: list[str]) -> None:
        """Update the overlay text lines (thread-safe)."""
        with self._overlay_lock:
            self._overlay_lines = list(lines)

    def start(self) -> None:
        """Start the preview thread (and aiohttp server if configured)."""
        if self._thread is not None:
            return

        if self._stream_port > 0:
            if self._camera is not None:
                self._jpeg_encode = self._init_jpeg_encoder()
            self._start_aiohttp_server()

        if self._camera is not None:
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._preview_loop, daemon=True, name="vision-preview"
            )
            self._thread.start()

        logger.info("VisionPreview: started (window=%s, stream_port=%d, camera=%s)",
                     self._show_window, self._stream_port, self._camera is not None)

    def stop(self) -> None:
        """Stop the preview thread and aiohttp server."""
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

        # Shut down aiohttp event loop
        if self._aio_loop is not None and self._aio_loop.is_running():
            self._aio_loop.call_soon_threadsafe(self._aio_loop.stop)
            self._aio_loop = None

        if self._show_window:
            try:
                import cv2
                cv2.destroyWindow("Vision")
            except Exception:
                pass

        logger.info("VisionPreview: stopped")

    def _preview_loop(self) -> None:
        """Main loop: grab frame, draw overlay, display/stream."""
        import cv2

        frame_interval = 1.0 / self._fps

        while not self._stop_event.is_set():
            loop_start = time.monotonic()

            # Grab raw frame under camera lock (no V4L2 flush — preview
            # reads at 10fps so the latest buffered frame is fine)
            frame = None
            with self._camera_lock:
                if self._camera.is_open:
                    frame = self._camera.capture_frame(flush=False)

            if frame is None:
                self._stop_event.wait(frame_interval)
                continue

            # Draw overlay
            annotated = self._draw_overlay(frame, cv2)

            # OpenCV window
            if self._show_window:
                cv2.imshow("Vision", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            # MJPEG stream: encode and swap reference
            if self._jpeg_encode is not None:
                self._latest_jpeg = self._jpeg_encode(annotated)

            # Maintain target FPS
            elapsed = time.monotonic() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)

    def _draw_overlay(self, frame: np.ndarray, cv2) -> np.ndarray:
        """Draw semi-transparent bar with overlay text at the bottom of the frame."""
        with self._overlay_lock:
            lines = list(self._overlay_lines)

        if not lines:
            return frame

        annotated = frame.copy()
        h, w = annotated.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        line_height = 22
        padding = 8

        bar_height = len(lines) * line_height + padding * 2
        bar_top = h - bar_height

        # Darken only the bar region in-place (no second full-frame copy)
        roi = annotated[bar_top:h, :]
        roi[:] = (roi * 0.4).astype(np.uint8)

        # Render each line
        for i, line in enumerate(lines):
            y = bar_top + padding + (i + 1) * line_height - 4
            cv2.putText(annotated, line, (10, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return annotated

    @staticmethod
    def _init_jpeg_encoder(quality: int = 70):
        """Initialize the fastest available JPEG encoder.

        Tries in order:
        1. nvjpeg (CUDA hardware encoder) — near-zero CPU cost
        2. TurboJPEG (SIMD-optimized libjpeg-turbo)
        3. cv2.imencode (fallback, also uses libjpeg-turbo internally)

        Returns a callable: (np.ndarray) -> bytes
        """
        # 1. Try nvjpeg (CUDA hardware encoder)
        try:
            from nvjpeg import NvJpeg
            nj = NvJpeg()
            test = np.zeros((2, 2, 3), dtype=np.uint8)
            nj.encode(test, quality)
            logger.info("VisionPreview: using NVJPEG hardware encoder")
            return lambda frame: nj.encode(frame, quality)
        except Exception:
            pass

        # 2. Try TurboJPEG
        try:
            from turbojpeg import TurboJPEG
            tj = TurboJPEG()
            test = np.zeros((2, 2, 3), dtype=np.uint8)
            tj.encode(test, quality=quality)
            logger.info("VisionPreview: using TurboJPEG encoder")
            return lambda frame: tj.encode(frame, quality=quality)
        except Exception:
            pass

        # 3. Fallback to cv2.imencode
        import cv2
        logger.info("VisionPreview: using cv2 JPEG encoder")
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]

        def _cv2_encode(frame: np.ndarray) -> bytes:
            _, buf = cv2.imencode(".jpg", frame, params)
            return buf.tobytes()

        return _cv2_encode

    def _start_aiohttp_server(self) -> None:
        """Start an aiohttp server in a daemon thread (HTTP + WebSocket)."""
        import aiohttp
        from aiohttp import web

        preview = self  # closure reference

        async def handle_root(request):
            return web.Response(text=_TRANSCRIPT_HTML, content_type="text/html")

        async def handle_video(request):
            resp = web.StreamResponse()
            resp.content_type = "multipart/x-mixed-replace; boundary=frame"
            await resp.prepare(request)
            try:
                while not preview._stop_event.is_set():
                    jpeg = preview._latest_jpeg
                    if jpeg is None:
                        await asyncio.sleep(0.1)
                        continue
                    await resp.write(
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                        + jpeg
                        + b"\r\n"
                    )
                    await asyncio.sleep(1.0 / preview._fps)
            except (ConnectionResetError, ConnectionError):
                pass
            return resp

        async def handle_events(request):
            resp = web.StreamResponse()
            resp.content_type = "text/event-stream"
            resp.headers["Cache-Control"] = "no-cache"
            resp.headers["Connection"] = "keep-alive"
            await resp.prepare(request)
            last_id = 0
            try:
                while not preview._stop_event.is_set():
                    entries = preview._transcript
                    new = [e for e in entries if e["id"] > last_id]
                    for e in new:
                        data = _json.dumps(e)
                        await resp.write(f"data: {data}\n\n".encode())
                        last_id = e["id"]
                    await asyncio.sleep(0.3)
            except (ConnectionResetError, ConnectionError):
                pass
            return resp

        async def handle_ws(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            preview._browser_ws = ws
            logger.info("Browser audio connected from %s", request.remote)

            # Auto-clear on new connection — each browser session starts fresh.
            # This prevents prior conversation history from leaking between
            # sessions (security) and ensures a clean state.
            preview._transcript.clear()
            preview._transcript_id = 0
            if preview._clear_callback:
                preview._clear_callback()
            logger.info("Session auto-cleared for new browser connection")

            # Send config
            await ws.send_json({
                "type": "config",
                "output_sample_rate": 22050,
                "input_sample_rate": 16000,
            })

            async def recv_loop():
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        pcm = np.frombuffer(msg.data, dtype=np.int16)
                        audio_f32 = pcm.astype(np.float32) / 32768.0
                        if preview._audio_callback:
                            preview._audio_callback(audio_f32)
                    elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
                        break

            async def send_loop():
                loop = asyncio.get_event_loop()
                while not ws.closed:
                    try:
                        data = await loop.run_in_executor(
                            None, lambda: preview._tts_audio_queue.get(timeout=0.05)
                        )
                        await ws.send_bytes(data)
                    except queue.Empty:
                        pass
                    except Exception as exc:
                        # Catch ALL exceptions (not just Connection*) to prevent
                        # silent disconnection that makes has_browser_client False
                        # and routes all subsequent TTS to non-existent local speaker.
                        logger.warning("send_loop error (will retry): %s", exc)
                        await asyncio.sleep(0.1)

            try:
                await asyncio.gather(recv_loop(), send_loop())
            finally:
                preview._browser_ws = None
                # Drain any leftover TTS audio
                while not preview._tts_audio_queue.empty():
                    try:
                        preview._tts_audio_queue.get_nowait()
                    except queue.Empty:
                        break
                logger.info("Browser audio disconnected")

            return ws

        def _get_local_ip_sans(x509_mod, ipaddress_mod):
            """Discover all non-loopback IPv4 addresses for TLS SAN."""
            import socket
            sans = []
            try:
                for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                    addr = info[4][0]
                    if addr.startswith("127."):
                        continue
                    try:
                        sans.append(x509_mod.IPAddress(ipaddress_mod.IPv4Address(addr)))
                    except ValueError:
                        pass
            except Exception:
                pass
            # Fallback: connect to external to find primary IP
            if not sans:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.connect(("8.8.8.8", 80))
                    addr = s.getsockname()[0]
                    s.close()
                    sans.append(x509_mod.IPAddress(ipaddress_mod.IPv4Address(addr)))
                except Exception:
                    pass
            return sans

        def _make_ssl_context():
            """Create a self-signed TLS cert for getUserMedia secure context."""
            import ssl
            import tempfile
            try:
                from cryptography import x509
                from cryptography.x509.oid import NameOID
                from cryptography.hazmat.primitives import hashes, serialization
                from cryptography.hazmat.primitives.asymmetric import rsa
                import datetime
                import ipaddress

                key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
                subject = issuer = x509.Name([
                    x509.NameAttribute(NameOID.COMMON_NAME, "jetson-assistant"),
                ])
                cert = (
                    x509.CertificateBuilder()
                    .subject_name(subject)
                    .issuer_name(issuer)
                    .public_key(key.public_key())
                    .serial_number(x509.random_serial_number())
                    .not_valid_before(datetime.datetime.utcnow())
                    .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
                    .add_extension(
                        x509.SubjectAlternativeName(
                            [x509.DNSName("localhost"),
                             x509.IPAddress(ipaddress.IPv4Address("127.0.0.1"))]
                            + _get_local_ip_sans(x509, ipaddress)
                        ),
                        critical=False,
                    )
                    .sign(key, hashes.SHA256())
                )

                cert_pem = cert.public_bytes(serialization.Encoding.PEM)
                key_pem = key.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.TraditionalOpenSSL,
                    serialization.NoEncryption(),
                )

                # Store cert PEM so /cert endpoint can serve it for iOS install
                preview._cert_pem = cert_pem

                cert_file = tempfile.NamedTemporaryFile(suffix=".pem", delete=False)
                cert_file.write(cert_pem)
                cert_file.close()

                key_file = tempfile.NamedTemporaryFile(suffix=".pem", delete=False)
                key_file.write(key_pem)
                key_file.close()

                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ctx.load_cert_chain(cert_file.name, key_file.name)
                logger.info("VisionPreview: TLS enabled (self-signed cert for getUserMedia)")
                return ctx
            except ImportError:
                logger.warning(
                    "VisionPreview: 'cryptography' not installed — serving HTTP only. "
                    "Browser mic will NOT work (getUserMedia requires HTTPS). "
                    "Install with: pip install cryptography"
                )
                return None

        async def handle_frame(request):
            """Accept a JPEG frame from the browser webcam (POST /frame)."""
            body = await request.read()
            if body:
                preview._latest_jpeg = body
                preview._browser_frame_jpeg = body
            return web.Response(text="ok")

        async def handle_clear(request):
            """Clear conversation history and transcript."""
            preview._transcript.clear()
            preview._transcript_id = 0
            if preview._clear_callback:
                preview._clear_callback()
            logger.info("Conversation and transcript cleared via /clear")
            return web.json_response({"status": "cleared"})

        async def handle_cert(request):
            """Serve the self-signed TLS certificate for iOS/mobile install."""
            if not hasattr(preview, '_cert_pem') or preview._cert_pem is None:
                return web.Response(text="No certificate available", status=404)
            return web.Response(
                body=preview._cert_pem,
                content_type="application/x-pem-file",
                headers={
                    "Content-Disposition": "attachment; filename=jetson-assistant.pem",
                },
            )

        async def _run_server():
            app = web.Application()
            app.router.add_get("/", handle_root)
            app.router.add_get("/video", handle_video)
            app.router.add_get("/events", handle_events)
            app.router.add_get("/ws", handle_ws)
            app.router.add_get("/cert", handle_cert)
            app.router.add_post("/clear", handle_clear)
            app.router.add_post("/frame", handle_frame)

            ssl_ctx = _make_ssl_context()

            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "0.0.0.0", preview._stream_port, ssl_context=ssl_ctx)
            await site.start()
            proto = "https" if ssl_ctx else "http"
            logger.info("VisionPreview: aiohttp server on %s://0.0.0.0:%d", proto, preview._stream_port)

            # Run until the stop event is set
            while not preview._stop_event.is_set():
                await asyncio.sleep(0.5)
            await runner.cleanup()

        def run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            preview._aio_loop = loop
            loop.run_until_complete(_run_server())

        thread = threading.Thread(target=run, daemon=True, name="aiohttp-server")
        thread.start()
