"""
Camera vision module for voice assistant.

Captures frames from a USB camera and encodes them as base64 JPEG
for use with vision-language models (VLMs) via Ollama.

Includes a live preview overlay (OpenCV window and/or MJPEG browser stream)
showing assistant state, VLM responses, and watch status.

Requires: opencv-python-headless >= 4.8.0 (or opencv-python for --show-vision)
    pip install jetson-assistant[vision]
"""

import logging
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
    <button id="clear-btn" onclick="document.getElementById('msgs').innerHTML=''" style="position:absolute;right:24px;top:20px;background:#1e1e1e;border:1px solid #333;color:#888;padding:6px 14px;border-radius:6px;font-size:12px;cursor:pointer;">Clear</button>
  </div>
  <div class="chat-messages" id="msgs"></div>
  <div class="pipeline">
    <div class="pipe-stage" id="ps-stt"><span class="label">STT</span><span class="val" id="pv-stt">--</span></div>
    <div class="pipe-stage" id="ps-llm"><span class="label">LLM</span><span class="val" id="pv-llm">--</span></div>
    <div class="pipe-stage" id="ps-tts"><span class="label">TTS</span><span class="val" id="pv-tts">--</span></div>
    <span class="badge">STREAMING</span>
  </div>
  <div class="status"><div class="dot" id="dot"></div><span id="status">Connected</span></div>
</div>
<script>
const msgs = document.getElementById('msgs');
const dot = document.getElementById('dot');
const status = document.getElementById('status');
const es = new EventSource('/events');
let streamEl = null;
let streamText = '';
es.onmessage = (e) => {
  const d = JSON.parse(e.data);
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
        camera: "Camera",
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

    def set_overlay(self, lines: list[str]) -> None:
        """Update the overlay text lines (thread-safe)."""
        with self._overlay_lock:
            self._overlay_lines = list(lines)

    def start(self) -> None:
        """Start the preview thread (and MJPEG server if configured)."""
        if self._thread is not None:
            return

        if self._stream_port > 0:
            self._jpeg_encode = self._init_jpeg_encoder()
            self._start_mjpeg_server()

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._preview_loop, daemon=True, name="vision-preview"
        )
        self._thread.start()
        logger.info("VisionPreview: started (window=%s, stream_port=%d)", self._show_window, self._stream_port)

    def stop(self) -> None:
        """Stop the preview thread and MJPEG server."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=3.0)
        self._thread = None

        if self._mjpeg_server is not None:
            self._mjpeg_server.shutdown()
            self._mjpeg_server = None

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

    def _start_mjpeg_server(self) -> None:
        """Start a simple HTTP MJPEG streaming server in a daemon thread."""
        import http.server
        import socketserver

        preview = self  # closure reference

        class MJPEGHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/video":
                    self._stream_mjpeg()
                elif self.path == "/events":
                    self._stream_sse()
                elif self.path == "/":
                    self._serve_page()
                else:
                    self.send_error(404)

            def _serve_page(self):
                html = _TRANSCRIPT_HTML
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html.encode())

            def _stream_mjpeg(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while not preview._stop_event.is_set():
                        jpeg = preview._latest_jpeg
                        if jpeg is None:
                            time.sleep(0.1)
                            continue
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                        time.sleep(1.0 / preview._fps)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def _stream_sse(self):
                import json as _json
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                last_id = 0
                try:
                    while not preview._stop_event.is_set():
                        entries = preview._transcript
                        new = [e for e in entries if e["id"] > last_id]
                        for e in new:
                            data = _json.dumps(e)
                            self.wfile.write(f"data: {data}\n\n".encode())
                            last_id = e["id"]
                        self.wfile.flush()
                        time.sleep(0.3)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, format, *args):
                pass

        class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        server = ThreadedHTTPServer(("0.0.0.0", self._stream_port), MJPEGHandler)
        self._mjpeg_server = server

        thread = threading.Thread(target=server.serve_forever, daemon=True, name="mjpeg-server")
        thread.start()
        logger.info("VisionPreview: MJPEG server on port %d", self._stream_port)
