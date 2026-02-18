"""PersonaPlex 7B full-duplex speech-to-speech backend for jetson-assistant."""

import asyncio
import json as _json
import logging
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Transcript HTML page (served at /transcript) ──
_PERSONAPLEX_TRANSCRIPT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PersonaPlex Transcript</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0f0f0f; color: #e0e0e0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; height: 100vh; display: flex; }
  .camera { width: 45%%; background: #000; display: flex; align-items: center; justify-content: center; position: relative; }
  .camera img { width: 100%%; height: 100%%; object-fit: contain; }
  .camera-label { position: absolute; top: 16px; left: 16px; font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1.5px; }
  .camera-off { color: #444; font-size: 14px; }
  .no-camera { display: none; }
  .full-width { width: 100%%; }
  .chat-panel { width: 55%%; display: flex; flex-direction: column; border-left: 1px solid #1a1a1a; }
  .chat-header { padding: 20px 24px; border-bottom: 1px solid #1a1a1a; position: relative; }
  .chat-header h1 { font-size: 16px; font-weight: 600; color: #fff; }
  .chat-header p { font-size: 12px; color: #666; margin-top: 4px; }
  .chat-messages { flex: 1; overflow-y: auto; padding: 20px 24px; display: flex; flex-direction: column; gap: 12px; }
  .msg { max-width: 85%%; padding: 10px 14px; border-radius: 12px; font-size: 14px; line-height: 1.5; animation: fadeIn 0.2s ease; }
  .msg .ts { font-size: 10px; color: #555; margin-top: 4px; }
  .msg-assistant { align-self: flex-start; background: #1e1e1e; color: #e0e0e0; border-bottom-left-radius: 4px; }
  .msg-tool { align-self: flex-start; background: #0d2818; border: 1px solid #1a4028; color: #a3e4b8; border-radius: 8px; max-width: 90%%; padding: 10px 14px; font-size: 13px; }
  .msg-tool .tool-name { font-weight: 600; color: #22c55e; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
  .msg-tool .tool-args { color: #6ee7a0; font-size: 12px; margin-top: 2px; }
  .msg-tool .tool-result { color: #d0d0d0; margin-top: 6px; font-size: 13px; line-height: 1.4; }
  .msg-tool .tool-running { color: #eab308; font-style: italic; }
  .status { padding: 12px 24px; border-top: 1px solid #1a1a1a; display: flex; align-items: center; gap: 8px; }
  .dot { width: 8px; height: 8px; border-radius: 50%%; background: #22c55e; }
  .dot.speaking { background: #6366f1; animation: pulse 0.8s infinite; }
  .dot.listening { background: #22c55e; }
  .status span { font-size: 12px; color: #888; }
  .pipeline { padding: 8px 24px; border-top: 1px solid #1a1a1a; display: flex; gap: 16px; align-items: center; }
  .pipe-stage { font-size: 10px; color: #555; text-transform: uppercase; letter-spacing: 1px; }
  .pipe-stage .val { color: #888; font-variant-numeric: tabular-nums; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 10px; background: #1a1a2e; color: #6366f1; margin-left: 8px; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
  @keyframes pulse { 0%%,100%% { opacity: 1; } 50%% { opacity: 0.4; } }
</style>
</head>
<body>
<div class="camera %(camera_class)s" id="camera-pane">
  <span class="camera-label">Live Camera</span>
  <img src="/video" alt="camera" onerror="this.parentElement.style.display='none';document.getElementById('chat').classList.add('full-width');">
</div>
<div class="chat-panel %(chat_class)s" id="chat">
  <div class="chat-header">
    <h1>PersonaPlex</h1>
    <p>Full-duplex voice AI &mdash; Jetson Thor</p>
    <button id="clear-btn" onclick="document.getElementById('msgs').innerHTML=''" style="position:absolute;right:24px;top:20px;background:#1e1e1e;border:1px solid #333;color:#888;padding:6px 14px;border-radius:6px;font-size:12px;cursor:pointer;">Clear</button>
  </div>
  <div class="chat-messages" id="msgs"></div>
  <div class="pipeline">
    <div class="pipe-stage">PersonaPlex <span class="val" id="pv-frame">--</span></div>
    <span class="badge">FULL-DUPLEX</span>
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
const toolEls = {};
es.onmessage = (e) => {
  const d = JSON.parse(e.data);
  if (d.type === 'frame_timing') {
    document.getElementById('pv-frame').textContent = d.ms + 'ms/frame';
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
    streamText += d.text;
    streamEl.querySelector('.stream-text').textContent = streamText;
    msgs.scrollTop = msgs.scrollHeight;
    dot.className = 'dot speaking'; status.textContent = 'Speaking...';
    return;
  }
  if (d.type === 'stream_end') {
    streamEl = null; streamText = '';
    dot.className = 'dot listening'; status.textContent = 'Listening';
    return;
  }
  if (d.type === 'tool') {
    streamEl = null; streamText = '';
    const div = document.createElement('div');
    div.className = 'msg msg-tool';
    div.id = 'tool-' + d.tool_id;
    let html = '<div class="tool-name">' + d.name + '</div>';
    html += '<div class="tool-args">' + escHtml(JSON.stringify(d.args)) + '</div>';
    html += '<div class="tool-result tool-running" id="tool-result-' + d.tool_id + '">Running...</div>';
    html += '<div class="ts">' + d.ts + '</div>';
    div.innerHTML = html;
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
    toolEls[d.tool_id] = div;
    return;
  }
  if (d.type === 'tool_result') {
    const el = document.getElementById('tool-result-' + d.tool_id);
    if (el) { el.className = 'tool-result'; el.textContent = d.result; }
    return;
  }
  if (d.type === 'state') {
    if (d.state === 'speaking') { dot.className = 'dot speaking'; status.textContent = 'Speaking...'; }
    else if (d.state === 'listening') { dot.className = 'dot listening'; status.textContent = 'Listening'; }
    else { dot.className = 'dot'; status.textContent = d.state; }
    return;
  }
};
es.onerror = () => { dot.className = 'dot'; status.textContent = 'Reconnecting...'; };
function escHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
</script>
</body>
</html>"""


class PersonaplexBackend:
    """Full-duplex speech-to-speech backend using PersonaPlex (Moshi 7B).

    Supports two audio modes:
    - browser: WebSocket + Opus codec, serves web UI on personaplex_port
    - local: Direct mic/speaker via AudioInput/AudioOutput

    Tool detection: parses text token stream for [tool commands] and dispatches
    to the jetson-assistant ToolRegistry.

    Transcript: serves a live transcript page at /transcript with SSE updates,
    tool call cards, and optional MJPEG camera feed.
    """

    def __init__(self, config):
        """Initialize backend (lazy — does not load model).

        Args:
            config: AssistantConfig with personaplex_* fields set.
        """
        self.config = config
        self._loaded = False
        self._running = False

        # Model components (set by load())
        self._mimi = None
        self._lm_gen = None
        self._text_tokenizer = None

        # Callbacks (set by set_callbacks())
        self._on_state_change: Optional[Callable] = None
        self._on_audio_chunk: Optional[Callable] = None
        self._tool_registry = None

        # State
        self._state = "idle"

        # Transcript state (for /transcript page)
        self._transcript: list[dict] = []
        self._transcript_id = 0
        self._tool_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tool")

        # Camera for /video MJPEG (set up in run_browser if available)
        self._transcript_camera = None
        self._latest_jpeg: Optional[bytes] = None
        self._camera_stop = threading.Event()

    def is_loaded(self) -> bool:
        return self._loaded

    def set_callbacks(
        self,
        on_state_change: Optional[Callable] = None,
        on_audio_chunk: Optional[Callable] = None,
        tool_registry=None,
    ):
        """Set callbacks for state changes, audio chunks, and tool dispatch.

        Args:
            on_state_change: Called with (old_state, new_state) strings.
            on_audio_chunk: Called with (audio_np, sample_rate) during model speech.
            tool_registry: ToolRegistry for dispatching parsed tool commands.
        """
        self._on_state_change = on_state_change
        self._on_audio_chunk = on_audio_chunk
        self._tool_registry = tool_registry

    # ── Transcript helpers ──

    def _add_transcript(self, entry: dict) -> None:
        """Append a transcript entry (thread-safe)."""
        self._transcript_id += 1
        entry["id"] = self._transcript_id
        entry["ts"] = time.strftime("%H:%M:%S")
        self._transcript.append(entry)
        if len(self._transcript) > 200:
            self._transcript = self._transcript[-200:]

    def add_transcript_stream(self, text: str) -> None:
        """Append a streaming text token to transcript."""
        self._add_transcript({"type": "stream", "text": text})

    def add_tool_call(self, name: str, args: dict) -> int:
        """Add a tool call entry (returns tool_id for later result update)."""
        tool_id = self._transcript_id + 1
        self._add_transcript({"type": "tool", "name": name, "args": args, "tool_id": tool_id})
        return tool_id

    def add_tool_result(self, tool_id: int, result: str) -> None:
        """Update a tool call entry with its result."""
        self._add_transcript({"type": "tool_result", "tool_id": tool_id, "result": result})

    def add_frame_timing(self, ms: float) -> None:
        """Add frame timing for pipeline display."""
        self._add_transcript({"type": "frame_timing", "ms": round(ms)})

    def add_stream_end(self) -> None:
        """Signal end of a streaming assistant message."""
        self._add_transcript({"type": "stream_end"})

    # ── Camera for /video MJPEG ──

    def _start_camera_stream(self) -> None:
        """Start background camera capture for MJPEG streaming."""
        if not self.config.vision_enabled:
            return
        try:
            from jetson_assistant.assistant.vision import Camera, CameraConfig
            cam = Camera(CameraConfig(device=self.config.camera_device))
            if cam.open():
                self._transcript_camera = cam
                self._camera_stop.clear()

                def _capture_loop():
                    import cv2
                    while not self._camera_stop.is_set():
                        frame = cam.capture_frame(flush=False)
                        if frame is not None:
                            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                            self._latest_jpeg = buf.tobytes()
                        self._camera_stop.wait(0.1)  # ~10fps

                t = threading.Thread(target=_capture_loop, daemon=True, name="transcript-camera")
                t.start()
                logger.info("Transcript camera started (device=%d)", self.config.camera_device)
            else:
                logger.warning("Transcript camera device %d failed to open", self.config.camera_device)
        except Exception as e:
            logger.warning("Could not start transcript camera: %s", e)

    def _stop_camera_stream(self) -> None:
        """Stop background camera capture."""
        self._camera_stop.set()
        if self._transcript_camera is not None:
            self._transcript_camera.close()
            self._transcript_camera = None

    def load(self):
        """Load Moshi model, Mimi codec, apply FP8 quantization.

        This is expensive (~15s) and requires GPU. Call once at startup.
        """
        import torch
        from huggingface_hub import hf_hub_download

        personaplex_dir = os.path.expanduser(self.config.personaplex_dir)
        if personaplex_dir not in sys.path:
            sys.path.insert(0, personaplex_dir)

        from moshi.models import loaders, LMGen
        import sentencepiece

        device = torch.device("cuda")

        # Load Mimi (audio codec) — only one copy, skip other_mimi
        logger.info("Loading Mimi...")
        mimi_weight = hf_hub_download(self.config.personaplex_hf_repo, loaders.MIMI_NAME)
        self._mimi = loaders.get_mimi(mimi_weight, device)
        self._mimi = self._mimi.half()
        self._mimi.torch_compile_encoder_decoder = True
        self._mimi = torch.compile(self._mimi)
        logger.info("Mimi loaded (FP16 + compiled)")

        # Load text tokenizer
        tokenizer_path = hf_hub_download(self.config.personaplex_hf_repo, loaders.TEXT_TOKENIZER_NAME)
        self._text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

        # Load Moshi LM
        logger.info("Loading Moshi LM...")
        moshi_weight = hf_hub_download(self.config.personaplex_hf_repo, loaders.MOSHI_NAME)
        lm = loaders.get_moshi_lm(moshi_weight, device=device)
        lm.eval()

        # Apply FP8 quantization
        logger.info("Applying FP8 quantization...")
        from fp8_quantize import quantize_model, free_bf16_inproj
        quantize_model(lm)
        logger.info("FP8 quantization complete")

        # Resolve voice prompt directory
        voice_prompt_dir = hf_hub_download(self.config.personaplex_hf_repo, "voices.tgz")
        voices_dir = Path(voice_prompt_dir).parent / "voices"
        if not voices_dir.exists():
            import tarfile
            with tarfile.open(voice_prompt_dir, "r:gz") as tar:
                tar.extractall(path=Path(voice_prompt_dir).parent)

        # Create LMGen
        self._lm_gen = LMGen(
            lm,
            audio_silence_frame_cnt=int(0.5 * self._mimi.frame_rate),
            sample_rate=self._mimi.sample_rate,
            device=device,
            frame_rate=self._mimi.frame_rate,
        )

        # Set voice prompt
        voice_file = voices_dir / f"{self.config.personaplex_voice}.pt"
        if voice_file.exists():
            self._lm_gen.load_voice_prompt_embeddings(str(voice_file))
        else:
            voice_wav = voices_dir / f"{self.config.personaplex_voice}.wav"
            if voice_wav.exists():
                self._lm_gen.load_voice_prompt(str(voice_wav))
            else:
                logger.warning("Voice prompt %s not found, using default", self.config.personaplex_voice)

        # Set text prompt
        if self.config.personaplex_text_prompt:
            prompt_text = self.config.personaplex_text_prompt.strip()
            if not prompt_text.startswith("<system>"):
                prompt_text = f"<system> {prompt_text} <system>"
            self._lm_gen.text_prompt_tokens = self._text_tokenizer.encode(prompt_text)

        # Setup streaming
        self._mimi.streaming_forever(1)
        self._lm_gen.streaming_forever(1)

        # Warmup
        logger.info("Warming up...")
        frame_size = int(self._mimi.sample_rate / self._mimi.frame_rate)
        for _ in range(4):
            chunk = torch.zeros(1, 1, frame_size, dtype=torch.float16, device=device)
            codes = self._mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self._lm_gen.step(codes[:, :, c:c + 1])
                if tokens is not None:
                    _ = self._mimi.decode(tokens[:, 1:9])
        torch.cuda.synchronize()

        # Free bf16 copies after warmup
        free_bf16_inproj(lm)
        logger.info("PersonaPlex loaded. GPU memory: %.2f GB", torch.cuda.memory_allocated() / 1e9)

        # Pre-allocate pinned buffer for DtoH
        self._pinned_pcm = torch.empty(1920, dtype=torch.float32, pin_memory=True)
        self._frame_size = frame_size
        self._device = device
        self._loaded = True

    def _set_state(self, new_state: str):
        """Update state and fire callback."""
        if new_state != self._state:
            old = self._state
            self._state = new_state
            if self._on_state_change:
                try:
                    self._on_state_change(old, new_state)
                except Exception as e:
                    logger.error("on_state_change error: %s", e)

    def stop(self):
        """Stop the backend."""
        self._running = False
        self._stop_camera_stream()

    def run_browser(self):
        """Run with browser audio (WebSocket + Opus). Blocking.

        Serves PersonaPlex web UI on config.personaplex_port.
        Browser captures mic, streams Opus over WebSocket.
        Server runs inference, streams audio + text back.
        Also serves /transcript page with live SSE updates.
        """
        import torch
        import sphn
        import aiohttp
        from aiohttp import web
        from jetson_assistant.backends.tool_parser import ToolParser, KeywordToolDetector

        self._running = True
        lock = asyncio.Lock()

        # Start camera stream for /video endpoint
        self._start_camera_stream()
        has_camera = self._transcript_camera is not None

        backend = self  # closure reference

        async def handle_chat(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            logger.info("Browser connected from %s", request.remote)

            self._set_state("listening")

            async with lock:
                # Reset streaming state for new session
                self._mimi.reset_streaming()
                self._lm_gen.reset_streaming()

                # Opus codec
                opus_writer = sphn.OpusStreamWriter(self._mimi.sample_rate)
                opus_reader = sphn.OpusStreamReader(self._mimi.sample_rate)

                # Process voice/text system prompts through mimi
                async def is_alive():
                    return not ws.closed

                await self._lm_gen.step_system_prompts_async(self._mimi, is_alive=is_alive)
                self._mimi.reset_streaming()

                # Send handshake
                if not ws.closed:
                    await ws.send_bytes(b"\x00")

                # Tool execution helper (shared by bracket parser + keyword detector)
                loop = asyncio.get_event_loop()

                def _execute_tool(name, args):
                    """Add tool to transcript and execute async. Send result to browser UI."""
                    tool_id = backend.add_tool_call(name, args)
                    if self._tool_registry:
                        def _run_tool():
                            class _TC:
                                def __init__(self, n, a):
                                    self.name = n
                                    self.arguments = a
                            try:
                                result = self._tool_registry.execute(_TC(name, args))
                                result_str = str(result) if result else "No result"
                            except Exception as e:
                                result_str = f"Error: {e}"
                            backend.add_tool_result(tool_id, result_str)
                            logger.info("Tool %s -> %s", name, result_str[:100])
                            # Send result to browser as visual notification
                            # 0x03 prefix = tool result message
                            try:
                                import json
                                msg_data = json.dumps({"tool": name, "result": result_str[:300]})
                                msg = b"\x03" + bytes(msg_data, encoding="utf8")
                                asyncio.run_coroutine_threadsafe(
                                    ws.send_bytes(msg), loop
                                )
                            except Exception as e:
                                logger.warning("Failed to send tool result to browser: %s", e)
                        self._tool_executor.submit(_run_tool)
                    else:
                        backend.add_tool_result(tool_id, "No tool registry configured")

                # Bracket parser ([tool command] in model output)
                def on_tool_detected(tool_info):
                    _execute_tool(tool_info["name"], tool_info["args"])

                tool_parser = ToolParser(
                    on_text=lambda t: None,
                    on_tool=on_tool_detected,
                )

                # Keyword detector (monitors model text for tool-triggering patterns)
                keyword_detector = None
                if self.config.personaplex_tool_detection == "keyword":
                    registered = set()
                    if self._tool_registry:
                        registered = set(self._tool_registry._tools.keys())
                    keyword_detector = KeywordToolDetector(
                        on_tool=_execute_tool,
                        cooldown=15.0,
                        registered_tools=registered,
                    )

                close = False

                async def recv_loop():
                    nonlocal close
                    try:
                        async for message in ws:
                            if message.type == aiohttp.WSMsgType.BINARY:
                                data = message.data
                                if len(data) > 0 and data[0] == 1:  # Audio
                                    opus_reader.append_bytes(data[1:])
                            elif message.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE):
                                break
                    finally:
                        close = True

                async def opus_loop():
                    all_pcm_data = None
                    _frame_count = 0
                    _profile_interval = 50  # log every 50 frames

                    while not close:
                        await asyncio.sleep(0.001)
                        pcm = opus_reader.read_pcm()
                        if pcm.shape[-1] == 0:
                            continue
                        if all_pcm_data is None:
                            all_pcm_data = pcm
                        else:
                            all_pcm_data = np.concatenate((all_pcm_data, pcm))

                        while all_pcm_data is not None and all_pcm_data.shape[-1] >= self._frame_size:
                            _frame_count += 1
                            _do_log = (_frame_count % _profile_interval == 1)
                            _t0 = time.time()

                            chunk = all_pcm_data[:self._frame_size]
                            all_pcm_data = all_pcm_data[self._frame_size:]
                            if all_pcm_data.shape[-1] == 0:
                                all_pcm_data = None

                            chunk_t = torch.from_numpy(chunk).to(device=self._device, dtype=torch.float16)[None, None]

                            if _do_log:
                                torch.cuda.synchronize()
                                _t1 = time.time()

                            codes = self._mimi.encode(chunk_t)

                            if _do_log:
                                torch.cuda.synchronize()
                                _t2 = time.time()

                            _step_total = 0.0
                            _decode_total = 0.0
                            for c in range(codes.shape[-1]):
                                if _do_log:
                                    torch.cuda.synchronize()
                                    _ts0 = time.time()

                                tokens = self._lm_gen.step(codes[:, :, c:c + 1])

                                if _do_log:
                                    torch.cuda.synchronize()
                                    _ts1 = time.time()
                                    _step_total += (_ts1 - _ts0)

                                if tokens is None:
                                    continue

                                if _do_log:
                                    torch.cuda.synchronize()
                                    _td0 = time.time()

                                # Decode audio
                                main_pcm = self._mimi.decode(tokens[:, 1:9])

                                if _do_log:
                                    torch.cuda.synchronize()
                                    _td1 = time.time()
                                    _decode_total += (_td1 - _td0)

                                main_pcm = main_pcm.float()
                                self._pinned_pcm.copy_(main_pcm[0, 0], non_blocking=True)
                                torch.cuda.current_stream().synchronize()

                                pcm_np = self._pinned_pcm.detach().numpy()

                                # Feed audio chunk for motion/animation callbacks
                                rms = float(np.sqrt(np.mean(pcm_np ** 2)))
                                if rms > 0.001:
                                    self._set_state("speaking")
                                    if self._on_audio_chunk:
                                        try:
                                            self._on_audio_chunk(pcm_np, self._mimi.sample_rate)
                                        except Exception as e:
                                            logger.error("on_audio_chunk error: %s", e)
                                else:
                                    self._set_state("listening")

                                # Send audio back to browser
                                opus_writer.append_pcm(pcm_np)

                                # Parse text token for tools + transcript
                                text_token = tokens[0, 0, 0].item()
                                if text_token not in (0, 3):
                                    _text = self._text_tokenizer.id_to_piece(text_token)
                                    _text = _text.replace("\u2581", " ")
                                    tool_parser.feed(_text)
                                    if keyword_detector:
                                        keyword_detector.feed(_text)
                                    # Send text to browser (PersonaPlex audio UI)
                                    msg = b"\x02" + bytes(_text, encoding="utf8")
                                    await ws.send_bytes(msg)
                                    # Stream to transcript page
                                    backend.add_transcript_stream(_text)

                            _t3 = time.time()
                            _total_ms = (_t3 - _t0) * 1000
                            if _do_log:
                                _prep_ms = (_t1 - _t0) * 1000
                                _encode_ms = (_t2 - _t1) * 1000
                                _step_ms = _step_total * 1000
                                _dec_ms = _decode_total * 1000
                                _other_ms = _total_ms - _prep_ms - _encode_ms - _step_ms - _dec_ms
                                print(f"PROFILE frame={_frame_count} total={_total_ms:.1f}ms | prep={_prep_ms:.1f} encode={_encode_ms:.1f} lm_step={_step_ms:.1f} decode={_dec_ms:.1f} other={_other_ms:.1f}", flush=True)
                                backend.add_frame_timing(_total_ms)

                    tool_parser.flush()
                    if keyword_detector:
                        keyword_detector.flush()
                    backend.add_stream_end()

                async def send_loop():
                    while not close:
                        await asyncio.sleep(0.001)
                        msg = opus_writer.read_bytes()
                        if len(msg) > 0:
                            await ws.send_bytes(b"\x01" + msg)

                # Run recv + opus + send concurrently
                tasks = [
                    asyncio.create_task(recv_loop()),
                    asyncio.create_task(opus_loop()),
                    asyncio.create_task(send_loop()),
                ]
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                await ws.close()

            self._set_state("idle")
            logger.info("Browser disconnected")
            return ws

        # ── Transcript page routes ──

        async def handle_transcript(_):
            """Serve the transcript HTML page."""
            camera_class = "" if has_camera else "no-camera"
            chat_class = "full-width" if not has_camera else ""
            html = _PERSONAPLEX_TRANSCRIPT_HTML % {
                "camera_class": camera_class,
                "chat_class": chat_class,
            }
            return web.Response(text=html, content_type="text/html")

        async def handle_events(request):
            """SSE endpoint for live transcript updates."""
            response = web.StreamResponse()
            response.headers["Content-Type"] = "text/event-stream"
            response.headers["Cache-Control"] = "no-cache"
            response.headers["Connection"] = "keep-alive"
            await response.prepare(request)

            last_id = 0
            try:
                while True:
                    entries = self._transcript
                    new = [e for e in entries if e["id"] > last_id]
                    for e in new:
                        data = _json.dumps(e)
                        await response.write(f"data: {data}\n\n".encode())
                        last_id = e["id"]
                    await asyncio.sleep(0.3)
            except (asyncio.CancelledError, ConnectionResetError):
                pass
            return response

        async def handle_video(request):
            """MJPEG camera stream for transcript page."""
            if not has_camera:
                return web.Response(status=404, text="No camera configured")

            response = web.StreamResponse()
            response.headers["Content-Type"] = "multipart/x-mixed-replace; boundary=frame"
            await response.prepare(request)

            try:
                while True:
                    jpeg = self._latest_jpeg
                    if jpeg is not None:
                        await response.write(
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                            + jpeg
                            + b"\r\n"
                        )
                    await asyncio.sleep(0.1)  # ~10fps
            except (asyncio.CancelledError, ConnectionResetError):
                pass
            return response

        # ── Build aiohttp app ──

        app = web.Application()
        app.router.add_get("/api/chat", handle_chat)
        app.router.add_get("/transcript", handle_transcript)
        app.router.add_get("/events", handle_events)
        app.router.add_get("/video", handle_video)

        # Serve PersonaPlex web UI — prefer local (has patched audio buffer)
        personaplex_dir = os.path.expanduser(self.config.personaplex_dir)
        local_dist = Path(personaplex_dir) / "client" / "dist"
        if local_dist.exists():
            dist_dir = local_dist
            logger.info("Serving local web UI from %s", dist_dir)
        else:
            from huggingface_hub import hf_hub_download
            dist_tgz = hf_hub_download(self.config.personaplex_hf_repo, "dist.tgz")
            dist_dir = Path(dist_tgz).parent / "dist"
            if not dist_dir.exists():
                import tarfile
                with tarfile.open(dist_tgz, "r:gz") as tar:
                    tar.extractall(path=Path(dist_tgz).parent)
            logger.info("Serving HuggingFace web UI from %s", dist_dir)

        async def handle_root(_):
            return web.FileResponse(os.path.join(str(dist_dir), "index.html"))

        app.router.add_get("/", handle_root)
        app.router.add_static("/", path=str(dist_dir), follow_symlinks=True, name="static")

        # SSL
        ssl_context = None
        if self.config.personaplex_ssl_dir:
            from moshi.utils.connection import create_ssl_context
            ssl_context, _ = create_ssl_context(os.path.expanduser(self.config.personaplex_ssl_dir))

        from moshi.utils.connection import get_lan_ip
        host_ip = get_lan_ip()
        protocol = "https" if ssl_context else "http"
        logger.info("PersonaPlex Web UI: %s://%s:%d", protocol, host_ip, self.config.personaplex_port)
        logger.info("PersonaPlex Transcript: %s://%s:%d/transcript", protocol, host_ip, self.config.personaplex_port)

        web.run_app(app, port=self.config.personaplex_port, ssl_context=ssl_context)

    def run_local(self):
        """Run with local mic/speaker. Blocking.

        Uses jetson-assistant's AudioInput/AudioOutput for on-device audio.
        """
        import torch
        from jetson_assistant.assistant.audio_io import AudioInput, AudioOutput, AudioConfig
        from jetson_assistant.backends.tool_parser import ToolParser, KeywordToolDetector
        import resampy

        self._running = True

        # Reset streaming for new session
        self._mimi.reset_streaming()
        self._lm_gen.reset_streaming()

        # Process system prompts synchronously
        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            self._lm_gen.step_system_prompts_async(self._mimi, is_alive=lambda: True)
        )
        loop.close()
        self._mimi.reset_streaming()

        # Audio I/O at Mimi's sample rate (24kHz)
        audio_out = AudioOutput(sample_rate=self._mimi.sample_rate)

        # Tool execution helper
        backend = self

        def _execute_tool(name, args):
            tool_id = backend.add_tool_call(name, args)
            if self._tool_registry:
                def _run_tool():
                    class _TC:
                        def __init__(self, n, a):
                            self.name = n
                            self.arguments = a
                    try:
                        result = self._tool_registry.execute(_TC(name, args))
                        result_str = str(result) if result else "No result"
                    except Exception as e:
                        result_str = f"Error: {e}"
                    backend.add_tool_result(tool_id, result_str)
                    logger.info("Tool %s -> %s", name, result_str[:100])
                self._tool_executor.submit(_run_tool)
            else:
                backend.add_tool_result(tool_id, "No tool registry configured")

        def on_tool_detected(tool_info):
            _execute_tool(tool_info["name"], tool_info["args"])

        tool_parser = ToolParser(on_text=lambda t: None, on_tool=on_tool_detected)

        # Keyword detector
        keyword_detector = None
        if self.config.personaplex_tool_detection == "keyword":
            registered = set()
            if self._tool_registry:
                registered = set(self._tool_registry._tools.keys())
            keyword_detector = KeywordToolDetector(
                on_tool=_execute_tool,
                cooldown=15.0,
                registered_tools=registered,
            )

        # Audio buffer (mic -> model)
        audio_buffer = []
        buffer_lock = threading.Lock()

        def on_mic_chunk(chunk_int16):
            # Resample 16kHz -> 24kHz
            chunk_f32 = chunk_int16.astype(np.float32) / 32768.0
            chunk_24k = resampy.resample(chunk_f32, 16000, self._mimi.sample_rate)
            with buffer_lock:
                audio_buffer.append(chunk_24k)

        audio_in = AudioInput(
            config=AudioConfig(sample_rate=16000),
            device=self.config.audio_input_device,
        )
        audio_in.start(on_mic_chunk)

        logger.info("PersonaPlex local mode running. Speak into microphone.")

        try:
            while self._running:
                # Drain buffer
                with buffer_lock:
                    if not audio_buffer:
                        time.sleep(0.005)
                        continue
                    all_pcm = np.concatenate(audio_buffer)
                    audio_buffer.clear()

                # Process in frame-sized chunks
                offset = 0
                while offset + self._frame_size <= len(all_pcm):
                    frame = all_pcm[offset:offset + self._frame_size]
                    offset += self._frame_size

                    chunk_t = torch.from_numpy(frame).to(device=self._device, dtype=torch.float16)[None, None]
                    codes = self._mimi.encode(chunk_t)

                    for c in range(codes.shape[-1]):
                        tokens = self._lm_gen.step(codes[:, :, c:c + 1])
                        if tokens is None:
                            continue

                        main_pcm = self._mimi.decode(tokens[:, 1:9])
                        main_pcm = main_pcm.float()
                        self._pinned_pcm.copy_(main_pcm[0, 0], non_blocking=True)
                        torch.cuda.current_stream().synchronize()

                        pcm_np = self._pinned_pcm.detach().numpy().copy()

                        # Play audio
                        pcm_int16 = (pcm_np * 32768).astype(np.int16)
                        audio_out.play(pcm_int16, self._mimi.sample_rate)

                        # State + motion callbacks
                        rms = float(np.sqrt(np.mean(pcm_np ** 2)))
                        if rms > 0.001:
                            self._set_state("speaking")
                            if self._on_audio_chunk:
                                self._on_audio_chunk(pcm_np, self._mimi.sample_rate)
                        else:
                            self._set_state("listening")

                        # Text -> tool parsing + transcript
                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            _text = self._text_tokenizer.id_to_piece(text_token)
                            _text = _text.replace("\u2581", " ")
                            tool_parser.feed(_text)
                            if keyword_detector:
                                keyword_detector.feed(_text)
                            backend.add_transcript_stream(_text)

        except KeyboardInterrupt:
            logger.info("Stopping PersonaPlex local mode...")
        finally:
            audio_in.stop()
            audio_out.stop()
            tool_parser.flush()
            if keyword_detector:
                keyword_detector.flush()
            self._running = False
            self._set_state("idle")

    def run(self):
        """Run the backend in configured audio mode. Blocking."""
        if not self._loaded:
            self.load()

        if self.config.personaplex_audio == "browser":
            self.run_browser()
        elif self.config.personaplex_audio == "local":
            self.run_local()
        else:
            raise ValueError(f"Unknown personaplex_audio mode: {self.config.personaplex_audio}")
