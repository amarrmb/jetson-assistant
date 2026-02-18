"""
Voice Assistant core - main conversation loop and state machine.

This is the heart of the assistant that ties together:
- Wake word detection
- Speech-to-Text
- LLM response generation
- Text-to-Speech
"""

import base64
import logging
import queue
import re
import time
import threading
from concurrent.futures import Future, ThreadPoolExecutor

logger = logging.getLogger(__name__)
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from jetson_assistant.assistant.audio_io import (
    AudioConfig,
    AudioInput,
    AudioOutput,
    ChimeSounds,
    VoiceActivityDetector,
)
from jetson_assistant.assistant.llm import LLMBackend, create_llm
from jetson_assistant.assistant.tts_cache import TTSCache
from jetson_assistant.assistant.wakeword import WakeWordDetector, create_wakeword_detector


class AssistantState(Enum):
    """Assistant state machine states."""

    IDLE = "idle"  # Waiting for wake word
    LISTENING = "listening"  # Recording user speech
    PROCESSING = "processing"  # STT -> LLM -> TTS
    SPEAKING = "speaking"  # Playing response
    ERROR = "error"  # Error state


@dataclass
class AssistantConfig:
    """Configuration for voice assistant."""

    # Server mode (recommended - much faster!)
    use_server: bool = False  # If True, connect to running jetson-assistant server
    server_host: str = "localhost"
    server_port: int = 8080
    use_server_llm: bool = True  # If True (and use_server=True), use server's LLM/RAG

    # Wake word settings
    wake_word: str = "hey_jarvis"
    wake_word_backend: str = "openwakeword"
    wake_word_threshold: float = 0.5

    # LLM settings (used when use_server_llm=False)
    llm_backend: str = "ollama"
    llm_model: str = "llama3.2:3b"
    llm_host: str = "http://localhost:11434"
    system_prompt: Optional[str] = None
    rag_collection: Optional[str] = None  # Enable RAG with collection name (e.g., "dota2")

    # TTS settings
    tts_backend: str = "piper"
    tts_voice: str = "en_US-amy-medium"
    tts_language: str = "en"

    # STT settings
    stt_backend: str = "whisper"
    stt_model: str = "base"
    stt_host: Optional[str] = None  # STT server URL (for vllm backend, e.g. http://localhost:8002/v1)
    stt_min_confidence: float = 0.15  # Minimum confidence (0-1, from avg_logprob)
    stt_min_words: int = 2  # Minimum words to accept (filter noise)

    # Audio settings
    input_sample_rate: int = 16000
    output_sample_rate: int = 22050  # Piper uses 22050 Hz
    silence_timeout_ms: int = 1500  # End of speech detection
    max_listen_time_s: float = 10.0  # Max recording time
    audio_input_device: Optional[int] = None  # Input device index (None for system default)

    # Behavior
    play_chimes: bool = True
    verbose: bool = False
    conversation_history: int = 5  # Number of turns to remember
    stream_llm: bool = True  # Enable LLM streaming for pipelined TTS

    # Vision
    vision_enabled: bool = False
    camera_device: int = 0
    camera_width: int = 640
    camera_height: int = 480
    vision_system_prompt: Optional[str] = None
    watch_poll_interval: float = 5.0  # Vision monitor polling interval (seconds)
    watch_confidence_threshold: int = 2  # Minimum positive votes to trigger detection
    watch_vote_window: int = 3  # Sliding window size for confidence voting
    show_vision: bool = False  # OpenCV preview window
    stream_vision_port: int = 0  # MJPEG stream port (0 = disabled)

    # Multi-camera
    camera_config_path: str = "~/.assistant_cameras.json"
    watch_cooldown: float = 60.0  # Seconds between repeated alerts on same camera

    # Remote camera (Aether SFU WebRTC stream via UDP)
    remote_camera_port: int = 0  # UDP port for remote camera (0 = disabled)

    # Aether Hub
    aether_hub_host: Optional[str] = None  # Hub hostname (None = disabled)
    aether_hub_port: int = 8000
    aether_pin: str = ""

    # Knowledge base (RAG tool — LLM decides when to search)
    knowledge_collection: Optional[str] = None  # Collection name (None = disabled)

    # External tool plugins (list of importable module paths)
    external_tools: Optional[list] = None  # e.g. ["reachy_tools", "my_pkg.tools"]

    # PersonaPlex mode
    mode: str = "pipeline"  # "pipeline" (existing STT→LLM→TTS) or "personaplex"
    personaplex_audio: str = "browser"  # "browser" (WebSocket) or "local" (pyaudio)
    personaplex_port: int = 8998
    personaplex_ssl_dir: Optional[str] = None  # Path to dir with cert.pem + key.pem
    personaplex_voice: str = "NATF1"  # PersonaPlex voice persona filename
    personaplex_text_prompt: Optional[str] = None  # System prompt for PersonaPlex
    personaplex_tool_detection: str = "prompt"  # "prompt", "keyword", or "llm"
    personaplex_dir: str = "~/personaplex"  # Path to fp8_quantize.py etc.
    personaplex_cpu_cores: str = "4-13"  # CPU affinity for inference
    personaplex_hf_repo: str = "nvidia/personaplex-7b-v1"

    # Callbacks
    on_wake: Optional[Callable[[], None]] = None
    on_listen_start: Optional[Callable[[], None]] = None
    on_listen_end: Optional[Callable[[str], None]] = None
    on_response: Optional[Callable[[str], None]] = None
    on_error: Optional[Callable[[Exception], None]] = None

    @classmethod
    def from_yaml(cls, path: str) -> dict:
        """Load config values from a YAML file.

        Returns a dict of config keys → values (not an AssistantConfig instance).
        Caller is responsible for merging with CLI overrides before constructing.
        Only keys that correspond to AssistantConfig fields are returned;
        unknown keys are silently ignored.
        """
        import yaml

        yaml_path = Path(path).expanduser()
        with open(yaml_path) as f:
            raw = yaml.safe_load(f) or {}

        valid_keys = {f.name for f in fields(cls) if f.name not in (
            "on_wake", "on_listen_start", "on_listen_end", "on_response", "on_error",
        )}
        return {k: v for k, v in raw.items() if k in valid_keys}


class VoiceAssistant:
    """
    Voice assistant with wake word, STT, LLM, and TTS.

    Usage (Server Mode - RECOMMENDED, much faster!):
        # Terminal 1: Start server once
        jetson-assistant serve --port 8080

        # Terminal 2: Run assistant
        from jetson_assistant.assistant import VoiceAssistant, AssistantConfig

        config = AssistantConfig(use_server=True)
        assistant = VoiceAssistant(config=config)
        assistant.run()

    Usage (In-process Mode - slower, loads models each time):
        from jetson_assistant import Engine
        from jetson_assistant.assistant import VoiceAssistant, AssistantConfig

        engine = Engine()
        engine.load_tts_backend("qwen")
        engine.load_stt_backend("whisper")

        config = AssistantConfig(wake_word="hey_jarvis")
        assistant = VoiceAssistant(engine, config)
        assistant.run()
    """

    VISION_SYSTEM_PROMPT = (
        "You are a helpful voice assistant with a camera. "
        "You can see through the camera and have conversation history.\n\n"
        "RULES:\n"
        "1. Use conversation history to understand references like 'the other one', "
        "'what you said', or 'I meant'. Previous messages give you context.\n"
        "2. For general questions (date, time, who you are, facts), answer from "
        "your knowledge — do NOT say 'not in the image'.\n"
        "3. For visual questions, be specific and factual about what you see.\n"
        "4. Maximum 2 sentences. Speak naturally. No formatting or markdown."
    )

    TOOL_SYSTEM_PROMPT_HEADER = (
        "You are a helpful voice assistant with tools. "
        "You MUST use your tools when they are relevant.\n\n"
        "To call a tool, respond with ONLY a JSON object like this:\n"
        '{"tool": "tool_name", "args": {"param": "value"}}\n\n'
        "RULES:\n"
        "- If a tool can handle the request, respond with ONLY the JSON tool call.\n"
        "- Do NOT add any text before or after the JSON.\n"
        "- Call ONLY ONE tool per response. If the user wants multiple things, "
        "pick the most important one.\n"
        "- If no tool is needed, just answer normally in 1-2 sentences.\n"
        "- No markdown or formatting.\n"
    )

    # ── Intent Router ──
    # Fast LLM classification (~90ms) to separate tool commands from
    # vision queries and chat. Replaces regex-based _has_visual_intent.
    _INTENT_ROUTER_PROMPT = (
        "Classify this voice command. Answer TOOL or CHAT. When in doubt, answer TOOL.\n\n"
        "CHAT = ONLY greetings and pure knowledge questions:\n"
        "hello/hi/hey, what is X, who is X, tell me a joke, your name\n\n"
        "TOOL = EVERYTHING ELSE including:\n"
        "look/turn/move, dance, emotions, sleep/wake, see/camera/show,\n"
        "search, remember, time, any action or request\n\n"
        '"go to sleep" → TOOL\n'
        '"show me you are happy" → TOOL\n'
        '"what do you see" → TOOL\n'
        '"wake up" → TOOL\n'
        '"remember my name" → TOOL\n'
        '"hello" → CHAT\n'
        '"what is AI" → CHAT'
    )

    # ── Slow Tool Acknowledgment ──
    # Tools expected to take >200ms get a spoken "Let me check on that."
    # before execution, so the user knows the assistant is working.
    _SLOW_TOOLS = frozenset({"web_search", "look", "search", "describe_scene"})

    _VISUAL_PATTERNS = re.compile(
        r"\b(see|look|show|describe|camera|picture|image|photo|color|holding|wearing|read)\b"
        r"|\bwhat(?:'s|\s+is)\s+(this|that|here|there)\b"
        r"|\bin front\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _has_visual_intent(text: str) -> bool:
        """Regex fallback for vision gating (used when no LLM router available)."""
        return bool(VoiceAssistant._VISUAL_PATTERNS.search(text))

    def _classify_intent(self, text: str) -> str:
        """Classify user intent via fast LLM call (~90ms).

        Returns 'TOOL' or 'CHAT'. When tools are registered, all vision
        queries route to TOOL (handled by check_camera/reachy_see tools).
        Falls back to regex-based _has_visual_intent when LLM unavailable.
        """
        # Only use router when we have both LLM and tools
        if self.llm is None or not (self._tools and self._tools.definitions()):
            return "CHAT"

        result = self.llm.classify_intent(text, self._INTENT_ROUTER_PROMPT)
        if result and result in ("TOOL", "CHAT"):
            return result

        # Fallback: treat as TOOL if it looks like a command
        return "TOOL"

    def _capture_vision_async(self) -> Future:
        """Submit camera capture to background thread, returns a Future.

        The capture runs concurrently so the main thread can build LLM
        context (trim history, classify intent, etc.) in parallel with
        the ~50-100 ms camera grab.
        """

        def _grab():
            with self._camera_lock:
                if self.camera and self.camera.is_open:
                    return self.camera.capture_base64()
            return None

        return self._vision_executor.submit(_grab)

    # ── VLM Priority Queue ──
    # User requests preempt background watch threads on the shared vLLM instance.

    def _request_vlm_priority(self):
        """Signal that a user request needs VLM access."""
        self._vlm_priority.set()

    def _release_vlm_priority(self):
        """Clear priority flag after user request completes."""
        self._vlm_priority.clear()

    def _vlm_call_with_priority(self, call_fn, is_user_request=False):
        """Make a VLM call with priority handling.

        Args:
            call_fn: Callable that performs the actual VLM call and returns result.
            is_user_request: If True, set priority (preempts watch threads).
                If False (watch thread), skip if user has priority.

        Returns:
            Result of call_fn, or None if skipped (watch thread yielding).
        """
        if not is_user_request:
            # Watch thread: yield if user has priority
            if self._vlm_priority.is_set():
                logger.debug("VLM priority: watch thread yielding to user request")
                return None

        if is_user_request:
            self._request_vlm_priority()

        with self._vlm_lock:
            try:
                return call_fn()
            finally:
                if is_user_request:
                    self._release_vlm_priority()

    def _make_priority_aware_check_fn(self, check_fn):
        """Wrap a VLM check function with priority awareness.

        Returns a function with the same signature as check_fn
        (prompt, image_b64) -> bool, but that skips the call (returns False)
        when a user request has VLM priority.

        Used when passing check_fn to VisionMonitor and MultiWatchMonitor
        so their background threads automatically yield to user requests.
        """

        def _priority_check(prompt: str, image_b64: str) -> bool:
            result = self._vlm_call_with_priority(
                lambda: check_fn(prompt, image_b64),
                is_user_request=False,
            )
            if result is None:
                return False  # Skipped — report as "not detected"
            return result

        return _priority_check

    def __init__(
        self,
        engine=None,  # jetson_assistant.Engine or None for server mode
        config: Optional[AssistantConfig] = None,
        wakeword: Optional[WakeWordDetector] = None,
        llm: Optional[LLMBackend] = None,
    ):
        """
        Initialize voice assistant.

        Args:
            engine: jetson-assistant Engine (None to use server mode)
            config: Assistant configuration
            wakeword: Custom wake word detector (created from config if None)
            llm: Custom LLM backend (created from config if None)
        """
        self.config = config or AssistantConfig()
        self.state = AssistantState.IDLE

        # Initialize engine (remote or local)
        if self.config.use_server or engine is None:
            self._init_remote_engine()
        else:
            self.engine = engine

        # Initialize components
        self._init_wakeword(wakeword)
        self._init_llm(llm)
        self._init_direct_stt()
        self._init_audio()
        self._init_vision()

        # Camera lock for thread-safe access (shared between processing and monitor)
        self._camera_lock = threading.Lock()

        # VLM priority queue — user requests preempt background watch threads.
        # Both the watch thread (scene description) and user tool calls use the
        # same vLLM instance. The lock serializes access; the priority event
        # signals that a user request is pending so watch threads should yield.
        self._vlm_lock = threading.Lock()
        self._vlm_priority = threading.Event()

        # Vision capture executor — runs camera grab concurrently with context building
        self._vision_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision")

        # Tool execution executor — runs multiple tool calls concurrently
        self._tool_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool")

        # Multi-camera pool (USB + RTSP cameras from config file)
        self._camera_pool = None
        self._init_camera_pool()

        # Vision monitor (background "watch for X" feature) — single camera legacy
        self._vision_monitor = None
        self._init_vision_monitor()

        # Multi-camera watch monitor
        self._multi_watch = None
        self._init_multi_watch()

        # Vision preview (live camera overlay)
        self._vision_preview = None
        self._init_vision_preview()

        # Aether Hub bridge (camera alerts, remote commands)
        self._aether_bridge = None
        self._init_aether()

        # Language instruction appended to system prompt (set by set_language tool)
        self._language_instruction = ""

        # External tool plugin modules (loaded in _init_tools)
        self._external_tool_modules: list = []

        # LLM-callable tools (vision watch, etc.)
        self._init_tools()

        # Update system prompt with tool descriptions (must happen after _init_tools)
        self._apply_tool_system_prompt()

        # State
        self._running = False
        self._audio_buffer: list[np.ndarray] = []
        self._conversations: dict[str, list[dict]] = {}
        self._listen_start_time = 0.0

        # Barge-in support: interrupt assistant while speaking
        self._bargein_event = threading.Event()
        self._aplay_proc = None  # Current aplay subprocess (for killing on barge-in)

        # Echo suppression: grace period after speech ends to avoid
        # picking up speaker echo as a new utterance
        self._speech_end_time = 0.0  # monotonic time when SPEAKING ended
        self._echo_grace_s = 1.5  # seconds to suppress after speech ends (USB mic/speaker need longer)

        # TTS audio cache — avoids re-synthesizing repeated short phrases
        self._tts_cache = TTSCache(max_entries=64, max_text_len=80)

        # Streaming TTS overlap: background thread synthesizes + plays queued
        # sentences so the LLM stream can keep generating while TTS is busy.
        self._tts_queue: queue.Queue = queue.Queue(maxsize=4)
        self._tts_thread = threading.Thread(
            target=self._tts_worker, daemon=True, name="tts-worker"
        )
        self._tts_thread.start()

        # Audio-reactive motion: background thread feeding chunks during playback
        self._audio_feed_thread: Optional[threading.Thread] = None

        # Persistent speech processing thread (keeps CUDA context consistent
        # for NeMo Nemotron STT which uses CUDA graph capture)
        self._speech_queue: queue.Queue = queue.Queue()
        self._speech_worker = threading.Thread(
            target=self._speech_worker_loop, daemon=True
        )
        self._speech_worker.start()

        # Enable debug logging when verbose so timing info actually appears
        if self.config.verbose:
            logger.setLevel(logging.DEBUG)

        logger.info("Voice assistant initialized")

    def _set_language_instruction(self, instruction: str) -> None:
        """Set the language instruction (called by set_language builtin tool)."""
        self._language_instruction = instruction

    def _set_state(self, new_state: AssistantState) -> None:
        """Set assistant state and notify external tool plugins."""
        old_state = self.state
        self.state = new_state
        if old_state != new_state:
            # Track when speaking ends for echo suppression
            if old_state == AssistantState.SPEAKING and new_state != AssistantState.SPEAKING:
                self._speech_end_time = time.monotonic()
            for mod in self._external_tool_modules:
                if hasattr(mod, "on_state_change"):
                    try:
                        mod.on_state_change(old_state.value, new_state.value)
                    except Exception:
                        pass

    def _notify_audio_playback(self, audio: np.ndarray, sample_rate: int) -> None:
        """Feed audio chunks to external tool plugins in sync with playback.

        Spawns a background thread that delivers ~50ms chunks at the
        correct rate, so audio-reactive motion stays synchronized with speech.
        """
        # Check if any external module has on_audio_chunk
        callbacks = [
            mod.on_audio_chunk
            for mod in self._external_tool_modules
            if hasattr(mod, "on_audio_chunk")
        ]
        if not callbacks:
            return

        chunk_ms = 50
        chunk_samples = int(sample_rate * chunk_ms / 1000)

        def _feed():
            for i in range(0, len(audio), chunk_samples):
                chunk = audio[i : i + chunk_samples]
                for cb in callbacks:
                    try:
                        cb(chunk, sample_rate)
                    except Exception:
                        pass
                import time
                time.sleep(chunk_ms / 1000.0)

        # Stop any previous feed thread
        self._audio_feed_thread = threading.Thread(target=_feed, daemon=True, name="AudioFeed")
        self._audio_feed_thread.start()

    def _get_conversation(self, session_id: str = "local") -> list[dict]:
        """Get or create conversation history for a session."""
        if session_id not in self._conversations:
            self._conversations[session_id] = []
        return self._conversations[session_id]

    def _init_remote_engine(self) -> None:
        """Initialize remote engine connecting to server."""
        from jetson_assistant.core.remote import RemoteEngine, RemoteEngineConfig

        remote_config = RemoteEngineConfig(
            host=self.config.server_host,
            port=self.config.server_port,
        )
        self.engine = RemoteEngine(remote_config)

        logger.info("Connecting to server at %s...", remote_config.base_url)
        if not self.engine.wait_for_server(timeout=10.0):
            raise ConnectionError(
                f"Could not connect to jetson-assistant server at {remote_config.base_url}. "
                "Make sure to start it first: jetson-assistant serve"
            )

    def _init_wakeword(self, wakeword: Optional[WakeWordDetector]) -> None:
        """Initialize wake word detector."""
        if wakeword:
            self.wakeword = wakeword
        else:
            self.wakeword = create_wakeword_detector(
                backend=self.config.wake_word_backend,
                wake_word=self.config.wake_word,
                threshold=self.config.wake_word_threshold,
            )

    def _init_llm(self, llm: Optional[LLMBackend]) -> None:
        """Initialize LLM."""
        # If using server LLM, skip local initialization
        if self.config.use_server and self.config.use_server_llm:
            self.llm = None  # Will use self.engine.chat() instead
            logger.info("Using server LLM (no local LLM loaded)")
            return

        # Use vision system prompt if vision enabled and no custom prompt.
        # Tool prompt is applied later by _apply_tool_system_prompt() after
        # tools are registered (overrides this if tools are available).
        system_prompt = self.config.system_prompt
        if self.config.vision_enabled and not system_prompt:
            system_prompt = self.config.vision_system_prompt or self.VISION_SYSTEM_PROMPT

        if llm:
            self.llm = llm
        else:
            kwargs = {
                "model": self.config.llm_model,
                "system_prompt": system_prompt,
            }
            if self.config.llm_backend in ("ollama", "vllm"):
                kwargs["host"] = self.config.llm_host
            if self.config.llm_backend == "ollama":
                kwargs["rag_collection"] = self.config.rag_collection
            self.llm = create_llm(
                backend=self.config.llm_backend,
                **kwargs,
            )

    def _init_direct_stt(self) -> None:
        """Initialize direct STT backend (bypasses speech server for STT)."""
        self._local_stt = None
        if self.config.stt_backend == "vllm":
            from jetson_assistant.stt.registry import get_stt_backend

            stt = get_stt_backend("vllm")
            stt.load(host=self.config.stt_host, model_size=self.config.stt_model)
            self._local_stt = stt
        elif self.config.stt_backend == "nemotron":
            from jetson_assistant.stt.registry import get_stt_backend

            stt = get_stt_backend("nemotron")
            stt.load(model_size=self.config.stt_model)
            self._local_stt = stt

    def _transcribe(self, audio: np.ndarray, sample_rate: int, language: str | None = None):
        """Transcribe audio using direct STT backend or engine."""
        if self._local_stt is not None:
            return self._local_stt.transcribe(audio, sample_rate, language)
        return self.engine.transcribe(audio, sample_rate=sample_rate, language=language)

    def _init_audio(self) -> None:
        """Initialize audio I/O."""
        self.audio_config = AudioConfig(
            sample_rate=self.config.input_sample_rate,
            chunk_duration_ms=100,
        )
        self.vad = VoiceActivityDetector(sample_rate=self.config.input_sample_rate)
        self.chimes = ChimeSounds(sample_rate=self.config.output_sample_rate)
        self.audio_output = AudioOutput(sample_rate=self.config.output_sample_rate)

    def _init_vision(self) -> None:
        """Initialize camera for vision support."""
        self.camera = None
        if not self.config.vision_enabled:
            return

        try:
            from jetson_assistant.assistant.vision import Camera, CameraConfig

            cam_config = CameraConfig(
                device=self.config.camera_device,
                width=self.config.camera_width,
                height=self.config.camera_height,
            )
            camera = Camera(cam_config)
            if camera.open():
                self.camera = camera
            else:
                logger.warning("Vision: Camera failed to open, falling back to text-only")
        except ImportError:
            logger.error(
                "Vision: opencv-python-headless not installed. "
                "Install with: pip install opencv-python-headless"
            )
        except Exception as e:
            logger.error("Vision: Camera init error: %s", e)

    def _init_vision_monitor(self) -> None:
        """Initialize vision monitor for 'watch for X' commands."""
        # Requires camera + local LLM with check_condition support
        if self.camera is None or self.llm is None:
            return
        if not hasattr(self.llm, 'check_condition'):
            return

        from jetson_assistant.assistant.vision import VisionMonitor

        self._vision_monitor = VisionMonitor(
            camera=self.camera,
            camera_lock=self._camera_lock,
            check_fn=self._make_priority_aware_check_fn(self.llm.check_condition),
            on_detected=self._on_watch_detected,
            can_speak_fn=lambda: self.state == AssistantState.IDLE,
            poll_interval=self.config.watch_poll_interval,
            confidence_threshold=self.config.watch_confidence_threshold,
            vote_window=self.config.watch_vote_window,
        )
        logger.info("VisionMonitor: available (use 'watch for...' or 'tell me when you see...')")

    def _init_camera_pool(self) -> None:
        """Initialize multi-camera pool from config file."""
        try:
            from jetson_assistant.assistant.cameras import CameraPool
            from pathlib import Path

            config_path = Path(self.config.camera_config_path).expanduser()
            self._camera_pool = CameraPool(config_path)

            # Auto-register local USB camera if available
            if self.camera is not None and self.camera.is_open:
                self._camera_pool.add_local(self.camera)

            # Auto-register remote camera (Aether SFU WebRTC stream) if configured
            if self.config.remote_camera_port > 0:
                try:
                    from jetson_assistant.assistant.remote_camera import RemoteCamera
                    remote_cam = RemoteCamera(port=self.config.remote_camera_port)
                    if remote_cam.open():
                        self._camera_pool.add_remote(remote_cam)
                        logger.info("RemoteCamera: listening on UDP port %d", self.config.remote_camera_port)
                    else:
                        logger.warning("RemoteCamera: failed to open")
                except Exception as e:
                    logger.error("RemoteCamera: init error: %s", e)

            count = len(self._camera_pool)
            if count > 0:
                names = ", ".join(s.name for s in self._camera_pool.list_cameras())
                logger.info("CameraPool: loaded %d cameras (%s)", count, names)
            else:
                logger.info("CameraPool: no cameras configured")

        except Exception as e:
            logger.error("CameraPool: init error: %s", e)
            self._camera_pool = None

    def _init_multi_watch(self) -> None:
        """Initialize multi-camera watch monitor."""
        if self._camera_pool is None or self.llm is None:
            return
        if not hasattr(self.llm, 'check_condition'):
            return

        try:
            from jetson_assistant.assistant.multi_watch import MultiWatchMonitor

            self._multi_watch = MultiWatchMonitor(
                camera_pool=self._camera_pool,
                check_fn=self._make_priority_aware_check_fn(self.llm.check_condition),
                on_detected=self._on_multi_watch_detected,
                can_speak_fn=lambda: self.state == AssistantState.IDLE,
                poll_interval=self.config.watch_poll_interval,
                confidence_threshold=self.config.watch_confidence_threshold,
                vote_window=self.config.watch_vote_window,
                cooldown_s=self.config.watch_cooldown,
            )
            logger.info("MultiWatch: available (use 'watch <camera> for...')")
        except Exception as e:
            logger.error("MultiWatch: init error: %s", e)

    def _init_aether(self) -> None:
        """Connect to Aether Hub if configured."""
        if self.config.aether_hub_host is None:
            return

        try:
            from jetson_assistant.assistant.aether_bridge import AetherBridge

            self._aether_bridge = AetherBridge.create(
                hub_host=self.config.aether_hub_host,
                hub_port=self.config.aether_hub_port,
                pin=self.config.aether_pin,
            )
            if self._aether_bridge is not None:
                self._aether_bridge.on_command(self._on_aether_command)
                self._start_status_publisher()
        except Exception as e:
            logger.error("AetherBridge: init error: %s", e)

    def _init_vision_preview(self) -> None:
        """Initialize vision preview if show_vision or stream_vision_port is set."""
        if self.camera is None:
            return
        if not self.config.show_vision and self.config.stream_vision_port <= 0:
            return

        from jetson_assistant.assistant.vision import VisionPreview

        self._vision_preview = VisionPreview(
            camera=self.camera,
            camera_lock=self._camera_lock,
            show_window=self.config.show_vision,
            stream_port=self.config.stream_vision_port,
        )
        self._vision_preview.start()

    def _update_preview(self, **kwargs) -> None:
        """Update vision preview overlay with current state info.

        Accepts keyword arguments: state, user_text, vlm_text, watch_text.
        Builds overlay lines from whichever are provided and updates the preview.
        """
        if self._vision_preview is None:
            return

        lines = []
        if "state" in kwargs:
            lines.append(f"State: {kwargs['state']}")
        if "user_text" in kwargs:
            lines.append(f"You: {kwargs['user_text'][:60]}")
        if "vlm_text" in kwargs:
            text = kwargs["vlm_text"]
            if len(text) > 60:
                text = text[:57] + "..."
            lines.append(f"VLM: {text}")
        if "watch_text" in kwargs:
            lines.append(f"Watch: {kwargs['watch_text']}")

        self._vision_preview.set_overlay(lines)

    def _apply_tool_system_prompt(self) -> None:
        """Update LLM system prompt with tool descriptions (prompt-based tool calling)."""
        if not self._tools or self.llm is None:
            return

        # Build tool descriptions from registry
        tool_lines = ["\nAvailable tools:"]
        for defn in self._tools.definitions():
            func = defn["function"]
            name = func["name"]
            desc = func["description"]
            params = func["parameters"].get("properties", {})
            required = func["parameters"].get("required", [])

            if params:
                param_parts = []
                for pname, pinfo in params.items():
                    ptype = pinfo.get("type", "string")
                    pdesc = pinfo.get("description", "")
                    req = " (required)" if pname in required else ""
                    param_parts.append(f"{pname}: {ptype}{req} - {pdesc}")
                tool_lines.append(f"- {name}: {desc}")
                for pp in param_parts:
                    tool_lines.append(f"    {pp}")
            else:
                tool_lines.append(f"- {name}: {desc}")

        tool_lines.append(
            '\nExamples:\n'
            'User: "What time is it?" → {"tool": "get_time", "args": {}}\n'
            'User: "What cameras do I have?" → {"tool": "list_cameras", "args": {}}\n'
            'User: "How many cameras?" → {"tool": "list_cameras", "args": {}}\n'
            'User: "Check the garage" → {"tool": "check_camera", "args": {"camera_name": "garage"}}\n'
            'User: "What do you see?" → {"tool": "check_camera", "args": {"camera_name": "local"}}\n'
            'User: "Search for latest AI news" → {"tool": "web_search", "args": {"query": "latest AI news"}}\n'
            'User: "What\'s the latest news on NVIDIA?" → {"tool": "web_search", "args": {"query": "NVIDIA latest news"}}\n'
            'User: "Tell me about Microsoft updates" → {"tool": "web_search", "args": {"query": "Microsoft latest updates"}}\n'
            'User: "What happened with OpenAI?" → {"tool": "web_search", "args": {"query": "OpenAI latest news"}}\n'
            'User: "Set a 30 second timer" → {"tool": "set_timer", "args": {"seconds": 30}}\n'
            'User: "Watch the garage for someone" → {"tool": "watch_camera", "args": {"camera_name": "garage", "condition": "Is there a person visible?"}}\n'
            'User: "Remember I need milk" → {"tool": "remember", "args": {"info": "Need to buy milk"}}\n'
            'User: "Speak in Hindi" → {"tool": "set_language", "args": {"language": "hindi"}}\n'
            'User: "Switch to English" → {"tool": "set_language", "args": {"language": "english"}}\n'
            'User: "Talk to me in Japanese" → {"tool": "set_language", "args": {"language": "japanese"}}\n'
            'User: "Hello, how are you?" → Just reply normally, no tool needed.'
        )

        prompt = self.TOOL_SYSTEM_PROMPT_HEADER + "\n".join(tool_lines)

        # Append custom system prompt from config (personality, extra rules).
        # Without this, config system_prompt (e.g. Reachy personality + tool
        # usage rules) would be completely overwritten by the tool prompt.
        if self.config.system_prompt:
            prompt += "\n\n" + self.config.system_prompt

        # Append language instruction if active (e.g., "Respond in Hindi")
        if self._language_instruction:
            prompt += "\n\n" + self._language_instruction

        self.llm.set_system_prompt(prompt)
        logger.info("Tool system prompt set (%d tools registered)", len(self._tools))

    def _init_tools(self) -> None:
        """Register LLM-callable tools via ToolRegistry."""
        from jetson_assistant.assistant.tools import ToolRegistry
        from jetson_assistant.assistant.builtin_tools import register_builtin_tools

        self._tools = ToolRegistry()

        # Build context dict with everything builtin tools need
        context = {
            "say": self.say,
            "engine": self.engine,
            "config": self.config,
            "llm": self.llm,
            "camera": self.camera,
            "camera_lock": self._camera_lock,
            "camera_pool": self._camera_pool,
            "vision_monitor": self._vision_monitor,
            "multi_watch": self._multi_watch,
            "update_preview": self._update_preview,
            "set_language_instruction": self._set_language_instruction,
            "apply_tool_system_prompt": self._apply_tool_system_prompt,
            "knowledge_collection": self.config.knowledge_collection,
        }

        register_builtin_tools(self._tools, context)

        # ── External tool plugins ──

        if self.config.external_tools:
            import importlib

            ext_context = {"llm": self.llm, "camera_pool": self._camera_pool, "say": self.say}
            for module_path in self.config.external_tools:
                try:
                    mod = importlib.import_module(module_path)
                    mod.register_tools(self._tools, ext_context)
                    self._external_tool_modules.append(mod)
                    logger.info("Loaded external tools from %s", module_path)
                except Exception as e:
                    logger.error("Failed to load external tools from %s: %s", module_path, e)

    def _handle_watch_command(self, user_text: str) -> Optional[str]:
        """
        Fast-path stop-watch detection (keyword only).

        Start-watching is handled by LLM tool calling in _process_speech.
        Returns confirmation string if stop matched, None otherwise.
        """
        text_lower = user_text.lower().strip()

        stop_patterns = ["stop watching", "stop looking", "cancel watch"]
        for pattern in stop_patterns:
            if pattern in text_lower:
                # Try multi-watch first
                if self._multi_watch is not None:
                    watches = self._multi_watch.list_watches()
                    if watches:
                        result = self._multi_watch.stop_all()
                        return f"OK, {result.lower()}"
                # Fall back to single-camera vision monitor
                if self._vision_monitor is not None and self._vision_monitor.is_watching:
                    self._vision_monitor.stop_watching()
                    return "OK, I've stopped watching."
                return "I wasn't watching for anything."

        return None

    def _handle_news_query(self, user_text: str) -> Optional[str]:
        """Fast-path for news/latest queries — bypass LLM, call web_search directly.

        The LLM is unreliable at calling web_search for news queries.
        If the user asks about "latest news" or "updates", we skip the LLM,
        search directly, and summarize with a single LLM call.

        Returns spoken response string, or None if not a news query.
        """
        import re

        text_lower = user_text.lower().strip()

        # Check for news-related keywords
        news_patterns = [
            r"\blatest\b", r"\bnews\b", r"\bupdate[sd]?\b",
            r"\brecent\b", r"\bcurrent events?\b", r"\bwhat.s happening\b",
            r"\bwhat happened\b", r"\btoday.s\b.*\bnews\b",
        ]
        if not any(re.search(p, text_lower) for p in news_patterns):
            return None

        # Extract the topic (strip news keywords to get the search subject)
        if not self._tools:
            return None

        logger.debug("Fast-path: news query detected")

        # Quick acknowledgment so the user feels heard
        self._set_state(AssistantState.SPEAKING)
        self._speak_quick("Sure, let me look that up.")

        # Build a search query from the user text
        query = user_text

        # Execute web_search tool directly
        from jetson_assistant.assistant.llm import ToolCallResult
        tc = ToolCallResult(name="web_search", arguments={"query": query})
        raw_result = self._tools.execute(tc)

        if not raw_result or raw_result.startswith("No results") or raw_result.startswith("Search failed"):
            return None  # Fall through to normal LLM path

        logger.debug("Tool: web_search('%s')", query)

        # Summarize with LLM for natural speech
        if self.llm is not None and len(raw_result) > 120:
            try:
                summary_prompt = (
                    f"The user asked: \"{user_text}\"\n"
                    f"Search results: {raw_result[:600]}\n\n"
                    "Summarize the key points in 2-3 natural spoken sentences. "
                    "No markdown, no formatting. Be concise."
                )
                summary = self.llm.generate(summary_prompt)
                return summary.text.strip()
            except Exception:
                return raw_result
        return raw_result

    def _is_slow_tool(self, tool_name: str) -> bool:
        """Check if a tool is expected to take >200ms."""
        return tool_name in self._SLOW_TOOLS

    def _maybe_acknowledge_slow_tools(self, tool_calls: list[dict]) -> None:
        """Speak brief acknowledgment if any tool in the list is slow."""
        if any(self._is_slow_tool(tc["name"]) for tc in tool_calls):
            self._speak_sentences("Let me check on that.")

    def _execute_tools_parallel(self, tool_calls: list[dict]) -> list[str]:
        """Execute tool calls concurrently. Single tool runs inline (no executor overhead).

        Each dict must have 'name' and 'arguments' keys.
        Returns list of result strings in same order as input.
        Exceptions are caught per-tool and returned as error strings.
        """
        from concurrent.futures import as_completed
        from jetson_assistant.assistant.llm import ToolCallResult

        def _run_one(tc_dict: dict) -> str:
            """Execute a single tool call dict, return result string."""
            obj = ToolCallResult(name=tc_dict["name"], arguments=tc_dict.get("arguments", {}))
            result = self._tools.execute(obj)
            return result if result is not None else f"Tool '{tc_dict['name']}' returned None."

        if len(tool_calls) == 1:
            try:
                return [_run_one(tool_calls[0])]
            except Exception as e:
                return [f"Error: {e}"]

        # Multiple tools — run in parallel via ThreadPoolExecutor
        futures = {}
        for i, tc in enumerate(tool_calls):
            fut = self._tool_executor.submit(_run_one, tc)
            futures[fut] = i

        results: list[str | None] = [None] * len(tool_calls)
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result(timeout=30)
            except Exception as e:
                results[idx] = f"Error: {e}"
        return results

    def _try_parse_tool_call(self, text: str) -> Optional[str]:
        """Try to parse prompt-based tool call(s) from the model's text output.

        Handles single: {"tool": "get_time", "args": {}}
        And multiple: {"tool": "set_language", ...}\n{"tool": "web_search", ...}

        Returns combined tool execution results, or None if no tool calls found.
        """
        import json

        text = text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.startswith("json"):
                text = text[4:].strip()

        # Handle malformed tool calls: 'tool_name {"arg": ...}' → proper JSON
        # Small LLMs sometimes emit 'reachy_see {"question": "..."}' instead of
        # '{"tool": "reachy_see", "args": {"question": "..."}}'
        if self._tools and not text.startswith("{"):
            import re
            m = re.match(r'^(\w+)\s*(\{.+)', text, re.DOTALL)
            if m:
                tool_name_prefix = m.group(1)
                registered = {d["function"]["name"] for d in self._tools.definitions()}
                if tool_name_prefix in registered:
                    try:
                        args_obj = json.loads(m.group(2))
                        text = json.dumps({"tool": tool_name_prefix, "args": args_obj})
                        logger.debug("Reformatted malformed tool call: %s", text[:120])
                    except json.JSONDecodeError:
                        pass

        # Find all JSON objects in the text by matching balanced braces
        tool_calls = []
        i = 0
        while i < len(text):
            if text[i] == "{":
                depth = 0
                start = i
                for j in range(i, len(text)):
                    if text[j] == "{":
                        depth += 1
                    elif text[j] == "}":
                        depth -= 1
                        if depth == 0:
                            json_str = text[start:j + 1]
                            try:
                                data = json.loads(json_str)
                                tool_name = data.get("tool") or data.get("name")
                                if tool_name:
                                    args = data.get("args") or data.get("arguments") or {}
                                    tool_calls.append((tool_name, args))
                            except json.JSONDecodeError:
                                pass
                            i = j + 1
                            break
                else:
                    i += 1
            else:
                i += 1

        if not tool_calls:
            return None

        # Deduplicate (model sometimes outputs the same tool call twice)
        seen = set()
        unique_calls = []
        for tool_name, args in tool_calls:
            key = (tool_name, str(sorted(args.items())))
            if key not in seen:
                seen.add(key)
                unique_calls.append((tool_name, args))
        tool_calls = unique_calls

        # Resolve fuzzy tool names before execution
        registered = {d["function"]["name"] for d in self._tools.definitions()}
        resolved_calls = []
        for tool_name, args in tool_calls:
            # Fuzzy-match hallucinated tool names from small LLMs
            if tool_name not in registered:
                resolved = self._fuzzy_resolve_tool(tool_name, registered, args)
                if resolved:
                    logger.info("Fuzzy tool resolve: '%s' → '%s'", tool_name, resolved)
                    tool_name = resolved
                else:
                    logger.warning("Unknown tool '%s', no fuzzy match", tool_name)
            logger.debug("Tool: %s(%s)", tool_name, args)
            resolved_calls.append({"name": tool_name, "arguments": args})

        # Acknowledge slow tools before executing
        self._maybe_acknowledge_slow_tools(resolved_calls)

        # Execute tools — single tool inline, multiple tools in parallel
        results = self._execute_tools_parallel(resolved_calls)

        return " ".join(results) if results else "Got it."

    @staticmethod
    def _fuzzy_resolve_tool(
        hallucinated: str, registered: set[str], args: dict
    ) -> str | None:
        """Resolve a hallucinated tool name to the closest registered tool.

        Small LLMs (7B) frequently hallucinate tool names like
        'camera_candy_monitor' instead of 'watch_camera'. This uses
        keyword overlap + argument hints to find the right tool.
        """
        h = hallucinated.lower().replace("-", "_")
        h_parts = set(h.split("_"))

        # Keyword → tool mapping for common hallucinations
        keyword_map = {
            "watch": "watch_camera",
            "monitor": "watch_camera",
            "watching": "watch_camera",
            "surveillance": "watch_camera",
            "alert": "watch_camera",
            "stop_watch": "stop_watching_camera",
            "stop_monitor": "stop_watching_camera",
            "camera_check": "check_camera",
            "look_camera": "check_camera",
            "see_camera": "check_camera",
            "search": "web_search",
            "timer": "set_timer",
            "language": "set_language",
            "time": "get_time",
        }

        # Try keyword match first
        for keyword, tool in keyword_map.items():
            if tool in registered and keyword in h:
                return tool

        # Try argument-based hints
        if "condition" in args and "watch_camera" in registered:
            return "watch_camera"
        if "camera_name" in args:
            if "watch" in h_parts or "monitor" in h_parts:
                if "watch_camera" in registered:
                    return "watch_camera"
            if "check_camera" in registered:
                return "check_camera"

        # Try word overlap scoring
        best, best_score = None, 0
        for tool in registered:
            t_parts = set(tool.split("_"))
            overlap = len(h_parts & t_parts)
            if overlap > best_score:
                best_score = overlap
                best = tool
        if best_score >= 1:
            return best

        return None

    def _on_watch_detected(self, condition, frame_b64: str) -> None:
        """Callback when the vision monitor detects the watched condition."""
        logger.info("VisionMonitor: detected '%s'", condition.description)
        self._update_preview(state="SPEAKING", watch_text=f"detected {condition.description}!")
        self.say(condition.announce_template)

        # Add to conversation history so user can follow up
        self._get_conversation("local").append(
            {"role": "assistant", "content": condition.announce_template}
        )

    def _on_multi_watch_detected(
        self, camera_name: str, condition, frame_b64
    ) -> None:
        """Callback when the multi-watch monitor detects a condition on a camera."""
        import time as _time

        # Wait up to 10s for assistant to not be speaking/processing before announcing
        # IDLE = wake-word mode, LISTENING = always-listen mode — both are "available"
        for _ in range(20):
            if self.state not in (AssistantState.SPEAKING, AssistantState.PROCESSING):
                break
            logger.debug("Watch alert waiting (state=%s)", self.state.value)
            _time.sleep(0.5)

        msg = f"Alert: {condition.description} detected on {camera_name}!"
        logger.info("MultiWatch: %s", msg)
        self._update_preview(state="SPEAKING", watch_text=f"{camera_name}: detected!")
        self.say(msg)

        # Add to conversation history so user can follow up
        self._get_conversation("local").append({"role": "assistant", "content": msg})

        # Publish to Aether Hub if connected
        if self._aether_bridge is not None:
            try:
                self._aether_bridge.publish_alert(camera_name, condition.description)
            except Exception as e:
                logger.error("AetherBridge: alert publish error: %s", e)

    # ── Aether command dispatcher (mobile/web → assistant) ──

    def _on_aether_command(self, data: dict) -> None:
        """Handle incoming command from Aether Hub (DIRECT_MESSAGE or SEND_TO)."""
        msg_type = data.get("type", "")

        # Extract sender ID and payload based on message type
        if msg_type == "DIRECT_MESSAGE":
            from_id = data.get("from", "")
            payload = data.get("payload", {})
        elif msg_type in ("SEND_TO", "ASSISTANT_CMD"):
            from_id = data.get("from", data.get("clientId", ""))
            payload = data.get("payload", data)
        else:
            return

        # Extract command from payload
        if isinstance(payload, dict):
            cmd_id = payload.get("id", "")
            command = payload.get("command", "")
            args = payload.get("args", {})
        else:
            return

        if not command:
            return

        logger.info("AetherBridge: command '%s' from %s...", command, from_id[:12])

        # Dispatch in background thread to avoid blocking message handler
        t = threading.Thread(
            target=self._execute_aether_command,
            args=(from_id, cmd_id, command, args),
            daemon=True,
        )
        t.start()

    def _execute_aether_command(
        self, from_id: str, cmd_id: str, command: str, args: dict
    ) -> None:
        """Execute a command and send the response back to the requester."""
        try:
            result = self._dispatch_command(command, args)
            response = {
                "type": "CMD_RESPONSE",
                "id": cmd_id,
                "success": True,
                "result": result,
            }
        except Exception as e:
            logger.error("AetherBridge: command error: %s", e)
            response = {
                "type": "CMD_RESPONSE",
                "id": cmd_id,
                "success": False,
                "error": str(e),
            }

        if self._aether_bridge is not None and from_id:
            try:
                self._aether_bridge.send_to(from_id, response)
            except Exception as e:
                logger.error("AetherBridge: response send error: %s", e)

    def _dispatch_command(self, command: str, args: dict) -> dict:
        """Route a command to the appropriate handler. Returns result dict."""
        if command == "list_cameras":
            if self._camera_pool is None:
                return {"cameras": []}
            cameras = self._camera_pool.list_cameras()
            return {
                "cameras": [
                    {"name": c.name, "url": c.url, "location": c.location}
                    for c in cameras
                ]
            }

        elif command == "check_camera":
            camera_name = args.get("camera_name", "")
            question = args.get(
                "question",
                "Describe what you see in this image. Be specific and concise.",
            )

            if self._camera_pool is None or not self._camera_pool.has(camera_name):
                available = []
                if self._camera_pool is not None:
                    available = [s.name for s in self._camera_pool.list_cameras()]
                raise ValueError(
                    f"Camera '{camera_name}' not found. Available: {available}"
                )

            frame_b64 = self._camera_pool.capture_base64(camera_name)
            if frame_b64 is None:
                raise RuntimeError(f"Failed to capture frame from '{camera_name}'.")

            description = ""
            if self.llm is not None:
                try:
                    response = self._vlm_call_with_priority(
                        lambda: self.llm.generate(question, images=[frame_b64]),
                        is_user_request=True,
                    )
                    description = response.text.strip()
                except Exception as e:
                    description = f"Vision analysis failed: {e}"

            return {"description": description, "image": frame_b64}

        elif command == "watch_camera":
            camera_name = args.get("camera_name", "")
            condition_text = args.get("condition", "")

            if self._multi_watch is None:
                raise RuntimeError("Multi-watch monitor not available.")

            from jetson_assistant.assistant.vision import WatchCondition

            wc = WatchCondition(
                description=condition_text,
                prompt=f"{condition_text} Answer only YES or NO.",
                announce_template=f"Alert on {camera_name}: {condition_text}",
            )
            result = self._multi_watch.start_watching(camera_name, wc)
            return {"message": result}

        elif command == "stop_watching":
            camera_name = args.get("camera_name", "all")
            if self._multi_watch is None:
                return {"message": "No watches active."}
            result = self._multi_watch.stop_watching(camera_name)
            return {"message": result}

        elif command == "list_watches":
            if self._multi_watch is None:
                return {"watches": []}
            return {"watches": self._multi_watch.list_watches()}

        elif command == "get_status":
            return self._build_status()

        elif command == "add_camera":
            if self._camera_pool is None:
                raise RuntimeError("Camera pool not available.")
            result = self._camera_pool.add(
                name=args.get("name", ""),
                url=args.get("url", ""),
                location=args.get("location", ""),
            )
            return {"message": result}

        elif command == "remove_camera":
            if self._camera_pool is None:
                raise RuntimeError("Camera pool not available.")
            result = self._camera_pool.remove(args.get("name", ""))
            return {"message": result}

        elif command == "voice_query":
            query_text = args.get("text", "")
            if not query_text.strip():
                raise ValueError("Empty query text")
            include_audio = args.get("include_audio", True)
            session_id = args.get("session_id", "default")
            remote_image = args.get("image_base64")

            # Vision gating — explicit snapshots always pass through.
            # Intent router decides auto-capture (TOOL = no image, CHAT = legacy).
            images = None
            if remote_image:
                # Explicit base64 snapshot from phone (always allowed)
                images = [remote_image]
            elif self._classify_intent(query_text) == "CHAT" and not (self._tools and self._tools.definitions()):
                # Legacy path: no tools, use regex vision gating
                if self._camera_pool is not None and self._has_visual_intent(query_text):
                    phone_frame = self._camera_pool.capture_frame("phone")
                    if phone_frame is not None:
                        import cv2
                        _, buf = cv2.imencode(".jpg", phone_frame)
                        images = [base64.b64encode(buf).decode()]
                    elif self.camera is not None and self.camera.is_open:
                        with self._camera_lock:
                            frame_b64 = self.camera.capture_base64()
                        if frame_b64:
                            images = [frame_b64]
                elif self.camera is not None and self.camera.is_open:
                    if self._has_visual_intent(query_text):
                        with self._camera_lock:
                            frame_b64 = self.camera.capture_base64()
                        if frame_b64:
                            images = [frame_b64]

            if self.llm is None:
                raise RuntimeError("LLM not available")

            conversation = self._get_conversation(session_id)
            context = conversation[-self.config.conversation_history * 2:]
            response = self._vlm_call_with_priority(
                lambda: self.llm.generate(query_text, context=context, images=images),
                is_user_request=True,
            )
            response_text = response.text.strip()

            # Tool call detection (reuse existing parser)
            if self._tools and response_text:
                tool_result = self._try_parse_tool_call(response_text)
                if tool_result is not None:
                    response_text = tool_result

            conversation.append({"role": "user", "content": query_text})
            conversation.append({"role": "assistant", "content": response_text})

            result = {"text": response_text}

            if include_audio and response_text:
                import io
                from scipy.io import wavfile

                tts_result = self.engine.synthesize(
                    response_text,
                    voice=self.config.tts_voice,
                    language=self.config.tts_language,
                )
                audio = tts_result.audio
                if audio.dtype != np.int16:
                    if audio.dtype in (np.float32, np.float64):
                        audio = (audio * 32767).astype(np.int16)
                    else:
                        audio = audio.astype(np.int16)
                buf = io.BytesIO()
                wavfile.write(buf, tts_result.sample_rate, audio)
                result["audio_base64"] = base64.b64encode(buf.getvalue()).decode()

            return result

        elif command == "rag_collections":
            # List all knowledge base collections
            from jetson_assistant.config import get_default_cache_dir
            persist_dir = get_default_cache_dir() / "rag"
            if not persist_dir.exists():
                return {"collections": []}
            try:
                import chromadb
                from chromadb.config import Settings
                client = chromadb.PersistentClient(
                    path=str(persist_dir),
                    settings=Settings(anonymized_telemetry=False),
                )
                collections = []
                for col in client.list_collections():
                    collections.append({
                        "name": col.name,
                        "count": col.count(),
                    })
                return {"collections": collections}
            except ImportError:
                return {"collections": []}

        elif command == "rag_ingest":
            collection_name = args.get("collection", "")
            source_type = args.get("source_type", "text")
            content = args.get("content", "")
            if not collection_name or not content:
                raise ValueError("collection and content required")
            from jetson_assistant.rag import RAGPipeline
            rag = RAGPipeline(collection_name=collection_name)
            if source_type == "url":
                count = rag.ingest_url(content, follow_links=False, verbose=False)
            else:
                count = rag.ingest_text(content, verbose=False)
            return {"message": f"Added {count} chunks to '{collection_name}'", "count": count}

        elif command == "rag_search":
            collection_name = args.get("collection", "")
            query = args.get("query", "")
            top_k = int(args.get("top_k", 3))
            if not collection_name or not query:
                raise ValueError("collection and query required")
            from jetson_assistant.rag import RAGPipeline
            rag = RAGPipeline(collection_name=collection_name)
            results = rag.retrieve(query, top_k=top_k)
            return {
                "results": [
                    {"content": r["content"][:500], "score": round(r["score"], 3)}
                    for r in results
                ]
            }

        else:
            raise ValueError(f"Unknown command: {command}")

    @staticmethod
    def _sanitize_url(url: str) -> str:
        """Strip credentials from URLs for safe broadcast.

        Replaces 'rtsp://user:pass@host' with 'rtsp://***@host'.
        """
        import re
        return re.sub(r'(rtsp://)([^@]+)@', r'\1***@', url)

    def _build_status(self) -> dict:
        """Build current assistant status for publishing."""
        cameras = []
        if self._camera_pool is not None:
            cameras = [
                {"name": c.name, "url": self._sanitize_url(c.url), "location": c.location}
                for c in self._camera_pool.list_cameras()
            ]

        watches = []
        if self._multi_watch is not None:
            watches = self._multi_watch.list_watches()

        return {
            "cameras": cameras,
            "watches": watches,
            "timestamp": time.time(),
        }

    def _start_status_publisher(self) -> None:
        """Start a daemon thread that publishes ASSISTANT_STATUS every 30s."""
        def _publish_loop():
            while self._aether_bridge is not None and self._aether_bridge.is_connected:
                try:
                    status = self._build_status()
                    self._aether_bridge.publish_status(
                        cameras=status["cameras"],
                        watches=status["watches"],
                    )
                except Exception as e:
                    logger.error("AetherBridge: status publish error: %s", e)
                # Sleep in small increments so we can exit cleanly
                for _ in range(30):
                    if self._aether_bridge is None:
                        return
                    time.sleep(1.0)

        t = threading.Thread(target=_publish_loop, daemon=True)
        t.start()
        logger.info("AetherBridge: status publisher started (30s interval)")

    def run(self) -> None:
        """
        Run the assistant (blocking).

        This is the main loop that:
        1. Listens for wake word
        2. Records user speech
        3. Processes with STT -> LLM -> TTS
        4. Plays response
        """
        self._running = True
        logger.info("Assistant ready! Say '%s' to start...", self.config.wake_word.replace('_', ' '))

        audio_input = AudioInput(config=self.audio_config, device=self.config.audio_input_device)

        try:
            audio_input.start(self._on_audio_chunk)

            while self._running:
                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Stopping assistant...")
        finally:
            if self._vision_preview is not None:
                self._vision_preview.stop()
            if self._vision_monitor is not None:
                self._vision_monitor.stop_watching()
            if self._multi_watch is not None:
                self._multi_watch.stop_all()
            if self._aether_bridge is not None:
                self._aether_bridge.disconnect()
            audio_input.stop()
            if self._camera_pool is not None:
                self._camera_pool.close_all()
            elif self.camera is not None:
                self.camera.close()
            for mod in self._external_tool_modules:
                if hasattr(mod, "cleanup"):
                    try:
                        mod.cleanup()
                    except Exception as e:
                        logger.error("External tool cleanup error (%s): %s", mod.__name__, e)
            self._running = False

    def run_async(self) -> threading.Thread:
        """
        Run the assistant in background thread.

        Returns:
            Thread running the assistant
        """
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        """Stop the assistant."""
        self._running = False

    def _on_audio_chunk(self, audio: np.ndarray) -> None:
        """
        Callback for each audio chunk from microphone.

        Implements the state machine.
        """
        if self.state == AssistantState.SPEAKING:
            # Suppress all mic processing during playback.
            # Without hardware AEC, the speaker echo easily exceeds any
            # reasonable barge-in threshold and kills aplay mid-sentence.
            return

        elif self.state == AssistantState.IDLE:
            # Echo suppression: skip wake detection briefly after speech ends
            if time.monotonic() - self._speech_end_time < self._echo_grace_s:
                return

            # Check for wake word
            if self.wakeword.detect(audio):
                self._on_wake_detected()

        elif self.state == AssistantState.LISTENING:
            # Record audio and check for end of speech
            self._audio_buffer.append(audio.copy())

            # Check for timeout
            if time.time() - self._listen_start_time > self.config.max_listen_time_s:
                self._on_listen_complete()
                return

            # Check for end of speech (silence after talking)
            if self.vad.detect_end_of_speech(
                audio,
                silence_threshold_ms=self.config.silence_timeout_ms,
                chunk_duration_ms=self.audio_config.chunk_duration_ms,
            ):
                self._on_listen_complete()

    def _on_wake_detected(self) -> None:
        """Handle wake word detection."""
        if self.config.verbose:
            logger.debug("Wake word detected!")

        self._set_state(AssistantState.LISTENING)
        self._audio_buffer = []
        self._listen_start_time = time.time()
        self.wakeword.reset()
        self.vad.reset()

        self._update_preview(state="LISTENING")

        # Play chime
        if self.config.play_chimes:
            try:
                self.audio_output.play_blocking(
                    self.chimes.wake_chime(),
                    self.config.output_sample_rate,
                )
            except Exception:
                pass

        # Callback
        if self.config.on_wake:
            self.config.on_wake()

        if self.config.on_listen_start:
            self.config.on_listen_start()

        logger.info("Listening...")

    def _on_listen_complete(self) -> None:
        """Handle end of user speech."""
        self._set_state(AssistantState.PROCESSING)

        # Play thinking chime to indicate processing
        if self.config.play_chimes:
            try:
                self.audio_output.play_blocking(
                    self.chimes.thinking_chime(),
                    self.config.output_sample_rate,
                )
            except Exception:
                pass

        if self.config.verbose:
            logger.debug("Processing...")

        # Queue for persistent worker thread (CUDA context stays consistent)
        self._speech_queue.put("process")

    # Known Whisper hallucination patterns (generated on silence/noise)
    _HALLUCINATION_PATTERNS = {
        "thank you",
        "thanks for watching",
        "please subscribe",
        "subscribe to my channel",
        "like and subscribe",
        "see you in the next video",
        "i'll see you in the next",
        "i'm sorry",
        "bye",
        "goodbye",
        "you",
        "oh",
        "hmm",
        "the end",
        "silence",
    }

    def _is_hallucination(self, text: str) -> bool:
        """Check if transcription is a known Whisper hallucination."""
        normalized = text.lower().strip().rstrip(".")
        # Check exact match
        if normalized in self._HALLUCINATION_PATTERNS:
            return True
        # Check if text is just repetition of a short phrase
        words = normalized.split()
        if len(words) >= 4:
            # e.g. "I'm sorry I'm sorry I'm sorry I'm sorry"
            half = len(words) // 2
            if words[:half] == words[half:2 * half]:
                return True
        # Check contains known pattern
        for pattern in self._HALLUCINATION_PATTERNS:
            if len(pattern) > 5 and pattern in normalized:
                return True
        return False

    def _speech_worker_loop(self) -> None:
        """Persistent worker thread for speech processing.

        Keeps the CUDA context consistent across STT calls — NeMo's
        Nemotron RNNT uses CUDA graph capture which must be replayed
        on the same thread that captured it.
        """
        # Initialize CUDA context on this thread
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
        except Exception:
            pass

        while True:
            try:
                self._speech_queue.get()
                self._process_speech()
            except Exception:
                logger.exception("Speech worker error")
                self._set_state(AssistantState.IDLE)

    def _process_speech(self) -> None:
        """Process recorded speech: STT -> LLM -> TTS."""
        try:
            # Concatenate audio buffer
            if not self._audio_buffer:
                self._set_state(AssistantState.IDLE)
                return

            audio = np.concatenate(self._audio_buffer)
            self._audio_buffer = []

            # Audio energy gate — skip if too quiet (no actual speech)
            rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
            if rms < 0.02:
                if self.config.verbose:
                    logger.debug("(rejected: audio too quiet, RMS=%.4f)", rms)
                self._set_state(AssistantState.IDLE)
                return

            # STT - force English to avoid wrong language detection
            start = time.perf_counter()
            result = self._transcribe(
                audio,
                sample_rate=self.config.input_sample_rate,
                language="en",
            )
            user_text = result.text.strip()
            stt_time = time.perf_counter() - start

            if not user_text:
                logger.debug("(no speech detected)")
                self._set_state(AssistantState.IDLE)
                return

            word_count = len(user_text.split())

            if word_count < self.config.stt_min_words:
                if self.config.verbose:
                    logger.debug("(rejected: too few words)")
                self._set_state(AssistantState.IDLE)
                return

            # Filter known Whisper hallucinations
            if self._is_hallucination(user_text):
                if self.config.verbose:
                    logger.debug("(rejected: hallucination) \"%s\"", user_text)
                self._set_state(AssistantState.IDLE)
                return

            logger.info("You: %s", user_text)
            self._update_preview(state="PROCESSING", user_text=user_text)
            if self._vision_preview is not None:
                self._vision_preview.add_transcript("user", user_text)

            if self.config.verbose:
                logger.debug("STT: %.0fms", stt_time * 1000)
            if self._vision_preview is not None:
                self._vision_preview.add_timing("stt", stt_time * 1000)

            # Callback
            if self.config.on_listen_end:
                self.config.on_listen_end(user_text)

            # Check for watch commands before LLM (instant keyword match)
            watch_response = self._handle_watch_command(user_text)
            if watch_response is not None:
                self._set_state(AssistantState.SPEAKING)
                self.say(watch_response)
                logger.info("Assistant: %s", watch_response)
                if self._vision_preview is not None:
                    self._vision_preview.add_transcript("assistant", watch_response)
                self._get_conversation("local").append({"role": "user", "content": user_text})
                self._get_conversation("local").append({"role": "assistant", "content": watch_response})
                self._set_state(AssistantState.IDLE)
                return

            # Fast-path: force web_search for news/latest queries (LLM is unreliable)
            news_response = self._handle_news_query(user_text)
            if news_response is not None:
                self._set_state(AssistantState.SPEAKING)
                self._speak_sentences(news_response)
                logger.info("Assistant: %s", news_response)
                if self._vision_preview is not None:
                    self._vision_preview.add_transcript("assistant", news_response)
                self._get_conversation("local").append({"role": "user", "content": user_text})
                self._get_conversation("local").append({"role": "assistant", "content": news_response})
                self._set_state(AssistantState.IDLE)
                self._update_preview(state="IDLE")

                # Callback
                if self.config.on_response:
                    self.config.on_response(news_response)
                return

            # Pipelined LLM + TTS: Stream LLM and synthesize sentences as they complete
            self._set_state(AssistantState.SPEAKING)

            # Start vision capture speculatively in background thread.
            # The camera grab takes ~50-100ms; by running it concurrently
            # with context building and intent classification, we save
            # that time from the critical path.
            vision_future = None
            if (
                self.camera is not None
                and self.camera.is_open
                and self.config.vision_enabled
                and self._has_visual_intent(user_text)
            ):
                vision_future = self._capture_vision_async()

            context = self._get_conversation("local")[-self.config.conversation_history * 2 :]

            # Intent router: fast LLM classification (~90ms) decides whether
            # to attach a camera frame. TOOL queries get text-only (tools
            # handle their own vision). CHAT queries also text-only.
            # Only legacy mode (no tools) uses regex-based vision gating.
            images = None
            intent = self._classify_intent(user_text)
            if self.config.verbose:
                router_ms = (time.perf_counter() - time.perf_counter()) if intent == "CHAT" else 0
                logger.debug("Router: %s", intent)

            # Collect the vision future — only use the frame for legacy
            # CHAT path (no tools). The capture already ran concurrently.
            if vision_future is not None:
                if intent == "CHAT" and not (self._tools and self._tools.definitions()):
                    frame_b64 = vision_future.result(timeout=2.0)
                    if frame_b64:
                        images = [frame_b64]
                        if self.config.verbose:
                            logger.debug("Vision: frame captured (async)")
                else:
                    # Intent resolved to TOOL or tools are loaded — discard
                    vision_future.cancel()

            full_response = []
            sentence_count = 0
            first_audio_time = None
            tool_call_json = None  # Original JSON for conversation history
            llm_start = time.perf_counter()

            # Determine if using server LLM or local LLM
            use_server_llm = self.config.use_server and self.config.use_server_llm and self.llm is None

            if use_server_llm:
                # SERVER LLM MODE: Use server's chat_stream endpoint
                if images:
                    logger.debug("Vision: server LLM mode does not support images yet")
                    images = None
                if self.config.stream_llm:
                    try:
                        for sentence in self.engine.chat_stream(
                            message=user_text,
                            context=context,
                            use_rag=bool(self.config.rag_collection),
                            rag_collection=self.config.rag_collection,
                            llm_model=self.config.llm_model,
                            system_prompt=self.config.system_prompt,
                        ):
                            if not sentence.strip():
                                continue

                            full_response.append(sentence)
                            sentence_count += 1

                            # Log first sentence timing
                            if sentence_count == 1:
                                first_sentence_time = time.perf_counter() - llm_start
                                if self.config.verbose:
                                    logger.debug("LLM first sentence: %.0fms", first_sentence_time * 1000)

                            # Enqueue for background TTS worker (overlaps with
                            # LLM generating the next sentence)
                            self._tts_queue.put((
                                sentence,
                                self.config.tts_voice,
                                self.config.tts_language,
                            ))

                    except Exception as e:
                        logger.error("Server stream error: %s", e)

                    # Wait for the TTS worker to finish all queued sentences
                    self._tts_queue.join()
                    response_text = " ".join(full_response) if full_response else "(no response)"
                else:
                    # Non-streaming server call
                    result = self.engine.chat(
                        message=user_text,
                        context=context,
                        use_rag=bool(self.config.rag_collection),
                        rag_collection=self.config.rag_collection,
                        llm_model=self.config.llm_model,
                        system_prompt=self.config.system_prompt,
                    )
                    response_text = result.get("response", "(no response)")
                    llm_time = time.perf_counter() - llm_start

                    if self.config.verbose:
                        logger.debug("LLM: %.0fms", llm_time * 1000)

                    # Synthesize full response
                    tts_result = self.engine.synthesize(
                        response_text,
                        voice=self.config.tts_voice,
                        language=self.config.tts_language,
                    )
                    self.audio_output.play_blocking(tts_result.audio, tts_result.sample_rate)

            elif self.config.stream_llm and hasattr(self.llm, 'generate_stream'):
                # LOCAL LLM STREAMING MODE (with early stream fork)
                # First sentence decides: starts with '{' → buffer for tool call,
                # otherwise → speak immediately for lowest latency.
                import json as _json
                from jetson_assistant.assistant.llm import ToolCallResult

                llm_kwargs: dict = {"context": context, "images": images}

                had_tool_calls = False
                tool_results = []
                buffer_mode = None  # None=undecided, True=tool call, False=speak
                tool_call_json = None  # Original JSON for conversation history

                try:
                    for item in self.llm.generate_stream(user_text, **llm_kwargs):
                        if isinstance(item, ToolCallResult):
                            result = self._tools.execute(item)
                            if result:
                                tool_results.append(result)
                            had_tool_calls = True
                            continue

                        sentence = item
                        if not sentence.strip():
                            continue

                        full_response.append(sentence)
                        sentence_count += 1

                        # Decide mode on first sentence
                        if buffer_mode is None:
                            stripped_s = sentence.strip()
                            # Detect tool calls: '{"tool":...}' or 'tool_name {"arg":...}'
                            looks_like_tool = (
                                stripped_s.startswith("{")
                                or ("{" in stripped_s and self._tools and any(
                                    stripped_s.startswith(d["function"]["name"])
                                    for d in self._tools.definitions()
                                ))
                            )
                            if self._tools and looks_like_tool:
                                buffer_mode = True
                            else:
                                buffer_mode = False
                                if self.config.verbose:
                                    llm_time = time.perf_counter() - llm_start
                                    logger.debug("LLM: %.0fms", llm_time * 1000)
                                if self._vision_preview is not None:
                                    self._vision_preview.add_timing("llm", (time.perf_counter() - llm_start) * 1000)

                        if buffer_mode:
                            # Accumulate — will parse as tool call after stream ends
                            continue

                        # Speak mode — enqueue for background TTS worker
                        # (overlaps synthesis+playback with LLM generation)
                        if self._bargein_event.is_set():
                            break

                        if sentence_count == 1 and self.config.verbose:
                            if self._vision_preview is not None:
                                self._vision_preview.add_timing("tts", 0)  # TTS starts immediately via queue

                        self._tts_queue.put((
                            sentence,
                            self.config.tts_voice,
                            self.config.tts_language,
                        ))

                except Exception as e:
                    logger.error("Stream error: %s", e)

                # Wait for the TTS worker to finish all queued sentences
                if buffer_mode is False:
                    self._tts_queue.join()

                response_text = " ".join(full_response) if full_response else ""

                # Tool call detection — only when we buffered (or empty response)
                if buffer_mode is True and self._tools and response_text.strip():
                    stripped = response_text.strip()
                    logger.debug("Tool buffer: %s", stripped[:200])

                    t_exec = time.perf_counter()
                    tool_result = self._try_parse_tool_call(stripped)
                    if self.config.verbose:
                        logger.debug("  Tool exec: %.0fms", (time.perf_counter() - t_exec) * 1000)
                    logger.debug("Tool result: %s", tool_result)
                    if tool_result is not None:
                        had_tool_calls = True
                        tool_call_json = stripped  # Preserve for conversation history
                        tool_results.append(tool_result)
                        response_text = ""

                # If model called tools, decide whether to speak the result
                if had_tool_calls:
                    raw_result = " ".join(tool_results) if tool_results else ""

                    # Detect robot action confirmations that shouldn't be spoken
                    # (user can see/hear the robot). These are short phrases like
                    # "Looking left.", "Expressing happy.", "Antennas set to ...",
                    # "Dancing: wiggle!", "Nodding yes."
                    _action_words = ("Looking ", "Expressing ", "Antennas ", "Dancing",
                                     "Nodding ", "Shaking ", "Powering ", "Reachy is ")
                    is_action = raw_result.startswith(_action_words) and len(raw_result) < 60

                    if not raw_result or is_action:
                        response_text = ""
                    elif len(raw_result) > 80 and hasattr(self.llm, '_client'):
                        # Summarize long results (search snippets, etc.)
                        t_sum = time.perf_counter()
                        try:
                            resp = self.llm._client.chat.completions.create(
                                model=self.llm.model,
                                messages=[{"role": "user", "content":
                                    f"Q: {user_text}\nFacts: {raw_result[:500]}\nAnswer the question in 2 short sentences using ONLY facts above. Be specific and relevant:"}],
                                max_tokens=80,
                                temperature=0,
                            )
                            response_text = resp.choices[0].message.content.strip()
                        except Exception:
                            response_text = raw_result
                        if self.config.verbose:
                            logger.debug("  Summary: %.0fms", (time.perf_counter() - t_sum) * 1000)
                    else:
                        # Informational result (vision, timer, etc.) — speak directly
                        response_text = raw_result

                    llm_time = time.perf_counter() - llm_start
                    if self.config.verbose:
                        logger.debug("LLM+Tool: %.0fms", llm_time * 1000)
                    if self._vision_preview is not None:
                        self._vision_preview.add_timing("llm", llm_time * 1000)
                    if response_text:
                        self._speak_sentences(response_text)
                elif not response_text:
                    response_text = "(no response)"
            else:
                # LOCAL LLM NON-STREAMING FALLBACK
                import json as _json

                llm_kwargs: dict = {"context": context, "images": images}

                response = self.llm.generate(user_text, **llm_kwargs)
                response_text = response.text.strip()
                llm_time = time.perf_counter() - llm_start

                if self.config.verbose:
                    logger.debug("LLM: %.0fms", llm_time * 1000)

                # Check for prompt-based tool call
                tool_results = []
                if self._tools and response_text:
                    tool_result = self._try_parse_tool_call(response_text)
                    if tool_result is not None:
                        tool_call_json = response_text.strip()
                        tool_results.append(tool_result)
                        response_text = " ".join(tool_results) if tool_results else "Got it."

                # Also check native tool calls (unlikely)
                if not tool_results:
                    for tc in response.tool_calls:
                        result = self._tools.execute(tc)
                        if result:
                            tool_results.append(result)
                    if not response_text and response.tool_calls:
                        response_text = " ".join(tool_results) if tool_results else "Got it."

                # Speak response sentence-by-sentence (supports barge-in)
                self._speak_sentences(response_text)

            logger.info("Assistant: %s", response_text)
            self._update_preview(state="SPEAKING", vlm_text=response_text)
            if self._vision_preview is not None:
                self._vision_preview.add_transcript("assistant", response_text)

            # Update conversation history — when tools were called, store the
            # JSON tool call (not the result text) so the LLM sees the correct
            # output pattern in subsequent turns and keeps outputting JSON.
            self._get_conversation("local").append({"role": "user", "content": user_text})
            if tool_call_json:
                self._get_conversation("local").append({"role": "assistant", "content": tool_call_json})
            else:
                self._get_conversation("local").append({"role": "assistant", "content": response_text})

            # Callback
            if self.config.on_response:
                self.config.on_response(response_text)

            self._set_state(AssistantState.IDLE)
            self._update_preview(state="IDLE")

        except Exception as e:
            logger.error("Error processing speech: %s", e)
            self._set_state(AssistantState.ERROR)

            if self.config.play_chimes:
                self.audio_output.play_blocking(
                    self.chimes.error_chime(),
                    self.config.output_sample_rate,
                )

            if self.config.on_error:
                self.config.on_error(e)

            self._set_state(AssistantState.IDLE)

    def _speak_quick(self, text: str) -> None:
        """Speak a short acknowledgment immediately (no barge-in, blocking).

        Used for quick responses like "Sure, let me search for that."
        before starting a longer operation.
        """
        try:
            result = self.engine.synthesize(
                text,
                voice=self.config.tts_voice,
                language=self.config.tts_language,
            )
            self._notify_audio_playback(result.audio, result.sample_rate)
            self.audio_output.play_blocking(result.audio, result.sample_rate)
        except Exception:
            pass  # Don't let acknowledgment failures block the actual work

    def _tts_worker(self):
        """Background thread: synthesize and play queued sentences.

        Pulls (sentence, voice, lang) tuples from ``_tts_queue``, checks the
        TTS cache, synthesises on miss, and plays audio via
        ``audio_output.play_blocking``.  Barge-in is checked both before
        synthesis and before playback so interrupted turns are skipped quickly.

        Send ``None`` as a sentinel to shut the worker down.
        """
        while True:
            item = self._tts_queue.get()
            if item is None:
                self._tts_queue.task_done()
                break

            sentence, voice, lang = item

            try:
                # Skip if user barged in
                if self._bargein_event.is_set():
                    continue

                # Check TTS cache first
                cached = self._tts_cache.get(sentence, voice, lang)
                if cached:
                    audio_data, sample_rate = cached.audio, cached.sample_rate
                else:
                    tts_result = self.engine.synthesize(
                        sentence, voice=voice, language=lang,
                    )
                    audio_data = tts_result.audio
                    sample_rate = tts_result.sample_rate
                    self._tts_cache.put(sentence, voice, lang, audio_data, sample_rate)

                # Check barge-in again after (potentially slow) synthesis
                if self._bargein_event.is_set():
                    continue

                self.audio_output.play_blocking(audio_data, sample_rate)

                # Stream sentence to transcript overlay
                if self._vision_preview is not None:
                    self._vision_preview.add_transcript_stream(sentence)
            except Exception as exc:
                logger.error("TTS worker error: %s", exc)
            finally:
                self._tts_queue.task_done()

    def _speak_sentences(self, text: str) -> bool:
        """Split text into sentences, synthesize and play each one.

        Supports mid-sentence barge-in: if the user speaks loudly during
        playback, aplay is killed and this returns False immediately.

        Returns True if all sentences were spoken, False if interrupted.
        """
        import os
        import re
        import subprocess
        import tempfile
        from scipy.io import wavfile

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return True

        # Clear any stale barge-in flag
        self._bargein_event.clear()

        for sentence in sentences:
            # Check barge-in before synthesizing next sentence
            if self._bargein_event.is_set():
                logger.debug("Barge-in — stopping speech")
                return False

            # Stream sentence to transcript page
            if self._vision_preview is not None:
                self._vision_preview.add_transcript_stream(sentence)

            # Check TTS cache before synthesizing
            voice = self.config.tts_voice
            lang = self.config.tts_language
            cached = self._tts_cache.get(sentence, voice, lang)
            if cached:
                audio = cached.audio
                sr = cached.sample_rate
            else:
                tts_result = self.engine.synthesize(
                    sentence,
                    voice=voice,
                    language=lang,
                )
                audio = tts_result.audio
                sr = tts_result.sample_rate
                self._tts_cache.put(sentence, voice, lang, audio, sr)

            # Play with interruptible aplay (Popen, not blocking run)

            if audio.dtype != np.int16:
                if audio.dtype in (np.float32, np.float64):
                    audio = (audio * 32767).astype(np.int16)
                else:
                    audio = audio.astype(np.int16)

            # Feed audio to external tool plugins for motion sync
            self._notify_audio_playback(audio, sr)

            tmpf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wavfile.write(tmpf.name, sr, audio)
            tmpf.close()

            try:
                proc = subprocess.Popen(["aplay", "-q", tmpf.name])
                self._aplay_proc = proc
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait()
            finally:
                self._aplay_proc = None
                os.unlink(tmpf.name)

            # Check if barge-in killed aplay
            if self._bargein_event.is_set():
                logger.debug("Barge-in — stopping speech")
                return False

        return True

    def say(self, text: str) -> None:
        """
        Make the assistant say something.

        Args:
            text: Text to speak
        """
        prev_state = self.state
        self._set_state(AssistantState.SPEAKING)

        result = self.engine.synthesize(
            text,
            voice=self.config.tts_voice,
            language=self.config.tts_language,
        )
        self._notify_audio_playback(result.audio, result.sample_rate)
        self.audio_output.play_blocking(result.audio, result.sample_rate)

        self._set_state(prev_state)

    def listen_once(self, timeout: float = 10.0) -> Optional[str]:
        """
        Listen for speech once (without wake word).

        Args:
            timeout: Maximum listen time

        Returns:
            Transcribed text or None
        """
        logger.info("Listening...")

        if self.config.play_chimes:
            self.audio_output.play_blocking(
                self.chimes.listening_chime(),
                self.config.output_sample_rate,
            )

        audio_buffer = []
        start_time = time.time()
        done = threading.Event()

        def on_chunk(audio):
            if done.is_set():
                return
            audio_buffer.append(audio.copy())
            if time.time() - start_time > timeout:
                done.set()
            elif self.vad.detect_end_of_speech(
                audio,
                silence_threshold_ms=self.config.silence_timeout_ms,
                chunk_duration_ms=self.audio_config.chunk_duration_ms,
            ):
                done.set()

        audio_input = AudioInput(config=self.audio_config, device=self.config.audio_input_device)
        audio_input.start(on_chunk)

        done.wait(timeout=timeout + 1)
        audio_input.stop()

        if not audio_buffer:
            return None

        audio = np.concatenate(audio_buffer)
        result = self._transcribe(audio, sample_rate=self.config.input_sample_rate)
        return result.text.strip() if result.text else None

    def ask(self, question: str) -> Optional[str]:
        """
        Ask a question and get voice response.

        Args:
            question: Question to ask

        Returns:
            User's transcribed response
        """
        self.say(question)
        return self.listen_once()

    def get_conversation_history(self, session_id: str = "local") -> list[dict]:
        """Get conversation history for a session."""
        return self._get_conversation(session_id).copy()

    def clear_conversation(self, session_id: str | None = None) -> None:
        """Clear conversation history. If session_id is None, clears all sessions."""
        if session_id is None:
            self._conversations.clear()
        elif session_id in self._conversations:
            del self._conversations[session_id]


def run_assistant(
    tts_backend: str = "qwen",
    stt_backend: str = "whisper",
    llm_backend: str = "ollama",
    llm_model: str = "llama3.2:3b",
    wake_word: str = "hey_jarvis",
    tts_voice: str = "serena",
    verbose: bool = False,
) -> None:
    """
    Quick start function to run the voice assistant.

    Args:
        tts_backend: TTS backend ("qwen" or "piper")
        stt_backend: STT backend ("whisper")
        llm_backend: LLM backend ("ollama", "openai", "anthropic", "simple")
        llm_model: LLM model name
        wake_word: Wake word to listen for
        tts_voice: TTS voice name
        verbose: Show timing info
    """
    from jetson_assistant import Engine

    # Create and configure engine
    engine = Engine()

    logger.info("Loading TTS backend...")
    engine.load_tts_backend(tts_backend)

    logger.info("Loading STT backend...")
    engine.load_stt_backend(stt_backend)

    # Configure assistant
    config = AssistantConfig(
        wake_word=wake_word,
        llm_backend=llm_backend,
        llm_model=llm_model,
        tts_backend=tts_backend,
        tts_voice=tts_voice,
        verbose=verbose,
    )

    # Create and run assistant
    assistant = VoiceAssistant(engine, config)
    assistant.run()
