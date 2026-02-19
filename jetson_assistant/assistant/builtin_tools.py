"""
Builtin tools for the voice assistant.

Extracted from core.py to keep the assistant class focused on the conversation
loop and state machine. All tools follow the same register pattern used by
external tool plugins: a top-level ``register_builtin_tools(registry, context)``
function that registers closures with the ToolRegistry.

Context dict keys consumed by builtin tools:

    say             callable(str)           Speak text aloud
    engine          Engine                  TTS/STT engine (for set_language)
    config          AssistantConfig         Assistant config (for set_language)
    llm             LLMBackend | None       LLM for vision analysis
    camera          Camera | None           Local USB camera
    camera_lock     threading.Lock          Lock for camera access
    camera_pool     CameraPool | None       Multi-camera pool
    vision_monitor  VisionMonitor | None    Single-camera watch (legacy)
    multi_watch     MultiWatchMonitor|None  Multi-camera watch
    update_preview  callable(**kwargs)      Update vision preview overlay
    set_language_instruction  callable(str) Set language instruction on assistant
    apply_tool_system_prompt  callable()    Reapply tool system prompt after changes
    knowledge_collection      str | None    RAG collection name
"""

import logging
import threading
from typing import Annotated

logger = logging.getLogger(__name__)

from jetson_assistant.assistant.tools import ToolRegistry


def register_builtin_tools(registry: ToolRegistry, context: dict) -> None:
    """Register all builtin tools with the given registry.

    Args:
        registry: ToolRegistry instance to register tools on.
        context: Dict of assistant state and helpers that tools need.
    """
    _register_always_available(registry, context)
    _register_camera_tools(registry, context)
    _register_vision_monitor_tools(registry, context)
    _register_multi_camera_tools(registry, context)
    _register_multi_watch_tools(registry, context)
    _register_knowledge_tools(registry, context)


# ---------------------------------------------------------------------------
# Always-available tools
# ---------------------------------------------------------------------------

def _register_always_available(registry: ToolRegistry, context: dict) -> None:
    """Register tools that are always available regardless of hardware."""

    say = context["say"]
    engine = context["engine"]
    config = context["config"]
    set_language_instruction = context["set_language_instruction"]
    apply_tool_system_prompt = context["apply_tool_system_prompt"]

    @registry.register(
        "Get the current date and time. Use when the user asks what time it is, "
        "what today's date is, or anything about the current time."
    )
    def get_time() -> str:
        from datetime import datetime

        now = datetime.now()
        return now.strftime("It's %I:%M %p on %A, %B %d, %Y.").replace(" 0", " ")

    @registry.register(
        "Get system stats like uptime, memory usage, and CPU temperature. "
        "Use when the user asks about system health, how the Jetson is doing, "
        "or wants hardware status."
    )
    def system_stats() -> str:
        import os

        parts = []
        # Uptime
        try:
            with open("/proc/uptime") as f:
                secs = float(f.read().split()[0])
            hours = int(secs // 3600)
            mins = int((secs % 3600) // 60)
            parts.append(f"Uptime {hours} hours {mins} minutes")
        except OSError:
            pass
        # Memory
        try:
            with open("/proc/meminfo") as f:
                meminfo = {}
                for line in f:
                    k, v = line.split(":")
                    meminfo[k.strip()] = int(v.strip().split()[0])
            total_mb = meminfo["MemTotal"] // 1024
            avail_mb = meminfo.get("MemAvailable", meminfo.get("MemFree", 0)) // 1024
            used_mb = total_mb - avail_mb
            pct = int(used_mb / total_mb * 100) if total_mb else 0
            parts.append(f"Memory: {used_mb}MB of {total_mb}MB used, {pct}%")
        except (OSError, KeyError, ValueError):
            pass
        # CPU temperature
        try:
            thermal_dir = "/sys/devices/virtual/thermal"
            for entry in os.listdir(thermal_dir):
                if entry.startswith("thermal_zone"):
                    temp_path = os.path.join(thermal_dir, entry, "temp")
                    with open(temp_path) as f:
                        temp_c = int(f.read().strip()) / 1000
                    parts.append(f"CPU temperature {temp_c:.0f}C")
                    break
        except (OSError, ValueError):
            pass
        return ". ".join(parts) + "." if parts else "Could not read system stats."

    @registry.register(
        "Set a countdown timer. Use when the user asks to set a timer, "
        "reminder, or countdown for a number of seconds."
    )
    def set_timer(
        seconds: Annotated[int, "Number of seconds for the timer"],
    ) -> str:
        def _timer_done():
            say(f"Time's up! Your {seconds} second timer is done.")

        t = threading.Timer(seconds, _timer_done)
        t.daemon = True
        t.start()
        return f"Timer set for {seconds} seconds."

    @registry.register(
        "Remember a piece of information for the user. Use when the user asks "
        "you to remember, save, or note something."
    )
    def remember(
        info: Annotated[str, "The information to remember"],
    ) -> str:
        import json
        from pathlib import Path

        mem_path = Path.home() / ".assistant_memory.json"
        memories: list[str] = []
        if mem_path.exists():
            try:
                memories = json.loads(mem_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        memories.append(info)
        mem_path.write_text(json.dumps(memories, indent=2))
        return "I'll remember that."

    @registry.register(
        "Recall previously remembered information. Use when the user asks what "
        "you remember, or asks about something they previously told you to remember."
    )
    def recall(
        query: Annotated[str, (
            "What to search for. Use 'all' to get everything, "
            "or a keyword to search memories."
        )] = "all",
    ) -> str:
        import json
        from pathlib import Path

        mem_path = Path.home() / ".assistant_memory.json"
        if not mem_path.exists():
            return "I don't have any memories saved yet."
        try:
            memories: list[str] = json.loads(mem_path.read_text())
        except (json.JSONDecodeError, OSError):
            return "I don't have any memories saved yet."
        if not memories:
            return "I don't have any memories saved yet."

        if query.lower() in ("all", "everything", ""):
            numbered = [f"{i+1}. {m}" for i, m in enumerate(memories)]
            return "Here's what I remember: " + " ".join(numbered)

        matches = [m for m in memories if query.lower() in m.lower()]
        if not matches:
            return f"I don't remember anything about '{query}'."
        numbered = [f"{i+1}. {m}" for i, m in enumerate(matches)]
        return "Here's what I found: " + " ".join(numbered)

    @registry.register(
        "Switch the assistant's spoken language. Use when the user asks you to "
        "speak in a different language, respond in Hindi, switch to English, "
        "talk in Japanese, etc. Supported: english, hindi, japanese, chinese, "
        "french, spanish, portuguese."
    )
    def set_language(
        language: Annotated[str, (
            "Language to switch to: 'english', 'hindi', 'japanese', 'chinese', "
            "'french', 'spanish', 'portuguese'"
        )],
    ) -> str:
        lang_map = {
            "english": "en", "hindi": "hi", "japanese": "ja",
            "chinese": "zh", "korean": "ko", "french": "fr",
            "spanish": "es", "portuguese": "pt", "british": "en-gb",
            # Also accept codes directly
            "en": "en", "hi": "hi", "ja": "ja", "zh": "zh",
            "ko": "ko", "fr": "fr", "es": "es", "pt": "pt",
        }

        # Language instructions for the LLM (tells it to generate text in target language)
        lang_instructions = {
            "en": "",  # English = default, no extra instruction
            "en-gb": "",
            "hi": "IMPORTANT: You MUST respond in Hindi using Devanagari script. All your responses must be in Hindi, not English. Tool calls remain in JSON.",
            "ja": "IMPORTANT: You MUST respond in Japanese. All your responses must be in Japanese. Tool calls remain in JSON.",
            "zh": "IMPORTANT: You MUST respond in Chinese. All your responses must be in Chinese. Tool calls remain in JSON.",
            "ko": "IMPORTANT: You MUST respond in Korean. All your responses must be in Korean. Tool calls remain in JSON.",
            "fr": "IMPORTANT: You MUST respond in French. All your responses must be in French. Tool calls remain in JSON.",
            "es": "IMPORTANT: You MUST respond in Spanish. All your responses must be in Spanish. Tool calls remain in JSON.",
            "pt": "IMPORTANT: You MUST respond in Portuguese. All your responses must be in Portuguese. Tool calls remain in JSON.",
        }

        # Native-language confirmations (spoken by TTS in the target language)
        lang_confirmations = {
            "en": "Switched to English.",
            "en-gb": "Switched to British English.",
            "hi": "\u0920\u0940\u0915 \u0939\u0948, \u0905\u092c \u092e\u0948\u0902 \u0939\u093f\u0902\u0926\u0940 \u092e\u0947\u0902 \u092c\u094b\u0932\u0942\u0901\u0917\u0940\u0964",
            "ja": "\u306f\u3044\u3001\u65e5\u672c\u8a9e\u3067\u8a71\u3057\u307e\u3059\u3002",
            "zh": "\u597d\u7684\uff0c\u6211\u73b0\u5728\u8bf4\u4e2d\u6587\u3002",
            "ko": "\ub124, \uc774\uc81c \ud55c\uad6d\uc5b4\ub85c \ub9d0\ud560\uac8c\uc694.",
            "fr": "D'accord, je parle maintenant en fran\u00e7ais.",
            "es": "De acuerdo, ahora hablo en espa\u00f1ol.",
            "pt": "Ok, agora vou falar em portugu\u00eas.",
        }

        lang_code = lang_map.get(language.lower().strip())
        if lang_code is None:
            return f"Language '{language}' not supported. Try: english, hindi, japanese, chinese, korean, french, spanish, portuguese."

        backend = engine._tts_backend
        if backend is None or not hasattr(backend, "switch_language"):
            return "Language switching requires Kokoro TTS backend."

        try:
            backend.switch_language(lang_code)
            # Update config so subsequent synthesize calls use the new voice
            config.tts_voice = backend._current_voice
            config.tts_language = lang_code

            # Update LLM system prompt to generate text in the target language
            set_language_instruction(lang_instructions.get(lang_code, ""))
            apply_tool_system_prompt()

            return lang_confirmations.get(lang_code, f"Switched to {language}.")
        except Exception as e:
            return f"Failed to switch language: {e}"

    @registry.register(
        "Search the web for current information. MANDATORY for any question about "
        "news, latest, updates, recent events, what happened, or any company/person "
        "updates. Your training data is outdated -- NEVER answer news questions from "
        "memory. ALWAYS use this tool for: latest, news, update, recent, current."
    )
    def web_search(
        query: Annotated[str, "The search query"],
    ) -> str:
        # Try new 'ddgs' package first, then legacy 'duckduckgo_search'
        DDGS = None
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                return (
                    "Web search is not available. "
                    "Install with: pip install ddgs"
                )

        try:
            from datetime import datetime
            date_str = datetime.now().strftime("%B %Y")

            ddgs = DDGS()

            # Text search is primary — better for factual queries (medal counts,
            # scores, "who won X"). Append date for recency.
            results = list(ddgs.text(f"{query} {date_str}", max_results=5))

            # Fallback to news search if text returns nothing
            if not results:
                results = list(ddgs.news(query, max_results=5))

            if not results:
                return f"No results found for '{query}'."

            # Return top 3 results so the LLM summarizer has enough facts.
            # The summarization step in core.py condenses this for TTS.
            parts = []
            for r in results[:3]:
                title = r.get("title", "")
                body = r.get("body", r.get("description", ""))
                if body:
                    parts.append(f"{title}: {body}")
                elif title:
                    parts.append(title)
            return "\n".join(parts) if parts else f"No useful results for '{query}'."
        except Exception as e:
            return f"Search failed: {e}"


# ---------------------------------------------------------------------------
# Camera tools (require a local USB camera)
# ---------------------------------------------------------------------------

def _register_camera_tools(registry: ToolRegistry, context: dict) -> None:
    """Register tools that need a local camera."""
    camera = context.get("camera")
    if camera is None:
        return

    camera_lock = context["camera_lock"]

    @registry.register(
        "Take a photo with the camera and save it. Use when the user asks "
        "to take a picture, photo, snapshot, or capture an image."
    )
    def take_photo() -> str:
        import os
        from datetime import datetime

        try:
            import cv2
        except ImportError:
            return "Camera not available -- opencv not installed."

        photos_dir = os.path.expanduser("~/photos")
        os.makedirs(photos_dir, exist_ok=True)

        with camera_lock:
            frame = camera.capture_frame()
        if frame is None:
            return "Failed to capture a photo."

        filename = f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(photos_dir, filename)
        cv2.imwrite(filepath, frame)
        return f"Photo saved as {filename}"


# ---------------------------------------------------------------------------
# Single-camera vision monitor tools (legacy, used when multi-watch absent)
# ---------------------------------------------------------------------------

def _register_vision_monitor_tools(registry: ToolRegistry, context: dict) -> None:
    """Register legacy single-camera watch tools."""
    vision_monitor = context.get("vision_monitor")
    multi_watch = context.get("multi_watch")

    # Only register legacy tools when multi-watch is not available
    if vision_monitor is None or multi_watch is not None:
        return

    vm = vision_monitor
    update_preview = context["update_preview"]

    @registry.register(
        "Start continuously monitoring the camera and alert the user when a "
        "condition is detected. Use this when the user asks you to watch for "
        "something, keep an eye on something, monitor something, or alert/notify "
        "them about a visual event."
    )
    def start_watching(
        condition: Annotated[str, (
            "A yes/no question to periodically check against the camera "
            "feed. Example: 'Is someone taking the candies?'"
        )],
    ) -> None:
        from jetson_assistant.assistant.vision import WatchCondition

        wc = WatchCondition(
            description=condition,
            prompt=f"{condition} Answer only YES or NO.",
            announce_template="Alert: the condition you asked about has been detected!",
        )
        vm.start_watching(wc)
        update_preview(state="IDLE", watch_text=f"monitoring: {condition[:40]}")

    @registry.register(
        "Stop the current camera monitoring. Use when the user asks to stop "
        "watching, cancel monitoring, or no longer needs alerts."
    )
    def stop_watching() -> None:
        if vm.is_watching:
            vm.stop_watching()
            update_preview(state="IDLE", watch_text="")


# ---------------------------------------------------------------------------
# Multi-camera tools (camera pool)
# ---------------------------------------------------------------------------

def _register_multi_camera_tools(registry: ToolRegistry, context: dict) -> None:
    """Register tools for multi-camera pool management and vision."""
    camera_pool = context.get("camera_pool")
    if camera_pool is None:
        return

    pool = camera_pool
    llm = context.get("llm")

    @registry.register(
        "List all available cameras and their locations. Use when the user "
        "asks what cameras they have, which cameras are connected, or wants "
        "to see available camera names."
    )
    def list_cameras() -> str:
        cameras = pool.list_cameras()
        if not cameras:
            return "No cameras configured."
        parts = []
        for cam in cameras:
            loc = f" ({cam.location})" if cam.location else ""
            parts.append(f"{cam.name}{loc}")
        return f"You have {len(cameras)} cameras: {', '.join(parts)}."

    @registry.register(
        "Check a specific camera by name and describe what you see. Use when "
        "the user asks what's happening at a specific camera, wants to see a "
        "camera feed, or asks about a specific location's camera. "
        "Example: 'What's happening at the front door?'"
    )
    def check_camera(
        camera_name: Annotated[str, "Name of the camera to check (e.g., 'garage', 'front_door')"],
        question: Annotated[str, "What to look for or describe (e.g., 'Is anyone there?', 'Describe what you see')"] = "Describe what you see in this image. Be specific and concise.",
    ) -> str:
        if not pool.has(camera_name):
            cameras = pool.list_cameras()
            if len(cameras) == 1:
                camera_name = cameras[0].name
            else:
                available = ", ".join(s.name for s in cameras)
                return f"Camera '{camera_name}' not found. Available: {available}."

        frame_b64 = pool.capture_base64(camera_name)
        if frame_b64 is None:
            return f"Failed to capture frame from '{camera_name}'. Camera may be offline."

        # Use VLM to analyze the frame
        if llm is None:
            return "No LLM available for image analysis."

        try:
            response = llm.generate(
                question,
                images=[frame_b64],
            )
            return response.text.strip()
        except Exception as e:
            return f"Vision analysis failed: {e}"

    @registry.register(
        "Add a new camera by name and URL. Use when the user wants to register "
        "a new camera. URL can be RTSP (rtsp://...) or USB (usb:0)."
    )
    def add_camera(
        name: Annotated[str, "Name for the camera (e.g., 'patio', 'nursery')"],
        url: Annotated[str, "Camera URL: RTSP (rtsp://ip:port/path) or USB (usb:0)"],
    ) -> str:
        return pool.add(name, url)

    @registry.register(
        "Remove a camera by name. Use when the user wants to delete or "
        "unregister a camera."
    )
    def remove_camera(
        name: Annotated[str, "Name of the camera to remove"],
    ) -> str:
        return pool.remove(name)


# ---------------------------------------------------------------------------
# Multi-camera watch tools
# ---------------------------------------------------------------------------

def _rewrite_condition_to_state_based(condition: str) -> tuple[str, bool]:
    """Rewrite action-based watch conditions to state-based ones.

    The VLM polls every ~5s so it can't catch brief actions like a hand
    grabbing something. Instead we detect the resulting state change.

    Returns (rewritten_condition, invert). When invert=True, the watch
    triggers when the VLM says NO (absence detection).

    'Is someone taking the candy?' → ('Is there candy visible?', True)
      → triggers when VLM says NO (candy is gone)
    'Is someone opening the door?' → ('Is the door open?', False)
    """
    import re

    c = condition.strip()

    # Pattern: any sentence about taking/stealing/removing/grabbing an object
    # Broad match — handles "taking", "trying to take", "going to steal", etc.
    # → Ask if object is VISIBLE, trigger on NO (absence = someone took it)
    m = re.search(
        r"(?:(?:trying|going|about)\s+to\s+)?"
        r"(?:tak(?:e|ing)|steal(?:ing)?|remov(?:e|ing)|grab(?:bing)?|eat(?:ing)?|snatch(?:ing)?)\s+"
        r"(?:the\s+|my\s+|our\s+|a\s+)?(.+?)[\?\.]?\s*$",
        c, re.IGNORECASE,
    )
    if m:
        obj = m.group(1).rstrip("?. ")
        rewritten = f"Is there {obj} visible in this image?"
        logger.info("Condition rewrite: %r → %r (inverted)", c, rewritten)
        return rewritten, True

    # Pattern: "Is someone opening/closing the [object]?"
    m = re.match(
        r"(?:Is|Are)\s+(?:someone|anyone|somebody)\s+"
        r"(opening|closing)\s+"
        r"(?:the\s+|my\s+)?(.+?)[\?\.]?\s*$",
        c, re.IGNORECASE,
    )
    if m:
        action, obj = m.group(1).lower(), m.group(2).rstrip("?. ")
        state = "open" if action == "opening" else "closed"
        rewritten = f"Is the {obj} {state}?"
        logger.info("Condition rewrite: %r → %r", c, rewritten)
        return rewritten, False

    # Pattern: "Is someone arriving/entering/leaving?"
    m = re.match(
        r"(?:Is|Are)\s+(?:someone|anyone|somebody|there someone)\s+"
        r"(arriving|entering|coming|leaving|departing|walking)",
        c, re.IGNORECASE,
    )
    if m:
        rewritten = "Is there a person visible in the scene?"
        logger.info("Condition rewrite: %r → %r", c, rewritten)
        return rewritten, False

    return condition, False


def _register_multi_watch_tools(registry: ToolRegistry, context: dict) -> None:
    """Register tools for multi-camera watching/monitoring."""
    multi_watch = context.get("multi_watch")
    if multi_watch is None:
        return

    mw = multi_watch
    update_preview = context["update_preview"]

    @registry.register(
        "Start watching a specific camera and alert when a condition is detected. "
        "Use when the user asks to watch a camera for something, monitor a location, "
        "or get alerts from a specific camera. Each camera can have one active watch. "
        "Example: 'Watch the garage and tell me when the door opens'"
    )
    def watch_camera(
        camera_name: Annotated[str, "Name of the camera to watch (e.g., 'garage')"],
        condition: Annotated[str, (
            "A yes/no question to periodically check. "
            "Example: 'Is the garage door open?'"
        )],
    ) -> str:
        from jetson_assistant.assistant.vision import WatchCondition

        # Rewrite action-based conditions to state-based ones.
        # The VLM polls every 5s — it can't catch brief actions like
        # "someone taking X" but CAN detect the resulting state "X is gone".
        original_condition = condition
        condition, invert = _rewrite_condition_to_state_based(condition)

        wc = WatchCondition(
            description=original_condition,
            prompt=f"{condition} Answer only YES or NO.",
            announce_template=f"Alert on {camera_name}: {original_condition}",
            invert=invert,
        )
        result = mw.start_watching(camera_name, wc)
        update_preview(
            state="IDLE",
            watch_text=f"watching {camera_name}: {condition[:30]}",
        )
        return result

    @registry.register(
        "Stop watching a camera or all cameras. Use when the user asks to "
        "stop monitoring, cancel a watch, or stop all alerts. "
        "Use camera_name='all' to stop everything."
    )
    def stop_watching_camera(
        camera_name: Annotated[str, (
            "Camera to stop watching, or 'all' to stop all watches"
        )] = "all",
    ) -> str:
        result = mw.stop_watching(camera_name)
        update_preview(state="IDLE", watch_text="")
        return result

    @registry.register(
        "List all active camera watches. Use when the user asks what you're "
        "currently watching or monitoring."
    )
    def list_watches() -> str:
        watches = mw.list_watches()
        if not watches:
            return "No active watches."
        parts = []
        for w in watches:
            status = "active" if w["active"] else "stopped"
            parts.append(f"{w['camera']}: {w['condition']} ({status})")
        return f"Active watches: {'; '.join(parts)}"


# ---------------------------------------------------------------------------
# Knowledge base tool (RAG)
# ---------------------------------------------------------------------------

def _register_knowledge_tools(registry: ToolRegistry, context: dict) -> None:
    """Register knowledge base lookup tool if a collection is configured."""
    knowledge_collection = context.get("knowledge_collection")
    if not knowledge_collection:
        return

    collection_name = knowledge_collection

    @registry.register(
        "Look up personal or specific information from the knowledge base. "
        "Use this when the user asks about personal details, contacts, family info, "
        "birthdays, phone numbers, preferences, schedules, or any factual info that "
        "would have been stored in advance. Also use this for any domain-specific "
        "knowledge that has been loaded into the knowledge base."
    )
    def lookup_info(
        query: Annotated[str, "What to look up (e.g., 'mom birthday', 'dentist phone number', 'wifi password')"],
    ) -> str:
        try:
            from jetson_assistant.rag.pipeline import RAGPipeline
        except ImportError:
            return (
                "Knowledge base not available. "
                "Install with: pip install chromadb sentence-transformers"
            )

        try:
            rag = RAGPipeline(collection_name)
            results = rag.retrieve(query, top_k=3)

            if not results:
                return f"No information found for '{query}'."

            # Filter low-confidence results
            good_results = [r for r in results if r.get("score", 0) > 0.3]

            if not good_results:
                return f"No relevant information found for '{query}'."

            parts = []
            for r in good_results:
                parts.append(r["content"])
            return " | ".join(parts)

        except Exception as e:
            return f"Knowledge base lookup failed: {e}"
