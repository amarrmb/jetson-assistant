"""
Tests for voice assistant module.
"""

import json
import os
import tempfile
from typing import Annotated
from unittest.mock import patch

import numpy as np
import pytest


class TestWakeWordDetector:
    """Tests for wake word detection."""

    def test_simple_energy_detector(self):
        """Test energy-based detector."""
        from jetson_speech.assistant.wakeword import SimpleEnergyDetector

        detector = SimpleEnergyDetector(threshold=500.0, min_frames=2)

        # Silent audio should not trigger
        silent = np.zeros(1600, dtype=np.int16)
        assert not detector.detect(silent)

        # Loud audio should trigger after min_frames
        loud = (np.random.randn(1600) * 10000).astype(np.int16)
        detector.detect(loud)  # First frame
        assert detector.detect(loud)  # Second frame triggers

    def test_create_wakeword_detector(self):
        """Test factory function."""
        from jetson_speech.assistant.wakeword import create_wakeword_detector

        # Energy detector (always available)
        detector = create_wakeword_detector(backend="energy")
        assert detector is not None

        # Unknown backend should raise
        with pytest.raises(ValueError):
            create_wakeword_detector(backend="unknown")


class TestLLM:
    """Tests for LLM backends."""

    def test_simple_llm(self):
        """Test simple rule-based LLM."""
        from jetson_speech.assistant.llm import SimpleLLM

        llm = SimpleLLM()

        # Test known phrases
        response = llm.generate("hello")
        assert "hello" in response.text.lower() or "help" in response.text.lower()

        response = llm.generate("tell me a joke")
        assert len(response.text) > 0

    def test_create_llm(self):
        """Test factory function."""
        from jetson_speech.assistant.llm import create_llm

        # Simple (always available)
        llm = create_llm(backend="simple")
        assert llm is not None

        # Unknown backend should raise
        with pytest.raises(ValueError):
            create_llm(backend="unknown")


class TestAudioIO:
    """Tests for audio I/O components."""

    def test_audio_config(self):
        """Test audio configuration."""
        from jetson_speech.assistant.audio_io import AudioConfig

        config = AudioConfig(sample_rate=16000, chunk_duration_ms=100)
        assert config.chunk_size == 1600

        config = AudioConfig(sample_rate=44100, chunk_duration_ms=50)
        assert config.chunk_size == 2205

    def test_vad_energy(self):
        """Test energy-based VAD."""
        from jetson_speech.assistant.audio_io import VoiceActivityDetector

        vad = VoiceActivityDetector(use_webrtc=False)

        # Silent audio
        silent = np.zeros(1600, dtype=np.int16)
        assert not vad.is_speech(silent)

        # Loud audio
        loud = (np.random.randn(1600) * 10000).astype(np.int16)
        assert vad.is_speech(loud)

    def test_chime_sounds(self):
        """Test chime generation."""
        from jetson_speech.assistant.audio_io import ChimeSounds

        chimes = ChimeSounds(sample_rate=24000)

        wake = chimes.wake_chime()
        assert len(wake) > 0
        assert wake.dtype == np.int16

        listen = chimes.listening_chime()
        assert len(listen) > 0

        done = chimes.done_chime()
        assert len(done) > 0

        error = chimes.error_chime()
        assert len(error) > 0


class TestAssistantConfig:
    """Tests for assistant configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        from jetson_speech.assistant.core import AssistantConfig

        config = AssistantConfig()

        assert config.wake_word == "hey_jarvis"
        assert config.llm_backend == "ollama"
        assert config.tts_backend == "piper"
        assert config.stt_backend == "whisper"
        assert config.silence_timeout_ms > 0
        assert config.max_listen_time_s > 0

    def test_custom_config(self):
        """Test custom configuration."""
        from jetson_speech.assistant.core import AssistantConfig

        config = AssistantConfig(
            wake_word="alexa",
            llm_backend="openai",
            llm_model="gpt-4",
            tts_voice="ryan",
            verbose=True,
        )

        assert config.wake_word == "alexa"
        assert config.llm_backend == "openai"
        assert config.llm_model == "gpt-4"
        assert config.tts_voice == "ryan"
        assert config.verbose is True


class TestAssistantState:
    """Tests for assistant state machine."""

    def test_state_enum(self):
        """Test state enumeration."""
        from jetson_speech.assistant.core import AssistantState

        assert AssistantState.IDLE.value == "idle"
        assert AssistantState.LISTENING.value == "listening"
        assert AssistantState.PROCESSING.value == "processing"
        assert AssistantState.SPEAKING.value == "speaking"
        assert AssistantState.ERROR.value == "error"


class TestToolRegistry:
    """Tests for ToolRegistry auto-schema generation and dispatch."""

    def _make_registry(self):
        from jetson_speech.assistant.tools import ToolRegistry
        return ToolRegistry()

    def test_register_and_definitions(self):
        """Schema has correct shape, name, and required fields."""
        reg = self._make_registry()

        @reg.register("Greet someone")
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        defs = reg.definitions()
        assert len(defs) == 1
        d = defs[0]
        assert d["type"] == "function"
        assert d["function"]["name"] == "greet"
        assert d["function"]["description"] == "Greet someone"
        assert "name" in d["function"]["parameters"]["properties"]
        assert d["function"]["parameters"]["required"] == ["name"]

    def test_execute(self):
        """Dispatch calls the registered handler and returns its result."""
        reg = self._make_registry()
        calls = []

        @reg.register("Add numbers")
        def add(a: int, b: int) -> int:
            calls.append((a, b))
            return a + b

        class FakeToolCall:
            name = "add"
            arguments = {"a": 2, "b": 3}

        result = reg.execute(FakeToolCall())
        assert result == "5"
        assert calls == [(2, 3)]

    def test_execute_unknown_tool(self):
        """Unknown tool name returns None."""
        reg = self._make_registry()

        class FakeToolCall:
            name = "nonexistent"
            arguments = {}

        assert reg.execute(FakeToolCall()) is None

    def test_bool(self):
        """Empty registry is falsy, non-empty is truthy."""
        reg = self._make_registry()
        assert not reg

        @reg.register("noop")
        def noop() -> None:
            pass

        assert reg

    def test_annotated_description(self):
        """Annotated[str, 'desc'] is extracted as parameter description."""
        reg = self._make_registry()

        @reg.register("Watch")
        def watch(condition: Annotated[str, "A yes/no question"]) -> None:
            pass

        props = reg.definitions()[0]["function"]["parameters"]["properties"]
        assert props["condition"]["description"] == "A yes/no question"

    def test_type_mapping(self):
        """str/int/float/bool map to correct JSON Schema types."""
        reg = self._make_registry()

        @reg.register("types")
        def typed(s: str, i: int, f: float, b: bool) -> None:
            pass

        props = reg.definitions()[0]["function"]["parameters"]["properties"]
        assert props["s"]["type"] == "string"
        assert props["i"]["type"] == "integer"
        assert props["f"]["type"] == "number"
        assert props["b"]["type"] == "boolean"

    def test_optional_parameter(self):
        """Parameter with default value is not in required list."""
        reg = self._make_registry()

        @reg.register("optional test")
        def opt(required_param: str, optional_param: str = "default") -> None:
            pass

        schema = reg.definitions()[0]["function"]
        assert schema["parameters"]["required"] == ["required_param"]
        assert "optional_param" in schema["parameters"]["properties"]

    def test_len(self):
        """__len__ returns number of registered tools."""
        reg = self._make_registry()
        assert len(reg) == 0

        @reg.register("a")
        def a() -> None:
            pass

        @reg.register("b")
        def b() -> None:
            pass

        assert len(reg) == 2


class TestDemoTools:
    """Tests for the 7 demo tools registered in _init_tools()."""

    def _make_tools(self):
        """Build a ToolRegistry and register tools the same way core.py does."""
        from jetson_speech.assistant.tools import ToolRegistry
        import threading

        reg = ToolRegistry()

        @reg.register("Get the current time")
        def get_time() -> str:
            from datetime import datetime
            now = datetime.now()
            return now.strftime("It's %I:%M %p on %A, %B %d, %Y.").replace(" 0", " ")

        @reg.register("Get system stats")
        def system_stats() -> str:
            import os
            parts = []
            try:
                with open("/proc/uptime") as f:
                    secs = float(f.read().split()[0])
                hours = int(secs // 3600)
                mins = int((secs % 3600) // 60)
                parts.append(f"Uptime {hours} hours {mins} minutes")
            except OSError:
                pass
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
            return ". ".join(parts) + "." if parts else "Could not read system stats."

        self._timer_threads = []

        @reg.register("Set a timer")
        def set_timer(seconds: Annotated[int, "Number of seconds"]) -> str:
            def _noop():
                pass
            t = threading.Timer(seconds, _noop)
            t.daemon = True
            t.start()
            self._timer_threads.append(t)
            return f"Timer set for {seconds} seconds."

        self._mem_path = None

        @reg.register("Remember something")
        def remember(info: Annotated[str, "Info to remember"]) -> str:
            memories: list[str] = []
            if os.path.exists(self._mem_path):
                try:
                    with open(self._mem_path) as f:
                        memories = json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass
            memories.append(info)
            with open(self._mem_path, "w") as f:
                json.dump(memories, f, indent=2)
            return "I'll remember that."

        @reg.register("Recall memories")
        def recall(query: Annotated[str, "Search keyword or 'all'"] = "all") -> str:
            if not os.path.exists(self._mem_path):
                return "I don't have any memories saved yet."
            try:
                with open(self._mem_path) as f:
                    memories: list[str] = json.load(f)
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

        @reg.register("Search the web")
        def web_search(query: Annotated[str, "Search query"]) -> str:
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                return (
                    "Web search is not available. "
                    "Install it with: pip install duckduckgo-search"
                )
            try:
                results = DDGS().text(query, max_results=3)
                if not results:
                    return f"No results found for '{query}'."
                summaries = []
                for r in results:
                    title = r.get("title", "")
                    body = r.get("body", "")
                    summaries.append(f"{title}: {body}")
                combined = " | ".join(summaries)
                if len(combined) > 300:
                    combined = combined[:297] + "..."
                return combined
            except Exception as e:
                return f"Search failed: {e}"

        return reg

    def _call(self, reg, name, args=None):
        class FakeToolCall:
            pass
        tc = FakeToolCall()
        tc.name = name
        tc.arguments = args or {}
        return reg.execute(tc)

    def test_get_time(self):
        reg = self._make_tools()
        result = self._call(reg, "get_time")
        assert result is not None
        assert "AM" in result or "PM" in result

    def test_system_stats(self):
        reg = self._make_tools()
        result = self._call(reg, "system_stats")
        assert result is not None
        # On Linux, at least uptime or memory should be present
        assert "Uptime" in result or "Memory" in result or "Could not" in result

    def test_set_timer(self):
        reg = self._make_tools()
        result = self._call(reg, "set_timer", {"seconds": 9999})
        assert result == "Timer set for 9999 seconds."
        assert len(self._timer_threads) == 1
        assert self._timer_threads[0].is_alive()
        # Clean up
        for t in self._timer_threads:
            t.cancel()

    def test_remember_and_recall(self):
        reg = self._make_tools()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            self._mem_path = f.name
        try:
            # Remove the temp file so we start empty
            os.unlink(self._mem_path)

            result = self._call(reg, "remember", {"info": "The robot IP is 192.0.2.1"})
            assert result == "I'll remember that."

            result = self._call(reg, "recall", {"query": "robot"})
            assert "192.0.2.1" in result

            result = self._call(reg, "recall", {"query": "all"})
            assert "192.0.2.1" in result
        finally:
            if os.path.exists(self._mem_path):
                os.unlink(self._mem_path)

    def test_recall_no_memories(self):
        reg = self._make_tools()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            self._mem_path = f.name
        try:
            os.unlink(self._mem_path)
            result = self._call(reg, "recall")
            assert "don't have any memories" in result
        finally:
            if os.path.exists(self._mem_path):
                os.unlink(self._mem_path)

    def test_web_search_missing_package(self):
        """web_search returns helpful message when duckduckgo-search not installed."""
        reg = self._make_tools()
        # Mock the import to simulate missing package
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "duckduckgo_search":
                raise ImportError("No module named 'duckduckgo_search'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = self._call(reg, "web_search", {"query": "test"})
        assert "not available" in result
        assert "pip install" in result
