"""
Tests for core utilities.
"""

import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


class TestTextProcessing:
    """Test text extraction and processing."""

    def test_split_into_chunks(self):
        """Test text chunking."""
        from jetson_assistant.core.text import split_into_chunks

        text = "Hello world. How are you? I am fine."
        chunks = split_into_chunks(text, max_chars=50)

        assert len(chunks) >= 1
        assert all(len(c) <= 50 or len(c.split()) == 1 for c in chunks)

    def test_split_by_sentence(self):
        """Test sentence-based chunking."""
        from jetson_assistant.core.text import split_into_chunks

        text = "First sentence. Second sentence. Third sentence."
        chunks = split_into_chunks(text, by_sentence=True)

        assert len(chunks) == 3
        assert chunks[0] == "First sentence."

    def test_extract_text_txt(self):
        """Test extracting text from TXT file."""
        from jetson_assistant.core.text import extract_text

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello world")
            f.flush()

            text = extract_text(f.name)
            assert text == "Hello world"

    def test_extract_text_file_not_found(self):
        """Test extraction from non-existent file."""
        from jetson_assistant.core.text import extract_text

        with pytest.raises(FileNotFoundError):
            extract_text("/nonexistent/file.txt")

    def test_clean_text_for_speech(self):
        """Test text cleaning."""
        from jetson_assistant.core.text import clean_text_for_speech

        text = "Check https://example.com and email@test.com"
        cleaned = clean_text_for_speech(text)

        assert "https://" not in cleaned
        assert "@" not in cleaned


class TestAudioProcessing:
    """Test audio processing utilities."""

    def test_concatenate_audio(self):
        """Test audio concatenation."""
        from jetson_assistant.core.audio import concatenate_audio

        audio1 = np.ones(1000, dtype=np.int16)
        audio2 = np.ones(1000, dtype=np.int16) * 2

        result = concatenate_audio([audio1, audio2], sample_rate=1000, silence_duration=0.5)

        # Should be audio1 + 500 samples silence + audio2
        assert len(result) == 2500
        assert result[0] == 1  # From audio1
        assert result[1250] == 0  # Silence (indices 1000-1499)
        assert result[-1] == 2  # From audio2

    def test_audio_to_bytes(self):
        """Test converting audio to bytes."""
        from jetson_assistant.core.audio import audio_to_bytes

        audio = np.zeros(1000, dtype=np.int16)
        wav_bytes = audio_to_bytes(audio, sample_rate=16000)

        assert len(wav_bytes) > 0
        assert wav_bytes[:4] == b"RIFF"

    def test_bytes_to_audio(self):
        """Test converting bytes to audio."""
        from jetson_assistant.core.audio import audio_to_bytes, bytes_to_audio

        original = np.arange(1000, dtype=np.int16)
        wav_bytes = audio_to_bytes(original, sample_rate=16000)

        audio, sample_rate = bytes_to_audio(wav_bytes)

        assert sample_rate == 16000
        np.testing.assert_array_equal(audio, original)

    def test_get_audio_duration(self):
        """Test duration calculation."""
        from jetson_assistant.core.audio import get_audio_duration

        audio = np.zeros(16000, dtype=np.int16)
        duration = get_audio_duration(audio, sample_rate=16000)

        assert duration == 1.0

    def test_normalize_audio(self):
        """Test audio normalization."""
        from jetson_assistant.core.audio import normalize_audio

        # Quiet audio
        audio = np.array([100, -100, 50, -50], dtype=np.int16)
        normalized = normalize_audio(audio, target_db=-3.0)

        # Should be louder
        assert np.max(np.abs(normalized)) > np.max(np.abs(audio))


class TestEngine:
    """Test the main Engine class."""

    def test_engine_creation(self):
        """Test creating engine instance."""
        from jetson_assistant.core.engine import Engine

        engine = Engine()
        assert engine is not None
        assert engine.get_tts_info()["loaded"] is False
        assert engine.get_stt_info()["loaded"] is False

    def test_engine_info(self):
        """Test getting engine info."""
        from jetson_assistant.core.engine import Engine

        engine = Engine()
        info = engine.get_info()

        assert "is_jetson" in info
        assert "tts" in info
        assert "stt" in info

    def test_synthesize_without_backend(self):
        """Test synthesis fails without backend."""
        from jetson_assistant.core.engine import Engine

        engine = Engine()

        with pytest.raises(RuntimeError, match="No TTS backend loaded"):
            engine.synthesize("Hello")

    def test_transcribe_without_backend(self):
        """Test transcription fails without backend."""
        from jetson_assistant.core.engine import Engine

        engine = Engine()
        audio = np.zeros(1000, dtype=np.int16)

        with pytest.raises(RuntimeError, match="No STT backend loaded"):
            engine.transcribe(audio, sample_rate=16000)


class TestIntentRouter:
    """Test the intent classification router in VoiceAssistant."""

    @staticmethod
    def _make_assistant(**overrides):
        """Create a minimal VoiceAssistant without running __init__."""
        from jetson_assistant.assistant.core import VoiceAssistant, AssistantConfig

        assistant = VoiceAssistant.__new__(VoiceAssistant)
        assistant.config = AssistantConfig(
            llm_backend="vllm", llm_host="http://localhost:8001/v1"
        )
        assistant._tools = None
        assistant.llm = MagicMock()
        for key, value in overrides.items():
            setattr(assistant, key, value)
        return assistant

    def test_intent_router_skipped_when_tools_none(self):
        """Intent router should not call LLM when _tools is None."""
        assistant = self._make_assistant(_tools=None)

        result = assistant._classify_intent("hello world")

        assert result == "CHAT"
        assistant.llm.classify_intent.assert_not_called()

    def test_intent_router_skipped_when_tools_empty(self):
        """Intent router should not call LLM when tools registry has no definitions."""
        empty_registry = MagicMock()
        empty_registry.definitions.return_value = []
        assistant = self._make_assistant(_tools=empty_registry)

        result = assistant._classify_intent("hello world")

        assert result == "CHAT"
        assistant.llm.classify_intent.assert_not_called()

    def test_intent_router_skipped_when_llm_none(self):
        """Intent router should not call LLM when llm is None."""
        tools_with_defs = MagicMock()
        tools_with_defs.definitions.return_value = [{"function": {"name": "test"}}]
        assistant = self._make_assistant(_tools=tools_with_defs, llm=None)

        result = assistant._classify_intent("hello world")

        assert result == "CHAT"

    def test_intent_router_calls_llm_when_tools_loaded(self):
        """Intent router should call LLM when tools are loaded."""
        tools_with_defs = MagicMock()
        tools_with_defs.definitions.return_value = [{"function": {"name": "test"}}]
        assistant = self._make_assistant(_tools=tools_with_defs)
        assistant.llm.classify_intent.return_value = "TOOL"

        result = assistant._classify_intent("take a photo")

        assert result == "TOOL"
        assistant.llm.classify_intent.assert_called_once()


class TestVisionCaptureAsync:
    """Test parallel vision capture via ThreadPoolExecutor."""

    @staticmethod
    def _make_assistant(**overrides):
        """Create a minimal VoiceAssistant with vision executor."""
        from jetson_assistant.assistant.core import VoiceAssistant, AssistantConfig

        assistant = VoiceAssistant.__new__(VoiceAssistant)
        assistant.config = AssistantConfig(vision_enabled=True)
        assistant._camera_lock = threading.Lock()
        assistant.camera = MagicMock()
        assistant.camera.is_open = True
        assistant.camera.capture_base64.return_value = "base64img"
        # _vision_executor will be created by the method under test
        from concurrent.futures import ThreadPoolExecutor
        assistant._vision_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="vision"
        )
        for key, value in overrides.items():
            setattr(assistant, key, value)
        return assistant

    def test_vision_capture_runs_in_parallel(self):
        """Vision capture should not block context building."""
        assistant = self._make_assistant()
        # Simulate slow camera capture (100ms)
        assistant.camera.capture_base64.side_effect = lambda: (
            time.sleep(0.1),
            "base64img",
        )[1]

        t0 = time.perf_counter()
        future = assistant._capture_vision_async()
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.05, f"_capture_vision_async blocked for {elapsed * 1000:.0f}ms"

        result = future.result(timeout=1.0)
        assert result == "base64img"

    def test_vision_capture_returns_none_when_camera_closed(self):
        """Should return None when camera is not open."""
        assistant = self._make_assistant()
        assistant.camera.is_open = False

        future = assistant._capture_vision_async()
        result = future.result(timeout=1.0)

        assert result is None
        assistant.camera.capture_base64.assert_not_called()

    def test_vision_capture_returns_none_when_camera_missing(self):
        """Should return None when camera is None."""
        assistant = self._make_assistant()
        assistant.camera = None

        future = assistant._capture_vision_async()
        result = future.result(timeout=1.0)

        assert result is None

    def test_vision_capture_uses_camera_lock(self):
        """Should acquire camera lock during capture."""
        assistant = self._make_assistant()
        lock_acquired_in_thread = threading.Event()

        original_capture = assistant.camera.capture_base64

        def capture_and_check():
            # Verify the lock is held by trying to acquire it (should fail)
            locked = assistant._camera_lock.locked()
            if locked:
                lock_acquired_in_thread.set()
            return "base64img"

        assistant.camera.capture_base64.side_effect = capture_and_check

        future = assistant._capture_vision_async()
        future.result(timeout=1.0)

        assert lock_acquired_in_thread.is_set(), "Camera lock was not held during capture"


class TestTTSCache:
    """Test TTS cache integration in _speak_sentences."""

    def test_speak_sentences_uses_tts_cache(self):
        """_speak_sentences should use cached audio when available."""
        from unittest.mock import patch
        from jetson_assistant.assistant.core import VoiceAssistant
        from jetson_assistant.assistant.tts_cache import CachedAudio

        assistant = VoiceAssistant.__new__(VoiceAssistant)
        assistant.config = MagicMock()
        assistant.config.tts_voice = "af_heart"
        assistant.config.tts_language = "en"
        assistant._tts_cache = MagicMock()
        assistant.engine = MagicMock()
        assistant.audio_output = MagicMock()
        assistant._bargein_event = MagicMock()
        assistant._bargein_event.is_set.return_value = False
        assistant._vision_preview = None
        assistant._external_tool_modules = []

        cached = CachedAudio(audio=np.zeros(1000, dtype=np.int16), sample_rate=24000)
        assistant._tts_cache.get.return_value = cached

        # Mock out aplay subprocess, tempfile, wavfile, and os.unlink
        with patch("subprocess.Popen") as mock_popen, \
             patch("tempfile.NamedTemporaryFile") as mock_tmpf, \
             patch("scipy.io.wavfile.write"), \
             patch("os.unlink"):
            mock_proc = MagicMock()
            mock_popen.return_value = mock_proc
            mock_proc.wait.return_value = 0
            mock_tmpf.return_value.__enter__ = MagicMock(return_value=mock_tmpf.return_value)
            mock_tmpf.return_value.__exit__ = MagicMock(return_value=False)
            mock_tmpf.return_value.name = "/tmp/fake.wav"

            assistant._speak_sentences("Hello there.")

        # TTS engine should NOT be called (cache hit)
        assistant.engine.synthesize.assert_not_called()

    def test_speak_sentences_caches_on_miss(self):
        """_speak_sentences should store synthesis result in cache on miss."""
        from unittest.mock import patch
        from jetson_assistant.assistant.core import VoiceAssistant

        assistant = VoiceAssistant.__new__(VoiceAssistant)
        assistant.config = MagicMock()
        assistant.config.tts_voice = "af_heart"
        assistant.config.tts_language = "en"
        assistant._tts_cache = MagicMock()
        assistant.engine = MagicMock()
        assistant.audio_output = MagicMock()
        assistant._bargein_event = MagicMock()
        assistant._bargein_event.is_set.return_value = False
        assistant._vision_preview = None
        assistant._external_tool_modules = []

        # Cache miss
        assistant._tts_cache.get.return_value = None

        # Engine returns audio
        synth_result = MagicMock()
        synth_result.audio = np.zeros(1000, dtype=np.int16)
        synth_result.sample_rate = 24000
        assistant.engine.synthesize.return_value = synth_result

        with patch("subprocess.Popen") as mock_popen, \
             patch("tempfile.NamedTemporaryFile") as mock_tmpf, \
             patch("scipy.io.wavfile.write"), \
             patch("os.unlink"):
            mock_proc = MagicMock()
            mock_popen.return_value = mock_proc
            mock_proc.wait.return_value = 0
            mock_tmpf.return_value.__enter__ = MagicMock(return_value=mock_tmpf.return_value)
            mock_tmpf.return_value.__exit__ = MagicMock(return_value=False)
            mock_tmpf.return_value.name = "/tmp/fake.wav"

            assistant._speak_sentences("Hello there.")

        # Engine SHOULD be called (cache miss)
        assistant.engine.synthesize.assert_called_once()
        # Result should be stored in cache
        assistant._tts_cache.put.assert_called_once_with(
            "Hello there.", "af_heart", "en",
            synth_result.audio, synth_result.sample_rate,
        )


class TestTTSWorker:
    """Test streaming TTS overlap — background TTS worker thread."""

    @staticmethod
    def _make_assistant(**overrides):
        """Create a minimal VoiceAssistant with TTS worker infrastructure."""
        import queue as _queue
        from jetson_assistant.assistant.core import VoiceAssistant, AssistantConfig
        from jetson_assistant.assistant.tts_cache import TTSCache

        assistant = VoiceAssistant.__new__(VoiceAssistant)
        assistant.config = AssistantConfig()
        assistant._tts_cache = TTSCache(max_entries=16, max_text_len=80)
        assistant.engine = MagicMock()
        assistant.audio_output = MagicMock()
        assistant._bargein_event = threading.Event()
        assistant._vision_preview = None
        assistant._external_tool_modules = []
        assistant._aplay_proc = None

        # Initialise the TTS queue and worker thread
        assistant._tts_queue = _queue.Queue(maxsize=4)
        assistant._tts_thread = threading.Thread(
            target=assistant._tts_worker, daemon=True, name="tts-worker"
        )
        assistant._tts_thread.start()

        for key, value in overrides.items():
            setattr(assistant, key, value)
        return assistant

    def test_tts_worker_exists(self):
        """VoiceAssistant should have a _tts_queue and _tts_worker method."""
        from jetson_assistant.assistant.core import VoiceAssistant

        assert hasattr(VoiceAssistant, '_tts_worker'), "_tts_worker method missing"
        assistant = self._make_assistant()
        assert assistant._tts_thread.is_alive(), "TTS worker thread not running"
        # Shutdown
        assistant._tts_queue.put(None)
        assistant._tts_thread.join(timeout=2)

    def test_tts_worker_synthesizes_and_plays(self):
        """TTS worker should synthesize and play sentences from the queue."""
        assistant = self._make_assistant()

        synth_result = MagicMock()
        synth_result.audio = np.zeros(1000, dtype=np.int16)
        synth_result.sample_rate = 24000
        assistant.engine.synthesize.return_value = synth_result

        # Enqueue a sentence
        assistant._tts_queue.put(("Hello world.", "af_heart", "en"))
        assistant._tts_queue.join()  # Wait for processing

        assistant.engine.synthesize.assert_called_once_with(
            "Hello world.", voice="af_heart", language="en"
        )
        assistant.audio_output.play_blocking.assert_called_once()

        # Shutdown
        assistant._tts_queue.put(None)
        assistant._tts_thread.join(timeout=2)

    def test_tts_worker_uses_cache(self):
        """TTS worker should use cached audio when available."""
        from jetson_assistant.assistant.tts_cache import CachedAudio

        assistant = self._make_assistant()

        # Pre-populate cache
        cached_audio = np.ones(500, dtype=np.int16)
        assistant._tts_cache.put("cached.", "af_heart", "en", cached_audio, 24000)

        # Enqueue cached sentence
        assistant._tts_queue.put(("cached.", "af_heart", "en"))
        assistant._tts_queue.join()

        # Engine should NOT be called (cache hit)
        assistant.engine.synthesize.assert_not_called()
        # Audio should be played
        assistant.audio_output.play_blocking.assert_called_once()
        played_audio = assistant.audio_output.play_blocking.call_args[0][0]
        np.testing.assert_array_equal(played_audio, cached_audio)

        # Shutdown
        assistant._tts_queue.put(None)
        assistant._tts_thread.join(timeout=2)

    def test_tts_worker_caches_on_miss(self):
        """TTS worker should store synthesis results in cache on miss."""
        assistant = self._make_assistant()

        synth_result = MagicMock()
        synth_result.audio = np.zeros(800, dtype=np.int16)
        synth_result.sample_rate = 22050
        assistant.engine.synthesize.return_value = synth_result

        assistant._tts_queue.put(("New phrase.", "af_heart", "en"))
        assistant._tts_queue.join()

        # Verify it was cached
        cached = assistant._tts_cache.get("New phrase.", "af_heart", "en")
        assert cached is not None
        np.testing.assert_array_equal(cached.audio, synth_result.audio)
        assert cached.sample_rate == 22050

        # Shutdown
        assistant._tts_queue.put(None)
        assistant._tts_thread.join(timeout=2)

    def test_tts_worker_skips_on_bargein(self):
        """TTS worker should skip synthesis when barge-in is set."""
        assistant = self._make_assistant()
        assistant._bargein_event.set()  # Simulate barge-in

        synth_result = MagicMock()
        assistant.engine.synthesize.return_value = synth_result

        assistant._tts_queue.put(("Should skip.", "af_heart", "en"))
        assistant._tts_queue.join()

        # Neither synthesize nor play should be called
        assistant.engine.synthesize.assert_not_called()
        assistant.audio_output.play_blocking.assert_not_called()

        # Shutdown
        assistant._tts_queue.put(None)
        assistant._tts_thread.join(timeout=2)

    def test_tts_worker_processes_multiple_sentences(self):
        """TTS worker should process multiple sentences in order."""
        assistant = self._make_assistant()

        synth_result = MagicMock()
        synth_result.audio = np.zeros(500, dtype=np.int16)
        synth_result.sample_rate = 24000
        assistant.engine.synthesize.return_value = synth_result

        # Enqueue 3 sentences
        for s in ["One.", "Two.", "Three."]:
            assistant._tts_queue.put((s, "af_heart", "en"))
        assistant._tts_queue.join()

        assert assistant.engine.synthesize.call_count == 3
        assert assistant.audio_output.play_blocking.call_count == 3

        # Verify order
        calls = [c[0][0] for c in assistant.engine.synthesize.call_args_list]
        assert calls == ["One.", "Two.", "Three."]

        # Shutdown
        assistant._tts_queue.put(None)
        assistant._tts_thread.join(timeout=2)

    def test_tts_worker_stops_on_sentinel(self):
        """TTS worker should exit when None sentinel is received."""
        assistant = self._make_assistant()
        assert assistant._tts_thread.is_alive()

        assistant._tts_queue.put(None)
        assistant._tts_thread.join(timeout=2)

        assert not assistant._tts_thread.is_alive()

    def test_tts_worker_skips_play_on_late_bargein(self):
        """TTS worker should skip playback if barge-in set after synthesis."""
        assistant = self._make_assistant()

        # Make synthesis succeed but set barge-in during synthesis
        def synth_side_effect(*args, **kwargs):
            assistant._bargein_event.set()
            result = MagicMock()
            result.audio = np.zeros(500, dtype=np.int16)
            result.sample_rate = 24000
            return result

        assistant.engine.synthesize.side_effect = synth_side_effect

        assistant._tts_queue.put(("Late bargein.", "af_heart", "en"))
        assistant._tts_queue.join()

        # Synthesis happened but play should be skipped
        assistant.engine.synthesize.assert_called_once()
        assistant.audio_output.play_blocking.assert_not_called()

        # Shutdown
        assistant._tts_queue.put(None)
        assistant._tts_thread.join(timeout=2)


class TestParallelToolExecution:
    """Test parallel tool execution via ThreadPoolExecutor."""

    @staticmethod
    def _make_assistant(**overrides):
        """Create a minimal VoiceAssistant with tool executor."""
        from concurrent.futures import ThreadPoolExecutor
        from jetson_assistant.assistant.core import VoiceAssistant, AssistantConfig

        assistant = VoiceAssistant.__new__(VoiceAssistant)
        assistant.config = AssistantConfig()
        assistant._tool_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool")
        assistant._tools = MagicMock()
        for key, value in overrides.items():
            setattr(assistant, key, value)
        return assistant

    def test_multiple_tools_run_in_parallel(self):
        """When LLM calls multiple tools, they should run concurrently."""
        assistant = self._make_assistant()

        # Simulate: tool1 takes 200ms, tool2 takes 10ms
        def mock_execute(tc):
            if tc.name == "web_search":
                time.sleep(0.2)
                return "search result"
            return "fast result"

        assistant._tools.execute.side_effect = mock_execute

        tool_calls = [
            {"name": "web_search", "arguments": {"query": "test"}},
            {"name": "set_antennas", "arguments": {"left": 45}},
        ]

        t0 = time.perf_counter()
        results = assistant._execute_tools_parallel(tool_calls)
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.25, f"Took {elapsed*1000:.0f}ms — not parallel"
        assert len(results) == 2
        assert results[0] == "search result"
        assert results[1] == "fast result"

    def test_single_tool_runs_inline(self):
        """Single tool call should run inline without executor overhead."""
        assistant = self._make_assistant()

        assistant._tools.execute.return_value = "result"

        tool_calls = [{"name": "get_time", "arguments": {}}]
        results = assistant._execute_tools_parallel(tool_calls)

        assert results == ["result"]
        assistant._tools.execute.assert_called_once()

    def test_exception_in_one_tool_does_not_crash_others(self):
        """If one tool throws, other tools should still return results."""
        assistant = self._make_assistant()

        def mock_execute(tc):
            if tc.name == "bad_tool":
                raise RuntimeError("tool exploded")
            return "good result"

        assistant._tools.execute.side_effect = mock_execute

        tool_calls = [
            {"name": "bad_tool", "arguments": {}},
            {"name": "good_tool", "arguments": {}},
        ]

        results = assistant._execute_tools_parallel(tool_calls)

        assert len(results) == 2
        assert "Error" in results[0]
        assert "tool exploded" in results[0]
        assert results[1] == "good result"

    def test_results_preserve_input_order(self):
        """Results should be in same order as input tool_calls regardless of completion order."""
        assistant = self._make_assistant()

        def mock_execute(tc):
            # First tool is slow, second is fast — but results should match input order
            if tc.name == "slow":
                time.sleep(0.1)
                return "slow-result"
            return "fast-result"

        assistant._tools.execute.side_effect = mock_execute

        tool_calls = [
            {"name": "slow", "arguments": {}},
            {"name": "fast", "arguments": {}},
        ]

        results = assistant._execute_tools_parallel(tool_calls)

        assert results[0] == "slow-result"
        assert results[1] == "fast-result"

    def test_single_tool_exception_returns_error_string(self):
        """Single tool exception should return error string, not raise."""
        assistant = self._make_assistant()

        assistant._tools.execute.side_effect = ValueError("bad arg")

        tool_calls = [{"name": "broken", "arguments": {}}]
        results = assistant._execute_tools_parallel(tool_calls)

        assert len(results) == 1
        assert "Error" in results[0]
        assert "bad arg" in results[0]

    def test_tool_returning_none_gets_fallback_message(self):
        """Tool returning None should get a fallback message."""
        assistant = self._make_assistant()

        assistant._tools.execute.return_value = None

        tool_calls = [{"name": "unknown_tool", "arguments": {}}]
        results = assistant._execute_tools_parallel(tool_calls)

        assert len(results) == 1
        assert "unknown_tool" in results[0]

    def test_empty_arguments_default(self):
        """Tool call without 'arguments' key should default to empty dict."""
        assistant = self._make_assistant()

        assistant._tools.execute.return_value = "ok"

        tool_calls = [{"name": "get_time"}]
        results = assistant._execute_tools_parallel(tool_calls)

        assert results == ["ok"]
        # Verify the ToolCallResult had empty arguments
        called_tc = assistant._tools.execute.call_args[0][0]
        assert called_tc.arguments == {}

    def test_three_slow_tools_run_concurrently(self):
        """Three 100ms tools should complete in ~100ms total, not 300ms."""
        assistant = self._make_assistant()

        def mock_execute(tc):
            time.sleep(0.1)
            return f"result-{tc.name}"

        assistant._tools.execute.side_effect = mock_execute

        tool_calls = [
            {"name": "tool_a", "arguments": {}},
            {"name": "tool_b", "arguments": {}},
            {"name": "tool_c", "arguments": {}},
        ]

        t0 = time.perf_counter()
        results = assistant._execute_tools_parallel(tool_calls)
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.18, f"Took {elapsed*1000:.0f}ms — not parallel (expected ~100ms)"
        assert len(results) == 3
        assert results[0] == "result-tool_a"
        assert results[1] == "result-tool_b"
        assert results[2] == "result-tool_c"


class TestSlowToolAcknowledgment:
    """Test slow tool acknowledgment — spoken ack only for slow tools."""

    @staticmethod
    def _make_assistant(**overrides):
        """Create a minimal VoiceAssistant with _speak_sentences mock."""
        from concurrent.futures import ThreadPoolExecutor
        from jetson_assistant.assistant.core import VoiceAssistant, AssistantConfig

        assistant = VoiceAssistant.__new__(VoiceAssistant)
        assistant.config = AssistantConfig()
        assistant._tool_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool")
        assistant._tools = MagicMock()
        assistant._speak_sentences = MagicMock()
        for key, value in overrides.items():
            setattr(assistant, key, value)
        return assistant

    def test_is_slow_tool_web_search(self):
        """web_search should be classified as a slow tool."""
        assistant = self._make_assistant()
        assert assistant._is_slow_tool("web_search") is True

    def test_is_slow_tool_look(self):
        """look should be classified as a slow tool."""
        assistant = self._make_assistant()
        assert assistant._is_slow_tool("look") is True

    def test_is_slow_tool_search(self):
        """search should be classified as a slow tool."""
        assistant = self._make_assistant()
        assert assistant._is_slow_tool("search") is True

    def test_is_slow_tool_describe_scene(self):
        """describe_scene should be classified as a slow tool."""
        assistant = self._make_assistant()
        assert assistant._is_slow_tool("describe_scene") is True

    def test_is_not_slow_tool_set_antennas(self):
        """set_antennas should NOT be classified as a slow tool."""
        assistant = self._make_assistant()
        assert assistant._is_slow_tool("set_antennas") is False

    def test_is_not_slow_tool_get_time(self):
        """get_time should NOT be classified as a slow tool."""
        assistant = self._make_assistant()
        assert assistant._is_slow_tool("get_time") is False

    def test_acknowledgment_spoken_for_slow_tool(self):
        """_maybe_acknowledge_slow_tools should speak for slow tools."""
        assistant = self._make_assistant()

        tool_calls = [{"name": "web_search", "arguments": {"query": "weather"}}]
        assistant._maybe_acknowledge_slow_tools(tool_calls)

        assistant._speak_sentences.assert_called_once_with("Let me check on that.")

    def test_acknowledgment_spoken_when_mixed_tools(self):
        """Acknowledgment should fire if ANY tool in the list is slow."""
        assistant = self._make_assistant()

        tool_calls = [
            {"name": "set_antennas", "arguments": {"left": 45}},
            {"name": "web_search", "arguments": {"query": "test"}},
        ]
        assistant._maybe_acknowledge_slow_tools(tool_calls)

        assistant._speak_sentences.assert_called_once_with("Let me check on that.")

    def test_no_acknowledgment_for_fast_only_tools(self):
        """No acknowledgment should be spoken when all tools are fast."""
        assistant = self._make_assistant()

        tool_calls = [
            {"name": "set_antennas", "arguments": {"left": 45}},
            {"name": "get_time", "arguments": {}},
        ]
        assistant._maybe_acknowledge_slow_tools(tool_calls)

        assistant._speak_sentences.assert_not_called()

    def test_no_acknowledgment_for_empty_tool_list(self):
        """No acknowledgment for empty tool list."""
        assistant = self._make_assistant()

        assistant._maybe_acknowledge_slow_tools([])

        assistant._speak_sentences.assert_not_called()


class TestVLMPriorityQueue:
    """Test VLM priority queue — user requests preempt watch thread."""

    @staticmethod
    def _make_assistant(**overrides):
        """Create a minimal VoiceAssistant with VLM priority infrastructure."""
        from jetson_assistant.assistant.core import VoiceAssistant, AssistantConfig

        assistant = VoiceAssistant.__new__(VoiceAssistant)
        assistant.config = AssistantConfig()
        assistant._vlm_lock = threading.Lock()
        assistant._vlm_priority = threading.Event()
        assistant.llm = MagicMock()
        for key, value in overrides.items():
            setattr(assistant, key, value)
        return assistant

    def test_request_vlm_priority_sets_event(self):
        """_request_vlm_priority should set the priority event."""
        assistant = self._make_assistant()

        assert not assistant._vlm_priority.is_set()
        assistant._request_vlm_priority()
        assert assistant._vlm_priority.is_set()

    def test_release_vlm_priority_clears_event(self):
        """_release_vlm_priority should clear the priority event."""
        assistant = self._make_assistant()

        assistant._request_vlm_priority()
        assert assistant._vlm_priority.is_set()

        assistant._release_vlm_priority()
        assert not assistant._vlm_priority.is_set()

    def test_vlm_user_request_preempts_watch(self):
        """User VLM requests should have higher priority than watch thread."""
        assistant = self._make_assistant()

        # Signal user priority
        assistant._request_vlm_priority()
        assert assistant._vlm_priority.is_set()

        # Release
        assistant._release_vlm_priority()
        assert not assistant._vlm_priority.is_set()

    def test_vlm_call_with_priority_user_request(self):
        """User request should set/clear priority and acquire lock."""
        assistant = self._make_assistant()

        # Track priority state during the call
        priority_during_call = []

        def mock_call():
            priority_during_call.append(assistant._vlm_priority.is_set())
            return "vlm_result"

        result = assistant._vlm_call_with_priority(
            mock_call,
            is_user_request=True,
        )

        assert result == "vlm_result"
        # Priority should have been set during the call
        assert priority_during_call[0] is True
        # Priority should be cleared after the call
        assert not assistant._vlm_priority.is_set()

    def test_vlm_call_with_priority_watch_skips_when_priority_set(self):
        """Watch thread VLM call should return None when user has priority."""
        assistant = self._make_assistant()

        # Simulate user holding priority
        assistant._request_vlm_priority()

        result = assistant._vlm_call_with_priority(
            lambda: assistant.llm.generate("watch query"),
            is_user_request=False,
        )

        # Should skip (return None) because user has priority
        assert result is None
        assistant.llm.generate.assert_not_called()

    def test_vlm_call_with_priority_watch_proceeds_when_no_priority(self):
        """Watch thread VLM call should proceed when no user priority."""
        assistant = self._make_assistant()
        expected = MagicMock()
        assistant.llm.generate.return_value = expected

        result = assistant._vlm_call_with_priority(
            lambda: assistant.llm.generate("watch query"),
            is_user_request=False,
        )

        # Should proceed normally
        assert result == expected
        assistant.llm.generate.assert_called_once()

    def test_vlm_call_with_priority_user_releases_on_exception(self):
        """Priority should be released even if the VLM call raises."""
        assistant = self._make_assistant()

        def failing_call():
            raise RuntimeError("VLM error")

        with pytest.raises(RuntimeError, match="VLM error"):
            assistant._vlm_call_with_priority(failing_call, is_user_request=True)

        # Priority must be cleared despite the exception
        assert not assistant._vlm_priority.is_set()

    def test_vlm_call_serializes_access(self):
        """VLM calls should be serialized via the lock."""
        assistant = self._make_assistant()
        call_order = []

        def slow_call(label):
            def _call():
                call_order.append(f"{label}_start")
                time.sleep(0.1)
                call_order.append(f"{label}_end")
                return label

            return _call

        # Run two "user" calls concurrently — they should serialize
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(
                assistant._vlm_call_with_priority,
                slow_call("call1"),
                True,
            )
            time.sleep(0.01)  # Small delay to ensure call1 starts first
            f2 = pool.submit(
                assistant._vlm_call_with_priority,
                slow_call("call2"),
                True,
            )
            f1.result(timeout=2)
            f2.result(timeout=2)

        # Verify serialization: call1 should complete before call2 starts
        assert call_order.index("call1_end") < call_order.index("call2_start")

    def test_priority_aware_check_fn_skips_when_priority_set(self):
        """The priority-aware check_fn wrapper should skip when priority is set."""
        assistant = self._make_assistant()
        assistant.llm.check_condition = MagicMock(return_value=True)

        # Get the wrapped check function
        wrapped_fn = assistant._make_priority_aware_check_fn(assistant.llm.check_condition)

        # Without priority — should call through
        result = wrapped_fn("prompt", "image_b64")
        assert result is True
        assistant.llm.check_condition.assert_called_once()

        # With priority set — should skip (return False)
        assistant.llm.check_condition.reset_mock()
        assistant._request_vlm_priority()

        result = wrapped_fn("prompt", "image_b64")
        assert result is False
        assistant.llm.check_condition.assert_not_called()

        assistant._release_vlm_priority()
