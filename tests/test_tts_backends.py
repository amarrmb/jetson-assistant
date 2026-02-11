"""
Tests for TTS backends.
"""

import pytest

from jetson_speech.tts.base import SynthesisResult, TTSBackend, Voice
from jetson_speech.tts.registry import get_tts_backend, list_tts_backends, register_tts_backend


class TestTTSBase:
    """Test base TTS classes."""

    def test_voice_creation(self):
        """Test Voice dataclass."""
        voice = Voice(
            id="test",
            name="Test Voice",
            language="en",
            gender="male",
            description="Test description",
        )

        assert voice.id == "test"
        assert voice.name == "Test Voice"
        assert voice.language == "en"

        # Test to_dict
        d = voice.to_dict()
        assert d["id"] == "test"
        assert d["name"] == "Test Voice"

    def test_synthesis_result(self):
        """Test SynthesisResult dataclass."""
        import numpy as np

        audio = np.zeros(24000, dtype=np.int16)  # 1 second of silence
        result = SynthesisResult(
            audio=audio,
            sample_rate=24000,
            voice="test",
            text="Hello",
        )

        assert result.duration == 1.0
        assert result.sample_rate == 24000
        assert len(result.audio) == 24000

        # Test to_bytes
        wav_bytes = result.to_bytes()
        assert len(wav_bytes) > 0
        assert wav_bytes[:4] == b"RIFF"  # WAV header


class TestTTSRegistry:
    """Test TTS backend registry."""

    def test_list_backends(self):
        """Test listing backends."""
        backends = list_tts_backends()
        assert isinstance(backends, list)

    def test_get_unknown_backend(self):
        """Test getting unknown backend raises error."""
        with pytest.raises(ValueError, match="not found"):
            get_tts_backend("nonexistent_backend")

    def test_register_backend(self):
        """Test registering a custom backend."""

        @register_tts_backend("test_backend")
        class TestBackend(TTSBackend):
            name = "test_backend"

            def load(self, **kwargs):
                self._loaded = True

            def synthesize(self, text, voice="default", language="en", **kwargs):
                import numpy as np

                return SynthesisResult(
                    audio=np.zeros(1000, dtype=np.int16),
                    sample_rate=24000,
                    voice=voice,
                    text=text,
                )

            def get_voices(self):
                return [Voice(id="default", name="Default", language="en")]

            def is_loaded(self):
                return self._loaded

        # Should be able to get it
        backend = get_tts_backend("test_backend")
        assert backend.name == "test_backend"


class TestQwenBackend:
    """Test Qwen TTS backend."""

    def test_qwen_voices(self):
        """Test Qwen voice definitions."""
        try:
            from jetson_speech.tts.qwen import QWEN_LANGUAGES, QWEN_SPEAKERS
        except ImportError:
            pytest.skip("Qwen TTS not installed")

        assert "ryan" in QWEN_SPEAKERS
        assert "serena" in QWEN_SPEAKERS
        assert "English" in QWEN_LANGUAGES

    def test_qwen_backend_creation(self):
        """Test creating Qwen backend."""
        try:
            backend = get_tts_backend("qwen")
            assert backend.name == "qwen"
            assert backend.supports_streaming is True
        except (ValueError, ImportError):
            pytest.skip("Qwen TTS not installed")


class TestPiperBackend:
    """Test Piper TTS backend."""

    def test_piper_voices(self):
        """Test Piper voice definitions."""
        try:
            from jetson_speech.tts.piper import PIPER_VOICES
        except ImportError:
            pytest.skip("Piper TTS not installed")

        assert "en_US-lessac-medium" in PIPER_VOICES

    def test_piper_backend_creation(self):
        """Test creating Piper backend."""
        try:
            backend = get_tts_backend("piper")
            assert backend.name == "piper"
        except (ValueError, ImportError):
            pytest.skip("Piper TTS not installed")


class TestKokoroBackend:
    """Test Kokoro TTS backend."""

    def test_kokoro_voices(self):
        """Test Kokoro voice definitions."""
        try:
            from jetson_speech.tts.kokoro import KOKORO_LANG_CODES, KOKORO_VOICES
        except ImportError:
            pytest.skip("Kokoro TTS not installed")

        assert "af_heart" in KOKORO_VOICES
        assert "am_adam" in KOKORO_VOICES
        assert "bf_emma" in KOKORO_VOICES
        assert "bm_george" in KOKORO_VOICES
        assert "en" in KOKORO_LANG_CODES
        assert "ja" in KOKORO_LANG_CODES

    def test_kokoro_backend_creation(self):
        """Test creating Kokoro backend."""
        try:
            backend = get_tts_backend("kokoro")
            assert backend.name == "kokoro"
            assert backend.supports_streaming is True
        except (ValueError, ImportError):
            pytest.skip("Kokoro TTS not installed")

    def test_kokoro_voice_list(self):
        """Test Kokoro voice listing."""
        try:
            backend = get_tts_backend("kokoro")
            voices = backend.get_voices()
            assert len(voices) > 0
            voice_ids = [v.id for v in voices]
            assert "af_heart" in voice_ids
        except (ValueError, ImportError):
            pytest.skip("Kokoro TTS not installed")

    def test_kokoro_languages(self):
        """Test Kokoro language support."""
        try:
            backend = get_tts_backend("kokoro")
            languages = backend.get_languages()
            assert "en" in languages
            assert "ja" in languages
        except (ValueError, ImportError):
            pytest.skip("Kokoro TTS not installed")
