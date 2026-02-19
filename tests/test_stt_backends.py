"""
Tests for STT backends.
"""

import numpy as np
import pytest

from jetson_assistant.stt.base import STTBackend, TranscriptionResult, TranscriptionSegment
from jetson_assistant.stt.registry import get_stt_backend, list_stt_backends


class TestSTTBase:
    """Test base STT classes."""

    def test_transcription_segment(self):
        """Test TranscriptionSegment dataclass."""
        segment = TranscriptionSegment(
            text="Hello world",
            start=0.0,
            end=1.5,
            confidence=0.95,
        )

        assert segment.text == "Hello world"
        assert segment.start == 0.0
        assert segment.end == 1.5
        assert segment.confidence == 0.95

        # Test to_dict
        d = segment.to_dict()
        assert d["text"] == "Hello world"

    def test_transcription_result(self):
        """Test TranscriptionResult dataclass."""
        segments = [
            TranscriptionSegment(text="Hello", start=0.0, end=0.5, confidence=0.9),
            TranscriptionSegment(text="world", start=0.5, end=1.0, confidence=0.95),
        ]

        result = TranscriptionResult(
            text="Hello world",
            segments=segments,
            language="en",
            duration=1.0,
        )

        assert result.text == "Hello world"
        assert len(result.segments) == 2
        assert result.language == "en"

        # Test to_dict
        d = result.to_dict()
        assert d["text"] == "Hello world"
        assert len(d["segments"]) == 2


class TestSTTRegistry:
    """Test STT backend registry."""

    def test_list_backends(self):
        """Test listing backends."""
        backends = list_stt_backends()
        assert isinstance(backends, list)

    def test_get_unknown_backend(self):
        """Test getting unknown backend raises error."""
        with pytest.raises(ValueError, match="not found"):
            get_stt_backend("nonexistent_backend")


class TestWhisperBackend:
    """Test Whisper STT backend."""

    def test_whisper_backend_creation(self):
        """Test creating Whisper backend."""
        try:
            backend = get_stt_backend("whisper")
            assert backend.name == "whisper"
        except (ValueError, ImportError):
            pytest.skip("Whisper not installed")

    def test_whisper_languages(self):
        """Test Whisper language support."""
        try:
            backend = get_stt_backend("whisper")
            languages = backend.get_languages()
            assert "en" in languages
            assert "zh" in languages
        except (ValueError, ImportError):
            pytest.skip("Whisper not installed")

    def test_whisper_model_sizes(self):
        """Test Whisper model sizes."""
        try:
            backend = get_stt_backend("whisper")
            sizes = backend.get_model_sizes()
            assert "tiny" in sizes
            assert "base" in sizes
            assert "large-v3" in sizes
        except (ValueError, ImportError):
            pytest.skip("Whisper not installed")


class TestNemotronBackend:
    """Test Nemotron Speech STT backend."""

    def test_nemotron_backend_creation(self):
        """Test creating Nemotron backend."""
        try:
            backend = get_stt_backend("nemotron")
            assert backend.name == "nemotron"
            assert backend.supports_streaming is False
        except (ValueError, ImportError):
            pytest.skip("Nemotron (NeMo) not installed")

    def test_nemotron_languages(self):
        """Test Nemotron language support (English-only)."""
        try:
            backend = get_stt_backend("nemotron")
            languages = backend.get_languages()
            assert languages == ["en"]
        except (ValueError, ImportError):
            pytest.skip("Nemotron (NeMo) not installed")

    def test_nemotron_model_sizes(self):
        """Test Nemotron model sizes."""
        try:
            backend = get_stt_backend("nemotron")
            sizes = backend.get_model_sizes()
            assert "0.6b" in sizes
        except (ValueError, ImportError):
            pytest.skip("Nemotron (NeMo) not installed")


class TestNemotronFastBackend:
    """Test NemotronFast STT backend (direct forward path)."""

    def test_nemotron_fast_backend_creation(self):
        """Test creating NemotronFast backend via registry."""
        try:
            backend = get_stt_backend("nemotron_fast")
            assert backend.name == "nemotron_fast"
            assert backend.supports_streaming is False
        except (ValueError, ImportError):
            pytest.skip("NemotronFast (NeMo) not installed")

    def test_nemotron_fast_transcribe_returns_result(self):
        """Test NemotronFast transcribe returns TranscriptionResult.

        Loads the actual model and transcribes 1 second of synthetic audio.
        Skipped if NeMo or CUDA is not available.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")
        except ImportError:
            pytest.skip("PyTorch not installed")

        try:
            backend = get_stt_backend("nemotron_fast")
        except (ValueError, ImportError):
            pytest.skip("NemotronFast (NeMo) not installed")

        try:
            backend.load()
        except Exception as e:
            pytest.skip(f"Could not load NemotronFast model: {e}")

        # Generate 1 second of synthetic 16kHz int16 audio (silence with noise)
        rng = np.random.default_rng(42)
        audio = (rng.standard_normal(16000) * 100).astype(np.int16)

        result = backend.transcribe(audio, sample_rate=16000)

        assert isinstance(result, TranscriptionResult)
        assert result.language == "en"
        assert result.duration > 0
        assert result.metadata.get("backend") == "nemotron_fast"

        backend.unload()
