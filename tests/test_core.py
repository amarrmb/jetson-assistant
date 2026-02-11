"""
Tests for core utilities.
"""

import tempfile
from pathlib import Path

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
        assert result[1500] == 0  # Silence
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
