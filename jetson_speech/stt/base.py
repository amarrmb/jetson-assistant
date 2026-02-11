"""
Abstract base class for STT backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np


@dataclass
class TranscriptionSegment:
    """A segment of transcribed audio."""

    text: str
    start: float  # Start time in seconds
    end: float  # End time in seconds
    confidence: float = 1.0  # Confidence score (0.0 - 1.0)
    words: list[dict] = field(default_factory=list)  # Word-level timestamps

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "words": self.words,
        }


@dataclass
class TranscriptionResult:
    """Result from speech transcription."""

    text: str  # Full transcription
    segments: list[TranscriptionSegment]  # Segments with timestamps
    language: str  # Detected language
    duration: float  # Audio duration in seconds
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "segments": [s.to_dict() for s in self.segments],
            "language": self.language,
            "duration": self.duration,
            "metadata": self.metadata,
        }


class STTBackend(ABC):
    """Abstract base class for STT backends."""

    name: str = "base"
    supports_streaming: bool = False

    def __init__(self):
        """Initialize the backend."""
        self._loaded = False
        self._model = None
        self._model_size = "base"

    @abstractmethod
    def load(self, model_size: str = "base", **kwargs) -> None:
        """
        Load the model into memory.

        Args:
            model_size: Model size (e.g., "tiny", "base", "small", "medium", "large")
            **kwargs: Backend-specific options
        """
        pass

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: str | None = None,
        **kwargs,
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data (int16 PCM)
            sample_rate: Sample rate in Hz
            language: Language code (auto-detect if None)
            **kwargs: Backend-specific options

        Returns:
            TranscriptionResult with text and segments
        """
        pass

    def stream(
        self,
        audio_stream: Iterator[np.ndarray],
        sample_rate: int,
        **kwargs,
    ) -> Iterator[TranscriptionSegment]:
        """
        Stream transcription for real-time recognition.

        Override this method for streaming support.

        Args:
            audio_stream: Iterator of audio chunks
            sample_rate: Sample rate in Hz
            **kwargs: Backend-specific options

        Yields:
            TranscriptionSegment for each recognized phrase
        """
        raise NotImplementedError("Streaming not supported by this backend")

    @abstractmethod
    def get_languages(self) -> list[str]:
        """
        Get supported languages.

        Returns:
            List of language codes
        """
        pass

    def get_model_sizes(self) -> list[str]:
        """
        Get available model sizes.

        Returns:
            List of model size names
        """
        return ["tiny", "base", "small", "medium", "large"]

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def unload(self) -> None:
        """Unload model from memory."""
        self._model = None
        self._loaded = False

    def get_info(self) -> dict[str, Any]:
        """
        Get backend information.

        Returns:
            Dictionary with backend info
        """
        return {
            "name": self.name,
            "loaded": self._loaded,
            "model_size": self._model_size,
            "supports_streaming": self.supports_streaming,
        }
