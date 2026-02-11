"""
Abstract base class for TTS backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np


@dataclass
class Voice:
    """Voice information."""

    id: str
    name: str
    language: str
    gender: str = ""
    description: str = ""
    sample_rate: int = 24000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "language": self.language,
            "gender": self.gender,
            "description": self.description,
            "sample_rate": self.sample_rate,
        }


@dataclass
class SynthesisResult:
    """Result from speech synthesis."""

    audio: np.ndarray  # int16 PCM audio data
    sample_rate: int
    duration: float = 0.0
    voice: str = ""
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate duration if not set."""
        if self.duration == 0.0 and len(self.audio) > 0:
            self.duration = len(self.audio) / self.sample_rate

    def save(self, path: str) -> None:
        """Save audio to WAV file."""
        from scipy.io import wavfile

        wavfile.write(path, self.sample_rate, self.audio)

    def to_bytes(self) -> bytes:
        """Convert to WAV bytes."""
        import io

        from scipy.io import wavfile

        buffer = io.BytesIO()
        wavfile.write(buffer, self.sample_rate, self.audio)
        return buffer.getvalue()


class TTSBackend(ABC):
    """Abstract base class for TTS backends."""

    name: str = "base"
    supports_streaming: bool = False
    supports_voice_cloning: bool = False

    def __init__(self):
        """Initialize the backend."""
        self._loaded = False
        self._model = None

    @abstractmethod
    def load(self, **kwargs) -> None:
        """
        Load the model into memory.

        Args:
            **kwargs: Backend-specific options (model_size, device, etc.)
        """
        pass

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice: str = "default",
        language: str = "en",
        **kwargs,
    ) -> SynthesisResult:
        """
        Generate speech from text.

        Args:
            text: Text to synthesize
            voice: Voice ID to use
            language: Language code or name
            **kwargs: Backend-specific options

        Returns:
            SynthesisResult with audio data
        """
        pass

    def stream(
        self,
        text: str,
        voice: str = "default",
        language: str = "en",
        **kwargs,
    ) -> Iterator[SynthesisResult]:
        """
        Stream audio chunks for real-time playback.

        Override this method for streaming support.

        Args:
            text: Text to synthesize
            voice: Voice ID to use
            language: Language code or name
            **kwargs: Backend-specific options

        Yields:
            SynthesisResult for each audio chunk
        """
        # Default implementation: yield single result
        yield self.synthesize(text, voice, language, **kwargs)

    @abstractmethod
    def get_voices(self) -> list[Voice]:
        """
        Get available voices.

        Returns:
            List of Voice objects
        """
        pass

    def get_languages(self) -> list[str]:
        """
        Get supported languages.

        Returns:
            List of language names or codes
        """
        return ["en"]

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
            "supports_streaming": self.supports_streaming,
            "supports_voice_cloning": self.supports_voice_cloning,
            "voices": len(self.get_voices()) if self._loaded else 0,
        }
