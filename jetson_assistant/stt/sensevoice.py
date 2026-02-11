"""
SenseVoice STT backend implementation.

Alibaba's SenseVoice for multilingual speech recognition.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

import numpy as np

from jetson_assistant.stt.base import STTBackend, TranscriptionResult, TranscriptionSegment
from jetson_assistant.stt.registry import register_stt_backend


@register_stt_backend("sensevoice")
class SenseVoiceBackend(STTBackend):
    """SenseVoice STT backend using FunASR."""

    name = "sensevoice"
    supports_streaming = False

    def __init__(self):
        super().__init__()
        self._model_name = "iic/SenseVoiceSmall"

    def load(
        self,
        model_size: str = "small",
        device: str = "cuda",
        **kwargs,
    ) -> None:
        """
        Load SenseVoice model.

        Args:
            model_size: Model size ("small" or "large")
            device: Device to use ("cuda" or "cpu")
        """
        try:
            from funasr import AutoModel
        except ImportError as e:
            raise ImportError(
                "FunASR not installed. "
                "Install with: pip install jetson-assistant[sensevoice]"
            ) from e

        self._model_size = model_size

        # Select model based on size
        if model_size == "large":
            self._model_name = "iic/SenseVoiceLarge"
        else:
            self._model_name = "iic/SenseVoiceSmall"

        logger.info("Loading SenseVoice (%s)...", model_size)

        self._model = AutoModel(
            model=self._model_name,
            device=device,
            disable_update=True,
        )

        self._loaded = True
        logger.info("SenseVoice model loaded")

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
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Convert int16 to float32 in range [-1, 1]
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        else:
            audio_float = audio.astype(np.float32)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            from scipy import signal

            num_samples = int(len(audio_float) * 16000 / sample_rate)
            audio_float = signal.resample(audio_float, num_samples)

        # Transcribe
        result = self._model.generate(
            input=audio_float,
            cache={},
            language=language or "auto",
            use_itn=True,
        )

        # Parse result
        if result and len(result) > 0:
            text = result[0].get("text", "")
            detected_lang = result[0].get("language", language or "unknown")
        else:
            text = ""
            detected_lang = language or "unknown"

        # Calculate duration
        duration = len(audio) / sample_rate

        # Create single segment (SenseVoice doesn't provide timestamps by default)
        segments = [
            TranscriptionSegment(
                text=text,
                start=0.0,
                end=duration,
                confidence=1.0,
            )
        ]

        return TranscriptionResult(
            text=text,
            segments=segments,
            language=detected_lang,
            duration=duration,
            metadata={
                "model": self._model_name,
            },
        )

    def get_languages(self) -> list[str]:
        """Get supported languages."""
        # SenseVoice supports these languages well
        return [
            "zh",  # Chinese
            "en",  # English
            "ja",  # Japanese
            "ko",  # Korean
            "yue",  # Cantonese
        ]

    def get_model_sizes(self) -> list[str]:
        """Get available model sizes."""
        return ["small", "large"]

    def get_info(self) -> dict[str, Any]:
        """Get backend information."""
        info = super().get_info()
        info.update({
            "model_name": self._model_name,
            "model_sizes": self.get_model_sizes(),
        })
        return info
