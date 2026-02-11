"""
vLLM Whisper STT backend.

Uses vLLM's OpenAI-compatible /v1/audio/transcriptions endpoint
for GPU-accelerated Whisper inference (~126ms vs 700-900ms CPU).

Requires a vLLM container serving a Whisper model:
    docker run --runtime=nvidia --network host vllm-whisper-audio:latest \
        vllm serve openai/whisper-large-v3-turbo \
        --host 0.0.0.0 --port 8002 --max-model-len 448 \
        --gpu-memory-utilization 0.1 --enforce-eager
"""

import io
import logging
from typing import Any

logger = logging.getLogger(__name__)

import numpy as np

from jetson_assistant.stt.base import STTBackend, TranscriptionResult, TranscriptionSegment
from jetson_assistant.stt.registry import register_stt_backend


@register_stt_backend("vllm")
class VLLMWhisperBackend(STTBackend):
    """vLLM Whisper STT backend using OpenAI-compatible audio API."""

    name = "vllm"
    supports_streaming = False

    def __init__(self):
        super().__init__()
        self._host = "http://localhost:8002/v1"
        self._model_name = "openai/whisper-large-v3-turbo"
        self._client = None

    def load(
        self,
        host: str = "http://localhost:8002/v1",
        model_size: str = "large-v3-turbo",
        **kwargs,
    ) -> None:
        """
        Connect to vLLM Whisper server.

        Args:
            host: vLLM server base URL (e.g. http://localhost:8002/v1)
            model_size: Ignored (model is determined by the vLLM server)
        """
        import httpx

        self._host = host.rstrip("/")
        self._model_size = model_size
        self._client = httpx.Client(timeout=30.0)

        # Auto-detect model name from server
        try:
            resp = self._client.get(f"{self._host}/models")
            resp.raise_for_status()
            models = resp.json().get("data", [])
            if models:
                self._model_name = models[0]["id"]
                logger.info("vLLM Whisper: connected, model=%s", self._model_name)
            else:
                logger.warning("vLLM Whisper: connected but no models found")
        except Exception as e:
            logger.warning("vLLM Whisper: could not query models (%s), using default=%s", e, self._model_name)

        self._loaded = True

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: str | None = None,
        **kwargs,
    ) -> TranscriptionResult:
        """
        Transcribe audio via vLLM Whisper server.

        Args:
            audio: Audio data (int16 PCM)
            sample_rate: Sample rate in Hz
            language: Language code (e.g. "en")
        """
        if not self._loaded or self._client is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        import scipy.io.wavfile

        # Convert int16 PCM to WAV bytes
        buf = io.BytesIO()
        scipy.io.wavfile.write(buf, sample_rate, audio)
        wav_bytes = buf.getvalue()

        # POST multipart form to /audio/transcriptions
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {"model": self._model_name}
        if language:
            data["language"] = language

        resp = self._client.post(
            f"{self._host}/audio/transcriptions",
            files=files,
            data=data,
        )
        resp.raise_for_status()
        result = resp.json()

        text = result.get("text", "").strip()
        duration = len(audio) / sample_rate

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
            language=language or "en",
            duration=duration,
            metadata={"model": self._model_name, "backend": "vllm"},
        )

    def get_languages(self) -> list[str]:
        """Get supported languages (Whisper supports many)."""
        return [
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
            "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
        ]

    def get_model_sizes(self) -> list[str]:
        """Get available model sizes (server-determined)."""
        return ["large-v3-turbo", "large-v3", "large-v2", "medium", "small", "base", "tiny"]

    def get_info(self) -> dict[str, Any]:
        """Get backend information."""
        info = super().get_info()
        info.update({
            "host": self._host,
            "model_name": self._model_name,
            "type": "remote_gpu",
        })
        return info
