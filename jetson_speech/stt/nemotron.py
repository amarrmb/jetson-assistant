"""
Nemotron Speech STT backend implementation.

Uses NVIDIA's Nemotron Speech 0.6B streaming model for fast,
accurate English transcription. Runs natively on CUDA without
a container — ~24ms latency on Jetson Thor.
"""

import sys
import tempfile
from typing import Any

import numpy as np

from jetson_speech.stt.base import STTBackend, TranscriptionResult, TranscriptionSegment
from jetson_speech.stt.registry import register_stt_backend


@register_stt_backend("nemotron")
class NemotronBackend(STTBackend):
    """Nemotron Speech STT backend using NeMo ASR.

    Uses nvidia/nemotron-speech-streaming-en-0.6b for fast English
    transcription with native punctuation and capitalization.
    """

    name = "nemotron"
    supports_streaming = False  # Batch-only for v1; streaming via PipelineBuilder is complex

    def __init__(self):
        super().__init__()
        self._model_size = "0.6b"
        self._device = "cuda"

    def load(
        self,
        model_size: str = "0.6b",
        device: str = "cuda",
        model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b",
        **kwargs,
    ) -> None:
        """
        Load Nemotron Speech model.

        Args:
            model_size: Model size identifier (for display)
            device: Device to use ("cuda" or "cpu")
            model_name: HuggingFace model name
        """
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as e:
            raise ImportError(
                "NeMo ASR not installed. "
                "Install with: pip install jetson-speech[nemotron]"
            ) from e

        self._model_size = model_size
        self._device = device

        print(f"Loading Nemotron Speech ({model_name})...", file=sys.stderr)

        # Determine device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
            self._device = device

        # Load the model from HuggingFace
        self._model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

        # Move to device
        if device == "cuda":
            self._model = self._model.cuda()

        # Set to eval mode
        self._model.eval()

        self._loaded = True
        print(f"Nemotron Speech loaded on {device}!", file=sys.stderr)

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
            language: Ignored (English-only model)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Convert int16 to float32 in range [-1, 1]
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        else:
            audio_float = audio.astype(np.float32)

        # Resample to 16kHz if needed (NeMo expects 16kHz)
        if sample_rate != 16000:
            from scipy import signal

            num_samples = int(len(audio_float) * 16000 / sample_rate)
            audio_float = signal.resample(audio_float, num_samples)
            sample_rate = 16000

        # NeMo's transcribe API is most reliable with file paths
        # Write to temp WAV file
        import scipy.io.wavfile as wavfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            # Convert back to int16 for WAV file
            audio_int16 = (audio_float * 32767).astype(np.int16)
            wavfile.write(tmp.name, 16000, audio_int16)

            # Transcribe using file path
            transcriptions = self._model.transcribe([tmp.name])

        # NeMo returns a list of strings (or Hypothesis objects)
        if isinstance(transcriptions, list) and len(transcriptions) > 0:
            # Handle both str and Hypothesis objects
            result = transcriptions[0]
            if hasattr(result, "text"):
                text = result.text
            elif isinstance(result, str):
                text = result
            else:
                text = str(result)
        else:
            text = ""

        # Calculate duration from original audio
        duration = len(audio) / sample_rate

        # Create a single segment for the full transcription
        segments = []
        if text.strip():
            segments.append(
                TranscriptionSegment(
                    text=text.strip(),
                    start=0.0,
                    end=duration,
                    confidence=1.0,  # NeMo doesn't expose per-segment confidence easily
                )
            )

        return TranscriptionResult(
            text=text.strip(),
            segments=segments,
            language="en",
            duration=duration,
            metadata={
                "model_size": self._model_size,
                "device": self._device,
                "backend": "nemotron",
            },
        )

    def get_languages(self) -> list[str]:
        """Get supported languages (English-only for streaming variant)."""
        return ["en"]

    def get_model_sizes(self) -> list[str]:
        """Get available model sizes."""
        return ["0.6b"]

    def unload(self) -> None:
        """Unload model and free memory."""
        if self._model is not None:
            del self._model
            self._model = None

            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        self._loaded = False

    def get_info(self) -> dict[str, Any]:
        """Get backend information."""
        info = super().get_info()
        info.update({
            "device": self._device,
            "model_name": "nvidia/nemotron-speech-streaming-en-0.6b",
        })
        return info
