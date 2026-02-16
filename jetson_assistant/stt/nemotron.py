"""
Nemotron Speech STT backend implementation.

Uses NVIDIA's Nemotron Speech 0.6B streaming model for fast,
accurate English transcription. Runs natively on CUDA without
a container — ~24ms latency on Jetson Thor.
"""

import logging
import tempfile
from typing import Any

logger = logging.getLogger(__name__)

import numpy as np

from jetson_assistant.stt.base import STTBackend, TranscriptionResult, TranscriptionSegment
from jetson_assistant.stt.registry import register_stt_backend


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
        self._device = "auto"

    def load(
        self,
        model_size: str = "0.6b",
        device: str = "auto",
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
                "Install with: pip install jetson-assistant[nemotron]"
            ) from e

        self._model_size = model_size
        self._device = device

        logger.info("Loading Nemotron Speech (%s)...", model_name)

        # Determine device
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    cap = torch.cuda.get_device_capability(0)
                    if cap[0] >= 12:
                        # Blackwell GPUs (sm_12.1+) trigger cuBLAS errors with
                        # the pip PyTorch build (max supported sm_12.0). Use CPU.
                        device = "cpu"
                        logger.info("Blackwell GPU (sm_%d.%d) — using CPU for Nemotron STT", *cap)
                    else:
                        device = "cuda"
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"
            self._device = device

        # Load the model from HuggingFace
        self._model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

        # Move to device — NeMo's from_pretrained auto-loads onto CUDA,
        # so we must explicitly move to CPU when needed (e.g. Blackwell).
        if device == "cpu":
            self._model = self._model.cpu()
        elif device == "cuda":
            self._model = self._model.cuda()

        # Set to eval mode
        self._model.eval()

        # Disable CUDA graphs for the RNNT decoder — they cause
        # "Capture must end on the same stream" errors in Docker
        # containers on Jetson Thor (SBSA CUDA runtime).  The graph
        # controls live on the *inner* decoding object, not the
        # top-level RNNTBPEDecoding wrapper.
        self._disable_cuda_graphs()

        self._loaded = True
        logger.info("Nemotron Speech loaded on %s", device)

    def _disable_cuda_graphs(self) -> None:
        """Disable CUDA graph capture in the RNNT decoder.

        NeMo's RNNT greedy decoder uses CUDA graphs for faster
        inference, but this fails in Docker containers on Jetson Thor
        with 'Capture must end on the same stream it began on'.

        The graph controls live on the inner GreedyBatchedRNNTInfer
        (model.decoding.decoding), NOT the outer RNNTBPEDecoding
        wrapper (model.decoding).
        """
        try:
            inner = getattr(self._model.decoding, "decoding", None)
            if inner is None:
                return

            # Disable the top-level flag
            if hasattr(inner, "use_cuda_graph_decoder"):
                inner.use_cuda_graph_decoder = False
                logger.info("Set use_cuda_graph_decoder=False on %s", type(inner).__name__)

            # Disable on the decoding computer (label looping)
            dc = getattr(inner, "decoding_computer", None)
            if dc is not None and hasattr(dc, "disable_cuda_graphs"):
                dc.disable_cuda_graphs()
                logger.info("Disabled CUDA graphs on %s", type(dc).__name__)
        except Exception:
            logger.warning("Could not disable CUDA graphs", exc_info=True)

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

            # Transcribe using file path.
            # Force CUDA context on this thread (NeMo's RNNT uses CUDA
            # graph capture, which fails if called from a different thread
            # than the one that loaded the model).
            import torch
            if self._device == "cuda":
                torch.cuda.set_device(0)
            with torch.no_grad():
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
