"""
NemotronFast STT backend — direct forward() bypass of NeMo batch API.

The standard Nemotron backend (nemotron.py) calls model.transcribe(),
which internally spins up a Lhotse DatalayerLoader per call and writes
to a temp WAV file. Profiling on Jetson Thor shows the GPU forward pass
is ~6ms, but transcribe() adds ~70ms of Python/Lhotse/IO overhead.

This backend calls the model's submodules directly:
  preprocessor → encoder → RNNT decode
No Lhotse dataloader, no temp file I/O, no batch API wrapper.

Expected latency improvement: ~76ms → ~15-20ms on Jetson Thor.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

import numpy as np

from jetson_assistant.stt.base import STTBackend, TranscriptionResult, TranscriptionSegment
from jetson_assistant.stt.registry import register_stt_backend


@register_stt_backend("nemotron_fast")
class NemotronFastBackend(STTBackend):
    """Nemotron Speech STT backend with direct forward() path.

    Uses nvidia/nemotron-speech-streaming-en-0.6b, same as the standard
    Nemotron backend, but bypasses NeMo's transcribe() batch API to call
    preprocessor → encoder → RNNT decode directly on GPU tensors.
    """

    name = "nemotron_fast"
    supports_streaming = False

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
        Load Nemotron Speech model for direct forward inference.

        Args:
            model_size: Model size identifier (for display)
            device: Device to use ("cuda" or "cpu")
            model_name: HuggingFace model name
        """
        try:
            import nemo.collections.asr as nemo_asr  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "NeMo ASR not installed. "
                "Install with: pip install jetson-assistant[nemotron]"
            ) from e

        self._model_size = model_size
        self._device = device

        logger.info("Loading Nemotron Speech FAST (%s)...", model_name)

        # Determine device — probe cuBLAS with a small matmul to check
        # GPU compatibility (standard pip wheels lack Blackwell sm_12.1;
        # SBSA wheels work fine).
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    a = torch.randn(8, 8, device="cuda")
                    _ = a @ a.T  # triggers cuBLAS
                    device = "cuda"
                    logger.info("CUDA matmul probe: OK — NemotronFast will use GPU")
                else:
                    device = "cpu"
            except Exception as e:
                device = "cpu"
                logger.info("CUDA matmul probe failed (%s) — NemotronFast will use CPU", e)
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
        # containers on Jetson Thor (SBSA CUDA runtime).
        self._disable_cuda_graphs()

        self._loaded = True
        logger.info("Nemotron Speech FAST loaded on %s", device)

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
        Transcribe audio via direct forward() — no Lhotse, no temp files.

        Calls preprocessor → encoder → RNNT decode directly, bypassing
        NeMo's transcribe() batch API and its per-call overhead.

        Args:
            audio: Audio data (int16 PCM)
            sample_rate: Sample rate in Hz
            language: Ignored (English-only model)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch

        # Convert int16 to float32 in range [-1, 1]
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        else:
            audio_float = audio.astype(np.float32)

        # Resample to 16kHz if needed (NeMo expects 16kHz)
        if sample_rate != 16000:
            try:
                import torchaudio
                audio_tensor = torch.from_numpy(audio_float).unsqueeze(0)  # [1, T]
                audio_tensor = torchaudio.functional.resample(
                    audio_tensor, orig_freq=sample_rate, new_freq=16000
                )
                audio_float = audio_tensor.squeeze(0).numpy()
            except ImportError:
                from scipy import signal
                num_samples = int(len(audio_float) * 16000 / sample_rate)
                audio_float = signal.resample(audio_float, num_samples)
            sample_rate = 16000

        # Build tensor directly on device — no temp WAV, no Lhotse
        audio_tensor = torch.from_numpy(audio_float).unsqueeze(0).to(self._device)  # [1, T]
        length = torch.tensor([audio_tensor.shape[1]], dtype=torch.long, device=self._device)

        # Force CUDA context on this thread (NeMo's RNNT uses CUDA
        # graph capture, which fails if called from a different thread
        # than the one that loaded the model).
        if self._device == "cuda":
            torch.cuda.set_device(0)

        # Direct forward: preprocessor → encoder → RNNT decode
        with torch.no_grad():
            # 1. Mel spectrogram extraction
            features, feat_length = self._model.preprocessor(
                input_signal=audio_tensor, length=length
            )

            # 2. Encoder forward pass (~6ms on Jetson Thor)
            encoded, enc_length = self._model.encoder(
                audio_signal=features, length=feat_length
            )

            # 3. RNNT greedy decode — produces hypothesis objects
            hypotheses = self._model.decoding.rnnt_decoder_predictions_tensor(
                encoded, enc_length
            )

        # Extract text from hypotheses.
        # rnnt_decoder_predictions_tensor returns (best_hyps, all_hyps).
        # best_hyps is a list of Hypothesis objects (one per batch item).
        text = ""
        if hypotheses and len(hypotheses) > 0:
            best_hyps = hypotheses[0]  # First element is the list of best hypotheses
            if isinstance(best_hyps, list) and len(best_hyps) > 0:
                hyp = best_hyps[0]  # First (and only) batch item
                if hasattr(hyp, "text"):
                    text = hyp.text
                elif isinstance(hyp, str):
                    text = hyp
                else:
                    text = str(hyp)
            elif hasattr(best_hyps, "text"):
                text = best_hyps.text
            elif isinstance(best_hyps, str):
                text = best_hyps
            else:
                text = str(best_hyps)

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
                "backend": "nemotron_fast",
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
            "fast_path": True,
        })
        return info
