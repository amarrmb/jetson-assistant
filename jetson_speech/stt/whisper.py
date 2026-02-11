"""
Whisper STT backend implementation.

Uses faster-whisper for optimized inference.
"""

import sys
from typing import Any, Iterator

import numpy as np

from jetson_speech.stt.base import STTBackend, TranscriptionResult, TranscriptionSegment
from jetson_speech.stt.registry import register_stt_backend


@register_stt_backend("whisper")
class WhisperBackend(STTBackend):
    """Whisper STT backend using faster-whisper."""

    name = "whisper"
    supports_streaming = False  # Basic Whisper doesn't support streaming

    def __init__(self):
        super().__init__()
        self._model_size = "base"
        self._device = "auto"
        self._compute_type = "auto"

    def load(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
        **kwargs,
    ) -> None:
        """
        Load Whisper model.

        Args:
            model_size: Model size (tiny, base, small, medium, large-v2, large-v3)
            device: Device to use ("cuda", "cpu", or "auto")
            compute_type: Compute type ("float16", "int8", or "auto")
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError(
                "faster-whisper not installed. "
                "Install with: pip install jetson-speech[whisper]"
            ) from e

        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type

        print(f"Loading Whisper ({model_size})...", file=sys.stderr)

        # Determine device - try CUDA first, fall back to CPU
        if device == "auto":
            device = "cpu"  # Default to CPU
            try:
                import ctranslate2
                # Check if ctranslate2 was compiled with CUDA
                if "cuda" in ctranslate2.get_supported_compute_types("cuda"):
                    import torch
                    if torch.cuda.is_available():
                        device = "cuda"
            except Exception:
                pass

        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        # Try loading model, fall back to CPU if CUDA fails
        try:
            self._model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
            )
        except ValueError as e:
            if "CUDA" in str(e) and device == "cuda":
                print("CUDA not available in ctranslate2, falling back to CPU", file=sys.stderr)
                device = "cpu"
                compute_type = "int8"
                self._model = WhisperModel(
                    model_size,
                    device=device,
                    compute_type=compute_type,
                )
            else:
                raise

        self._loaded = True
        print(f"Whisper model loaded on {device}!", file=sys.stderr)

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: str | None = None,
        task: str = "transcribe",
        beam_size: int = 5,
        word_timestamps: bool = True,
        **kwargs,
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data (int16 PCM)
            sample_rate: Sample rate in Hz
            language: Language code (auto-detect if None)
            task: "transcribe" or "translate" (to English)
            beam_size: Beam size for decoding
            word_timestamps: Include word-level timestamps
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Convert int16 to float32 in range [-1, 1]
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        else:
            audio_float = audio.astype(np.float32)

        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            from scipy import signal

            num_samples = int(len(audio_float) * 16000 / sample_rate)
            audio_float = signal.resample(audio_float, num_samples)

        # Transcribe — use no_speech_threshold to filter silent segments
        segments_gen, info = self._model.transcribe(
            audio_float,
            language=language,
            task=task,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
            no_speech_threshold=0.6,
        )

        # Collect segments
        segments = []
        full_text_parts = []

        for segment in segments_gen:
            words = []
            if word_timestamps and segment.words:
                words = [
                    {
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "probability": w.probability,
                    }
                    for w in segment.words
                ]

            # no_speech_prob > 0.5 means Whisper thinks this is not speech
            no_speech_prob = getattr(segment, "no_speech_prob", 0.0)
            if no_speech_prob > 0.6:
                continue

            # avg_logprob is negative (e.g., -0.3); convert to 0-1 range
            # Values closer to 0 are higher confidence
            avg_logprob = getattr(segment, "avg_logprob", 0.0)
            confidence = max(0.0, min(1.0, 1.0 + avg_logprob))  # -1.0 → 0.0, 0.0 → 1.0

            segments.append(
                TranscriptionSegment(
                    text=segment.text.strip(),
                    start=segment.start,
                    end=segment.end,
                    confidence=confidence,
                    words=words,
                )
            )
            full_text_parts.append(segment.text.strip())

        # Calculate duration from audio
        duration = len(audio) / sample_rate

        return TranscriptionResult(
            text=" ".join(full_text_parts),
            segments=segments,
            language=info.language if hasattr(info, "language") else (language or "en"),
            duration=duration,
            metadata={
                "model_size": self._model_size,
                "language_probability": getattr(info, "language_probability", None),
                "task": task,
            },
        )

    def stream(
        self,
        audio_stream: Iterator[np.ndarray],
        sample_rate: int,
        chunk_duration: float = 5.0,
        **kwargs,
    ) -> Iterator[TranscriptionSegment]:
        """
        Stream transcription (basic implementation).

        Note: This is a basic chunked implementation, not true streaming.
        Buffers audio for `chunk_duration` seconds, then transcribes.

        Args:
            audio_stream: Iterator of audio chunks
            sample_rate: Sample rate in Hz
            chunk_duration: Duration of each transcription chunk
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Buffer for accumulating audio
        buffer = np.array([], dtype=np.int16)
        chunk_samples = int(sample_rate * chunk_duration)
        time_offset = 0.0

        for audio_chunk in audio_stream:
            # Add to buffer
            buffer = np.concatenate([buffer, audio_chunk])

            # Process when buffer is full enough
            while len(buffer) >= chunk_samples:
                # Extract chunk
                chunk = buffer[:chunk_samples]
                buffer = buffer[chunk_samples:]

                # Transcribe chunk
                result = self.transcribe(chunk, sample_rate, **kwargs)

                # Yield segments with adjusted timestamps
                for segment in result.segments:
                    yield TranscriptionSegment(
                        text=segment.text,
                        start=segment.start + time_offset,
                        end=segment.end + time_offset,
                        confidence=segment.confidence,
                        words=segment.words,
                    )

                time_offset += chunk_duration

        # Process remaining buffer
        if len(buffer) > 0:
            result = self.transcribe(buffer, sample_rate, **kwargs)
            for segment in result.segments:
                yield TranscriptionSegment(
                    text=segment.text,
                    start=segment.start + time_offset,
                    end=segment.end + time_offset,
                    confidence=segment.confidence,
                    words=segment.words,
                )

    def get_languages(self) -> list[str]:
        """Get supported languages."""
        # Whisper supports many languages
        return [
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
            "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
            "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
            "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
            "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
            "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
            "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
            "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
            "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
            "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
        ]

    def get_model_sizes(self) -> list[str]:
        """Get available model sizes."""
        return ["tiny", "base", "small", "medium", "large-v2", "large-v3"]

    def unload(self) -> None:
        """Unload model and free memory."""
        if self._model is not None:
            del self._model
            self._model = None

            # Try to free GPU memory
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
            "compute_type": self._compute_type,
            "model_sizes": self.get_model_sizes(),
        })
        return info
