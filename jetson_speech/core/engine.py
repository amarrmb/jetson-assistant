"""
Main speech engine that orchestrates TTS and STT backends.
"""

from pathlib import Path
from typing import Any, Iterator

import numpy as np

from jetson_speech.config import get_config
from jetson_speech.tts.base import SynthesisResult, TTSBackend, Voice


class Engine:
    """
    Main speech engine for TTS and STT operations.

    Manages backend loading, synthesis, and transcription.
    """

    def __init__(self):
        """Initialize the engine."""
        self._tts_backend: TTSBackend | None = None
        self._stt_backend: Any = None  # STTBackend when implemented
        self._config = get_config()

    # === TTS Methods ===

    def load_tts_backend(
        self,
        backend: TTSBackend | str,
        **kwargs,
    ) -> None:
        """
        Load a TTS backend.

        Args:
            backend: TTSBackend instance or backend name (e.g., "qwen", "piper")
            **kwargs: Backend-specific options
        """
        if isinstance(backend, str):
            from jetson_speech.tts.registry import get_tts_backend

            backend = get_tts_backend(backend)

        # Unload existing backend
        if self._tts_backend is not None:
            self._tts_backend.unload()

        # Load new backend
        backend.load(**kwargs)
        self._tts_backend = backend

    def unload_tts_backend(self) -> None:
        """Unload the current TTS backend."""
        if self._tts_backend is not None:
            self._tts_backend.unload()
            self._tts_backend = None

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        language: str | None = None,
        **kwargs,
    ) -> SynthesisResult:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            voice: Voice ID (uses default if None)
            language: Language name/code (uses default if None)
            **kwargs: Backend-specific options

        Returns:
            SynthesisResult with audio data
        """
        if self._tts_backend is None:
            raise RuntimeError("No TTS backend loaded. Call load_tts_backend() first.")

        voice = voice or self._config.tts.default_voice
        language = language or self._config.tts.default_language

        return self._tts_backend.synthesize(text, voice, language, **kwargs)

    def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        language: str | None = None,
        **kwargs,
    ) -> Iterator[SynthesisResult]:
        """
        Stream synthesis for real-time playback.

        Args:
            text: Text to synthesize
            voice: Voice ID
            language: Language name/code
            **kwargs: Backend-specific options

        Yields:
            SynthesisResult for each chunk
        """
        if self._tts_backend is None:
            raise RuntimeError("No TTS backend loaded. Call load_tts_backend() first.")

        voice = voice or self._config.tts.default_voice
        language = language or self._config.tts.default_language

        yield from self._tts_backend.stream(text, voice, language, **kwargs)

    def synthesize_file(
        self,
        filepath: str | Path,
        output: str | Path | None = None,
        voice: str | None = None,
        language: str | None = None,
        **kwargs,
    ) -> SynthesisResult:
        """
        Synthesize speech from a file (txt, pdf, docx, md).

        Args:
            filepath: Path to input file
            output: Path to save audio (optional)
            voice: Voice ID
            language: Language name/code
            **kwargs: Backend-specific options

        Returns:
            SynthesisResult with audio data
        """
        from jetson_speech.core.audio import concatenate_audio, save_audio
        from jetson_speech.core.text import extract_text, split_into_chunks

        # Extract text from file
        text = extract_text(filepath)

        # Split into chunks
        chunks = split_into_chunks(text, max_chars=self._config.tts.chunk_size)

        # Synthesize each chunk
        audio_list = []
        sample_rate = 0

        for chunk in chunks:
            result = self.synthesize(chunk, voice, language, **kwargs)
            audio_list.append(result.audio)
            sample_rate = result.sample_rate

        # Concatenate
        combined_audio = concatenate_audio(audio_list, sample_rate)

        result = SynthesisResult(
            audio=combined_audio,
            sample_rate=sample_rate,
            voice=voice or self._config.tts.default_voice,
            text=text[:100] + "..." if len(text) > 100 else text,
            metadata={"source_file": str(filepath), "chunks": len(chunks)},
        )

        # Save if output specified
        if output:
            save_audio(combined_audio, sample_rate, output)

        return result

    def get_tts_voices(self) -> list[Voice]:
        """Get available TTS voices."""
        if self._tts_backend is None:
            return []
        return self._tts_backend.get_voices()

    def get_tts_languages(self) -> list[str]:
        """Get supported TTS languages."""
        if self._tts_backend is None:
            return []
        return self._tts_backend.get_languages()

    def get_tts_info(self) -> dict[str, Any]:
        """Get TTS backend information."""
        if self._tts_backend is None:
            return {"loaded": False}
        return self._tts_backend.get_info()

    # === STT Methods ===

    def load_stt_backend(
        self,
        backend: Any,  # STTBackend when implemented
        **kwargs,
    ) -> None:
        """
        Load an STT backend.

        Args:
            backend: STTBackend instance or backend name
            **kwargs: Backend-specific options
        """
        if isinstance(backend, str):
            from jetson_speech.stt.registry import get_stt_backend

            backend = get_stt_backend(backend)

        # Unload existing backend
        if self._stt_backend is not None:
            self._stt_backend.unload()

        # Load new backend
        backend.load(**kwargs)
        self._stt_backend = backend

    def unload_stt_backend(self) -> None:
        """Unload the current STT backend."""
        if self._stt_backend is not None:
            self._stt_backend.unload()
            self._stt_backend = None

    def transcribe(
        self,
        audio: np.ndarray | str | Path,
        sample_rate: int | None = None,
        language: str | None = None,
        **kwargs,
    ) -> Any:  # TranscriptionResult when implemented
        """
        Transcribe audio to text.

        Args:
            audio: Audio data (np.ndarray) or path to audio file
            sample_rate: Sample rate (required if audio is np.ndarray)
            language: Language code (auto-detect if None)
            **kwargs: Backend-specific options

        Returns:
            TranscriptionResult with text and segments
        """
        if self._stt_backend is None:
            raise RuntimeError("No STT backend loaded. Call load_stt_backend() first.")

        # Load audio from file if path provided
        if isinstance(audio, (str, Path)):
            from jetson_speech.core.audio import load_audio

            audio, sample_rate = load_audio(audio)

        if sample_rate is None:
            raise ValueError("sample_rate required when audio is np.ndarray")

        return self._stt_backend.transcribe(audio, sample_rate, language, **kwargs)

    def get_stt_info(self) -> dict[str, Any]:
        """Get STT backend information."""
        if self._stt_backend is None:
            return {"loaded": False}
        return self._stt_backend.get_info()

    # === General Methods ===

    def get_info(self) -> dict[str, Any]:
        """Get engine information."""
        from jetson_speech.config import get_jetson_power_mode, is_jetson

        return {
            "is_jetson": is_jetson(),
            "power_mode": get_jetson_power_mode(),
            "tts": self.get_tts_info(),
            "stt": self.get_stt_info(),
        }

    def say(
        self,
        text: str,
        voice: str | None = None,
        language: str | None = None,
        stream: bool = False,
        **kwargs,
    ) -> SynthesisResult | None:
        """
        Synthesize and play text.

        Args:
            text: Text to speak
            voice: Voice ID
            language: Language name/code
            stream: If True, use streaming playback
            **kwargs: Backend-specific options

        Returns:
            SynthesisResult (or None if streaming)
        """
        from jetson_speech.core.audio import concatenate_audio, play_audio

        if stream:
            # Streaming playback
            from jetson_speech.core.streaming import stream_and_play
            from jetson_speech.core.text import split_into_chunks

            chunks = split_into_chunks(text, by_sentence=True)

            def synth_fn(chunk: str) -> SynthesisResult:
                return self.synthesize(chunk, voice, language, **kwargs)

            results = stream_and_play(chunks, synth_fn)

            # Concatenate for return
            if results:
                audio_list = [r.audio for r in results]
                combined = concatenate_audio(audio_list, results[0].sample_rate)
                return SynthesisResult(
                    audio=combined,
                    sample_rate=results[0].sample_rate,
                    voice=results[0].voice,
                    text=text,
                )
            return None
        else:
            # Regular playback
            result = self.synthesize(text, voice, language, **kwargs)
            play_audio(result.audio, result.sample_rate)
            return result
