"""
Piper TTS backend implementation.

Piper is a fast, local TTS system that runs well on CPU.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)
from typing import Any, Iterator

import numpy as np

from jetson_speech.config import get_default_cache_dir
from jetson_speech.tts.base import SynthesisResult, TTSBackend, Voice
from jetson_speech.tts.registry import register_tts_backend

# Common Piper voice models
PIPER_VOICES = {
    "en_US-lessac-medium": {
        "name": "Lessac",
        "language": "en-US",
        "gender": "male",
        "description": "US English medium quality",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
    },
    "en_US-amy-medium": {
        "name": "Amy",
        "language": "en-US",
        "gender": "female",
        "description": "US English medium quality",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx",
        "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
    },
    "en_GB-alba-medium": {
        "name": "Alba",
        "language": "en-GB",
        "gender": "female",
        "description": "British English medium quality",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium/en_GB-alba-medium.onnx",
        "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium/en_GB-alba-medium.onnx.json",
    },
}


@register_tts_backend("piper")
class PiperBackend(TTSBackend):
    """Piper TTS backend for fast, lightweight synthesis."""

    name = "piper"
    supports_streaming = True
    supports_voice_cloning = False

    def __init__(self):
        super().__init__()
        self._sample_rate = 22050
        self._voice_path: Path | None = None
        self._config_path: Path | None = None
        self._current_voice = "en_US-lessac-medium"

    def load(
        self,
        voice: str = "en_US-lessac-medium",
        model_path: str | None = None,
        **kwargs,
    ) -> None:
        """
        Load Piper TTS model.

        Args:
            voice: Voice model name or path
            model_path: Direct path to ONNX model file
        """
        try:
            from piper import PiperVoice
        except ImportError as e:
            raise ImportError(
                "Piper TTS not installed. "
                "Install with: pip install jetson-speech[piper]"
            ) from e

        if model_path:
            # Use provided model path
            self._voice_path = Path(model_path)
            self._config_path = Path(str(model_path) + ".json")
        elif voice in PIPER_VOICES:
            # Download from known voices
            self._voice_path, self._config_path = self._download_voice(voice)
            self._current_voice = voice
        else:
            raise ValueError(f"Unknown voice: {voice}")

        logger.info("Loading Piper voice: %s...", self._voice_path.name)

        self._model = PiperVoice.load(
            str(self._voice_path),
            config_path=str(self._config_path) if self._config_path.exists() else None,
        )

        # Get sample rate from config
        if hasattr(self._model, "config") and hasattr(self._model.config, "sample_rate"):
            self._sample_rate = self._model.config.sample_rate

        self._loaded = True
        logger.info("Piper model loaded")

    def _download_voice(self, voice: str) -> tuple[Path, Path]:
        """Download voice model if not cached."""
        import urllib.request

        voice_info = PIPER_VOICES[voice]
        cache_dir = get_default_cache_dir() / "piper"
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_path = cache_dir / f"{voice}.onnx"
        config_path = cache_dir / f"{voice}.onnx.json"

        if not model_path.exists():
            logger.info("Downloading %s model...", voice)
            urllib.request.urlretrieve(voice_info["url"], model_path)

        if not config_path.exists() and "config_url" in voice_info:
            logger.info("Downloading %s config...", voice)
            urllib.request.urlretrieve(voice_info["config_url"], config_path)

        return model_path, config_path

    def synthesize(
        self,
        text: str,
        voice: str = "default",
        language: str = "en",
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
        **kwargs,
    ) -> SynthesisResult:
        """
        Generate speech from text.

        Args:
            text: Text to synthesize
            voice: Ignored (use load() to change voice)
            language: Ignored (determined by loaded voice)
            length_scale: Speaking rate (higher = slower)
            noise_scale: Variability in pitch
            noise_w: Variability in duration
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Create synthesis config if parameters differ from defaults
        syn_config = None
        try:
            from piper.config import SynthesisConfig
            syn_config = SynthesisConfig(
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_w=noise_w,
            )
        except (ImportError, TypeError):
            pass  # Use defaults if config not available

        # Synthesize and collect audio chunks
        audio_arrays = []
        for audio_chunk in self._model.synthesize(text, syn_config):
            audio_arrays.append(audio_chunk.audio_int16_array)

        # Concatenate all chunks
        if audio_arrays:
            audio = np.concatenate(audio_arrays)
        else:
            audio = np.array([], dtype=np.int16)

        return SynthesisResult(
            audio=audio,
            sample_rate=self._sample_rate,
            voice=self._current_voice,
            text=text,
            metadata={
                "backend": "piper",
                "length_scale": length_scale,
            },
        )

    def stream(
        self,
        text: str,
        voice: str = "default",
        language: str = "en",
        **kwargs,
    ) -> Iterator[SynthesisResult]:
        """
        Stream audio chunks.

        Piper supports native streaming via synthesize_stream_raw.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        from jetson_speech.core.text import split_into_chunks

        chunks = split_into_chunks(text, by_sentence=True)

        for i, chunk in enumerate(chunks):
            result = self.synthesize(chunk, voice, language, **kwargs)
            result.metadata["chunk_index"] = i
            result.metadata["total_chunks"] = len(chunks)
            yield result

    def get_voices(self) -> list[Voice]:
        """Get available Piper voices."""
        voices = []
        for voice_id, info in PIPER_VOICES.items():
            voices.append(
                Voice(
                    id=voice_id,
                    name=info["name"],
                    language=info["language"],
                    gender=info["gender"],
                    description=info["description"],
                    sample_rate=22050,
                )
            )
        return voices

    def get_languages(self) -> list[str]:
        """Get supported languages."""
        languages = set()
        for info in PIPER_VOICES.values():
            lang = info["language"].split("-")[0]
            languages.add(lang)
        return sorted(languages)

    def get_info(self) -> dict[str, Any]:
        """Get backend information."""
        info = super().get_info()
        info.update({
            "current_voice": self._current_voice,
            "sample_rate": self._sample_rate,
            "available_voices": list(PIPER_VOICES.keys()),
        })
        return info
