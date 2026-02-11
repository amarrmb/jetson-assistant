"""
Qwen3-TTS backend implementation.

Ported from qwen-tts.sh standalone script.
"""

import sys
from typing import Any, Iterator

import numpy as np

from jetson_speech.tts.base import SynthesisResult, TTSBackend, Voice
from jetson_speech.tts.registry import register_tts_backend

# Speaker definitions (from Qwen3-TTS CustomVoice model)
# Valid speakers: aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian
QWEN_SPEAKERS = {
    # Female voices
    "serena": {"name": "Serena", "gender": "female", "description": "Warm, friendly"},
    "vivian": {"name": "Vivian", "gender": "female", "description": "Clear, professional"},
    "sohee": {"name": "Sohee", "gender": "female", "description": "Korean accent"},
    "ono_anna": {"name": "Ono Anna", "gender": "female", "description": "Japanese accent"},
    # Male voices
    "aiden": {"name": "Aiden", "gender": "male", "description": "Young, energetic"},
    "dylan": {"name": "Dylan", "gender": "male", "description": "Casual"},
    "eric": {"name": "Eric", "gender": "male", "description": "Professional"},
    "ryan": {"name": "Ryan", "gender": "male", "description": "Neutral (default)"},
    "uncle_fu": {"name": "Uncle Fu", "gender": "male", "description": "Chinese accent, mature"},
}

QWEN_LANGUAGES = [
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "French",
    "German",
    "Spanish",
    "Portuguese",
    "Russian",
]


@register_tts_backend("qwen")
class QwenBackend(TTSBackend):
    """Qwen3-TTS backend for high-quality neural TTS."""

    name = "qwen"
    supports_streaming = True
    supports_voice_cloning = True

    def __init__(self):
        super().__init__()
        self._model_size = "0.6B"
        self._sample_rate = 24000

    def load(
        self,
        model_size: str = "0.6B",
        device: str = "cuda",
        dtype: str = "bfloat16",
        **kwargs,
    ) -> None:
        """
        Load Qwen3-TTS model.

        Args:
            model_size: Model size ("0.6B" or "1.7B")
            device: Device to use ("cuda" or "cpu")
            dtype: Data type ("bfloat16" or "float16")
        """
        try:
            import torch
            from huggingface_hub import snapshot_download
            from qwen_tts import Qwen3TTSModel
        except ImportError as e:
            raise ImportError(
                "Qwen TTS dependencies not installed. "
                "Install with: pip install jetson-speech[qwen]"
            ) from e

        self._model_size = model_size

        print(f"Loading Qwen3-TTS ({model_size})...", file=sys.stderr)

        model_path = snapshot_download(
            f"Qwen/Qwen3-TTS-12Hz-{model_size}-CustomVoice"
        )

        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

        # Match standalone qwen-tts.sh: use model's default attention
        self._model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=torch_dtype,
        )

        self._loaded = True
        print("Model loaded!", file=sys.stderr)

    def synthesize(
        self,
        text: str,
        voice: str = "ryan",
        language: str = "English",
        temperature: float = 1.0,  # Match standalone script default
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        max_new_tokens: int = 2048,
        **kwargs,
    ) -> SynthesisResult:
        """
        Generate speech from text.

        Args:
            text: Text to synthesize
            voice: Speaker voice ID
            language: Language name
            temperature: Sampling temperature (higher = more varied)
            top_p: Top-p sampling parameter
            repetition_penalty: Penalty for repetition
            max_new_tokens: Maximum tokens to generate
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Normalize voice name
        voice = voice.lower()
        if voice not in QWEN_SPEAKERS:
            voice = "ryan"

        # Build generation kwargs (matching standalone qwen-tts.sh behavior)
        gen_kwargs: dict[str, Any] = {
            "text": text,
            "language": language,
            "speaker": voice,
            "non_streaming_mode": True,
            "max_new_tokens": max_new_tokens,
        }

        # Only pass tuning parameters if not default (matches standalone script)
        if temperature != 1.0:
            gen_kwargs["temperature"] = temperature
        if top_p != 0.9:
            gen_kwargs["top_p"] = top_p
        if repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = repetition_penalty

        # Generate audio
        wavs, sr = self._model.generate_custom_voice(**gen_kwargs)

        # Convert to int16 PCM
        wav_int16 = np.clip(wavs[0] * 32767, -32768, 32767).astype(np.int16)

        self._sample_rate = sr

        return SynthesisResult(
            audio=wav_int16,
            sample_rate=sr,
            voice=voice,
            text=text,
            metadata={
                "language": language,
                "model_size": self._model_size,
            },
        )

    def stream(
        self,
        text: str,
        voice: str = "ryan",
        language: str = "English",
        chunk_by_sentence: bool = True,
        **kwargs,
    ) -> Iterator[SynthesisResult]:
        """
        Stream audio chunks for real-time playback.

        Splits text into sentences and yields each as it's generated.

        Args:
            text: Text to synthesize
            voice: Speaker voice ID
            language: Language name
            chunk_by_sentence: Split by sentences (True) or fixed size (False)
        """
        from jetson_speech.core.text import split_into_chunks

        chunks = split_into_chunks(text, by_sentence=chunk_by_sentence)

        for i, chunk in enumerate(chunks):
            result = self.synthesize(chunk, voice, language, **kwargs)
            result.metadata["chunk_index"] = i
            result.metadata["total_chunks"] = len(chunks)
            yield result

    def get_voices(self) -> list[Voice]:
        """Get available Qwen voices."""
        voices = []
        for voice_id, info in QWEN_SPEAKERS.items():
            voices.append(
                Voice(
                    id=voice_id,
                    name=info["name"],
                    language="multilingual",
                    gender=info["gender"],
                    description=info["description"],
                    sample_rate=self._sample_rate,
                )
            )
        return voices

    def get_languages(self) -> list[str]:
        """Get supported languages."""
        return QWEN_LANGUAGES

    def unload(self) -> None:
        """Unload model and free GPU memory."""
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
            "model_size": self._model_size,
            "sample_rate": self._sample_rate,
            "languages": self.get_languages(),
        })
        return info
