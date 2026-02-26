"""
Kokoro TTS backend implementation.

Kokoro is a lightweight, near-human-quality TTS model with 82M parameters
and 54+ voices. Outputs 24kHz audio. Requires espeak-ng system dependency.
"""

import logging
from typing import Any, Iterator

logger = logging.getLogger(__name__)

import numpy as np

from jetson_assistant.tts.base import SynthesisResult, TTSBackend, Voice
from jetson_assistant.tts.registry import register_tts_backend

# Kokoro language code mapping
# Pipeline lang_code is set at creation time, not per-synthesis
KOKORO_LANG_CODES = {
    "en": "a",       # American English
    "en-us": "a",    # American English
    "en-gb": "b",    # British English
    "ja": "j",       # Japanese
    "zh": "z",       # Chinese
    "ko": "k",       # Korean
    "fr": "f",       # French
    "es": "e",       # Spanish (not confirmed, placeholder)
    "hi": "h",       # Hindi
    "pt": "p",       # Portuguese (Brazilian)
}

# Default voice for each language code (first char of voice ID = lang code)
KOKORO_LANG_DEFAULT_VOICES = {
    "a": "af_heart",    # American English
    "b": "bf_emma",     # British English
    "j": "jf_alpha",    # Japanese
    "z": "zf_xiaobei",  # Chinese
    "k": "kf_sarah",    # Korean
    "f": "ff_siwis",    # French
    "e": "ef_dora",     # Spanish
    "h": "hf_alpha",    # Hindi
    "p": "pf_dora",     # Portuguese
}

# Key Kokoro voices organized by language prefix and gender
KOKORO_VOICES = {
    # American English - Female
    "af_heart": {"name": "Heart", "language": "en-US", "gender": "female", "description": "Warm, expressive (default)"},
    "af_alloy": {"name": "Alloy", "language": "en-US", "gender": "female", "description": "Balanced, versatile"},
    "af_aoede": {"name": "Aoede", "language": "en-US", "gender": "female", "description": "Clear, musical"},
    "af_bella": {"name": "Bella", "language": "en-US", "gender": "female", "description": "Bright, friendly"},
    "af_jessica": {"name": "Jessica", "language": "en-US", "gender": "female", "description": "Professional, neutral"},
    "af_kore": {"name": "Kore", "language": "en-US", "gender": "female", "description": "Soft, gentle"},
    "af_nicole": {"name": "Nicole", "language": "en-US", "gender": "female", "description": "Warm, conversational"},
    "af_nova": {"name": "Nova", "language": "en-US", "gender": "female", "description": "Energetic, modern"},
    "af_river": {"name": "River", "language": "en-US", "gender": "female", "description": "Calm, soothing"},
    "af_sarah": {"name": "Sarah", "language": "en-US", "gender": "female", "description": "Friendly, natural"},
    "af_sky": {"name": "Sky", "language": "en-US", "gender": "female", "description": "Light, airy"},
    # American English - Male
    "am_adam": {"name": "Adam", "language": "en-US", "gender": "male", "description": "Deep, authoritative"},
    "am_echo": {"name": "Echo", "language": "en-US", "gender": "male", "description": "Clear, resonant"},
    "am_eric": {"name": "Eric", "language": "en-US", "gender": "male", "description": "Professional, neutral"},
    "am_liam": {"name": "Liam", "language": "en-US", "gender": "male", "description": "Young, friendly"},
    "am_michael": {"name": "Michael", "language": "en-US", "gender": "male", "description": "Warm, mature"},
    "am_onyx": {"name": "Onyx", "language": "en-US", "gender": "male", "description": "Rich, deep"},
    # British English - Female
    "bf_emma": {"name": "Emma", "language": "en-GB", "gender": "female", "description": "British, elegant"},
    "bf_isabella": {"name": "Isabella", "language": "en-GB", "gender": "female", "description": "British, refined"},
    # British English - Male
    "bm_george": {"name": "George", "language": "en-GB", "gender": "male", "description": "British, distinguished"},
    "bm_lewis": {"name": "Lewis", "language": "en-GB", "gender": "male", "description": "British, casual"},
    # Hindi - Female
    "hf_alpha": {"name": "Alpha", "language": "hi", "gender": "female", "description": "Hindi, natural"},
    "hf_beta": {"name": "Beta", "language": "hi", "gender": "female", "description": "Hindi, warm"},
    # Hindi - Male
    "hm_omega": {"name": "Omega", "language": "hi", "gender": "male", "description": "Hindi, deep"},
    "hm_psi": {"name": "Psi", "language": "hi", "gender": "male", "description": "Hindi, clear"},
}


@register_tts_backend("kokoro")
class KokoroBackend(TTSBackend):
    """Kokoro TTS backend for near-human-quality synthesis.

    Kokoro is a lightweight 82M parameter model with 54+ voices.
    Outputs 24kHz audio. Requires espeak-ng system dependency.
    """

    name = "kokoro"
    supports_streaming = True
    supports_voice_cloning = False

    def __init__(self):
        super().__init__()
        self._sample_rate = 24000
        self._pipelines: dict[str, Any] = {}  # lang_code -> KPipeline
        self._lang_code = "a"  # Active language code
        self._current_voice = "af_heart"
        self._KPipeline = None  # Cached class reference

    @staticmethod
    def _resolve_lang(lang_code: str) -> str:
        """Resolve a language code to Kokoro's single-char format."""
        resolved = KOKORO_LANG_CODES.get(lang_code.lower(), lang_code)
        if len(resolved) > 1:
            resolved = resolved[0]
        return resolved

    def load(
        self,
        lang_code: str = "a",
        voice: str = "af_heart",
        **kwargs,
    ) -> None:
        """
        Load Kokoro TTS pipeline.

        Args:
            lang_code: Language code ('a'=American, 'b'=British, 'j'=Japanese, etc.)
                       Also accepts full codes like 'en', 'en-us', 'ja'.
            voice: Default voice ID (e.g., 'af_heart', 'am_adam')
        """
        try:
            from kokoro import KPipeline
            self._KPipeline = KPipeline
        except ImportError as e:
            raise ImportError(
                "Kokoro TTS not installed. "
                "Install with: pip install jetson-assistant[kokoro]\n"
                "Also requires: apt-get install espeak-ng"
            ) from e

        resolved_lang = self._resolve_lang(lang_code)
        self._lang_code = resolved_lang
        self._current_voice = voice

        # Blackwell GPUs (sm_12.1+) may not be supported by all PyTorch builds.
        # Standard pip wheels lack sm_12.1 cuBLAS; SBSA wheels work fine.
        # Probe with a small matmul to decide.
        self._force_cpu = False
        try:
            import torch
            if torch.cuda.is_available():
                a = torch.randn(8, 8, device="cuda")
                _ = a @ a.T  # triggers cuBLAS
                logger.info("CUDA matmul probe: OK — Kokoro will use GPU")

                # Blackwell (sm_12x): PyTorch's built-in SDPA Cutlass FMHA
                # kernels only support sm80-sm100.  On sm121 they spam FATAL
                # errors and hang TTS synthesis.  Disable them and use the
                # math SDPA backend (plenty fast for Kokoro's 82M params).
                cap = torch.cuda.get_device_capability()
                if cap[0] >= 12:
                    torch.backends.cuda.enable_flash_sdp(False)
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                    logger.info(
                        "Disabled flash/efficient SDPA for sm_%d%d — using math backend",
                        cap[0], cap[1],
                    )
        except Exception as e:
            self._force_cpu = True
            logger.info("CUDA matmul probe failed (%s) — Kokoro will use CPU", e)

        logger.info("Loading Kokoro TTS (lang=%s, voice=%s)...", resolved_lang, voice)

        pipeline = KPipeline(lang_code=resolved_lang)

        # Move model to CPU if GPU is unsupported (setting CUDA_VISIBLE_DEVICES
        # doesn't work after torch is initialized — must move tensors explicitly)
        if self._force_cpu and hasattr(pipeline, 'model') and pipeline.model is not None:
            pipeline.model = pipeline.model.cpu()
            logger.info("Kokoro model moved to CPU")

        self._pipelines[resolved_lang] = pipeline

        # Pre-load voice weights to avoid first-call delay
        logger.info("Pre-loading voice %s...", voice)
        try:
            for _ in pipeline("warmup", voice=voice):
                pass
        except Exception:
            pass  # Voice will download on first synthesize if this fails

        self._loaded = True
        logger.info("Kokoro model loaded")

    def switch_language(self, lang_code: str, voice: str | None = None) -> str:
        """Switch the active language, creating a new pipeline if needed.

        Args:
            lang_code: Language code ('en', 'hi', 'ja', etc. or single-char 'a', 'h', 'j')
            voice: Voice to use. If None, picks the default for that language.

        Returns:
            Confirmation message.
        """
        if not self._loaded or self._KPipeline is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        resolved = self._resolve_lang(lang_code)

        # Pick default voice for the language if not specified
        if voice is None:
            voice = KOKORO_LANG_DEFAULT_VOICES.get(resolved, f"{resolved}f_alpha")

        # Create pipeline on first use for this language
        if resolved not in self._pipelines:
            logger.info("Kokoro: loading pipeline for lang=%s...", resolved)
            pipeline = self._KPipeline(lang_code=resolved)
            self._pipelines[resolved] = pipeline

            # Warm up the voice
            logger.info("Kokoro: pre-loading voice %s...", voice)
            try:
                for _ in pipeline("warmup", voice=voice):
                    pass
            except Exception:
                pass

            logger.info("Kokoro: lang=%s ready", resolved)

        self._lang_code = resolved
        self._current_voice = voice

        lang_names = {"a": "American English", "b": "British English", "h": "Hindi",
                      "j": "Japanese", "z": "Chinese", "k": "Korean", "f": "French",
                      "e": "Spanish", "p": "Portuguese"}
        name = lang_names.get(resolved, resolved)
        return f"Switched to {name} (voice: {voice})."

    @property
    def _pipeline(self):
        """Get the active pipeline for the current language."""
        return self._pipelines.get(self._lang_code)

    def synthesize(
        self,
        text: str,
        voice: str = "default",
        language: str = "en",
        speed: float = 1.0,
        **kwargs,
    ) -> SynthesisResult:
        """
        Generate speech from text.

        Args:
            text: Text to synthesize
            voice: Voice ID (e.g., 'af_heart'). 'default' uses the loaded voice.
            language: Language code. If different from active, switches pipeline.
            speed: Speaking rate multiplier (default 1.0)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if voice == "default":
            voice = self._current_voice

        pipeline = self._pipeline
        if pipeline is None:
            raise RuntimeError(f"No pipeline for lang_code={self._lang_code}")

        # Collect all audio chunks from the generator
        audio_chunks = []
        for _graphemes, _phonemes, audio_chunk in pipeline(
            text, voice=voice, speed=speed
        ):
            if audio_chunk is not None:
                # Kokoro outputs float32 tensors, convert to int16
                chunk_np = self._to_int16(audio_chunk)
                audio_chunks.append(chunk_np)

        if audio_chunks:
            audio = np.concatenate(audio_chunks)
        else:
            audio = np.array([], dtype=np.int16)

        return SynthesisResult(
            audio=audio,
            sample_rate=self._sample_rate,
            voice=voice,
            text=text,
            metadata={
                "backend": "kokoro",
                "lang_code": self._lang_code,
                "speed": speed,
            },
        )

    def stream(
        self,
        text: str,
        voice: str = "default",
        language: str = "en",
        speed: float = 1.0,
        **kwargs,
    ) -> Iterator[SynthesisResult]:
        """
        Stream audio chunks per sentence.

        Kokoro's generator naturally yields per-sentence audio, making
        this ideal for pipelined TTS (synthesize while LLM is still generating).
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if voice == "default":
            voice = self._current_voice

        pipeline = self._pipeline
        if pipeline is None:
            raise RuntimeError(f"No pipeline for lang_code={self._lang_code}")

        chunk_index = 0
        for _graphemes, _phonemes, audio_chunk in pipeline(
            text, voice=voice, speed=speed
        ):
            if audio_chunk is not None:
                audio = self._to_int16(audio_chunk)

                yield SynthesisResult(
                    audio=audio,
                    sample_rate=self._sample_rate,
                    voice=voice,
                    text=text,
                    metadata={
                        "backend": "kokoro",
                        "chunk_index": chunk_index,
                        "lang_code": self._lang_code,
                    },
                )
                chunk_index += 1

    def _to_int16(self, audio) -> np.ndarray:
        """Convert Kokoro's float32 output to int16 PCM."""
        # Handle both torch tensors and numpy arrays
        if hasattr(audio, "cpu"):
            audio_np = audio.cpu().numpy()
        elif isinstance(audio, np.ndarray):
            audio_np = audio
        else:
            audio_np = np.array(audio, dtype=np.float32)

        # Ensure float32
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)

        # Flatten if multi-dimensional
        audio_np = audio_np.flatten()

        # Clip and convert to int16
        audio_np = np.clip(audio_np, -1.0, 1.0)
        return (audio_np * 32767).astype(np.int16)

    def get_voices(self) -> list[Voice]:
        """Get available Kokoro voices."""
        voices = []
        for voice_id, info in KOKORO_VOICES.items():
            voices.append(
                Voice(
                    id=voice_id,
                    name=info["name"],
                    language=info["language"],
                    gender=info["gender"],
                    description=info["description"],
                    sample_rate=24000,
                )
            )
        return voices

    def get_languages(self) -> list[str]:
        """Get supported languages."""
        return ["en", "en-gb", "ja", "zh", "ko", "fr", "es", "hi", "pt"]

    def unload(self) -> None:
        """Unload all pipelines and free memory."""
        if self._pipelines:
            for lang, pipeline in self._pipelines.items():
                del pipeline
            self._pipelines.clear()

            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        self._KPipeline = None
        self._loaded = False

    def get_info(self) -> dict[str, Any]:
        """Get backend information."""
        info = super().get_info()
        info.update({
            "current_voice": self._current_voice,
            "lang_code": self._lang_code,
            "sample_rate": self._sample_rate,
            "available_voices": list(KOKORO_VOICES.keys()),
        })
        return info
