"""
TTS (Text-to-Speech) backends.
"""

from jetson_speech.tts.base import SynthesisResult, TTSBackend, Voice
from jetson_speech.tts.registry import get_tts_backend, list_tts_backends, register_tts_backend

__all__ = [
    "TTSBackend",
    "Voice",
    "SynthesisResult",
    "register_tts_backend",
    "get_tts_backend",
    "list_tts_backends",
]
