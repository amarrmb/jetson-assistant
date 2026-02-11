"""
STT (Speech-to-Text) backends.
"""

from jetson_assistant.stt.base import STTBackend, TranscriptionResult, TranscriptionSegment
from jetson_assistant.stt.registry import get_stt_backend, list_stt_backends, register_stt_backend

__all__ = [
    "STTBackend",
    "TranscriptionResult",
    "TranscriptionSegment",
    "register_stt_backend",
    "get_stt_backend",
    "list_stt_backends",
]
