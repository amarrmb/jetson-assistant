"""
TTS backend registry for discovery and instantiation.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jetson_speech.tts.base import TTSBackend

# Registry of available backends
_tts_backends: dict[str, type["TTSBackend"]] = {}


def register_tts_backend(name: str):
    """
    Decorator to register a TTS backend.

    Args:
        name: Backend name (e.g., "qwen", "piper")

    Example:
        @register_tts_backend("qwen")
        class QwenBackend(TTSBackend):
            ...
    """

    def decorator(cls: type["TTSBackend"]) -> type["TTSBackend"]:
        cls.name = name
        _tts_backends[name] = cls
        return cls

    return decorator


def get_tts_backend(name: str) -> "TTSBackend":
    """
    Get a TTS backend instance by name.

    Args:
        name: Backend name

    Returns:
        TTSBackend instance

    Raises:
        ValueError: If backend not found
    """
    # Lazy import backends to avoid import errors when dependencies missing
    _discover_backends()

    if name not in _tts_backends:
        available = ", ".join(_tts_backends.keys())
        raise ValueError(f"TTS backend '{name}' not found. Available: {available}")

    return _tts_backends[name]()


def list_tts_backends() -> list[dict]:
    """
    List all available TTS backends.

    Returns:
        List of backend info dictionaries
    """
    _discover_backends()

    result = []
    for name, cls in _tts_backends.items():
        result.append({
            "name": name,
            "class": cls.__name__,
            "supports_streaming": getattr(cls, "supports_streaming", False),
            "supports_voice_cloning": getattr(cls, "supports_voice_cloning", False),
        })
    return result


def _discover_backends() -> None:
    """Discover and import available backends."""
    # Try to import each backend module
    # They will auto-register via the decorator

    # Qwen backend
    try:
        from jetson_speech.tts import qwen  # noqa: F401
    except ImportError:
        pass

    # Piper backend
    try:
        from jetson_speech.tts import piper  # noqa: F401
    except ImportError:
        pass

    # Kokoro backend
    try:
        from jetson_speech.tts import kokoro  # noqa: F401
    except ImportError:
        pass
