"""
STT backend registry for discovery and instantiation.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jetson_assistant.stt.base import STTBackend

# Registry of available backends
_stt_backends: dict[str, type["STTBackend"]] = {}


def register_stt_backend(name: str):
    """
    Decorator to register an STT backend.

    Args:
        name: Backend name (e.g., "whisper", "sensevoice")

    Example:
        @register_stt_backend("whisper")
        class WhisperBackend(STTBackend):
            ...
    """

    def decorator(cls: type["STTBackend"]) -> type["STTBackend"]:
        cls.name = name
        _stt_backends[name] = cls
        return cls

    return decorator


def get_stt_backend(name: str) -> "STTBackend":
    """
    Get an STT backend instance by name.

    Args:
        name: Backend name

    Returns:
        STTBackend instance

    Raises:
        ValueError: If backend not found
    """
    # Lazy import backends to avoid import errors when dependencies missing
    _discover_backends()

    if name not in _stt_backends:
        available = ", ".join(_stt_backends.keys())
        raise ValueError(f"STT backend '{name}' not found. Available: {available}")

    return _stt_backends[name]()


def list_stt_backends() -> list[dict]:
    """
    List all available STT backends.

    Returns:
        List of backend info dictionaries
    """
    _discover_backends()

    result = []
    for name, cls in _stt_backends.items():
        result.append({
            "name": name,
            "class": cls.__name__,
            "supports_streaming": getattr(cls, "supports_streaming", False),
        })
    return result


def _discover_backends() -> None:
    """Discover and import available backends."""
    # Try to import each backend module
    # They will auto-register via the decorator

    # Whisper backend
    try:
        from jetson_assistant.stt import whisper  # noqa: F401
    except ImportError:
        pass

    # SenseVoice backend
    try:
        from jetson_assistant.stt import sensevoice  # noqa: F401
    except ImportError:
        pass

    # vLLM Whisper backend (GPU-accelerated via remote vLLM server)
    try:
        from jetson_assistant.stt import vllm_whisper  # noqa: F401
    except ImportError:
        pass

    # Nemotron Speech backend (NVIDIA NeMo, fast English STT)
    try:
        from jetson_assistant.stt import nemotron  # noqa: F401
    except ImportError:
        pass

    # NemotronFast backend (direct forward, no Lhotse overhead)
    try:
        from jetson_assistant.stt import nemotron_fast  # noqa: F401
    except ImportError:
        pass
