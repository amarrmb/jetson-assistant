"""
STT REST API routes.
"""

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from jetson_assistant.server.app import get_engine
from jetson_assistant.server.schemas import (
    BackendInfo,
    BackendListResponse,
    LoadBackendRequest,
    LoadBackendResponse,
    TranscribeResponse,
)
from jetson_assistant.stt.registry import list_stt_backends

router = APIRouter()


@router.get("/backends", response_model=BackendListResponse)
async def list_backends() -> BackendListResponse:
    """List available STT backends."""
    backends = []

    for info in list_stt_backends():
        backends.append(
            BackendInfo(
                name=info["name"],
                loaded=False,
                supports_streaming=info.get("supports_streaming", False),
            )
        )

    # Check which is loaded
    engine = get_engine()
    stt_info = engine.get_stt_info()
    if stt_info.get("loaded"):
        for backend in backends:
            if backend.name == stt_info.get("name"):
                backend.loaded = True
                break

    return BackendListResponse(backends=backends)


@router.post("/backends/{name}/load", response_model=LoadBackendResponse)
async def load_backend(name: str, request: LoadBackendRequest) -> LoadBackendResponse:
    """Load an STT backend."""
    engine = get_engine()

    try:
        # Build load kwargs
        kwargs: dict[str, Any] = {}
        if request.model_size:
            kwargs["model_size"] = request.model_size
        if request.device:
            kwargs["device"] = request.device
        kwargs.update(request.options)

        # Load backend
        engine.load_stt_backend(name, **kwargs)

        # Get backend info
        info = engine.get_stt_info()

        return LoadBackendResponse(
            success=True,
            message=f"Loaded STT backend: {name}",
            backend=BackendInfo(
                name=info["name"],
                loaded=True,
                supports_streaming=info.get("supports_streaming", False),
                extra=info,
            ),
        )

    except ValueError as e:
        return LoadBackendResponse(
            success=False,
            message=str(e),
        )
    except Exception as e:
        return LoadBackendResponse(
            success=False,
            message=f"Failed to load backend: {e}",
        )


@router.post("/backends/unload")
async def unload_backend() -> dict[str, Any]:
    """Unload the current STT backend."""
    engine = get_engine()
    engine.unload_stt_backend()
    return {"success": True, "message": "STT backend unloaded"}


@router.get("/languages")
async def get_languages() -> dict[str, Any]:
    """Get supported languages for the current backend."""
    engine = get_engine()

    stt_info = engine.get_stt_info()
    if not stt_info.get("loaded"):
        raise HTTPException(status_code=400, detail="No STT backend loaded")

    # Get languages from backend
    backend = engine._stt_backend
    if backend:
        languages = backend.get_languages()
    else:
        languages = []

    return {"languages": languages}


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    audio: UploadFile = File(...),
    language: str | None = Form(None),
) -> TranscribeResponse:
    """
    Transcribe audio file to text.

    Accepts WAV, MP3, or other audio formats.
    """
    engine = get_engine()

    if not engine.get_stt_info().get("loaded"):
        return TranscribeResponse(
            success=False,
            text="",
            language="",
            duration=0,
            error="No STT backend loaded",
        )

    try:
        import tempfile
        from pathlib import Path

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            suffix=Path(audio.filename or "audio.wav").suffix,
            delete=False,
        ) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Transcribe
            result = engine.transcribe(tmp_path, language=language)

            return TranscribeResponse(
                success=True,
                text=result.text,
                language=result.language,
                duration=result.duration,
                segments=[s.to_dict() for s in result.segments],
            )

        finally:
            # Clean up temp file
            import os

            os.unlink(tmp_path)

    except Exception as e:
        return TranscribeResponse(
            success=False,
            text="",
            language="",
            duration=0,
            error=str(e),
        )


@router.get("/info")
async def get_info() -> dict[str, Any]:
    """Get STT backend information."""
    engine = get_engine()
    return engine.get_stt_info()
