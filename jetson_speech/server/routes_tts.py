"""
TTS REST API routes.
"""

import base64
from typing import Any

from fastapi import APIRouter, HTTPException, Response

from jetson_speech.server.app import get_engine
from jetson_speech.server.schemas import (
    BackendInfo,
    BackendListResponse,
    LoadBackendRequest,
    LoadBackendResponse,
    SynthesizeRequest,
    SynthesizeResponse,
    VoiceInfo,
    VoiceListResponse,
)
from jetson_speech.tts.registry import list_tts_backends

router = APIRouter()


@router.get("/backends", response_model=BackendListResponse)
async def list_backends() -> BackendListResponse:
    """List available TTS backends."""
    backends = []

    for info in list_tts_backends():
        backends.append(
            BackendInfo(
                name=info["name"],
                loaded=False,  # Will be updated below
                supports_streaming=info.get("supports_streaming", False),
                supports_voice_cloning=info.get("supports_voice_cloning", False),
            )
        )

    # Check which is loaded
    engine = get_engine()
    tts_info = engine.get_tts_info()
    if tts_info.get("loaded"):
        for backend in backends:
            if backend.name == tts_info.get("name"):
                backend.loaded = True
                break

    return BackendListResponse(backends=backends)


@router.post("/backends/{name}/load", response_model=LoadBackendResponse)
async def load_backend(name: str, request: LoadBackendRequest) -> LoadBackendResponse:
    """Load a TTS backend."""
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
        engine.load_tts_backend(name, **kwargs)

        # Get backend info
        info = engine.get_tts_info()

        return LoadBackendResponse(
            success=True,
            message=f"Loaded TTS backend: {name}",
            backend=BackendInfo(
                name=info["name"],
                loaded=True,
                supports_streaming=info.get("supports_streaming", False),
                supports_voice_cloning=info.get("supports_voice_cloning", False),
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
    """Unload the current TTS backend."""
    engine = get_engine()
    engine.unload_tts_backend()
    return {"success": True, "message": "TTS backend unloaded"}


@router.get("/voices", response_model=VoiceListResponse)
async def get_voices() -> VoiceListResponse:
    """Get available voices for the current backend."""
    engine = get_engine()

    if not engine.get_tts_info().get("loaded"):
        raise HTTPException(status_code=400, detail="No TTS backend loaded")

    voices = []
    for voice in engine.get_tts_voices():
        voices.append(
            VoiceInfo(
                id=voice.id,
                name=voice.name,
                language=voice.language,
                gender=voice.gender,
                description=voice.description,
                sample_rate=voice.sample_rate,
            )
        )

    languages = engine.get_tts_languages()

    return VoiceListResponse(voices=voices, languages=languages)


@router.post("/synthesize")
async def synthesize(request: SynthesizeRequest) -> Response:
    """
    Synthesize speech from text.

    Returns WAV audio data by default.
    """
    engine = get_engine()

    if not engine.get_tts_info().get("loaded"):
        raise HTTPException(status_code=400, detail="No TTS backend loaded")

    try:
        result = engine.synthesize(
            text=request.text,
            voice=request.voice,
            language=request.language,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        # Return WAV audio
        audio_bytes = result.to_bytes()

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'attachment; filename="speech.wav"',
                "X-Duration": str(result.duration),
                "X-Sample-Rate": str(result.sample_rate),
                "X-Voice": result.voice,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/synthesize/json", response_model=SynthesizeResponse)
async def synthesize_json(request: SynthesizeRequest) -> SynthesizeResponse:
    """
    Synthesize speech and return JSON response with base64 audio.
    """
    engine = get_engine()

    if not engine.get_tts_info().get("loaded"):
        return SynthesizeResponse(
            success=False,
            duration=0,
            sample_rate=0,
            voice="",
            error="No TTS backend loaded",
        )

    try:
        result = engine.synthesize(
            text=request.text,
            voice=request.voice,
            language=request.language,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        # Encode audio as base64
        audio_bytes = result.to_bytes()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return SynthesizeResponse(
            success=True,
            duration=result.duration,
            sample_rate=result.sample_rate,
            voice=result.voice,
            audio_base64=audio_base64,
        )

    except Exception as e:
        return SynthesizeResponse(
            success=False,
            duration=0,
            sample_rate=0,
            voice="",
            error=str(e),
        )
