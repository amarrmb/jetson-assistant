"""
FastAPI application for Jetson Assistant server.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from jetson_assistant import __version__
from jetson_assistant.config import get_config, get_jetson_power_mode, is_jetson
from jetson_assistant.core.engine import Engine
from jetson_assistant.server.schemas import HealthResponse

# Global engine instance
_engine: Engine | None = None


def get_engine() -> Engine:
    """Get the global engine instance."""
    global _engine
    if _engine is None:
        _engine = Engine()
    return _engine


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    import os

    # Startup
    config = get_config()
    engine = get_engine()

    # Check for preload options from environment
    preload_tts = os.environ.get("JETSON_ASSISTANT_PRELOAD_TTS")
    preload_stt = os.environ.get("JETSON_ASSISTANT_PRELOAD_STT")
    do_warmup = os.environ.get("JETSON_ASSISTANT_WARMUP") == "1"

    # Preload TTS backend if specified
    if preload_tts:
        logger.info("Preloading TTS backend: %s...", preload_tts)
        engine.load_tts_backend(preload_tts)
        logger.info("TTS backend loaded")

        # Warmup with a short synthesis
        if do_warmup:
            logger.info("Warming up TTS...")
            engine.synthesize("Hello.", voice="serena", language="English")
            logger.info("TTS warmup complete")

    # Preload STT backend if specified
    if preload_stt:
        logger.info("Preloading STT backend: %s...", preload_stt)
        engine.load_stt_backend(preload_stt)
        logger.info("STT backend loaded")

    yield

    # Shutdown
    engine.unload_tts_backend()
    engine.unload_stt_backend()


def create_app(
    cors_origins: list[str] | None = None,
    enable_webui: bool = False,
) -> FastAPI:
    """
    Create the FastAPI application.

    Args:
        cors_origins: CORS allowed origins (default: ["*"])
        enable_webui: Enable Gradio web UI at /ui

    Returns:
        FastAPI application
    """
    config = get_config()

    app = FastAPI(
        title="Jetson Assistant API",
        description="Modular TTS + STT server for Jetson/edge devices",
        version=__version__,
        lifespan=lifespan,
    )

    # CORS middleware
    origins = cors_origins or config.server.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    from jetson_assistant.server.routes_stt import router as stt_router
    from jetson_assistant.server.routes_tts import router as tts_router
    from jetson_assistant.server.routes_llm import router as llm_router
    from jetson_assistant.server.routes_rag import router as rag_router
    from jetson_assistant.server.websocket import router as ws_router

    app.include_router(tts_router, prefix="/tts", tags=["TTS"])
    app.include_router(stt_router, prefix="/stt", tags=["STT"])
    app.include_router(llm_router, tags=["LLM"])
    app.include_router(rag_router, tags=["RAG"])
    app.include_router(ws_router, tags=["WebSocket"])

    # Health check endpoint
    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Health check endpoint."""
        engine = get_engine()
        tts_info = engine.get_tts_info()
        stt_info = engine.get_stt_info()

        return HealthResponse(
            status="ok",
            version=__version__,
            is_jetson=is_jetson(),
            power_mode=get_jetson_power_mode(),
            tts_backend=tts_info.get("name") if tts_info.get("loaded") else None,
            stt_backend=stt_info.get("name") if stt_info.get("loaded") else None,
        )

    @app.get("/info")
    async def get_info() -> dict[str, Any]:
        """Get detailed server information."""
        engine = get_engine()
        return engine.get_info()

    # Mount Gradio UI if enabled
    if enable_webui:
        try:
            import gradio as gr
            from webui.app import create_gradio_app

            gradio_app = create_gradio_app()
            app = gr.mount_gradio_app(app, gradio_app, path="/ui")
        except ImportError:
            pass  # Gradio not installed

    return app


def run_server(
    host: str | None = None,
    port: int | None = None,
    reload: bool = False,
    workers: int = 1,
    enable_webui: bool = False,
) -> None:
    """
    Run the server.

    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload (development)
        workers: Number of workers
        enable_webui: Enable Gradio web UI
    """
    import uvicorn

    config = get_config()

    uvicorn.run(
        "jetson_assistant.server.app:create_app",
        factory=True,
        host=host or config.server.host,
        port=port or config.server.port,
        reload=reload,
        workers=workers,
    )
