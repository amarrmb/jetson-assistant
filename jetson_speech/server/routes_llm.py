"""
LLM/Chat routes for the speech server.

Provides chat endpoints that integrate LLM with optional RAG.
"""

import logging
import time

logger = logging.getLogger(__name__)
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from jetson_speech.server.schemas import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
)

router = APIRouter(prefix="/llm", tags=["llm"])

# Global LLM instance (loaded once)
_llm_instance = None
_llm_model = None

# Global RAG instances (keyed by collection name)
_rag_instances: dict = {}


def get_llm(model: Optional[str] = None):
    """Get or create LLM instance."""
    global _llm_instance, _llm_model

    model = model or "llama3.2:3b"

    if _llm_instance is None or _llm_model != model:
        try:
            import ollama
            _llm_instance = ollama.Client(host="http://localhost:11434")
            _llm_model = model

            # Check if model is available, pull if not
            try:
                _llm_instance.show(model)
                logger.info("LLM ready: %s", model)
            except Exception:
                logger.info("Pulling model: %s...", model)
                _llm_instance.pull(model)
                logger.info("LLM ready: %s", model)

        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="Ollama not installed. Install with: pip install ollama"
            )

    return _llm_instance, model


def get_rag(collection: str):
    """Get or create RAG instance for a collection."""
    global _rag_instances

    if collection not in _rag_instances:
        try:
            from jetson_speech.rag import RAGPipeline
            rag = RAGPipeline(collection_name=collection)
            if rag.count() == 0:
                return None  # Collection is empty
            _rag_instances[collection] = rag
            logger.info("RAG loaded: %s (%d chunks)", collection, rag.count())
        except ImportError:
            return None

    return _rag_instances.get(collection)


# Default system prompts
DEFAULT_SYSTEM_PROMPT = "You are a helpful voice assistant. Keep responses concise (1-3 sentences). Speak naturally without formatting."

RAG_SYSTEM_PROMPT = """You are a knowledgeable assistant. Use the provided context to give accurate, educational answers. Explain WHY things work, not just what to do. Keep answers under 30 words but make them insightful."""


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Process a chat request with optional RAG context.

    This is the main endpoint for LLM interaction. It:
    1. Optionally retrieves context from RAG
    2. Builds the prompt with context
    3. Calls the LLM
    4. Returns the response
    """
    start_time = time.perf_counter()

    try:
        client, model = get_llm(request.llm_model)
    except HTTPException as e:
        return ChatResponse(
            success=False,
            response="",
            model="",
            error=str(e.detail)
        )

    # Build messages
    system_prompt = request.system_prompt or DEFAULT_SYSTEM_PROMPT
    rag_context = None
    rag_sources = []

    # Get RAG context if requested
    if request.use_rag and request.rag_collection:
        rag = get_rag(request.rag_collection)
        if rag:
            results = rag.retrieve(request.message, top_k=request.rag_top_k)
            if results:
                rag_context = "\n\n".join([r["content"] for r in results])
                rag_sources = list(set(r.get("metadata", {}).get("source", "unknown") for r in results))
                system_prompt = request.system_prompt or RAG_SYSTEM_PROMPT

    # Build message list
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation context
    for msg in request.context:
        messages.append({"role": msg.role, "content": msg.content})

    # Add current message (with RAG context if available)
    if rag_context:
        user_content = f"Context:\n{rag_context}\n\nQuestion: {request.message}"
    else:
        user_content = request.message

    messages.append({"role": "user", "content": user_content})

    # Call LLM
    try:
        response = client.chat(
            model=model,
            messages=messages,
            options={"num_predict": 100},
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return ChatResponse(
            success=True,
            response=response["message"]["content"],
            model=model,
            tokens_used=response.get("eval_count", 0),
            latency_ms=latency_ms,
            rag_context=rag_context,
            rag_sources=rag_sources,
        )

    except Exception as e:
        return ChatResponse(
            success=False,
            response="",
            model=model,
            error=str(e)
        )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream chat response using Server-Sent Events (SSE).

    Yields chunks as they're generated, enabling pipelined TTS.
    """
    import json
    import re

    try:
        client, model = get_llm(request.llm_model)
    except HTTPException as e:
        async def error_generator():
            yield f"data: {json.dumps({'type': 'error', 'error': str(e.detail)})}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")

    # Build messages (same as non-streaming)
    system_prompt = request.system_prompt or DEFAULT_SYSTEM_PROMPT
    rag_context = None

    if request.use_rag and request.rag_collection:
        rag = get_rag(request.rag_collection)
        if rag:
            results = rag.retrieve(request.message, top_k=request.rag_top_k)
            if results:
                rag_context = "\n\n".join([r["content"] for r in results])
                system_prompt = request.system_prompt or RAG_SYSTEM_PROMPT

    messages = [{"role": "system", "content": system_prompt}]
    for msg in request.context:
        messages.append({"role": msg.role, "content": msg.content})

    if rag_context:
        user_content = f"Context:\n{rag_context}\n\nQuestion: {request.message}"
    else:
        user_content = request.message
    messages.append({"role": "user", "content": user_content})

    async def generate():
        """Generator that yields SSE chunks."""
        # Send start message
        yield f"data: {json.dumps({'type': 'start', 'model': model})}\n\n"

        buffer = ""
        sentence_pattern = re.compile(r'[.!?:]\s+')

        try:
            for chunk in client.chat(
                model=model,
                messages=messages,
                stream=True,
                options={"num_predict": 100},
            ):
                if chunk.get("done", False):
                    break

                content = chunk.get("message", {}).get("content", "")
                if not content:
                    continue

                buffer += content

                # Check for complete sentences
                match = sentence_pattern.search(buffer)
                if match:
                    end_pos = match.end()
                    sentence = buffer[:end_pos].strip()
                    buffer = buffer[end_pos:].strip()

                    if sentence and len(sentence) > 2:
                        yield f"data: {json.dumps({'type': 'chunk', 'content': sentence})}\n\n"

            # Yield remaining buffer
            if buffer.strip():
                yield f"data: {json.dumps({'type': 'chunk', 'content': buffer.strip()})}\n\n"

            # Send done message
            yield f"data: {json.dumps({'type': 'done', 'done': True})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/models")
async def list_models():
    """List available LLM models from Ollama."""
    try:
        import ollama
        client = ollama.Client(host="http://localhost:11434")
        models = client.list()
        return {
            "success": True,
            "models": [m["name"] for m in models.get("models", [])]
        }
    except Exception as e:
        return {
            "success": False,
            "models": [],
            "error": str(e)
        }


@router.post("/models/{name}/pull")
async def pull_model(name: str):
    """Pull/download an LLM model."""
    try:
        import ollama
        client = ollama.Client(host="http://localhost:11434")
        client.pull(name)
        return {"success": True, "model": name, "message": f"Model {name} pulled successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
