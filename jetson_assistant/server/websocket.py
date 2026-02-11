"""
WebSocket endpoints for streaming TTS and STT.
"""

import asyncio
import base64
import json
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from jetson_assistant.server.app import get_engine

router = APIRouter()


@router.websocket("/tts/stream")
async def tts_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming TTS.

    Protocol:
    1. Client connects
    2. Client sends: {"text": "Hello", "voice": "ryan", "language": "English"}
    3. Server sends: {"type": "start", "chunks": N}
    4. Server sends: {"type": "audio", "chunk": 1, "data": "<base64 wav>"}
    5. ... more audio chunks ...
    6. Server sends: {"type": "done", "total_time": 5.2}
    """
    await websocket.accept()

    engine = get_engine()

    try:
        while True:
            # Receive request
            data = await websocket.receive_text()
            request = json.loads(data)

            text = request.get("text", "")
            voice = request.get("voice")
            language = request.get("language")

            if not text:
                await websocket.send_json({"type": "error", "error": "No text provided"})
                continue

            if not engine.get_tts_info().get("loaded"):
                await websocket.send_json({"type": "error", "error": "No TTS backend loaded"})
                continue

            try:
                start_time = time.time()

                # Get chunks for streaming
                from jetson_assistant.core.text import split_into_chunks

                chunks = split_into_chunks(text, by_sentence=True)

                # Send start message
                await websocket.send_json({
                    "type": "start",
                    "chunks": len(chunks),
                })

                # Stream each chunk
                for i, chunk in enumerate(chunks):
                    # Synthesize chunk
                    result = engine.synthesize(chunk, voice, language)

                    # Encode audio
                    audio_bytes = result.to_bytes()
                    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

                    # Send audio chunk
                    await websocket.send_json({
                        "type": "audio",
                        "chunk": i + 1,
                        "data": audio_base64,
                        "duration": result.duration,
                    })

                    # Small delay to prevent overwhelming the client
                    await asyncio.sleep(0.01)

                # Send done message
                total_time = time.time() - start_time
                await websocket.send_json({
                    "type": "done",
                    "total_time": total_time,
                })

            except Exception as e:
                await websocket.send_json({"type": "error", "error": str(e)})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass


@router.websocket("/stt/stream")
async def stt_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming STT.

    Protocol:
    1. Client connects
    2. Client sends: {"type": "start", "language": "en", "sample_rate": 16000}
    3. Client sends binary audio chunks
    4. Server sends: {"type": "segment", "text": "hello", "start": 0.0, "end": 1.0}
    5. Client sends: {"type": "stop"}
    6. Server sends: {"type": "done", "text": "full transcription"}
    """
    await websocket.accept()

    engine = get_engine()

    sample_rate = 16000
    language = None
    audio_buffer = b""
    all_text_parts = []

    try:
        while True:
            message = await websocket.receive()

            # Handle binary audio data
            if "bytes" in message:
                audio_buffer += message["bytes"]

                # Process when we have enough data (e.g., 1 second of audio)
                chunk_samples = sample_rate * 1  # 1 second
                chunk_bytes = chunk_samples * 2  # 16-bit audio

                while len(audio_buffer) >= chunk_bytes:
                    # Extract chunk
                    chunk_data = audio_buffer[:chunk_bytes]
                    audio_buffer = audio_buffer[chunk_bytes:]

                    # Convert to numpy
                    import numpy as np

                    audio = np.frombuffer(chunk_data, dtype=np.int16)

                    # Transcribe if backend is loaded
                    if engine.get_stt_info().get("loaded"):
                        try:
                            result = engine.transcribe(audio, sample_rate, language)
                            if result.text.strip():
                                all_text_parts.append(result.text.strip())
                                await websocket.send_json({
                                    "type": "segment",
                                    "text": result.text.strip(),
                                    "is_final": False,
                                })
                        except Exception as e:
                            await websocket.send_json({
                                "type": "error",
                                "error": str(e),
                            })

            # Handle text messages (control messages)
            elif "text" in message:
                data = json.loads(message["text"])
                msg_type = data.get("type", "")

                if msg_type == "start":
                    # Initialize streaming session
                    sample_rate = data.get("sample_rate", 16000)
                    language = data.get("language")
                    audio_buffer = b""
                    all_text_parts = []

                    if not engine.get_stt_info().get("loaded"):
                        await websocket.send_json({
                            "type": "error",
                            "error": "No STT backend loaded",
                        })
                    else:
                        await websocket.send_json({"type": "ready"})

                elif msg_type == "stop":
                    # Process remaining audio
                    if audio_buffer and engine.get_stt_info().get("loaded"):
                        import numpy as np

                        audio = np.frombuffer(audio_buffer, dtype=np.int16)
                        if len(audio) > 0:
                            try:
                                result = engine.transcribe(audio, sample_rate, language)
                                if result.text.strip():
                                    all_text_parts.append(result.text.strip())
                            except Exception:
                                pass

                    # Send final result
                    full_text = " ".join(all_text_parts)
                    await websocket.send_json({
                        "type": "done",
                        "text": full_text,
                    })

                    # Reset state
                    audio_buffer = b""
                    all_text_parts = []

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass
