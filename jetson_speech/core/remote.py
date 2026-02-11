"""
Remote engine that connects to a running jetson-speech server.

This is the recommended approach for the voice assistant:
1. Start server once: `jetson-speech serve`
2. Assistant connects to server for fast TTS/STT
"""

import io
from dataclasses import dataclass
from typing import Optional

import numpy as np

from jetson_speech.tts.base import SynthesisResult
from jetson_speech.stt.base import TranscriptionResult, TranscriptionSegment


@dataclass
class RemoteEngineConfig:
    """Configuration for remote engine."""

    host: str = "localhost"
    port: int = 8080
    timeout: float = 120.0  # Long timeout for TTS synthesis

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class RemoteEngine:
    """
    Engine that connects to a running jetson-speech server.

    Much faster than loading models in-process because:
    - Models are already loaded and warm on the server
    - Just HTTP calls, no initialization overhead

    Usage:
        # Start server first: jetson-speech serve

        engine = RemoteEngine()
        engine.wait_for_server()

        # Fast TTS
        result = engine.synthesize("Hello world")

        # Fast STT
        text = engine.transcribe(audio_data)
    """

    def __init__(self, config: Optional[RemoteEngineConfig] = None):
        """
        Initialize remote engine.

        Args:
            config: Server connection config
        """
        try:
            import httpx
        except ImportError as e:
            raise ImportError(
                "httpx not installed. Install with: pip install httpx"
            ) from e

        self.config = config or RemoteEngineConfig()
        self._client = httpx.Client(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )
        self._connected = False

    def wait_for_server(self, timeout: float = 60.0, interval: float = 1.0) -> bool:
        """
        Wait for server to be ready.

        Args:
            timeout: Maximum time to wait
            interval: Time between checks

        Returns:
            True if server is ready
        """
        import time

        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = self._client.get("/health")
                if resp.status_code == 200:
                    self._connected = True
                    return True
            except Exception:
                pass
            time.sleep(interval)

        return False

    def is_connected(self) -> bool:
        """Check if connected to server."""
        try:
            resp = self._client.get("/health")
            self._connected = resp.status_code == 200
            return self._connected
        except Exception:
            self._connected = False
            return False

    def get_server_info(self) -> dict:
        """Get server information."""
        resp = self._client.get("/info")
        resp.raise_for_status()
        return resp.json()

    # === TTS Methods ===

    def load_tts_backend(self, backend: str, **kwargs) -> None:
        """
        Load TTS backend on server.

        Args:
            backend: Backend name (e.g., "qwen", "piper")
            **kwargs: Backend-specific options
        """
        resp = self._client.post(
            f"/tts/backends/{backend}/load",
            json=kwargs,
        )
        resp.raise_for_status()

    def synthesize(
        self,
        text: str,
        voice: str = "serena",
        language: str = "English",
        **kwargs,
    ) -> SynthesisResult:
        """
        Synthesize speech via server API.

        Args:
            text: Text to synthesize
            voice: Voice name
            language: Language

        Returns:
            SynthesisResult with audio
        """
        import scipy.io.wavfile as wavfile

        resp = self._client.post(
            "/tts/synthesize",
            json={
                "text": text,
                "voice": voice,
                "language": language,
                **kwargs,
            },
        )
        resp.raise_for_status()

        # Parse WAV response
        audio_bytes = resp.content
        sample_rate, audio = wavfile.read(io.BytesIO(audio_bytes))

        # Get metadata from headers
        duration = float(resp.headers.get("X-Duration", len(audio) / sample_rate))

        return SynthesisResult(
            audio=audio,
            sample_rate=sample_rate,
            duration=duration,
            voice=voice,
            text=text,
            metadata={"backend": "remote"},
        )

    def get_voices(self) -> list[dict]:
        """Get available voices from server."""
        resp = self._client.get("/tts/voices")
        resp.raise_for_status()
        return resp.json().get("voices", [])

    # === STT Methods ===

    def load_stt_backend(self, backend: str, **kwargs) -> None:
        """
        Load STT backend on server.

        Args:
            backend: Backend name (e.g., "whisper")
            **kwargs: Backend-specific options
        """
        resp = self._client.post(
            f"/stt/backends/{backend}/load",
            json=kwargs,
        )
        resp.raise_for_status()

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        **kwargs,
    ) -> TranscriptionResult:
        """
        Transcribe audio via server API.

        Args:
            audio: Audio samples (int16 or float32)
            sample_rate: Sample rate
            language: Language code (optional)

        Returns:
            TranscriptionResult with text
        """
        import scipy.io.wavfile as wavfile

        # Convert to WAV bytes
        if audio.dtype == np.float32:
            audio = (audio * 32767).astype(np.int16)

        wav_buffer = io.BytesIO()
        wavfile.write(wav_buffer, sample_rate, audio)
        wav_buffer.seek(0)

        # Send to server
        files = {"audio": ("audio.wav", wav_buffer, "audio/wav")}
        data = {}
        if language:
            data["language"] = language

        resp = self._client.post(
            "/stt/transcribe",
            files=files,
            data=data,
        )
        resp.raise_for_status()

        result = resp.json()

        # Parse segments
        segments = []
        for seg in result.get("segments", []):
            segments.append(TranscriptionSegment(
                text=seg["text"],
                start=seg["start"],
                end=seg["end"],
                confidence=seg.get("confidence", 1.0),
            ))

        return TranscriptionResult(
            text=result["text"],
            segments=segments,
            language=result.get("language", "en"),
            duration=result.get("duration", 0.0),
        )

    # === LLM/Chat Methods ===

    def chat(
        self,
        message: str,
        context: Optional[list[dict]] = None,
        use_rag: bool = False,
        rag_collection: Optional[str] = None,
        rag_top_k: int = 3,
        llm_model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """
        Send a chat message to the server LLM.

        Args:
            message: User message
            context: Conversation history (list of {"role": "...", "content": "..."})
            use_rag: Whether to use RAG for context
            rag_collection: RAG collection name
            rag_top_k: Number of RAG results to retrieve
            llm_model: LLM model to use
            system_prompt: Custom system prompt

        Returns:
            dict with response, model, tokens_used, latency_ms, etc.
        """
        resp = self._client.post(
            "/llm/chat",
            json={
                "message": message,
                "context": context or [],
                "use_rag": use_rag,
                "rag_collection": rag_collection,
                "rag_top_k": rag_top_k,
                "llm_model": llm_model,
                "system_prompt": system_prompt,
            },
        )
        resp.raise_for_status()
        return resp.json()

    def chat_stream(
        self,
        message: str,
        context: Optional[list[dict]] = None,
        use_rag: bool = False,
        rag_collection: Optional[str] = None,
        rag_top_k: int = 3,
        llm_model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Stream chat response from server LLM using SSE.

        Yields complete sentences as they're generated.

        Args:
            message: User message
            context: Conversation history
            use_rag: Whether to use RAG
            rag_collection: RAG collection name
            rag_top_k: Number of RAG results
            llm_model: LLM model to use
            system_prompt: Custom system prompt

        Yields:
            str: Complete sentences
        """
        import json
        import httpx

        # Use streaming client for SSE
        with httpx.Client(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        ) as client:
            with client.stream(
                "POST",
                "/llm/chat/stream",
                json={
                    "message": message,
                    "context": context or [],
                    "use_rag": use_rag,
                    "rag_collection": rag_collection,
                    "rag_top_k": rag_top_k,
                    "llm_model": llm_model,
                    "system_prompt": system_prompt,
                    "stream": True,
                },
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    try:
                        data = json.loads(line[6:])  # Remove "data: " prefix

                        if data.get("type") == "chunk":
                            content = data.get("content", "")
                            if content:
                                yield content

                        elif data.get("type") == "error":
                            raise Exception(data.get("error", "Unknown error"))

                        elif data.get("type") == "done":
                            break

                    except json.JSONDecodeError:
                        continue

    # === RAG Methods ===

    def rag_search(
        self,
        collection: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search a RAG collection.

        Args:
            collection: Collection name
            query: Search query
            top_k: Number of results

        Returns:
            List of search results
        """
        resp = self._client.post(
            f"/rag/{collection}/search",
            json={"query": query, "top_k": top_k},
        )
        resp.raise_for_status()
        result = resp.json()
        return result.get("results", [])

    def rag_info(self, collection: str) -> dict:
        """Get RAG collection info."""
        resp = self._client.get(f"/rag/{collection}/info")
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        """Close connection."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
