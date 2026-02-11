"""
Pydantic schemas for API requests and responses.
"""

from typing import Any

from pydantic import BaseModel, Field


# === General ===


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str
    is_jetson: bool
    power_mode: str | None = None
    tts_backend: str | None = None
    stt_backend: str | None = None


class BackendInfo(BaseModel):
    """Backend information."""

    name: str
    loaded: bool
    supports_streaming: bool = False
    supports_voice_cloning: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)


class BackendListResponse(BaseModel):
    """List of available backends."""

    backends: list[BackendInfo]


class LoadBackendRequest(BaseModel):
    """Request to load a backend."""

    model_size: str | None = None
    device: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class LoadBackendResponse(BaseModel):
    """Response after loading a backend."""

    success: bool
    message: str
    backend: BackendInfo | None = None


# === TTS ===


class VoiceInfo(BaseModel):
    """Voice information."""

    id: str
    name: str
    language: str
    gender: str = ""
    description: str = ""
    sample_rate: int = 24000


class VoiceListResponse(BaseModel):
    """List of available voices."""

    voices: list[VoiceInfo]
    languages: list[str]


class SynthesizeRequest(BaseModel):
    """TTS synthesis request."""

    text: str = Field(..., min_length=1, max_length=10000)
    voice: str | None = None
    language: str | None = None
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)
    output_format: str = Field(default="wav", pattern="^(wav|mp3|ogg)$")


class SynthesizeResponse(BaseModel):
    """TTS synthesis response (for JSON mode)."""

    success: bool
    duration: float
    sample_rate: int
    voice: str
    audio_base64: str | None = None
    error: str | None = None


class TTSStreamRequest(BaseModel):
    """WebSocket TTS stream request."""

    text: str
    voice: str | None = None
    language: str | None = None


class TTSStreamMessage(BaseModel):
    """WebSocket TTS stream message."""

    type: str  # "start", "audio", "done", "error"
    chunk: int | None = None
    total_chunks: int | None = None
    data: str | None = None  # Base64 audio data
    total_time: float | None = None
    error: str | None = None


# === STT ===


class TranscribeResponse(BaseModel):
    """STT transcription response."""

    success: bool
    text: str
    language: str
    duration: float
    segments: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


class STTStreamMessage(BaseModel):
    """WebSocket STT stream message."""

    type: str  # "start", "segment", "done", "error"
    text: str | None = None
    start: float | None = None
    end: float | None = None
    confidence: float | None = None
    is_final: bool = False
    error: str | None = None


# === Benchmark ===


class BenchmarkRequest(BaseModel):
    """Benchmark request."""

    type: str = Field(..., pattern="^(tts|stt)$")
    backends: list[str] = Field(default_factory=list)
    iterations: int = Field(default=3, ge=1, le=20)
    text: str | None = None  # For TTS
    audio_path: str | None = None  # For STT


class BenchmarkResult(BaseModel):
    """Single benchmark result."""

    backend: str
    iterations: int
    avg_time: float
    min_time: float
    max_time: float
    avg_rtf: float  # Real-time factor
    memory_mb: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class BenchmarkResponse(BaseModel):
    """Benchmark response."""

    success: bool
    type: str
    results: list[BenchmarkResult]
    error: str | None = None


# === LLM / Chat ===


class ChatMessage(BaseModel):
    """A single chat message."""

    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    """Chat/LLM request."""

    message: str = Field(..., min_length=1, max_length=10000)
    context: list[ChatMessage] = Field(default_factory=list)
    use_rag: bool = False
    rag_collection: str | None = None
    rag_top_k: int = Field(default=3, ge=1, le=10)
    llm_model: str | None = None
    system_prompt: str | None = None
    stream: bool = False  # If true, use SSE streaming


class ChatResponse(BaseModel):
    """Chat/LLM response."""

    success: bool
    response: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    rag_context: str | None = None  # The context retrieved from RAG
    rag_sources: list[str] = Field(default_factory=list)
    error: str | None = None


class ChatStreamChunk(BaseModel):
    """Streaming chat chunk (for SSE)."""

    type: str  # "start", "chunk", "done", "error"
    content: str | None = None
    model: str | None = None
    done: bool = False
    error: str | None = None


# === RAG ===


class RAGIngestRequest(BaseModel):
    """Request to ingest documents into RAG."""

    collection: str = Field(..., min_length=1, max_length=100)
    source_type: str = Field(..., pattern="^(url|file|text|directory)$")
    content: str  # URL, file path, or raw text
    follow_links: bool = False  # For URL source
    max_pages: int = Field(default=10, ge=1, le=100)  # For URL crawling
    glob_pattern: str = "**/*"  # For directory source


class RAGIngestResponse(BaseModel):
    """Response after ingesting documents."""

    success: bool
    collection: str
    documents_added: int
    chunks_added: int
    error: str | None = None


class RAGSearchRequest(BaseModel):
    """Request to search RAG collection."""

    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)


class RAGSearchResult(BaseModel):
    """A single RAG search result."""

    content: str
    score: float
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGSearchResponse(BaseModel):
    """Response from RAG search."""

    success: bool
    results: list[RAGSearchResult]
    query: str
    error: str | None = None


class RAGInfoResponse(BaseModel):
    """RAG collection information."""

    success: bool
    collection: str
    chunk_count: int
    sources: list[str]
    embedding_model: str
    error: str | None = None


class RAGCollectionsResponse(BaseModel):
    """List of RAG collections."""

    success: bool
    collections: list[str]
    error: str | None = None
