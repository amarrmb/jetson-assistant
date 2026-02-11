"""
Embedding models for RAG.

Converts text to dense vector representations for semantic search.
"""

import sys
from typing import Optional

import numpy as np


class EmbeddingModel:
    """
    Text embedding model.

    Supports:
    - sentence-transformers (local, default)
    - OpenAI embeddings (API)
    """

    # Popular embedding models
    MODELS = {
        "minilm": "sentence-transformers/all-MiniLM-L6-v2",  # 80MB, fast
        "mpnet": "sentence-transformers/all-mpnet-base-v2",  # 420MB, better
        "bge-small": "BAAI/bge-small-en-v1.5",  # 130MB, good
        "bge-base": "BAAI/bge-base-en-v1.5",  # 440MB, better
    }

    def __init__(
        self,
        model_name: str = "minilm",
        device: str = "cpu",
        normalize: bool = True,
    ):
        """
        Initialize embedding model.

        Args:
            model_name: Model name or path. Use keys from MODELS or full HF path.
            device: Device to run on ("cpu", "cuda")
            normalize: Whether to L2-normalize embeddings
        """
        self.model_name = self.MODELS.get(model_name, model_name)
        self.device = device
        self.normalize = normalize
        self._model = None
        self._dimension: Optional[int] = None

    def load(self) -> None:
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        print(f"Loading embedding model: {self.model_name}...", file=sys.stderr)
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._dimension = self._model.get_sentence_embedding_dimension()
        print(f"Embedding model loaded! (dim={self._dimension})", file=sys.stderr)

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if self._model is None:
            self.load()

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 10,
        )

        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        return self.embed([query])[0]

    def embed_documents(self, documents: list[str]) -> np.ndarray:
        """Embed multiple documents."""
        return self.embed(documents)


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI embedding model (API-based)."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.api_key = api_key
        self._client = None
        self._dimension = 1536 if "small" in model else 3072

    def load(self) -> None:
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai not installed. Install with: pip install openai"
            )

        import os
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required")

        self._client = OpenAI(api_key=api_key)
        print(f"OpenAI embedding model: {self.model}", file=sys.stderr)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using OpenAI API."""
        if self._client is None:
            self.load()

        response = self._client.embeddings.create(
            model=self.model,
            input=texts,
        )

        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype=np.float32)
