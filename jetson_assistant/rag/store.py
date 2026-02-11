"""
Vector store for RAG.

Stores embeddings and enables similarity search.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

import numpy as np

from jetson_assistant.rag.chunker import Chunk
from jetson_assistant.rag.embeddings import EmbeddingModel
from jetson_assistant.config import get_default_cache_dir


class VectorStore:
    """
    Vector store using ChromaDB.

    Supports:
    - Persistent storage
    - Similarity search
    - Metadata filtering
    """

    def __init__(
        self,
        collection_name: str,
        persist_dir: Optional[str | Path] = None,
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        """
        Initialize vector store.

        Args:
            collection_name: Name of the collection
            persist_dir: Directory for persistent storage (default: cache dir)
            embedding_model: Embedding model to use (default: MiniLM)
        """
        self.collection_name = collection_name
        self.persist_dir = Path(persist_dir) if persist_dir else get_default_cache_dir() / "rag"
        self.embedding_model = embedding_model or EmbeddingModel()

        self._client = None
        self._collection = None

    def _init_client(self) -> None:
        """Initialize ChromaDB client."""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "chromadb not installed. Install with: pip install chromadb"
            )

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # Load embedding model
        self.embedding_model.load()

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        logger.info("Vector store ready: %s (%d documents)", self.collection_name, self.count())

    @property
    def collection(self):
        """Get ChromaDB collection, initializing if needed."""
        if self._collection is None:
            self._init_client()
        return self._collection

    def add(self, chunks: list[Chunk]) -> int:
        """
        Add chunks to the vector store.

        Args:
            chunks: List of chunks to add

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.embed_documents(texts)

        # Prepare data for ChromaDB - use hash of content for unique IDs
        import hashlib
        ids = []
        for i, chunk in enumerate(chunks):
            # Create unique ID from source + chunk index + content hash
            content_hash = hashlib.md5(chunk.content.encode()).hexdigest()[:8]
            chunk_id = f"{chunk.source}_{i}_{content_hash}"
            ids.append(chunk_id)

        metadatas = [chunk.metadata for chunk in chunks]

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
        )

        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> list[dict]:
        """
        Search for similar documents.

        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filter (ChromaDB where clause)

        Returns:
            List of results with content, metadata, and distance
        """
        # Embed query
        query_embedding = self.embedding_model.embed_query(query)

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "score": 1 - results["distances"][0][i],  # Convert distance to similarity
            })

        return formatted

    def count(self) -> int:
        """Get number of documents in the collection."""
        return self.collection.count()

    def clear(self) -> None:
        """Clear all documents from the collection."""
        if self._client is not None:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    def delete(self, ids: list[str]) -> None:
        """Delete specific documents by ID."""
        self.collection.delete(ids=ids)

    def get_all_sources(self) -> list[str]:
        """Get list of all unique sources."""
        results = self.collection.get(include=["metadatas"])
        sources = set()
        for metadata in results["metadatas"]:
            if "source" in metadata:
                sources.add(metadata["source"])
        return sorted(sources)
