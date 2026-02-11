"""
RAG (Retrieval Augmented Generation) module.

A modular, reusable RAG system for building domain-specific knowledge bases.

Example usage:
    from jetson_assistant.rag import RAGPipeline, WebLoader, PDFLoader

    # Create pipeline
    rag = RAGPipeline(collection_name="dota2")

    # Ingest data from multiple sources
    rag.ingest(WebLoader("https://dota2.fandom.com/wiki/Heroes"))
    rag.ingest(PDFLoader("dota_guide.pdf"))
    rag.ingest(TextLoader("notes.txt"))

    # Query
    context = rag.retrieve("Who counters Anti-Mage?", top_k=3)
    answer = rag.query("Who counters Anti-Mage?", llm=my_llm)
"""

from jetson_assistant.rag.pipeline import RAGPipeline
from jetson_assistant.rag.loaders import (
    BaseLoader,
    WebLoader,
    PDFLoader,
    TextLoader,
    DirectoryLoader,
)
from jetson_assistant.rag.chunker import TextChunker
from jetson_assistant.rag.embeddings import EmbeddingModel
from jetson_assistant.rag.store import VectorStore

__all__ = [
    "RAGPipeline",
    "BaseLoader",
    "WebLoader",
    "PDFLoader",
    "TextLoader",
    "DirectoryLoader",
    "TextChunker",
    "EmbeddingModel",
    "VectorStore",
]
