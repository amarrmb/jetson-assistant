"""
RAG (Retrieval Augmented Generation) module.

A modular, reusable RAG system for building domain-specific knowledge bases.

Example usage:
    from jetson_speech.rag import RAGPipeline, WebLoader, PDFLoader

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

from jetson_speech.rag.pipeline import RAGPipeline
from jetson_speech.rag.loaders import (
    BaseLoader,
    WebLoader,
    PDFLoader,
    TextLoader,
    DirectoryLoader,
)
from jetson_speech.rag.chunker import TextChunker
from jetson_speech.rag.embeddings import EmbeddingModel
from jetson_speech.rag.store import VectorStore

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
