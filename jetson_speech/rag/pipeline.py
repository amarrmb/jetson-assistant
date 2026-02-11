"""
RAG Pipeline - Main interface for building and querying RAG systems.

Example:
    # Create a Dota 2 knowledge base
    rag = RAGPipeline("dota2")

    # Ingest data
    rag.ingest_url("https://dota2.fandom.com/wiki/Heroes")
    rag.ingest_file("dota_guide.pdf")

    # Query
    answer = rag.query("Who counters Anti-Mage?", llm=my_llm)
"""

import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

from jetson_speech.rag.loaders import (
    BaseLoader,
    Document,
    WebLoader,
    PDFLoader,
    TextLoader,
    DirectoryLoader,
)
from jetson_speech.rag.chunker import TextChunker, Chunk
from jetson_speech.rag.embeddings import EmbeddingModel
from jetson_speech.rag.store import VectorStore


class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Handles data ingestion, chunking, embedding, storage, and retrieval.
    """

    # Default prompt template for RAG queries
    DEFAULT_PROMPT_TEMPLATE = """Use the following context to answer the question. If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {question}

Answer:"""

    def __init__(
        self,
        collection_name: str,
        persist_dir: Optional[str | Path] = None,
        embedding_model: str = "minilm",
        chunk_strategy: str = "sentence",
        chunk_size: int = 500,
    ):
        """
        Initialize RAG pipeline.

        Args:
            collection_name: Name for the vector store collection
            persist_dir: Directory for persistent storage
            embedding_model: Embedding model name (minilm, mpnet, bge-small, bge-base)
            chunk_strategy: Chunking strategy (sentence, paragraph, fixed, semantic)
            chunk_size: Maximum chunk size in characters
        """
        self.collection_name = collection_name
        self.embedding_model = EmbeddingModel(model_name=embedding_model)
        self.chunker = TextChunker(strategy=chunk_strategy, chunk_size=chunk_size)
        self.store = VectorStore(
            collection_name=collection_name,
            persist_dir=persist_dir,
            embedding_model=self.embedding_model,
        )
        self.prompt_template = self.DEFAULT_PROMPT_TEMPLATE

    def ingest(self, loader: BaseLoader, verbose: bool = True) -> int:
        """
        Ingest documents from a loader.

        Args:
            loader: Document loader instance
            verbose: Print progress

        Returns:
            Number of chunks added
        """
        # Load documents
        if verbose:
            logger.info("Loading documents...")

        documents = loader.load_all()

        if verbose:
            logger.info("Loaded %d document(s)", len(documents))

        # Chunk documents
        if verbose:
            logger.info("Chunking documents...")

        chunks = self.chunker.chunk_documents(documents)

        if verbose:
            logger.info("Created %d chunk(s)", len(chunks))

        # Add to vector store
        if verbose:
            logger.info("Adding to vector store...")

        count = self.store.add(chunks)

        if verbose:
            logger.info("Added %d chunk(s) to '%s'", count, self.collection_name)

        return count

    def ingest_url(
        self,
        url: str,
        follow_links: bool = False,
        max_pages: int = 50,
        verbose: bool = True,
    ) -> int:
        """
        Ingest content from a URL.

        Args:
            url: URL to crawl
            follow_links: Whether to follow links on the page
            max_pages: Maximum pages to crawl
            verbose: Print progress

        Returns:
            Number of chunks added
        """
        loader = WebLoader(
            url=url,
            follow_links=follow_links,
            max_pages=max_pages,
        )
        return self.ingest(loader, verbose=verbose)

    def ingest_file(self, path: str | Path, verbose: bool = True) -> int:
        """
        Ingest content from a file.

        Args:
            path: Path to file (PDF, TXT, MD, etc.)
            verbose: Print progress

        Returns:
            Number of chunks added
        """
        path = Path(path)

        if path.suffix.lower() == ".pdf":
            loader = PDFLoader(path)
        else:
            loader = TextLoader(path)

        return self.ingest(loader, verbose=verbose)

    def ingest_directory(
        self,
        path: str | Path,
        glob_pattern: str = "**/*",
        verbose: bool = True,
    ) -> int:
        """
        Ingest all documents from a directory.

        Args:
            path: Directory path
            glob_pattern: File pattern to match
            verbose: Print progress

        Returns:
            Number of chunks added
        """
        loader = DirectoryLoader(path=path, glob_pattern=glob_pattern)
        return self.ingest(loader, verbose=verbose)

    def ingest_text(self, text: str, source: str = "inline", verbose: bool = True) -> int:
        """
        Ingest raw text directly.

        Args:
            text: Text content to ingest
            source: Source identifier
            verbose: Print progress

        Returns:
            Number of chunks added
        """
        document = Document(content=text, metadata={"source": source, "type": "text"})
        chunks = list(self.chunker.chunk(document))

        if verbose:
            logger.info("Created %d chunk(s) from text", len(chunks))

        return self.store.add(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> list[dict]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of results
            filter_metadata: Optional metadata filter

        Returns:
            List of relevant documents with scores
        """
        return self.store.search(query, top_k=top_k, filter_metadata=filter_metadata)

    def build_context(self, query: str, top_k: int = 5) -> str:
        """
        Build context string from retrieved documents.

        Args:
            query: Search query
            top_k: Number of documents to include

        Returns:
            Formatted context string
        """
        results = self.retrieve(query, top_k=top_k)

        if not results:
            return "No relevant information found."

        context_parts = []
        for i, result in enumerate(results, 1):
            source = result["metadata"].get("source", "unknown")
            content = result["content"]
            score = result["score"]
            context_parts.append(f"[{i}] (score: {score:.2f}, source: {source})\n{content}")

        return "\n\n".join(context_parts)

    def query(
        self,
        question: str,
        llm=None,
        top_k: int = 5,
        return_context: bool = False,
    ) -> Union[str, tuple[str, str]]:
        """
        Query the RAG system.

        Args:
            question: Question to answer
            llm: LLM instance with generate() method
            top_k: Number of context documents to retrieve
            return_context: Also return the context used

        Returns:
            Answer string, or (answer, context) if return_context=True
        """
        # Build context
        context = self.build_context(question, top_k=top_k)

        # Build prompt
        prompt = self.prompt_template.format(
            context=context,
            question=question,
        )

        # Generate answer
        if llm is not None:
            if hasattr(llm, "generate_stream"):
                answer = " ".join(llm.generate_stream(prompt))
            elif hasattr(llm, "generate"):
                result = llm.generate(prompt)
                answer = result.text
            else:
                raise ValueError("LLM must have generate() or generate_stream() method")
        else:
            # No LLM - just return the context
            answer = f"Context found:\n{context}"

        if return_context:
            return answer, context
        return answer

    def count(self) -> int:
        """Get number of chunks in the store."""
        return self.store.count()

    def clear(self) -> None:
        """Clear all data from the store."""
        self.store.clear()

    def get_sources(self) -> list[str]:
        """Get list of all ingested sources."""
        return self.store.get_all_sources()

    def info(self) -> dict:
        """Get information about the RAG pipeline."""
        return {
            "collection_name": self.collection_name,
            "chunk_count": self.count(),
            "sources": self.get_sources(),
            "embedding_model": self.embedding_model.model_name,
            "chunk_strategy": self.chunker.strategy,
            "chunk_size": self.chunker.chunk_size,
        }
