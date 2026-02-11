"""
Text chunking strategies for RAG.

Splits documents into smaller chunks for embedding and retrieval.
"""

import re
from dataclasses import dataclass
from typing import Iterator

from jetson_speech.rag.loaders import Document


@dataclass
class Chunk:
    """A chunk of text with metadata."""

    content: str
    metadata: dict

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Chunk(source={self.source}, content={preview!r})"


class TextChunker:
    """
    Split documents into chunks for embedding.

    Strategies:
    - sentence: Split by sentences (best for QA)
    - paragraph: Split by paragraphs
    - fixed: Fixed size chunks with overlap
    - semantic: Split by semantic boundaries (headers, sections)
    """

    def __init__(
        self,
        strategy: str = "sentence",
        chunk_size: int = 500,  # Max characters per chunk
        chunk_overlap: int = 50,  # Overlap for fixed strategy
        min_chunk_size: int = 50,  # Minimum chunk size
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk(self, document: Document) -> Iterator[Chunk]:
        """Split a document into chunks."""
        if self.strategy == "sentence":
            yield from self._chunk_by_sentence(document)
        elif self.strategy == "paragraph":
            yield from self._chunk_by_paragraph(document)
        elif self.strategy == "fixed":
            yield from self._chunk_fixed_size(document)
        elif self.strategy == "semantic":
            yield from self._chunk_semantic(document)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Chunk multiple documents."""
        chunks = []
        for doc in documents:
            chunks.extend(self.chunk(doc))
        return chunks

    def _chunk_by_sentence(self, document: Document) -> Iterator[Chunk]:
        """Split by sentences, combining small ones."""
        # Split by sentence-ending punctuation
        sentences = re.split(r"(?<=[.!?])\s+", document.content)

        current_chunk = ""
        chunk_idx = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If adding this sentence exceeds chunk_size, yield current chunk
            if current_chunk and len(current_chunk) + len(sentence) + 1 > self.chunk_size:
                if len(current_chunk) >= self.min_chunk_size:
                    yield Chunk(
                        content=current_chunk.strip(),
                        metadata={
                            **document.metadata,
                            "chunk_index": chunk_idx,
                            "chunk_strategy": "sentence",
                        }
                    )
                    chunk_idx += 1
                current_chunk = sentence
            else:
                current_chunk += (" " if current_chunk else "") + sentence

        # Yield remaining
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            yield Chunk(
                content=current_chunk.strip(),
                metadata={
                    **document.metadata,
                    "chunk_index": chunk_idx,
                    "chunk_strategy": "sentence",
                }
            )

    def _chunk_by_paragraph(self, document: Document) -> Iterator[Chunk]:
        """Split by paragraphs."""
        paragraphs = re.split(r"\n\s*\n", document.content)

        chunk_idx = 0
        for para in paragraphs:
            para = para.strip()
            if len(para) >= self.min_chunk_size:
                # If paragraph is too long, sub-chunk it
                if len(para) > self.chunk_size:
                    sub_doc = Document(content=para, metadata=document.metadata)
                    for sub_chunk in self._chunk_by_sentence(sub_doc):
                        sub_chunk.metadata["chunk_index"] = chunk_idx
                        sub_chunk.metadata["chunk_strategy"] = "paragraph"
                        yield sub_chunk
                        chunk_idx += 1
                else:
                    yield Chunk(
                        content=para,
                        metadata={
                            **document.metadata,
                            "chunk_index": chunk_idx,
                            "chunk_strategy": "paragraph",
                        }
                    )
                    chunk_idx += 1

    def _chunk_fixed_size(self, document: Document) -> Iterator[Chunk]:
        """Split into fixed-size chunks with overlap."""
        text = document.content
        chunk_idx = 0
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at word boundary
            if end < len(text):
                # Look for space near the end
                space_idx = text.rfind(" ", start + self.chunk_size - 50, end)
                if space_idx > start:
                    end = space_idx

            chunk_text = text[start:end].strip()

            if len(chunk_text) >= self.min_chunk_size:
                yield Chunk(
                    content=chunk_text,
                    metadata={
                        **document.metadata,
                        "chunk_index": chunk_idx,
                        "chunk_strategy": "fixed",
                        "start_char": start,
                        "end_char": end,
                    }
                )
                chunk_idx += 1

            start = end - self.chunk_overlap

    def _chunk_semantic(self, document: Document) -> Iterator[Chunk]:
        """Split by semantic boundaries (headers, sections)."""
        text = document.content

        # Look for markdown-style headers or section breaks
        sections = re.split(r"\n(?=#{1,3}\s|\n[A-Z][^a-z]*\n)", text)

        chunk_idx = 0
        for section in sections:
            section = section.strip()
            if len(section) >= self.min_chunk_size:
                # If section is too long, sub-chunk it
                if len(section) > self.chunk_size:
                    sub_doc = Document(content=section, metadata=document.metadata)
                    for sub_chunk in self._chunk_by_sentence(sub_doc):
                        sub_chunk.metadata["chunk_index"] = chunk_idx
                        sub_chunk.metadata["chunk_strategy"] = "semantic"
                        yield sub_chunk
                        chunk_idx += 1
                else:
                    yield Chunk(
                        content=section,
                        metadata={
                            **document.metadata,
                            "chunk_index": chunk_idx,
                            "chunk_strategy": "semantic",
                        }
                    )
                    chunk_idx += 1
