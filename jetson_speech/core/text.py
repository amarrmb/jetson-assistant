"""
Text extraction and processing utilities.

Supports extracting text from various file formats (txt, pdf, docx, md).
"""

import re
from pathlib import Path


def extract_text(filepath: str | Path) -> str:
    """
    Extract text from various file formats.

    Args:
        filepath: Path to the file

    Returns:
        Extracted text content

    Raises:
        FileNotFoundError: If file doesn't exist
        ImportError: If required library not installed
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = path.suffix.lower()

    if ext == ".txt":
        return path.read_text(encoding="utf-8")

    elif ext == ".pdf":
        return _extract_pdf(path)

    elif ext == ".docx":
        return _extract_docx(path)

    elif ext == ".md":
        return _extract_markdown(path)

    else:
        # Try reading as plain text
        return path.read_text(encoding="utf-8")


def _extract_pdf(path: Path) -> str:
    """Extract text from PDF file."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError(
            "pypdf not installed. Install with: pip install pypdf"
        )

    reader = PdfReader(str(path))
    text_parts = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)

    return "\n".join(text_parts)


def _extract_docx(path: Path) -> str:
    """Extract text from Word document."""
    try:
        import docx2txt
    except ImportError:
        raise ImportError(
            "docx2txt not installed. Install with: pip install docx2txt"
        )

    return docx2txt.process(str(path))


def _extract_markdown(path: Path) -> str:
    """Extract text from Markdown file, removing formatting."""
    text = path.read_text(encoding="utf-8")

    # Remove headers
    text = re.sub(r"#{1,6}\s*", "", text)

    # Remove bold
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)

    # Remove italic
    text = re.sub(r"\*(.+?)\*", r"\1", text)

    # Remove inline code
    text = re.sub(r"`(.+?)`", r"\1", text)

    # Remove links, keep text
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)

    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)

    # Remove images
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

    return text


def split_into_chunks(
    text: str,
    max_chars: int = 500,
    by_sentence: bool = False,
) -> list[str]:
    """
    Split text into speakable chunks.

    Args:
        text: Text to split
        max_chars: Maximum characters per chunk (when by_sentence=False)
        by_sentence: If True, split by sentences (one per chunk)

    Returns:
        List of text chunks
    """
    # Normalize: add space after sentence-ending punctuation if missing
    text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)

    # Split by paragraphs first
    paragraphs = re.split(r"\n\s*\n", text.strip())

    chunks = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Split by sentences
        sentences = re.split(r"(?<=[.!?])\s+", para)

        if by_sentence:
            # Each sentence is a chunk (for streaming)
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    chunks.append(sentence)
        else:
            # Combine sentences up to max_chars
            current_chunk = ""

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                if len(current_chunk) + len(sentence) + 1 <= max_chars:
                    current_chunk += (" " if current_chunk else "") + sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence

            if current_chunk:
                chunks.append(current_chunk)

    return chunks


def clean_text_for_speech(text: str) -> str:
    """
    Clean text for better TTS output.

    Args:
        text: Raw text

    Returns:
        Cleaned text suitable for speech synthesis
    """
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", "", text)

    # Remove special characters except basic punctuation
    text = re.sub(r"[^\w\s.,!?;:'\"-]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def estimate_speech_duration(text: str, words_per_minute: int = 150) -> float:
    """
    Estimate speech duration for text.

    Args:
        text: Text to estimate
        words_per_minute: Speaking rate

    Returns:
        Estimated duration in seconds
    """
    words = len(text.split())
    return (words / words_per_minute) * 60
