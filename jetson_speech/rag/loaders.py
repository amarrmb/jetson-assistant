"""
Document loaders for various data sources.

Supports: Web pages, PDFs, text files, directories.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import urljoin, urlparse


@dataclass
class Document:
    """A document with content and metadata."""

    content: str
    metadata: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Document(source={self.source}, content={preview!r})"


class BaseLoader(ABC):
    """Base class for document loaders."""

    @abstractmethod
    def load(self) -> Iterator[Document]:
        """Load documents from the source."""
        pass

    def load_all(self) -> list[Document]:
        """Load all documents into a list."""
        return list(self.load())


class TextLoader(BaseLoader):
    """Load text from a file."""

    def __init__(self, path: str | Path, encoding: str = "utf-8"):
        self.path = Path(path)
        self.encoding = encoding

    def load(self) -> Iterator[Document]:
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")

        content = self.path.read_text(encoding=self.encoding)
        yield Document(
            content=content,
            metadata={
                "source": str(self.path),
                "type": "text",
                "filename": self.path.name,
            }
        )


class PDFLoader(BaseLoader):
    """Load text from a PDF file."""

    def __init__(self, path: str | Path, pages: Optional[list[int]] = None):
        self.path = Path(path)
        self.pages = pages  # None = all pages

    def load(self) -> Iterator[Document]:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf not installed. Install with: pip install pypdf")

        if not self.path.exists():
            raise FileNotFoundError(f"PDF not found: {self.path}")

        reader = PdfReader(self.path)

        pages_to_load = self.pages or range(len(reader.pages))

        for page_num in pages_to_load:
            if page_num >= len(reader.pages):
                continue

            page = reader.pages[page_num]
            text = page.extract_text() or ""

            if text.strip():
                yield Document(
                    content=text,
                    metadata={
                        "source": str(self.path),
                        "type": "pdf",
                        "filename": self.path.name,
                        "page": page_num + 1,
                        "total_pages": len(reader.pages),
                    }
                )


class WebLoader(BaseLoader):
    """Load content from web pages."""

    def __init__(
        self,
        url: str,
        follow_links: bool = False,
        max_depth: int = 1,
        same_domain_only: bool = True,
        max_pages: int = 50,
    ):
        self.url = url
        self.follow_links = follow_links
        self.max_depth = max_depth
        self.same_domain_only = same_domain_only
        self.max_pages = max_pages
        self._visited: set[str] = set()

    def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch HTML content from URL."""
        import urllib.request
        import urllib.error

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"
            }
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.read().decode("utf-8", errors="ignore")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"Failed to fetch {url}: {e}")
            return None

    def _extract_text(self, html: str) -> str:
        """Extract clean text from HTML."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            # Fallback: basic HTML tag removal
            text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text)
            return text.strip()

        soup = BeautifulSoup(html, "html.parser")

        # Remove unwanted elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Get text
        text = soup.get_text(separator=" ", strip=True)

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _extract_links(self, html: str, base_url: str) -> list[str]:
        """Extract links from HTML."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            links = []
            for a in soup.find_all("a", href=True):
                href = a["href"]
                full_url = urljoin(base_url, href)
                # Remove fragments
                full_url = full_url.split("#")[0]
                if full_url.startswith("http"):
                    links.append(full_url)
            return links
        except ImportError:
            return []

    def _crawl(self, url: str, depth: int = 0) -> Iterator[Document]:
        """Recursively crawl pages."""
        if url in self._visited:
            return
        if len(self._visited) >= self.max_pages:
            return

        self._visited.add(url)

        html = self._fetch_page(url)
        if not html:
            return

        text = self._extract_text(html)
        if text and len(text) > 100:  # Skip very short pages
            yield Document(
                content=text,
                metadata={
                    "source": url,
                    "type": "web",
                    "depth": depth,
                }
            )

        # Follow links if enabled
        if self.follow_links and depth < self.max_depth:
            base_domain = urlparse(self.url).netloc

            for link in self._extract_links(html, url):
                link_domain = urlparse(link).netloc

                if self.same_domain_only and link_domain != base_domain:
                    continue

                yield from self._crawl(link, depth + 1)

    def load(self) -> Iterator[Document]:
        self._visited.clear()
        yield from self._crawl(self.url)


class DirectoryLoader(BaseLoader):
    """Load all documents from a directory."""

    def __init__(
        self,
        path: str | Path,
        glob_pattern: str = "**/*",
        recursive: bool = True,
    ):
        self.path = Path(path)
        self.glob_pattern = glob_pattern
        self.recursive = recursive

    def _get_loader(self, file_path: Path) -> Optional[BaseLoader]:
        """Get appropriate loader for file type."""
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return PDFLoader(file_path)
        elif suffix in [".txt", ".md", ".rst", ".json", ".yaml", ".yml"]:
            return TextLoader(file_path)
        else:
            return None

    def load(self) -> Iterator[Document]:
        if not self.path.exists():
            raise FileNotFoundError(f"Directory not found: {self.path}")

        pattern = self.glob_pattern
        files = self.path.glob(pattern) if self.recursive else self.path.glob(pattern.replace("**", "*"))

        for file_path in files:
            if file_path.is_file():
                loader = self._get_loader(file_path)
                if loader:
                    try:
                        yield from loader.load()
                    except Exception as e:
                        print(f"Failed to load {file_path}: {e}")


class APILoader(BaseLoader):
    """Load data from an API endpoint."""

    def __init__(
        self,
        url: str,
        headers: Optional[dict] = None,
        params: Optional[dict] = None,
        json_path: Optional[str] = None,  # JSONPath to extract text
    ):
        self.url = url
        self.headers = headers or {}
        self.params = params or {}
        self.json_path = json_path

    def load(self) -> Iterator[Document]:
        import json
        import urllib.request
        import urllib.parse

        # Build URL with params
        if self.params:
            url = f"{self.url}?{urllib.parse.urlencode(self.params)}"
        else:
            url = self.url

        req = urllib.request.Request(url, headers=self.headers)

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as e:
            print(f"API request failed: {e}")
            return

        # Extract text from JSON
        if self.json_path:
            # Simple JSONPath support (e.g., "data.items[*].text")
            content = self._extract_json_path(data, self.json_path)
        else:
            content = json.dumps(data, indent=2)

        yield Document(
            content=content,
            metadata={
                "source": self.url,
                "type": "api",
            }
        )

    def _extract_json_path(self, data: dict, path: str) -> str:
        """Simple JSON path extraction."""
        parts = path.split(".")
        current = data

        for part in parts:
            if "[*]" in part:
                key = part.replace("[*]", "")
                if key:
                    current = current.get(key, [])
                if isinstance(current, list):
                    texts = []
                    for item in current:
                        if isinstance(item, str):
                            texts.append(item)
                        elif isinstance(item, dict):
                            texts.append(str(item))
                    return "\n".join(texts)
            else:
                current = current.get(part, {})

        if isinstance(current, str):
            return current
        return str(current)
