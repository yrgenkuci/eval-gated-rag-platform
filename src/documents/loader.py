"""Document loader interface and implementations."""

from abc import ABC, abstractmethod
from pathlib import Path

from src.documents.models import Document
from src.exceptions import DocumentError, ErrorCode


class DocumentLoader(ABC):
    """Abstract base class for document loaders.

    Defines the interface for loading documents from various sources.
    """

    @abstractmethod
    def load(self, source: str | Path) -> Document:
        """Load a document from a source.

        Args:
            source: Path or identifier for the document source.

        Returns:
            Loaded Document instance.

        Raises:
            DocumentError: If loading fails.
        """
        ...

    @abstractmethod
    def supports(self, source: str | Path) -> bool:
        """Check if this loader supports the given source.

        Args:
            source: Path or identifier to check.

        Returns:
            True if this loader can handle the source.
        """
        ...


class TextFileLoader(DocumentLoader):
    """Loader for plain text files.

    Supports .txt, .md, and other text-based files.
    """

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".text"}

    def __init__(self, encoding: str = "utf-8") -> None:
        """Initialize the text file loader.

        Args:
            encoding: Text encoding to use when reading files.
        """
        self.encoding = encoding

    def load(self, source: str | Path) -> Document:
        """Load a text file as a document.

        Args:
            source: Path to the text file.

        Returns:
            Document with file content.

        Raises:
            DocumentError: If file cannot be read.
        """
        path = Path(source) if isinstance(source, str) else source

        if not path.exists():
            raise DocumentError(
                f"File not found: {path}",
                code=ErrorCode.DOCUMENT_NOT_FOUND,
                details={"path": str(path)},
            )

        if not path.is_file():
            raise DocumentError(
                f"Not a file: {path}",
                code=ErrorCode.DOCUMENT_PARSE_ERROR,
                details={"path": str(path)},
            )

        try:
            content = path.read_text(encoding=self.encoding)
        except UnicodeDecodeError as e:
            raise DocumentError(
                f"Failed to decode file: {path}",
                code=ErrorCode.DOCUMENT_PARSE_ERROR,
                details={"path": str(path), "encoding": self.encoding, "error": str(e)},
            ) from e
        except OSError as e:
            raise DocumentError(
                f"Failed to read file: {path}",
                code=ErrorCode.DOCUMENT_PARSE_ERROR,
                details={"path": str(path), "error": str(e)},
            ) from e

        return Document.from_text(
            content=content,
            source=str(path),
            file_name=path.name,
            file_size=path.stat().st_size,
        )

    def supports(self, source: str | Path) -> bool:
        """Check if source is a supported text file.

        Args:
            source: Path to check.

        Returns:
            True if file has a supported extension.
        """
        path = Path(source) if isinstance(source, str) else source
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

