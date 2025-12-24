"""Document data models."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata associated with a document.

    Attributes:
        source: Original source path or identifier.
        created_at: When the document was loaded.
        file_type: File extension or MIME type.
        extra: Additional metadata fields.
    """

    source: str = Field(description="Original source path or identifier")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the document was loaded",
    )
    file_type: str = Field(default="text/plain", description="File type or MIME type")
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata fields",
    )


class Document(BaseModel):
    """A document with content and metadata.

    Attributes:
        content: The text content of the document.
        metadata: Associated metadata.
    """

    content: str = Field(description="Text content of the document")
    metadata: DocumentMetadata = Field(description="Document metadata")

    @classmethod
    def from_text(
        cls,
        content: str,
        source: str,
        file_type: str = "text/plain",
        **extra: Any,
    ) -> "Document":
        """Create a document from text content.

        Args:
            content: The text content.
            source: Source identifier.
            file_type: File type.
            **extra: Additional metadata.

        Returns:
            New Document instance.
        """
        metadata = DocumentMetadata(
            source=source,
            file_type=file_type,
            extra=extra,
        )
        return cls(content=content, metadata=metadata)

    @classmethod
    def from_file(cls, path: Path) -> "Document":
        """Create a document from a file path.

        Args:
            path: Path to the file.

        Returns:
            New Document instance.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = path.read_text(encoding="utf-8")
        file_type = _get_file_type(path)

        return cls.from_text(
            content=content,
            source=str(path),
            file_type=file_type,
            file_name=path.name,
            file_size=path.stat().st_size,
        )


def _get_file_type(path: Path) -> str:
    """Determine file type from extension."""
    extension_map = {
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".json": "application/json",
        ".html": "text/html",
        ".xml": "application/xml",
        ".csv": "text/csv",
        ".py": "text/x-python",
        ".js": "text/javascript",
        ".ts": "text/typescript",
    }
    return extension_map.get(path.suffix.lower(), "text/plain")

