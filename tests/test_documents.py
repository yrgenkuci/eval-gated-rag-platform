"""Tests for document models, loaders, and chunkers."""

from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import pytest

from src.documents.chunker import (
    CharacterChunker,
    ChunkerConfig,
    SentenceChunker,
)
from src.documents.loader import TextFileLoader
from src.documents.models import Document, DocumentMetadata
from src.exceptions import DocumentError, ErrorCode


class TestDocumentMetadata:
    """Tests for DocumentMetadata model."""

    def test_default_values(self) -> None:
        """Metadata has sensible defaults."""
        meta = DocumentMetadata(source="test.txt")
        assert meta.source == "test.txt"
        assert meta.file_type == "text/plain"
        assert meta.extra == {}
        assert meta.created_at is not None

    def test_extra_metadata(self) -> None:
        """Extra metadata is stored."""
        meta = DocumentMetadata(
            source="test.txt",
            extra={"author": "test", "version": 1},
        )
        assert meta.extra["author"] == "test"
        assert meta.extra["version"] == 1


class TestDocument:
    """Tests for Document model."""

    def test_from_text(self) -> None:
        """Document can be created from text."""
        doc = Document.from_text(
            content="Hello, world!",
            source="greeting.txt",
        )
        assert doc.content == "Hello, world!"
        assert doc.metadata.source == "greeting.txt"

    def test_from_text_with_extra(self) -> None:
        """Document preserves extra metadata."""
        doc = Document.from_text(
            content="Test content",
            source="test.txt",
            author="tester",
            version=2,
        )
        assert doc.metadata.extra["author"] == "tester"
        assert doc.metadata.extra["version"] == 2

    def test_from_file(self) -> None:
        """Document can be loaded from file."""
        with NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("File content here")
            f.flush()
            path = Path(f.name)

        try:
            doc = Document.from_file(path)
            assert doc.content == "File content here"
            assert doc.metadata.source == str(path)
            assert doc.metadata.file_type == "text/plain"
            assert doc.metadata.extra["file_name"] == path.name
        finally:
            path.unlink()

    def test_from_file_not_found(self) -> None:
        """Document.from_file raises for missing file."""
        with pytest.raises(FileNotFoundError):
            Document.from_file(Path("/nonexistent/file.txt"))

    def test_from_file_markdown(self) -> None:
        """Markdown files get correct type."""
        with NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Heading")
            f.flush()
            path = Path(f.name)

        try:
            doc = Document.from_file(path)
            assert doc.metadata.file_type == "text/markdown"
        finally:
            path.unlink()


class TestTextFileLoader:
    """Tests for TextFileLoader."""

    def test_load_text_file(self) -> None:
        """Loader reads text file content."""
        with NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test document content")
            f.flush()
            path = Path(f.name)

        try:
            loader = TextFileLoader()
            doc = loader.load(path)
            assert doc.content == "Test document content"
        finally:
            path.unlink()

    def test_load_with_string_path(self) -> None:
        """Loader accepts string paths."""
        with NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("String path test")
            f.flush()
            path_str = f.name

        try:
            loader = TextFileLoader()
            doc = loader.load(path_str)
            assert doc.content == "String path test"
        finally:
            Path(path_str).unlink()

    def test_load_missing_file(self) -> None:
        """Loader raises DocumentError for missing files."""
        loader = TextFileLoader()
        with pytest.raises(DocumentError) as exc_info:
            loader.load("/nonexistent/file.txt")

        assert exc_info.value.code == ErrorCode.DOCUMENT_NOT_FOUND

    def test_load_directory(self) -> None:
        """Loader raises DocumentError for directories."""
        with TemporaryDirectory() as tmpdir:
            loader = TextFileLoader()
            with pytest.raises(DocumentError) as exc_info:
                loader.load(tmpdir)

            assert exc_info.value.code == ErrorCode.DOCUMENT_PARSE_ERROR

    def test_supports_txt(self) -> None:
        """Loader supports .txt files."""
        loader = TextFileLoader()
        assert loader.supports("document.txt") is True
        assert loader.supports(Path("document.txt")) is True

    def test_supports_markdown(self) -> None:
        """Loader supports markdown files."""
        loader = TextFileLoader()
        assert loader.supports("readme.md") is True
        assert loader.supports("readme.markdown") is True

    def test_not_supports_other(self) -> None:
        """Loader does not support other file types."""
        loader = TextFileLoader()
        assert loader.supports("image.png") is False
        assert loader.supports("data.json") is False
        assert loader.supports("script.py") is False

    def test_custom_encoding(self) -> None:
        """Loader respects encoding parameter."""
        loader = TextFileLoader(encoding="latin-1")
        assert loader.encoding == "latin-1"


class TestChunkerConfig:
    """Tests for ChunkerConfig."""

    def test_default_values(self) -> None:
        """Config has sensible defaults."""
        config = ChunkerConfig()
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.min_chunk_size == 100

    def test_custom_values(self) -> None:
        """Config accepts custom values."""
        config = ChunkerConfig(chunk_size=500, chunk_overlap=50, min_chunk_size=50)
        assert config.chunk_size == 500
        assert config.chunk_overlap == 50

    def test_overlap_must_be_less_than_size(self) -> None:
        """Overlap cannot exceed chunk size."""
        with pytest.raises(ValueError):
            ChunkerConfig(chunk_size=100, chunk_overlap=100)

    def test_minimum_chunk_size(self) -> None:
        """Chunk size has minimum value."""
        with pytest.raises(ValueError):
            ChunkerConfig(chunk_size=50)  # Below minimum of 100


class TestCharacterChunker:
    """Tests for CharacterChunker."""

    def test_empty_document(self) -> None:
        """Empty document returns empty list."""
        doc = Document.from_text("", source="empty.txt")
        chunker = CharacterChunker()
        chunks = chunker.chunk(doc)
        assert chunks == []

    def test_whitespace_only(self) -> None:
        """Whitespace-only document returns empty list."""
        doc = Document.from_text("   \n\t  ", source="whitespace.txt")
        chunker = CharacterChunker()
        chunks = chunker.chunk(doc)
        assert chunks == []

    def test_small_document(self) -> None:
        """Document smaller than chunk size returns single chunk."""
        doc = Document.from_text("Short text", source="short.txt")
        config = ChunkerConfig(chunk_size=1000, chunk_overlap=100)
        chunker = CharacterChunker(config)
        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].content == "Short text"
        assert chunks[0].index == 0

    def test_multiple_chunks(self) -> None:
        """Long document is split into multiple chunks."""
        # Create document longer than chunk size
        text = "Word " * 300  # ~1500 characters
        doc = Document.from_text(text, source="long.txt")
        config = ChunkerConfig(chunk_size=500, chunk_overlap=50, min_chunk_size=50)
        chunker = CharacterChunker(config)
        chunks = chunker.chunk(doc)

        assert len(chunks) > 1
        # Check indices are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_chunk_metadata_preserved(self) -> None:
        """Chunk metadata includes source info."""
        doc = Document.from_text("Test content", source="test.txt")
        chunker = CharacterChunker()
        chunks = chunker.chunk(doc)

        assert chunks[0].metadata.source == "test.txt"
        assert "chunk_index" in chunks[0].metadata.extra

    def test_chunk_positions(self) -> None:
        """Chunks have correct start/end positions."""
        doc = Document.from_text("Hello world", source="test.txt")
        chunker = CharacterChunker()
        chunks = chunker.chunk(doc)

        assert chunks[0].start_char == 0
        assert chunks[0].end_char > 0


class TestSentenceChunker:
    """Tests for SentenceChunker."""

    def test_empty_document(self) -> None:
        """Empty document returns empty list."""
        doc = Document.from_text("", source="empty.txt")
        chunker = SentenceChunker()
        chunks = chunker.chunk(doc)
        assert chunks == []

    def test_single_sentence(self) -> None:
        """Single sentence returns single chunk."""
        doc = Document.from_text("This is a test.", source="single.txt")
        chunker = SentenceChunker()
        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert "This is a test." in chunks[0].content

    def test_multiple_sentences(self) -> None:
        """Multiple sentences are grouped into chunks."""
        text = "First sentence. Second sentence. Third sentence."
        doc = Document.from_text(text, source="multi.txt")
        config = ChunkerConfig(chunk_size=1000, chunk_overlap=0)
        chunker = SentenceChunker(config)
        chunks = chunker.chunk(doc)

        # All sentences fit in one chunk
        assert len(chunks) == 1

    def test_sentence_splitting(self) -> None:
        """Long text is split at sentence boundaries."""
        # Create text with many sentences
        sentences = [f"This is sentence number {i}." for i in range(20)]
        text = " ".join(sentences)
        doc = Document.from_text(text, source="sentences.txt")
        config = ChunkerConfig(chunk_size=200, chunk_overlap=50, min_chunk_size=50)
        chunker = SentenceChunker(config)
        chunks = chunker.chunk(doc)

        assert len(chunks) > 1
        # Each chunk should end with a sentence
        for chunk in chunks:
            assert chunk.content.endswith(".")

