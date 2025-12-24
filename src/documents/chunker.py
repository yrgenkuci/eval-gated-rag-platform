"""Text chunking strategies for document processing."""

import re
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from src.documents.models import Document, DocumentMetadata


class Chunk(BaseModel):
    """A chunk of text from a document.

    Attributes:
        content: The text content of the chunk.
        metadata: Metadata including source and chunk info.
        index: Position of this chunk in the sequence.
        start_char: Starting character position in original document.
        end_char: Ending character position in original document.
    """

    content: str = Field(description="Text content of the chunk")
    metadata: DocumentMetadata = Field(description="Chunk metadata")
    index: int = Field(description="Chunk index in sequence")
    start_char: int = Field(description="Start position in original document")
    end_char: int = Field(description="End position in original document")


class ChunkerConfig(BaseModel):
    """Configuration for text chunking.

    Attributes:
        chunk_size: Target size of each chunk in characters.
        chunk_overlap: Number of characters to overlap between chunks.
        min_chunk_size: Minimum chunk size (smaller chunks are merged).
    """

    chunk_size: int = Field(default=1000, ge=100, description="Target chunk size")
    chunk_overlap: int = Field(default=200, ge=0, description="Overlap between chunks")
    min_chunk_size: int = Field(default=100, ge=10, description="Minimum chunk size")

    def model_post_init(self, __context: Any) -> None:
        """Validate overlap is less than chunk size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")


class Chunker(ABC):
    """Abstract base class for text chunkers."""

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        """Initialize chunker with configuration.

        Args:
            config: Chunking configuration. Uses defaults if not provided.
        """
        self.config = config or ChunkerConfig()

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks.

        Args:
            document: The document to chunk.

        Returns:
            List of Chunk objects.

        Raises:
            DocumentError: If chunking fails.
        """
        ...

    def _create_chunk(
        self,
        content: str,
        source_metadata: DocumentMetadata,
        index: int,
        start_char: int,
        end_char: int,
    ) -> Chunk:
        """Create a chunk with metadata.

        Args:
            content: Chunk text content.
            source_metadata: Original document metadata.
            index: Chunk index.
            start_char: Start position.
            end_char: End position.

        Returns:
            New Chunk instance.
        """
        chunk_metadata = DocumentMetadata(
            source=source_metadata.source,
            file_type=source_metadata.file_type,
            extra={
                **source_metadata.extra,
                "chunk_index": index,
                "start_char": start_char,
                "end_char": end_char,
            },
        )
        return Chunk(
            content=content,
            metadata=chunk_metadata,
            index=index,
            start_char=start_char,
            end_char=end_char,
        )


class CharacterChunker(Chunker):
    """Chunk text by character count with overlap.

    Simple chunking strategy that splits text at character boundaries.
    Tries to break at whitespace when possible.
    """

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document by character count.

        Args:
            document: Document to chunk.

        Returns:
            List of chunks.
        """
        text = document.content
        if not text.strip():
            return []

        chunks: list[Chunk] = []
        start = 0
        index = 0

        while start < len(text):
            # Calculate end position
            end = min(start + self.config.chunk_size, len(text))

            # Try to break at whitespace if not at end of text
            if end < len(text):
                # Look for last whitespace in the chunk
                last_space = text.rfind(" ", start, end)
                if last_space > start + self.config.min_chunk_size:
                    end = last_space + 1  # Include the space

            chunk_text = text[start:end].strip()

            if chunk_text:  # Only add non-empty chunks
                chunks.append(
                    self._create_chunk(
                        content=chunk_text,
                        source_metadata=document.metadata,
                        index=index,
                        start_char=start,
                        end_char=end,
                    )
                )
                index += 1

            # Move start position with overlap
            start = end - self.config.chunk_overlap
            if start >= len(text) or start <= chunks[-1].start_char if chunks else False:
                break

        return chunks


class SentenceChunker(Chunker):
    """Chunk text by sentences, grouping to target size.

    Splits text into sentences first, then groups sentences
    to approximate target chunk size.
    """

    # Sentence boundary pattern
    SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document by sentences.

        Args:
            document: Document to chunk.

        Returns:
            List of chunks.
        """
        text = document.content
        if not text.strip():
            return []

        # Split into sentences
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_length = 0
        start_char = 0
        index = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence exceeds chunk size, save current chunk
            if (
                current_length + sentence_length > self.config.chunk_size
                and current_sentences
            ):
                chunk_text = " ".join(current_sentences)
                end_char = start_char + len(chunk_text)

                chunks.append(
                    self._create_chunk(
                        content=chunk_text,
                        source_metadata=document.metadata,
                        index=index,
                        start_char=start_char,
                        end_char=end_char,
                    )
                )
                index += 1

                # Calculate overlap sentences
                overlap_sentences = self._get_overlap_sentences(
                    current_sentences, self.config.chunk_overlap
                )
                start_char = end_char - sum(len(s) + 1 for s in overlap_sentences)
                current_sentences = overlap_sentences
                current_length = sum(len(s) + 1 for s in overlap_sentences)

            current_sentences.append(sentence)
            current_length += sentence_length + 1  # +1 for space

        # Add remaining sentences as final chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(
                self._create_chunk(
                    content=chunk_text,
                    source_metadata=document.metadata,
                    index=index,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                )
            )

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences.

        Args:
            text: Text to split.

        Returns:
            List of sentences.
        """
        # Use regex to split at sentence boundaries
        sentences = self.SENTENCE_PATTERN.split(text)
        # Clean up and filter empty sentences
        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap_sentences(
        self, sentences: list[str], target_overlap: int
    ) -> list[str]:
        """Get sentences for overlap.

        Args:
            sentences: List of sentences.
            target_overlap: Target overlap in characters.

        Returns:
            Sentences to include in overlap.
        """
        overlap: list[str] = []
        overlap_length = 0

        for sentence in reversed(sentences):
            if overlap_length >= target_overlap:
                break
            overlap.insert(0, sentence)
            overlap_length += len(sentence) + 1

        return overlap

