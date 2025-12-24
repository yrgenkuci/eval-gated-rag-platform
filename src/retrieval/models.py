"""Retrieval data models."""

from typing import Any

from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    """Result from a retrieval operation.

    Attributes:
        content: The retrieved text content.
        score: Relevance score (higher is more relevant).
        source: Source document identifier.
        metadata: Additional metadata from the chunk.
    """

    content: str = Field(description="Retrieved text content")
    score: float = Field(description="Relevance score")
    source: str = Field(description="Source document identifier")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

