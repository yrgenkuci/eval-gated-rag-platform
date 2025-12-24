"""Vector store data models."""

from typing import Any

from pydantic import BaseModel, Field


class VectorRecord(BaseModel):
    """A record to store in the vector database.

    Attributes:
        id: Unique identifier for the record.
        vector: The embedding vector.
        payload: Additional metadata to store with the vector.
    """

    id: str = Field(description="Unique record identifier")
    vector: list[float] = Field(description="Embedding vector")
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata payload",
    )


class SearchResult(BaseModel):
    """Result from a vector similarity search.

    Attributes:
        id: Record identifier.
        score: Similarity score (higher is more similar).
        payload: Stored metadata.
    """

    id: str = Field(description="Record identifier")
    score: float = Field(description="Similarity score")
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Record metadata",
    )

