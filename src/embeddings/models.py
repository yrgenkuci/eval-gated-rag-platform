"""Embedding data models."""

from pydantic import BaseModel, Field


class EmbeddingResult(BaseModel):
    """Result of an embedding operation.

    Attributes:
        text: The original text that was embedded.
        embedding: The embedding vector.
        model: The model used to generate the embedding.
        dimensions: Number of dimensions in the embedding.
    """

    text: str = Field(description="Original text")
    embedding: list[float] = Field(description="Embedding vector")
    model: str = Field(description="Model used for embedding")
    dimensions: int = Field(description="Vector dimensions")

    def model_post_init(self, __context: object) -> None:
        """Validate dimensions match embedding length."""
        if self.dimensions != len(self.embedding):
            raise ValueError(
                f"dimensions ({self.dimensions}) does not match "
                f"embedding length ({len(self.embedding)})"
            )

