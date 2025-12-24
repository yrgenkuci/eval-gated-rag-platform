"""RAG pipeline data models."""

from pydantic import BaseModel, Field


class SourceAttribution(BaseModel):
    """Attribution to a source document.

    Attributes:
        source: Source document identifier.
        content: Relevant content snippet.
        score: Relevance score.
    """

    source: str = Field(description="Source document identifier")
    content: str = Field(description="Relevant content snippet")
    score: float = Field(description="Relevance score")


class RAGQuery(BaseModel):
    """Input for RAG query.

    Attributes:
        question: The user's question.
        top_k: Number of documents to retrieve.
        score_threshold: Minimum relevance score.
    """

    question: str = Field(description="User question")
    top_k: int = Field(default=5, ge=1, le=20, description="Documents to retrieve")
    score_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score",
    )


class RAGResponse(BaseModel):
    """Response from RAG query.

    Attributes:
        answer: Generated answer.
        sources: Source attributions.
        model: LLM model used.
        tokens_used: Total tokens consumed.
    """

    answer: str = Field(description="Generated answer")
    sources: list[SourceAttribution] = Field(
        default_factory=list,
        description="Source attributions",
    )
    model: str = Field(description="LLM model used")
    tokens_used: int = Field(default=0, description="Total tokens consumed")

