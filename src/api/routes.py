"""API routes for RAG operations."""

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.logging_config import get_logger
from src.rag.models import RAGQuery, RAGResponse

logger = get_logger(__name__)


# Create router
router = APIRouter(prefix="/api/v1", tags=["RAG"])


class QueryRequest(BaseModel):
    """Request body for RAG query."""

    question: str = Field(description="Question to answer")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents")
    score_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score",
    )


class QueryResponse(BaseModel):
    """Response from RAG query."""

    answer: str = Field(description="Generated answer")
    sources: list[dict[str, Any]] = Field(description="Source attributions")
    model: str = Field(description="Model used")
    tokens_used: int = Field(description="Tokens consumed")


class IngestRequest(BaseModel):
    """Request body for document ingestion."""

    content: str = Field(description="Document content to ingest")
    source: str = Field(description="Source identifier")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class IngestResponse(BaseModel):
    """Response from document ingestion."""

    success: bool = Field(description="Whether ingestion succeeded")
    chunks_created: int = Field(description="Number of chunks created")
    source: str = Field(description="Source identifier")


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(_request: QueryRequest) -> QueryResponse:
    """Query the RAG system.

    Note: This endpoint requires the RAG pipeline to be configured.
    In a full implementation, the pipeline would be injected via dependency.
    """
    # For now, return a placeholder indicating pipeline not configured
    # In production, this would use dependency injection
    logger.warning("RAG pipeline not configured - returning placeholder")

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "error": "RAG pipeline not configured",
            "message": "The RAG pipeline requires embedding service, vector store, and LLM to be running",
        },
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(_request: IngestRequest) -> IngestResponse:
    """Ingest a document into the RAG system.

    Note: This endpoint requires the RAG pipeline to be configured.
    In a full implementation, services would be injected via dependency.
    """
    logger.warning("RAG pipeline not configured - returning placeholder")

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "error": "RAG pipeline not configured",
            "message": "The ingestion pipeline requires embedding service and vector store to be running",
        },
    )


# Helper function to convert RAGResponse to QueryResponse
def rag_response_to_query_response(rag_response: RAGResponse) -> QueryResponse:
    """Convert internal RAGResponse to API QueryResponse."""
    return QueryResponse(
        answer=rag_response.answer,
        sources=[
            {
                "source": s.source,
                "content": s.content,
                "score": s.score,
            }
            for s in rag_response.sources
        ],
        model=rag_response.model,
        tokens_used=rag_response.tokens_used,
    )


# Helper to convert QueryRequest to RAGQuery
def query_request_to_rag_query(request: QueryRequest) -> RAGQuery:
    """Convert API QueryRequest to internal RAGQuery."""
    return RAGQuery(
        question=request.question,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
    )

