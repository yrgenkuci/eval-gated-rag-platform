"""Tests for RAG API routes."""

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.app import app
from src.api.routes import (
    IngestRequest,
    QueryRequest,
    query_request_to_rag_query,
    rag_response_to_query_response,
)
from src.rag.models import RAGResponse, SourceAttribution


class TestQueryRequest:
    """Tests for QueryRequest model."""

    def test_defaults(self) -> None:
        """Request has sensible defaults."""
        req = QueryRequest(question="Test?")
        assert req.top_k == 5
        assert req.score_threshold == 0.0


class TestIngestRequest:
    """Tests for IngestRequest model."""

    def test_create_request(self) -> None:
        """Request can be created."""
        req = IngestRequest(content="Hello", source="test.txt")
        assert req.content == "Hello"
        assert req.metadata == {}


class TestConverters:
    """Tests for response converters."""

    def test_rag_response_to_query_response(self) -> None:
        """Converts RAGResponse to QueryResponse."""
        rag = RAGResponse(
            answer="Answer",
            sources=[SourceAttribution(source="s.txt", content="...", score=0.9)],
            model="test",
            tokens_used=100,
        )
        result = rag_response_to_query_response(rag)

        assert result.answer == "Answer"
        assert len(result.sources) == 1
        assert result.sources[0]["source"] == "s.txt"

    def test_query_request_to_rag_query(self) -> None:
        """Converts QueryRequest to RAGQuery."""
        req = QueryRequest(question="What?", top_k=3, score_threshold=0.5)
        result = query_request_to_rag_query(req)

        assert result.question == "What?"
        assert result.top_k == 3
        assert result.score_threshold == 0.5


class TestQueryEndpoint:
    """Tests for /api/v1/query endpoint."""

    @pytest.mark.asyncio
    async def test_query_returns_503_without_pipeline(self) -> None:
        """Query returns 503 when pipeline not configured."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/query",
                json={"question": "What is AI?"},
            )

        assert response.status_code == 503
        data = response.json()
        assert "not configured" in data["detail"]["error"]

    @pytest.mark.asyncio
    async def test_query_validates_request(self) -> None:
        """Query validates request parameters."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Missing question
            response = await client.post("/api/v1/query", json={})

        assert response.status_code == 422  # Validation error


class TestIngestEndpoint:
    """Tests for /api/v1/ingest endpoint."""

    @pytest.mark.asyncio
    async def test_ingest_returns_503_without_pipeline(self) -> None:
        """Ingest returns 503 when pipeline not configured."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/ingest",
                json={"content": "Hello world", "source": "test.txt"},
            )

        assert response.status_code == 503
        data = response.json()
        assert "not configured" in data["detail"]["error"]

    @pytest.mark.asyncio
    async def test_ingest_validates_request(self) -> None:
        """Ingest validates request parameters."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Missing required fields
            response = await client.post("/api/v1/ingest", json={})

        assert response.status_code == 422  # Validation error

