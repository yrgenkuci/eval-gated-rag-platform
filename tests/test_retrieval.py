"""Tests for retrieval module."""

from unittest.mock import AsyncMock

import pytest

from src.embeddings.models import EmbeddingResult
from src.exceptions import RetrievalError
from src.retrieval.models import RetrievalResult
from src.retrieval.retriever import SemanticRetriever
from src.vectorstore.models import SearchResult


class TestRetrievalResult:
    """Tests for RetrievalResult model."""

    def test_create_result(self) -> None:
        """Result can be created."""
        result = RetrievalResult(
            content="Hello world",
            score=0.95,
            source="doc.txt",
        )
        assert result.content == "Hello world"
        assert result.score == 0.95
        assert result.source == "doc.txt"
        assert result.metadata == {}

    def test_result_with_metadata(self) -> None:
        """Result can have metadata."""
        result = RetrievalResult(
            content="Test",
            score=0.8,
            source="test.txt",
            metadata={"chunk_index": 0, "file_type": "text/plain"},
        )
        assert result.metadata["chunk_index"] == 0


class TestSemanticRetriever:
    """Tests for SemanticRetriever."""

    def _create_mock_embedding_service(self) -> AsyncMock:
        """Create mock embedding service."""
        service = AsyncMock()
        service.embed = AsyncMock(
            return_value=EmbeddingResult(
                text="test query",
                embedding=[0.1, 0.2, 0.3],
                model="test-model",
                dimensions=3,
            )
        )
        return service

    def _create_mock_vector_store(self) -> AsyncMock:
        """Create mock vector store."""
        store = AsyncMock()
        store.search = AsyncMock(return_value=[])
        return store

    @pytest.mark.asyncio
    async def test_retrieve_empty_query(self) -> None:
        """Empty query returns empty results."""
        embedding_service = self._create_mock_embedding_service()
        vector_store = self._create_mock_vector_store()

        retriever = SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
            collection="test",
        )

        results = await retriever.retrieve("")
        assert results == []
        embedding_service.embed.assert_not_called()

    @pytest.mark.asyncio
    async def test_retrieve_calls_embedding(self) -> None:
        """Retriever embeds the query."""
        embedding_service = self._create_mock_embedding_service()
        vector_store = self._create_mock_vector_store()

        retriever = SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
            collection="test",
        )

        await retriever.retrieve("test query")

        embedding_service.embed.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_retrieve_searches_vector_store(self) -> None:
        """Retriever searches vector store with embedded query."""
        embedding_service = self._create_mock_embedding_service()
        vector_store = self._create_mock_vector_store()

        retriever = SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
            collection="documents",
        )

        await retriever.retrieve("test query", top_k=10)

        vector_store.search.assert_called_once()
        call_kwargs = vector_store.search.call_args.kwargs
        assert call_kwargs["collection"] == "documents"
        assert call_kwargs["vector"] == [0.1, 0.2, 0.3]
        assert call_kwargs["limit"] == 10

    @pytest.mark.asyncio
    async def test_retrieve_returns_results(self) -> None:
        """Retriever returns formatted results."""
        embedding_service = self._create_mock_embedding_service()
        vector_store = self._create_mock_vector_store()

        # Mock search results
        vector_store.search = AsyncMock(
            return_value=[
                SearchResult(
                    id="1",
                    score=0.95,
                    payload={"content": "Hello world", "source": "doc1.txt"},
                ),
                SearchResult(
                    id="2",
                    score=0.85,
                    payload={"content": "Goodbye world", "source": "doc2.txt"},
                ),
            ]
        )

        retriever = SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
            collection="test",
        )

        results = await retriever.retrieve("test query")

        assert len(results) == 2
        assert results[0].content == "Hello world"
        assert results[0].score == 0.95
        assert results[0].source == "doc1.txt"
        assert results[1].content == "Goodbye world"

    @pytest.mark.asyncio
    async def test_retrieve_with_score_threshold(self) -> None:
        """Retriever filters by score threshold."""
        embedding_service = self._create_mock_embedding_service()
        vector_store = self._create_mock_vector_store()

        vector_store.search = AsyncMock(
            return_value=[
                SearchResult(id="1", score=0.95, payload={"content": "High"}),
                SearchResult(id="2", score=0.5, payload={"content": "Low"}),
            ]
        )

        retriever = SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
            collection="test",
            score_threshold=0.7,
        )

        results = await retriever.retrieve("test query")

        assert len(results) == 1
        assert results[0].content == "High"

    @pytest.mark.asyncio
    async def test_retrieve_with_filters(self) -> None:
        """Retriever passes filters to vector store."""
        embedding_service = self._create_mock_embedding_service()
        vector_store = self._create_mock_vector_store()

        retriever = SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
            collection="test",
        )

        await retriever.retrieve(
            "test query",
            filters={"source": "specific.txt"},
        )

        call_kwargs = vector_store.search.call_args.kwargs
        assert call_kwargs["filters"] == {"source": "specific.txt"}

    @pytest.mark.asyncio
    async def test_retrieve_handles_missing_content(self) -> None:
        """Retriever handles payloads without content."""
        embedding_service = self._create_mock_embedding_service()
        vector_store = self._create_mock_vector_store()

        vector_store.search = AsyncMock(
            return_value=[
                SearchResult(id="1", score=0.9, payload={}),
            ]
        )

        retriever = SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
            collection="test",
        )

        results = await retriever.retrieve("test query")

        assert len(results) == 1
        assert results[0].content == ""
        assert results[0].source == "1"  # Falls back to ID

    @pytest.mark.asyncio
    async def test_retrieve_error_handling(self) -> None:
        """Retriever wraps errors in RetrievalError."""
        embedding_service = self._create_mock_embedding_service()
        embedding_service.embed = AsyncMock(side_effect=Exception("Service down"))

        vector_store = self._create_mock_vector_store()

        retriever = SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
            collection="test",
        )

        with pytest.raises(RetrievalError):
            await retriever.retrieve("test query")

