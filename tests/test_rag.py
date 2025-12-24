"""Tests for RAG pipeline module."""

from unittest.mock import AsyncMock

import pytest

from src.llm.models import GenerationResult
from src.rag.models import RAGQuery, RAGResponse, SourceAttribution
from src.rag.pipeline import RAGPipeline
from src.retrieval.models import RetrievalResult


class TestSourceAttribution:
    """Tests for SourceAttribution model."""

    def test_create_attribution(self) -> None:
        """Attribution can be created."""
        attr = SourceAttribution(
            source="doc.txt",
            content="Sample content",
            score=0.95,
        )
        assert attr.source == "doc.txt"
        assert attr.score == 0.95


class TestRAGQuery:
    """Tests for RAGQuery model."""

    def test_default_values(self) -> None:
        """Query has sensible defaults."""
        query = RAGQuery(question="What is AI?")
        assert query.top_k == 5
        assert query.score_threshold == 0.0

    def test_custom_values(self) -> None:
        """Query accepts custom values."""
        query = RAGQuery(
            question="What is AI?",
            top_k=10,
            score_threshold=0.7,
        )
        assert query.top_k == 10
        assert query.score_threshold == 0.7


class TestRAGResponse:
    """Tests for RAGResponse model."""

    def test_create_response(self) -> None:
        """Response can be created."""
        response = RAGResponse(
            answer="AI is artificial intelligence.",
            sources=[
                SourceAttribution(source="ai.txt", content="...", score=0.9),
            ],
            model="llama3:8b",
            tokens_used=100,
        )
        assert response.answer == "AI is artificial intelligence."
        assert len(response.sources) == 1
        assert response.tokens_used == 100


class TestRAGPipeline:
    """Tests for RAGPipeline."""

    def _create_mock_retriever(self) -> AsyncMock:
        """Create mock retriever."""
        retriever = AsyncMock()
        retriever.retrieve = AsyncMock(return_value=[])
        return retriever

    def _create_mock_llm(self) -> AsyncMock:
        """Create mock LLM client."""
        llm = AsyncMock()
        llm.model_name = "test-model"
        llm.generate_text = AsyncMock(
            return_value=GenerationResult(
                content="Generated answer",
                model="test-model",
                prompt_tokens=50,
                completion_tokens=20,
                total_tokens=70,
            )
        )
        return llm

    @pytest.mark.asyncio
    async def test_query_calls_retriever(self) -> None:
        """Pipeline calls retriever with query."""
        retriever = self._create_mock_retriever()
        llm = self._create_mock_llm()

        pipeline = RAGPipeline(retriever=retriever, llm_client=llm)
        await pipeline.query(RAGQuery(question="Test question", top_k=3))

        retriever.retrieve.assert_called_once()
        call_kwargs = retriever.retrieve.call_args.kwargs
        assert call_kwargs["query"] == "Test question"
        assert call_kwargs["top_k"] == 3

    @pytest.mark.asyncio
    async def test_query_no_results(self) -> None:
        """Pipeline handles no retrieval results."""
        retriever = self._create_mock_retriever()
        retriever.retrieve = AsyncMock(return_value=[])
        llm = self._create_mock_llm()

        pipeline = RAGPipeline(retriever=retriever, llm_client=llm)
        response = await pipeline.query(RAGQuery(question="Unknown topic"))

        assert "could not find" in response.answer.lower()
        assert response.sources == []
        llm.generate_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_with_results(self) -> None:
        """Pipeline generates answer from results."""
        retriever = self._create_mock_retriever()
        retriever.retrieve = AsyncMock(
            return_value=[
                RetrievalResult(
                    content="AI is artificial intelligence.",
                    score=0.95,
                    source="ai.txt",
                    metadata={},
                ),
            ]
        )
        llm = self._create_mock_llm()

        pipeline = RAGPipeline(retriever=retriever, llm_client=llm)
        response = await pipeline.query(RAGQuery(question="What is AI?"))

        assert response.answer == "Generated answer"
        assert response.model == "test-model"
        assert response.tokens_used == 70
        llm.generate_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_source_attribution(self) -> None:
        """Pipeline includes source attributions."""
        retriever = self._create_mock_retriever()
        retriever.retrieve = AsyncMock(
            return_value=[
                RetrievalResult(
                    content="First document content",
                    score=0.95,
                    source="doc1.txt",
                    metadata={},
                ),
                RetrievalResult(
                    content="Second document content",
                    score=0.85,
                    source="doc2.txt",
                    metadata={},
                ),
            ]
        )
        llm = self._create_mock_llm()

        pipeline = RAGPipeline(retriever=retriever, llm_client=llm)
        response = await pipeline.query(RAGQuery(question="Test"))

        assert len(response.sources) == 2
        assert response.sources[0].source == "doc1.txt"
        assert response.sources[0].score == 0.95
        assert response.sources[1].source == "doc2.txt"

    @pytest.mark.asyncio
    async def test_query_score_threshold(self) -> None:
        """Pipeline filters by score threshold."""
        retriever = self._create_mock_retriever()
        retriever.retrieve = AsyncMock(
            return_value=[
                RetrievalResult(content="High", score=0.9, source="high.txt", metadata={}),
                RetrievalResult(content="Low", score=0.3, source="low.txt", metadata={}),
            ]
        )
        llm = self._create_mock_llm()

        pipeline = RAGPipeline(retriever=retriever, llm_client=llm)
        response = await pipeline.query(
            RAGQuery(question="Test", score_threshold=0.5)
        )

        # Only high score source should be included
        assert len(response.sources) == 1
        assert response.sources[0].source == "high.txt"

    @pytest.mark.asyncio
    async def test_query_truncates_long_content(self) -> None:
        """Pipeline truncates long content in attributions."""
        retriever = self._create_mock_retriever()
        long_content = "x" * 500  # More than 200 chars
        retriever.retrieve = AsyncMock(
            return_value=[
                RetrievalResult(
                    content=long_content,
                    score=0.9,
                    source="long.txt",
                    metadata={},
                ),
            ]
        )
        llm = self._create_mock_llm()

        pipeline = RAGPipeline(retriever=retriever, llm_client=llm)
        response = await pipeline.query(RAGQuery(question="Test"))

        assert len(response.sources[0].content) < 250
        assert response.sources[0].content.endswith("...")

    @pytest.mark.asyncio
    async def test_query_simple(self) -> None:
        """Simple query returns just the answer."""
        retriever = self._create_mock_retriever()
        retriever.retrieve = AsyncMock(
            return_value=[
                RetrievalResult(content="Test", score=0.9, source="t.txt", metadata={}),
            ]
        )
        llm = self._create_mock_llm()

        pipeline = RAGPipeline(retriever=retriever, llm_client=llm)
        answer = await pipeline.query_simple("What is this?")

        assert answer == "Generated answer"

