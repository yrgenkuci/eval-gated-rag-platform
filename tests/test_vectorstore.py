"""Tests for vector store module."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import QdrantSettings
from src.exceptions import ErrorCode, VectorStoreError
from src.vectorstore.models import SearchResult, VectorRecord
from src.vectorstore.service import QdrantVectorStore


class TestVectorRecord:
    """Tests for VectorRecord model."""

    def test_create_record(self) -> None:
        """Record can be created with required fields."""
        record = VectorRecord(
            id="test-id",
            vector=[0.1, 0.2, 0.3],
        )
        assert record.id == "test-id"
        assert record.vector == [0.1, 0.2, 0.3]
        assert record.payload == {}

    def test_record_with_payload(self) -> None:
        """Record can have payload metadata."""
        record = VectorRecord(
            id="test-id",
            vector=[0.1, 0.2],
            payload={"text": "hello", "source": "doc.txt"},
        )
        assert record.payload["text"] == "hello"
        assert record.payload["source"] == "doc.txt"


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_create_result(self) -> None:
        """Result can be created."""
        result = SearchResult(
            id="test-id",
            score=0.95,
            payload={"text": "hello"},
        )
        assert result.id == "test-id"
        assert result.score == 0.95
        assert result.payload["text"] == "hello"


class TestQdrantVectorStore:
    """Tests for QdrantVectorStore."""

    def _create_mock_client(self) -> AsyncMock:
        """Create a mock Qdrant client."""
        client = AsyncMock()
        client.collection_exists = AsyncMock(return_value=False)
        client.create_collection = AsyncMock()
        client.delete_collection = AsyncMock()
        client.upsert = AsyncMock()
        # Mock query_points with proper response structure
        mock_response = MagicMock()
        mock_response.points = []
        client.query_points = AsyncMock(return_value=mock_response)
        client.delete = AsyncMock()
        client.close = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_create_collection(self) -> None:
        """Collection can be created."""
        mock_client = self._create_mock_client()
        settings = QdrantSettings(url="http://localhost:6333")
        store = QdrantVectorStore(settings=settings, client=mock_client)

        await store.create_collection("test", dimensions=1024)

        mock_client.create_collection.assert_called_once()
        call_kwargs = mock_client.create_collection.call_args.kwargs
        assert call_kwargs["collection_name"] == "test"

    @pytest.mark.asyncio
    async def test_create_collection_already_exists(self) -> None:
        """Creating existing collection raises error."""
        mock_client = self._create_mock_client()
        mock_client.collection_exists = AsyncMock(return_value=True)

        settings = QdrantSettings(url="http://localhost:6333")
        store = QdrantVectorStore(settings=settings, client=mock_client)

        with pytest.raises(VectorStoreError) as exc_info:
            await store.create_collection("test", dimensions=1024)

        assert exc_info.value.code == ErrorCode.COLLECTION_EXISTS

    @pytest.mark.asyncio
    async def test_delete_collection(self) -> None:
        """Collection can be deleted."""
        mock_client = self._create_mock_client()
        mock_client.collection_exists = AsyncMock(return_value=True)

        settings = QdrantSettings(url="http://localhost:6333")
        store = QdrantVectorStore(settings=settings, client=mock_client)

        await store.delete_collection("test")

        mock_client.delete_collection.assert_called_once_with("test")

    @pytest.mark.asyncio
    async def test_delete_collection_not_found(self) -> None:
        """Deleting non-existent collection raises error."""
        mock_client = self._create_mock_client()
        mock_client.collection_exists = AsyncMock(return_value=False)

        settings = QdrantSettings(url="http://localhost:6333")
        store = QdrantVectorStore(settings=settings, client=mock_client)

        with pytest.raises(VectorStoreError) as exc_info:
            await store.delete_collection("test")

        assert exc_info.value.code == ErrorCode.COLLECTION_NOT_FOUND

    @pytest.mark.asyncio
    async def test_collection_exists(self) -> None:
        """Can check if collection exists."""
        mock_client = self._create_mock_client()
        mock_client.collection_exists = AsyncMock(return_value=True)

        settings = QdrantSettings(url="http://localhost:6333")
        store = QdrantVectorStore(settings=settings, client=mock_client)

        result = await store.collection_exists("test")
        assert result is True

    @pytest.mark.asyncio
    async def test_upsert_records(self) -> None:
        """Records can be upserted."""
        mock_client = self._create_mock_client()
        settings = QdrantSettings(url="http://localhost:6333")
        store = QdrantVectorStore(settings=settings, client=mock_client)

        records = [
            VectorRecord(id="1", vector=[0.1, 0.2], payload={"text": "hello"}),
            VectorRecord(id="2", vector=[0.3, 0.4], payload={"text": "world"}),
        ]

        count = await store.upsert("test", records)

        assert count == 2
        mock_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_empty_list(self) -> None:
        """Upserting empty list returns 0."""
        mock_client = self._create_mock_client()
        settings = QdrantSettings(url="http://localhost:6333")
        store = QdrantVectorStore(settings=settings, client=mock_client)

        count = await store.upsert("test", [])

        assert count == 0
        mock_client.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_search(self) -> None:
        """Search returns results."""
        mock_client = self._create_mock_client()

        # Mock search results
        mock_point = MagicMock()
        mock_point.id = "1"
        mock_point.score = 0.95
        mock_point.payload = {"text": "hello"}
        mock_response = MagicMock()
        mock_response.points = [mock_point]
        mock_client.query_points = AsyncMock(return_value=mock_response)

        settings = QdrantSettings(url="http://localhost:6333")
        store = QdrantVectorStore(settings=settings, client=mock_client)

        results = await store.search("test", vector=[0.1, 0.2], limit=5)

        assert len(results) == 1
        assert results[0].id == "1"
        assert results[0].score == 0.95
        assert results[0].payload["text"] == "hello"

    @pytest.mark.asyncio
    async def test_search_with_filters(self) -> None:
        """Search can filter results."""
        mock_client = self._create_mock_client()

        settings = QdrantSettings(url="http://localhost:6333")
        store = QdrantVectorStore(settings=settings, client=mock_client)

        await store.search(
            "test",
            vector=[0.1, 0.2],
            filters={"source": "doc.txt"},
        )

        # Verify filter was passed
        call_kwargs = mock_client.query_points.call_args.kwargs
        assert call_kwargs["query_filter"] is not None

    @pytest.mark.asyncio
    async def test_delete_records(self) -> None:
        """Records can be deleted by ID."""
        mock_client = self._create_mock_client()
        settings = QdrantSettings(url="http://localhost:6333")
        store = QdrantVectorStore(settings=settings, client=mock_client)

        count = await store.delete("test", ids=["1", "2"])

        assert count == 2
        mock_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_empty_list(self) -> None:
        """Deleting empty list returns 0."""
        mock_client = self._create_mock_client()
        settings = QdrantSettings(url="http://localhost:6333")
        store = QdrantVectorStore(settings=settings, client=mock_client)

        count = await store.delete("test", ids=[])

        assert count == 0
        mock_client.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        """Store closes client properly."""
        mock_client = self._create_mock_client()
        settings = QdrantSettings(url="http://localhost:6333")
        store = QdrantVectorStore(settings=settings, client=mock_client)
        store._owns_client = True

        await store.close()

        mock_client.close.assert_called_once()

