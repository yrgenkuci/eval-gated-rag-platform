"""Tests for embedding service."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.config import EmbeddingSettings
from src.embeddings.models import EmbeddingResult
from src.embeddings.service import HTTPEmbeddingService
from src.exceptions import EmbeddingError


class TestEmbeddingResult:
    """Tests for EmbeddingResult model."""

    def test_valid_result(self) -> None:
        """Valid embedding result is created."""
        result = EmbeddingResult(
            text="test",
            embedding=[0.1, 0.2, 0.3],
            model="test-model",
            dimensions=3,
        )
        assert result.text == "test"
        assert len(result.embedding) == 3
        assert result.dimensions == 3

    def test_dimensions_mismatch(self) -> None:
        """Mismatched dimensions raise error."""
        with pytest.raises(ValueError, match="dimensions"):
            EmbeddingResult(
                text="test",
                embedding=[0.1, 0.2, 0.3],
                model="test-model",
                dimensions=5,  # Wrong!
            )


class TestHTTPEmbeddingService:
    """Tests for HTTPEmbeddingService."""

    def test_model_name(self) -> None:
        """Service returns configured model name."""
        settings = EmbeddingSettings(model="test-model")
        service = HTTPEmbeddingService(settings=settings)
        assert service.model_name == "test-model"

    def test_known_model_dimensions(self) -> None:
        """Known models return correct dimensions."""
        settings = EmbeddingSettings(model="BAAI/bge-large-en-v1.5")
        service = HTTPEmbeddingService(settings=settings)
        assert service.dimensions == 1024

    def test_unknown_model_dimensions(self) -> None:
        """Unknown models default to 1024."""
        settings = EmbeddingSettings(model="unknown-model")
        service = HTTPEmbeddingService(settings=settings)
        assert service.dimensions == 1024

    @pytest.mark.asyncio
    async def test_embed_single(self) -> None:
        """Single text embedding works."""
        settings = EmbeddingSettings(
            base_url="http://test:8080",
            model="test-model",
        )

        # Mock HTTP client
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response

        service = HTTPEmbeddingService(settings=settings, client=mock_client)
        result = await service.embed("test text")

        assert result.text == "test text"
        assert result.embedding == [0.1, 0.2, 0.3]
        assert result.model == "test-model"

    @pytest.mark.asyncio
    async def test_embed_batch(self) -> None:
        """Batch embedding works."""
        settings = EmbeddingSettings(
            base_url="http://test:8080",
            model="test-model",
            batch_size=10,
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2]},
                {"embedding": [0.3, 0.4]},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response

        service = HTTPEmbeddingService(settings=settings, client=mock_client)
        results = await service.embed_batch(["text1", "text2"])

        assert len(results) == 2
        assert results[0].text == "text1"
        assert results[1].text == "text2"

    @pytest.mark.asyncio
    async def test_embed_empty_list(self) -> None:
        """Empty list returns empty results."""
        settings = EmbeddingSettings(base_url="http://test:8080")
        service = HTTPEmbeddingService(settings=settings)
        results = await service.embed_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_embed_http_error(self) -> None:
        """HTTP error raises EmbeddingError."""
        settings = EmbeddingSettings(base_url="http://test:8080")

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error",
            request=MagicMock(),
            response=mock_response,
        )

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response

        service = HTTPEmbeddingService(settings=settings, client=mock_client)

        with pytest.raises(EmbeddingError):
            await service.embed("test")

    @pytest.mark.asyncio
    async def test_embed_connection_error(self) -> None:
        """Connection error raises EmbeddingError."""
        settings = EmbeddingSettings(base_url="http://test:8080")

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = httpx.RequestError("Connection failed")

        service = HTTPEmbeddingService(settings=settings, client=mock_client)

        with pytest.raises(EmbeddingError):
            await service.embed("test")

    @pytest.mark.asyncio
    async def test_batch_chunking(self) -> None:
        """Large batches are chunked correctly."""
        settings = EmbeddingSettings(
            base_url="http://test:8080",
            model="test-model",
            batch_size=2,  # Small batch size
        )

        call_count = 0

        def make_response(*_args: object, **_kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            mock_response = MagicMock()
            # Return 2 embeddings per call
            mock_response.json.return_value = {
                "data": [
                    {"embedding": [0.1]},
                    {"embedding": [0.2]},
                ]
            }
            mock_response.raise_for_status = MagicMock()
            return mock_response

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = make_response

        service = HTTPEmbeddingService(settings=settings, client=mock_client)
        # 4 texts with batch_size=2 should make 2 requests
        results = await service.embed_batch(["t1", "t2", "t3", "t4"])

        assert len(results) == 4
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        """Service closes owned client."""
        settings = EmbeddingSettings(base_url="http://test:8080")

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        service = HTTPEmbeddingService(settings=settings, client=mock_client)
        service._owns_client = True  # Simulate owning the client

        await service.close()
        mock_client.aclose.assert_called_once()

