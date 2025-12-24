"""Embedding service interface and implementations."""

from abc import ABC, abstractmethod

import httpx

from src.config import EmbeddingSettings, get_settings
from src.embeddings.models import EmbeddingResult
from src.exceptions import EmbeddingError, ErrorCode
from src.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingService(ABC):
    """Abstract base class for embedding services.

    Defines the interface for generating text embeddings.
    """

    @abstractmethod
    async def embed(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            EmbeddingResult with vector.

        Raises:
            EmbeddingError: If embedding fails.
        """
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of EmbeddingResult objects.

        Raises:
            EmbeddingError: If embedding fails.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name used for embeddings."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Get the embedding dimensions."""
        ...


class HTTPEmbeddingService(EmbeddingService):
    """Embedding service using HTTP API.

    Compatible with OpenAI-style embedding APIs and
    text-embeddings-inference (TEI) servers.
    """

    # Known model dimensions
    MODEL_DIMENSIONS = {
        "BAAI/bge-large-en-v1.5": 1024,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-small-en-v1.5": 384,
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    def __init__(
        self,
        settings: EmbeddingSettings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize the HTTP embedding service.

        Args:
            settings: Embedding configuration. Uses defaults if not provided.
            client: HTTP client. Creates new one if not provided.
        """
        self._settings = settings or get_settings().embedding
        self._client = client
        self._owns_client = client is None
        self._dimensions: int | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client if we own it."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._settings.model

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        if self._dimensions is not None:
            return self._dimensions

        # Try to get from known models
        if self._settings.model in self.MODEL_DIMENSIONS:
            return self.MODEL_DIMENSIONS[self._settings.model]

        # Default for BGE-large
        return 1024

    async def embed(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            EmbeddingResult with vector.

        Raises:
            EmbeddingError: If embedding fails.
        """
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of EmbeddingResult objects.

        Raises:
            EmbeddingError: If embedding fails.
        """
        if not texts:
            return []

        client = await self._get_client()
        url = f"{self._settings.base_url}/embeddings"

        # Process in batches
        all_results: list[EmbeddingResult] = []
        batch_size = self._settings.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_results = await self._embed_batch_request(client, url, batch)
            all_results.extend(batch_results)

        return all_results

    async def _embed_batch_request(
        self,
        client: httpx.AsyncClient,
        url: str,
        texts: list[str],
    ) -> list[EmbeddingResult]:
        """Make embedding request for a batch.

        Args:
            client: HTTP client.
            url: Embedding endpoint URL.
            texts: Batch of texts.

        Returns:
            List of EmbeddingResult objects.

        Raises:
            EmbeddingError: If request fails.
        """
        payload = {
            "input": texts,
            "model": self._settings.model,
        }

        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"Embedding request failed: {e.response.status_code}",
                extra={"url": url, "status": e.response.status_code},
            )
            raise EmbeddingError(
                f"Embedding service returned {e.response.status_code}",
                code=ErrorCode.EMBEDDING_SERVICE_ERROR,
                details={"status_code": e.response.status_code},
            ) from e
        except httpx.RequestError as e:
            logger.error(
                f"Embedding request error: {e}",
                extra={"url": url},
            )
            raise EmbeddingError(
                f"Failed to connect to embedding service: {e}",
                code=ErrorCode.EMBEDDING_SERVICE_ERROR,
                details={"url": url},
            ) from e

        try:
            data = response.json()
            embeddings = data.get("data", [])

            results: list[EmbeddingResult] = []
            for i, emb_data in enumerate(embeddings):
                embedding = emb_data.get("embedding", [])

                # Update dimensions if we learn them
                if self._dimensions is None and embedding:
                    self._dimensions = len(embedding)

                results.append(
                    EmbeddingResult(
                        text=texts[i],
                        embedding=embedding,
                        model=self._settings.model,
                        dimensions=len(embedding),
                    )
                )

            return results

        except (KeyError, IndexError, ValueError) as e:
            raise EmbeddingError(
                f"Invalid response from embedding service: {e}",
                code=ErrorCode.EMBEDDING_SERVICE_ERROR,
                details={"error": str(e)},
            ) from e

