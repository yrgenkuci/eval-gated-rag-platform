"""Vector store interface and Qdrant implementation."""

from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from src.config import QdrantSettings, get_settings
from src.exceptions import ErrorCode, VectorStoreError
from src.logging_config import get_logger
from src.vectorstore.models import SearchResult, VectorRecord

logger = get_logger(__name__)


class VectorStore(ABC):
    """Abstract base class for vector stores.

    Defines the interface for storing and searching vectors.
    """

    @abstractmethod
    async def create_collection(
        self,
        name: str,
        dimensions: int,
    ) -> None:
        """Create a new collection.

        Args:
            name: Collection name.
            dimensions: Vector dimensions.

        Raises:
            VectorStoreError: If creation fails.
        """
        ...

    @abstractmethod
    async def delete_collection(self, name: str) -> None:
        """Delete a collection.

        Args:
            name: Collection name.

        Raises:
            VectorStoreError: If deletion fails.
        """
        ...

    @abstractmethod
    async def collection_exists(self, name: str) -> bool:
        """Check if a collection exists.

        Args:
            name: Collection name.

        Returns:
            True if collection exists.
        """
        ...

    @abstractmethod
    async def upsert(
        self,
        collection: str,
        records: list[VectorRecord],
    ) -> int:
        """Insert or update records.

        Args:
            collection: Collection name.
            records: Records to upsert.

        Returns:
            Number of records upserted.

        Raises:
            VectorStoreError: If upsert fails.
        """
        ...

    @abstractmethod
    async def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors.

        Args:
            collection: Collection name.
            vector: Query vector.
            limit: Maximum results to return.
            filters: Optional payload filters.

        Returns:
            List of search results.

        Raises:
            VectorStoreError: If search fails.
        """
        ...

    @abstractmethod
    async def delete(
        self,
        collection: str,
        ids: list[str],
    ) -> int:
        """Delete records by ID.

        Args:
            collection: Collection name.
            ids: Record IDs to delete.

        Returns:
            Number of records deleted.

        Raises:
            VectorStoreError: If deletion fails.
        """
        ...


class QdrantVectorStore(VectorStore):
    """Qdrant vector store implementation."""

    def __init__(
        self,
        settings: QdrantSettings | None = None,
        client: AsyncQdrantClient | None = None,
    ) -> None:
        """Initialize Qdrant vector store.

        Args:
            settings: Qdrant configuration.
            client: Existing client (for testing).
        """
        self._settings = settings or get_settings().qdrant
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> AsyncQdrantClient:
        """Get or create Qdrant client."""
        if self._client is None:
            api_key = None
            if self._settings.api_key:
                api_key = self._settings.api_key.get_secret_value()

            self._client = AsyncQdrantClient(
                url=self._settings.url,
                api_key=api_key,
            )
        return self._client

    async def close(self) -> None:
        """Close the Qdrant client."""
        if self._owns_client and self._client is not None:
            await self._client.close()
            self._client = None

    async def create_collection(
        self,
        name: str,
        dimensions: int,
    ) -> None:
        """Create a new Qdrant collection."""
        client = await self._get_client()

        try:
            exists = await client.collection_exists(name)
            if exists:
                raise VectorStoreError(
                    f"Collection already exists: {name}",
                    code=ErrorCode.COLLECTION_EXISTS,
                    details={"collection": name},
                )

            await client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=dimensions,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created collection: {name}", extra={"dimensions": dimensions})

        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(
                f"Failed to create collection: {e}",
                code=ErrorCode.VECTOR_STORE_ERROR,
                details={"collection": name, "error": str(e)},
            ) from e

    async def delete_collection(self, name: str) -> None:
        """Delete a Qdrant collection."""
        client = await self._get_client()

        try:
            exists = await client.collection_exists(name)
            if not exists:
                raise VectorStoreError(
                    f"Collection not found: {name}",
                    code=ErrorCode.COLLECTION_NOT_FOUND,
                    details={"collection": name},
                )

            await client.delete_collection(name)
            logger.info(f"Deleted collection: {name}")

        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(
                f"Failed to delete collection: {e}",
                code=ErrorCode.VECTOR_STORE_ERROR,
                details={"collection": name, "error": str(e)},
            ) from e

    async def collection_exists(self, name: str) -> bool:
        """Check if collection exists."""
        client = await self._get_client()
        try:
            return await client.collection_exists(name)
        except Exception as e:
            raise VectorStoreError(
                f"Failed to check collection: {e}",
                code=ErrorCode.VECTOR_STORE_ERROR,
                details={"collection": name, "error": str(e)},
            ) from e

    async def upsert(
        self,
        collection: str,
        records: list[VectorRecord],
    ) -> int:
        """Upsert records into collection."""
        if not records:
            return 0

        client = await self._get_client()

        try:
            points = [
                PointStruct(
                    id=record.id or str(uuid4()),
                    vector=record.vector,
                    payload=record.payload,
                )
                for record in records
            ]

            await client.upsert(
                collection_name=collection,
                points=points,
            )

            logger.debug(
                f"Upserted {len(points)} records",
                extra={"collection": collection},
            )
            return len(points)

        except Exception as e:
            raise VectorStoreError(
                f"Failed to upsert records: {e}",
                code=ErrorCode.VECTOR_STORE_ERROR,
                details={"collection": collection, "error": str(e)},
            ) from e

    async def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors."""
        client = await self._get_client()

        try:
            # Build filter if provided
            query_filter = None
            if filters:
                conditions = [
                    FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in filters.items()
                ]
                query_filter = Filter(must=conditions)  # type: ignore[arg-type]

            results = await client.query_points(
                collection_name=collection,
                query=vector,
                limit=limit,
                query_filter=query_filter,
            )

            return [
                SearchResult(
                    id=str(point.id),
                    score=point.score if point.score is not None else 0.0,
                    payload=dict(point.payload) if point.payload else {},
                )
                for point in results.points
            ]

        except Exception as e:
            raise VectorStoreError(
                f"Failed to search: {e}",
                code=ErrorCode.VECTOR_STORE_ERROR,
                details={"collection": collection, "error": str(e)},
            ) from e

    async def delete(
        self,
        collection: str,
        ids: list[str],
    ) -> int:
        """Delete records by ID."""
        if not ids:
            return 0

        client = await self._get_client()

        try:
            await client.delete(
                collection_name=collection,
                points_selector=PointIdsList(points=ids),  # type: ignore[arg-type]
            )

            logger.debug(
                f"Deleted {len(ids)} records",
                extra={"collection": collection},
            )
            return len(ids)

        except Exception as e:
            raise VectorStoreError(
                f"Failed to delete records: {e}",
                code=ErrorCode.VECTOR_STORE_ERROR,
                details={"collection": collection, "error": str(e)},
            ) from e

