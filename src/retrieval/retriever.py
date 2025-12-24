"""Retriever interface and implementations."""

from abc import ABC, abstractmethod
from typing import Any

from src.embeddings.service import EmbeddingService
from src.exceptions import ErrorCode, RetrievalError
from src.logging_config import get_logger
from src.retrieval.models import RetrievalResult
from src.vectorstore.service import VectorStore

logger = get_logger(__name__)


class Retriever(ABC):
    """Abstract base class for retrievers.

    Defines the interface for retrieving relevant documents.
    """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant documents for a query.

        Args:
            query: The search query.
            top_k: Maximum number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of retrieval results ordered by relevance.

        Raises:
            RetrievalError: If retrieval fails.
        """
        ...


class SemanticRetriever(Retriever):
    """Semantic search retriever using embeddings and vector store.

    Embeds the query and finds similar vectors in the store.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        collection: str,
        score_threshold: float = 0.0,
    ) -> None:
        """Initialize the semantic retriever.

        Args:
            embedding_service: Service for generating embeddings.
            vector_store: Vector database for similarity search.
            collection: Name of the collection to search.
            score_threshold: Minimum score to include in results.
        """
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._collection = collection
        self._score_threshold = score_threshold

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve documents using semantic similarity.

        Args:
            query: The search query.
            top_k: Maximum number of results.
            filters: Optional payload filters.

        Returns:
            List of relevant results.

        Raises:
            RetrievalError: If retrieval fails.
        """
        if not query.strip():
            return []

        try:
            # Embed the query
            embedding_result = await self._embedding_service.embed(query)
            query_vector = embedding_result.embedding

            # Search vector store
            search_results = await self._vector_store.search(
                collection=self._collection,
                vector=query_vector,
                limit=top_k,
                filters=filters,
            )

            # Convert to retrieval results
            results: list[RetrievalResult] = []
            for sr in search_results:
                # Filter by score threshold
                if sr.score < self._score_threshold:
                    continue

                # Extract content and source from payload
                content = sr.payload.get("content", "")
                source = sr.payload.get("source", sr.id)

                results.append(
                    RetrievalResult(
                        content=content,
                        score=sr.score,
                        source=source,
                        metadata=sr.payload,
                    )
                )

            logger.debug(
                f"Retrieved {len(results)} results for query",
                extra={
                    "query_length": len(query),
                    "top_k": top_k,
                    "results_count": len(results),
                },
            )

            return results

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise RetrievalError(
                f"Failed to retrieve documents: {e}",
                code=ErrorCode.RETRIEVAL_ERROR,
                details={"query": query[:100], "error": str(e)},
            ) from e

