"""Application exception hierarchy.

All custom exceptions inherit from RAGPlatformError.
Each exception has an error code for structured error handling.
"""

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """Error codes for structured error handling."""

    # General errors (1xxx)
    INTERNAL_ERROR = "RAG-1000"
    CONFIGURATION_ERROR = "RAG-1001"
    VALIDATION_ERROR = "RAG-1002"

    # Document processing errors (2xxx)
    DOCUMENT_NOT_FOUND = "RAG-2000"
    DOCUMENT_PARSE_ERROR = "RAG-2001"
    CHUNK_ERROR = "RAG-2002"

    # Embedding errors (3xxx)
    EMBEDDING_SERVICE_ERROR = "RAG-3000"
    EMBEDDING_DIMENSION_MISMATCH = "RAG-3001"

    # Vector store errors (4xxx)
    VECTOR_STORE_ERROR = "RAG-4000"
    COLLECTION_NOT_FOUND = "RAG-4001"
    COLLECTION_EXISTS = "RAG-4002"

    # LLM errors (5xxx)
    LLM_SERVICE_ERROR = "RAG-5000"
    LLM_TIMEOUT = "RAG-5001"
    LLM_RATE_LIMIT = "RAG-5002"
    LLM_CONTEXT_LENGTH = "RAG-5003"

    # Retrieval errors (6xxx)
    RETRIEVAL_ERROR = "RAG-6000"
    NO_RESULTS_FOUND = "RAG-6001"

    # Evaluation errors (7xxx)
    EVALUATION_ERROR = "RAG-7000"
    GOLD_SET_ERROR = "RAG-7001"
    METRIC_ERROR = "RAG-7002"


class RAGPlatformError(Exception):
    """Base exception for all RAG platform errors.

    Attributes:
        message: Human-readable error message.
        code: Structured error code.
        details: Additional error context.
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": {
                "code": self.code.value,
                "message": self.message,
                "details": self.details,
            }
        }


class ConfigurationError(RAGPlatformError):
    """Configuration or environment error."""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, ErrorCode.CONFIGURATION_ERROR, details)


class ValidationError(RAGPlatformError):
    """Input validation error."""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, ErrorCode.VALIDATION_ERROR, details)


class DocumentError(RAGPlatformError):
    """Document processing error."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.DOCUMENT_NOT_FOUND,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code, details)


class EmbeddingError(RAGPlatformError):
    """Embedding service error."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.EMBEDDING_SERVICE_ERROR,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code, details)


class VectorStoreError(RAGPlatformError):
    """Vector store operation error."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.VECTOR_STORE_ERROR,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code, details)


class LLMError(RAGPlatformError):
    """LLM service error."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.LLM_SERVICE_ERROR,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code, details)


class RetrievalError(RAGPlatformError):
    """Retrieval operation error."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.RETRIEVAL_ERROR,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code, details)


class EvaluationError(RAGPlatformError):
    """Evaluation harness error."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.EVALUATION_ERROR,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code, details)

