"""Tests for application exceptions."""

from src.exceptions import (
    ConfigurationError,
    DocumentError,
    EmbeddingError,
    ErrorCode,
    EvaluationError,
    LLMError,
    RAGPlatformError,
    RetrievalError,
    ValidationError,
    VectorStoreError,
)


class TestErrorCode:
    """Tests for error codes."""

    def test_error_code_format(self) -> None:
        """Error codes follow RAG-XXXX format."""
        for code in ErrorCode:
            assert code.value.startswith("RAG-")
            assert len(code.value) == 8  # RAG-XXXX

    def test_error_code_uniqueness(self) -> None:
        """All error codes are unique."""
        codes = [code.value for code in ErrorCode]
        assert len(codes) == len(set(codes))


class TestRAGPlatformError:
    """Tests for base exception."""

    def test_basic_exception(self) -> None:
        """Base exception stores message and code."""
        error = RAGPlatformError("Something went wrong")
        assert error.message == "Something went wrong"
        assert error.code == ErrorCode.INTERNAL_ERROR
        assert error.details == {}

    def test_exception_with_code(self) -> None:
        """Exception can have custom code."""
        error = RAGPlatformError(
            "Not found",
            code=ErrorCode.DOCUMENT_NOT_FOUND,
        )
        assert error.code == ErrorCode.DOCUMENT_NOT_FOUND

    def test_exception_with_details(self) -> None:
        """Exception can have additional details."""
        error = RAGPlatformError(
            "Validation failed",
            code=ErrorCode.VALIDATION_ERROR,
            details={"field": "query", "reason": "too short"},
        )
        assert error.details == {"field": "query", "reason": "too short"}

    def test_to_dict(self) -> None:
        """Exception converts to API response dict."""
        error = RAGPlatformError(
            "Something went wrong",
            code=ErrorCode.INTERNAL_ERROR,
            details={"trace_id": "abc123"},
        )
        result = error.to_dict()

        assert result == {
            "error": {
                "code": "RAG-1000",
                "message": "Something went wrong",
                "details": {"trace_id": "abc123"},
            }
        }

    def test_str_representation(self) -> None:
        """Exception string is the message."""
        error = RAGPlatformError("Test error")
        assert str(error) == "Test error"

    def test_exception_is_raisable(self) -> None:
        """Exception can be raised and caught."""
        try:
            raise RAGPlatformError("Test error")
        except RAGPlatformError as e:
            assert e.message == "Test error"


class TestConfigurationError:
    """Tests for configuration exception."""

    def test_default_code(self) -> None:
        """ConfigurationError has correct default code."""
        error = ConfigurationError("Missing env var")
        assert error.code == ErrorCode.CONFIGURATION_ERROR

    def test_inherits_from_base(self) -> None:
        """ConfigurationError inherits from RAGPlatformError."""
        error = ConfigurationError("Missing env var")
        assert isinstance(error, RAGPlatformError)


class TestValidationError:
    """Tests for validation exception."""

    def test_default_code(self) -> None:
        """ValidationError has correct default code."""
        error = ValidationError("Invalid input")
        assert error.code == ErrorCode.VALIDATION_ERROR


class TestDocumentError:
    """Tests for document exception."""

    def test_default_code(self) -> None:
        """DocumentError has correct default code."""
        error = DocumentError("Document not found")
        assert error.code == ErrorCode.DOCUMENT_NOT_FOUND

    def test_custom_code(self) -> None:
        """DocumentError can have custom code."""
        error = DocumentError(
            "Failed to parse",
            code=ErrorCode.DOCUMENT_PARSE_ERROR,
        )
        assert error.code == ErrorCode.DOCUMENT_PARSE_ERROR


class TestEmbeddingError:
    """Tests for embedding exception."""

    def test_default_code(self) -> None:
        """EmbeddingError has correct default code."""
        error = EmbeddingError("Service unavailable")
        assert error.code == ErrorCode.EMBEDDING_SERVICE_ERROR


class TestVectorStoreError:
    """Tests for vector store exception."""

    def test_default_code(self) -> None:
        """VectorStoreError has correct default code."""
        error = VectorStoreError("Connection failed")
        assert error.code == ErrorCode.VECTOR_STORE_ERROR

    def test_custom_code(self) -> None:
        """VectorStoreError can have custom code."""
        error = VectorStoreError(
            "Collection not found",
            code=ErrorCode.COLLECTION_NOT_FOUND,
        )
        assert error.code == ErrorCode.COLLECTION_NOT_FOUND


class TestLLMError:
    """Tests for LLM exception."""

    def test_default_code(self) -> None:
        """LLMError has correct default code."""
        error = LLMError("Model unavailable")
        assert error.code == ErrorCode.LLM_SERVICE_ERROR

    def test_timeout_code(self) -> None:
        """LLMError can indicate timeout."""
        error = LLMError("Request timed out", code=ErrorCode.LLM_TIMEOUT)
        assert error.code == ErrorCode.LLM_TIMEOUT


class TestRetrievalError:
    """Tests for retrieval exception."""

    def test_default_code(self) -> None:
        """RetrievalError has correct default code."""
        error = RetrievalError("Search failed")
        assert error.code == ErrorCode.RETRIEVAL_ERROR


class TestEvaluationError:
    """Tests for evaluation exception."""

    def test_default_code(self) -> None:
        """EvaluationError has correct default code."""
        error = EvaluationError("Evaluation failed")
        assert error.code == ErrorCode.EVALUATION_ERROR

    def test_gold_set_code(self) -> None:
        """EvaluationError can indicate gold set issue."""
        error = EvaluationError(
            "Invalid gold set format",
            code=ErrorCode.GOLD_SET_ERROR,
        )
        assert error.code == ErrorCode.GOLD_SET_ERROR

