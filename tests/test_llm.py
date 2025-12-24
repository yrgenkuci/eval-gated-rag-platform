"""Tests for LLM module."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.config import LLMSettings
from src.exceptions import ErrorCode, LLMError
from src.llm.client import OpenAICompatibleClient
from src.llm.models import GenerationResult, Message, Role
from src.llm.prompts import RAGPromptTemplate


class TestMessage:
    """Tests for Message model."""

    def test_create_message(self) -> None:
        """Message can be created."""
        msg = Message(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"

    def test_role_values(self) -> None:
        """Role enum has expected values."""
        assert Role.SYSTEM.value == "system"
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"


class TestGenerationResult:
    """Tests for GenerationResult model."""

    def test_create_result(self) -> None:
        """Result can be created."""
        result = GenerationResult(
            content="Generated text",
            model="test-model",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )
        assert result.content == "Generated text"
        assert result.model == "test-model"
        assert result.total_tokens == 30


class TestOpenAICompatibleClient:
    """Tests for OpenAICompatibleClient."""

    def test_model_name(self) -> None:
        """Client returns configured model name."""
        settings = LLMSettings(model="llama3:8b")
        client = OpenAICompatibleClient(settings=settings)
        assert client.model_name == "llama3:8b"

    @pytest.mark.asyncio
    async def test_generate(self) -> None:
        """Client generates text."""
        settings = LLMSettings(
            base_url="http://test:11434/v1",
            model="test-model",
        )

        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Generated response"}}],
            "model": "test-model",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response

        client = OpenAICompatibleClient(settings=settings, client=mock_client)

        result = await client.generate([
            Message(role=Role.USER, content="Hello"),
        ])

        assert result.content == "Generated response"
        assert result.model == "test-model"
        assert result.total_tokens == 30

    @pytest.mark.asyncio
    async def test_generate_text(self) -> None:
        """Client generates text from simple prompt."""
        settings = LLMSettings(base_url="http://test:11434/v1")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}}],
            "usage": {},
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response

        client = OpenAICompatibleClient(settings=settings, client=mock_client)

        result = await client.generate_text(
            prompt="What is 2+2?",
            system_prompt="You are a calculator.",
        )

        assert result.content == "Response"

        # Verify both system and user messages were sent
        call_kwargs = mock_client.post.call_args.kwargs
        messages = call_kwargs["json"]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_timeout_error(self) -> None:
        """Timeout raises LLMError with correct code."""
        settings = LLMSettings(base_url="http://test:11434/v1")

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = httpx.TimeoutException("Timeout")

        client = OpenAICompatibleClient(settings=settings, client=mock_client)

        with pytest.raises(LLMError) as exc_info:
            await client.generate([Message(role=Role.USER, content="Hello")])

        assert exc_info.value.code == ErrorCode.LLM_TIMEOUT

    @pytest.mark.asyncio
    async def test_rate_limit_error(self) -> None:
        """Rate limit returns correct error code."""
        settings = LLMSettings(base_url="http://test:11434/v1")

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Rate limit",
            request=MagicMock(),
            response=mock_response,
        )

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response

        client = OpenAICompatibleClient(settings=settings, client=mock_client)

        with pytest.raises(LLMError) as exc_info:
            await client.generate([Message(role=Role.USER, content="Hello")])

        assert exc_info.value.code == ErrorCode.LLM_RATE_LIMIT

    @pytest.mark.asyncio
    async def test_connection_error(self) -> None:
        """Connection error raises LLMError."""
        settings = LLMSettings(base_url="http://test:11434/v1")

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = httpx.RequestError("Connection failed")

        client = OpenAICompatibleClient(settings=settings, client=mock_client)

        with pytest.raises(LLMError) as exc_info:
            await client.generate([Message(role=Role.USER, content="Hello")])

        assert exc_info.value.code == ErrorCode.LLM_SERVICE_ERROR

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        """Client closes properly."""
        settings = LLMSettings(base_url="http://test:11434/v1")
        mock_client = AsyncMock(spec=httpx.AsyncClient)

        client = OpenAICompatibleClient(settings=settings, client=mock_client)
        client._owns_client = True

        await client.close()

        mock_client.aclose.assert_called_once()


class TestRAGPromptTemplate:
    """Tests for RAGPromptTemplate."""

    def test_default_prompts(self) -> None:
        """Template has default prompts."""
        template = RAGPromptTemplate()
        assert "helpful assistant" in template.system_prompt
        assert "{context}" in template.user_template
        assert "{question}" in template.user_template

    def test_custom_prompts(self) -> None:
        """Template accepts custom prompts."""
        template = RAGPromptTemplate(
            system_prompt="Custom system",
            user_template="Q: {question}\nC: {context}",
        )
        assert template.system_prompt == "Custom system"
        assert "Q:" in template.user_template

    def test_format(self) -> None:
        """Template formats correctly."""
        template = RAGPromptTemplate()
        result = template.format(
            context="Some context here",
            question="What is this?",
        )
        assert "Some context here" in result
        assert "What is this?" in result

    def test_format_context(self) -> None:
        """Context chunks are combined."""
        template = RAGPromptTemplate()
        result = template.format_context(["Chunk 1", "Chunk 2", "Chunk 3"])
        assert "Chunk 1" in result
        assert "Chunk 2" in result
        assert "---" in result  # Default separator

    def test_format_context_custom_separator(self) -> None:
        """Custom separator is used."""
        template = RAGPromptTemplate()
        result = template.format_context(["A", "B"], separator="\n\n")
        assert result == "A\n\nB"

    def test_build_prompt(self) -> None:
        """Build prompt returns system and user prompts."""
        template = RAGPromptTemplate()
        system, user = template.build_prompt(
            question="What is AI?",
            chunks=["AI is artificial intelligence."],
        )

        assert "helpful assistant" in system
        assert "What is AI?" in user
        assert "artificial intelligence" in user

