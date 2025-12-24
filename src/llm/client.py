"""LLM client interface and implementations."""

from abc import ABC, abstractmethod

import httpx

from src.config import LLMSettings, get_settings
from src.exceptions import ErrorCode, LLMError
from src.llm.models import GenerationResult, Message, Role
from src.logging_config import get_logger

logger = get_logger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients.

    Defines the interface for generating text with LLMs.
    """

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """Generate text from messages.

        Args:
            messages: Conversation messages.
            temperature: Sampling temperature override.
            max_tokens: Maximum tokens override.

        Returns:
            GenerationResult with generated text.

        Raises:
            LLMError: If generation fails.
        """
        ...

    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """Generate text from a simple prompt.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature override.
            max_tokens: Maximum tokens override.

        Returns:
            GenerationResult with generated text.

        Raises:
            LLMError: If generation fails.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        ...


class OpenAICompatibleClient(LLMClient):
    """LLM client for OpenAI-compatible APIs.

    Works with:
    - Ollama (localhost:11434/v1)
    - vLLM
    - OpenAI API
    - Any OpenAI-compatible endpoint
    """

    def __init__(
        self,
        settings: LLMSettings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize the OpenAI-compatible client.

        Args:
            settings: LLM configuration.
            client: HTTP client (for testing).
        """
        self._settings = settings or get_settings().llm
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._settings.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._settings.model

    async def generate(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """Generate text using chat completions API."""
        client = await self._get_client()
        url = f"{self._settings.base_url}/chat/completions"

        # Build request payload
        payload = {
            "model": self._settings.model,
            "messages": [
                {"role": msg.role.value, "content": msg.content} for msg in messages
            ],
            "temperature": temperature or self._settings.temperature,
            "max_tokens": max_tokens or self._settings.max_tokens,
        }

        # Add API key header if provided
        headers = {}
        if self._settings.api_key:
            api_key = self._settings.api_key.get_secret_value()
            if api_key != "not-required":
                headers["Authorization"] = f"Bearer {api_key}"

        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()

        except httpx.TimeoutException as e:
            logger.error(f"LLM request timed out: {e}")
            raise LLMError(
                "LLM request timed out",
                code=ErrorCode.LLM_TIMEOUT,
                details={"timeout": self._settings.timeout},
            ) from e

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            logger.error(f"LLM request failed: {status}")

            if status == 429:
                raise LLMError(
                    "Rate limit exceeded",
                    code=ErrorCode.LLM_RATE_LIMIT,
                    details={"status_code": status},
                ) from e

            raise LLMError(
                f"LLM service returned {status}",
                code=ErrorCode.LLM_SERVICE_ERROR,
                details={"status_code": status},
            ) from e

        except httpx.RequestError as e:
            logger.error(f"LLM connection error: {e}")
            raise LLMError(
                f"Failed to connect to LLM service: {e}",
                code=ErrorCode.LLM_SERVICE_ERROR,
                details={"url": url},
            ) from e

        # Parse response
        try:
            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]
            usage = data.get("usage", {})

            return GenerationResult(
                content=message["content"],
                model=data.get("model", self._settings.model),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )

        except (KeyError, IndexError) as e:
            raise LLMError(
                f"Invalid response from LLM: {e}",
                code=ErrorCode.LLM_SERVICE_ERROR,
                details={"error": str(e)},
            ) from e

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """Generate text from a simple prompt."""
        messages: list[Message] = []

        if system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=system_prompt))

        messages.append(Message(role=Role.USER, content=prompt))

        return await self.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

