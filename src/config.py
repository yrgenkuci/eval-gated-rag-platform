"""Application configuration using Pydantic Settings.

All configuration is loaded from environment variables.
No secrets are hardcoded.
"""

from enum import Enum
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LLMSettings(BaseSettings):
    """LLM service configuration.

    Supports both Ollama (dev) and vLLM (prod) via OpenAI-compatible API.
    """

    model_config = SettingsConfigDict(env_prefix="LLM_")

    base_url: str = Field(
        default="http://localhost:11434/v1",
        description="LLM API base URL (Ollama default)",
    )
    model: str = Field(
        default="llama3:8b",
        description="Model name to use for generation",
    )
    api_key: SecretStr = Field(
        default=SecretStr("not-required"),
        description="API key (not required for Ollama)",
    )
    timeout: float = Field(
        default=120.0,
        description="Request timeout in seconds",
    )
    max_tokens: int = Field(
        default=2048,
        description="Maximum tokens in response",
    )
    temperature: float = Field(
        default=0.1,
        description="Sampling temperature (lower = more deterministic)",
    )


class EmbeddingSettings(BaseSettings):
    """Embedding service configuration."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    base_url: str = Field(
        default="http://localhost:8080",
        description="Embedding service base URL",
    )
    model: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="Embedding model name",
    )
    batch_size: int = Field(
        default=32,
        description="Batch size for embedding requests",
    )


class QdrantSettings(BaseSettings):
    """Qdrant vector database configuration."""

    model_config = SettingsConfigDict(env_prefix="QDRANT_")

    url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL",
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="Qdrant API key (optional for local)",
    )
    collection_name: str = Field(
        default="rag_documents",
        description="Default collection name",
    )


class Settings(BaseSettings):
    """Main application settings.

    Aggregates all configuration sections.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application settings
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )

    # API settings
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host",
    )
    api_port: int = Field(
        default=8000,
        description="API server port",
    )

    # Nested settings
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings.

    Returns:
        Settings instance loaded from environment.
    """
    return Settings()

