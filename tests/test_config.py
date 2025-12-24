"""Tests for application configuration."""

import os
from unittest.mock import patch

from src.config import (
    EmbeddingSettings,
    Environment,
    LLMSettings,
    QdrantSettings,
    Settings,
    get_settings,
)


class TestLLMSettings:
    """Tests for LLM configuration."""

    def test_default_values(self) -> None:
        """Default values point to local Ollama."""
        settings = LLMSettings()
        assert settings.base_url == "http://localhost:11434/v1"
        assert settings.model == "llama3:8b"
        assert settings.timeout == 120.0
        assert settings.max_tokens == 2048
        assert settings.temperature == 0.1

    def test_api_key_is_secret(self) -> None:
        """API key should be masked when printed."""
        settings = LLMSettings()
        assert "not-required" not in str(settings.api_key)
        assert settings.api_key.get_secret_value() == "not-required"

    def test_env_override(self) -> None:
        """Environment variables override defaults."""
        with patch.dict(os.environ, {"LLM_MODEL": "mistral:latest"}):
            settings = LLMSettings()
            assert settings.model == "mistral:latest"


class TestEmbeddingSettings:
    """Tests for embedding configuration."""

    def test_default_values(self) -> None:
        """Default values for embedding service."""
        settings = EmbeddingSettings()
        assert settings.base_url == "http://localhost:8080"
        assert settings.model == "BAAI/bge-large-en-v1.5"
        assert settings.batch_size == 32

    def test_env_override(self) -> None:
        """Environment variables override defaults."""
        with patch.dict(os.environ, {"EMBEDDING_BATCH_SIZE": "64"}):
            settings = EmbeddingSettings()
            assert settings.batch_size == 64


class TestQdrantSettings:
    """Tests for Qdrant configuration."""

    def test_default_values(self) -> None:
        """Default values for Qdrant."""
        settings = QdrantSettings()
        assert settings.url == "http://localhost:6333"
        assert settings.api_key is None
        assert settings.collection_name == "rag_documents"

    def test_api_key_optional(self) -> None:
        """API key is optional for local development."""
        settings = QdrantSettings()
        assert settings.api_key is None

    def test_api_key_is_secret_when_set(self) -> None:
        """API key should be masked when set."""
        with patch.dict(os.environ, {"QDRANT_API_KEY": "secret-key"}):
            settings = QdrantSettings()
            assert settings.api_key is not None
            assert "secret-key" not in str(settings.api_key)
            assert settings.api_key.get_secret_value() == "secret-key"


class TestSettings:
    """Tests for main application settings."""

    def test_default_environment(self) -> None:
        """Default environment is development."""
        settings = Settings()
        assert settings.environment == Environment.DEVELOPMENT

    def test_default_api_settings(self) -> None:
        """Default API host and port."""
        settings = Settings()
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000

    def test_nested_settings_loaded(self) -> None:
        """Nested settings are initialized."""
        settings = Settings()
        assert isinstance(settings.llm, LLMSettings)
        assert isinstance(settings.embedding, EmbeddingSettings)
        assert isinstance(settings.qdrant, QdrantSettings)

    def test_environment_enum(self) -> None:
        """Environment can be set via string."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            settings = Settings()
            assert settings.environment == Environment.PRODUCTION


class TestGetSettings:
    """Tests for settings singleton."""

    def test_returns_settings_instance(self) -> None:
        """get_settings returns a Settings instance."""
        get_settings.cache_clear()
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_caching(self) -> None:
        """Settings are cached."""
        get_settings.cache_clear()
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

