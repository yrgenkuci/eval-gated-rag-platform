"""Tests for logging configuration."""

import json
import logging
from unittest.mock import patch

from src.config import Environment, Settings
from src.logging_config import (
    DevFormatter,
    JSONFormatter,
    get_logger,
    setup_logging,
)


class TestJSONFormatter:
    """Tests for JSON log formatter."""

    def test_format_basic_message(self) -> None:
        """Basic log message is formatted as JSON."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert data["message"] == "Test message"
        assert "timestamp" in data

    def test_format_includes_file_info(self) -> None:
        """Log includes file and line information."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="/app/module.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["file"] == "/app/module.py:42"

    def test_format_with_exception(self) -> None:
        """Exception info is included in output."""
        formatter = JSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error",
            args=(),
            exc_info=exc_info,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert "exception" in data
        assert "ValueError" in data["exception"]


class TestDevFormatter:
    """Tests for development formatter."""

    def test_format_includes_level(self) -> None:
        """Development format includes level."""
        formatter = DevFormatter()
        record = logging.LogRecord(
            name="test.module",
            level=logging.WARNING,
            pathname="test.py",
            lineno=10,
            msg="Warning message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        assert "WARNING" in output
        assert "test.module" in output
        assert "Warning message" in output


class TestSetupLogging:
    """Tests for logging setup."""

    def test_returns_root_logger(self) -> None:
        """setup_logging returns root logger."""
        logger = setup_logging(level="INFO", json_output=False)
        assert logger is logging.getLogger()

    def test_uses_json_in_production(self) -> None:
        """JSON output is used in production environment."""
        mock_settings = Settings(environment=Environment.PRODUCTION)

        with patch("src.logging_config.get_settings", return_value=mock_settings):
            setup_logging()

        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JSONFormatter)

    def test_uses_dev_formatter_in_development(self) -> None:
        """Dev formatter is used in development environment."""
        mock_settings = Settings(environment=Environment.DEVELOPMENT)

        with patch("src.logging_config.get_settings", return_value=mock_settings):
            setup_logging()

        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, DevFormatter)

    def test_level_override(self) -> None:
        """Log level can be overridden."""
        setup_logging(level="DEBUG", json_output=False)
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_json_output_override(self) -> None:
        """JSON output can be forced."""
        mock_settings = Settings(environment=Environment.DEVELOPMENT)

        with patch("src.logging_config.get_settings", return_value=mock_settings):
            setup_logging(json_output=True)

        root = logging.getLogger()
        assert isinstance(root.handlers[0].formatter, JSONFormatter)


class TestGetLogger:
    """Tests for named logger retrieval."""

    def test_returns_named_logger(self) -> None:
        """get_logger returns a logger with the given name."""
        logger = get_logger("myapp.module")
        assert logger.name == "myapp.module"

    def test_logger_hierarchy(self) -> None:
        """Child loggers inherit from parent."""
        setup_logging(level="WARNING", json_output=False)
        _ = get_logger("myapp")  # Parent must exist for hierarchy
        child = get_logger("myapp.sub")

        # Child inherits effective level from root
        assert child.getEffectiveLevel() == logging.WARNING

