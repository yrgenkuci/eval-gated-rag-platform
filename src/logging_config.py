"""Structured logging configuration.

JSON output for production, human-readable for development.
"""

import logging
import sys
from datetime import UTC, datetime
from typing import Any

from src.config import Environment, get_settings


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra"):
            log_data["extra"] = record.extra

        # Add standard fields if present
        if record.funcName:
            log_data["function"] = record.funcName
        if record.pathname:
            log_data["file"] = f"{record.pathname}:{record.lineno}"

        # Manual JSON serialization to avoid import overhead
        return self._to_json(log_data)

    def _to_json(self, data: dict[str, Any]) -> str:
        """Convert dict to JSON string without external dependency."""
        import json

        return json.dumps(data, default=str, ensure_ascii=False)


class DevFormatter(logging.Formatter):
    """Human-readable formatter for development."""

    FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self.FORMAT, datefmt=self.DATE_FORMAT)


def setup_logging(
    level: str | None = None,
    json_output: bool | None = None,
) -> logging.Logger:
    """Configure application logging.

    Args:
        level: Log level override (default from settings).
        json_output: Force JSON output (default based on environment).

    Returns:
        Root logger instance.
    """
    settings = get_settings()

    # Determine log level
    log_level = level or settings.log_level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Determine output format
    if json_output is None:
        json_output = settings.environment != Environment.DEVELOPMENT

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)

    # Set formatter based on environment
    if json_output:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(DevFormatter())

    root_logger.addHandler(handler)

    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a named logger.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)

