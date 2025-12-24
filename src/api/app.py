"""FastAPI application entry point.

Configures the application with logging, exception handling, and health checks.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src import __version__
from src.config import get_settings
from src.exceptions import RAGPlatformError
from src.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    settings = get_settings()
    setup_logging(level=settings.log_level)
    logger.info(
        "Starting RAG Platform",
        extra={
            "version": __version__,
            "environment": settings.environment.value,
        },
    )

    yield

    # Shutdown
    logger.info("Shutting down RAG Platform")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title="Eval-Gated RAG Platform",
        description="Production-ready RAG system with evaluation-driven CI/CD",
        version=__version__,
        lifespan=lifespan,
        debug=settings.debug,
    )

    # Register exception handlers
    app.add_exception_handler(RAGPlatformError, rag_exception_handler)

    # Register routes
    app.add_api_route("/health", health_check, methods=["GET"], tags=["Health"])
    app.add_api_route("/health/ready", readiness_check, methods=["GET"], tags=["Health"])
    app.add_api_route("/health/live", liveness_check, methods=["GET"], tags=["Health"])

    return app


async def rag_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle RAGPlatformError exceptions.

    Converts exceptions to structured JSON responses.
    """
    # Type narrow to RAGPlatformError
    if not isinstance(exc, RAGPlatformError):
        # Should not happen, but handle gracefully
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "RAG-1000", "message": str(exc), "details": {}}},
        )

    logger.error(
        f"Request failed: {exc.message}",
        extra={
            "error_code": exc.code.value,
            "path": request.url.path,
            "details": exc.details,
        },
    )

    # Map error codes to HTTP status codes
    status_code = _get_status_code(exc.code.value)

    return JSONResponse(
        status_code=status_code,
        content=exc.to_dict(),
    )


def _get_status_code(error_code: str) -> int:
    """Map error code to HTTP status code."""
    # Validation errors -> 400
    if error_code in ("RAG-1002",):
        return 400

    # Not found errors -> 404
    if error_code in ("RAG-2000", "RAG-4001", "RAG-6001"):
        return 404

    # Conflict errors -> 409
    if error_code in ("RAG-4002",):
        return 409

    # Rate limit -> 429
    if error_code in ("RAG-5002",):
        return 429

    # Timeout -> 504
    if error_code in ("RAG-5001",):
        return 504

    # Default to 500 for internal errors
    return 500


async def health_check() -> dict[str, Any]:
    """Basic health check endpoint.

    Returns:
        Health status with version and timestamp.
    """
    return {
        "status": "healthy",
        "version": __version__,
        "timestamp": datetime.now(UTC).isoformat(),
    }


async def readiness_check() -> dict[str, Any]:
    """Kubernetes readiness probe.

    Checks if the service is ready to accept traffic.
    Future: Will check database and service connections.

    Returns:
        Readiness status with component checks.
    """
    # TODO: Add actual dependency checks (Qdrant, LLM service)
    checks: dict[str, str] = {
        "config": "ok",
    }

    all_ok = all(v == "ok" for v in checks.values())

    return {
        "status": "ready" if all_ok else "not_ready",
        "checks": checks,
        "timestamp": datetime.now(UTC).isoformat(),
    }


async def liveness_check() -> dict[str, str]:
    """Kubernetes liveness probe.

    Simple check that the service is running.

    Returns:
        Liveness status.
    """
    return {"status": "alive"}


# Create the application instance
app = create_app()

