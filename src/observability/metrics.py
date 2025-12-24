"""Prometheus metrics for RAG platform.

Provides metrics instrumentation for:
- HTTP request latency and counts
- LLM token usage and latency
- Embedding request latency
- Retrieval metrics (chunks, scores)
- Evaluation pass rates
"""

import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.logging_config import get_logger

logger = get_logger(__name__)

# HTTP Request Metrics
HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint", "status_code"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

HTTP_REQUEST_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

# RAG Query Metrics
RAG_QUERY_DURATION = Histogram(
    "rag_query_duration_seconds",
    "RAG query duration in seconds",
    ["status"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

RAG_QUERY_TOTAL = Counter(
    "rag_queries_total",
    "Total RAG queries",
    ["status"],
)

# LLM Metrics
LLM_REQUEST_DURATION = Histogram(
    "llm_request_duration_seconds",
    "LLM request duration in seconds",
    ["model", "status"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)

LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total LLM tokens used",
    ["model", "type"],  # "type" label values: prompt, completion
)

LLM_REQUEST_TOTAL = Counter(
    "llm_requests_total",
    "Total LLM requests",
    ["model", "status"],
)

# Embedding Metrics
EMBEDDING_REQUEST_DURATION = Histogram(
    "embedding_request_duration_seconds",
    "Embedding request duration in seconds",
    ["model", "status"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

EMBEDDING_REQUEST_TOTAL = Counter(
    "embedding_requests_total",
    "Total embedding requests",
    ["model", "status"],
)

EMBEDDING_BATCH_SIZE = Histogram(
    "embedding_batch_size",
    "Embedding batch size",
    ["model"],
    buckets=[1, 5, 10, 25, 50, 100, 250, 500],
)

# Retrieval Metrics
RETRIEVAL_CHUNKS_RETURNED = Histogram(
    "retrieval_chunks_returned",
    "Number of chunks returned per retrieval",
    buckets=[0, 1, 2, 3, 5, 10, 20, 50],
)

RETRIEVAL_TOP_SCORE = Histogram(
    "retrieval_top_score",
    "Top retrieval score per query",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Evaluation Metrics
EVAL_PASS_RATE = Gauge(
    "eval_pass_rate",
    "Current evaluation pass rate",
    ["gold_set"],
)

EVAL_METRIC_SCORE = Gauge(
    "eval_metric_score",
    "Current evaluation metric score",
    ["gold_set", "metric"],
)

# Vector Store Metrics
VECTORSTORE_OPERATION_DURATION = Histogram(
    "vectorstore_operation_duration_seconds",
    "Vector store operation duration",
    ["operation", "status"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect HTTP request metrics."""

    def __init__(self, app: ASGIApp) -> None:
        """Initialize the middleware."""
        super().__init__(app)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Process request and collect metrics."""
        # Skip metrics endpoint to avoid recursion
        if request.url.path == "/metrics":
            return await call_next(request)

        start_time = time.perf_counter()

        response = await call_next(request)

        duration = time.perf_counter() - start_time

        # Normalize endpoint for cardinality control
        endpoint = self._normalize_endpoint(request.url.path)

        HTTP_REQUEST_DURATION.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=response.status_code,
        ).observe(duration)

        HTTP_REQUEST_TOTAL.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=response.status_code,
        ).inc()

        return response

    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint path to reduce cardinality."""
        # Group health endpoints
        if path.startswith("/health"):
            return "/health"
        # Keep API versioned paths
        if path.startswith("/api/v1/"):
            parts = path.split("/")
            if len(parts) >= 4:
                return f"/api/v1/{parts[3]}"
        return path


def get_metrics() -> bytes:
    """Generate Prometheus metrics output."""
    return generate_latest()


def get_metrics_content_type() -> str:
    """Get the content type for metrics response."""
    return CONTENT_TYPE_LATEST


def track_llm_request(
    model: str,
    duration: float,
    prompt_tokens: int,
    completion_tokens: int,
    success: bool = True,
) -> None:
    """Track LLM request metrics.

    Args:
        model: LLM model name.
        duration: Request duration in seconds.
        prompt_tokens: Number of prompt tokens.
        completion_tokens: Number of completion tokens.
        success: Whether the request succeeded.
    """
    status = "success" if success else "error"

    LLM_REQUEST_DURATION.labels(model=model, status=status).observe(duration)
    LLM_REQUEST_TOTAL.labels(model=model, status=status).inc()

    if success:
        LLM_TOKENS_TOTAL.labels(model=model, type="prompt").inc(prompt_tokens)
        LLM_TOKENS_TOTAL.labels(model=model, type="completion").inc(completion_tokens)


def track_embedding_request(
    model: str,
    duration: float,
    batch_size: int,
    success: bool = True,
) -> None:
    """Track embedding request metrics.

    Args:
        model: Embedding model name.
        duration: Request duration in seconds.
        batch_size: Number of texts in the batch.
        success: Whether the request succeeded.
    """
    status = "success" if success else "error"

    EMBEDDING_REQUEST_DURATION.labels(model=model, status=status).observe(duration)
    EMBEDDING_REQUEST_TOTAL.labels(model=model, status=status).inc()
    EMBEDDING_BATCH_SIZE.labels(model=model).observe(batch_size)


def track_retrieval_request(
    chunks_returned: int,
    top_score: float,
) -> None:
    """Track retrieval request metrics.

    Args:
        chunks_returned: Number of chunks returned.
        top_score: Highest relevance score.
    """
    RETRIEVAL_CHUNKS_RETURNED.observe(chunks_returned)
    if top_score > 0:
        RETRIEVAL_TOP_SCORE.observe(top_score)


def update_eval_metrics(
    gold_set: str,
    pass_rate: float,
    metric_scores: dict[str, float],
) -> None:
    """Update evaluation metrics gauges.

    Args:
        gold_set: Name of the gold set.
        pass_rate: Current pass rate (0-1).
        metric_scores: Dict of metric name to score.
    """
    EVAL_PASS_RATE.labels(gold_set=gold_set).set(pass_rate)

    for metric, score in metric_scores.items():
        EVAL_METRIC_SCORE.labels(gold_set=gold_set, metric=metric).set(score)

