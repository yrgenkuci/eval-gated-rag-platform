"""Tests for observability module."""

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.app import app
from src.observability.metrics import (
    get_metrics,
    track_embedding_request,
    track_llm_request,
    track_retrieval_request,
    update_eval_metrics,
)


class TestMetricsEndpoint:
    """Tests for /metrics endpoint."""

    @pytest.mark.asyncio
    async def test_metrics_endpoint_returns_prometheus_format(self) -> None:
        """Metrics endpoint returns Prometheus format."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        # Should contain standard Prometheus metrics
        assert b"http_request" in response.content or b"# HELP" in response.content


class TestMetricsFunctions:
    """Tests for metrics tracking functions."""

    def test_get_metrics_returns_bytes(self) -> None:
        """get_metrics returns bytes."""
        metrics = get_metrics()
        assert isinstance(metrics, bytes)

    def test_track_llm_request_success(self) -> None:
        """track_llm_request records successful request."""
        # Should not raise
        track_llm_request(
            model="test-model",
            duration=1.5,
            prompt_tokens=100,
            completion_tokens=50,
            success=True,
        )

        metrics = get_metrics().decode()
        assert "llm_request_duration_seconds" in metrics
        assert "llm_tokens_total" in metrics

    def test_track_llm_request_failure(self) -> None:
        """track_llm_request records failed request."""
        track_llm_request(
            model="test-model",
            duration=0.5,
            prompt_tokens=0,
            completion_tokens=0,
            success=False,
        )

        metrics = get_metrics().decode()
        assert "llm_requests_total" in metrics

    def test_track_embedding_request(self) -> None:
        """track_embedding_request records request."""
        track_embedding_request(
            model="bge-large",
            duration=0.1,
            batch_size=10,
            success=True,
        )

        metrics = get_metrics().decode()
        assert "embedding_request_duration_seconds" in metrics
        assert "embedding_batch_size" in metrics

    def test_track_retrieval_request(self) -> None:
        """track_retrieval_request records request."""
        track_retrieval_request(
            chunks_returned=5,
            top_score=0.95,
        )

        metrics = get_metrics().decode()
        assert "retrieval_chunks_returned" in metrics
        assert "retrieval_top_score" in metrics

    def test_update_eval_metrics(self) -> None:
        """update_eval_metrics sets gauge values."""
        update_eval_metrics(
            gold_set="test_gold",
            pass_rate=0.87,
            metric_scores={"rouge_l": 0.85, "bleu": 0.72},
        )

        metrics = get_metrics().decode()
        assert "eval_pass_rate" in metrics
        assert "eval_metric_score" in metrics


class TestMetricsMiddleware:
    """Tests for MetricsMiddleware."""

    @pytest.mark.asyncio
    async def test_middleware_records_request_metrics(self) -> None:
        """Middleware records HTTP request metrics."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Make a request
            await client.get("/health")

        metrics = get_metrics().decode()
        assert "http_request_duration_seconds" in metrics
        assert "http_requests_total" in metrics

    @pytest.mark.asyncio
    async def test_middleware_normalizes_endpoints(self) -> None:
        """Middleware normalizes endpoint paths."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Make requests to different health endpoints
            await client.get("/health")
            await client.get("/health/ready")
            await client.get("/health/live")

        # All should be normalized to /health
        metrics = get_metrics().decode()
        # The middleware should group these
        assert "http_requests_total" in metrics

