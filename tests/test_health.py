"""Integration tests for health check endpoints."""

from httpx import AsyncClient

from src import __version__


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    async def test_health_returns_200(self, client: AsyncClient) -> None:
        """Health endpoint returns 200 OK."""
        response = await client.get("/health")
        assert response.status_code == 200

    async def test_health_returns_status(self, client: AsyncClient) -> None:
        """Health endpoint returns healthy status."""
        response = await client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    async def test_health_returns_version(self, client: AsyncClient) -> None:
        """Health endpoint returns application version."""
        response = await client.get("/health")
        data = response.json()
        assert data["version"] == __version__

    async def test_health_returns_timestamp(self, client: AsyncClient) -> None:
        """Health endpoint returns ISO timestamp."""
        response = await client.get("/health")
        data = response.json()
        assert "timestamp" in data
        # Verify ISO format (contains T separator)
        assert "T" in data["timestamp"]


class TestReadinessEndpoint:
    """Tests for /health/ready endpoint."""

    async def test_readiness_returns_200(self, client: AsyncClient) -> None:
        """Readiness endpoint returns 200 OK."""
        response = await client.get("/health/ready")
        assert response.status_code == 200

    async def test_readiness_returns_status(self, client: AsyncClient) -> None:
        """Readiness endpoint returns ready status."""
        response = await client.get("/health/ready")
        data = response.json()
        assert data["status"] == "ready"

    async def test_readiness_returns_checks(self, client: AsyncClient) -> None:
        """Readiness endpoint returns component checks."""
        response = await client.get("/health/ready")
        data = response.json()
        assert "checks" in data
        assert data["checks"]["config"] == "ok"

    async def test_readiness_returns_timestamp(self, client: AsyncClient) -> None:
        """Readiness endpoint returns timestamp."""
        response = await client.get("/health/ready")
        data = response.json()
        assert "timestamp" in data


class TestLivenessEndpoint:
    """Tests for /health/live endpoint."""

    async def test_liveness_returns_200(self, client: AsyncClient) -> None:
        """Liveness endpoint returns 200 OK."""
        response = await client.get("/health/live")
        assert response.status_code == 200

    async def test_liveness_returns_alive(self, client: AsyncClient) -> None:
        """Liveness endpoint returns alive status."""
        response = await client.get("/health/live")
        data = response.json()
        assert data["status"] == "alive"

