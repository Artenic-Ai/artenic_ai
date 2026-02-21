"""Integration tests for the full Artenic AI Platform application."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from artenic_ai_platform.app import _lifespan, create_app
from artenic_ai_platform.settings import PlatformSettings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


# ======================================================================
# Helpers
# ======================================================================


def _test_settings() -> PlatformSettings:
    """Return minimal settings for in-memory SQLite testing."""
    return PlatformSettings(
        database_url="sqlite+aiosqlite://",
        api_key="",
        secret_key="test-secret-key-for-testing",
        otel_enabled=False,
    )


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
async def app_with_lifespan() -> AsyncGenerator[FastAPI, None]:
    """Create a FastAPI app and manually enter/exit its lifespan."""
    settings = _test_settings()
    app = create_app(settings)

    async with _lifespan(app):
        yield app


@pytest.fixture
async def client(app_with_lifespan: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client wired to the fully-initialised test app."""
    transport = ASGITransport(app=app_with_lifespan)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ======================================================================
# App creation tests (no lifespan needed)
# ======================================================================


class TestCreateApp:
    """Verify create_app returns a properly configured FastAPI instance."""

    def test_create_app_returns_fastapi(self) -> None:
        app = create_app(_test_settings())
        assert isinstance(app, FastAPI)

    def test_create_app_has_routes(self) -> None:
        app = create_app(_test_settings())
        paths = {getattr(r, "path", None) for r in app.routes}

        expected = [
            "/health",
            "/api/v1/models",
            "/api/v1/training/dispatch",
            "/api/v1/budgets",
            "/api/v1/services/{service}/predict",
            "/api/v1/ensembles",
            "/api/v1/ab-tests",
            "/ws",
        ]
        for path in expected:
            assert path in paths, f"Missing route: {path}"

    def test_create_app_has_middleware(self) -> None:
        app = create_app(_test_settings())
        # Accessing any endpoint triggers middleware stack build;
        # at minimum the app object must have user_middleware entries.
        assert app.middleware_stack is not None or len(app.user_middleware) > 0

    def test_create_app_custom_settings(self) -> None:
        settings = _test_settings()
        app = create_app(settings)
        assert app.state.settings is settings


# ======================================================================
# Smoke tests with httpx (lifespan triggered via fixture)
# ======================================================================


class TestHealthEndpoints:
    """Health-check endpoints should respond without any DB seeding."""

    async def test_health_endpoint(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"

    async def test_health_ready_endpoint(self, client: AsyncClient) -> None:
        resp = await client.get("/health/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready"


class TestInferenceEndpoint:
    """Inference service smoke test."""

    async def test_predict_endpoint_no_model_returns_404(self, client: AsyncClient) -> None:
        """Without any loaded model plugin, predict returns 404."""
        resp = await client.post(
            "/api/v1/services/test/predict",
            json={"data": {"x": 1}},
        )
        assert resp.status_code == 404


class TestRegistryEndpoint:
    """Model registry smoke test."""

    async def test_list_models_endpoint_smoke(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/models")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestEnsembleEndpoint:
    """Ensemble management smoke test."""

    async def test_list_ensembles_endpoint_smoke(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/ensembles")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestABTestEndpoint:
    """A/B testing smoke test."""

    async def test_list_ab_tests_endpoint_smoke(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/ab-tests")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestBudgetEndpoint:
    """Budget governance smoke test."""

    async def test_list_budgets_endpoint_smoke(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/budgets")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


# ======================================================================
# Budget factory callable
# ======================================================================


class TestBudgetFactory:
    """app.state.budget_manager_factory is callable and creates a BudgetManager."""

    async def test_budget_factory_callable(self, app_with_lifespan: FastAPI) -> None:
        from unittest.mock import MagicMock

        factory = app_with_lifespan.state.budget_manager_factory
        assert callable(factory)
        # Create a mock session to call the factory
        result = factory(MagicMock())
        assert result is not None


# ======================================================================
# Health monitor auto-start
# ======================================================================


class TestHealthMonitorAutoStart:
    """Cover app.py line 108: health_monitor.start() when health.enabled=True."""

    async def test_health_monitor_starts_when_enabled(self) -> None:
        settings = PlatformSettings(
            database_url="sqlite+aiosqlite://",
            api_key="",
            secret_key="test-secret-key-for-testing",
            otel_enabled=False,
            health={"enabled": True, "check_interval_seconds": 3600},
        )
        app = create_app(settings)

        async with _lifespan(app):
            hm = app.state.health_monitor
            assert hm._running is True
            assert hm._task is not None

        # After lifespan exits, the monitor should be stopped
        assert hm._running is False
