"""Tests for artenic_ai_platform.health.router — 100% coverage."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from httpx import ASGITransport, AsyncClient

from artenic_ai_platform.app import create_app
from artenic_ai_platform.db.engine import create_async_engine, create_tables
from artenic_ai_platform.settings import PlatformSettings


def _make_settings() -> PlatformSettings:
    return PlatformSettings(
        database_url="sqlite+aiosqlite://",
        api_key="",
        secret_key="test",
        otel_enabled=False,
    )


async def _setup_app_state(app, settings: PlatformSettings) -> None:  # type: ignore[no-untyped-def]
    """Manually initialize app.state the same way the lifespan does."""
    engine = create_async_engine(settings.database_url)
    await create_tables(engine)
    app.state.engine = engine


# ======================================================================
# /health — liveness
# ======================================================================


class TestHealthLiveness:
    async def test_returns_healthy(self) -> None:
        settings = _make_settings()
        app = create_app(settings)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "healthy"}


# ======================================================================
# /health/ready — readiness
# ======================================================================


class TestHealthReadiness:
    async def test_ready_when_db_connected(self) -> None:
        settings = _make_settings()
        app = create_app(settings)
        await _setup_app_state(app, settings)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/ready")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["database"] == "connected"
        await app.state.engine.dispose()

    async def test_not_ready_when_db_fails(self) -> None:
        settings = _make_settings()
        app = create_app(settings)

        mock_engine = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=Exception("DB down"))
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_engine.connect = MagicMock(return_value=mock_cm)
        app.state.engine = mock_engine

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/ready")

        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "not_ready"
        assert data["database"] == "disconnected"


# ======================================================================
# /health/detailed — detailed health
# ======================================================================


class TestHealthDetailed:
    async def test_healthy_when_db_connected(self) -> None:
        settings = _make_settings()
        app = create_app(settings)
        await _setup_app_state(app, settings)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/detailed")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["components"]["database"] == "connected"
        assert "version" in data["components"]
        await app.state.engine.dispose()

    async def test_degraded_when_db_fails(self) -> None:
        settings = _make_settings()
        app = create_app(settings)

        mock_engine = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=Exception("DB error"))
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_engine.connect = MagicMock(return_value=mock_cm)
        app.state.engine = mock_engine

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/detailed")

        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["components"]["database"] == "disconnected"
