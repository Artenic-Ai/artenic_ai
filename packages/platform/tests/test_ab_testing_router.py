"""Tests for artenic_ai_platform.ab_testing.router — /api/v1/ab-tests/*."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from artenic_ai_platform.ab_testing.router import router
from artenic_ai_platform.db.models import Base

# ======================================================================
# Helpers
# ======================================================================


async def _make_app() -> tuple[FastAPI, Any]:
    """Build a minimal FastAPI app with the AB-testing router and in-memory DB."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)

    app = FastAPI()
    app.state.session_factory = factory
    app.include_router(router)
    return app, engine


def _create_payload(**overrides: Any) -> dict[str, Any]:
    """Default valid POST body for creating an A/B test."""
    payload: dict[str, Any] = {
        "name": "exp-1",
        "service": "chat",
        "variants": {
            "control": {"model_id": "model-a", "traffic_pct": 50},
            "treatment": {"model_id": "model-b", "traffic_pct": 50},
        },
        "primary_metric": "accuracy",
    }
    payload.update(overrides)
    return payload


async def _post_test(client: AsyncClient, **overrides: Any) -> dict[str, Any]:
    """Create an A/B test via the API and return the response JSON."""
    resp = await client.post("/api/v1/ab-tests", json=_create_payload(**overrides))
    assert resp.status_code == 201, resp.text
    return resp.json()


# ======================================================================
# POST /api/v1/ab-tests
# ======================================================================


class TestCreateTestEndpoint:
    async def test_create_test_endpoint(self) -> None:
        app, engine = await _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            data = await _post_test(client)
            assert "test_id" in data
            assert len(data["test_id"]) == 36  # UUID4

        await engine.dispose()


# ======================================================================
# GET /api/v1/ab-tests
# ======================================================================


class TestListTestsEndpoint:
    async def test_list_tests_endpoint(self) -> None:
        app, engine = await _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Create two tests.
            await _post_test(client, name="a")
            await _post_test(client, name="b")

            resp = await client.get("/api/v1/ab-tests")
            assert resp.status_code == 200
            tests = resp.json()
            assert isinstance(tests, list)
            assert len(tests) == 2

        await engine.dispose()


# ======================================================================
# GET /api/v1/ab-tests/{id}
# ======================================================================


class TestGetTestEndpoint:
    async def test_get_test_endpoint(self) -> None:
        app, engine = await _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            created = await _post_test(client, name="detail")
            test_id = created["test_id"]

            resp = await client.get(f"/api/v1/ab-tests/{test_id}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["id"] == test_id
            assert data["name"] == "detail"
            assert data["status"] == "running"

        await engine.dispose()


# ======================================================================
# GET /api/v1/ab-tests/{id}/results
# ======================================================================


class TestGetResultsEndpoint:
    async def test_get_results_endpoint(self) -> None:
        app, engine = await _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            created = await _post_test(client)
            test_id = created["test_id"]

            resp = await client.get(f"/api/v1/ab-tests/{test_id}/results")
            assert resp.status_code == 200
            data = resp.json()
            assert data["test_id"] == test_id
            assert "variants" in data

        await engine.dispose()


# ======================================================================
# POST /api/v1/ab-tests/{id}/conclude
# ======================================================================


class TestConcludeEndpoint:
    async def test_conclude_endpoint(self) -> None:
        app, engine = await _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            created = await _post_test(client)
            test_id = created["test_id"]

            resp = await client.post(
                f"/api/v1/ab-tests/{test_id}/conclude",
                json={"winner": "control", "reason": "better accuracy"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "concluded"
            assert data["winner"] == "control"

        await engine.dispose()


# ======================================================================
# POST /api/v1/ab-tests/{id}/pause
# ======================================================================


class TestPauseEndpoint:
    async def test_pause_endpoint(self) -> None:
        app, engine = await _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            created = await _post_test(client)
            test_id = created["test_id"]

            resp = await client.post(f"/api/v1/ab-tests/{test_id}/pause")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "paused"

        await engine.dispose()


# ======================================================================
# POST /api/v1/ab-tests/{id}/resume
# ======================================================================


class TestResumeEndpoint:
    async def test_resume_endpoint(self) -> None:
        app, engine = await _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            created = await _post_test(client)
            test_id = created["test_id"]

            # Pause first, then resume.
            await client.post(f"/api/v1/ab-tests/{test_id}/pause")
            resp = await client.post(f"/api/v1/ab-tests/{test_id}/resume")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "running"

        await engine.dispose()
