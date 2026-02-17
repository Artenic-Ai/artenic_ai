"""Tests for artenic_ai_platform.ensemble.router — /api/v1/ensembles/*."""

from __future__ import annotations

from typing import Any

from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from artenic_ai_platform.db.models import Base

# ======================================================================
# Helper — minimal FastAPI app with ensemble router
# ======================================================================


async def _make_client() -> tuple[AsyncClient, Any]:
    """Return (AsyncClient, engine) wired to a fresh in-memory DB."""
    from fastapi import FastAPI

    from artenic_ai_platform.ensemble.router import router

    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    app = FastAPI()
    app.state.session_factory = factory
    app.include_router(router)

    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://test")
    return client, engine


def _ensemble_payload(**overrides: Any) -> dict[str, Any]:
    """Standard ensemble creation payload."""
    payload: dict[str, Any] = {
        "name": "weighted",
        "service": "sentiment",
        "strategy": "weighted",
        "model_ids": ["model-a", "model-b"],
        "description": "Test ensemble",
    }
    payload.update(overrides)
    return payload


async def _create_ensemble(client: AsyncClient, **overrides: Any) -> str:
    """POST an ensemble and return the ensemble_id."""
    resp = await client.post("/api/v1/ensembles", json=_ensemble_payload(**overrides))
    assert resp.status_code == 200, resp.text
    return resp.json()["ensemble_id"]


# ======================================================================
# POST /api/v1/ensembles
# ======================================================================


class TestCreateEnsembleEndpoint:
    async def test_create_ensemble_endpoint(self) -> None:
        """POST /api/v1/ensembles returns 200 with ensemble_id."""
        client, engine = await _make_client()
        try:
            resp = await client.post("/api/v1/ensembles", json=_ensemble_payload())
            assert resp.status_code == 200
            body = resp.json()
            assert "ensemble_id" in body
            assert body["ensemble_id"] == "sentiment_weighted_v1"
        finally:
            await client.aclose()
            await engine.dispose()


# ======================================================================
# GET /api/v1/ensembles
# ======================================================================


class TestListEnsemblesEndpoint:
    async def test_list_ensembles_endpoint(self) -> None:
        """GET /api/v1/ensembles returns a list of ensembles."""
        client, engine = await _make_client()
        try:
            await _create_ensemble(client, name="a", service="svc1")
            await _create_ensemble(client, name="b", service="svc2")

            resp = await client.get("/api/v1/ensembles")
            assert resp.status_code == 200
            body = resp.json()
            assert isinstance(body, list)
            assert len(body) == 2
        finally:
            await client.aclose()
            await engine.dispose()


# ======================================================================
# GET /api/v1/ensembles/{id}
# ======================================================================


class TestGetEnsembleEndpoint:
    async def test_get_ensemble_endpoint(self) -> None:
        """GET /api/v1/ensembles/{id} returns the ensemble."""
        client, engine = await _make_client()
        try:
            eid = await _create_ensemble(client)

            resp = await client.get(f"/api/v1/ensembles/{eid}")
            assert resp.status_code == 200
            body = resp.json()
            assert body["id"] == eid
            assert body["name"] == "weighted"
            assert body["service"] == "sentiment"
        finally:
            await client.aclose()
            await engine.dispose()


# ======================================================================
# PUT /api/v1/ensembles/{id}
# ======================================================================


class TestUpdateEnsembleEndpoint:
    async def test_update_ensemble_endpoint(self) -> None:
        """PUT /api/v1/ensembles/{id} returns the updated ensemble."""
        client, engine = await _make_client()
        try:
            eid = await _create_ensemble(client)

            resp = await client.put(
                f"/api/v1/ensembles/{eid}",
                json={
                    "model_ids": ["model-a", "model-b", "model-c"],
                    "change_reason": "Added model-c",
                },
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["version"] == 2
            assert body["model_ids"] == ["model-a", "model-b", "model-c"]
        finally:
            await client.aclose()
            await engine.dispose()


# ======================================================================
# POST /api/v1/ensembles/{id}/train
# ======================================================================


class TestDispatchTrainingEndpoint:
    async def test_dispatch_training_endpoint(self) -> None:
        """POST /api/v1/ensembles/{id}/train returns a job_id."""
        client, engine = await _make_client()
        try:
            eid = await _create_ensemble(client)

            resp = await client.post(
                f"/api/v1/ensembles/{eid}/train",
                json={"provider": "mock"},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert "job_id" in body
            assert len(body["job_id"]) == 36  # UUID
        finally:
            await client.aclose()
            await engine.dispose()


# ======================================================================
# GET /api/v1/ensembles/{id}/versions
# ======================================================================


class TestGetVersionHistoryEndpoint:
    async def test_get_version_history_endpoint(self) -> None:
        """GET /api/v1/ensembles/{id}/versions returns version records."""
        client, engine = await _make_client()
        try:
            eid = await _create_ensemble(client)

            # Update once to create a second version
            await client.put(
                f"/api/v1/ensembles/{eid}",
                json={"model_ids": ["model-x"], "change_reason": "v2"},
            )

            resp = await client.get(f"/api/v1/ensembles/{eid}/versions")
            assert resp.status_code == 200
            body = resp.json()
            assert isinstance(body, list)
            assert len(body) == 2
            # Newest first
            assert body[0]["version"] == 2
            assert body[1]["version"] == 1
        finally:
            await client.aclose()
            await engine.dispose()


# ======================================================================
# GET /api/v1/ensembles/{id}/jobs/{job_id}
# ======================================================================


class TestGetEnsembleJobStatusEndpoint:
    """GET /api/v1/ensembles/{id}/jobs/{job_id} returns the job status."""

    async def test_get_job_status_endpoint(self) -> None:
        client, engine = await _make_client()
        try:
            eid = await _create_ensemble(client)

            # Dispatch training to create a job
            resp2 = await client.post(
                f"/api/v1/ensembles/{eid}/train",
                json={"provider": "mock"},
            )
            assert resp2.status_code == 200
            jid = resp2.json()["job_id"]

            # Get job status
            resp3 = await client.get(f"/api/v1/ensembles/{eid}/jobs/{jid}")
            assert resp3.status_code == 200
            body = resp3.json()
            assert body["id"] == jid
            assert body["ensemble_id"] == eid
            assert body["status"] == "pending"
        finally:
            await client.aclose()
            await engine.dispose()
