"""Tests for artenic_ai_platform.registry.router — 100% coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from httpx import AsyncClient


# ======================================================================
# Helpers
# ======================================================================


def _register_body(
    name: str = "sentiment",
    version: str = "1.0",
    model_type: str = "classifier",
) -> dict:
    return {
        "name": name,
        "version": version,
        "model_type": model_type,
        "framework": "pytorch",
        "description": "test model",
        "tags": {"env": "test"},
        "input_features": [{"name": "text", "type": "string"}],
        "output_schema": {"label": "string"},
    }


# ======================================================================
# POST /api/v1/models
# ======================================================================


class TestRegisterModel:
    async def test_register_returns_201(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v1/models", json=_register_body())
        assert resp.status_code == 201
        data = resp.json()
        assert data["model_id"] == "sentiment_v1.0"

    async def test_register_validation_error(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v1/models", json={})
        assert resp.status_code == 422


# ======================================================================
# GET /api/v1/models
# ======================================================================


class TestListModels:
    async def test_list_empty(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/models")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_list_after_register(self, client: AsyncClient) -> None:
        await client.post("/api/v1/models", json=_register_body())
        resp = await client.get("/api/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "sentiment"


# ======================================================================
# GET /api/v1/models/{model_id}
# ======================================================================


class TestGetModel:
    async def test_get_existing(self, client: AsyncClient) -> None:
        await client.post("/api/v1/models", json=_register_body())
        resp = await client.get("/api/v1/models/sentiment_v1.0")
        assert resp.status_code == 200
        assert resp.json()["model_id"] == "sentiment_v1.0"

    async def test_get_not_found(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/models/not-there")
        assert resp.status_code == 404


# ======================================================================
# POST /api/v1/models/{model_id}/promote
# ======================================================================


class TestPromoteModel:
    async def test_promote_returns_204(self, client: AsyncClient) -> None:
        await client.post("/api/v1/models", json=_register_body())
        resp = await client.post(
            "/api/v1/models/sentiment_v1.0/promote",
            json={"version": "1.0"},
        )
        assert resp.status_code == 204

    async def test_promote_not_found(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/models/not-there/promote",
            json={"version": "1.0"},
        )
        assert resp.status_code == 404

    async def test_promote_validation_error(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/models/sentinel_v1.0/promote",
            json={},
        )
        assert resp.status_code == 422


# ======================================================================
# POST /api/v1/models/{model_id}/retire
# ======================================================================


class TestRetireModel:
    async def test_retire_returns_204(self, client: AsyncClient) -> None:
        await client.post("/api/v1/models", json=_register_body())
        resp = await client.post("/api/v1/models/sentiment_v1.0/retire")
        assert resp.status_code == 204

    async def test_retire_not_found(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v1/models/not-there/retire")
        assert resp.status_code == 404
