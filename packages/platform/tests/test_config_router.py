"""Tests for artenic_ai_platform.routes.config — 100% coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from httpx import AsyncClient


# ======================================================================
# GET /api/v1/settings/schema
# ======================================================================


class TestGetSchema:
    async def test_returns_full_schema(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/settings/schema")
        assert resp.status_code == 200
        data = resp.json()
        assert "global" in data
        assert len(data["global"]) == 12


# ======================================================================
# GET /api/v1/settings/schema/{scope}
# ======================================================================


class TestGetScopeSchema:
    async def test_global_scope(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/settings/schema/global")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 12

    async def test_unknown_scope(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/settings/schema/nonexistent")
        assert resp.status_code == 200
        assert resp.json() == []


# ======================================================================
# GET /api/v1/settings/schema/{scope}/{section}
# ======================================================================


class TestGetSectionSchema:
    async def test_existing_section(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/settings/schema/global/core")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "core"

    async def test_unknown_section(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/settings/schema/global/nonexistent")
        assert resp.status_code == 200
        assert resp.json() is None


# ======================================================================
# GET /api/v1/settings/{scope}
# ======================================================================


class TestGetAllSettings:
    async def test_empty_scope(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/settings/global")
        assert resp.status_code == 200
        assert resp.json() == {}

    async def test_with_data(self, client: AsyncClient) -> None:
        # Create some settings first
        await client.put(
            "/api/v1/settings/global/core",
            json={"host": "localhost"},
        )
        resp = await client.get("/api/v1/settings/global")
        assert resp.status_code == 200
        data = resp.json()
        assert "core" in data
        assert data["core"]["host"] == "localhost"


# ======================================================================
# GET /api/v1/settings/{scope}/{section}
# ======================================================================


class TestGetSectionSettings:
    async def test_empty_section(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/settings/global/core")
        assert resp.status_code == 200
        assert resp.json() == {}

    async def test_with_data(self, client: AsyncClient) -> None:
        await client.put(
            "/api/v1/settings/global/mlflow",
            json={"mlflow_tracking_uri": "http://mlflow:5000"},
        )
        resp = await client.get("/api/v1/settings/global/mlflow")
        assert resp.status_code == 200
        data = resp.json()
        assert data["mlflow_tracking_uri"] == "http://mlflow:5000"


# ======================================================================
# PUT /api/v1/settings/{scope}/{section}
# ======================================================================


class TestUpdateSectionSettings:
    async def test_create_settings(self, client: AsyncClient) -> None:
        resp = await client.put(
            "/api/v1/settings/global/core",
            json={"host": "0.0.0.0", "port": "9000"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["host"] == "0.0.0.0"
        assert data["port"] == "9000"

    async def test_update_existing(self, client: AsyncClient) -> None:
        await client.put(
            "/api/v1/settings/global/core",
            json={"host": "old"},
        )
        resp = await client.put(
            "/api/v1/settings/global/core",
            json={"host": "new"},
        )
        assert resp.status_code == 200
        assert resp.json()["host"] == "new"


# ======================================================================
# GET /api/v1/settings/audit/log
# ======================================================================


class TestGetAuditLog:
    async def test_empty_log(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/settings/audit/log")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_log_after_update(self, client: AsyncClient) -> None:
        await client.put(
            "/api/v1/settings/global/core",
            json={"host": "test"},
        )
        resp = await client.get("/api/v1/settings/audit/log")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1

    async def test_pagination_params(self, client: AsyncClient) -> None:
        resp = await client.get(
            "/api/v1/settings/audit/log",
            params={"limit": 10, "offset": 0},
        )
        assert resp.status_code == 200
