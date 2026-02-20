"""Tests for artenic_ai_platform.entities.features.router — REST API endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from httpx import ASGITransport, AsyncClient

from artenic_ai_platform.app import _lifespan, create_app
from artenic_ai_platform.settings import PlatformSettings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from fastapi import FastAPI


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
async def app_with_lifespan(tmp_path: Path) -> AsyncGenerator[FastAPI, None]:
    settings = PlatformSettings(
        database_url="sqlite+aiosqlite://",
        api_key="",
        secret_key="test-secret",
        otel_enabled=False,
        dataset={"storage": {"local_dir": str(tmp_path / "datasets")}},
    )
    app = create_app(settings)
    async with _lifespan(app):
        yield app


@pytest.fixture
async def client(app_with_lifespan: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app_with_lifespan)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ======================================================================
# Helpers
# ======================================================================

BASE = "/api/v1/features"


async def _create_feature(
    client: AsyncClient,
    feat_id: str = "feat_test_v1",
    **kwargs: Any,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": feat_id,
        "name": kwargs.get("name", "my-feature"),
        "metadata": kwargs.get("metadata", {"type": "numerical"}),
    }
    resp = await client.post(BASE, json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()


# ======================================================================
# POST /api/v1/features
# ======================================================================


class TestCreateFeature:
    async def test_create_returns_201(self, client: AsyncClient) -> None:
        data = await _create_feature(client, feat_id="feat_cr_v1")
        assert data["id"] == "feat_cr_v1"
        assert data["name"] == "my-feature"
        assert data["version"] == 1
        assert data["metadata"] == {"type": "numerical"}

    async def test_create_auto_version(self, client: AsyncClient) -> None:
        await _create_feature(client, feat_id="feat_av_v1", name="auto-v")
        data = await _create_feature(client, feat_id="feat_av_v2", name="auto-v")
        assert data["version"] == 2

    async def test_create_validation_error(self, client: AsyncClient) -> None:
        resp = await client.post(BASE, json={})
        assert resp.status_code == 422


# ======================================================================
# GET /api/v1/features
# ======================================================================


class TestListFeatures:
    async def test_list_empty(self, client: AsyncClient) -> None:
        resp = await client.get(BASE)
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_list_after_create(self, client: AsyncClient) -> None:
        await _create_feature(client, feat_id="feat_l1_v1", name="f1")
        await _create_feature(client, feat_id="feat_l2_v1", name="f2")
        resp = await client.get(BASE)
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    async def test_list_with_name_filter(self, client: AsyncClient) -> None:
        await _create_feature(client, feat_id="feat_nf1_v1", name="specific")
        await _create_feature(client, feat_id="feat_nf2_v1", name="other")
        resp = await client.get(BASE, params={"name": "specific"})
        assert resp.status_code == 200
        assert len(resp.json()) == 1


# ======================================================================
# GET /api/v1/features/{id}
# ======================================================================


class TestGetFeature:
    async def test_get_returns_details(self, client: AsyncClient) -> None:
        await _create_feature(client, feat_id="feat_get_v1", name="detail")
        resp = await client.get(f"{BASE}/feat_get_v1")
        assert resp.status_code == 200
        assert resp.json()["name"] == "detail"

    async def test_get_not_found(self, client: AsyncClient) -> None:
        resp = await client.get(f"{BASE}/nonexistent")
        assert resp.status_code == 404


# ======================================================================
# PATCH /api/v1/features/{id}
# ======================================================================


class TestUpdateFeature:
    async def test_update_metadata(self, client: AsyncClient) -> None:
        await _create_feature(client, feat_id="feat_upd_v1")
        resp = await client.patch(
            f"{BASE}/feat_upd_v1",
            json={"metadata": {"type": "categorical", "values": ["a", "b"]}},
        )
        assert resp.status_code == 200
        assert resp.json()["metadata"]["type"] == "categorical"

    async def test_update_not_found(self, client: AsyncClient) -> None:
        resp = await client.patch(f"{BASE}/nonexistent", json={"metadata": {}})
        assert resp.status_code == 404


# ======================================================================
# DELETE /api/v1/features/{id}
# ======================================================================


class TestDeleteFeature:
    async def test_delete_returns_204(self, client: AsyncClient) -> None:
        await _create_feature(client, feat_id="feat_del_v1")
        resp = await client.delete(f"{BASE}/feat_del_v1")
        assert resp.status_code == 204

    async def test_delete_not_found(self, client: AsyncClient) -> None:
        resp = await client.delete(f"{BASE}/nonexistent")
        assert resp.status_code == 404


# ======================================================================
# Additional coverage tests
# ======================================================================


class TestFeatureCoverageEdgeCases:
    async def test_create_with_explicit_version(self, client: AsyncClient) -> None:
        body: dict[str, Any] = {
            "id": "feat_ev_v5",
            "name": "expl-v",
            "metadata": {},
            "version": 5,
        }
        resp = await client.post(BASE, json=body)
        assert resp.status_code == 201
        assert resp.json()["version"] == 5
