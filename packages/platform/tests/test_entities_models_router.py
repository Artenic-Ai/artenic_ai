"""Tests for artenic_ai_platform.entities.models.router — REST API endpoints."""

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
        dataset={"storage": {"local_dir": str(tmp_path / "models")}},
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

BASE = "/api/v1/models"


def _create_body(
    model_id: str = "test_model_v1",
    name: str = "my-model",
    framework: str = "pytorch",
    **kwargs: Any,
) -> dict[str, Any]:
    return {
        "id": model_id,
        "name": name,
        "framework": framework,
        "description": kwargs.get("description", "test model"),
        "metadata": kwargs.get("metadata", {}),
        "metrics": kwargs.get("metrics", {}),
    }


async def _create_model(
    client: AsyncClient,
    model_id: str = "test_model_v1",
    **kwargs: Any,
) -> dict[str, Any]:
    resp = await client.post(BASE, json=_create_body(model_id=model_id, **kwargs))
    assert resp.status_code == 201, resp.text
    return resp.json()


# ======================================================================
# POST /api/v1/models
# ======================================================================


class TestCreateModel:
    async def test_create_returns_201(self, client: AsyncClient) -> None:
        data = await _create_model(client, model_id="mdl_cr_v1")
        assert data["id"] == "mdl_cr_v1"
        assert data["name"] == "my-model"
        assert data["framework"] == "pytorch"
        assert data["stage"] == "draft"
        assert data["version"] == 1

    async def test_create_auto_version(self, client: AsyncClient) -> None:
        await _create_model(client, model_id="mdl_av_v1", name="auto-v")
        data = await _create_model(client, model_id="mdl_av_v2", name="auto-v")
        assert data["version"] == 2

    async def test_create_with_metrics(self, client: AsyncClient) -> None:
        data = await _create_model(
            client,
            model_id="mdl_met_v1",
            metrics={"accuracy": 0.95, "f1": 0.92},
        )
        assert data["metrics"] == {"accuracy": 0.95, "f1": 0.92}

    async def test_create_validation_error(self, client: AsyncClient) -> None:
        resp = await client.post(BASE, json={})
        assert resp.status_code == 422


# ======================================================================
# GET /api/v1/models
# ======================================================================


class TestListModels:
    async def test_list_empty(self, client: AsyncClient) -> None:
        resp = await client.get(BASE)
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_list_after_create(self, client: AsyncClient) -> None:
        await _create_model(client, model_id="mdl_l1_v1", name="m1")
        await _create_model(client, model_id="mdl_l2_v1", name="m2")
        resp = await client.get(BASE)
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    async def test_list_with_stage_filter(self, client: AsyncClient) -> None:
        await _create_model(client, model_id="mdl_sf_v1")
        resp = await client.get(BASE, params={"stage": "draft"})
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    async def test_list_with_framework_filter(self, client: AsyncClient) -> None:
        await _create_model(client, model_id="mdl_ff1_v1", framework="pytorch")
        await _create_model(client, model_id="mdl_ff2_v1", name="m2", framework="tensorflow")
        resp = await client.get(BASE, params={"framework": "tensorflow"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["framework"] == "tensorflow"


# ======================================================================
# GET /api/v1/models/{id}
# ======================================================================


class TestGetModel:
    async def test_get_returns_details(self, client: AsyncClient) -> None:
        await _create_model(client, model_id="mdl_get_v1", name="detail")
        resp = await client.get(f"{BASE}/mdl_get_v1")
        assert resp.status_code == 200
        assert resp.json()["name"] == "detail"

    async def test_get_not_found(self, client: AsyncClient) -> None:
        resp = await client.get(f"{BASE}/nonexistent")
        assert resp.status_code == 404


# ======================================================================
# PATCH /api/v1/models/{id}
# ======================================================================


class TestUpdateModel:
    async def test_update_description(self, client: AsyncClient) -> None:
        await _create_model(client, model_id="mdl_upd_v1")
        resp = await client.patch(
            f"{BASE}/mdl_upd_v1",
            json={"description": "updated"},
        )
        assert resp.status_code == 200
        assert resp.json()["description"] == "updated"

    async def test_update_metrics(self, client: AsyncClient) -> None:
        await _create_model(client, model_id="mdl_um_v1")
        resp = await client.patch(
            f"{BASE}/mdl_um_v1",
            json={"metrics": {"loss": 0.01}},
        )
        assert resp.status_code == 200
        assert resp.json()["metrics"] == {"loss": 0.01}

    async def test_update_not_found(self, client: AsyncClient) -> None:
        resp = await client.patch(f"{BASE}/nonexistent", json={"description": "x"})
        assert resp.status_code == 404


# ======================================================================
# DELETE /api/v1/models/{id}
# ======================================================================


class TestDeleteModel:
    async def test_delete_returns_204(self, client: AsyncClient) -> None:
        await _create_model(client, model_id="mdl_del_v1")
        resp = await client.delete(f"{BASE}/mdl_del_v1")
        assert resp.status_code == 204

    async def test_delete_not_found(self, client: AsyncClient) -> None:
        resp = await client.delete(f"{BASE}/nonexistent")
        assert resp.status_code == 404


# ======================================================================
# PATCH /api/v1/models/{id}/stage
# ======================================================================


class TestChangeStage:
    async def test_promote_draft_to_staging(self, client: AsyncClient) -> None:
        await _create_model(client, model_id="mdl_st_v1")
        resp = await client.patch(
            f"{BASE}/mdl_st_v1/stage",
            json={"stage": "staging"},
        )
        assert resp.status_code == 200
        assert resp.json()["stage"] == "staging"

    async def test_full_lifecycle(self, client: AsyncClient) -> None:
        await _create_model(client, model_id="mdl_lc_v1")
        # draft -> staging
        resp = await client.patch(f"{BASE}/mdl_lc_v1/stage", json={"stage": "staging"})
        assert resp.status_code == 200
        # staging -> production
        resp = await client.patch(f"{BASE}/mdl_lc_v1/stage", json={"stage": "production"})
        assert resp.status_code == 200
        # production -> retired
        resp = await client.patch(f"{BASE}/mdl_lc_v1/stage", json={"stage": "retired"})
        assert resp.status_code == 200

    async def test_invalid_transition(self, client: AsyncClient) -> None:
        await _create_model(client, model_id="mdl_inv_v1")
        # draft -> production (not allowed, must go through staging)
        resp = await client.patch(
            f"{BASE}/mdl_inv_v1/stage",
            json={"stage": "production"},
        )
        assert resp.status_code == 400


# ======================================================================
# PUT /api/v1/models/{id}/artifact
# ======================================================================


class TestUploadArtifact:
    async def test_upload_artifact(self, client: AsyncClient) -> None:
        await _create_model(client, model_id="mdl_art_v1")
        resp = await client.put(
            f"{BASE}/mdl_art_v1/artifact",
            files={"file": ("model.pt", b"fake-weights", "application/octet-stream")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["artifact_format"] == "pt"
        assert data["artifact_size_bytes"] == len(b"fake-weights")
        assert data["artifact_sha256"] is not None

    async def test_upload_to_nonexistent(self, client: AsyncClient) -> None:
        resp = await client.put(
            f"{BASE}/nonexistent/artifact",
            files={"file": ("model.pt", b"data", "application/octet-stream")},
        )
        assert resp.status_code == 404


# ======================================================================
# GET /api/v1/models/{id}/artifact
# ======================================================================


class TestDownloadArtifact:
    async def test_download_artifact(self, client: AsyncClient) -> None:
        await _create_model(client, model_id="mdl_dl_v1")
        content = b"model-weights-here"
        await client.put(
            f"{BASE}/mdl_dl_v1/artifact",
            files={"file": ("model.bin", content, "application/octet-stream")},
        )
        resp = await client.get(f"{BASE}/mdl_dl_v1/artifact")
        assert resp.status_code == 200
        assert resp.content == content

    async def test_download_no_artifact(self, client: AsyncClient) -> None:
        await _create_model(client, model_id="mdl_noart_v1")
        resp = await client.get(f"{BASE}/mdl_noart_v1/artifact")
        assert resp.status_code == 404

    async def test_download_nonexistent_model(self, client: AsyncClient) -> None:
        resp = await client.get(f"{BASE}/nonexistent/artifact")
        assert resp.status_code == 404
