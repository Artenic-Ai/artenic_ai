"""Tests for artenic_ai_platform.entities.ensembles.router — REST API endpoints."""

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

BASE = "/api/v1/ensembles"


async def _create_ensemble(
    client: AsyncClient,
    ens_id: str = "ens_test_v1",
    **kwargs: Any,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": ens_id,
        "name": kwargs.get("name", "my-ensemble"),
        "strategy_type": kwargs.get("strategy_type", "voting"),
        "metadata": kwargs.get("metadata", {}),
        "metrics": kwargs.get("metrics", {}),
        "model_ids": kwargs.get("model_ids", []),
    }
    resp = await client.post(BASE, json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()


# ======================================================================
# POST /api/v1/ensembles
# ======================================================================


class TestCreateEnsemble:
    async def test_create_returns_201(self, client: AsyncClient) -> None:
        data = await _create_ensemble(client, ens_id="ens_cr_v1")
        assert data["id"] == "ens_cr_v1"
        assert data["name"] == "my-ensemble"
        assert data["strategy_type"] == "voting"
        assert data["stage"] == "staging"
        assert data["version"] == 1

    async def test_create_with_model_ids(self, client: AsyncClient) -> None:
        data = await _create_ensemble(
            client,
            ens_id="ens_mids_v1",
            model_ids=["mdl_a_v1", "mdl_b_v1"],
        )
        assert data["model_ids"] == ["mdl_a_v1", "mdl_b_v1"]

    async def test_create_auto_version(self, client: AsyncClient) -> None:
        await _create_ensemble(client, ens_id="ens_av_v1", name="auto-v")
        data = await _create_ensemble(client, ens_id="ens_av_v2", name="auto-v")
        assert data["version"] == 2

    async def test_create_validation_error(self, client: AsyncClient) -> None:
        resp = await client.post(BASE, json={})
        assert resp.status_code == 422


# ======================================================================
# GET /api/v1/ensembles
# ======================================================================


class TestListEnsembles:
    async def test_list_empty(self, client: AsyncClient) -> None:
        resp = await client.get(BASE)
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_list_after_create(self, client: AsyncClient) -> None:
        await _create_ensemble(client, ens_id="ens_l1_v1", name="e1")
        await _create_ensemble(client, ens_id="ens_l2_v1", name="e2")
        resp = await client.get(BASE)
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    async def test_list_with_stage_filter(self, client: AsyncClient) -> None:
        await _create_ensemble(client, ens_id="ens_sf_v1")
        resp = await client.get(BASE, params={"stage": "staging"})
        assert resp.status_code == 200
        assert len(resp.json()) == 1


# ======================================================================
# GET /api/v1/ensembles/{id}
# ======================================================================


class TestGetEnsemble:
    async def test_get_returns_details(self, client: AsyncClient) -> None:
        await _create_ensemble(client, ens_id="ens_get_v1", name="detail")
        resp = await client.get(f"{BASE}/ens_get_v1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "ens_get_v1"
        assert "model_ids" in data

    async def test_get_not_found(self, client: AsyncClient) -> None:
        resp = await client.get(f"{BASE}/nonexistent")
        assert resp.status_code == 404


# ======================================================================
# PATCH /api/v1/ensembles/{id}
# ======================================================================


class TestUpdateEnsemble:
    async def test_update_metrics(self, client: AsyncClient) -> None:
        await _create_ensemble(client, ens_id="ens_upd_v1")
        resp = await client.patch(
            f"{BASE}/ens_upd_v1",
            json={"metrics": {"accuracy": 0.98}},
        )
        assert resp.status_code == 200
        assert resp.json()["metrics"] == {"accuracy": 0.98}

    async def test_update_not_found(self, client: AsyncClient) -> None:
        resp = await client.patch(f"{BASE}/nonexistent", json={"metrics": {}})
        assert resp.status_code == 404


# ======================================================================
# DELETE /api/v1/ensembles/{id}
# ======================================================================


class TestDeleteEnsemble:
    async def test_delete_returns_204(self, client: AsyncClient) -> None:
        await _create_ensemble(client, ens_id="ens_del_v1")
        resp = await client.delete(f"{BASE}/ens_del_v1")
        assert resp.status_code == 204

    async def test_delete_not_found(self, client: AsyncClient) -> None:
        resp = await client.delete(f"{BASE}/nonexistent")
        assert resp.status_code == 404


# ======================================================================
# PATCH /api/v1/ensembles/{id}/stage
# ======================================================================


class TestChangeStage:
    async def test_staging_to_production(self, client: AsyncClient) -> None:
        await _create_ensemble(client, ens_id="ens_st_v1")
        resp = await client.patch(
            f"{BASE}/ens_st_v1/stage",
            json={"stage": "production"},
        )
        assert resp.status_code == 200
        assert resp.json()["stage"] == "production"

    async def test_production_to_retired(self, client: AsyncClient) -> None:
        await _create_ensemble(client, ens_id="ens_ret_v1")
        await client.patch(f"{BASE}/ens_ret_v1/stage", json={"stage": "production"})
        resp = await client.patch(
            f"{BASE}/ens_ret_v1/stage",
            json={"stage": "retired"},
        )
        assert resp.status_code == 200
        assert resp.json()["stage"] == "retired"

    async def test_invalid_transition(self, client: AsyncClient) -> None:
        await _create_ensemble(client, ens_id="ens_inv_v1")
        await client.patch(f"{BASE}/ens_inv_v1/stage", json={"stage": "retired"})
        # retired -> staging is NOT allowed
        resp = await client.patch(
            f"{BASE}/ens_inv_v1/stage",
            json={"stage": "staging"},
        )
        assert resp.status_code == 400


# ======================================================================
# POST/GET/DELETE /api/v1/ensembles/{id}/models
# ======================================================================


class TestModelManagement:
    async def test_add_model(self, client: AsyncClient) -> None:
        await _create_ensemble(client, ens_id="ens_am_v1")
        resp = await client.post(
            f"{BASE}/ens_am_v1/models",
            json={"model_id": "mdl_x_v1"},
        )
        assert resp.status_code == 201
        assert resp.json()["model_id"] == "mdl_x_v1"

    async def test_list_models(self, client: AsyncClient) -> None:
        await _create_ensemble(
            client,
            ens_id="ens_lm_v1",
            model_ids=["mdl_a_v1", "mdl_b_v1"],
        )
        resp = await client.get(f"{BASE}/ens_lm_v1/models")
        assert resp.status_code == 200
        assert set(resp.json()) == {"mdl_a_v1", "mdl_b_v1"}

    async def test_remove_model(self, client: AsyncClient) -> None:
        await _create_ensemble(
            client,
            ens_id="ens_rm_v1",
            model_ids=["mdl_del_v1"],
        )
        resp = await client.delete(f"{BASE}/ens_rm_v1/models/mdl_del_v1")
        assert resp.status_code == 204

    async def test_remove_model_not_found(self, client: AsyncClient) -> None:
        await _create_ensemble(client, ens_id="ens_rmnf_v1")
        resp = await client.delete(f"{BASE}/ens_rmnf_v1/models/nonexistent")
        assert resp.status_code == 404


# ======================================================================
# Additional coverage tests
# ======================================================================


class TestEnsembleCoverageEdgeCases:
    async def test_create_with_explicit_version(self, client: AsyncClient) -> None:
        body: dict[str, Any] = {
            "id": "ens_ev_v5",
            "name": "expl-v",
            "strategy_type": "voting",
            "version": 5,
        }
        resp = await client.post(BASE, json=body)
        assert resp.status_code == 201
        assert resp.json()["version"] == 5

    async def test_list_with_strategy_type_filter(self, client: AsyncClient) -> None:
        await _create_ensemble(client, ens_id="ens_stf1_v1", strategy_type="voting")
        await _create_ensemble(client, ens_id="ens_stf2_v1", name="e2", strategy_type="stacking")
        resp = await client.get(BASE, params={"strategy_type": "stacking"})
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    async def test_update_metadata(self, client: AsyncClient) -> None:
        await _create_ensemble(client, ens_id="ens_umd_v1")
        resp = await client.patch(
            f"{BASE}/ens_umd_v1",
            json={"metadata": {"key": "value"}},
        )
        assert resp.status_code == 200
        assert resp.json()["metadata"] == {"key": "value"}

    async def test_add_model_nonexistent_ensemble(self, client: AsyncClient) -> None:
        resp = await client.post(
            f"{BASE}/nonexistent/models",
            json={"model_id": "mdl_x_v1"},
        )
        assert resp.status_code == 404

    async def test_list_models_nonexistent_ensemble(self, client: AsyncClient) -> None:
        resp = await client.get(f"{BASE}/nonexistent/models")
        assert resp.status_code == 404
