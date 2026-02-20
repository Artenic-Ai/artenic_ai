"""Tests for artenic_ai_platform.entities.runs.router — REST API endpoints."""

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

BASE = "/api/v1/runs"


async def _create_run(
    client: AsyncClient,
    run_id: str = "run_test_v1",
    **kwargs: Any,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": run_id,
        "config": kwargs.get("config", {"epochs": 10}),
        "triggered_by": kwargs.get("triggered_by", "manual"),
        "metrics": kwargs.get("metrics", {}),
    }
    resp = await client.post(BASE, json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()


# ======================================================================
# POST /api/v1/runs
# ======================================================================


class TestCreateRun:
    async def test_create_returns_201(self, client: AsyncClient) -> None:
        data = await _create_run(client, run_id="run_cr_v1")
        assert data["id"] == "run_cr_v1"
        assert data["status"] == "pending"
        assert data["config"] == {"epochs": 10}
        assert data["triggered_by"] == "manual"

    async def test_create_with_custom_config(self, client: AsyncClient) -> None:
        data = await _create_run(
            client,
            run_id="run_cfg_v1",
            config={"lr": 0.001, "batch_size": 32},
        )
        assert data["config"] == {"lr": 0.001, "batch_size": 32}

    async def test_create_validation_error(self, client: AsyncClient) -> None:
        resp = await client.post(BASE, json={})
        assert resp.status_code == 422


# ======================================================================
# GET /api/v1/runs
# ======================================================================


class TestListRuns:
    async def test_list_empty(self, client: AsyncClient) -> None:
        resp = await client.get(BASE)
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_list_after_create(self, client: AsyncClient) -> None:
        await _create_run(client, run_id="run_l1_v1")
        await _create_run(client, run_id="run_l2_v1")
        resp = await client.get(BASE)
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    async def test_list_with_status_filter(self, client: AsyncClient) -> None:
        await _create_run(client, run_id="run_sf_v1")
        resp = await client.get(BASE, params={"status": "pending"})
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    async def test_list_with_triggered_by_filter(self, client: AsyncClient) -> None:
        await _create_run(client, run_id="run_tb1_v1", triggered_by="ci")
        await _create_run(client, run_id="run_tb2_v1", triggered_by="manual")
        resp = await client.get(BASE, params={"triggered_by": "ci"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["triggered_by"] == "ci"


# ======================================================================
# GET /api/v1/runs/{id}
# ======================================================================


class TestGetRun:
    async def test_get_returns_details_with_io(self, client: AsyncClient) -> None:
        await _create_run(client, run_id="run_get_v1")
        resp = await client.get(f"{BASE}/run_get_v1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "run_get_v1"
        assert "inputs" in data
        assert "outputs" in data

    async def test_get_not_found(self, client: AsyncClient) -> None:
        resp = await client.get(f"{BASE}/nonexistent")
        assert resp.status_code == 404


# ======================================================================
# DELETE /api/v1/runs/{id}
# ======================================================================


class TestDeleteRun:
    async def test_delete_returns_204(self, client: AsyncClient) -> None:
        await _create_run(client, run_id="run_del_v1")
        resp = await client.delete(f"{BASE}/run_del_v1")
        assert resp.status_code == 204

    async def test_delete_not_found(self, client: AsyncClient) -> None:
        resp = await client.delete(f"{BASE}/nonexistent")
        assert resp.status_code == 404


# ======================================================================
# PATCH /api/v1/runs/{id}/status
# ======================================================================


class TestChangeStatus:
    async def test_pending_to_running(self, client: AsyncClient) -> None:
        await _create_run(client, run_id="run_st_v1")
        resp = await client.patch(
            f"{BASE}/run_st_v1/status",
            json={"status": "running"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"
        assert data["started_at"] is not None

    async def test_running_to_completed_with_metrics(self, client: AsyncClient) -> None:
        await _create_run(client, run_id="run_comp_v1")
        await client.patch(
            f"{BASE}/run_comp_v1/status",
            json={"status": "running"},
        )
        resp = await client.patch(
            f"{BASE}/run_comp_v1/status",
            json={
                "status": "completed",
                "metrics": {"accuracy": 0.95},
                "duration_seconds": 3600.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["completed_at"] is not None
        assert data["metrics"]["accuracy"] == 0.95
        assert data["duration_seconds"] == 3600.0

    async def test_pending_to_failed(self, client: AsyncClient) -> None:
        await _create_run(client, run_id="run_fail_v1")
        resp = await client.patch(
            f"{BASE}/run_fail_v1/status",
            json={"status": "failed"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "failed"

    async def test_invalid_transition(self, client: AsyncClient) -> None:
        await _create_run(client, run_id="run_inv_v1")
        # pending -> completed (not allowed, must go through running)
        resp = await client.patch(
            f"{BASE}/run_inv_v1/status",
            json={"status": "completed"},
        )
        assert resp.status_code == 400

    async def test_terminal_state_no_transition(self, client: AsyncClient) -> None:
        await _create_run(client, run_id="run_term_v1")
        await client.patch(
            f"{BASE}/run_term_v1/status",
            json={"status": "failed"},
        )
        # failed -> running (not allowed, terminal)
        resp = await client.patch(
            f"{BASE}/run_term_v1/status",
            json={"status": "running"},
        )
        assert resp.status_code == 400


# ======================================================================
# POST /api/v1/runs/{id}/io
# ======================================================================


class TestAddIO:
    async def test_add_input(self, client: AsyncClient) -> None:
        await _create_run(client, run_id="run_io_v1")
        resp = await client.post(
            f"{BASE}/run_io_v1/io",
            json={"entity_id": "ds_train_v1", "direction": "input"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["entity_id"] == "ds_train_v1"
        assert data["direction"] == "input"
        assert data["run_id"] == "run_io_v1"

    async def test_add_output(self, client: AsyncClient) -> None:
        await _create_run(client, run_id="run_io2_v1")
        resp = await client.post(
            f"{BASE}/run_io2_v1/io",
            json={"entity_id": "mdl_result_v1", "direction": "output"},
        )
        assert resp.status_code == 201
        assert resp.json()["direction"] == "output"

    async def test_add_io_to_nonexistent(self, client: AsyncClient) -> None:
        resp = await client.post(
            f"{BASE}/nonexistent/io",
            json={"entity_id": "ds_x_v1", "direction": "input"},
        )
        assert resp.status_code == 404

    async def test_invalid_direction(self, client: AsyncClient) -> None:
        await _create_run(client, run_id="run_iodir_v1")
        resp = await client.post(
            f"{BASE}/run_iodir_v1/io",
            json={"entity_id": "ds_x_v1", "direction": "sideways"},
        )
        assert resp.status_code == 422


# ======================================================================
# GET /api/v1/runs/{id}/io
# ======================================================================


class TestListIO:
    async def test_list_io(self, client: AsyncClient) -> None:
        await _create_run(client, run_id="run_lio_v1")
        await client.post(
            f"{BASE}/run_lio_v1/io",
            json={"entity_id": "ds_a_v1", "direction": "input"},
        )
        await client.post(
            f"{BASE}/run_lio_v1/io",
            json={"entity_id": "mdl_out_v1", "direction": "output"},
        )
        resp = await client.get(f"{BASE}/run_lio_v1/io")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    async def test_list_io_nonexistent(self, client: AsyncClient) -> None:
        resp = await client.get(f"{BASE}/nonexistent/io")
        assert resp.status_code == 404


# ======================================================================
# DELETE /api/v1/runs/{id}/io
# ======================================================================


class TestRemoveIO:
    async def test_remove_io(self, client: AsyncClient) -> None:
        await _create_run(client, run_id="run_rio_v1")
        await client.post(
            f"{BASE}/run_rio_v1/io",
            json={"entity_id": "ds_rm_v1", "direction": "input"},
        )
        resp = await client.delete(
            f"{BASE}/run_rio_v1/io",
            params={"entity_id": "ds_rm_v1", "direction": "input"},
        )
        assert resp.status_code == 200

    async def test_remove_io_not_found(self, client: AsyncClient) -> None:
        await _create_run(client, run_id="run_rionf_v1")
        resp = await client.delete(
            f"{BASE}/run_rionf_v1/io",
            params={"entity_id": "nonexistent", "direction": "input"},
        )
        assert resp.status_code == 404


# ======================================================================
# GET enriched detail includes IO
# ======================================================================


class TestGetWithIO:
    async def test_get_includes_inputs_outputs(self, client: AsyncClient) -> None:
        await _create_run(client, run_id="run_enr_v1")
        await client.post(
            f"{BASE}/run_enr_v1/io",
            json={"entity_id": "ds_train_v1", "direction": "input"},
        )
        await client.post(
            f"{BASE}/run_enr_v1/io",
            json={"entity_id": "mdl_out_v1", "direction": "output"},
        )
        resp = await client.get(f"{BASE}/run_enr_v1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["inputs"]) == 1
        assert data["inputs"][0]["entity_id"] == "ds_train_v1"
        assert len(data["outputs"]) == 1
        assert data["outputs"][0]["entity_id"] == "mdl_out_v1"
