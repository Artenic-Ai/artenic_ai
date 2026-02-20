"""Tests for artenic_ai_platform.entities.datasets.router — REST API endpoints."""

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

BASE = "/api/v1/datasets"


def _create_body(
    ds_id: str = "test_dataset_v1",
    name: str = "my-dataset",
    fmt: str = "csv",
    **kwargs: Any,
) -> dict[str, Any]:
    return {
        "id": ds_id,
        "name": name,
        "format": fmt,
        "description": kwargs.get("description", "test dataset"),
        "metadata": kwargs.get("metadata", {}),
    }


async def _create_dataset(
    client: AsyncClient,
    ds_id: str = "test_dataset_v1",
    **kwargs: Any,
) -> dict[str, Any]:
    resp = await client.post(BASE, json=_create_body(ds_id=ds_id, **kwargs))
    assert resp.status_code == 201, resp.text
    return resp.json()


# ======================================================================
# POST /api/v1/datasets
# ======================================================================


class TestCreateDataset:
    async def test_create_returns_201(self, client: AsyncClient) -> None:
        data = await _create_dataset(client, ds_id="ds_create_v1")
        assert data["id"] == "ds_create_v1"
        assert data["name"] == "my-dataset"
        assert data["format"] == "csv"
        assert data["status"] == "created"
        assert data["version"] == 1

    async def test_create_auto_increments_version(self, client: AsyncClient) -> None:
        await _create_dataset(client, ds_id="ds_autoV_v1", name="auto-v")
        data = await _create_dataset(client, ds_id="ds_autoV_v2", name="auto-v")
        assert data["version"] == 2

    async def test_create_with_explicit_version(self, client: AsyncClient) -> None:
        body = _create_body(ds_id="ds_explV_v5", name="expl")
        body["version"] = 5
        resp = await client.post(BASE, json=body)
        assert resp.status_code == 201
        assert resp.json()["version"] == 5

    async def test_create_validation_error(self, client: AsyncClient) -> None:
        resp = await client.post(BASE, json={})
        assert resp.status_code == 422

    async def test_create_with_metadata(self, client: AsyncClient) -> None:
        data = await _create_dataset(
            client,
            ds_id="ds_meta_v1",
            metadata={"source": "api", "tags": ["prod"]},
        )
        assert data["metadata"] == {"source": "api", "tags": ["prod"]}


# ======================================================================
# GET /api/v1/datasets
# ======================================================================


class TestListDatasets:
    async def test_list_empty(self, client: AsyncClient) -> None:
        resp = await client.get(BASE)
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_list_after_create(self, client: AsyncClient) -> None:
        await _create_dataset(client, ds_id="ds_l1_v1", name="ds1")
        await _create_dataset(client, ds_id="ds_l2_v1", name="ds2")
        resp = await client.get(BASE)
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    async def test_list_with_status_filter(self, client: AsyncClient) -> None:
        await _create_dataset(client, ds_id="ds_filter_v1")
        resp = await client.get(BASE, params={"status": "created"})
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    async def test_list_with_name_filter(self, client: AsyncClient) -> None:
        await _create_dataset(client, ds_id="ds_nf1_v1", name="specific-name")
        await _create_dataset(client, ds_id="ds_nf2_v1", name="other")
        resp = await client.get(BASE, params={"name": "specific-name"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "specific-name"


# ======================================================================
# GET /api/v1/datasets/{id}
# ======================================================================


class TestGetDataset:
    async def test_get_returns_details(self, client: AsyncClient) -> None:
        await _create_dataset(client, ds_id="ds_get_v1", name="detail")
        resp = await client.get(f"{BASE}/ds_get_v1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "ds_get_v1"
        assert data["name"] == "detail"

    async def test_get_not_found(self, client: AsyncClient) -> None:
        resp = await client.get(f"{BASE}/nonexistent")
        assert resp.status_code == 404


# ======================================================================
# PATCH /api/v1/datasets/{id}
# ======================================================================


class TestUpdateDataset:
    async def test_update_description(self, client: AsyncClient) -> None:
        await _create_dataset(client, ds_id="ds_upd_v1")
        resp = await client.patch(
            f"{BASE}/ds_upd_v1",
            json={"description": "updated desc"},
        )
        assert resp.status_code == 200
        assert resp.json()["description"] == "updated desc"

    async def test_update_metadata(self, client: AsyncClient) -> None:
        await _create_dataset(client, ds_id="ds_updm_v1")
        resp = await client.patch(
            f"{BASE}/ds_updm_v1",
            json={"metadata": {"new": "value"}},
        )
        assert resp.status_code == 200
        assert resp.json()["metadata"] == {"new": "value"}

    async def test_update_not_found(self, client: AsyncClient) -> None:
        resp = await client.patch(f"{BASE}/nonexistent", json={"description": "x"})
        assert resp.status_code == 404


# ======================================================================
# DELETE /api/v1/datasets/{id}
# ======================================================================


class TestDeleteDataset:
    async def test_delete_returns_204(self, client: AsyncClient) -> None:
        await _create_dataset(client, ds_id="ds_del_v1")
        resp = await client.delete(f"{BASE}/ds_del_v1")
        assert resp.status_code == 204

    async def test_delete_not_found(self, client: AsyncClient) -> None:
        resp = await client.delete(f"{BASE}/nonexistent")
        assert resp.status_code == 404


# ======================================================================
# PATCH /api/v1/datasets/{id}/status
# ======================================================================


class TestChangeStatus:
    async def test_change_status_created_to_active(self, client: AsyncClient) -> None:
        await _create_dataset(client, ds_id="ds_st_v1")
        resp = await client.patch(
            f"{BASE}/ds_st_v1/status",
            json={"status": "active"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "active"

    async def test_change_status_invalid_transition(self, client: AsyncClient) -> None:
        await _create_dataset(client, ds_id="ds_inv_v1")
        resp = await client.patch(
            f"{BASE}/ds_inv_v1/status",
            json={"status": "archived"},
        )
        # created -> archived is allowed
        assert resp.status_code == 200
        # Now archived -> active is NOT allowed
        resp = await client.patch(
            f"{BASE}/ds_inv_v1/status",
            json={"status": "active"},
        )
        assert resp.status_code == 400


# ======================================================================
# POST /api/v1/datasets/{id}/versions
# ======================================================================


class TestCreateVersion:
    async def test_create_new_version(self, client: AsyncClient) -> None:
        await _create_dataset(client, ds_id="ds_ver_v1", name="versioned-ds")
        resp = await client.post(
            f"{BASE}/ds_ver_v1/versions",
            json={"change_summary": "second version"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["version"] == 2
        assert data["name"] == "versioned-ds"

    async def test_create_version_not_found(self, client: AsyncClient) -> None:
        resp = await client.post(
            f"{BASE}/nonexistent/versions",
            json={"change_summary": "nope"},
        )
        assert resp.status_code == 404


# ======================================================================
# POST /api/v1/datasets/{id}/files
# ======================================================================


class TestUploadFile:
    async def test_upload_multipart(self, client: AsyncClient) -> None:
        await _create_dataset(client, ds_id="ds_upl_v1")
        resp = await client.post(
            f"{BASE}/ds_upl_v1/files",
            files={"file": ("test.csv", b"col1,col2\na,b\n", "text/csv")},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["path"] == "test.csv"
        assert data["size_bytes"] == len(b"col1,col2\na,b\n")
        assert data["dataset_id"] == "ds_upl_v1"

    async def test_upload_to_nonexistent(self, client: AsyncClient) -> None:
        resp = await client.post(
            f"{BASE}/nonexistent/files",
            files={"file": ("f.csv", b"data", "text/csv")},
        )
        assert resp.status_code == 404

    async def test_upload_disallowed_extension(self, client: AsyncClient) -> None:
        await _create_dataset(client, ds_id="ds_ext_v1")
        resp = await client.post(
            f"{BASE}/ds_ext_v1/files",
            files={"file": ("virus.exe", b"bad", "application/octet-stream")},
        )
        assert resp.status_code == 400

    async def test_upload_exceeds_max_size(self, tmp_path: Path) -> None:
        settings = PlatformSettings(
            database_url="sqlite+aiosqlite://",
            api_key="",
            secret_key="test-secret",
            otel_enabled=False,
            dataset={
                "storage": {"local_dir": str(tmp_path / "ds")},
                "max_upload_size_mb": 0,
            },
        )
        app = create_app(settings)
        async with _lifespan(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                await _create_dataset(c, ds_id="ds_big_v1")
                resp = await c.post(
                    f"{BASE}/ds_big_v1/files",
                    files={"file": ("data.csv", b"x", "text/csv")},
                )
                assert resp.status_code == 413


# ======================================================================
# GET /api/v1/datasets/{id}/files
# ======================================================================


class TestListFiles:
    async def test_list_files(self, client: AsyncClient) -> None:
        await _create_dataset(client, ds_id="ds_lf_v1")
        await client.post(
            f"{BASE}/ds_lf_v1/files",
            files={"file": ("a.csv", b"data_a", "text/csv")},
        )
        await client.post(
            f"{BASE}/ds_lf_v1/files",
            files={"file": ("b.csv", b"data_b", "text/csv")},
        )
        resp = await client.get(f"{BASE}/ds_lf_v1/files")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    async def test_list_files_nonexistent(self, client: AsyncClient) -> None:
        resp = await client.get(f"{BASE}/nonexistent/files")
        assert resp.status_code == 404


# ======================================================================
# GET /api/v1/datasets/{id}/files/{path}
# ======================================================================


class TestDownloadFile:
    async def test_download_file(self, client: AsyncClient) -> None:
        await _create_dataset(client, ds_id="ds_dl_v1")
        content = b"download me"
        await client.post(
            f"{BASE}/ds_dl_v1/files",
            files={"file": ("dl.txt", content, "text/plain")},
        )
        resp = await client.get(f"{BASE}/ds_dl_v1/files/dl.txt")
        assert resp.status_code == 200
        assert resp.content == content

    async def test_download_not_found(self, client: AsyncClient) -> None:
        await _create_dataset(client, ds_id="ds_dlnf_v1")
        resp = await client.get(f"{BASE}/ds_dlnf_v1/files/nonexistent.txt")
        assert resp.status_code == 404


# ======================================================================
# DELETE /api/v1/datasets/{id}/files/{path}
# ======================================================================


class TestDeleteFile:
    async def test_delete_file_returns_204(self, client: AsyncClient) -> None:
        await _create_dataset(client, ds_id="ds_df_v1")
        await client.post(
            f"{BASE}/ds_df_v1/files",
            files={"file": ("rm.csv", b"bye", "text/csv")},
        )
        resp = await client.delete(f"{BASE}/ds_df_v1/files/rm.csv")
        assert resp.status_code == 204

    async def test_delete_file_not_found(self, client: AsyncClient) -> None:
        await _create_dataset(client, ds_id="ds_dfnf_v1")
        resp = await client.delete(f"{BASE}/ds_dfnf_v1/files/missing.csv")
        assert resp.status_code == 404


# ======================================================================
# GET /api/v1/datasets/{id}/stats
# ======================================================================


class TestGetStats:
    async def test_stats(self, client: AsyncClient) -> None:
        await _create_dataset(client, ds_id="ds_stats_v1")
        await client.post(
            f"{BASE}/ds_stats_v1/files",
            files={"file": ("s.csv", b"name\nalice\nbob\n", "text/csv")},
        )
        resp = await client.get(f"{BASE}/ds_stats_v1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_files" in data
        assert "total_size_bytes" in data
        assert "format_breakdown" in data

    async def test_stats_not_found(self, client: AsyncClient) -> None:
        resp = await client.get(f"{BASE}/nonexistent/stats")
        assert resp.status_code == 404


# ======================================================================
# GET /api/v1/datasets/{id}/preview
# ======================================================================


class TestGetPreview:
    async def test_preview_csv(self, client: AsyncClient) -> None:
        await _create_dataset(client, ds_id="ds_prev_v1")
        csv_data = b"name,age\nalice,30\nbob,25\n"
        await client.post(
            f"{BASE}/ds_prev_v1/files",
            files={"file": ("preview.csv", csv_data, "text/csv")},
        )
        resp = await client.get(f"{BASE}/ds_prev_v1/preview")
        assert resp.status_code == 200
        data = resp.json()
        assert data["columns"] == ["name", "age"]
        assert len(data["rows"]) == 2

    async def test_preview_not_found(self, client: AsyncClient) -> None:
        resp = await client.get(f"{BASE}/nonexistent/preview")
        assert resp.status_code == 404


# ======================================================================
# GET /api/v1/datasets/storage-options
# ======================================================================


class TestStorageOptions:
    async def test_returns_list_with_filesystem(self, client: AsyncClient) -> None:
        resp = await client.get(f"{BASE}/storage-options")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        ids = [opt["id"] for opt in data]
        assert "filesystem" in ids
