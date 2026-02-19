"""Tests for artenic_ai_platform.datasets.router — REST API endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    """Create a FastAPI app with full lifespan using tmp_path for storage."""
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
    """Async HTTP client wired to the fully-initialised test app."""
    transport = ASGITransport(app=app_with_lifespan)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ======================================================================
# Helpers
# ======================================================================

BASE = "/api/v1/datasets"


def _create_body(
    name: str = "my-dataset",
    fmt: str = "csv",
    **kwargs: object,
) -> dict:
    return {
        "name": name,
        "format": fmt,
        "description": kwargs.get("description", "test dataset"),
        "storage_backend": kwargs.get("storage_backend", "filesystem"),
        "source": kwargs.get("source", "test"),
        "tags": kwargs.get("tags", {}),
    }


async def _create_dataset(client: AsyncClient, **kwargs: object) -> str:
    """Create a dataset and return its id."""
    resp = await client.post(BASE, json=_create_body(**kwargs))
    assert resp.status_code == 201
    return resp.json()["id"]


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
        # Filesystem should be available
        fs = next(o for o in data if o["id"] == "filesystem")
        assert fs["available"] is True


# ======================================================================
# POST /api/v1/datasets
# ======================================================================


class TestCreateDataset:
    async def test_create_returns_201(self, client: AsyncClient) -> None:
        resp = await client.post(BASE, json=_create_body())
        assert resp.status_code == 201
        data = resp.json()
        assert "id" in data
        assert len(data["id"]) == 36

    async def test_create_validation_error(self, client: AsyncClient) -> None:
        resp = await client.post(BASE, json={})
        assert resp.status_code == 422


# ======================================================================
# GET /api/v1/datasets
# ======================================================================


class TestListDatasets:
    async def test_list_empty(self, client: AsyncClient) -> None:
        resp = await client.get(BASE)
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_list_after_create(self, client: AsyncClient) -> None:
        await _create_dataset(client, name="ds1")
        await _create_dataset(client, name="ds2")
        resp = await client.get(BASE)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2


# ======================================================================
# GET /api/v1/datasets/{dataset_id}
# ======================================================================


class TestGetDataset:
    async def test_get_returns_details(self, client: AsyncClient) -> None:
        ds_id = await _create_dataset(client, name="detailed")
        resp = await client.get(f"{BASE}/{ds_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == ds_id
        assert data["name"] == "detailed"
        assert data["format"] == "csv"

    async def test_get_not_found(self, client: AsyncClient) -> None:
        resp = await client.get(f"{BASE}/nonexistent-id")
        assert resp.status_code == 404


# ======================================================================
# PATCH /api/v1/datasets/{dataset_id}
# ======================================================================


class TestUpdateDataset:
    async def test_update_metadata(self, client: AsyncClient) -> None:
        ds_id = await _create_dataset(client)
        resp = await client.patch(f"{BASE}/{ds_id}", json={"name": "updated-name"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "updated-name"

    async def test_update_not_found(self, client: AsyncClient) -> None:
        resp = await client.patch(f"{BASE}/nonexistent", json={"name": "x"})
        assert resp.status_code == 404


# ======================================================================
# DELETE /api/v1/datasets/{dataset_id}
# ======================================================================


class TestDeleteDataset:
    async def test_delete_returns_204(self, client: AsyncClient) -> None:
        ds_id = await _create_dataset(client)
        resp = await client.delete(f"{BASE}/{ds_id}")
        assert resp.status_code == 204

    async def test_delete_not_found(self, client: AsyncClient) -> None:
        resp = await client.delete(f"{BASE}/nonexistent")
        assert resp.status_code == 404


# ======================================================================
# POST /api/v1/datasets/{dataset_id}/files
# ======================================================================


class TestUploadFile:
    async def test_upload_multipart(self, client: AsyncClient) -> None:
        ds_id = await _create_dataset(client)
        resp = await client.post(
            f"{BASE}/{ds_id}/files",
            files={"file": ("test.csv", b"col1,col2\na,b\n", "text/csv")},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["filename"] == "test.csv"
        assert data["size_bytes"] == len(b"col1,col2\na,b\n")
        assert data["dataset_id"] == ds_id

    async def test_upload_to_nonexistent_dataset(self, client: AsyncClient) -> None:
        resp = await client.post(
            f"{BASE}/nonexistent/files",
            files={"file": ("f.csv", b"data", "text/csv")},
        )
        assert resp.status_code == 404

    async def test_upload_disallowed_extension(self, client: AsyncClient) -> None:
        ds_id = await _create_dataset(client)
        resp = await client.post(
            f"{BASE}/{ds_id}/files",
            files={"file": ("virus.exe", b"bad", "application/octet-stream")},
        )
        assert resp.status_code == 400
        body = resp.json()
        msg = body.get("error", {}).get("message", body.get("detail", ""))
        assert "extension" in msg.lower()

    async def test_upload_sanitizes_filename(self, client: AsyncClient) -> None:
        ds_id = await _create_dataset(client)
        resp = await client.post(
            f"{BASE}/{ds_id}/files",
            files={"file": ("../../etc/data.csv", b"data", "text/csv")},
        )
        assert resp.status_code == 201
        data = resp.json()
        # Filename should be sanitized (no path components)
        assert "/" not in data["filename"]
        assert ".." not in data["filename"]


# ======================================================================
# GET /api/v1/datasets/{dataset_id}/files
# ======================================================================


class TestListFiles:
    async def test_list_files(self, client: AsyncClient) -> None:
        ds_id = await _create_dataset(client)
        await client.post(
            f"{BASE}/{ds_id}/files",
            files={"file": ("a.csv", b"data_a", "text/csv")},
        )
        await client.post(
            f"{BASE}/{ds_id}/files",
            files={"file": ("b.csv", b"data_b", "text/csv")},
        )
        resp = await client.get(f"{BASE}/{ds_id}/files")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    async def test_list_files_nonexistent_dataset(self, client: AsyncClient) -> None:
        resp = await client.get(f"{BASE}/nonexistent/files")
        assert resp.status_code == 404


# ======================================================================
# GET /api/v1/datasets/{dataset_id}/files/{filename}
# ======================================================================


class TestDownloadFile:
    async def test_download_file(self, client: AsyncClient) -> None:
        ds_id = await _create_dataset(client)
        content = b"download me"
        await client.post(
            f"{BASE}/{ds_id}/files",
            files={"file": ("dl.txt", content, "text/plain")},
        )
        resp = await client.get(f"{BASE}/{ds_id}/files/dl.txt")
        assert resp.status_code == 200
        assert resp.content == content

    async def test_download_not_found(self, client: AsyncClient) -> None:
        ds_id = await _create_dataset(client)
        resp = await client.get(f"{BASE}/{ds_id}/files/nonexistent.txt")
        assert resp.status_code == 404


# ======================================================================
# DELETE /api/v1/datasets/{dataset_id}/files/{filename}
# ======================================================================


class TestDeleteFile:
    async def test_delete_file_returns_204(self, client: AsyncClient) -> None:
        ds_id = await _create_dataset(client)
        await client.post(
            f"{BASE}/{ds_id}/files",
            files={"file": ("rm.csv", b"bye", "text/csv")},
        )
        resp = await client.delete(f"{BASE}/{ds_id}/files/rm.csv")
        assert resp.status_code == 204

    async def test_delete_file_not_found(self, client: AsyncClient) -> None:
        ds_id = await _create_dataset(client)
        resp = await client.delete(f"{BASE}/{ds_id}/files/missing.csv")
        assert resp.status_code == 404


# ======================================================================
# POST /api/v1/datasets/{dataset_id}/versions
# ======================================================================


class TestCreateVersion:
    async def test_create_version(self, client: AsyncClient) -> None:
        ds_id = await _create_dataset(client)
        await client.post(
            f"{BASE}/{ds_id}/files",
            files={"file": ("v.csv", b"col\n1", "text/csv")},
        )
        resp = await client.post(
            f"{BASE}/{ds_id}/versions",
            json={"change_summary": "first version"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["version"] == 1
        assert data["change_summary"] == "first version"
        assert data["num_files"] == 1

    async def test_create_version_not_found(self, client: AsyncClient) -> None:
        resp = await client.post(
            f"{BASE}/nonexistent/versions",
            json={"change_summary": "nope"},
        )
        assert resp.status_code == 404


# ======================================================================
# GET /api/v1/datasets/{dataset_id}/versions
# ======================================================================


class TestListVersions:
    async def test_list_versions(self, client: AsyncClient) -> None:
        ds_id = await _create_dataset(client)
        await client.post(
            f"{BASE}/{ds_id}/files",
            files={"file": ("f.csv", b"d", "text/csv")},
        )
        await client.post(f"{BASE}/{ds_id}/versions", json={"change_summary": "v1"})
        await client.post(f"{BASE}/{ds_id}/versions", json={"change_summary": "v2"})
        resp = await client.get(f"{BASE}/{ds_id}/versions")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        # Descending order
        assert data[0]["version"] > data[1]["version"]

    async def test_list_versions_nonexistent_dataset(self, client: AsyncClient) -> None:
        resp = await client.get(f"{BASE}/nonexistent/versions")
        assert resp.status_code == 404


# ======================================================================
# GET /api/v1/datasets/{dataset_id}/stats
# ======================================================================


class TestGetStats:
    async def test_stats_returns_stats(self, client: AsyncClient) -> None:
        ds_id = await _create_dataset(client)
        await client.post(
            f"{BASE}/{ds_id}/files",
            files={"file": ("s.csv", b"name\nalice\nbob\n", "text/csv")},
        )
        resp = await client.get(f"{BASE}/{ds_id}/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_files" in data
        assert "total_size_bytes" in data
        assert "format_breakdown" in data

    async def test_stats_not_found(self, client: AsyncClient) -> None:
        resp = await client.get(f"{BASE}/nonexistent/stats")
        assert resp.status_code == 404


# ======================================================================
# GET /api/v1/datasets/{dataset_id}/preview
# ======================================================================


class TestGetPreview:
    async def test_preview_csv(self, client: AsyncClient) -> None:
        ds_id = await _create_dataset(client)
        csv_data = b"name,age\nalice,30\nbob,25\n"
        await client.post(
            f"{BASE}/{ds_id}/files",
            files={"file": ("preview.csv", csv_data, "text/csv")},
        )
        resp = await client.get(f"{BASE}/{ds_id}/preview")
        assert resp.status_code == 200
        data = resp.json()
        assert data["columns"] == ["name", "age"]
        assert len(data["rows"]) == 2

    async def test_preview_not_found(self, client: AsyncClient) -> None:
        resp = await client.get(f"{BASE}/nonexistent/preview")
        assert resp.status_code == 404


# ======================================================================
# POST /api/v1/datasets/{dataset_id}/lineage
# ======================================================================


class TestAddLineage:
    async def test_add_lineage(self, client: AsyncClient) -> None:
        ds_id = await _create_dataset(client)
        await client.post(
            f"{BASE}/{ds_id}/files",
            files={"file": ("l.csv", b"d", "text/csv")},
        )
        await client.post(f"{BASE}/{ds_id}/versions", json={"change_summary": "v1"})
        resp = await client.post(
            f"{BASE}/{ds_id}/lineage",
            json={
                "dataset_version": 1,
                "entity_type": "model",
                "entity_id": "model-abc",
                "role": "input",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["dataset_id"] == ds_id
        assert data["entity_type"] == "model"
        assert data["entity_id"] == "model-abc"

    async def test_add_lineage_not_found(self, client: AsyncClient) -> None:
        resp = await client.post(
            f"{BASE}/nonexistent/lineage",
            json={
                "dataset_version": 1,
                "entity_type": "model",
                "entity_id": "m1",
                "role": "input",
            },
        )
        assert resp.status_code == 404


# ======================================================================
# GET /api/v1/datasets/{dataset_id}/lineage
# ======================================================================


class TestGetLineage:
    async def test_get_lineage(self, client: AsyncClient) -> None:
        ds_id = await _create_dataset(client)
        await client.post(
            f"{BASE}/{ds_id}/files",
            files={"file": ("l.csv", b"d", "text/csv")},
        )
        await client.post(f"{BASE}/{ds_id}/versions", json={"change_summary": "v1"})
        await client.post(
            f"{BASE}/{ds_id}/lineage",
            json={
                "dataset_version": 1,
                "entity_type": "model",
                "entity_id": "model-1",
                "role": "input",
            },
        )
        resp = await client.get(f"{BASE}/{ds_id}/lineage")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["entity_type"] == "model"

    async def test_get_lineage_empty(self, client: AsyncClient) -> None:
        ds_id = await _create_dataset(client)
        resp = await client.get(f"{BASE}/{ds_id}/lineage")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_get_lineage_nonexistent_dataset(self, client: AsyncClient) -> None:
        resp = await client.get(f"{BASE}/nonexistent/lineage")
        assert resp.status_code == 404
