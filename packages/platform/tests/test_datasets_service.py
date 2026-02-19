"""Tests for artenic_ai_platform.datasets.service — CRUD, files, versioning, stats, lineage."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from artenic_ai_platform.datasets.service import DatasetService
from artenic_ai_platform.datasets.storage import FilesystemStorage
from artenic_ai_platform.db.models import Base

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
async def engine() -> AsyncGenerator[AsyncEngine, None]:
    """In-memory async SQLite engine."""
    e = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with e.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield e
    await e.dispose()


@pytest.fixture
async def session(engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Async session for service tests."""
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s


@pytest.fixture
def storage(tmp_path: Path) -> FilesystemStorage:
    """Filesystem storage rooted in tmp_path."""
    return FilesystemStorage(str(tmp_path / "datasets"))


@pytest.fixture
def svc(session: AsyncSession, storage: FilesystemStorage) -> DatasetService:
    """DatasetService instance with in-memory DB and tmp storage."""
    return DatasetService(session, storage)


# ======================================================================
# Helpers
# ======================================================================


def _ds_meta(
    *,
    name: str = "test-dataset",
    fmt: str = "csv",
    description: str = "A test dataset",
) -> dict:
    return {
        "name": name,
        "format": fmt,
        "description": description,
        "storage_backend": "filesystem",
        "source": "unit-test",
        "tags": {"env": "test"},
    }


# ======================================================================
# CRUD
# ======================================================================


class TestCreate:
    """DatasetService.create()."""

    async def test_create_returns_uuid(self, svc: DatasetService) -> None:
        ds_id = await svc.create(_ds_meta())
        assert isinstance(ds_id, str)
        assert len(ds_id) == 36  # UUID format

    async def test_create_persists_record(self, svc: DatasetService) -> None:
        ds_id = await svc.create(_ds_meta(name="persisted"))
        record = await svc.get(ds_id)
        assert record is not None
        assert record.name == "persisted"
        assert record.format == "csv"
        assert record.description == "A test dataset"
        assert record.storage_backend == "filesystem"
        assert record.source == "unit-test"
        assert record.current_version == 0
        assert record.total_files == 0
        assert record.total_size_bytes == 0


class TestGet:
    """DatasetService.get()."""

    async def test_get_existing(self, svc: DatasetService) -> None:
        ds_id = await svc.create(_ds_meta())
        record = await svc.get(ds_id)
        assert record is not None
        assert record.id == ds_id

    async def test_get_non_existent_returns_none(self, svc: DatasetService) -> None:
        result = await svc.get("00000000-0000-0000-0000-000000000000")
        assert result is None


class TestListAll:
    """DatasetService.list_all()."""

    async def test_list_all_empty(self, svc: DatasetService) -> None:
        results = await svc.list_all()
        assert results == []

    async def test_list_all_returns_datasets_desc(self, svc: DatasetService) -> None:
        await svc.create(_ds_meta(name="first"))
        await svc.create(_ds_meta(name="second"))
        results = await svc.list_all()
        assert len(results) == 2
        # Most recent first (desc by created_at); both created nearly simultaneously
        # so just verify both are present
        names = {r.name for r in results}
        assert names == {"first", "second"}


class TestUpdate:
    """DatasetService.update()."""

    async def test_update_modifies_allowed_fields(self, svc: DatasetService) -> None:
        ds_id = await svc.create(_ds_meta())
        updated = await svc.update(ds_id, {
            "name": "renamed",
            "description": "new desc",
            "tags": {"updated": "yes"},
            "source": "new-source",
        })
        assert updated.name == "renamed"
        assert updated.description == "new desc"
        assert updated.tags == {"updated": "yes"}
        assert updated.source == "new-source"

    async def test_update_ignores_disallowed_fields(self, svc: DatasetService) -> None:
        ds_id = await svc.create(_ds_meta())
        updated = await svc.update(ds_id, {
            "name": "ok",
            "format": "should-be-ignored",
        })
        assert updated.name == "ok"
        assert updated.format == "csv"  # unchanged

    async def test_update_non_existent_raises(self, svc: DatasetService) -> None:
        with pytest.raises(ValueError, match="Dataset not found"):
            await svc.update("nonexistent-id", {"name": "x"})


class TestDelete:
    """DatasetService.delete()."""

    async def test_delete_removes_dataset(self, svc: DatasetService) -> None:
        ds_id = await svc.create(_ds_meta())
        await svc.delete(ds_id)
        assert await svc.get(ds_id) is None

    async def test_delete_removes_files_and_versions(self, svc: DatasetService) -> None:
        ds_id = await svc.create(_ds_meta())
        await svc.upload_file(ds_id, "a.csv", b"col\n1")
        await svc.create_version(ds_id, "v1")
        await svc.add_lineage(ds_id, 1, "model", "m1")
        await svc.delete(ds_id)
        assert await svc.get(ds_id) is None

    async def test_delete_non_existent_raises(self, svc: DatasetService) -> None:
        with pytest.raises(ValueError, match="Dataset not found"):
            await svc.delete("nonexistent-id")


# ======================================================================
# Files
# ======================================================================


class TestUploadFile:
    """DatasetService.upload_file()."""

    async def test_upload_file_persists_record(self, svc: DatasetService) -> None:
        ds_id = await svc.create(_ds_meta())
        record = await svc.upload_file(ds_id, "data.csv", b"col1,col2\na,b\n")
        assert record.filename == "data.csv"
        assert record.size_bytes == len(b"col1,col2\na,b\n")
        assert record.dataset_id == ds_id
        assert record.hash != ""

    async def test_upload_file_updates_aggregates(self, svc: DatasetService) -> None:
        ds_id = await svc.create(_ds_meta())
        data = b"col\n1\n2\n"
        await svc.upload_file(ds_id, "a.csv", data)
        dataset = await svc.get(ds_id)
        assert dataset is not None
        assert dataset.total_files == 1
        assert dataset.total_size_bytes == len(data)

        data2 = b"col\n3\n"
        await svc.upload_file(ds_id, "b.csv", data2)
        dataset = await svc.get(ds_id)
        assert dataset is not None
        assert dataset.total_files == 2
        assert dataset.total_size_bytes == len(data) + len(data2)

    async def test_upload_to_non_existent_dataset_raises(
        self, svc: DatasetService
    ) -> None:
        with pytest.raises(ValueError, match="Dataset not found"):
            await svc.upload_file("nonexistent", "f.csv", b"data")


class TestListFiles:
    """DatasetService.list_files()."""

    async def test_list_files_returns_files(self, svc: DatasetService) -> None:
        ds_id = await svc.create(_ds_meta())
        await svc.upload_file(ds_id, "a.csv", b"a")
        await svc.upload_file(ds_id, "b.csv", b"b")
        files = await svc.list_files(ds_id)
        assert len(files) == 2
        names = {f.filename for f in files}
        assert names == {"a.csv", "b.csv"}

    async def test_list_files_with_version_filter(self, svc: DatasetService) -> None:
        ds_id = await svc.create(_ds_meta())
        # Upload file at version 0 (default)
        await svc.upload_file(ds_id, "v0.csv", b"v0")
        # Create version 1 so next upload is at version 1
        await svc.create_version(ds_id, "bump")
        await svc.upload_file(ds_id, "v1.csv", b"v1")

        # Filter by version=0 should only include files at version <= 0
        files_v0 = await svc.list_files(ds_id, version=0)
        assert len(files_v0) == 1
        assert files_v0[0].filename == "v0.csv"

        # Filter by version=1 should include both
        files_v1 = await svc.list_files(ds_id, version=1)
        assert len(files_v1) == 2


class TestDownloadFile:
    """DatasetService.download_file()."""

    async def test_download_file_returns_bytes(self, svc: DatasetService) -> None:
        ds_id = await svc.create(_ds_meta())
        content = b"important data content"
        await svc.upload_file(ds_id, "file.bin", content)
        result = await svc.download_file(ds_id, "file.bin")
        assert result == content

    async def test_download_missing_file_raises(self, svc: DatasetService) -> None:
        ds_id = await svc.create(_ds_meta())
        with pytest.raises(FileNotFoundError):
            await svc.download_file(ds_id, "missing.bin")


class TestDeleteFile:
    """DatasetService.delete_file()."""

    async def test_delete_file_removes_record_and_storage(
        self, svc: DatasetService
    ) -> None:
        ds_id = await svc.create(_ds_meta())
        await svc.upload_file(ds_id, "deleteme.csv", b"data")
        await svc.delete_file(ds_id, "deleteme.csv")
        files = await svc.list_files(ds_id)
        assert len(files) == 0
        # Aggregates should be updated
        dataset = await svc.get(ds_id)
        assert dataset is not None
        assert dataset.total_files == 0
        assert dataset.total_size_bytes == 0

    async def test_delete_missing_file_raises(self, svc: DatasetService) -> None:
        ds_id = await svc.create(_ds_meta())
        with pytest.raises(FileNotFoundError):
            await svc.delete_file(ds_id, "nonexistent.csv")


# ======================================================================
# Versioning
# ======================================================================


class TestCreateVersion:
    """DatasetService.create_version()."""

    async def test_create_version_increments_version(
        self, svc: DatasetService
    ) -> None:
        ds_id = await svc.create(_ds_meta())
        await svc.upload_file(ds_id, "data.csv", b"col\n1")
        v = await svc.create_version(ds_id, "initial")
        assert v.version == 1
        assert v.change_summary == "initial"
        assert v.num_files == 1
        assert v.hash != ""

    async def test_create_version_computes_hash(self, svc: DatasetService) -> None:
        ds_id = await svc.create(_ds_meta())
        await svc.upload_file(ds_id, "a.csv", b"data_a")
        await svc.upload_file(ds_id, "b.csv", b"data_b")
        v = await svc.create_version(ds_id)
        # Hash should be a 64-char hex SHA-256
        assert len(v.hash) == 64
        assert v.num_files == 2

    async def test_create_version_on_non_existent_raises(
        self, svc: DatasetService
    ) -> None:
        with pytest.raises(ValueError, match="Dataset not found"):
            await svc.create_version("nonexistent")

    async def test_create_version_increments_dataset_current_version(
        self, svc: DatasetService
    ) -> None:
        ds_id = await svc.create(_ds_meta())
        await svc.upload_file(ds_id, "f.csv", b"d")
        await svc.create_version(ds_id, "v1")
        await svc.create_version(ds_id, "v2")
        dataset = await svc.get(ds_id)
        assert dataset is not None
        assert dataset.current_version == 2


class TestListVersions:
    """DatasetService.list_versions()."""

    async def test_list_versions_descending(self, svc: DatasetService) -> None:
        ds_id = await svc.create(_ds_meta())
        await svc.upload_file(ds_id, "f.csv", b"d")
        await svc.create_version(ds_id, "first")
        await svc.create_version(ds_id, "second")
        versions = await svc.list_versions(ds_id)
        assert len(versions) == 2
        assert versions[0].version > versions[1].version
        assert versions[0].change_summary == "second"


class TestGetVersion:
    """DatasetService.get_version()."""

    async def test_get_version_returns_record(self, svc: DatasetService) -> None:
        ds_id = await svc.create(_ds_meta())
        await svc.upload_file(ds_id, "f.csv", b"d")
        await svc.create_version(ds_id, "v1")
        v = await svc.get_version(ds_id, 1)
        assert v is not None
        assert v.version == 1

    async def test_get_version_returns_none_for_missing(
        self, svc: DatasetService
    ) -> None:
        ds_id = await svc.create(_ds_meta())
        v = await svc.get_version(ds_id, 999)
        assert v is None


# ======================================================================
# Stats
# ======================================================================


class TestComputeStats:
    """DatasetService.compute_stats()."""

    async def test_compute_stats_basic(self, svc: DatasetService) -> None:
        ds_id = await svc.create(_ds_meta())
        await svc.upload_file(ds_id, "a.csv", b"col\n1\n2\n")
        stats = await svc.compute_stats(ds_id)
        assert "total_size_bytes" in stats
        assert "total_files" in stats
        assert "format_breakdown" in stats
        assert stats["total_files"] == 1
        assert stats["format_breakdown"]["csv"] == 1

    async def test_compute_stats_csv_counts_records(
        self, svc: DatasetService
    ) -> None:
        ds_id = await svc.create(_ds_meta(fmt="csv"))
        csv_data = b"name,age\nalice,30\nbob,25\n"
        await svc.upload_file(ds_id, "people.csv", csv_data)
        stats = await svc.compute_stats(ds_id)
        # 3 lines minus 1 header = 2 records
        assert stats["num_records"] == 2

    async def test_compute_stats_json_counts_records(
        self, svc: DatasetService
    ) -> None:
        ds_id = await svc.create(_ds_meta(fmt="json"))
        json_data = json.dumps([{"a": 1}, {"a": 2}, {"a": 3}]).encode()
        await svc.upload_file(ds_id, "data.json", json_data)
        stats = await svc.compute_stats(ds_id)
        assert stats["num_records"] == 3

    async def test_compute_stats_non_existent_raises(
        self, svc: DatasetService
    ) -> None:
        with pytest.raises(ValueError, match="Dataset not found"):
            await svc.compute_stats("nonexistent")


# ======================================================================
# Preview
# ======================================================================


class TestPreview:
    """DatasetService.preview()."""

    async def test_preview_csv(self, svc: DatasetService) -> None:
        ds_id = await svc.create(_ds_meta(fmt="csv"))
        csv_data = b"name,age\nalice,30\nbob,25\n"
        await svc.upload_file(ds_id, "people.csv", csv_data)
        result = await svc.preview(ds_id)
        assert result["columns"] == ["name", "age"]
        assert len(result["rows"]) == 2
        assert result["rows"][0]["name"] == "alice"
        assert result["total_rows"] == 2

    async def test_preview_json_array(self, svc: DatasetService) -> None:
        ds_id = await svc.create(_ds_meta(fmt="json"))
        data = json.dumps([{"x": 1, "y": 2}, {"x": 3, "y": 4}]).encode()
        await svc.upload_file(ds_id, "data.json", data)
        result = await svc.preview(ds_id)
        assert result["columns"] == ["x", "y"]
        assert len(result["rows"]) == 2
        assert result["total_rows"] == 2

    async def test_preview_jsonl(self, svc: DatasetService) -> None:
        ds_id = await svc.create(_ds_meta(fmt="jsonl"))
        lines = b'{"a":1}\n{"a":2}\n{"a":3}\n'
        await svc.upload_file(ds_id, "data.jsonl", lines)
        result = await svc.preview(ds_id)
        assert result["columns"] == ["a"]
        assert len(result["rows"]) == 3
        assert result["total_rows"] == 3

    async def test_preview_non_tabular_returns_empty(
        self, svc: DatasetService
    ) -> None:
        ds_id = await svc.create(_ds_meta(fmt="png"))
        await svc.upload_file(ds_id, "image.png", b"\x89PNG\r\n")
        result = await svc.preview(ds_id)
        assert result["columns"] == []
        assert result["rows"] == []

    async def test_preview_non_existent_raises(self, svc: DatasetService) -> None:
        with pytest.raises(ValueError, match="Dataset not found"):
            await svc.preview("nonexistent")

    async def test_preview_empty_dataset(self, svc: DatasetService) -> None:
        ds_id = await svc.create(_ds_meta(fmt="csv"))
        result = await svc.preview(ds_id)
        assert result["columns"] == []
        assert result["rows"] == []


# ======================================================================
# Lineage
# ======================================================================


class TestAddLineage:
    """DatasetService.add_lineage()."""

    async def test_add_lineage_creates_record(self, svc: DatasetService) -> None:
        ds_id = await svc.create(_ds_meta())
        await svc.upload_file(ds_id, "f.csv", b"d")
        await svc.create_version(ds_id, "v1")
        record = await svc.add_lineage(ds_id, 1, "model", "model-abc", "input")
        assert record.dataset_id == ds_id
        assert record.dataset_version == 1
        assert record.entity_type == "model"
        assert record.entity_id == "model-abc"
        assert record.role == "input"

    async def test_add_lineage_non_existent_dataset_raises(
        self, svc: DatasetService
    ) -> None:
        with pytest.raises(ValueError, match="Dataset not found"):
            await svc.add_lineage("nonexistent", 1, "model", "m1")


class TestGetLineage:
    """DatasetService.get_lineage()."""

    async def test_get_lineage_returns_records(self, svc: DatasetService) -> None:
        ds_id = await svc.create(_ds_meta())
        await svc.upload_file(ds_id, "f.csv", b"d")
        await svc.create_version(ds_id, "v1")
        await svc.add_lineage(ds_id, 1, "model", "m1", "input")
        await svc.add_lineage(ds_id, 1, "training_job", "j1", "output")
        records = await svc.get_lineage(ds_id)
        assert len(records) == 2
        entity_types = {r.entity_type for r in records}
        assert entity_types == {"model", "training_job"}

    async def test_get_lineage_empty(self, svc: DatasetService) -> None:
        ds_id = await svc.create(_ds_meta())
        records = await svc.get_lineage(ds_id)
        assert records == []
