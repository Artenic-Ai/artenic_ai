"""Tests for artenic_ai_platform.entities.base_service — GenericEntityService."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: TC002

from artenic_ai_platform.db.engine import (
    create_async_engine,
    create_session_factory,
    create_tables,
)
from artenic_ai_platform.db.models import MLDataset
from artenic_ai_platform.entities.base_service import GenericEntityService


class _DatasetSvc(GenericEntityService[MLDataset]):
    _model_class = MLDataset


@pytest.fixture
async def session() -> AsyncSession:  # type: ignore[misc]
    engine = create_async_engine("sqlite+aiosqlite://")
    await create_tables(engine)
    factory = create_session_factory(engine)
    async with factory() as s:
        yield s  # type: ignore[misc]
    await engine.dispose()


@pytest.fixture
def svc(session: AsyncSession) -> _DatasetSvc:
    return _DatasetSvc(session)


# ======================================================================
# create
# ======================================================================


class TestCreate:
    async def test_create_returns_record(self, svc: _DatasetSvc) -> None:
        record = await svc.create(
            "ds:test:v1",
            {
                "name": "test",
                "version": 1,
                "format": "csv",
            },
        )
        assert record.id == "ds:test:v1"
        assert record.name == "test"
        assert record.format == "csv"

    async def test_create_with_metadata(self, svc: _DatasetSvc) -> None:
        record = await svc.create(
            "ds:meta:v1",
            {
                "name": "meta",
                "version": 1,
                "format": "json",
                "metadata_": {"source": "api"},
            },
        )
        assert record.metadata_ == {"source": "api"}


# ======================================================================
# get / get_or_raise
# ======================================================================


class TestGet:
    async def test_get_existing(self, svc: _DatasetSvc) -> None:
        await svc.create("ds:x:v1", {"name": "x", "version": 1, "format": "csv"})
        record = await svc.get("ds:x:v1")
        assert record is not None
        assert record.name == "x"

    async def test_get_missing_returns_none(self, svc: _DatasetSvc) -> None:
        result = await svc.get("nonexistent")
        assert result is None

    async def test_get_or_raise_existing(self, svc: _DatasetSvc) -> None:
        await svc.create("ds:y:v1", {"name": "y", "version": 1, "format": "csv"})
        record = await svc.get_or_raise("ds:y:v1")
        assert record.name == "y"

    async def test_get_or_raise_missing(self, svc: _DatasetSvc) -> None:
        with pytest.raises(ValueError, match="Not found"):
            await svc.get_or_raise("nonexistent")


# ======================================================================
# list_all
# ======================================================================


class TestListAll:
    async def test_list_empty(self, svc: _DatasetSvc) -> None:
        results = await svc.list_all()
        assert results == []

    async def test_list_with_records(self, svc: _DatasetSvc) -> None:
        await svc.create("ds:a:v1", {"name": "a", "version": 1, "format": "csv"})
        await svc.create("ds:b:v1", {"name": "b", "version": 1, "format": "csv"})
        results = await svc.list_all()
        assert len(results) == 2

    async def test_list_with_pagination(self, svc: _DatasetSvc) -> None:
        for i in range(5):
            await svc.create(f"ds:{i}:v1", {"name": f"d{i}", "version": 1, "format": "csv"})
        page = await svc.list_all(offset=2, limit=2)
        assert len(page) == 2

    async def test_list_with_filter(self, svc: _DatasetSvc) -> None:
        await svc.create("ds:f1:v1", {"name": "f1", "version": 1, "format": "csv"})
        await svc.create("ds:f2:v1", {"name": "f2", "version": 1, "format": "json"})
        results = await svc.list_all(filters={"format": "json"})
        assert len(results) == 1
        assert results[0].format == "json"

    async def test_list_ignores_unknown_filter(self, svc: _DatasetSvc) -> None:
        await svc.create("ds:ig:v1", {"name": "ig", "version": 1, "format": "csv"})
        results = await svc.list_all(filters={"nonexistent_col": "x"})
        assert len(results) == 1

    async def test_list_limit_clamped(self, svc: _DatasetSvc) -> None:
        await svc.create("ds:c:v1", {"name": "c", "version": 1, "format": "csv"})
        results = await svc.list_all(limit=0)
        assert len(results) == 1  # min clamp to 1

    async def test_list_offset_clamped(self, svc: _DatasetSvc) -> None:
        await svc.create("ds:oc:v1", {"name": "oc", "version": 1, "format": "csv"})
        results = await svc.list_all(offset=-5)
        assert len(results) == 1  # clamp to 0

    async def test_list_filter_none_value_skipped(self, svc: _DatasetSvc) -> None:
        await svc.create("ds:n:v1", {"name": "n", "version": 1, "format": "csv"})
        results = await svc.list_all(filters={"status": None})
        assert len(results) == 1


# ======================================================================
# update
# ======================================================================


class TestUpdate:
    async def test_update_field(self, svc: _DatasetSvc) -> None:
        await svc.create("ds:u:v1", {"name": "u", "version": 1, "format": "csv"})
        record = await svc.update("ds:u:v1", {"description": "updated"})
        assert record.description == "updated"

    async def test_update_ignores_id(self, svc: _DatasetSvc) -> None:
        await svc.create("ds:uid:v1", {"name": "uid", "version": 1, "format": "csv"})
        record = await svc.update("ds:uid:v1", {"id": "hacked"})
        assert record.id == "ds:uid:v1"

    async def test_update_missing_raises(self, svc: _DatasetSvc) -> None:
        with pytest.raises(ValueError, match="Not found"):
            await svc.update("nonexistent", {"description": "x"})


# ======================================================================
# delete
# ======================================================================


class TestDelete:
    async def test_delete_existing(self, svc: _DatasetSvc) -> None:
        await svc.create("ds:d:v1", {"name": "d", "version": 1, "format": "csv"})
        await svc.delete("ds:d:v1")
        assert await svc.get("ds:d:v1") is None

    async def test_delete_missing_raises(self, svc: _DatasetSvc) -> None:
        with pytest.raises(ValueError, match="Not found"):
            await svc.delete("nonexistent")


# ======================================================================
# next_version
# ======================================================================


class TestNextVersion:
    async def test_first_version(self, svc: _DatasetSvc) -> None:
        version = await svc.next_version("new-dataset")
        assert version == 1

    async def test_increments_version(self, svc: _DatasetSvc) -> None:
        await svc.create("ds:nv:v1", {"name": "versioned", "version": 1, "format": "csv"})
        await svc.create("ds:nv:v2", {"name": "versioned", "version": 2, "format": "csv"})
        version = await svc.next_version("versioned")
        assert version == 3
