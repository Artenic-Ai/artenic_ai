"""Tests for artenic_ai_platform.ensemble.service — PlatformEnsembleManager."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from artenic_ai_platform.db.models import Base, EnsembleVersionRecord
from artenic_ai_platform.ensemble.service import PlatformEnsembleManager

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
async def session() -> AsyncGenerator[AsyncSession, None]:
    """In-memory aiosqlite session for ensemble service tests."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as sess:
        yield sess

    await engine.dispose()


async def _create_ensemble(
    session: AsyncSession,
    *,
    name: str = "weighted",
    service: str = "sentiment",
    strategy: str = "weighted",
    model_ids: list[str] | None = None,
) -> str:
    """Helper to create a standard ensemble and return its ID."""
    mgr = PlatformEnsembleManager(session)
    return await mgr.create_ensemble(
        name=name,
        service=service,
        strategy=strategy,
        model_ids=model_ids or ["model-a", "model-b"],
    )


# ======================================================================
# create_ensemble
# ======================================================================


class TestCreateEnsemble:
    async def test_create_ensemble(self, session: AsyncSession) -> None:
        """create_ensemble returns an ID in '{service}_{name}_v1' format."""
        ensemble_id = await _create_ensemble(session)
        assert ensemble_id == "sentiment_weighted_v1"

    async def test_create_ensemble_initial_version(self, session: AsyncSession) -> None:
        """An EnsembleVersionRecord with version=1 is created."""
        ensemble_id = await _create_ensemble(session)

        stmt = select(EnsembleVersionRecord).where(EnsembleVersionRecord.ensemble_id == ensemble_id)
        result = await session.execute(stmt)
        versions = result.scalars().all()

        assert len(versions) == 1
        assert versions[0].version == 1
        assert versions[0].change_reason == "Initial creation"


# ======================================================================
# get_ensemble
# ======================================================================


class TestGetEnsemble:
    async def test_get_ensemble(self, session: AsyncSession) -> None:
        """get_ensemble returns a dict with all expected fields."""
        ensemble_id = await _create_ensemble(session)
        mgr = PlatformEnsembleManager(session)
        result = await mgr.get_ensemble(ensemble_id)

        assert result["id"] == ensemble_id
        assert result["name"] == "weighted"
        assert result["service"] == "sentiment"
        assert result["strategy"] == "weighted"
        assert result["model_ids"] == ["model-a", "model-b"]
        assert result["version"] == 1
        assert result["stage"] == "registered"
        assert result["enabled"] is True

    async def test_get_ensemble_not_found(self, session: AsyncSession) -> None:
        """get_ensemble raises KeyError for nonexistent ensemble."""
        mgr = PlatformEnsembleManager(session)

        with pytest.raises(KeyError, match="not-there"):
            await mgr.get_ensemble("not-there")


# ======================================================================
# list_ensembles
# ======================================================================


class TestListEnsembles:
    async def test_list_ensembles(self, session: AsyncSession) -> None:
        """list_ensembles returns all created ensembles."""
        await _create_ensemble(session, name="a", service="svc1")
        await _create_ensemble(session, name="b", service="svc2")

        mgr = PlatformEnsembleManager(session)
        result = await mgr.list_ensembles()

        assert len(result) == 2

    async def test_list_ensembles_filter_service(self, session: AsyncSession) -> None:
        """list_ensembles filters by service when provided."""
        await _create_ensemble(session, name="a", service="svc1")
        await _create_ensemble(session, name="b", service="svc2")

        mgr = PlatformEnsembleManager(session)
        result = await mgr.list_ensembles(service="svc1")

        assert len(result) == 1
        assert result[0]["service"] == "svc1"


# ======================================================================
# update_ensemble
# ======================================================================


class TestUpdateEnsemble:
    async def test_update_ensemble_bumps_version(self, session: AsyncSession) -> None:
        """update_ensemble increments the version number."""
        ensemble_id = await _create_ensemble(session)
        mgr = PlatformEnsembleManager(session)

        updated = await mgr.update_ensemble(
            ensemble_id,
            model_ids=["model-a", "model-b", "model-c"],
            change_reason="Added model-c",
        )

        assert updated["version"] == 2

    async def test_update_ensemble_creates_version_record(self, session: AsyncSession) -> None:
        """update_ensemble creates a new EnsembleVersionRecord."""
        ensemble_id = await _create_ensemble(session)
        mgr = PlatformEnsembleManager(session)

        await mgr.update_ensemble(
            ensemble_id,
            model_ids=["model-x"],
            change_reason="Replaced models",
        )

        stmt = (
            select(EnsembleVersionRecord)
            .where(EnsembleVersionRecord.ensemble_id == ensemble_id)
            .order_by(EnsembleVersionRecord.version.desc())
        )
        result = await session.execute(stmt)
        versions = result.scalars().all()

        assert len(versions) == 2
        assert versions[0].version == 2
        assert versions[0].change_reason == "Replaced models"
        assert versions[0].model_ids == ["model-x"]

    async def test_update_ensemble_name(self, session: AsyncSession) -> None:
        """update_ensemble can change the name field."""
        ensemble_id = await _create_ensemble(session)
        mgr = PlatformEnsembleManager(session)

        updated = await mgr.update_ensemble(
            ensemble_id,
            name="renamed-ensemble",
            change_reason="Renaming",
        )

        assert updated["name"] == "renamed-ensemble"

    async def test_update_ensemble_not_found(self, session: AsyncSession) -> None:
        """update_ensemble raises KeyError for nonexistent ensemble."""
        mgr = PlatformEnsembleManager(session)

        with pytest.raises(KeyError, match="ghost"):
            await mgr.update_ensemble("ghost", model_ids=["m"])


# ======================================================================
# dispatch_ensemble_training
# ======================================================================


class TestDispatchEnsembleTraining:
    async def test_dispatch_ensemble_training(self, session: AsyncSession) -> None:
        """dispatch creates an EnsembleJobRecord and returns a job_id."""
        ensemble_id = await _create_ensemble(session)
        mgr = PlatformEnsembleManager(session)

        job_id = await mgr.dispatch_ensemble_training(ensemble_id, provider="mock")

        assert isinstance(job_id, str)
        assert len(job_id) == 36  # UUID

    async def test_dispatch_training_not_found(self, session: AsyncSession) -> None:
        """dispatch raises KeyError for nonexistent ensemble."""
        mgr = PlatformEnsembleManager(session)

        with pytest.raises(KeyError, match="missing"):
            await mgr.dispatch_ensemble_training("missing", provider="mock")


# ======================================================================
# get_ensemble_job_status
# ======================================================================


class TestGetEnsembleJobStatus:
    async def test_get_ensemble_job_status(self, session: AsyncSession) -> None:
        """get_ensemble_job_status returns a job dict."""
        ensemble_id = await _create_ensemble(session)
        mgr = PlatformEnsembleManager(session)
        job_id = await mgr.dispatch_ensemble_training(ensemble_id, provider="mock")

        job = await mgr.get_ensemble_job_status(job_id)

        assert job["id"] == job_id
        assert job["ensemble_id"] == ensemble_id
        assert job["status"] == "pending"
        assert job["total_models"] == 2

    async def test_get_job_status_not_found(self, session: AsyncSession) -> None:
        """get_ensemble_job_status raises KeyError for unknown job."""
        mgr = PlatformEnsembleManager(session)

        with pytest.raises(KeyError, match="no-such-job"):
            await mgr.get_ensemble_job_status("no-such-job")


# ======================================================================
# get_version_history
# ======================================================================


class TestGetVersionHistory:
    async def test_get_version_history(self, session: AsyncSession) -> None:
        """Version history is ordered newest-first."""
        ensemble_id = await _create_ensemble(session)
        mgr = PlatformEnsembleManager(session)

        await mgr.update_ensemble(ensemble_id, model_ids=["m1"], change_reason="v2")
        await mgr.update_ensemble(ensemble_id, model_ids=["m2"], change_reason="v3")

        history = await mgr.get_version_history(ensemble_id)

        assert len(history) == 3
        # Newest first
        assert history[0]["version"] == 3
        assert history[1]["version"] == 2
        assert history[2]["version"] == 1


# ======================================================================
# event_bus integration
# ======================================================================


class TestUpdatePublishesEvent:
    async def test_update_publishes_event(self, session: AsyncSession) -> None:
        """update_ensemble publishes an 'ensemble' event via event_bus."""
        bus = MagicMock()
        bus.publish = MagicMock()

        ensemble_id = await _create_ensemble(session)
        mgr = PlatformEnsembleManager(session, event_bus=bus)

        await mgr.update_ensemble(
            ensemble_id,
            model_ids=["model-z"],
            change_reason="swap",
        )

        bus.publish.assert_called_once()
        topic = bus.publish.call_args[0][0]
        assert topic == "ensemble"
        payload = bus.publish.call_args[0][1]
        assert payload["action"] == "updated"
        assert payload["ensemble_id"] == ensemble_id
        assert payload["version"] == 2
        assert payload["change_reason"] == "swap"


# ======================================================================
# list_ensembles — stage filter
# ======================================================================


class TestListEnsemblesFilterStage:
    """list_ensembles filters by stage when provided."""

    async def test_list_filter_stage(self, session: AsyncSession) -> None:
        mgr = PlatformEnsembleManager(session)
        await mgr.create_ensemble("e1", "svc", "weighted", ["m1"])

        results = await mgr.list_ensembles(stage="registered")
        assert len(results) == 1
        assert results[0]["stage"] == "registered"

    async def test_list_filter_stage_no_match(self, session: AsyncSession) -> None:
        mgr = PlatformEnsembleManager(session)
        await mgr.create_ensemble("e1", "svc", "weighted", ["m1"])

        results = await mgr.list_ensembles(stage="production")
        assert len(results) == 0


# ======================================================================
# update_ensemble — individual optional fields
# ======================================================================


class TestUpdateEnsembleAllFields:
    """update_ensemble updates strategy, strategy_config, and description."""

    async def test_update_all_optional_fields(self, session: AsyncSession) -> None:
        mgr = PlatformEnsembleManager(session)
        eid = await mgr.create_ensemble("e", "svc", "weighted", ["m1"])

        result = await mgr.update_ensemble(
            eid,
            strategy="round_robin",
            strategy_config={"key": "val"},
            description="updated desc",
        )

        assert result["strategy"] == "round_robin"
        assert result["description"] == "updated desc"
