"""Tests for MLRun integration in training dispatch + outcome writer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from artenic_ai_platform.db.models import Base, MLRun, TrainingJob
from artenic_ai_platform.providers.base import JobStatus
from artenic_ai_platform.providers.mock import MockProvider
from artenic_ai_platform.training.outcome_writer import OutcomeWriter
from artenic_ai_platform.training.service import TrainingManager

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncSession


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
async def session() -> AsyncGenerator[AsyncSession, None]:
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as sess:
        yield sess

    await engine.dispose()


@pytest.fixture
def manager(session: AsyncSession) -> TrainingManager:
    return TrainingManager(session, {"mock": MockProvider()})


@pytest.fixture
def writer(session: AsyncSession) -> OutcomeWriter:
    return OutcomeWriter(session)


# ======================================================================
# dispatch() creates MLRun
# ======================================================================


class TestDispatchCreatesMLRun:
    async def test_dispatch_creates_mlrun(
        self, session: AsyncSession, manager: TrainingManager
    ) -> None:
        job_id = await manager.dispatch("svc", "model", "mock")

        run_id = f"run:{job_id}"
        result = await session.execute(select(MLRun).where(MLRun.id == run_id))
        ml_run = result.scalar_one_or_none()

        assert ml_run is not None
        assert ml_run.status == "running"
        assert ml_run.triggered_by == "mock"
        assert ml_run.started_at is not None

    async def test_dispatch_mlrun_has_config(
        self, session: AsyncSession, manager: TrainingManager
    ) -> None:
        job_id = await manager.dispatch(
            "svc", "model", "mock", config={"lr": 0.01}
        )

        run_id = f"run:{job_id}"
        result = await session.execute(select(MLRun).where(MLRun.id == run_id))
        ml_run = result.scalar_one()
        assert ml_run.config == {"lr": 0.01}


# ======================================================================
# write_outcome() updates MLRun
# ======================================================================


class TestOutcomeUpdatesMLRun:
    async def test_outcome_updates_mlrun_completed(
        self,
        session: AsyncSession,
        manager: TrainingManager,
        writer: OutcomeWriter,
    ) -> None:
        job_id = await manager.dispatch("svc", "model", "mock")

        # Simulate job completion
        result = await session.execute(
            select(TrainingJob).where(TrainingJob.id == job_id)
        )
        job = result.scalar_one()
        job.status = JobStatus.COMPLETED.value
        job.metrics = {"accuracy": 0.95}
        await session.commit()

        # Write outcome
        outcome = await writer.write_outcome(job_id)
        assert outcome is not None

        # Verify MLRun was updated
        run_id = f"run:{job_id}"
        run_result = await session.execute(select(MLRun).where(MLRun.id == run_id))
        ml_run = run_result.scalar_one()
        assert ml_run.status == "completed"
        assert ml_run.metrics == {"accuracy": 0.95}

    async def test_outcome_updates_mlrun_failed(
        self,
        session: AsyncSession,
        manager: TrainingManager,
        writer: OutcomeWriter,
    ) -> None:
        job_id = await manager.dispatch("svc", "model", "mock")

        result = await session.execute(
            select(TrainingJob).where(TrainingJob.id == job_id)
        )
        job = result.scalar_one()
        job.status = JobStatus.FAILED.value
        await session.commit()

        await writer.write_outcome(job_id)

        run_id = f"run:{job_id}"
        run_result = await session.execute(select(MLRun).where(MLRun.id == run_id))
        ml_run = run_result.scalar_one()
        assert ml_run.status == "failed"
