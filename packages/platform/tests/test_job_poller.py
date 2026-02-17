"""Tests for artenic_ai_platform.training.job_poller — background polling loop."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from artenic_ai_platform.db.models import Base, TrainingJob
from artenic_ai_platform.providers.base import JobStatus, TrainingSpec
from artenic_ai_platform.providers.mock import MockProvider
from artenic_ai_platform.training.job_poller import JobPoller


@pytest.fixture
async def session_factory() -> async_sessionmaker[AsyncSession]:
    """Create an in-memory aiosqlite DB with all tables."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory  # type: ignore[misc]
    await engine.dispose()


class TestStartStop:
    async def test_start_stop(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        poller = JobPoller(
            session_factory=session_factory,
            providers={},
            interval_seconds=0.05,
        )
        poller.start()
        assert poller._running is True
        assert poller._task is not None

        await poller.stop()
        assert poller._running is False
        assert poller._task is None


class TestPollRunningJobCompleted:
    async def test_poll_running_job_completed(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        # Create a MockProvider and submit a job so it has a known job_id
        mock = MockProvider()
        spec = TrainingSpec(service="nlp", model="bert", provider="mock")
        provider_job_id = await mock.submit_job(spec)

        # Insert a running TrainingJob record in the DB
        job_id = uuid.uuid4().hex[:36]
        async with session_factory() as session:
            job = TrainingJob(
                id=job_id,
                service="nlp",
                model="bert",
                provider="mock",
                status=JobStatus.RUNNING.value,
                provider_job_id=provider_job_id,
                started_at=datetime.now(UTC),
            )
            session.add(job)
            await session.commit()

        # Create poller and run a single poll cycle
        poller = JobPoller(
            session_factory=session_factory,
            providers={"mock": mock},
            interval_seconds=999,  # we won't use the loop
        )
        await poller._poll_all_running()

        # Verify status was updated to completed
        async with session_factory() as session:
            result = await session.get(TrainingJob, job_id)
            assert result is not None
            assert result.status == JobStatus.COMPLETED.value
            assert result.completed_at is not None


class TestPollStuckJob:
    async def test_poll_stuck_job(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        mock = MockProvider()
        spec = TrainingSpec(service="nlp", model="bert", provider="mock")
        provider_job_id = await mock.submit_job(spec)

        job_id = uuid.uuid4().hex[:36]
        # started_at is 100 hours ago, exceeding default 48h max_runtime
        async with session_factory() as session:
            job = TrainingJob(
                id=job_id,
                service="nlp",
                model="bert",
                provider="mock",
                status=JobStatus.RUNNING.value,
                provider_job_id=provider_job_id,
                started_at=datetime.now(UTC) - timedelta(hours=100),
            )
            session.add(job)
            await session.commit()

        poller = JobPoller(
            session_factory=session_factory,
            providers={"mock": mock},
            interval_seconds=999,
            max_runtime_hours=48.0,
        )
        await poller._poll_all_running()

        async with session_factory() as session:
            result = await session.get(TrainingJob, job_id)
            assert result is not None
            assert result.status == JobStatus.FAILED.value
            assert result.error is not None
            assert "max runtime" in result.error.lower()
            assert result.completed_at is not None


class TestPollUnknownProvider:
    async def test_poll_unknown_provider(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        """A running job whose provider is not registered should be skipped gracefully."""
        job_id = uuid.uuid4().hex[:36]
        async with session_factory() as session:
            job = TrainingJob(
                id=job_id,
                service="nlp",
                model="bert",
                provider="nonexistent",
                status=JobStatus.RUNNING.value,
                provider_job_id="some-id",
                started_at=datetime.now(UTC),
            )
            session.add(job)
            await session.commit()

        poller = JobPoller(
            session_factory=session_factory,
            providers={},  # no providers registered
            interval_seconds=999,
        )
        # Should not raise
        await poller._poll_all_running()

        # Job status should remain unchanged
        async with session_factory() as session:
            result = await session.get(TrainingJob, job_id)
            assert result is not None
            assert result.status == JobStatus.RUNNING.value


class TestPollNoRunningJobs:
    async def test_poll_no_running_jobs(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        """When there are no running jobs, the poll should complete without errors."""
        # Insert a non-running job
        job_id = uuid.uuid4().hex[:36]
        async with session_factory() as session:
            job = TrainingJob(
                id=job_id,
                service="nlp",
                model="bert",
                provider="mock",
                status=JobStatus.COMPLETED.value,
            )
            session.add(job)
            await session.commit()

        poller = JobPoller(
            session_factory=session_factory,
            providers={"mock": MockProvider()},
            interval_seconds=999,
        )
        # Should not raise
        await poller._poll_all_running()

        # Completed job should remain completed
        async with session_factory() as session:
            result = await session.get(TrainingJob, job_id)
            assert result is not None
            assert result.status == JobStatus.COMPLETED.value


class TestStartAlreadyRunning:
    """Calling start() twice is a no-op (line 52)."""

    async def test_start_already_running(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        poller = JobPoller(
            session_factory=session_factory,
            providers={},
            interval_seconds=0.05,
        )
        poller.start()
        first_task = poller._task

        # Second start is a no-op
        poller.start()
        assert poller._task is first_task

        await poller.stop()


class TestPollLoopRunsAndHandlesErrors:
    """_poll_loop runs repeatedly and handles exceptions (lines 73-78)."""

    async def test_poll_loop_handles_exception(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        from unittest.mock import patch

        poller = JobPoller(
            session_factory=session_factory,
            providers={},
            interval_seconds=0.01,
        )

        call_count = 0
        original_running = True

        async def _failing_poll():
            nonlocal call_count, original_running
            call_count += 1
            if call_count >= 2:
                # Stop after 2 iterations
                poller._running = False
            raise RuntimeError("Simulated poll error")

        with patch.object(poller, "_poll_all_running", new=_failing_poll):
            poller._running = True
            await poller._poll_loop()

        # It ran at least twice despite errors
        assert call_count >= 2


class TestPollSingleError:
    """Exception in _poll_single is caught by _poll_all_running (lines 92-93)."""

    async def test_poll_single_error(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        from unittest.mock import AsyncMock

        # Provider that raises on poll_status
        bad_provider = AsyncMock()
        bad_provider.poll_status = AsyncMock(side_effect=RuntimeError("API down"))

        job_id = uuid.uuid4().hex[:36]
        async with session_factory() as session:
            job = TrainingJob(
                id=job_id,
                service="nlp",
                model="bert",
                provider="bad",
                status=JobStatus.RUNNING.value,
                provider_job_id="pj-xxx",
                started_at=datetime.now(UTC),
            )
            session.add(job)
            await session.commit()

        poller = JobPoller(
            session_factory=session_factory,
            providers={"bad": bad_provider},
            interval_seconds=999,
        )
        # Should not raise
        await poller._poll_all_running()

        # Job should still be running (error was caught)
        async with session_factory() as session:
            result = await session.get(TrainingJob, job_id)
            assert result is not None
            assert result.status == JobStatus.RUNNING.value


class TestPollNoChangeStatus:
    """When cloud status is same as DB status, no update (line 116)."""

    async def test_poll_no_change_status(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        from unittest.mock import AsyncMock

        from artenic_ai_platform.providers.base import CloudJobStatus

        # Provider returns RUNNING (same as DB)
        mock_provider = AsyncMock()
        mock_provider.poll_status = AsyncMock(
            return_value=CloudJobStatus(
                provider_job_id="pj-same",
                status=JobStatus.RUNNING,
            )
        )

        job_id = uuid.uuid4().hex[:36]
        async with session_factory() as session:
            job = TrainingJob(
                id=job_id,
                service="nlp",
                model="bert",
                provider="mock",
                status=JobStatus.RUNNING.value,
                provider_job_id="pj-same",
                started_at=datetime.now(UTC),
            )
            session.add(job)
            await session.commit()

        poller = JobPoller(
            session_factory=session_factory,
            providers={"mock": mock_provider},
            interval_seconds=999,
        )
        await poller._poll_all_running()

        # Status unchanged
        async with session_factory() as session:
            result = await session.get(TrainingJob, job_id)
            assert result is not None
            assert result.status == JobStatus.RUNNING.value


class TestPollWithErrorAndArtifacts:
    """Poll updates error and artifacts_uri fields (lines 123, 125)."""

    async def test_poll_with_error_and_artifacts(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        from unittest.mock import AsyncMock

        from artenic_ai_platform.providers.base import CloudJobStatus

        mock_provider = AsyncMock()
        mock_provider.poll_status = AsyncMock(
            return_value=CloudJobStatus(
                provider_job_id="pj-err",
                status=JobStatus.FAILED,
                error="OOM killed",
                artifacts_uri="s3://bucket/logs",
                cost_eur=12.5,
                duration_seconds=7200.0,
            )
        )

        job_id = uuid.uuid4().hex[:36]
        async with session_factory() as session:
            job = TrainingJob(
                id=job_id,
                service="nlp",
                model="bert",
                provider="mock",
                status=JobStatus.RUNNING.value,
                provider_job_id="pj-err",
                started_at=datetime.now(UTC),
            )
            session.add(job)
            await session.commit()

        poller = JobPoller(
            session_factory=session_factory,
            providers={"mock": mock_provider},
            interval_seconds=999,
        )
        await poller._poll_all_running()

        async with session_factory() as session:
            result = await session.get(TrainingJob, job_id)
            assert result is not None
            assert result.status == JobStatus.FAILED.value
            assert result.error == "OOM killed"
            assert result.artifacts_uri == "s3://bucket/logs"
            assert result.cost_actual_eur == 12.5
            assert result.duration_seconds == 7200.0
            assert result.completed_at is not None


class TestIsStuckNoStartedAt:
    """_is_stuck returns False when started_at is None (line 157)."""

    async def test_is_stuck_no_started_at(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        job_id = uuid.uuid4().hex[:36]
        async with session_factory() as session:
            job = TrainingJob(
                id=job_id,
                service="nlp",
                model="bert",
                provider="mock",
                status=JobStatus.RUNNING.value,
                started_at=None,
            )
            session.add(job)
            await session.flush()

            poller = JobPoller(
                session_factory=session_factory,
                providers={},
                interval_seconds=999,
            )
            assert poller._is_stuck(job) is False


class TestFinalizeStuckCleanupFails:
    """When provider.cancel_job fails during stuck finalization (lines 194-195)."""

    async def test_finalize_stuck_cleanup_fails(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        from unittest.mock import AsyncMock

        bad_provider = AsyncMock()
        bad_provider.cancel_job = AsyncMock(side_effect=RuntimeError("cleanup failed"))

        job_id = uuid.uuid4().hex[:36]
        async with session_factory() as session:
            job = TrainingJob(
                id=job_id,
                service="nlp",
                model="bert",
                provider="bad",
                status=JobStatus.RUNNING.value,
                provider_job_id="pj-stuck",
                started_at=datetime.now(UTC) - timedelta(hours=100),
            )
            session.add(job)
            await session.commit()

        poller = JobPoller(
            session_factory=session_factory,
            providers={"bad": bad_provider},
            interval_seconds=999,
            max_runtime_hours=48.0,
        )
        await poller._poll_all_running()

        # Job should still be marked as failed despite cleanup failure
        async with session_factory() as session:
            result = await session.get(TrainingJob, job_id)
            assert result is not None
            assert result.status == JobStatus.FAILED.value
            assert result.completed_at is not None
