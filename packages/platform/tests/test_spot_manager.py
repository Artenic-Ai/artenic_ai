"""Tests for artenic_ai_platform.training.spot_manager."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

from artenic_ai_platform.db.engine import (
    create_async_engine,
    create_session_factory,
    create_tables,
)
from artenic_ai_platform.db.models import TrainingJob
from artenic_ai_platform.providers.base import CloudJobStatus, JobStatus
from artenic_ai_platform.training.spot_manager import SpotManager

SQLITE_URL = "sqlite+aiosqlite://"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


async def _make_session():
    """Create an in-memory DB engine, create tables, return (engine, session)."""
    engine = create_async_engine(SQLITE_URL)
    await create_tables(engine)
    factory = create_session_factory(engine)
    session = factory()
    return engine, session


async def _insert_job(session, **overrides):
    """Insert a TrainingJob with sensible defaults, return the object."""
    defaults = {
        "id": str(uuid.uuid4()),
        "service": "sentiment",
        "model": "bert-base",
        "provider": "mock",
        "status": "running",
        "is_spot": True,
        "preemption_count": 0,
        "region": "us-central1",
        "instance_type": "n1-standard-4",
        "config": {},
    }
    defaults.update(overrides)
    job = TrainingJob(**defaults)
    session.add(job)
    await session.flush()
    return job


# ======================================================================
# Detection tests
# ======================================================================


class TestDetectPreemptionSpotPreemptedStatus:
    """detect_preemption returns True when cloud status is PREEMPTED."""

    async def test_detect_preemption_spot_preempted_status(self) -> None:
        engine, session = await _make_session()
        try:
            job = await _insert_job(session)
            cloud = CloudJobStatus(
                provider_job_id="pj-1",
                status=JobStatus.PREEMPTED,
            )
            mgr = SpotManager(session, {}, max_retries=3)
            assert mgr.detect_preemption(job, cloud) is True
        finally:
            await session.close()
            await engine.dispose()


class TestDetectPreemptionNotSpot:
    """detect_preemption returns False for non-spot jobs."""

    async def test_detect_preemption_not_spot(self) -> None:
        engine, session = await _make_session()
        try:
            job = await _insert_job(session, is_spot=False)
            cloud = CloudJobStatus(
                provider_job_id="pj-2",
                status=JobStatus.PREEMPTED,
            )
            mgr = SpotManager(session, {})
            assert mgr.detect_preemption(job, cloud) is False
        finally:
            await session.close()
            await engine.dispose()


class TestDetectPreemptionKeywordInError:
    """detect_preemption returns True when error contains preemption keyword."""

    async def test_detect_preemption_keyword_in_error(self) -> None:
        engine, session = await _make_session()
        try:
            job = await _insert_job(session)
            cloud = CloudJobStatus(
                provider_job_id="pj-3",
                status=JobStatus.FAILED,
                error="VM was preempted by the cloud provider",
            )
            mgr = SpotManager(session, {})
            assert mgr.detect_preemption(job, cloud) is True
        finally:
            await session.close()
            await engine.dispose()


class TestDetectPreemptionShortDuration:
    """detect_preemption returns True when duration is under 5 minutes."""

    async def test_detect_preemption_short_duration(self) -> None:
        engine, session = await _make_session()
        try:
            job = await _insert_job(session)
            cloud = CloudJobStatus(
                provider_job_id="pj-4",
                status=JobStatus.FAILED,
                duration_seconds=120.0,  # 2 min — well under 5 min
            )
            mgr = SpotManager(session, {})
            assert mgr.detect_preemption(job, cloud) is True
        finally:
            await session.close()
            await engine.dispose()


class TestDetectPreemptionNormalFailure:
    """detect_preemption returns False for a normal failure (long duration, no keywords)."""

    async def test_detect_preemption_normal_failure(self) -> None:
        engine, session = await _make_session()
        try:
            job = await _insert_job(session)
            cloud = CloudJobStatus(
                provider_job_id="pj-5",
                status=JobStatus.FAILED,
                error="Out of memory",
                duration_seconds=7200.0,  # 2 hours — normal
            )
            mgr = SpotManager(session, {})
            assert mgr.detect_preemption(job, cloud) is False
        finally:
            await session.close()
            await engine.dispose()


# ======================================================================
# handle_preemption tests
# ======================================================================


class TestHandlePreemptionRetry:
    """handle_preemption creates a retry job and returns its ID."""

    async def test_handle_preemption_retry(self) -> None:
        engine, session = await _make_session()
        try:
            mock_provider = AsyncMock()
            mock_provider.submit_job = AsyncMock(return_value="provider-new-123")

            job = await _insert_job(session, preemption_count=0)
            mgr = SpotManager(
                session,
                {"mock": mock_provider},
                max_retries=3,
            )

            new_id = await mgr.handle_preemption(job)
            assert new_id is not None

            # The retry job should exist in the DB
            retry = await session.get(TrainingJob, new_id)
            assert retry is not None
            assert retry.preemption_count == 1
            assert retry.resumed_from_job_id == job.id
            assert retry.status == JobStatus.RUNNING.value
        finally:
            await session.close()
            await engine.dispose()


class TestHandlePreemptionMaxRetries:
    """handle_preemption returns None when max retries exceeded."""

    async def test_handle_preemption_max_retries(self) -> None:
        engine, session = await _make_session()
        try:
            job = await _insert_job(session, preemption_count=3)
            mgr = SpotManager(session, {}, max_retries=3)

            result = await mgr.handle_preemption(job)
            assert result is None

            # Original job should be marked as preempted
            updated = await session.get(TrainingJob, job.id)
            assert updated is not None
            assert updated.preempted is True
            assert updated.status == JobStatus.PREEMPTED.value
        finally:
            await session.close()
            await engine.dispose()


class TestHandlePreemptionProviderSubmitFails:
    """When provider.submit_job raises, the retry job is marked as failed."""

    async def test_handle_preemption_provider_submit_fails(self) -> None:
        engine, session = await _make_session()
        try:
            mock_provider = AsyncMock()
            mock_provider.submit_job = AsyncMock(side_effect=RuntimeError("provider unavailable"))

            job = await _insert_job(session, preemption_count=0)
            mgr = SpotManager(
                session,
                {"mock": mock_provider},
                max_retries=3,
            )

            new_id = await mgr.handle_preemption(job)
            assert new_id is not None

            retry = await session.get(TrainingJob, new_id)
            assert retry is not None
            assert retry.status == JobStatus.FAILED.value
            assert "Failed to re-submit" in (retry.error or "")
        finally:
            await session.close()
            await engine.dispose()


class TestDetectPreemptionNonFailedStatus:
    """detect_preemption returns False for non-FAILED status (line 84)."""

    async def test_detect_preemption_non_failed_status(self) -> None:
        engine, session = await _make_session()
        try:
            job = await _insert_job(session)
            cloud = CloudJobStatus(
                provider_job_id="pj-running",
                status=JobStatus.RUNNING,
            )
            mgr = SpotManager(session, {})
            assert mgr.detect_preemption(job, cloud) is False
        finally:
            await session.close()
            await engine.dispose()


class TestHandlePreemptionProviderNotFound:
    """handle_preemption returns new_job_id when provider not found (lines 130-131)."""

    async def test_handle_preemption_provider_not_found(self) -> None:
        engine, session = await _make_session()
        try:
            job = await _insert_job(session, preemption_count=0)
            # Empty providers dict - provider won't be found
            mgr = SpotManager(
                session,
                {},  # no providers!
                max_retries=3,
            )

            new_id = await mgr.handle_preemption(job)
            assert new_id is not None

            # The retry job should exist but still in pending (provider wasn't found)
            retry = await session.get(TrainingJob, new_id)
            assert retry is not None
            assert retry.status == JobStatus.PENDING.value
            assert retry.resumed_from_job_id == job.id
        finally:
            await session.close()
            await engine.dispose()


# ======================================================================
# Failover region selection
# ======================================================================


class TestFailoverRegionSelection:
    """_select_failover_region cycles through configured regions."""

    async def test_failover_region_selection(self) -> None:
        engine, session = await _make_session()
        try:
            regions = ["us-east1", "us-west1", "eu-west1"]
            mgr = SpotManager(session, {}, max_retries=5, failover_regions=regions)

            # preemption_count=0 => idx=0 => "us-east1"
            job0 = await _insert_job(session, preemption_count=0, region="other")
            assert mgr._select_failover_region(job0) == "us-east1"

            # preemption_count=1 => idx=1 => "us-west1"
            job1 = await _insert_job(session, preemption_count=1, region="other")
            assert mgr._select_failover_region(job1) == "us-west1"

            # preemption_count=2 => idx=2 => "eu-west1"
            job2 = await _insert_job(session, preemption_count=2, region="other")
            assert mgr._select_failover_region(job2) == "eu-west1"

            # preemption_count=3 => idx=0 => wraps back to "us-east1"
            job3 = await _insert_job(session, preemption_count=3, region="other")
            assert mgr._select_failover_region(job3) == "us-east1"

            # When the selected region matches current region, skip to next
            job_skip = await _insert_job(session, preemption_count=0, region="us-east1")
            assert mgr._select_failover_region(job_skip) == "us-west1"
        finally:
            await session.close()
            await engine.dispose()
