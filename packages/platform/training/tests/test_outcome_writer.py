"""Tests for artenic_ai_platform_training.outcome_writer."""

from __future__ import annotations

import uuid

import pytest

from artenic_ai_platform.db.engine import (
    create_async_engine,
    create_session_factory,
    create_tables,
)
from artenic_ai_platform.db.models import TrainingJob
from artenic_ai_platform_providers.base import JobStatus
from artenic_ai_platform_training.outcome_writer import OutcomeWriter

SQLITE_URL = "sqlite+aiosqlite://"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


async def _make_session():
    """Create an in-memory DB, return (engine, session)."""
    engine = create_async_engine(SQLITE_URL)
    await create_tables(engine)
    factory = create_session_factory(engine)
    session = factory()
    return engine, session


async def _insert_job(session, **overrides):
    """Insert a TrainingJob with sensible defaults."""
    defaults = {
        "id": str(uuid.uuid4()),
        "service": "nlp",
        "model": "bert",
        "provider": "mock",
        "status": JobStatus.COMPLETED.value,
        "is_spot": False,
        "instance_type": "n1-standard-4",
        "region": "us-central1",
        "config": {},
    }
    defaults.update(overrides)
    job = TrainingJob(**defaults)
    session.add(job)
    await session.flush()
    return job


# ======================================================================
# write_outcome
# ======================================================================


class TestWriteOutcomeCompleted:
    """write_outcome creates an outcome record for a completed job."""

    async def test_write_outcome_completed(self) -> None:
        engine, session = await _make_session()
        try:
            job = await _insert_job(
                session,
                status=JobStatus.COMPLETED.value,
                duration_seconds=3600.0,
                cost_actual_eur=5.0,
            )
            await session.commit()

            writer = OutcomeWriter(session)
            result = await writer.write_outcome(job.id)

            assert result is not None
            assert result["job_id"] == job.id
            assert result["success"] is True
            assert result["actual_duration_hours"] == pytest.approx(1.0)
            assert result["actual_cost_eur"] == pytest.approx(5.0)
        finally:
            await session.close()
            await engine.dispose()


class TestWriteOutcomeFailed:
    """write_outcome creates an outcome with success=False for a failed job."""

    async def test_write_outcome_failed(self) -> None:
        engine, session = await _make_session()
        try:
            job = await _insert_job(
                session,
                status=JobStatus.FAILED.value,
                duration_seconds=1800.0,
                cost_actual_eur=2.5,
            )
            await session.commit()

            writer = OutcomeWriter(session)
            result = await writer.write_outcome(job.id)

            assert result is not None
            assert result["success"] is False
        finally:
            await session.close()
            await engine.dispose()


class TestWriteOutcomeNotFinalized:
    """write_outcome returns None when the job is still running."""

    async def test_write_outcome_not_finalized(self) -> None:
        engine, session = await _make_session()
        try:
            job = await _insert_job(session, status=JobStatus.RUNNING.value)
            await session.commit()

            writer = OutcomeWriter(session)
            result = await writer.write_outcome(job.id)

            assert result is None
        finally:
            await session.close()
            await engine.dispose()


class TestWriteOutcomeAlreadyExists:
    """write_outcome returns None when an outcome already exists (no duplicate)."""

    async def test_write_outcome_already_exists(self) -> None:
        engine, session = await _make_session()
        try:
            job = await _insert_job(
                session,
                status=JobStatus.COMPLETED.value,
                duration_seconds=3600.0,
                cost_actual_eur=5.0,
            )
            await session.commit()

            writer = OutcomeWriter(session)
            # First write succeeds
            first = await writer.write_outcome(job.id)
            assert first is not None

            # Second write returns None (already exists)
            second = await writer.write_outcome(job.id)
            assert second is None
        finally:
            await session.close()
            await engine.dispose()


class TestWriteOutcomeJobNotFound:
    """write_outcome returns None when the job doesn't exist."""

    async def test_write_outcome_job_not_found(self) -> None:
        engine, session = await _make_session()
        try:
            writer = OutcomeWriter(session)
            result = await writer.write_outcome("nonexistent-job-id")
            assert result is None
        finally:
            await session.close()
            await engine.dispose()


class TestWriteOutcomeExtractsMetrics:
    """Extracts metrics from job.metrics dict when primary_metric_after is None."""

    async def test_write_outcome_extracts_metrics(self) -> None:
        engine, session = await _make_session()
        try:
            job = await _insert_job(
                session,
                status=JobStatus.COMPLETED.value,
                duration_seconds=7200.0,
                cost_actual_eur=10.0,
                metrics={"accuracy": 0.92, "loss": 0.08},
                primary_metric_name=None,
                primary_metric_after=None,
            )
            await session.commit()

            writer = OutcomeWriter(session)
            result = await writer.write_outcome(job.id)

            assert result is not None
            # Should extract "accuracy" (first matching key)
            assert result["primary_metric_name"] == "accuracy"
            assert result["primary_metric_value"] == pytest.approx(0.92)
        finally:
            await session.close()
            await engine.dispose()


# ======================================================================
# list_outcomes
# ======================================================================


class TestWriteOutcomeDurationFromTimestamps:
    """Duration from started_at/completed_at when duration_seconds is None."""

    async def test_write_outcome_duration_from_timestamps(self) -> None:
        from datetime import UTC, datetime, timedelta

        engine, session = await _make_session()
        try:
            started = datetime.now(UTC) - timedelta(hours=2)
            completed = datetime.now(UTC)
            job = await _insert_job(
                session,
                status=JobStatus.COMPLETED.value,
                duration_seconds=None,
                cost_actual_eur=8.0,
                started_at=started,
                completed_at=completed,
            )
            await session.commit()

            writer = OutcomeWriter(session)
            result = await writer.write_outcome(job.id)

            assert result is not None
            # Duration should be ~2.0 hours (computed from timestamps)
            assert result["actual_duration_hours"] == pytest.approx(2.0, abs=0.01)
        finally:
            await session.close()
            await engine.dispose()


class TestListOutcomes:
    """list_outcomes supports pagination and filtering."""

    async def test_list_outcomes(self) -> None:
        engine, session = await _make_session()
        try:
            writer = OutcomeWriter(session)

            # Insert 3 completed jobs with outcomes
            for i in range(3):
                job = await _insert_job(
                    session,
                    status=JobStatus.COMPLETED.value,
                    duration_seconds=3600.0,
                    cost_actual_eur=float(i + 1),
                    provider="mock" if i < 2 else "other",
                )
                await session.commit()
                await writer.write_outcome(job.id)

            # Insert a failed job with outcome
            failed_job = await _insert_job(
                session,
                status=JobStatus.FAILED.value,
                duration_seconds=600.0,
                cost_actual_eur=0.5,
                provider="mock",
            )
            await session.commit()
            await writer.write_outcome(failed_job.id)

            # List all outcomes
            all_outcomes = await writer.list_outcomes()
            assert len(all_outcomes) == 4

            # Filter by provider
            mock_outcomes = await writer.list_outcomes(provider="mock")
            assert len(mock_outcomes) == 3

            # Filter success_only
            success_outcomes = await writer.list_outcomes(success_only=True)
            assert len(success_outcomes) == 3
            for o in success_outcomes:
                assert o["success"] is True

            # Pagination: limit=2
            page1 = await writer.list_outcomes(limit=2, offset=0)
            assert len(page1) == 2

            page2 = await writer.list_outcomes(limit=2, offset=2)
            assert len(page2) == 2
        finally:
            await session.close()
            await engine.dispose()
