"""Tests for artenic_ai_platform.health.monitor — HealthMonitor."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from artenic_ai_platform.db.models import Base, ModelHealthRecord, RegisteredModel
from artenic_ai_platform.health.monitor import HealthMonitor

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
async def db_factory() -> ...:
    """In-memory SQLite database with all tables created."""
    engine: AsyncEngine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


async def _seed_registered_model(
    factory: async_sessionmaker,  # type: ignore[type-arg]
    model_id: str,
) -> None:
    """Insert a minimal RegisteredModel row to satisfy the FK constraint."""
    async with factory() as session:
        session.add(
            RegisteredModel(
                id=model_id,
                name=model_id,
                version="1",
                model_type="test",
                framework="test",
            )
        )
        await session.commit()


# ======================================================================
# record_inference
# ======================================================================


class TestRecordInferenceBuffers:
    """record_inference appends observations to the internal buffer."""

    def test_record_inference_buffers(self) -> None:
        monitor = HealthMonitor(session_factory=None)  # type: ignore[arg-type]
        monitor.record_inference("model-a", latency_ms=10.0)
        monitor.record_inference("model-a", latency_ms=20.0)
        monitor.record_inference("model-a", latency_ms=30.0)

        assert len(monitor._observations["model-a"]) == 3


class TestRecordInferenceCapsBuffer:
    """Buffer is capped at 1000 entries per model."""

    def test_record_inference_caps_buffer(self) -> None:
        monitor = HealthMonitor(session_factory=None)  # type: ignore[arg-type]
        for i in range(1005):
            monitor.record_inference("model-b", latency_ms=float(i))

        assert len(monitor._observations["model-b"]) == 1000


# ======================================================================
# compute_health_snapshot
# ======================================================================


class TestComputeSnapshotEmpty:
    """Snapshot with no observations returns healthy with 0 sample_count."""

    def test_compute_snapshot_empty(self) -> None:
        monitor = HealthMonitor(session_factory=None)  # type: ignore[arg-type]
        snap = monitor.compute_health_snapshot("model-x")

        assert snap["status"] == "healthy"
        assert snap["sample_count"] == 0
        assert snap["model_id"] == "model-x"
        assert snap["error_rate"] == 0.0


class TestComputeSnapshotHealthy:
    """Low error rate and good confidence yields status=healthy."""

    def test_compute_snapshot_healthy(self) -> None:
        monitor = HealthMonitor(session_factory=None)  # type: ignore[arg-type]
        # 100 inferences, 0 errors, confidence close to 1.0
        for _ in range(100):
            monitor.record_inference("model-h", latency_ms=50.0, confidence=0.95)

        snap = monitor.compute_health_snapshot("model-h")

        assert snap["status"] == "healthy"
        assert snap["sample_count"] == 100
        assert snap["error_rate"] == 0.0
        assert snap["avg_confidence"] == pytest.approx(0.95)


class TestComputeSnapshotDegraded:
    """~20% error rate yields status=degraded."""

    def test_compute_snapshot_degraded(self) -> None:
        monitor = HealthMonitor(session_factory=None)  # type: ignore[arg-type]
        # 80 successes, 20 errors => 20% error rate
        for _ in range(80):
            monitor.record_inference("model-d", latency_ms=50.0, error=False)
        for _ in range(20):
            monitor.record_inference("model-d", latency_ms=50.0, error=True)

        snap = monitor.compute_health_snapshot("model-d")

        assert snap["status"] == "degraded"
        assert snap["error_rate"] == pytest.approx(0.2)
        assert snap["sample_count"] == 100


class TestComputeSnapshotUnhealthy:
    """>50% error rate yields status=unhealthy."""

    def test_compute_snapshot_unhealthy(self) -> None:
        monitor = HealthMonitor(session_factory=None)  # type: ignore[arg-type]
        # 40 successes, 60 errors => 60% error rate
        for _ in range(40):
            monitor.record_inference("model-u", latency_ms=50.0, error=False)
        for _ in range(60):
            monitor.record_inference("model-u", latency_ms=50.0, error=True)

        snap = monitor.compute_health_snapshot("model-u")

        assert snap["status"] == "unhealthy"
        assert snap["error_rate"] == pytest.approx(0.6)


class TestComputeSnapshotConfidenceDrift:
    """Avg confidence=0.8 gives drift=0.2 (>0.1 threshold), yielding degraded."""

    def test_compute_snapshot_confidence_drift(self) -> None:
        monitor = HealthMonitor(
            session_factory=None,  # type: ignore[arg-type]
            drift_threshold=0.1,
        )
        # All confidence values at 0.8 => avg=0.8, drift=|0.8-1.0|=0.2
        # error_rate=0.0 so the first condition fails only on drift
        for _ in range(50):
            monitor.record_inference("model-c", latency_ms=30.0, confidence=0.8)

        snap = monitor.compute_health_snapshot("model-c")

        assert snap["status"] == "degraded"
        assert snap["confidence_drift"] == pytest.approx(0.2)
        assert snap["avg_confidence"] == pytest.approx(0.8)
        assert snap["error_rate"] == 0.0


class TestComputeSnapshotP99Latency:
    """p99 latency is computed correctly for 100 observations."""

    def test_compute_snapshot_p99_latency(self) -> None:
        monitor = HealthMonitor(session_factory=None)  # type: ignore[arg-type]
        # Record latencies 1.0, 2.0, ..., 100.0
        for i in range(1, 101):
            monitor.record_inference("model-p", latency_ms=float(i))

        snap = monitor.compute_health_snapshot("model-p")

        # p99_index = max(0, int(100 * 0.99) - 1) = max(0, 99 - 1) = 98
        # sorted_latencies[98] = 99.0
        assert snap["p99_latency_ms"] == pytest.approx(99.0)
        assert snap["sample_count"] == 100
        assert snap["avg_latency_ms"] == pytest.approx(50.5)


# ======================================================================
# check_all_models (async — real DB)
# ======================================================================


class TestCheckAllModelsPersists:
    """check_all_models persists a ModelHealthRecord to the database."""

    async def test_check_all_models_persists(
        self,
        db_factory: async_sessionmaker,  # type: ignore[type-arg]
    ) -> None:
        model_id = "persist-model"
        await _seed_registered_model(db_factory, model_id)

        monitor = HealthMonitor(db_factory)
        monitor.record_inference(model_id, latency_ms=42.0, confidence=0.95)
        monitor.record_inference(model_id, latency_ms=43.0, confidence=0.96)

        snapshots = await monitor.check_all_models()

        assert len(snapshots) == 1
        assert snapshots[0]["model_id"] == model_id

        # Verify the row was persisted.
        async with db_factory() as session:
            result = await session.execute(
                select(ModelHealthRecord).where(ModelHealthRecord.model_id == model_id)
            )
            rows = result.scalars().all()

        assert len(rows) == 1
        assert rows[0].model_id == model_id
        assert rows[0].metric_name == "health_snapshot"
        assert rows[0].sample_count == 2


class TestCheckAllModelsPublishesAlert:
    """EventBus.publish is called when a model is degraded."""

    async def test_check_all_models_publishes_alert(
        self,
        db_factory: async_sessionmaker,  # type: ignore[type-arg]
    ) -> None:
        model_id = "alert-model"
        await _seed_registered_model(db_factory, model_id)

        mock_bus = MagicMock()
        mock_bus.publish = MagicMock()

        monitor = HealthMonitor(db_factory, event_bus=mock_bus)
        # Create a degraded model: ~20% error rate
        for _ in range(80):
            monitor.record_inference(model_id, latency_ms=10.0, error=False)
        for _ in range(20):
            monitor.record_inference(model_id, latency_ms=10.0, error=True)

        await monitor.check_all_models()

        mock_bus.publish.assert_called_once()
        call_args = mock_bus.publish.call_args
        assert call_args[0][0] == "health"
        assert call_args[0][1]["status"] == "degraded"
        assert call_args[0][1]["model_id"] == model_id


# ======================================================================
# get_health_history (async — real DB)
# ======================================================================


class TestGetHealthHistory:
    """get_health_history returns persisted ModelHealthRecord rows."""

    async def test_get_health_history(
        self,
        db_factory: async_sessionmaker,  # type: ignore[type-arg]
    ) -> None:
        model_id = "history-model"
        await _seed_registered_model(db_factory, model_id)

        # Manually insert some rows.
        async with db_factory() as session:
            for i in range(3):
                session.add(
                    ModelHealthRecord(
                        model_id=model_id,
                        metric_name="health_snapshot",
                        metric_value=float(i) * 0.1,
                        drift_score=0.01,
                        alert_triggered=False,
                        sample_count=10 + i,
                    )
                )
            await session.commit()

        monitor = HealthMonitor(db_factory)
        history = await monitor.get_health_history(model_id, limit=20)

        assert len(history) == 3
        # Should be ordered by created_at descending — all have same
        # server-default timestamp so at minimum we check all are present.
        model_ids = {h["model_id"] for h in history}
        assert model_ids == {model_id}
        sample_counts = sorted(h["sample_count"] for h in history)
        assert sample_counts == [10, 11, 12]


# ======================================================================
# start / stop
# ======================================================================


class TestStartStop:
    """start() sets _running and stop() resets state."""

    async def test_start_stop(self) -> None:
        monitor = HealthMonitor(
            session_factory=None,  # type: ignore[arg-type]
            check_interval_seconds=999.0,
        )

        monitor.start()
        assert monitor._running is True
        assert monitor._task is not None

        monitor.stop()
        assert monitor._running is False
        assert monitor._task is None


# ======================================================================
# _monitor_loop
# ======================================================================


class TestMonitorLoopRuns:
    """The background loop calls check_all_models periodically."""

    async def test_monitor_loop_runs(self) -> None:
        monitor = HealthMonitor(
            session_factory=None,  # type: ignore[arg-type]
            check_interval_seconds=0.05,
        )

        with patch.object(
            HealthMonitor,
            "check_all_models",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_check:
            monitor.start()
            # Give the loop enough time for at least one iteration.
            await asyncio.sleep(0.15)
            monitor.stop()

        assert mock_check.call_count >= 1


# ======================================================================
# start() already running
# ======================================================================


class TestStartAlreadyRunning:
    """Calling start() when already running should be a no-op."""

    def test_start_already_running(self) -> None:
        monitor = HealthMonitor(
            session_factory=None,  # type: ignore[arg-type]
            check_interval_seconds=999,
        )
        monitor._running = True
        # start() should detect _running=True, log a warning, and return
        # without creating a new task.
        monitor.start()
        assert monitor._task is None  # no new task created


# ======================================================================
# check_all_models — DB exception during persist
# ======================================================================


class TestCheckAllModelsDbError:
    """check_all_models logs but does not raise when DB persist fails."""

    async def test_check_all_models_db_error(
        self,
        db_factory: async_sessionmaker,  # type: ignore[type-arg]
    ) -> None:
        bad_factory = MagicMock()
        bad_session = AsyncMock()
        bad_session.__aenter__ = AsyncMock(side_effect=RuntimeError("DB down"))
        bad_factory.return_value = bad_session

        monitor = HealthMonitor(bad_factory, check_interval_seconds=999)
        monitor.record_inference("m1", latency_ms=10.0)

        # Should not raise — error is caught and logged.
        results = await monitor.check_all_models()
        assert len(results) == 1  # snapshot computed even though DB persist failed


# ======================================================================
# _monitor_loop — error handling & loop exit
# ======================================================================


class TestMonitorLoopError:
    """_monitor_loop catches errors and continues; logs on exit."""

    async def test_monitor_loop_handles_error(self) -> None:
        monitor = HealthMonitor(
            session_factory=None,  # type: ignore[arg-type]
            check_interval_seconds=0.01,
        )

        call_count = 0

        async def _bad_check(self_inner: object) -> list:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("boom")
            return []

        with patch.object(HealthMonitor, "check_all_models", _bad_check):
            monitor._running = True
            task = asyncio.create_task(monitor._monitor_loop())
            await asyncio.sleep(0.08)
            monitor._running = False
            await asyncio.sleep(0.03)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        # The loop should have called check_all_models at least twice
        # (once erroring, once succeeding) before we stopped it.
        assert call_count >= 2
