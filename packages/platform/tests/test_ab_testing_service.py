"""Tests for artenic_ai_platform.ab_testing.service — ABTestManager."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from artenic_ai_platform.ab_testing.service import ABTestManager
from artenic_ai_platform.db.models import ABTestMetricRecord, Base

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


# ======================================================================
# Fixtures & helpers
# ======================================================================


@pytest.fixture
async def db() -> AsyncGenerator[AsyncSession, None]:
    """In-memory aiosqlite session for service tests."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session

    await engine.dispose()


def _variants() -> dict:
    """Two-variant configuration that sums to 100 %."""
    return {
        "control": {"model_id": "model-a", "traffic_pct": 50},
        "treatment": {"model_id": "model-b", "traffic_pct": 50},
    }


async def _create_running_test(
    db: AsyncSession,
    *,
    name: str = "test-1",
    service: str = "chat",
    event_bus: object | None = None,
) -> str:
    """Helper — create a running A/B test and return its ID."""
    mgr = ABTestManager(db, event_bus=event_bus)
    return await mgr.create_test(
        name=name,
        service=service,
        variants=_variants(),
        primary_metric="accuracy",
    )


# ======================================================================
# create_test
# ======================================================================


class TestCreateTest:
    async def test_create_test(self, db: AsyncSession) -> None:
        mgr = ABTestManager(db)
        test_id = await mgr.create_test(
            name="exp-1",
            service="chat",
            variants=_variants(),
            primary_metric="accuracy",
        )

        # Returns a UUID-like string.
        assert isinstance(test_id, str)
        assert len(test_id) == 36  # UUID4 format

        # Verify stored record is "running".
        result = await mgr.get_test(test_id)
        assert result["status"] == "running"
        assert result["name"] == "exp-1"
        assert result["service"] == "chat"

    async def test_create_test_validates_min_variants(self, db: AsyncSession) -> None:
        mgr = ABTestManager(db)
        with pytest.raises(ValueError, match="two variants"):
            await mgr.create_test(
                name="bad",
                service="s",
                variants={"only_one": {"model_id": "m", "traffic_pct": 100}},
                primary_metric="acc",
            )

    async def test_create_test_validates_model_id(self, db: AsyncSession) -> None:
        mgr = ABTestManager(db)
        with pytest.raises(ValueError, match="model_id"):
            await mgr.create_test(
                name="bad",
                service="s",
                variants={
                    "a": {"traffic_pct": 50},
                    "b": {"model_id": "m", "traffic_pct": 50},
                },
                primary_metric="acc",
            )

    async def test_create_test_validates_traffic_pct(self, db: AsyncSession) -> None:
        mgr = ABTestManager(db)
        with pytest.raises(ValueError, match="traffic_pct"):
            await mgr.create_test(
                name="bad",
                service="s",
                variants={
                    "a": {"model_id": "m"},
                    "b": {"model_id": "n", "traffic_pct": 50},
                },
                primary_metric="acc",
            )

    async def test_create_test_validates_pct_sum(self, db: AsyncSession) -> None:
        mgr = ABTestManager(db)
        with pytest.raises(ValueError, match="sum to 100"):
            await mgr.create_test(
                name="bad",
                service="s",
                variants={
                    "a": {"model_id": "m", "traffic_pct": 40},
                    "b": {"model_id": "n", "traffic_pct": 40},
                },
                primary_metric="acc",
            )


# ======================================================================
# select_variant
# ======================================================================


class TestSelectVariant:
    async def test_select_variant_running_test(self, db: AsyncSession) -> None:
        test_id = await _create_running_test(db)
        mgr = ABTestManager(db)

        result = await mgr.select_variant("chat")
        assert result is not None
        assert result["test_id"] == test_id
        assert result["variant_name"] in {"control", "treatment"}
        assert "model_id" in result

    async def test_select_variant_no_test(self, db: AsyncSession) -> None:
        mgr = ABTestManager(db)
        result = await mgr.select_variant("nonexistent-service")
        assert result is None

    async def test_select_variant_paused_test(self, db: AsyncSession) -> None:
        test_id = await _create_running_test(db)
        mgr = ABTestManager(db)

        # Pause the test so it should not be selected.
        await mgr.pause_test(test_id)

        result = await mgr.select_variant("chat")
        assert result is None


# ======================================================================
# record_metric
# ======================================================================


class TestRecordMetric:
    async def test_record_metric(self, db: AsyncSession) -> None:
        test_id = await _create_running_test(db)
        mgr = ABTestManager(db)

        await mgr.record_metric(test_id, "control", "accuracy", 0.95, latency_ms=42.0)

        # Verify the metric row was persisted.
        stmt = select(ABTestMetricRecord).where(ABTestMetricRecord.ab_test_id == test_id)
        result = await db.execute(stmt)
        rows = result.scalars().all()
        assert len(rows) == 1
        assert rows[0].variant_name == "control"
        assert rows[0].metric_name == "accuracy"
        assert rows[0].metric_value == pytest.approx(0.95)
        assert rows[0].latency_ms == pytest.approx(42.0)
        assert rows[0].error is False


# ======================================================================
# get_results
# ======================================================================


class TestGetResults:
    async def test_get_results_aggregation(self, db: AsyncSession) -> None:
        test_id = await _create_running_test(db)
        mgr = ABTestManager(db)

        # Record several metric values for the control variant.
        values = [0.90, 0.92, 0.94, 0.96, 0.98]
        for v in values:
            await mgr.record_metric(test_id, "control", "accuracy", v, latency_ms=10.0)

        results = await mgr.get_results(test_id)
        assert results["test_id"] == test_id
        assert results["status"] == "running"

        ctrl = results["variants"]["control"]
        acc = ctrl["accuracy"]
        assert acc["count"] == 5
        assert acc["mean"] == pytest.approx(sum(values) / len(values))
        assert acc["min"] == pytest.approx(0.90)
        assert acc["max"] == pytest.approx(0.98)
        assert acc["std"] >= 0
        assert ctrl["sample_count"] == 5
        assert ctrl["avg_latency_ms"] == pytest.approx(10.0)

    async def test_get_results_error_rate(self, db: AsyncSession) -> None:
        test_id = await _create_running_test(db)
        mgr = ABTestManager(db)

        # 4 successful + 1 error = 20 % error rate.
        for _ in range(4):
            await mgr.record_metric(test_id, "control", "accuracy", 0.9)
        await mgr.record_metric(test_id, "control", "accuracy", 0.0, error=True)

        results = await mgr.get_results(test_id)
        ctrl = results["variants"]["control"]
        assert ctrl["error_rate"] == pytest.approx(0.2)

    async def test_get_results_not_found(self, db: AsyncSession) -> None:
        mgr = ABTestManager(db)
        with pytest.raises(KeyError, match="not found"):
            await mgr.get_results("nonexistent-id")


# ======================================================================
# conclude_test
# ======================================================================


class TestConcludeTest:
    async def test_conclude_test(self, db: AsyncSession) -> None:
        test_id = await _create_running_test(db)
        mgr = ABTestManager(db)

        result = await mgr.conclude_test(test_id, winner="control", reason="higher acc")
        assert result["status"] == "concluded"
        assert result["winner"] == "control"
        assert result["conclusion_reason"] == "higher acc"
        assert result["concluded_at"] is not None

    async def test_conclude_test_not_found(self, db: AsyncSession) -> None:
        mgr = ABTestManager(db)
        with pytest.raises(KeyError, match="not found"):
            await mgr.conclude_test("does-not-exist")


# ======================================================================
# pause / resume
# ======================================================================


class TestPauseResume:
    async def test_pause_test(self, db: AsyncSession) -> None:
        test_id = await _create_running_test(db)
        mgr = ABTestManager(db)

        result = await mgr.pause_test(test_id)
        assert result["status"] == "paused"

    async def test_resume_test(self, db: AsyncSession) -> None:
        test_id = await _create_running_test(db)
        mgr = ABTestManager(db)

        await mgr.pause_test(test_id)
        result = await mgr.resume_test(test_id)
        assert result["status"] == "running"

    async def test_resume_not_paused(self, db: AsyncSession) -> None:
        test_id = await _create_running_test(db)
        mgr = ABTestManager(db)

        with pytest.raises(ValueError, match="Cannot resume"):
            await mgr.resume_test(test_id)


# ======================================================================
# get_test / list_tests
# ======================================================================


class TestGetTest:
    async def test_get_test(self, db: AsyncSession) -> None:
        test_id = await _create_running_test(db, name="my-test")
        mgr = ABTestManager(db)

        result = await mgr.get_test(test_id)
        assert result["id"] == test_id
        assert result["name"] == "my-test"
        assert result["service"] == "chat"
        assert result["variants"] == _variants()

    async def test_get_test_not_found(self, db: AsyncSession) -> None:
        mgr = ABTestManager(db)
        with pytest.raises(KeyError, match="not found"):
            await mgr.get_test("missing-id")


class TestListTests:
    async def test_list_tests(self, db: AsyncSession) -> None:
        mgr = ABTestManager(db)
        await _create_running_test(db, name="a", service="svc-1")
        await _create_running_test(db, name="b", service="svc-2")

        tests = await mgr.list_tests()
        assert len(tests) == 2

    async def test_list_tests_filter_service(self, db: AsyncSession) -> None:
        mgr = ABTestManager(db)
        await _create_running_test(db, name="a", service="svc-1")
        await _create_running_test(db, name="b", service="svc-2")

        tests = await mgr.list_tests(service="svc-1")
        assert len(tests) == 1
        assert tests[0]["service"] == "svc-1"

    async def test_list_tests_filter_status(self, db: AsyncSession) -> None:
        test_id = await _create_running_test(db, name="a", service="svc")
        mgr = ABTestManager(db)
        await mgr.pause_test(test_id)

        # Create a second test that stays running.
        await _create_running_test(db, name="b", service="svc")

        running = await mgr.list_tests(status="running")
        assert len(running) == 1
        assert running[0]["status"] == "running"

        paused = await mgr.list_tests(status="paused")
        assert len(paused) == 1
        assert paused[0]["status"] == "paused"


# ======================================================================
# Event publishing
# ======================================================================


class TestEventPublishing:
    async def test_create_publishes_event(self, db: AsyncSession) -> None:
        bus = MagicMock()
        mgr = ABTestManager(db, event_bus=bus)

        test_id = await mgr.create_test(
            name="evt-test",
            service="svc",
            variants=_variants(),
            primary_metric="accuracy",
        )

        bus.publish.assert_called_once()
        topic, payload = bus.publish.call_args.args
        assert topic == "ab_test"
        assert payload["action"] == "created"
        assert payload["test_id"] == test_id
        assert payload["name"] == "evt-test"
        assert payload["service"] == "svc"


# ======================================================================
# conclude_test publishes event
# ======================================================================


class TestConcludePublishesEvent:
    """conclude_test publishes an event when event_bus is provided."""

    async def test_conclude_publishes_event(self, db: AsyncSession) -> None:
        bus = MagicMock()
        mgr = ABTestManager(db, event_bus=bus)
        test_id = await mgr.create_test(
            name="t",
            service="svc",
            variants=_variants(),
            primary_metric="acc",
        )
        # Reset the mock so we only see the conclude event.
        bus.reset_mock()

        await mgr.conclude_test(test_id, winner="control")

        bus.publish.assert_called_once()
        topic, payload = bus.publish.call_args.args
        assert topic == "ab_test"
        assert payload["action"] == "concluded"
        assert payload["test_id"] == test_id
        assert payload["winner"] == "control"


# ======================================================================
# pause_test not found
# ======================================================================


class TestPauseNotFound:
    """pause_test raises KeyError for nonexistent test_id."""

    async def test_pause_not_found(self, db: AsyncSession) -> None:
        mgr = ABTestManager(db)
        with pytest.raises(KeyError, match="not found"):
            await mgr.pause_test("nonexistent")


# ======================================================================
# resume_test not found
# ======================================================================


class TestResumeNotFound:
    """resume_test raises KeyError for nonexistent test_id."""

    async def test_resume_not_found(self, db: AsyncSession) -> None:
        mgr = ABTestManager(db)
        with pytest.raises(KeyError, match="not found"):
            await mgr.resume_test("nonexistent")
