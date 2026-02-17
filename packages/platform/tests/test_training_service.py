"""Tests for artenic_ai_platform.training.service — 100% coverage."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from artenic_ai_platform.db.models import Base, TrainingJob
from artenic_ai_platform.providers.base import CloudJobStatus, JobStatus
from artenic_ai_platform.providers.mock import MockProvider
from artenic_ai_platform.training.service import TrainingManager

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
async def session() -> AsyncGenerator[AsyncSession, None]:
    """In-memory aiosqlite session for service tests."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as sess:
        yield sess

    await engine.dispose()


@pytest.fixture
def mock_provider() -> MockProvider:
    """A fresh MockProvider instance."""
    return MockProvider()


@pytest.fixture
def providers(mock_provider: MockProvider) -> dict[str, MockProvider]:
    """Provider registry with one mock provider."""
    return {"mock": mock_provider}


@pytest.fixture
def manager(
    session: AsyncSession,
    providers: dict[str, MockProvider],
) -> TrainingManager:
    """TrainingManager wired to the in-memory DB and mock provider."""
    return TrainingManager(session, providers)


async def _dispatch_default(
    mgr: TrainingManager,
    *,
    service: str = "nlp",
    model: str = "bert",
    provider: str = "mock",
    config: dict | None = None,
    instance_type: str | None = None,
    region: str | None = None,
    is_spot: bool = False,
    max_runtime_hours: float = 24.0,
) -> str:
    """Helper to dispatch a job with sensible defaults."""
    return await mgr.dispatch(
        service=service,
        model=model,
        provider=provider,
        config=config,
        instance_type=instance_type,
        region=region,
        is_spot=is_spot,
        max_runtime_hours=max_runtime_hours,
    )


# ======================================================================
# TestDispatch
# ======================================================================


class TestDispatch:
    async def test_dispatch_success(
        self,
        session: AsyncSession,
        manager: TrainingManager,
    ) -> None:
        """Dispatches with MockProvider, returns job_id UUID, DB has running job."""
        job_id = await _dispatch_default(manager)

        # Returned value is a valid UUID
        uuid.UUID(job_id)

        # DB row exists and is running
        row = await session.get(TrainingJob, job_id)
        assert row is not None
        assert row.status == JobStatus.RUNNING.value
        assert row.service == "nlp"
        assert row.model == "bert"
        assert row.provider == "mock"
        assert row.provider_job_id is not None
        assert row.provider_job_id.startswith("mock-")
        assert row.started_at is not None

    async def test_dispatch_with_cost_estimation(
        self,
        session: AsyncSession,
        manager: TrainingManager,
    ) -> None:
        """Dispatch with instance_type populates cost_estimate_eur."""
        job_id = await _dispatch_default(
            manager,
            instance_type="mock-gpu-a100",
            max_runtime_hours=10.0,
        )

        row = await session.get(TrainingJob, job_id)
        assert row is not None
        # mock-gpu-a100 on-demand: 2.50 EUR/hr * 10 hrs = 25.00
        assert row.cost_estimate_eur == pytest.approx(25.0)
        assert row.cost_per_hour_eur == pytest.approx(2.50)

    async def test_dispatch_with_spot_cost_estimation(
        self,
        session: AsyncSession,
        manager: TrainingManager,
    ) -> None:
        """Dispatch with is_spot=True uses spot pricing."""
        job_id = await _dispatch_default(
            manager,
            instance_type="mock-gpu-a100",
            is_spot=True,
            max_runtime_hours=10.0,
        )

        row = await session.get(TrainingJob, job_id)
        assert row is not None
        # mock-gpu-a100 spot: 0.75 EUR/hr * 10 hrs = 7.50
        assert row.cost_estimate_eur == pytest.approx(7.50)
        assert row.cost_per_hour_eur == pytest.approx(0.75)

    async def test_dispatch_unknown_provider(
        self,
        manager: TrainingManager,
    ) -> None:
        """Raises ValueError for an unknown provider."""
        with pytest.raises(ValueError, match="Unknown provider 'nonexistent'"):
            await _dispatch_default(manager, provider="nonexistent")

    async def test_dispatch_provider_submit_fails(
        self,
        session: AsyncSession,
        providers: dict[str, MockProvider],
    ) -> None:
        """If provider.submit_job raises, job is marked failed in DB and re-raised."""
        failing_provider = AsyncMock()
        failing_provider.submit_job = AsyncMock(
            side_effect=RuntimeError("GPU quota exceeded"),
        )
        failing_provider.list_instance_types = AsyncMock(return_value=[])
        providers["failing"] = failing_provider

        mgr = TrainingManager(session, providers)

        with pytest.raises(RuntimeError, match="GPU quota exceeded"):
            await mgr.dispatch(
                service="nlp",
                model="bert",
                provider="failing",
            )

        # Verify a failed job row exists in DB
        result = await session.execute(select(TrainingJob).where(TrainingJob.provider == "failing"))
        row = result.scalar_one()
        assert row.status == JobStatus.FAILED.value
        assert "GPU quota exceeded" in (row.error or "")

    async def test_dispatch_with_budget_check(
        self,
        session: AsyncSession,
        providers: dict[str, MockProvider],
    ) -> None:
        """Budget manager is called with estimated cost when provided."""
        budget_mock = AsyncMock()
        budget_mock.check_budget = AsyncMock()

        mgr = TrainingManager(session, providers, budget_manager=budget_mock)
        await mgr.dispatch(
            service="nlp",
            model="bert",
            provider="mock",
            instance_type="mock-gpu-a100",
            max_runtime_hours=10.0,
        )

        budget_mock.check_budget.assert_awaited_once_with(
            scope="service",
            scope_value="nlp",
            estimated_cost=pytest.approx(25.0),
        )

    async def test_dispatch_cost_estimation_skipped_on_exception(
        self,
        session: AsyncSession,
    ) -> None:
        """Cost estimation failure is silently skipped (debug log only)."""
        broken_provider = AsyncMock()
        broken_provider.list_instance_types = AsyncMock(
            side_effect=RuntimeError("API down"),
        )
        broken_provider.submit_job = AsyncMock(return_value="prov-123")

        mgr = TrainingManager(session, {"broken": broken_provider})
        job_id = await mgr.dispatch(
            service="nlp",
            model="bert",
            provider="broken",
            instance_type="gpu-xl",
        )

        row = await session.get(TrainingJob, job_id)
        assert row is not None
        assert row.cost_estimate_eur is None
        assert row.cost_per_hour_eur is None

    async def test_dispatch_instance_type_not_found(
        self,
        session: AsyncSession,
        manager: TrainingManager,
    ) -> None:
        """When instance_type does not match any listed, cost remains None."""
        job_id = await _dispatch_default(
            manager,
            instance_type="nonexistent-instance",
        )

        row = await session.get(TrainingJob, job_id)
        assert row is not None
        assert row.cost_estimate_eur is None

    async def test_dispatch_budget_not_called_without_cost(
        self,
        session: AsyncSession,
        providers: dict[str, MockProvider],
    ) -> None:
        """Budget manager not called when cost_estimate is None."""
        budget_mock = AsyncMock()
        budget_mock.check_budget = AsyncMock()

        mgr = TrainingManager(session, providers, budget_manager=budget_mock)
        # No instance_type means no cost estimate
        await mgr.dispatch(
            service="nlp",
            model="bert",
            provider="mock",
        )

        budget_mock.check_budget.assert_not_awaited()


# ======================================================================
# TestGetStatus
# ======================================================================


class TestGetStatus:
    async def test_get_status_pending_job(
        self,
        session: AsyncSession,
        manager: TrainingManager,
    ) -> None:
        """Returns a job dict for a pending job (no live poll needed)."""
        # Insert a pending job directly
        job_id = str(uuid.uuid4())
        job = TrainingJob(
            id=job_id,
            service="nlp",
            model="bert",
            provider="mock",
            config={},
            status=JobStatus.PENDING.value,
        )
        session.add(job)
        await session.commit()

        result = await manager.get_status(job_id)
        assert result["id"] == job_id
        assert result["status"] == "pending"
        assert result["service"] == "nlp"

    async def test_get_status_running_job_live_poll(
        self,
        session: AsyncSession,
        manager: TrainingManager,
        mock_provider: MockProvider,
    ) -> None:
        """Live poll updates status for a running job."""
        # Dispatch creates a running job
        job_id = await _dispatch_default(manager)

        # The mock provider marks jobs as completed immediately, so
        # poll_status will return COMPLETED status.
        result = await manager.get_status(job_id)

        # The live poll should have updated status to completed
        assert result["status"] == JobStatus.COMPLETED.value

    async def test_get_status_completed_job(
        self,
        session: AsyncSession,
        manager: TrainingManager,
    ) -> None:
        """Completed job does not trigger a live poll."""
        job_id = str(uuid.uuid4())
        job = TrainingJob(
            id=job_id,
            service="nlp",
            model="bert",
            provider="mock",
            config={},
            status=JobStatus.COMPLETED.value,
            provider_job_id="mock-abc",
        )
        session.add(job)
        await session.commit()

        result = await manager.get_status(job_id)
        assert result["status"] == "completed"

    async def test_get_status_not_found(
        self,
        manager: TrainingManager,
    ) -> None:
        """Raises KeyError when job does not exist."""
        with pytest.raises(KeyError, match="not found"):
            await manager.get_status("nonexistent-id")

    async def test_get_status_live_poll_fails(
        self,
        session: AsyncSession,
    ) -> None:
        """When live poll fails, warns but returns DB status."""
        failing_provider = AsyncMock()
        failing_provider.poll_status = AsyncMock(
            side_effect=RuntimeError("Connection refused"),
        )

        mgr = TrainingManager(session, {"broken": failing_provider})

        job_id = str(uuid.uuid4())
        job = TrainingJob(
            id=job_id,
            service="nlp",
            model="bert",
            provider="broken",
            config={},
            status=JobStatus.RUNNING.value,
            provider_job_id="prov-xyz",
        )
        session.add(job)
        await session.commit()

        result = await mgr.get_status(job_id)
        # Should still return the DB status despite poll failure
        assert result["status"] == "running"
        assert result["id"] == job_id

    async def test_get_status_live_poll_updates_metrics(
        self,
        session: AsyncSession,
    ) -> None:
        """Live poll that returns metrics/artifacts updates DB."""
        cloud_status = CloudJobStatus(
            provider_job_id="prov-1",
            status=JobStatus.COMPLETED,
            metrics={"accuracy": 0.99},
            artifacts_uri="s3://bucket/artifacts",
            cost_eur=12.5,
            duration_seconds=3600.0,
            error=None,
        )
        poll_provider = AsyncMock()
        poll_provider.poll_status = AsyncMock(return_value=cloud_status)

        mgr = TrainingManager(session, {"test": poll_provider})

        job_id = str(uuid.uuid4())
        job = TrainingJob(
            id=job_id,
            service="nlp",
            model="bert",
            provider="test",
            config={},
            status=JobStatus.RUNNING.value,
            provider_job_id="prov-1",
        )
        session.add(job)
        await session.commit()

        result = await mgr.get_status(job_id)
        assert result["status"] == "completed"
        assert result["metrics"] == {"accuracy": 0.99}
        assert result["artifacts_uri"] == "s3://bucket/artifacts"
        assert result["cost_actual_eur"] == pytest.approx(12.5)
        assert result["duration_seconds"] == pytest.approx(3600.0)
        assert result["completed_at"] is not None

    async def test_get_status_live_poll_same_status_no_update(
        self,
        session: AsyncSession,
    ) -> None:
        """Live poll returning the same status does not update DB."""
        cloud_status = CloudJobStatus(
            provider_job_id="prov-1",
            status=JobStatus.RUNNING,
        )
        poll_provider = AsyncMock()
        poll_provider.poll_status = AsyncMock(return_value=cloud_status)

        mgr = TrainingManager(session, {"test": poll_provider})

        job_id = str(uuid.uuid4())
        job = TrainingJob(
            id=job_id,
            service="nlp",
            model="bert",
            provider="test",
            config={},
            status=JobStatus.RUNNING.value,
            provider_job_id="prov-1",
        )
        session.add(job)
        await session.commit()

        result = await mgr.get_status(job_id)
        assert result["status"] == "running"

    async def test_get_status_live_poll_with_error(
        self,
        session: AsyncSession,
    ) -> None:
        """Live poll returning FAILED status with error updates DB."""
        cloud_status = CloudJobStatus(
            provider_job_id="prov-1",
            status=JobStatus.FAILED,
            error="OOM killed",
            cost_eur=5.0,
            duration_seconds=600.0,
        )
        poll_provider = AsyncMock()
        poll_provider.poll_status = AsyncMock(return_value=cloud_status)

        mgr = TrainingManager(session, {"test": poll_provider})

        job_id = str(uuid.uuid4())
        job = TrainingJob(
            id=job_id,
            service="nlp",
            model="bert",
            provider="test",
            config={},
            status=JobStatus.RUNNING.value,
            provider_job_id="prov-1",
        )
        session.add(job)
        await session.commit()

        result = await mgr.get_status(job_id)
        assert result["status"] == "failed"
        assert result["error"] == "OOM killed"

    async def test_get_status_running_no_provider_job_id(
        self,
        session: AsyncSession,
        manager: TrainingManager,
    ) -> None:
        """Running job without provider_job_id skips live poll."""
        job_id = str(uuid.uuid4())
        job = TrainingJob(
            id=job_id,
            service="nlp",
            model="bert",
            provider="mock",
            config={},
            status=JobStatus.RUNNING.value,
            provider_job_id=None,
        )
        session.add(job)
        await session.commit()

        result = await manager.get_status(job_id)
        assert result["status"] == "running"

    async def test_get_status_running_unknown_provider(
        self,
        session: AsyncSession,
        manager: TrainingManager,
    ) -> None:
        """Running job with unknown provider skips live poll."""
        job_id = str(uuid.uuid4())
        job = TrainingJob(
            id=job_id,
            service="nlp",
            model="bert",
            provider="unknown_prov",
            config={},
            status=JobStatus.RUNNING.value,
            provider_job_id="prov-123",
        )
        session.add(job)
        await session.commit()

        result = await manager.get_status(job_id)
        assert result["status"] == "running"


# ======================================================================
# TestCancel
# ======================================================================


class TestCancel:
    async def test_cancel_running_job(
        self,
        session: AsyncSession,
        manager: TrainingManager,
    ) -> None:
        """Cancel a running job marks it as cancelled."""
        job_id = await _dispatch_default(manager)

        result = await manager.cancel(job_id)
        assert result["status"] == JobStatus.CANCELLED.value
        assert result["completed_at"] is not None

    async def test_cancel_pending_job(
        self,
        session: AsyncSession,
        manager: TrainingManager,
    ) -> None:
        """Cancel a pending job marks it as cancelled."""
        job_id = str(uuid.uuid4())
        job = TrainingJob(
            id=job_id,
            service="nlp",
            model="bert",
            provider="mock",
            config={},
            status=JobStatus.PENDING.value,
        )
        session.add(job)
        await session.commit()

        result = await manager.cancel(job_id)
        assert result["status"] == "cancelled"

    async def test_cancel_completed_job(
        self,
        session: AsyncSession,
        manager: TrainingManager,
    ) -> None:
        """Raises ValueError when trying to cancel a completed job."""
        job_id = str(uuid.uuid4())
        job = TrainingJob(
            id=job_id,
            service="nlp",
            model="bert",
            provider="mock",
            config={},
            status=JobStatus.COMPLETED.value,
        )
        session.add(job)
        await session.commit()

        with pytest.raises(ValueError, match="Cannot cancel job"):
            await manager.cancel(job_id)

    async def test_cancel_not_found(
        self,
        manager: TrainingManager,
    ) -> None:
        """Raises KeyError when job does not exist."""
        with pytest.raises(KeyError, match="not found"):
            await manager.cancel("nonexistent-id")

    async def test_cancel_provider_fails(
        self,
        session: AsyncSession,
    ) -> None:
        """Provider cancel failure is logged, but job still marked cancelled."""
        failing_provider = AsyncMock()
        failing_provider.cancel_job = AsyncMock(
            side_effect=RuntimeError("Provider unreachable"),
        )

        mgr = TrainingManager(session, {"broken": failing_provider})

        job_id = str(uuid.uuid4())
        job = TrainingJob(
            id=job_id,
            service="nlp",
            model="bert",
            provider="broken",
            config={},
            status=JobStatus.RUNNING.value,
            provider_job_id="prov-xyz",
        )
        session.add(job)
        await session.commit()

        result = await mgr.cancel(job_id)
        assert result["status"] == "cancelled"
        assert result["completed_at"] is not None

    async def test_cancel_failed_job(
        self,
        session: AsyncSession,
        manager: TrainingManager,
    ) -> None:
        """Raises ValueError when trying to cancel a failed job."""
        job_id = str(uuid.uuid4())
        job = TrainingJob(
            id=job_id,
            service="nlp",
            model="bert",
            provider="mock",
            config={},
            status=JobStatus.FAILED.value,
        )
        session.add(job)
        await session.commit()

        with pytest.raises(ValueError, match="Cannot cancel job"):
            await manager.cancel(job_id)

    async def test_cancel_pending_no_provider_job_id(
        self,
        session: AsyncSession,
        manager: TrainingManager,
    ) -> None:
        """Cancel a pending job without provider_job_id works (no provider call)."""
        job_id = str(uuid.uuid4())
        job = TrainingJob(
            id=job_id,
            service="nlp",
            model="bert",
            provider="mock",
            config={},
            status=JobStatus.PENDING.value,
            provider_job_id=None,
        )
        session.add(job)
        await session.commit()

        result = await manager.cancel(job_id)
        assert result["status"] == "cancelled"


# ======================================================================
# TestListJobs
# ======================================================================


class TestListJobs:
    async def test_list_empty(
        self,
        manager: TrainingManager,
    ) -> None:
        """Returns empty list when no jobs exist."""
        result = await manager.list_jobs()
        assert result == []

    async def test_list_with_jobs(
        self,
        session: AsyncSession,
        manager: TrainingManager,
    ) -> None:
        """Returns list of job dicts."""
        await _dispatch_default(manager, service="svc-a", model="m1")
        await _dispatch_default(manager, service="svc-b", model="m2")

        result = await manager.list_jobs()
        assert len(result) == 2
        services = {j["service"] for j in result}
        assert services == {"svc-a", "svc-b"}

    async def test_list_filter_by_service(
        self,
        session: AsyncSession,
        manager: TrainingManager,
    ) -> None:
        """Filters jobs by service name."""
        await _dispatch_default(manager, service="nlp")
        await _dispatch_default(manager, service="vision")

        result = await manager.list_jobs(service="nlp")
        assert len(result) == 1
        assert result[0]["service"] == "nlp"

    async def test_list_filter_by_provider(
        self,
        session: AsyncSession,
        manager: TrainingManager,
    ) -> None:
        """Filters jobs by provider."""
        await _dispatch_default(manager)

        # Only mock provider jobs exist
        result = await manager.list_jobs(provider="mock")
        assert len(result) == 1
        assert result[0]["provider"] == "mock"

        # No jobs for non-existent provider
        result = await manager.list_jobs(provider="aws")
        assert result == []

    async def test_list_filter_by_status(
        self,
        session: AsyncSession,
        manager: TrainingManager,
    ) -> None:
        """Filters jobs by status."""
        await _dispatch_default(manager)

        # Dispatched jobs are running
        result = await manager.list_jobs(status="running")
        assert len(result) == 1

        result = await manager.list_jobs(status="pending")
        assert result == []

    async def test_list_pagination(
        self,
        session: AsyncSession,
        manager: TrainingManager,
    ) -> None:
        """Limit and offset control pagination."""
        for i in range(5):
            await _dispatch_default(manager, service=f"svc-{i}")

        # First page
        page1 = await manager.list_jobs(limit=2, offset=0)
        assert len(page1) == 2

        # Second page
        page2 = await manager.list_jobs(limit=2, offset=2)
        assert len(page2) == 2

        # Third page (only 1 left)
        page3 = await manager.list_jobs(limit=2, offset=4)
        assert len(page3) == 1

        # All IDs are unique
        all_ids = {j["id"] for j in page1 + page2 + page3}
        assert len(all_ids) == 5
