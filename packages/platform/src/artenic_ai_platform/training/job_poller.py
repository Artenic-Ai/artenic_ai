"""Background polling loop for running training jobs."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import select, update

from artenic_ai_platform.db.models import TrainingJob
from artenic_ai_platform.providers.base import JobStatus

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from artenic_ai_platform.providers.base import TrainingProvider

logger = logging.getLogger(__name__)


class JobPoller:
    """Periodically polls cloud providers for job status updates.

    Runs as a background task during the platform lifespan.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        providers: dict[str, TrainingProvider],
        *,
        interval_seconds: float = 60.0,
        max_runtime_hours: float = 48.0,
    ) -> None:
        self._session_factory = session_factory
        self._providers = providers
        self._interval = interval_seconds
        self._max_runtime = max_runtime_hours
        self._task: asyncio.Task[None] | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background polling loop."""
        if self._task is not None:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("Job poller started (interval=%ss)", self._interval)

    async def stop(self) -> None:
        """Stop the polling loop gracefully."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("Job poller stopped")

    # ------------------------------------------------------------------
    # Poll loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Main polling loop — runs until stopped."""
        while self._running:
            try:
                await self._poll_all_running()
            except Exception:
                logger.exception("Error in poll loop")
            await asyncio.sleep(self._interval)

    async def _poll_all_running(self) -> None:
        """Poll all jobs in 'running' state."""
        async with self._session_factory() as session:
            stmt = select(TrainingJob).where(
                TrainingJob.status == JobStatus.RUNNING.value,
            )
            result = await session.execute(stmt)
            jobs = result.scalars().all()

            for job in jobs:
                try:
                    await self._poll_single(session, job)
                except Exception:
                    logger.warning("Failed to poll job %s", job.id, exc_info=True)

    async def _poll_single(
        self,
        session: AsyncSession,
        job: TrainingJob,
    ) -> None:
        """Poll a single job and update its state."""
        # Check for stuck jobs first
        if self._is_stuck(job):
            await self._finalize_stuck(session, job)
            return

        provider = self._providers.get(job.provider)
        if provider is None or job.provider_job_id is None:
            return

        cloud_status = await provider.poll_status(job.provider_job_id)
        new_status = cloud_status.status.value

        if new_status == job.status:
            return  # No change

        values: dict[str, Any] = {"status": new_status}

        if cloud_status.metrics:
            values["metrics"] = cloud_status.metrics
        if cloud_status.error:
            values["error"] = cloud_status.error
        if cloud_status.artifacts_uri:
            values["artifacts_uri"] = cloud_status.artifacts_uri

        # Finalize completed/failed jobs
        if new_status in (JobStatus.COMPLETED.value, JobStatus.FAILED.value):
            now = datetime.now(UTC)
            values["completed_at"] = now
            if cloud_status.cost_eur is not None:
                values["cost_actual_eur"] = cloud_status.cost_eur
            if cloud_status.duration_seconds is not None:
                values["duration_seconds"] = cloud_status.duration_seconds

        await session.execute(update(TrainingJob).where(TrainingJob.id == job.id).values(**values))
        await session.commit()

        logger.info(
            "Job %s status: %s → %s",
            job.id,
            job.status,
            new_status,
        )

    # ------------------------------------------------------------------
    # Stuck detection
    # ------------------------------------------------------------------

    def _is_stuck(self, job: TrainingJob) -> bool:
        """Check if a running job has exceeded max runtime."""
        if job.started_at is None:
            return False
        now = datetime.now(UTC)
        started = job.started_at
        if started.tzinfo is None:
            started = started.replace(tzinfo=UTC)
        elapsed_hours = (now - started).total_seconds() / 3600
        return elapsed_hours > self._max_runtime

    async def _finalize_stuck(
        self,
        session: AsyncSession,
        job: TrainingJob,
    ) -> None:
        """Mark a stuck job as failed and attempt cleanup."""
        logger.warning(
            "Job %s stuck (exceeded %.1fh max runtime)",
            job.id,
            self._max_runtime,
        )

        now = datetime.now(UTC)
        await session.execute(
            update(TrainingJob)
            .where(TrainingJob.id == job.id)
            .values(
                status=JobStatus.FAILED.value,
                error=f"Exceeded max runtime of {self._max_runtime}h",
                completed_at=now,
            )
        )
        await session.commit()

        # Attempt provider cleanup
        provider = self._providers.get(job.provider)
        if provider and job.provider_job_id:
            try:
                await provider.cancel_job(job.provider_job_id)
            except Exception:
                logger.warning(
                    "Failed to cleanup stuck job %s at provider",
                    job.id,
                )
