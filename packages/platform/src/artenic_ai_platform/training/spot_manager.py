"""Spot / preemptible instance management — detection and retry."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import update

from artenic_ai_platform.db.models import TrainingJob
from artenic_ai_platform.providers.base import (
    CloudJobStatus,
    JobStatus,
    TrainingSpec,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from artenic_ai_platform.providers.base import TrainingProvider

logger = logging.getLogger(__name__)

# Keywords that indicate preemption in provider error messages
_PREEMPTION_KEYWORDS = frozenset(
    {
        "preempted",
        "preemption",
        "spot interruption",
        "instance terminated",
        "capacity",
        "reclaimed",
        "evicted",
    }
)


class SpotManager:
    """Detects spot preemptions and handles retry logic.

    When a spot instance is preempted, the manager:
    1. Marks the original job as preempted
    2. Creates a new retry job (with incremented preemption_count)
    3. Re-submits to the same or a failover region
    """

    def __init__(
        self,
        session: AsyncSession,
        providers: dict[str, TrainingProvider],
        *,
        max_retries: int = 3,
        failover_regions: list[str] | None = None,
    ) -> None:
        self._session = session
        self._providers = providers
        self._max_retries = max_retries
        self._failover_regions = failover_regions or []

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_preemption(
        self,
        job: TrainingJob,
        cloud_status: CloudJobStatus,
    ) -> bool:
        """Determine if a job failure was caused by spot preemption.

        Heuristics:
        - Job was a spot instance
        - Status is FAILED or PREEMPTED
        - Error message contains preemption keywords
        - Duration was abnormally short (< 5 min for a job expected to run hours)
        """
        if not job.is_spot:
            return False

        if cloud_status.status == JobStatus.PREEMPTED:
            return True

        if cloud_status.status != JobStatus.FAILED:
            return False

        # Check error keywords
        error = (cloud_status.error or "").lower()
        if any(kw in error for kw in _PREEMPTION_KEYWORDS):
            return True

        # Check abnormal short duration (< 5 min)
        return cloud_status.duration_seconds is not None and cloud_status.duration_seconds < 300

    # ------------------------------------------------------------------
    # Retry
    # ------------------------------------------------------------------

    async def handle_preemption(
        self,
        job: TrainingJob,
    ) -> str | None:
        """Handle a preempted job: mark as preempted and retry if possible.

        Returns the new job ID if retried, None if max retries exceeded.
        """
        if job.preemption_count >= self._max_retries:
            logger.warning(
                "Job %s exceeded max preemption retries (%d)",
                job.id,
                self._max_retries,
            )
            await self._mark_preempted(job, final=True)
            return None

        # Mark original as preempted
        await self._mark_preempted(job, final=False)

        # Choose region for retry
        region = self._select_failover_region(job)

        # Create retry job
        new_job_id = await self._create_retry_job(job, region)

        # Re-submit to provider
        provider = self._providers.get(job.provider)
        if provider is None:
            logger.error("Provider '%s' not found for retry", job.provider)
            return new_job_id

        spec = TrainingSpec(
            service=job.service,
            model=job.model,
            provider=job.provider,
            config=job.config or {},
            instance_type=job.instance_type,
            region=region,
            is_spot=job.is_spot,
        )

        try:
            provider_job_id = await provider.submit_job(spec)
            now = datetime.now(UTC)
            await self._session.execute(
                update(TrainingJob)
                .where(TrainingJob.id == new_job_id)
                .values(
                    status=JobStatus.RUNNING.value,
                    provider_job_id=provider_job_id,
                    started_at=now,
                )
            )
            await self._session.commit()
            logger.info(
                "Preempted job %s retried as %s (region=%s, attempt=%d)",
                job.id,
                new_job_id,
                region,
                job.preemption_count + 1,
            )
        except Exception:
            logger.exception("Failed to re-submit preempted job %s", job.id)
            await self._session.execute(
                update(TrainingJob)
                .where(TrainingJob.id == new_job_id)
                .values(
                    status=JobStatus.FAILED.value,
                    error="Failed to re-submit after preemption",
                )
            )
            await self._session.commit()

        return new_job_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _mark_preempted(
        self,
        job: TrainingJob,
        *,
        final: bool,
    ) -> None:
        """Mark a job as preempted in the database."""
        now = datetime.now(UTC)
        await self._session.execute(
            update(TrainingJob)
            .where(TrainingJob.id == job.id)
            .values(
                status=JobStatus.PREEMPTED.value,
                preempted=True,
                completed_at=now,
                error=(
                    "Max preemption retries exceeded" if final else "Preempted by cloud provider"
                ),
            )
        )
        await self._session.commit()

    def _select_failover_region(self, job: TrainingJob) -> str | None:
        """Pick the next failover region, cycling through the list."""
        if not self._failover_regions:
            return job.region

        # Use preemption count to cycle through regions
        idx = job.preemption_count % len(self._failover_regions)
        region = self._failover_regions[idx]

        # Skip the current region if possible
        if region == job.region and len(self._failover_regions) > 1:
            idx = (idx + 1) % len(self._failover_regions)
            region = self._failover_regions[idx]

        return region

    async def _create_retry_job(
        self,
        original: TrainingJob,
        region: str | None,
    ) -> str:
        """Create a new TrainingJob as a retry of the preempted one."""
        new_id = str(uuid.uuid4())
        retry = TrainingJob(
            id=new_id,
            service=original.service,
            model=original.model,
            provider=original.provider,
            config=original.config,
            status=JobStatus.PENDING.value,
            instance_type=original.instance_type,
            region=region,
            is_spot=original.is_spot,
            workload_spec=original.workload_spec,
            cost_estimate_eur=original.cost_estimate_eur,
            cost_per_hour_eur=original.cost_per_hour_eur,
            resumed_from_job_id=original.id,
            preemption_count=original.preemption_count + 1,
        )
        self._session.add(retry)
        await self._session.flush()
        return new_id
