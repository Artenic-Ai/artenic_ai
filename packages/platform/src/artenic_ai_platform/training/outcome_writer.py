"""Training outcome persistence for future optimizer integration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from artenic_ai_platform.db.models import MLRun, TrainingJob, TrainingOutcomeRecord
from artenic_ai_platform.providers.base import JobStatus

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class OutcomeWriter:
    """Writes denormalized training outcomes for the optimizer interface.

    Each completed job produces one ``TrainingOutcomeRecord`` that captures
    the workload spec, chosen instance, actual cost/duration, and whether
    the training succeeded.  This data is the bridge between the platform
    and a future optimizer feedback loop.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def write_outcome(self, job_id: str) -> dict[str, Any] | None:
        """Create a TrainingOutcomeRecord from a completed job.

        Returns the outcome dict, or None if the job isn't complete
        or an outcome already exists.
        """
        result = await self._session.execute(select(TrainingJob).where(TrainingJob.id == job_id))
        job = result.scalar_one_or_none()
        if job is None:
            logger.warning("Job %s not found for outcome", job_id)
            return None

        if job.status not in (
            JobStatus.COMPLETED.value,
            JobStatus.FAILED.value,
        ):
            logger.debug("Job %s not finalized, skipping outcome", job_id)
            return None

        # Check for existing outcome
        existing = await self._session.execute(
            select(TrainingOutcomeRecord).where(TrainingOutcomeRecord.job_id == job_id)
        )
        if existing.scalar_one_or_none() is not None:
            logger.debug("Outcome already exists for job %s", job_id)
            return None

        # Compute duration in hours
        duration_hours = 0.0
        if job.duration_seconds is not None:
            duration_hours = job.duration_seconds / 3600.0
        elif job.started_at and job.completed_at:
            delta = job.completed_at - job.started_at
            duration_hours = delta.total_seconds() / 3600.0

        # Determine primary metric
        metric_name: str | None = job.primary_metric_name
        metric_value: float | None = job.primary_metric_after
        if metric_value is None and job.metrics:
            # Try to extract from metrics dict
            for key in ("accuracy", "f1", "loss", "rmse", "mae"):
                if key in job.metrics:
                    metric_name = key
                    metric_value = float(job.metrics[key])
                    break

        outcome = TrainingOutcomeRecord(
            job_id=job_id,
            workload_spec=job.workload_spec or {},
            provider=job.provider,
            instance_type=job.instance_type or "unknown",
            region=job.region,
            is_spot=job.is_spot,
            actual_duration_hours=duration_hours,
            actual_cost_eur=job.cost_actual_eur or 0.0,
            predicted_duration_hours=job.duration_predicted_hours,
            predicted_cost_eur=job.cost_predicted_eur,
            success=job.status == JobStatus.COMPLETED.value,
            primary_metric_name=metric_name,
            primary_metric_value=metric_value,
        )
        self._session.add(outcome)
        await self._session.flush()
        await self._session.commit()

        # --- Update associated MLRun (if exists) -------------------------
        run_id = f"run:{job_id}"
        run_result = await self._session.execute(select(MLRun).where(MLRun.id == run_id))
        ml_run = run_result.scalar_one_or_none()
        if ml_run is not None:
            ml_run.status = "completed" if outcome.success else "failed"
            ml_run.completed_at = job.completed_at
            ml_run.duration_seconds = job.duration_seconds
            if job.metrics:
                ml_run.metrics = job.metrics
            await self._session.commit()

        logger.info("Wrote outcome for job %s (success=%s)", job_id, outcome.success)
        return self._to_dict(outcome)

    async def list_outcomes(
        self,
        *,
        provider: str | None = None,
        success_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List training outcomes with optional filters."""
        stmt = select(TrainingOutcomeRecord)
        if provider:
            stmt = stmt.where(TrainingOutcomeRecord.provider == provider)
        if success_only:
            stmt = stmt.where(TrainingOutcomeRecord.success.is_(True))
        stmt = stmt.order_by(TrainingOutcomeRecord.created_at.desc())
        stmt = stmt.limit(limit).offset(offset)

        result = await self._session.execute(stmt)
        return [self._to_dict(o) for o in result.scalars().all()]

    @staticmethod
    def _to_dict(outcome: TrainingOutcomeRecord) -> dict[str, Any]:
        return {
            "id": outcome.id,
            "job_id": outcome.job_id,
            "provider": outcome.provider,
            "instance_type": outcome.instance_type,
            "region": outcome.region,
            "is_spot": outcome.is_spot,
            "actual_duration_hours": outcome.actual_duration_hours,
            "actual_cost_eur": outcome.actual_cost_eur,
            "predicted_duration_hours": outcome.predicted_duration_hours,
            "predicted_cost_eur": outcome.predicted_cost_eur,
            "success": outcome.success,
            "primary_metric_name": outcome.primary_metric_name,
            "primary_metric_value": outcome.primary_metric_value,
            "created_at": (outcome.created_at.isoformat() if outcome.created_at else None),
        }
