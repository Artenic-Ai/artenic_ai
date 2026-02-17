"""Training orchestrator — dispatch, status, cancel."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import select, update

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


class TrainingManager:
    """Orchestrates training dispatch, status polling and cancellation.

    Bridges the gap between the REST layer and cloud providers, persisting
    job state in the database.
    """

    def __init__(
        self,
        session: AsyncSession,
        providers: dict[str, TrainingProvider],
        *,
        budget_manager: Any | None = None,
        mlflow: Any | None = None,
    ) -> None:
        self._session = session
        self._providers = providers
        self._budget = budget_manager
        self._mlflow = mlflow

    # ------------------------------------------------------------------
    # Provider helpers
    # ------------------------------------------------------------------

    def _get_provider(self, name: str) -> TrainingProvider:
        """Resolve provider by name, raising ValueError if unknown."""
        provider = self._providers.get(name)
        if provider is None:
            available = ", ".join(sorted(self._providers)) or "(none)"
            msg = f"Unknown provider '{name}'. Available: {available}"
            raise ValueError(msg)
        return provider

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def dispatch(
        self,
        service: str,
        model: str,
        provider: str,
        config: dict[str, Any] | None = None,
        *,
        instance_type: str | None = None,
        region: str | None = None,
        is_spot: bool = False,
        max_runtime_hours: float = 24.0,
        workload_spec: dict[str, Any] | None = None,
    ) -> str:
        """Dispatch a training job: validate → budget check → persist → submit.

        Returns the platform job ID (UUID).
        """
        prov = self._get_provider(provider)

        spec = TrainingSpec(
            service=service,
            model=model,
            provider=provider,
            config=config or {},
            instance_type=instance_type,
            region=region,
            is_spot=is_spot,
            max_runtime_hours=max_runtime_hours,
            workload_spec=workload_spec,
        )

        # --- Cost estimation (optional) ---------------------------------
        cost_estimate: float | None = None
        cost_per_hour: float | None = None
        if instance_type:
            try:
                instances = await prov.list_instance_types(region=region)
                match = next(
                    (i for i in instances if i.name == instance_type),
                    None,
                )
                if match:
                    rate = (
                        match.spot_price_per_hour_eur
                        if is_spot and match.spot_price_per_hour_eur
                        else match.price_per_hour_eur
                    )
                    cost_per_hour = rate
                    cost_estimate = rate * max_runtime_hours
            except Exception:
                logger.debug("Cost estimation skipped for %s", instance_type)

        # --- Budget check (optional) ------------------------------------
        if self._budget is not None and cost_estimate is not None:
            await self._budget.check_budget(
                scope="service",
                scope_value=service,
                estimated_cost=cost_estimate,
            )

        # --- Persist job in DB -------------------------------------------
        job_id = str(uuid.uuid4())
        job = TrainingJob(
            id=job_id,
            service=service,
            model=model,
            provider=provider,
            config=spec.config,
            status=JobStatus.PENDING.value,
            instance_type=instance_type,
            region=region,
            is_spot=is_spot,
            workload_spec=workload_spec,
            cost_estimate_eur=cost_estimate,
            cost_per_hour_eur=cost_per_hour,
        )
        self._session.add(job)
        await self._session.flush()

        # --- Submit to provider -----------------------------------------
        try:
            provider_job_id = await prov.submit_job(spec)
        except Exception as exc:
            await self._session.execute(
                update(TrainingJob)
                .where(TrainingJob.id == job_id)
                .values(
                    status=JobStatus.FAILED.value,
                    error=str(exc),
                )
            )
            await self._session.commit()
            raise

        now = datetime.now(UTC)
        await self._session.execute(
            update(TrainingJob)
            .where(TrainingJob.id == job_id)
            .values(
                status=JobStatus.RUNNING.value,
                provider_job_id=provider_job_id,
                started_at=now,
            )
        )
        await self._session.commit()

        logger.info(
            "Training job %s dispatched to %s (provider_job=%s)",
            job_id,
            provider,
            provider_job_id,
        )
        return job_id

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def get_status(self, job_id: str) -> dict[str, Any]:
        """Get job status from DB, optionally polling the provider for live updates."""
        result = await self._session.execute(select(TrainingJob).where(TrainingJob.id == job_id))
        job = result.scalar_one_or_none()
        if job is None:
            msg = f"Training job '{job_id}' not found"
            raise KeyError(msg)

        # If job is still running, do a live poll
        if (
            job.status == JobStatus.RUNNING.value
            and job.provider_job_id
            and job.provider in self._providers
        ):
            try:
                cloud_status = await self._live_poll(job)
                job = await self._apply_cloud_status(job, cloud_status)
            except Exception:
                logger.warning("Live poll failed for job %s", job_id)

        return self._job_to_dict(job)

    async def _live_poll(self, job: TrainingJob) -> CloudJobStatus:
        """Poll the cloud provider for the latest status."""
        prov = self._providers[job.provider]
        return await prov.poll_status(job.provider_job_id)  # type: ignore[arg-type]

    async def _apply_cloud_status(
        self,
        job: TrainingJob,
        cloud: CloudJobStatus,
    ) -> TrainingJob:
        """Apply provider status to the DB record if changed."""
        new_status = cloud.status.value
        if new_status == job.status:
            return job

        values: dict[str, Any] = {"status": new_status}

        if cloud.metrics:
            values["metrics"] = cloud.metrics
        if cloud.error:
            values["error"] = cloud.error
        if cloud.artifacts_uri:
            values["artifacts_uri"] = cloud.artifacts_uri

        if new_status in (JobStatus.COMPLETED.value, JobStatus.FAILED.value):
            now = datetime.now(UTC)
            values["completed_at"] = now
            if cloud.cost_eur is not None:
                values["cost_actual_eur"] = cloud.cost_eur
            if cloud.duration_seconds is not None:
                values["duration_seconds"] = cloud.duration_seconds

        await self._session.execute(
            update(TrainingJob).where(TrainingJob.id == job.id).values(**values)
        )
        await self._session.commit()

        # Refresh the object
        result = await self._session.execute(select(TrainingJob).where(TrainingJob.id == job.id))
        return result.scalar_one()

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    async def cancel(self, job_id: str) -> dict[str, Any]:
        """Cancel a running job."""
        result = await self._session.execute(select(TrainingJob).where(TrainingJob.id == job_id))
        job = result.scalar_one_or_none()
        if job is None:
            msg = f"Training job '{job_id}' not found"
            raise KeyError(msg)

        if job.status not in (JobStatus.PENDING.value, JobStatus.RUNNING.value):
            msg = f"Cannot cancel job in '{job.status}' state"
            raise ValueError(msg)

        # Cancel at provider level
        if job.provider_job_id and job.provider in self._providers:
            prov = self._providers[job.provider]
            try:
                await prov.cancel_job(job.provider_job_id)
            except Exception:
                logger.warning("Provider cancel failed for job %s", job_id)

        now = datetime.now(UTC)
        await self._session.execute(
            update(TrainingJob)
            .where(TrainingJob.id == job_id)
            .values(
                status=JobStatus.CANCELLED.value,
                completed_at=now,
            )
        )
        await self._session.commit()

        result = await self._session.execute(select(TrainingJob).where(TrainingJob.id == job_id))
        return self._job_to_dict(result.scalar_one())

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    async def list_jobs(
        self,
        *,
        service: str | None = None,
        provider: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List training jobs with optional filters."""
        stmt = select(TrainingJob)
        if service:
            stmt = stmt.where(TrainingJob.service == service)
        if provider:
            stmt = stmt.where(TrainingJob.provider == provider)
        if status:
            stmt = stmt.where(TrainingJob.status == status)
        stmt = stmt.order_by(TrainingJob.created_at.desc())
        stmt = stmt.limit(limit).offset(offset)

        result = await self._session.execute(stmt)
        return [self._job_to_dict(j) for j in result.scalars().all()]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _job_to_dict(job: TrainingJob) -> dict[str, Any]:
        """Convert a TrainingJob ORM object to a plain dict."""
        return {
            "id": job.id,
            "service": job.service,
            "model": job.model,
            "provider": job.provider,
            "config": job.config,
            "status": job.status,
            "metrics": job.metrics,
            "error": job.error,
            "provider_job_id": job.provider_job_id,
            "instance_type": job.instance_type,
            "region": job.region,
            "is_spot": job.is_spot,
            "cost_estimate_eur": job.cost_estimate_eur,
            "cost_actual_eur": job.cost_actual_eur,
            "cost_per_hour_eur": job.cost_per_hour_eur,
            "duration_seconds": job.duration_seconds,
            "artifacts_uri": job.artifacts_uri,
            "created_at": (job.created_at.isoformat() if job.created_at else None),
            "started_at": (job.started_at.isoformat() if job.started_at else None),
            "completed_at": (job.completed_at.isoformat() if job.completed_at else None),
        }
