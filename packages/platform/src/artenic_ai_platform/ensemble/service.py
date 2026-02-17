"""Ensemble management — create, update, version, train."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import select, update

from artenic_ai_platform.db.models import (
    EnsembleJobRecord,
    EnsembleRecord,
    EnsembleVersionRecord,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from artenic_ai_platform.events.event_bus import EventBus

logger = logging.getLogger(__name__)


class PlatformEnsembleManager:
    """Manages ensemble lifecycle: creation, updates, versioning, and training dispatch.

    Each ensemble groups multiple model IDs under a routing strategy
    (e.g. weighted, round-robin, fallback) and tracks immutable version
    snapshots whenever the configuration changes.
    """

    def __init__(
        self,
        session: AsyncSession,
        event_bus: EventBus | None = None,
    ) -> None:
        self._session = session
        self._event_bus = event_bus

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    async def create_ensemble(
        self,
        name: str,
        service: str,
        strategy: str,
        model_ids: list[str],
        *,
        description: str = "",
        strategy_config: dict[str, Any] | None = None,
    ) -> str:
        """Create a new ensemble and its initial version snapshot.

        Returns the ensemble ID (``{service}_{name}_v1``).
        """
        ensemble_id = f"{service}_{name}_v1"
        config = strategy_config or {}

        ensemble = EnsembleRecord(
            id=ensemble_id,
            name=name,
            service=service,
            strategy=strategy,
            strategy_config=config,
            model_ids=model_ids,
            description=description,
            version=1,
            stage="registered",
            enabled=True,
        )
        self._session.add(ensemble)
        await self._session.flush()

        # Initial version snapshot
        version_record = EnsembleVersionRecord(
            ensemble_id=ensemble_id,
            version=1,
            model_ids=model_ids,
            strategy=strategy,
            strategy_config=config,
            change_reason="Initial creation",
        )
        self._session.add(version_record)
        await self._session.flush()
        await self._session.commit()

        logger.info("Created ensemble %s with %d models", ensemble_id, len(model_ids))
        return ensemble_id

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_ensemble(self, ensemble_id: str) -> dict[str, Any]:
        """Fetch a single ensemble by ID.

        Raises ``KeyError`` if the ensemble does not exist.
        """
        result = await self._session.execute(
            select(EnsembleRecord).where(EnsembleRecord.id == ensemble_id)
        )
        record = result.scalar_one_or_none()
        if record is None:
            msg = f"Ensemble '{ensemble_id}' not found"
            raise KeyError(msg)
        return self._ensemble_to_dict(record)

    async def list_ensembles(
        self,
        *,
        service: str | None = None,
        stage: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List ensembles with optional filters."""
        stmt = select(EnsembleRecord)
        if service:
            stmt = stmt.where(EnsembleRecord.service == service)
        if stage:
            stmt = stmt.where(EnsembleRecord.stage == stage)
        stmt = stmt.order_by(EnsembleRecord.created_at.desc())
        stmt = stmt.limit(limit).offset(offset)

        result = await self._session.execute(stmt)
        return [self._ensemble_to_dict(r) for r in result.scalars().all()]

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    async def update_ensemble(
        self,
        ensemble_id: str,
        *,
        name: str | None = None,
        model_ids: list[str] | None = None,
        strategy: str | None = None,
        strategy_config: dict[str, Any] | None = None,
        description: str | None = None,
        change_reason: str = "",
    ) -> dict[str, Any]:
        """Update ensemble fields, bump version, and create a version snapshot.

        Publishes an ``ensemble`` event via the event bus when available.
        """
        result = await self._session.execute(
            select(EnsembleRecord).where(EnsembleRecord.id == ensemble_id)
        )
        record = result.scalar_one_or_none()
        if record is None:
            msg = f"Ensemble '{ensemble_id}' not found"
            raise KeyError(msg)

        # Collect field updates
        values: dict[str, Any] = {}
        if name is not None:
            values["name"] = name
        if model_ids is not None:
            values["model_ids"] = model_ids
        if strategy is not None:
            values["strategy"] = strategy
        if strategy_config is not None:
            values["strategy_config"] = strategy_config
        if description is not None:
            values["description"] = description

        new_version = record.version + 1
        values["version"] = new_version

        await self._session.execute(
            update(EnsembleRecord).where(EnsembleRecord.id == ensemble_id).values(**values)
        )

        # Create version snapshot with resolved values
        version_record = EnsembleVersionRecord(
            ensemble_id=ensemble_id,
            version=new_version,
            model_ids=model_ids if model_ids is not None else record.model_ids,
            strategy=strategy if strategy is not None else record.strategy,
            strategy_config=(
                strategy_config if strategy_config is not None else record.strategy_config
            ),
            change_reason=change_reason,
        )
        self._session.add(version_record)
        await self._session.flush()
        await self._session.commit()

        # Publish event
        if self._event_bus is not None:
            self._event_bus.publish(
                "ensemble",
                {
                    "action": "updated",
                    "ensemble_id": ensemble_id,
                    "version": new_version,
                    "change_reason": change_reason,
                },
            )

        logger.info("Updated ensemble %s to version %d", ensemble_id, new_version)

        # Re-fetch to return up-to-date state
        result = await self._session.execute(
            select(EnsembleRecord).where(EnsembleRecord.id == ensemble_id)
        )
        return self._ensemble_to_dict(result.scalar_one())

    # ------------------------------------------------------------------
    # Training dispatch
    # ------------------------------------------------------------------

    async def dispatch_ensemble_training(
        self,
        ensemble_id: str,
        provider: str,
        config: dict[str, Any] | None = None,
    ) -> str:
        """Create a pending ensemble training job.

        Returns the job ID (UUID).
        """
        result = await self._session.execute(
            select(EnsembleRecord).where(EnsembleRecord.id == ensemble_id)
        )
        record = result.scalar_one_or_none()
        if record is None:
            msg = f"Ensemble '{ensemble_id}' not found"
            raise KeyError(msg)

        job_id = str(uuid.uuid4())
        job = EnsembleJobRecord(
            id=job_id,
            ensemble_id=ensemble_id,
            status="pending",
            training_job_ids=[],
            total_models=len(record.model_ids),
            completed_models=0,
            failed_models=0,
        )
        self._session.add(job)
        await self._session.flush()
        await self._session.commit()

        logger.info(
            "Dispatched ensemble training job %s for %s (%d models, provider=%s)",
            job_id,
            ensemble_id,
            len(record.model_ids),
            provider,
        )
        return job_id

    # ------------------------------------------------------------------
    # Job status
    # ------------------------------------------------------------------

    async def get_ensemble_job_status(self, job_id: str) -> dict[str, Any]:
        """Fetch an ensemble job by ID.

        Raises ``KeyError`` if the job does not exist.
        """
        result = await self._session.execute(
            select(EnsembleJobRecord).where(EnsembleJobRecord.id == job_id)
        )
        record = result.scalar_one_or_none()
        if record is None:
            msg = f"Ensemble job '{job_id}' not found"
            raise KeyError(msg)
        return self._job_to_dict(record)

    # ------------------------------------------------------------------
    # Version history
    # ------------------------------------------------------------------

    async def get_version_history(
        self,
        ensemble_id: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return version snapshots for an ensemble, newest first."""
        stmt = (
            select(EnsembleVersionRecord)
            .where(EnsembleVersionRecord.ensemble_id == ensemble_id)
            .order_by(EnsembleVersionRecord.version.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return [
            {
                "id": v.id,
                "ensemble_id": v.ensemble_id,
                "version": v.version,
                "model_ids": v.model_ids,
                "strategy": v.strategy,
                "strategy_config": v.strategy_config,
                "change_reason": v.change_reason,
                "created_at": v.created_at.isoformat() if v.created_at else None,
            }
            for v in result.scalars().all()
        ]

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensemble_to_dict(record: EnsembleRecord) -> dict[str, Any]:
        """Convert an EnsembleRecord to a plain dict."""
        return {
            "id": record.id,
            "name": record.name,
            "description": record.description,
            "service": record.service,
            "strategy": record.strategy,
            "strategy_config": record.strategy_config,
            "model_ids": record.model_ids,
            "stage": record.stage,
            "version": record.version,
            "enabled": record.enabled,
            "created_at": record.created_at.isoformat() if record.created_at else None,
            "updated_at": record.updated_at.isoformat() if record.updated_at else None,
        }

    @staticmethod
    def _job_to_dict(record: EnsembleJobRecord) -> dict[str, Any]:
        """Convert an EnsembleJobRecord to a plain dict."""
        return {
            "id": record.id,
            "ensemble_id": record.ensemble_id,
            "status": record.status,
            "training_job_ids": record.training_job_ids,
            "total_models": record.total_models,
            "completed_models": record.completed_models,
            "failed_models": record.failed_models,
            "total_cost_eur": record.total_cost_eur,
            "created_at": record.created_at.isoformat() if record.created_at else None,
            "completed_at": (record.completed_at.isoformat() if record.completed_at else None),
        }
