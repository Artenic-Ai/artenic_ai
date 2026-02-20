"""Run service — CRUD, status transitions, input/output management."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from artenic_ai_platform.db.models import MLRun, MLRunIO
from artenic_ai_platform.entities.base_service import GenericEntityService
from artenic_ai_platform.entities.schemas import RUN_TRANSITIONS

logger = logging.getLogger(__name__)


class RunService(GenericEntityService[MLRun]):
    """Extended service for runs with status transitions and IO management."""

    _model_class = MLRun

    # ------------------------------------------------------------------
    # Status lifecycle
    # ------------------------------------------------------------------

    async def change_status(
        self,
        entity_id: str,
        new_status: str,
        *,
        metrics: dict[str, Any] | None = None,
        duration_seconds: float | None = None,
    ) -> MLRun:
        """Transition run to *new_status* with optional metric updates."""
        record = await self.get_or_raise(entity_id)
        allowed = RUN_TRANSITIONS.get(record.status, set())
        if new_status not in allowed:
            msg = f"Cannot transition from {record.status} to {new_status}"
            raise ValueError(msg)

        record.status = new_status

        if new_status == "running" and record.started_at is None:
            record.started_at = datetime.now(UTC)

        if new_status in ("completed", "failed"):
            record.completed_at = datetime.now(UTC)

        if metrics is not None:
            record.metrics = {**record.metrics, **metrics}

        if duration_seconds is not None:
            record.duration_seconds = duration_seconds

        await self._session.commit()
        await self._session.refresh(record)
        return record

    # ------------------------------------------------------------------
    # Input/Output management
    # ------------------------------------------------------------------

    async def add_io(
        self, run_id: str, entity_id: str, direction: str
    ) -> MLRunIO:
        """Add an input or output entity reference to a run."""
        await self.get_or_raise(run_id)
        io_record = MLRunIO(run_id=run_id, entity_id=entity_id, direction=direction)
        self._session.add(io_record)
        await self._session.commit()
        await self._session.refresh(io_record)
        logger.info("Added %s %s to run %s", direction, entity_id, run_id)
        return io_record

    async def list_io(self, run_id: str) -> list[MLRunIO]:
        """List all input/output references for a run."""
        await self.get_or_raise(run_id)
        result = await self._session.execute(
            select(MLRunIO).where(MLRunIO.run_id == run_id)
        )
        return list(result.scalars().all())

    async def remove_io(
        self, run_id: str, entity_id: str, direction: str
    ) -> None:
        """Remove an input/output reference from a run."""
        result = await self._session.execute(
            select(MLRunIO).where(
                MLRunIO.run_id == run_id,
                MLRunIO.entity_id == entity_id,
                MLRunIO.direction == direction,
            )
        )
        record = result.scalar_one_or_none()
        if record is None:
            msg = f"IO reference not found: {direction} {entity_id} on run {run_id}"
            raise ValueError(msg)
        await self._session.delete(record)
        await self._session.commit()

    # ------------------------------------------------------------------
    # Get with IO (enriched detail)
    # ------------------------------------------------------------------

    async def get_with_io(self, run_id: str) -> dict[str, Any]:
        """Get run details including its input/output references."""
        record = await self.get_or_raise(run_id)
        io_records = await self.list_io(run_id)
        return {
            **_run_to_dict(record),
            "inputs": [
                {"entity_id": io.entity_id}
                for io in io_records if io.direction == "input"
            ],
            "outputs": [
                {"entity_id": io.entity_id}
                for io in io_records if io.direction == "output"
            ],
        }


def _run_to_dict(r: MLRun) -> dict[str, Any]:
    """Convert an MLRun to a JSON-safe dict."""
    return {
        "id": r.id,
        "config": r.config,
        "status": r.status,
        "metrics": r.metrics,
        "triggered_by": r.triggered_by,
        "started_at": r.started_at.isoformat() if r.started_at else None,
        "completed_at": r.completed_at.isoformat() if r.completed_at else None,
        "duration_seconds": r.duration_seconds,
        "created_at": r.created_at.isoformat() if r.created_at else "",
    }
