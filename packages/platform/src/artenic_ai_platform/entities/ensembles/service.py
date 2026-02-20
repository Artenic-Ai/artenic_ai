"""Ensemble service — CRUD, model list, stage transitions."""

from __future__ import annotations

import logging

from sqlalchemy import select

from artenic_ai_platform.db.models import MLEnsemble, MLEnsembleModel
from artenic_ai_platform.entities.base_service import GenericEntityService
from artenic_ai_platform.entities.schemas import ENSEMBLE_TRANSITIONS

logger = logging.getLogger(__name__)


class EnsembleService(GenericEntityService[MLEnsemble]):
    """Extended service for ensembles with model list and stage management."""

    _model_class = MLEnsemble

    # ------------------------------------------------------------------
    # Stage lifecycle
    # ------------------------------------------------------------------

    async def change_stage(self, entity_id: str, new_stage: str) -> MLEnsemble:
        """Transition ensemble to *new_stage*."""
        record = await self.get_or_raise(entity_id)
        allowed = ENSEMBLE_TRANSITIONS.get(record.stage, set())
        if new_stage not in allowed:
            msg = f"Cannot transition from {record.stage} to {new_stage}"
            raise ValueError(msg)
        record.stage = new_stage
        await self._session.commit()
        await self._session.refresh(record)
        return record

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    async def add_model(self, ensemble_id: str, model_id: str) -> MLEnsembleModel:
        """Add a model reference to an ensemble."""
        await self.get_or_raise(ensemble_id)
        link = MLEnsembleModel(ensemble_id=ensemble_id, model_id=model_id)
        self._session.add(link)
        await self._session.commit()
        await self._session.refresh(link)
        logger.info("Added model %s to ensemble %s", model_id, ensemble_id)
        return link

    async def list_models(self, ensemble_id: str) -> list[str]:
        """List all model IDs in an ensemble."""
        await self.get_or_raise(ensemble_id)
        result = await self._session.execute(
            select(MLEnsembleModel.model_id).where(
                MLEnsembleModel.ensemble_id == ensemble_id
            )
        )
        return list(result.scalars().all())

    async def remove_model(self, ensemble_id: str, model_id: str) -> None:
        """Remove a model reference from an ensemble."""
        result = await self._session.execute(
            select(MLEnsembleModel).where(
                MLEnsembleModel.ensemble_id == ensemble_id,
                MLEnsembleModel.model_id == model_id,
            )
        )
        record = result.scalar_one_or_none()
        if record is None:
            msg = f"Model {model_id} not in ensemble {ensemble_id}"
            raise ValueError(msg)
        await self._session.delete(record)
        await self._session.commit()

    # ------------------------------------------------------------------
    # Delete override — also removes model links
    # ------------------------------------------------------------------

    async def delete(self, entity_id: str) -> None:
        """Delete an ensemble and its model links (CASCADE handles this)."""
        await super().delete(entity_id)
