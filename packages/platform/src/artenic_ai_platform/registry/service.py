"""Model registry — CRUD, promote, retire, best-model lookup."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from artenic_ai_platform.db.models import PromotionRecord, RegisteredModel

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from artenic_ai_platform.tracking.mlflow_client import MLflowTracker

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Service layer for registered-model CRUD and lifecycle."""

    def __init__(
        self,
        session: AsyncSession,
        mlflow: MLflowTracker | None = None,
    ) -> None:
        self._session = session
        self._mlflow = mlflow

    # ------------------------------------------------------------------
    # Register
    # ------------------------------------------------------------------

    async def register(
        self,
        metadata: dict[str, Any],
        mlflow_run_id: str | None = None,
    ) -> str:
        """Register a new model.  Returns model_id ``{name}_v{version}``."""
        name: str = metadata["name"]
        version: str = metadata["version"]
        model_id = f"{name}_v{version}"

        model = RegisteredModel(
            id=model_id,
            name=name,
            version=version,
            model_type=metadata.get("model_type", ""),
            framework=metadata.get("framework", "custom"),
            description=metadata.get("description", ""),
            tags=metadata.get("tags", {}),
            input_features=metadata.get("input_features", []),
            output_schema=metadata.get("output_schema", {}),
            mlflow_run_id=mlflow_run_id,
        )
        self._session.add(model)
        await self._session.commit()

        logger.info("Registered model %s", model_id)
        return model_id

    # ------------------------------------------------------------------
    # Get / List
    # ------------------------------------------------------------------

    async def get(self, model_id: str) -> dict[str, Any]:
        """Get model by ID.  Raises ``ModelNotFoundError`` if missing."""
        from artenic_ai_sdk.exceptions import ModelNotFoundError

        result = await self._session.get(RegisteredModel, model_id)
        if result is None:
            raise ModelNotFoundError(f"Model '{model_id}' not found")
        return self._to_dict(result)

    async def list_all(self) -> list[dict[str, Any]]:
        """Return all registered models."""
        stmt = select(RegisteredModel).order_by(RegisteredModel.created_at.desc())
        result = await self._session.execute(stmt)
        rows = result.scalars().all()
        return [self._to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Promote / Retire
    # ------------------------------------------------------------------

    async def promote(self, model_id: str, version: str) -> None:
        """Promote a model version to production."""
        from artenic_ai_sdk.exceptions import ModelNotFoundError

        model = await self._session.get(RegisteredModel, model_id)
        if model is None:
            raise ModelNotFoundError(f"Model '{model_id}' not found")

        old_stage = model.stage
        model.stage = "production"

        promotion = PromotionRecord(
            model_id=model_id,
            from_stage=old_stage,
            to_stage="production",
            version=version,
        )
        self._session.add(promotion)
        await self._session.commit()

        # Sync with MLflow if available
        if self._mlflow and self._mlflow.available:
            await self._mlflow.transition_stage(model.name, version, "Production")

        logger.info("Promoted %s to production (v%s)", model_id, version)

    async def retire(self, model_id: str) -> None:
        """Archive a model."""
        from artenic_ai_sdk.exceptions import ModelNotFoundError

        model = await self._session.get(RegisteredModel, model_id)
        if model is None:
            raise ModelNotFoundError(f"Model '{model_id}' not found")

        old_stage = model.stage
        model.stage = "archived"

        promotion = PromotionRecord(
            model_id=model_id,
            from_stage=old_stage,
            to_stage="archived",
            version=model.version,
        )
        self._session.add(promotion)
        await self._session.commit()

        # Sync with MLflow if available
        if self._mlflow and self._mlflow.available:
            await self._mlflow.transition_stage(model.name, model.version, "Archived")

        logger.info("Retired %s", model_id)

    # ------------------------------------------------------------------
    # Best model
    # ------------------------------------------------------------------

    async def get_best_model(self, experiment_id: str, metric: str) -> dict[str, Any] | None:
        """Find the best model from MLflow experiment results."""
        if not self._mlflow or not self._mlflow.available:
            return None

        best = await self._mlflow.get_best_run(experiment_id, metric)
        if best is None:
            return None
        return best

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dict(model: RegisteredModel) -> dict[str, Any]:
        """Convert ORM model to dict (SDK-compatible)."""
        return {
            "name": model.name,
            "version": model.version,
            "model_type": model.model_type,
            "framework": model.framework,
            "description": model.description,
            "tags": model.tags,
            "input_features": model.input_features,
            "output_schema": model.output_schema,
            "created_at": (model.created_at.isoformat() if model.created_at else None),
            "updated_at": (model.updated_at.isoformat() if model.updated_at else None),
            "model_id": model.id,
            "stage": model.stage,
            "mlflow_run_id": model.mlflow_run_id,
        }
