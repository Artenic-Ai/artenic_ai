"""Model service — CRUD, artifact management, stage transitions."""

from __future__ import annotations

import contextlib
import hashlib
import logging
from typing import TYPE_CHECKING

from artenic_ai_platform.db.models import MLModel
from artenic_ai_platform.entities.base_service import GenericEntityService
from artenic_ai_platform.entities.schemas import MODEL_TRANSITIONS

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from artenic_ai_platform.entities.datasets.storage import StorageBackend

logger = logging.getLogger(__name__)


class ModelService(GenericEntityService[MLModel]):
    """Extended service for models with artifact operations and stage management."""

    _model_class = MLModel

    def __init__(self, session: AsyncSession, storage: StorageBackend) -> None:
        super().__init__(session)
        self._storage = storage

    # ------------------------------------------------------------------
    # Stage lifecycle
    # ------------------------------------------------------------------

    async def change_stage(self, entity_id: str, new_stage: str) -> MLModel:
        """Transition model to *new_stage*."""
        record = await self.get_or_raise(entity_id)
        allowed = MODEL_TRANSITIONS.get(record.stage, set())
        if new_stage not in allowed:
            msg = f"Cannot transition from {record.stage} to {new_stage}"
            raise ValueError(msg)
        record.stage = new_stage
        await self._session.commit()
        await self._session.refresh(record)
        return record

    # ------------------------------------------------------------------
    # Artifact management
    # ------------------------------------------------------------------

    async def upload_artifact(
        self, model_id: str, filename: str, data: bytes
    ) -> MLModel:
        """Upload a model artifact (weights, checkpoint, etc.)."""
        record = await self.get_or_raise(model_id)

        storage_key = f"models/{model_id}/{filename}"
        storage_path = await self._storage.save(storage_key, data)
        file_hash = hashlib.sha256(data).hexdigest()

        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        record.artifact_path = storage_path
        record.artifact_format = ext
        record.artifact_size_bytes = len(data)
        record.artifact_sha256 = file_hash

        await self._session.commit()
        await self._session.refresh(record)
        logger.info(
            "Uploaded artifact %s for model %s (%d bytes)",
            filename, model_id, len(data),
        )
        return record

    async def download_artifact(self, model_id: str) -> bytes:
        """Download the model artifact."""
        record = await self.get_or_raise(model_id)
        if not record.artifact_path:
            msg = f"No artifact for model {model_id}"
            raise FileNotFoundError(msg)
        return await self._storage.load(record.artifact_path)

    # ------------------------------------------------------------------
    # Delete override — also removes artifact
    # ------------------------------------------------------------------

    async def delete(self, entity_id: str) -> None:
        """Delete a model and its artifact."""
        record = await self.get_or_raise(entity_id)
        if record.artifact_path:
            with contextlib.suppress(FileNotFoundError):
                await self._storage.delete(record.artifact_path)
        await self._session.delete(record)
        await self._session.commit()
        logger.info("Deleted model %s", entity_id)
