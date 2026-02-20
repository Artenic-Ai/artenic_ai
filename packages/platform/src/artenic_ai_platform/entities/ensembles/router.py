"""REST API for ensembles — /api/v1/ensembles/*."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: TC002

from artenic_ai_platform.entities.ensembles.schemas import (  # noqa: TC001
    AddModelRequest,
    ChangeStageRequest,
    CreateEnsembleRequest,
    UpdateEnsembleRequest,
)
from artenic_ai_platform.entities.ensembles.service import EnsembleService

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from artenic_ai_platform.db.models import MLEnsemble

router = APIRouter(prefix="/api/v1/ensembles", tags=["ensembles"])


# ------------------------------------------------------------------
# Dependency helpers
# ------------------------------------------------------------------


async def _get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    factory = request.app.state.session_factory
    async with factory() as session:
        yield session


def _get_service(session: Annotated[AsyncSession, Depends(_get_db)]) -> EnsembleService:
    return EnsembleService(session)


Svc = Annotated[EnsembleService, Depends(_get_service)]


# ------------------------------------------------------------------
# CRUD
# ------------------------------------------------------------------


@router.post("", status_code=201)
async def create_ensemble(body: CreateEnsembleRequest, svc: Svc) -> dict[str, Any]:
    """Create a new ensemble."""
    data = body.model_dump(exclude={"id", "version", "model_ids"})
    if body.version is not None:
        data["version"] = body.version
    else:
        data["version"] = await svc.next_version(body.name)
    if "metadata" in data:
        data["metadata_"] = data.pop("metadata")

    record = await svc.create(body.id, data)

    # Add model references
    for model_id in body.model_ids:
        await svc.add_model(body.id, model_id)

    model_ids = await svc.list_models(body.id)
    return {**_ensemble_to_dict(record), "model_ids": model_ids}


@router.get("")
async def list_ensembles(
    svc: Svc,
    offset: int = 0,
    limit: int = 50,
    stage: str | None = None,
    strategy_type: str | None = None,
) -> list[dict[str, Any]]:
    """List ensembles with optional filters."""
    filters: dict[str, Any] = {}
    if stage is not None:
        filters["stage"] = stage
    if strategy_type is not None:
        filters["strategy_type"] = strategy_type
    records = await svc.list_all(offset=offset, limit=limit, filters=filters)
    return [_ensemble_to_dict(r) for r in records]


@router.get("/{ensemble_id}")
async def get_ensemble(ensemble_id: str, svc: Svc) -> dict[str, Any]:
    """Get ensemble details including model IDs."""
    record = await svc.get(ensemble_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Ensemble not found")
    model_ids = await svc.list_models(ensemble_id)
    return {**_ensemble_to_dict(record), "model_ids": model_ids}


@router.patch("/{ensemble_id}")
async def update_ensemble(
    ensemble_id: str, body: UpdateEnsembleRequest, svc: Svc
) -> dict[str, Any]:
    """Update ensemble metadata/metrics."""
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if "metadata" in updates:
        updates["metadata_"] = updates.pop("metadata")
    try:
        record = await svc.update(ensemble_id, updates)
    except ValueError:
        raise HTTPException(status_code=404, detail="Ensemble not found") from None
    return _ensemble_to_dict(record)


@router.delete("/{ensemble_id}", status_code=204)
async def delete_ensemble(ensemble_id: str, svc: Svc) -> None:
    """Delete an ensemble and its model links."""
    try:
        await svc.delete(ensemble_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Ensemble not found") from None


# ------------------------------------------------------------------
# Stage lifecycle
# ------------------------------------------------------------------


@router.patch("/{ensemble_id}/stage")
async def change_stage(
    ensemble_id: str, body: ChangeStageRequest, svc: Svc
) -> dict[str, Any]:
    """Transition ensemble stage (staging -> production -> retired)."""
    try:
        record = await svc.change_stage(ensemble_id, body.stage)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return _ensemble_to_dict(record)


# ------------------------------------------------------------------
# Model management
# ------------------------------------------------------------------


@router.post("/{ensemble_id}/models", status_code=201)
async def add_model(
    ensemble_id: str, body: AddModelRequest, svc: Svc
) -> dict[str, Any]:
    """Add a model to an ensemble."""
    try:
        link = await svc.add_model(ensemble_id, body.model_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Ensemble not found") from None
    return {
        "id": link.id,
        "ensemble_id": link.ensemble_id,
        "model_id": link.model_id,
    }


@router.get("/{ensemble_id}/models")
async def list_models(ensemble_id: str, svc: Svc) -> list[str]:
    """List all model IDs in an ensemble."""
    try:
        return await svc.list_models(ensemble_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Ensemble not found") from None


@router.delete("/{ensemble_id}/models/{model_id}", status_code=204)
async def remove_model(ensemble_id: str, model_id: str, svc: Svc) -> None:
    """Remove a model from an ensemble."""
    try:
        await svc.remove_model(ensemble_id, model_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _ensemble_to_dict(r: MLEnsemble) -> dict[str, Any]:
    return {
        "id": r.id,
        "name": r.name,
        "version": r.version,
        "strategy_type": r.strategy_type,
        "metadata": r.metadata_,
        "metrics": r.metrics,
        "stage": r.stage,
        "created_at": r.created_at.isoformat() if r.created_at else "",
    }
