"""REST API for ensemble management — /api/v1/ensembles/*."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from artenic_ai_platform.ensemble.service import PlatformEnsembleManager

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ensembles", tags=["ensembles"])


# ------------------------------------------------------------------
# DB session dependency
# ------------------------------------------------------------------


async def _get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Yield a DB session from app state session_factory."""
    factory = request.app.state.session_factory
    async with factory() as session:
        yield session


DbSession = Annotated[AsyncSession, Depends(_get_db)]


# ------------------------------------------------------------------
# Request schemas
# ------------------------------------------------------------------


class CreateEnsembleRequest(BaseModel):
    """Body for POST /ensembles."""

    name: str
    service: str
    strategy: str
    model_ids: list[str]
    description: str = ""
    strategy_config: dict[str, Any] | None = Field(default=None)


class UpdateEnsembleRequest(BaseModel):
    """Body for PUT /ensembles/{ensemble_id}."""

    name: str | None = None
    model_ids: list[str] | None = None
    strategy: str | None = None
    strategy_config: dict[str, Any] | None = None
    description: str | None = None
    change_reason: str = ""


class DispatchTrainingRequest(BaseModel):
    """Body for POST /ensembles/{ensemble_id}/train."""

    provider: str
    config: dict[str, Any] | None = Field(default=None)


# ------------------------------------------------------------------
# Dependency helpers
# ------------------------------------------------------------------


def _get_ensemble_manager(
    request: Request,
    session: AsyncSession,
) -> PlatformEnsembleManager:
    """Build a PlatformEnsembleManager from app state."""
    event_bus = getattr(request.app.state, "event_bus", None)
    return PlatformEnsembleManager(session=session, event_bus=event_bus)


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@router.post("")
async def create_ensemble(
    body: CreateEnsembleRequest,
    request: Request,
    session: DbSession,
) -> dict[str, Any]:
    """Create a new ensemble."""
    mgr = _get_ensemble_manager(request, session)
    ensemble_id = await mgr.create_ensemble(
        name=body.name,
        service=body.service,
        strategy=body.strategy,
        model_ids=body.model_ids,
        description=body.description,
        strategy_config=body.strategy_config,
    )
    return {"ensemble_id": ensemble_id}


@router.get("")
async def list_ensembles(
    request: Request,
    session: DbSession,
    service: str | None = Query(default=None),
    stage: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[dict[str, Any]]:
    """List ensembles with optional filters."""
    mgr = _get_ensemble_manager(request, session)
    return await mgr.list_ensembles(
        service=service,
        stage=stage,
        limit=limit,
        offset=offset,
    )


@router.get("/{ensemble_id}")
async def get_ensemble(
    ensemble_id: str,
    request: Request,
    session: DbSession,
) -> dict[str, Any]:
    """Get a single ensemble by ID."""
    mgr = _get_ensemble_manager(request, session)
    return await mgr.get_ensemble(ensemble_id)


@router.put("/{ensemble_id}")
async def update_ensemble(
    ensemble_id: str,
    body: UpdateEnsembleRequest,
    request: Request,
    session: DbSession,
) -> dict[str, Any]:
    """Update an ensemble (bumps version)."""
    mgr = _get_ensemble_manager(request, session)
    return await mgr.update_ensemble(
        ensemble_id,
        name=body.name,
        model_ids=body.model_ids,
        strategy=body.strategy,
        strategy_config=body.strategy_config,
        description=body.description,
        change_reason=body.change_reason,
    )


@router.post("/{ensemble_id}/train")
async def dispatch_ensemble_training(
    ensemble_id: str,
    body: DispatchTrainingRequest,
    request: Request,
    session: DbSession,
) -> dict[str, Any]:
    """Dispatch training for all models in an ensemble."""
    mgr = _get_ensemble_manager(request, session)
    job_id = await mgr.dispatch_ensemble_training(
        ensemble_id=ensemble_id,
        provider=body.provider,
        config=body.config,
    )
    return {"job_id": job_id}


@router.get("/{ensemble_id}/jobs/{job_id}")
async def get_ensemble_job_status(
    ensemble_id: str,
    job_id: str,
    request: Request,
    session: DbSession,
) -> dict[str, Any]:
    """Get status of an ensemble training job."""
    mgr = _get_ensemble_manager(request, session)
    return await mgr.get_ensemble_job_status(job_id)


@router.get("/{ensemble_id}/versions")
async def get_version_history(
    ensemble_id: str,
    request: Request,
    session: DbSession,
    limit: int = Query(default=20, ge=1, le=100),
) -> list[dict[str, Any]]:
    """Get version history for an ensemble."""
    mgr = _get_ensemble_manager(request, session)
    return await mgr.get_version_history(ensemble_id, limit=limit)
