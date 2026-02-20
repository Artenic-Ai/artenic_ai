"""REST API for training operations — /api/v1/training/*."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from artenic_ai_platform_training.service import TrainingManager

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

router = APIRouter(prefix="/api/v1/training", tags=["training"])


async def _get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Yield a DB session from app state session_factory."""
    factory = request.app.state.session_factory
    async with factory() as session:
        yield session


DbSession = Annotated[AsyncSession, Depends(_get_db)]


# ------------------------------------------------------------------
# Request / Response schemas
# ------------------------------------------------------------------


class DispatchRequest(BaseModel):
    """Body for POST /dispatch."""

    service: str
    model: str
    provider: str
    config: dict[str, Any] = Field(default_factory=dict)
    instance_type: str | None = None
    region: str | None = None
    is_spot: bool = False
    max_runtime_hours: float = 24.0
    workload_spec: dict[str, Any] | None = None


class DispatchResponse(BaseModel):
    """Response from POST /dispatch."""

    job_id: str


# ------------------------------------------------------------------
# Dependency helpers
# ------------------------------------------------------------------


def _get_training_manager(
    request: Request,
    session: AsyncSession,
) -> TrainingManager:
    """Build a TrainingManager from app state."""
    providers = getattr(request.app.state, "training_providers", {})
    budget = getattr(request.app.state, "budget_manager_factory", None)
    mlflow = getattr(request.app.state, "mlflow", None)

    budget_mgr = budget(session) if budget else None

    return TrainingManager(
        session=session,
        providers=providers,
        budget_manager=budget_mgr,
        mlflow=mlflow,
    )


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@router.post("/dispatch")
async def dispatch_training(
    body: DispatchRequest,
    request: Request,
    session: DbSession,
) -> DispatchResponse:
    """Dispatch a new training job."""
    mgr = _get_training_manager(request, session)
    job_id = await mgr.dispatch(
        service=body.service,
        model=body.model,
        provider=body.provider,
        config=body.config,
        instance_type=body.instance_type,
        region=body.region,
        is_spot=body.is_spot,
        max_runtime_hours=body.max_runtime_hours,
        workload_spec=body.workload_spec,
    )
    return DispatchResponse(job_id=job_id)


@router.get("/jobs")
async def list_jobs(
    request: Request,
    session: DbSession,
    service: str | None = Query(default=None),
    provider: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[dict[str, Any]]:
    """List training jobs with optional filters."""
    mgr = _get_training_manager(request, session)
    return await mgr.list_jobs(
        service=service,
        provider=provider,
        status=status,
        limit=limit,
        offset=offset,
    )


@router.get("/{job_id}")
async def get_training_status(
    job_id: str,
    request: Request,
    session: DbSession,
) -> dict[str, Any]:
    """Get detailed status for a training job."""
    mgr = _get_training_manager(request, session)
    return await mgr.get_status(job_id)


@router.post("/{job_id}/cancel")
async def cancel_training(
    job_id: str,
    request: Request,
    session: DbSession,
) -> dict[str, Any]:
    """Cancel a running or pending training job."""
    mgr = _get_training_manager(request, session)
    return await mgr.cancel(job_id)
