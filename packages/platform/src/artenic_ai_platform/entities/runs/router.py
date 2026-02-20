"""REST API for runs — /api/v1/runs/*."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: TC002

from artenic_ai_platform.entities.runs.schemas import (  # noqa: TC001
    AddRunIORequest,
    CreateRunRequest,
    UpdateRunStatusRequest,
)
from artenic_ai_platform.entities.runs.service import RunService, _run_to_dict

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

router = APIRouter(prefix="/api/v1/runs", tags=["runs"])


# ------------------------------------------------------------------
# Dependency helpers
# ------------------------------------------------------------------


async def _get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Yield a DB session from app state."""
    factory = request.app.state.session_factory
    async with factory() as session:
        yield session


def _get_service(session: Annotated[AsyncSession, Depends(_get_db)]) -> RunService:
    return RunService(session)


Svc = Annotated[RunService, Depends(_get_service)]


# ------------------------------------------------------------------
# CRUD
# ------------------------------------------------------------------


@router.post("", status_code=201)
async def create_run(body: CreateRunRequest, svc: Svc) -> dict[str, Any]:
    """Create a new run record."""
    data = body.model_dump(exclude={"id"})
    record = await svc.create(body.id, data)
    return _run_to_dict(record)


@router.get("")
async def list_runs(
    svc: Svc,
    offset: int = 0,
    limit: int = 50,
    status: str | None = None,
    triggered_by: str | None = None,
) -> list[dict[str, Any]]:
    """List runs with optional filters."""
    filters: dict[str, Any] = {}
    if status is not None:
        filters["status"] = status
    if triggered_by is not None:
        filters["triggered_by"] = triggered_by
    records = await svc.list_all(offset=offset, limit=limit, filters=filters)
    return [_run_to_dict(r) for r in records]


@router.get("/{run_id}")
async def get_run(run_id: str, svc: Svc) -> dict[str, Any]:
    """Get run details including inputs/outputs."""
    try:
        return await svc.get_with_io(run_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Run not found") from None


@router.delete("/{run_id}", status_code=204)
async def delete_run(run_id: str, svc: Svc) -> None:
    """Delete a run record."""
    try:
        await svc.delete(run_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Run not found") from None


# ------------------------------------------------------------------
# Status lifecycle
# ------------------------------------------------------------------


@router.patch("/{run_id}/status")
async def change_status(
    run_id: str, body: UpdateRunStatusRequest, svc: Svc
) -> dict[str, Any]:
    """Update run status with optional metrics and duration."""
    try:
        record = await svc.change_status(
            run_id,
            body.status,
            metrics=body.metrics,
            duration_seconds=body.duration_seconds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return _run_to_dict(record)


# ------------------------------------------------------------------
# Input/Output references
# ------------------------------------------------------------------


@router.post("/{run_id}/io", status_code=201)
async def add_io(run_id: str, body: AddRunIORequest, svc: Svc) -> dict[str, Any]:
    """Add an input or output entity reference to a run."""
    try:
        io_record = await svc.add_io(run_id, body.entity_id, body.direction)
    except ValueError:
        raise HTTPException(status_code=404, detail="Run not found") from None
    return {
        "id": io_record.id,
        "run_id": io_record.run_id,
        "entity_id": io_record.entity_id,
        "direction": io_record.direction,
    }


@router.get("/{run_id}/io")
async def list_io(run_id: str, svc: Svc) -> list[dict[str, Any]]:
    """List all input/output references for a run."""
    try:
        records = await svc.list_io(run_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Run not found") from None
    return [
        {
            "id": r.id,
            "run_id": r.run_id,
            "entity_id": r.entity_id,
            "direction": r.direction,
        }
        for r in records
    ]


@router.delete("/{run_id}/io")
async def remove_io(
    run_id: str,
    entity_id: str,
    direction: str,
    svc: Svc,
) -> None:
    """Remove an input/output entity reference from a run."""
    try:
        await svc.remove_io(run_id, entity_id, direction)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
