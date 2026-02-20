"""REST API for lineage — /api/v1/lineage/*."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: TC002

from artenic_ai_platform.entities.lineage.schemas import AddLineageRequest  # noqa: TC001
from artenic_ai_platform.entities.lineage.service import LineageService

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

router = APIRouter(prefix="/api/v1/lineage", tags=["lineage"])


# ------------------------------------------------------------------
# Dependency helpers
# ------------------------------------------------------------------


async def _get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    factory = request.app.state.session_factory
    async with factory() as session:
        yield session


def _get_service(session: Annotated[AsyncSession, Depends(_get_db)]) -> LineageService:
    return LineageService(session)


Svc = Annotated[LineageService, Depends(_get_service)]


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@router.post("", status_code=201)
async def add_link(body: AddLineageRequest, svc: Svc) -> dict[str, Any]:
    """Add a lineage link between two entities."""
    link = await svc.add_link(body.source_id, body.target_id, body.relation_type)
    return {
        "id": link.id,
        "source_id": link.source_id,
        "target_id": link.target_id,
        "relation_type": link.relation_type,
        "created_at": link.created_at.isoformat() if link.created_at else "",
    }


@router.get("/{entity_id}")
async def get_lineage(entity_id: str, svc: Svc) -> dict[str, Any]:
    """Get all lineage for an entity (upstream + downstream)."""
    return await svc.get_links(entity_id)


@router.get("/{entity_id}/graph")
async def get_graph(entity_id: str, svc: Svc) -> dict[str, Any]:
    """Get full dependency graph starting from an entity."""
    return await svc.get_graph(entity_id)


@router.delete("")
async def remove_link(
    source_id: str,
    target_id: str,
    relation_type: str,
    svc: Svc,
) -> None:
    """Remove a lineage link."""
    try:
        await svc.remove_link(source_id, target_id, relation_type)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
