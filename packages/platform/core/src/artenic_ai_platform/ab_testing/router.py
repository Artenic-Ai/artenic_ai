"""REST API for A/B testing — /api/v1/ab-tests/*."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from artenic_ai_platform.ab_testing.service import ABTestManager

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


# ------------------------------------------------------------------
# Dependency helpers
# ------------------------------------------------------------------


async def _get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Yield a DB session from app state session_factory."""
    factory = request.app.state.session_factory
    async with factory() as session:
        yield session


DbSession = Annotated[AsyncSession, Depends(_get_db)]

router = APIRouter(prefix="/api/v1/ab-tests", tags=["ab-tests"])


# ------------------------------------------------------------------
# Request schemas
# ------------------------------------------------------------------


class CreateTestRequest(BaseModel):
    """Body for ``POST /api/v1/ab-tests``."""

    name: str
    service: str
    variants: dict[str, Any]
    primary_metric: str
    min_samples: int = 100


class ConcludeTestRequest(BaseModel):
    """Body for ``POST /api/v1/ab-tests/{test_id}/conclude``."""

    winner: str | None = None
    reason: str = ""


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------


def _get_manager(request: Request, session: AsyncSession) -> ABTestManager:
    """Build an ABTestManager from app state."""
    event_bus = getattr(request.app.state, "event_bus", None)
    return ABTestManager(session, event_bus=event_bus)


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@router.post("", status_code=201)
async def create_test(
    body: CreateTestRequest,
    request: Request,
    session: DbSession,
) -> dict[str, Any]:
    """Create a new A/B test."""
    mgr = _get_manager(request, session)
    test_id = await mgr.create_test(
        name=body.name,
        service=body.service,
        variants=body.variants,
        primary_metric=body.primary_metric,
        min_samples=body.min_samples,
    )
    return {"test_id": test_id}


@router.get("")
async def list_tests(
    request: Request,
    session: DbSession,
    service: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[dict[str, Any]]:
    """List A/B tests with optional filters."""
    mgr = _get_manager(request, session)
    return await mgr.list_tests(
        service=service,
        status=status,
        limit=limit,
        offset=offset,
    )


@router.get("/{test_id}")
async def get_test(
    test_id: str,
    request: Request,
    session: DbSession,
) -> dict[str, Any]:
    """Get a single A/B test by ID."""
    mgr = _get_manager(request, session)
    return await mgr.get_test(test_id)


@router.get("/{test_id}/results")
async def get_results(
    test_id: str,
    request: Request,
    session: DbSession,
) -> dict[str, Any]:
    """Get aggregated metric results for an A/B test."""
    mgr = _get_manager(request, session)
    return await mgr.get_results(test_id)


@router.post("/{test_id}/conclude")
async def conclude_test(
    test_id: str,
    body: ConcludeTestRequest,
    request: Request,
    session: DbSession,
) -> dict[str, Any]:
    """Conclude an A/B test, optionally declaring a winner."""
    mgr = _get_manager(request, session)
    return await mgr.conclude_test(
        test_id,
        winner=body.winner,
        reason=body.reason,
    )


@router.post("/{test_id}/pause")
async def pause_test(
    test_id: str,
    request: Request,
    session: DbSession,
) -> dict[str, Any]:
    """Pause a running A/B test."""
    mgr = _get_manager(request, session)
    return await mgr.pause_test(test_id)


@router.post("/{test_id}/resume")
async def resume_test(
    test_id: str,
    request: Request,
    session: DbSession,
) -> dict[str, Any]:
    """Resume a paused A/B test."""
    mgr = _get_manager(request, session)
    return await mgr.resume_test(test_id)
