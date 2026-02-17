"""REST API for budget governance — /api/v1/budgets/*."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from artenic_ai_platform.budget.service import BudgetManager

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


async def _get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Yield a DB session from app state session_factory."""
    factory = request.app.state.session_factory
    async with factory() as session:
        yield session


DbSession = Annotated[AsyncSession, Depends(_get_db)]

router = APIRouter(prefix="/api/v1/budgets", tags=["budgets"])


# ------------------------------------------------------------------
# Request schemas
# ------------------------------------------------------------------


class CreateBudgetRequest(BaseModel):
    """Body for POST /budgets."""

    scope: str
    scope_value: str
    period: str
    limit_eur: float
    alert_threshold_pct: float | None = None


class UpdateBudgetRequest(BaseModel):
    """Body for PUT /budgets/{budget_id}."""

    limit_eur: float | None = None
    alert_threshold_pct: float | None = None
    enabled: bool | None = None
    period: str | None = None


# ------------------------------------------------------------------
# Dependency helpers
# ------------------------------------------------------------------


def _get_budget_manager(
    request: Request,
    session: Any,
) -> BudgetManager:
    """Build a BudgetManager from app state."""
    settings = request.app.state.settings
    return BudgetManager(
        session,
        enforcement_mode=settings.budget.enforcement_mode,
        alert_threshold_pct=settings.budget.alert_threshold_pct,
    )


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@router.get("")
async def list_budgets(
    request: Request,
    session: DbSession,
    scope: str | None = Query(default=None),
    enabled_only: bool = Query(default=True),
) -> list[dict[str, Any]]:
    """List all budget rules."""
    mgr = _get_budget_manager(request, session)
    return await mgr.list_budgets(scope=scope, enabled_only=enabled_only)


@router.post("")
async def create_budget(
    body: CreateBudgetRequest,
    request: Request,
    session: DbSession,
) -> dict[str, Any]:
    """Create a new budget rule."""
    mgr = _get_budget_manager(request, session)
    return await mgr.create_budget(
        scope=body.scope,
        scope_value=body.scope_value,
        period=body.period,
        limit_eur=body.limit_eur,
        alert_threshold_pct=body.alert_threshold_pct,
    )


@router.put("/{budget_id}")
async def update_budget(
    budget_id: int,
    body: UpdateBudgetRequest,
    request: Request,
    session: DbSession,
) -> dict[str, Any]:
    """Update a budget rule."""
    mgr = _get_budget_manager(request, session)
    updates = body.model_dump(exclude_none=True)
    return await mgr.update_budget(budget_id, updates)


@router.get("/spending")
async def get_spending(
    request: Request,
    session: DbSession,
    scope: str = Query(...),
    scope_value: str = Query(...),
) -> list[dict[str, Any]]:
    """Get current spending summary for a scope."""
    mgr = _get_budget_manager(request, session)
    summaries = await mgr.get_spending(scope, scope_value)
    return [
        {
            "scope": s.scope,
            "scope_value": s.scope_value,
            "period": s.period,
            "spent_eur": s.spent_eur,
            "limit_eur": s.limit_eur,
            "pct_used": s.pct_used,
            "alerts": s.alerts,
        }
        for s in summaries
    ]
