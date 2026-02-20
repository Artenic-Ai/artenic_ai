"""Budget governance — multi-scope, multi-period spending control."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select, update

from artenic_ai_platform.db.models import (
    BudgetAlertRecord,
    BudgetRecord,
    TrainingJob,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

PERIOD_DAYS: dict[str, int] = {
    "daily": 1,
    "weekly": 7,
    "monthly": 30,
}


@dataclass
class BudgetCheckResult:
    """Result of a budget check."""

    allowed: bool
    spent_eur: float
    limit_eur: float
    remaining_eur: float
    pct_used: float
    warning: str | None = None
    budget_id: int | None = None


@dataclass
class SpendingSummary:
    """Aggregated spending for a scope."""

    scope: str
    scope_value: str
    period: str
    spent_eur: float
    limit_eur: float
    pct_used: float
    alerts: list[dict[str, Any]] = field(default_factory=list)


class BudgetExceededError(Exception):
    """Raised when a budget limit is exceeded in block mode."""

    def __init__(self, result: BudgetCheckResult) -> None:
        self.result = result
        super().__init__(
            f"Budget exceeded: {result.pct_used:.1f}% used "
            f"({result.spent_eur:.2f}/{result.limit_eur:.2f} EUR)"
        )


# ------------------------------------------------------------------
# Service
# ------------------------------------------------------------------


class BudgetManager:
    """Multi-scope, multi-period budget governance.

    Scopes: "global", "service", "provider"
    Periods: "daily", "weekly", "monthly"
    Enforcement: "block" raises BudgetExceededError, "warn" returns warning
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        enforcement_mode: str = "block",
        alert_threshold_pct: float = 80.0,
    ) -> None:
        self._session = session
        self._enforcement = enforcement_mode
        self._alert_threshold = alert_threshold_pct

    # ------------------------------------------------------------------
    # Budget CRUD
    # ------------------------------------------------------------------

    async def create_budget(
        self,
        scope: str,
        scope_value: str,
        period: str,
        limit_eur: float,
        *,
        alert_threshold_pct: float | None = None,
    ) -> dict[str, Any]:
        """Create a new budget rule."""
        if period not in PERIOD_DAYS:
            msg = f"Invalid period '{period}'. Must be one of: {', '.join(PERIOD_DAYS)}"
            raise ValueError(msg)
        if limit_eur <= 0:
            msg = "Budget limit must be positive"
            raise ValueError(msg)

        budget = BudgetRecord(
            scope=scope,
            scope_value=scope_value,
            period=period,
            limit_eur=limit_eur,
            alert_threshold_pct=alert_threshold_pct or self._alert_threshold,
            enabled=True,
        )
        self._session.add(budget)
        await self._session.flush()
        await self._session.commit()
        return self._budget_to_dict(budget)

    async def update_budget(
        self,
        budget_id: int,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        """Update a budget rule."""
        result = await self._session.execute(
            select(BudgetRecord).where(BudgetRecord.id == budget_id)
        )
        budget = result.scalar_one_or_none()
        if budget is None:
            msg = f"Budget {budget_id} not found"
            raise KeyError(msg)

        allowed_fields = {
            "limit_eur",
            "alert_threshold_pct",
            "enabled",
            "period",
        }
        values = {k: v for k, v in updates.items() if k in allowed_fields}
        if not values:
            return self._budget_to_dict(budget)

        if "period" in values and values["period"] not in PERIOD_DAYS:
            msg = f"Invalid period '{values['period']}'"
            raise ValueError(msg)

        await self._session.execute(
            update(BudgetRecord).where(BudgetRecord.id == budget_id).values(**values)
        )
        await self._session.commit()

        result = await self._session.execute(
            select(BudgetRecord).where(BudgetRecord.id == budget_id)
        )
        return self._budget_to_dict(result.scalar_one())

    async def list_budgets(
        self,
        *,
        scope: str | None = None,
        enabled_only: bool = True,
    ) -> list[dict[str, Any]]:
        """List all budget rules with optional filters."""
        stmt = select(BudgetRecord)
        if scope:
            stmt = stmt.where(BudgetRecord.scope == scope)
        if enabled_only:
            stmt = stmt.where(BudgetRecord.enabled.is_(True))
        stmt = stmt.order_by(BudgetRecord.created_at.desc())

        result = await self._session.execute(stmt)
        return [self._budget_to_dict(b) for b in result.scalars().all()]

    # ------------------------------------------------------------------
    # Budget check
    # ------------------------------------------------------------------

    async def check_budget(
        self,
        scope: str,
        scope_value: str,
        estimated_cost: float,
    ) -> BudgetCheckResult:
        """Check if estimated spending fits within active budgets.

        In "block" mode, raises BudgetExceededError when over limit.
        In "warn" mode, returns a warning but allows the operation.
        """
        # Find active budgets for this scope
        stmt = select(BudgetRecord).where(
            BudgetRecord.scope == scope,
            BudgetRecord.scope_value == scope_value,
            BudgetRecord.enabled.is_(True),
        )
        result = await self._session.execute(stmt)
        budgets = result.scalars().all()

        if not budgets:
            return BudgetCheckResult(
                allowed=True,
                spent_eur=0.0,
                limit_eur=0.0,
                remaining_eur=0.0,
                pct_used=0.0,
            )

        # Check each budget — if any blocks, the operation is denied
        for budget in budgets:
            spent = await self._compute_spending(scope, scope_value, budget.period)
            projected = spent + estimated_cost
            pct = (projected / budget.limit_eur * 100) if budget.limit_eur > 0 else 0.0

            check = BudgetCheckResult(
                allowed=projected <= budget.limit_eur,
                spent_eur=spent,
                limit_eur=budget.limit_eur,
                remaining_eur=max(0.0, budget.limit_eur - spent),
                pct_used=pct,
                budget_id=budget.id,
            )

            # Fire alert if threshold crossed
            if pct >= budget.alert_threshold_pct:
                await self._fire_alert(budget, spent, pct)

            if not check.allowed:
                if self._enforcement == "block":
                    raise BudgetExceededError(check)
                check.warning = (
                    f"Budget warning: projected {projected:.2f} EUR "
                    f"exceeds {budget.limit_eur:.2f} EUR limit"
                )
                return check

        # All budgets passed
        return BudgetCheckResult(
            allowed=True,
            spent_eur=0.0,
            limit_eur=0.0,
            remaining_eur=0.0,
            pct_used=0.0,
        )

    # ------------------------------------------------------------------
    # Spending
    # ------------------------------------------------------------------

    async def record_spending(
        self,
        job_id: str,
        cost_eur: float,
        service: str,
        provider: str,
    ) -> None:
        """Record actual spending after job completion (updates TrainingJob)."""
        await self._session.execute(
            update(TrainingJob).where(TrainingJob.id == job_id).values(cost_actual_eur=cost_eur)
        )
        await self._session.commit()
        logger.info(
            "Recorded %.2f EUR for job %s (%s/%s)",
            cost_eur,
            job_id,
            service,
            provider,
        )

    async def get_spending(
        self,
        scope: str,
        scope_value: str,
    ) -> list[SpendingSummary]:
        """Get spending summaries for all active budgets of a scope."""
        stmt = select(BudgetRecord).where(
            BudgetRecord.scope == scope,
            BudgetRecord.scope_value == scope_value,
            BudgetRecord.enabled.is_(True),
        )
        result = await self._session.execute(stmt)
        budgets = result.scalars().all()

        summaries = []
        for budget in budgets:
            spent = await self._compute_spending(scope, scope_value, budget.period)
            pct = (spent / budget.limit_eur * 100) if budget.limit_eur > 0 else 0.0

            # Fetch recent alerts
            alert_stmt = (
                select(BudgetAlertRecord)
                .where(BudgetAlertRecord.budget_id == budget.id)
                .order_by(BudgetAlertRecord.created_at.desc())
                .limit(5)
            )
            alert_result = await self._session.execute(alert_stmt)
            alerts = [self._alert_to_dict(a) for a in alert_result.scalars().all()]

            summaries.append(
                SpendingSummary(
                    scope=scope,
                    scope_value=scope_value,
                    period=budget.period,
                    spent_eur=spent,
                    limit_eur=budget.limit_eur,
                    pct_used=pct,
                    alerts=alerts,
                )
            )
        return summaries

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _compute_spending(
        self,
        scope: str,
        scope_value: str,
        period: str,
    ) -> float:
        """Compute actual spending in the current period window."""
        days = PERIOD_DAYS.get(period, 30)
        cutoff = datetime.now(UTC) - timedelta(days=days)

        stmt = select(func.coalesce(func.sum(TrainingJob.cost_actual_eur), 0.0))

        if scope == "service":
            stmt = stmt.where(TrainingJob.service == scope_value)
        elif scope == "provider":
            stmt = stmt.where(TrainingJob.provider == scope_value)
        # "global" → no filter on scope, just time window

        stmt = stmt.where(TrainingJob.created_at >= cutoff)

        result = await self._session.execute(stmt)
        value = result.scalar()
        return float(value) if value else 0.0

    async def _fire_alert(
        self,
        budget: BudgetRecord,
        spent_eur: float,
        pct_used: float,
    ) -> None:
        """Create a budget alert record."""
        alert = BudgetAlertRecord(
            budget_id=budget.id,
            alert_type="threshold_exceeded",
            spent_eur=spent_eur,
            limit_eur=budget.limit_eur,
            pct_used=pct_used,
            message=(
                f"{budget.scope}/{budget.scope_value} "
                f"{budget.period} budget at {pct_used:.1f}% "
                f"({spent_eur:.2f}/{budget.limit_eur:.2f} EUR)"
            ),
            webhook_sent=False,
        )
        self._session.add(alert)
        await self._session.flush()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _budget_to_dict(budget: BudgetRecord) -> dict[str, Any]:
        return {
            "id": budget.id,
            "scope": budget.scope,
            "scope_value": budget.scope_value,
            "period": budget.period,
            "limit_eur": budget.limit_eur,
            "alert_threshold_pct": budget.alert_threshold_pct,
            "enabled": budget.enabled,
            "created_at": (budget.created_at.isoformat() if budget.created_at else None),
        }

    @staticmethod
    def _alert_to_dict(alert: BudgetAlertRecord) -> dict[str, Any]:
        return {
            "id": alert.id,
            "alert_type": alert.alert_type,
            "spent_eur": alert.spent_eur,
            "limit_eur": alert.limit_eur,
            "pct_used": alert.pct_used,
            "message": alert.message,
            "webhook_sent": alert.webhook_sent,
            "created_at": (alert.created_at.isoformat() if alert.created_at else None),
        }
