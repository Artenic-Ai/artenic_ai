"""Comprehensive tests for artenic_ai_platform.budget.service — BudgetManager."""

from __future__ import annotations

import uuid
from datetime import datetime  # noqa: TC003

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from artenic_ai_platform.budget.service import (
    BudgetCheckResult,
    BudgetExceededError,
    BudgetManager,
    SpendingSummary,
)
from artenic_ai_platform.db.models import (
    Base,
    BudgetAlertRecord,
    BudgetRecord,
    TrainingJob,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
async def session() -> AsyncSession:  # type: ignore[misc]
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as sess:
        yield sess  # type: ignore[misc]
    await engine.dispose()


@pytest.fixture
def manager(session: AsyncSession) -> BudgetManager:
    """Default BudgetManager in 'block' enforcement mode."""
    return BudgetManager(session, enforcement_mode="block", alert_threshold_pct=80.0)


@pytest.fixture
def warn_manager(session: AsyncSession) -> BudgetManager:
    """BudgetManager in 'warn' enforcement mode."""
    return BudgetManager(session, enforcement_mode="warn", alert_threshold_pct=80.0)


async def _insert_training_job(
    session: AsyncSession,
    *,
    service: str = "sentiment",
    provider: str = "mock",
    cost_actual_eur: float | None = None,
    created_at: datetime | None = None,
) -> TrainingJob:
    """Helper to insert a TrainingJob into the DB."""
    job = TrainingJob(
        id=uuid.uuid4().hex[:36],
        service=service,
        model="bert-base",
        provider=provider,
        cost_actual_eur=cost_actual_eur,
    )
    session.add(job)
    await session.flush()

    # Override created_at if requested (for time-window tests)
    if created_at is not None:
        job.created_at = created_at
        await session.flush()

    await session.commit()
    return job


# ======================================================================
# TestCreateBudget
# ======================================================================


class TestCreateBudget:
    async def test_create_budget(self, session: AsyncSession, manager: BudgetManager) -> None:
        """Creates a budget with all fields and returns a correct dict."""
        result = await manager.create_budget(
            scope="global",
            scope_value="*",
            period="monthly",
            limit_eur=1000.0,
            alert_threshold_pct=90.0,
        )

        assert result["scope"] == "global"
        assert result["scope_value"] == "*"
        assert result["period"] == "monthly"
        assert result["limit_eur"] == 1000.0
        assert result["alert_threshold_pct"] == 90.0
        assert result["enabled"] is True
        assert result["id"] is not None
        assert result["created_at"] is not None

        # Verify it was persisted in the DB
        row = await session.execute(select(BudgetRecord).where(BudgetRecord.id == result["id"]))
        budget = row.scalar_one()
        assert budget.limit_eur == 1000.0

    async def test_create_budget_default_threshold(
        self, session: AsyncSession, manager: BudgetManager
    ) -> None:
        """When alert_threshold_pct is not provided, uses the manager default."""
        result = await manager.create_budget(
            scope="service",
            scope_value="sentiment",
            period="daily",
            limit_eur=50.0,
        )
        assert result["alert_threshold_pct"] == 80.0

    async def test_create_budget_invalid_period(self, manager: BudgetManager) -> None:
        """Raises ValueError for an unsupported period."""
        with pytest.raises(ValueError, match="Invalid period 'yearly'"):
            await manager.create_budget(
                scope="global",
                scope_value="*",
                period="yearly",
                limit_eur=100.0,
            )

    async def test_create_budget_negative_limit(self, manager: BudgetManager) -> None:
        """Raises ValueError for a non-positive limit."""
        with pytest.raises(ValueError, match="Budget limit must be positive"):
            await manager.create_budget(
                scope="global",
                scope_value="*",
                period="daily",
                limit_eur=-10.0,
            )

    async def test_create_budget_zero_limit(self, manager: BudgetManager) -> None:
        """Raises ValueError when limit is exactly zero."""
        with pytest.raises(ValueError, match="Budget limit must be positive"):
            await manager.create_budget(
                scope="global",
                scope_value="*",
                period="weekly",
                limit_eur=0.0,
            )


# ======================================================================
# TestUpdateBudget
# ======================================================================


class TestUpdateBudget:
    async def test_update_budget(self, session: AsyncSession, manager: BudgetManager) -> None:
        """Updates limit_eur and verifies the change is persisted."""
        created = await manager.create_budget(
            scope="global", scope_value="*", period="monthly", limit_eur=500.0
        )
        updated = await manager.update_budget(created["id"], {"limit_eur": 750.0})
        assert updated["limit_eur"] == 750.0
        assert updated["id"] == created["id"]

    async def test_update_budget_not_found(self, manager: BudgetManager) -> None:
        """Raises KeyError when budget_id does not exist."""
        with pytest.raises(KeyError, match="Budget 99999 not found"):
            await manager.update_budget(99999, {"limit_eur": 100.0})

    async def test_update_budget_invalid_period(self, manager: BudgetManager) -> None:
        """Raises ValueError when updating period to an invalid value."""
        created = await manager.create_budget(
            scope="global", scope_value="*", period="monthly", limit_eur=500.0
        )
        with pytest.raises(ValueError, match="Invalid period 'hourly'"):
            await manager.update_budget(created["id"], {"period": "hourly"})

    async def test_update_budget_no_changes(self, manager: BudgetManager) -> None:
        """When no allowed fields are in updates, returns unchanged budget."""
        created = await manager.create_budget(
            scope="service",
            scope_value="nlp",
            period="weekly",
            limit_eur=200.0,
        )
        unchanged = await manager.update_budget(created["id"], {"unknown_field": "whatever"})
        assert unchanged["limit_eur"] == 200.0
        assert unchanged["period"] == "weekly"
        assert unchanged["id"] == created["id"]

    async def test_update_budget_multiple_fields(self, manager: BudgetManager) -> None:
        """Updates multiple allowed fields in a single call."""
        created = await manager.create_budget(
            scope="provider",
            scope_value="gcp",
            period="daily",
            limit_eur=100.0,
        )
        updated = await manager.update_budget(
            created["id"],
            {
                "limit_eur": 250.0,
                "alert_threshold_pct": 95.0,
                "enabled": False,
                "period": "weekly",
            },
        )
        assert updated["limit_eur"] == 250.0
        assert updated["alert_threshold_pct"] == 95.0
        assert updated["enabled"] is False
        assert updated["period"] == "weekly"


# ======================================================================
# TestListBudgets
# ======================================================================


class TestListBudgets:
    async def test_list_empty(self, manager: BudgetManager) -> None:
        """Returns empty list when no budgets exist."""
        result = await manager.list_budgets()
        assert result == []

    async def test_list_with_budgets(self, manager: BudgetManager) -> None:
        """Returns all enabled budgets."""
        await manager.create_budget(
            scope="global", scope_value="*", period="monthly", limit_eur=1000.0
        )
        await manager.create_budget(
            scope="service", scope_value="nlp", period="daily", limit_eur=50.0
        )

        result = await manager.list_budgets()
        assert len(result) == 2

    async def test_list_filter_by_scope(self, manager: BudgetManager) -> None:
        """Filters budgets by scope."""
        await manager.create_budget(
            scope="global", scope_value="*", period="monthly", limit_eur=1000.0
        )
        await manager.create_budget(
            scope="service", scope_value="nlp", period="daily", limit_eur=50.0
        )
        await manager.create_budget(
            scope="service", scope_value="vision", period="weekly", limit_eur=200.0
        )

        global_budgets = await manager.list_budgets(scope="global")
        assert len(global_budgets) == 1
        assert global_budgets[0]["scope"] == "global"

        service_budgets = await manager.list_budgets(scope="service")
        assert len(service_budgets) == 2
        assert all(b["scope"] == "service" for b in service_budgets)

    async def test_list_enabled_only(self, session: AsyncSession, manager: BudgetManager) -> None:
        """Filters to enabled-only budgets by default; disabled are excluded."""
        b1 = await manager.create_budget(
            scope="global", scope_value="*", period="monthly", limit_eur=1000.0
        )
        await manager.create_budget(
            scope="service", scope_value="nlp", period="daily", limit_eur=50.0
        )

        # Disable the first budget
        await manager.update_budget(b1["id"], {"enabled": False})

        enabled = await manager.list_budgets(enabled_only=True)
        assert len(enabled) == 1
        assert enabled[0]["scope"] == "service"

        # Get all (including disabled)
        all_budgets = await manager.list_budgets(enabled_only=False)
        assert len(all_budgets) == 2


# ======================================================================
# TestCheckBudget
# ======================================================================


class TestCheckBudget:
    async def test_check_no_budgets(self, manager: BudgetManager) -> None:
        """When no budgets exist for the scope, returns allowed with zero limits."""
        result = await manager.check_budget("global", "*", 100.0)

        assert isinstance(result, BudgetCheckResult)
        assert result.allowed is True
        assert result.spent_eur == 0.0
        assert result.limit_eur == 0.0
        assert result.remaining_eur == 0.0
        assert result.pct_used == 0.0
        assert result.warning is None
        assert result.budget_id is None

    async def test_check_under_limit(self, session: AsyncSession, manager: BudgetManager) -> None:
        """When spending is under the limit, returns allowed."""
        await manager.create_budget(
            scope="service", scope_value="sentiment", period="monthly", limit_eur=1000.0
        )
        # Insert a training job with some cost
        await _insert_training_job(session, service="sentiment", cost_actual_eur=200.0)

        result = await manager.check_budget("service", "sentiment", 100.0)

        assert result.allowed is True
        assert result.pct_used == 0.0  # The "all budgets passed" path

    async def test_check_over_limit_block(
        self, session: AsyncSession, manager: BudgetManager
    ) -> None:
        """In block mode, raises BudgetExceededError when over limit."""
        await manager.create_budget(
            scope="service",
            scope_value="sentiment",
            period="monthly",
            limit_eur=100.0,
        )
        # Insert spending that nearly reaches the limit
        await _insert_training_job(session, service="sentiment", cost_actual_eur=90.0)

        with pytest.raises(BudgetExceededError) as exc_info:
            await manager.check_budget("service", "sentiment", 20.0)

        err = exc_info.value
        assert err.result.allowed is False
        assert err.result.budget_id is not None
        assert "Budget exceeded" in str(err)
        assert err.result.spent_eur == 90.0
        assert err.result.limit_eur == 100.0

    async def test_check_over_limit_warn(
        self, session: AsyncSession, warn_manager: BudgetManager
    ) -> None:
        """In warn mode, returns a warning but does not raise."""
        await warn_manager.create_budget(
            scope="service",
            scope_value="sentiment",
            period="monthly",
            limit_eur=100.0,
        )
        await _insert_training_job(session, service="sentiment", cost_actual_eur=90.0)

        result = await warn_manager.check_budget("service", "sentiment", 20.0)

        assert result.allowed is False
        assert result.warning is not None
        assert "Budget warning" in result.warning
        assert "exceeds" in result.warning
        assert result.budget_id is not None

    async def test_check_alert_threshold(
        self, session: AsyncSession, manager: BudgetManager
    ) -> None:
        """Fires a BudgetAlertRecord when the alert threshold is crossed."""
        created = await manager.create_budget(
            scope="service",
            scope_value="sentiment",
            period="monthly",
            limit_eur=100.0,
            alert_threshold_pct=50.0,
        )
        # Spending that puts us at 60% when projected (50 existing + 10 new = 60%)
        await _insert_training_job(session, service="sentiment", cost_actual_eur=50.0)

        # 50 + 10 = 60 => 60% which exceeds 50% threshold but is under 100 limit
        result = await manager.check_budget("service", "sentiment", 10.0)

        assert result.allowed is True

        # Verify an alert was created
        alert_result = await session.execute(
            select(BudgetAlertRecord).where(BudgetAlertRecord.budget_id == created["id"])
        )
        alerts = alert_result.scalars().all()
        assert len(alerts) == 1
        assert alerts[0].alert_type == "threshold_exceeded"
        assert alerts[0].pct_used == pytest.approx(60.0)
        assert alerts[0].spent_eur == 50.0
        assert alerts[0].limit_eur == 100.0
        assert alerts[0].webhook_sent is False
        assert "sentiment" in alerts[0].message

    async def test_check_provider_scope(
        self, session: AsyncSession, manager: BudgetManager
    ) -> None:
        """Budget check works for 'provider' scope."""
        await manager.create_budget(
            scope="provider",
            scope_value="mock",
            period="daily",
            limit_eur=50.0,
        )
        await _insert_training_job(session, provider="mock", cost_actual_eur=10.0)

        result = await manager.check_budget("provider", "mock", 5.0)
        assert result.allowed is True

    async def test_check_global_scope(self, session: AsyncSession, manager: BudgetManager) -> None:
        """Budget check works for 'global' scope (no scope filter on jobs)."""
        await manager.create_budget(
            scope="global",
            scope_value="*",
            period="monthly",
            limit_eur=500.0,
        )
        await _insert_training_job(session, service="nlp", provider="gcp", cost_actual_eur=100.0)
        await _insert_training_job(session, service="vision", provider="aws", cost_actual_eur=100.0)

        result = await manager.check_budget("global", "*", 50.0)
        assert result.allowed is True


# ======================================================================
# TestRecordSpending
# ======================================================================


class TestRecordSpending:
    async def test_record_spending(self, session: AsyncSession, manager: BudgetManager) -> None:
        """Updates training job cost via record_spending."""
        job = await _insert_training_job(session, service="sentiment")

        await manager.record_spending(
            job_id=job.id,
            cost_eur=42.50,
            service="sentiment",
            provider="mock",
        )

        # Refresh and verify
        await session.refresh(job)
        assert job.cost_actual_eur == 42.50


# ======================================================================
# TestGetSpending
# ======================================================================


class TestGetSpending:
    async def test_get_spending_with_budget(
        self, session: AsyncSession, manager: BudgetManager
    ) -> None:
        """Returns spending summaries for active budgets."""
        await manager.create_budget(
            scope="service",
            scope_value="sentiment",
            period="monthly",
            limit_eur=1000.0,
        )
        await _insert_training_job(session, service="sentiment", cost_actual_eur=300.0)

        summaries = await manager.get_spending("service", "sentiment")

        assert len(summaries) == 1
        s = summaries[0]
        assert isinstance(s, SpendingSummary)
        assert s.scope == "service"
        assert s.scope_value == "sentiment"
        assert s.period == "monthly"
        assert s.spent_eur == 300.0
        assert s.limit_eur == 1000.0
        assert s.pct_used == pytest.approx(30.0)
        assert s.alerts == []

    async def test_get_spending_with_alerts(
        self, session: AsyncSession, manager: BudgetManager
    ) -> None:
        """Returns spending summaries including recent alerts."""
        await manager.create_budget(
            scope="service",
            scope_value="sentiment",
            period="monthly",
            limit_eur=100.0,
            alert_threshold_pct=50.0,
        )
        await _insert_training_job(session, service="sentiment", cost_actual_eur=60.0)

        # Trigger an alert by checking budget (60 + 5 = 65% > 50% threshold)
        await manager.check_budget("service", "sentiment", 5.0)
        await session.commit()

        summaries = await manager.get_spending("service", "sentiment")

        assert len(summaries) == 1
        s = summaries[0]
        assert len(s.alerts) >= 1
        alert = s.alerts[0]
        assert alert["alert_type"] == "threshold_exceeded"
        assert alert["id"] is not None
        assert alert["spent_eur"] == 60.0
        assert alert["webhook_sent"] is False

    async def test_get_spending_no_budgets(self, manager: BudgetManager) -> None:
        """Returns empty list when no budgets match the scope."""
        summaries = await manager.get_spending("service", "nonexistent")
        assert summaries == []


# ======================================================================
# TestBudgetExceededError
# ======================================================================


class TestBudgetExceededError:
    def test_error_message_format(self) -> None:
        """BudgetExceededError formats a human-readable message."""
        result = BudgetCheckResult(
            allowed=False,
            spent_eur=90.0,
            limit_eur=100.0,
            remaining_eur=10.0,
            pct_used=110.0,
            budget_id=1,
        )
        error = BudgetExceededError(result)
        assert "110.0% used" in str(error)
        assert "90.00/100.00 EUR" in str(error)
        assert error.result is result
