"""Monitoring ORM models (health, budget, A/B testing)."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 - SQLAlchemy needs runtime access

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from artenic_ai_platform.db.models.base import Base


class ModelHealthRecord(Base):
    """Health / drift metric snapshot for a registered model."""

    __tablename__ = "model_health"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    model_id: Mapped[str] = mapped_column(
        String(500),
        ForeignKey("ml_models.id"),
    )
    metric_name: Mapped[str] = mapped_column(String(100))
    metric_value: Mapped[float] = mapped_column(Float)
    drift_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    alert_triggered: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )
    window_start: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    window_end: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    sample_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class BudgetRecord(Base):
    """Spending budget for a given scope and period."""

    __tablename__ = "budgets"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    scope: Mapped[str] = mapped_column(String(50))
    scope_value: Mapped[str] = mapped_column(String(255))
    period: Mapped[str] = mapped_column(String(20))
    limit_eur: Mapped[float] = mapped_column(Float)
    alert_threshold_pct: Mapped[float] = mapped_column(
        Float,
        default=80.0,
    )
    enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        onupdate=func.now(),
    )


class BudgetAlertRecord(Base):
    """Alert triggered when budget thresholds are breached."""

    __tablename__ = "budget_alerts"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    budget_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("budgets.id"),
    )
    alert_type: Mapped[str] = mapped_column(String(50))
    spent_eur: Mapped[float] = mapped_column(Float)
    limit_eur: Mapped[float] = mapped_column(Float)
    pct_used: Mapped[float] = mapped_column(Float)
    message: Mapped[str] = mapped_column(Text)
    webhook_sent: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class ABTestRecord(Base):
    """A/B test comparing model variants on a service."""

    __tablename__ = "ab_tests"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
    )
    name: Mapped[str] = mapped_column(String(255))
    service: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(
        String(50),
        default="running",
    )
    variants: Mapped[dict] = mapped_column(  # type: ignore[type-arg]
        JSON,
    )
    primary_metric: Mapped[str] = mapped_column(
        String(100),
    )
    min_samples: Mapped[int] = mapped_column(
        Integer,
        default=100,
    )
    winner: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    conclusion_reason: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    concluded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class ABTestMetricRecord(Base):
    """Individual metric observation for an A/B test variant."""

    __tablename__ = "ab_test_metrics"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    ab_test_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("ab_tests.id"),
    )
    variant_name: Mapped[str] = mapped_column(String(255))
    metric_name: Mapped[str] = mapped_column(String(100))
    metric_value: Mapped[float] = mapped_column(Float)
    latency_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    error: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
