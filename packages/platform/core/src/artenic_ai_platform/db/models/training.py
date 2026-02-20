"""Training ORM models."""

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


class TrainingJob(Base):
    """Represents a single model training run."""

    __tablename__ = "training_jobs"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
    )
    service: Mapped[str] = mapped_column(String(255))
    model: Mapped[str] = mapped_column(String(255))
    provider: Mapped[str] = mapped_column(String(50))
    config: Mapped[dict] = mapped_column(  # type: ignore[type-arg]
        JSON,
        default=dict,
    )
    status: Mapped[str] = mapped_column(
        String(50),
        default="pending",
    )
    metrics: Mapped[dict | None] = mapped_column(  # type: ignore[type-arg]
        JSON,
        nullable=True,
    )
    error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    mlflow_run_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    provider_job_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    instance_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )
    cost_estimate_eur: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    artifacts_uri: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    cost_actual_eur: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    cost_per_hour_eur: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    duration_seconds: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    cost_predicted_eur: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    duration_predicted_hours: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    primary_metric_before: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    primary_metric_after: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    primary_metric_name: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )
    is_spot: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )
    preempted: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )
    resumed_from_job_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
    )
    preemption_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )
    region: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )
    workload_spec: Mapped[dict | None] = mapped_column(  # type: ignore[type-arg]
        JSON,
        nullable=True,
    )
    ensemble_job_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("ensemble_jobs.id"),
        nullable=True,
    )


class TrainingOutcomeRecord(Base):
    """Post-hoc outcome for a completed training job."""

    __tablename__ = "training_outcomes"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("training_jobs.id"),
        unique=True,
    )
    workload_spec: Mapped[dict] = mapped_column(  # type: ignore[type-arg]
        JSON,
    )
    provider: Mapped[str] = mapped_column(String(50))
    instance_type: Mapped[str] = mapped_column(String(100))
    region: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )
    is_spot: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )
    actual_duration_hours: Mapped[float] = mapped_column(
        Float,
    )
    actual_cost_eur: Mapped[float] = mapped_column(Float)
    predicted_duration_hours: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    predicted_cost_eur: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    success: Mapped[bool] = mapped_column(Boolean)
    primary_metric_name: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )
    primary_metric_value: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
