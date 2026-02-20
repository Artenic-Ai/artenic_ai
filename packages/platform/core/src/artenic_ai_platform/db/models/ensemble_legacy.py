"""Legacy ensemble and optimizer ORM models."""

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


class EnsembleRecord(Base):
    """Ensemble of models with a routing strategy."""

    __tablename__ = "ensembles"

    id: Mapped[str] = mapped_column(
        String(255),
        primary_key=True,
    )
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(
        Text,
        default="",
    )
    service: Mapped[str] = mapped_column(String(255))
    strategy: Mapped[str] = mapped_column(String(50))
    strategy_config: Mapped[dict] = mapped_column(  # type: ignore[type-arg]
        JSON,
        default=dict,
    )
    model_ids: Mapped[list] = mapped_column(  # type: ignore[type-arg]
        JSON,
        default=list,
    )
    stage: Mapped[str] = mapped_column(
        String(50),
        default="registered",
    )
    version: Mapped[int] = mapped_column(
        Integer,
        default=1,
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


class EnsembleJobRecord(Base):
    """Aggregate training job that spawns per-model jobs."""

    __tablename__ = "ensemble_jobs"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
    )
    ensemble_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("ensembles.id"),
    )
    status: Mapped[str] = mapped_column(
        String(50),
        default="pending",
    )
    training_job_ids: Mapped[list] = mapped_column(  # type: ignore[type-arg]
        JSON,
        default=list,
    )
    total_models: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )
    completed_models: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )
    failed_models: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )
    total_cost_eur: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class EnsembleVersionRecord(Base):
    """Immutable snapshot of an ensemble at a given version."""

    __tablename__ = "ensemble_versions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    ensemble_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("ensembles.id"),
    )
    version: Mapped[int] = mapped_column(Integer)
    model_ids: Mapped[list] = mapped_column(  # type: ignore[type-arg]
        JSON,
    )
    strategy: Mapped[str] = mapped_column(String(50))
    strategy_config: Mapped[dict] = mapped_column(  # type: ignore[type-arg]
        JSON,
        default=dict,
    )
    change_reason: Mapped[str] = mapped_column(
        String(255),
        default="",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class OptimizerRecommendationRecord(Base):
    """Instance-selection recommendation from the optimizer."""

    __tablename__ = "optimizer_recommendations"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
    )
    training_job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "training_jobs.id",
            ondelete="CASCADE",
        ),
        unique=True,
    )
    model_version: Mapped[str] = mapped_column(
        String(50),
    )
    prediction_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    recommended_provider: Mapped[str] = mapped_column(
        String(50),
    )
    recommended_instance: Mapped[str] = mapped_column(
        String(100),
    )
    recommended_rank: Mapped[int] = mapped_column(
        Integer,
        default=1,
    )
    estimated_duration_hours: Mapped[float] = mapped_column(
        Float,
    )
    estimated_cost_eur: Mapped[float] = mapped_column(
        Float,
    )
    confidence_score: Mapped[float] = mapped_column(Float)
    alternatives: Mapped[list] = mapped_column(  # type: ignore[type-arg]
        JSON,
    )
    total_instances_considered: Mapped[int] = mapped_column(
        Integer,
    )
    instances_after_filter: Mapped[int] = mapped_column(
        Integer,
    )
    workload_spec: Mapped[dict] = mapped_column(  # type: ignore[type-arg]
        JSON,
    )


class OptimizerTrainingSampleRecord(Base):
    """LTR training sample generated by the optimizer."""

    __tablename__ = "optimizer_training_samples"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
    )
    recommendation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "optimizer_recommendations.id",
            ondelete="CASCADE",
        ),
    )
    sample_type: Mapped[str] = mapped_column(String(20))
    query_id: Mapped[str] = mapped_column(String(36))
    features: Mapped[dict] = mapped_column(  # type: ignore[type-arg]
        JSON,
    )
    relevance_label: Mapped[int] = mapped_column(Integer)
    workload_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
