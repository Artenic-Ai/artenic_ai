"""SQLAlchemy 2.0 ORM models for the Artenic AI platform."""

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
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
)


class Base(DeclarativeBase):
    """Declarative base for all Artenic AI platform models."""


# ------------------------------------------------------------------
# 9. EnsembleRecord  (must be defined early — referenced by FK)
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# 11. EnsembleJobRecord  (referenced by TrainingJob FK)
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# 1. RegisteredModel
# ------------------------------------------------------------------


class RegisteredModel(Base):
    """A registered ML model with metadata and lineage."""

    __tablename__ = "artenic_models"

    id: Mapped[str] = mapped_column(
        String(255),
        primary_key=True,
    )
    name: Mapped[str] = mapped_column(String(255))
    version: Mapped[str] = mapped_column(String(50))
    model_type: Mapped[str] = mapped_column(String(100))
    framework: Mapped[str] = mapped_column(String(50))
    description: Mapped[str] = mapped_column(
        Text,
        default="",
    )
    tags: Mapped[dict] = mapped_column(  # type: ignore[type-arg]
        JSON,
        default=dict,
    )
    input_features: Mapped[list] = mapped_column(  # type: ignore[type-arg]
        JSON,
        default=list,
    )
    output_schema: Mapped[dict] = mapped_column(  # type: ignore[type-arg]
        JSON,
        default=dict,
    )
    stage: Mapped[str] = mapped_column(
        String(50),
        default="registered",
    )
    mlflow_run_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    mlflow_model_uri: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )
    ensemble_id: Mapped[str | None] = mapped_column(
        String(255),
        ForeignKey("ensembles.id"),
        nullable=True,
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


# ------------------------------------------------------------------
# 2. PromotionRecord
# ------------------------------------------------------------------


class PromotionRecord(Base):
    """Tracks stage promotions for registered models."""

    __tablename__ = "promotions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    model_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("artenic_models.id"),
    )
    from_stage: Mapped[str] = mapped_column(String(50))
    to_stage: Mapped[str] = mapped_column(String(50))
    version: Mapped[str] = mapped_column(String(50))
    promoted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


# ------------------------------------------------------------------
# 3. TrainingJob
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# 4. BudgetRecord
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# 5. BudgetAlertRecord
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# 6. TrainingOutcomeRecord
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# 7. OptimizerRecommendationRecord
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# 8. OptimizerTrainingSampleRecord
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# 10. EnsembleVersionRecord
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# 12. ABTestRecord
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# 13. ABTestMetricRecord
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# 14. ModelHealthRecord
# ------------------------------------------------------------------


class ModelHealthRecord(Base):
    """Health / drift metric snapshot for a registered model."""

    __tablename__ = "model_health"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    model_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("artenic_models.id"),
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


# ------------------------------------------------------------------
# 15. ConfigSettingRecord
# ------------------------------------------------------------------


class ConfigSettingRecord(Base):
    """Persisted configuration key-value pair."""

    __tablename__ = "config_settings"
    __table_args__ = (UniqueConstraint("scope", "section", "key"),)

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    scope: Mapped[str] = mapped_column(String(50))
    section: Mapped[str] = mapped_column(String(100))
    key: Mapped[str] = mapped_column(String(100))
    value: Mapped[str] = mapped_column(Text)
    encrypted: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_by: Mapped[str] = mapped_column(
        String(100),
        default="system",
    )


# ------------------------------------------------------------------
# 16. ConfigAuditRecord
# ------------------------------------------------------------------


class ConfigAuditRecord(Base):
    """Audit log entry for configuration changes."""

    __tablename__ = "config_audit_log"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    scope: Mapped[str] = mapped_column(String(50))
    section: Mapped[str] = mapped_column(String(100))
    key: Mapped[str] = mapped_column(String(100))
    old_value: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    new_value: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    action: Mapped[str] = mapped_column(String(20))
    changed_by: Mapped[str] = mapped_column(
        String(100),
        default="api",
    )
    changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


# ------------------------------------------------------------------
# 17. ConfigOverrideRecord  (legacy)
# ------------------------------------------------------------------


class ConfigOverrideRecord(Base):
    """Legacy configuration override (section + key)."""

    __tablename__ = "config_overrides"
    __table_args__ = (UniqueConstraint("section", "key"),)

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    section: Mapped[str] = mapped_column(String(100))
    key: Mapped[str] = mapped_column(String(100))
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
