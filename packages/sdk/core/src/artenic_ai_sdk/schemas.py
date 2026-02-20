"""Pydantic schemas for model I/O, metadata, config, and health."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from artenic_ai_sdk.types import (
    ConfigPhase,
    DriftType,
    EnsemblePhase,
    EnsembleStrategyType,
    EvolutionTrigger,
    ModelFramework,
    ModelPhase,
)

# =============================================================================
# Predictions — Generic base
# =============================================================================


class BasePrediction(BaseModel):
    """Generic prediction output. Every model produces this format.

    Domain extensions can subclass this with specialized fields.
    """

    model_config = ConfigDict(frozen=True)

    prediction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence")
    metadata: dict[str, Any] = Field(default_factory=dict)
    model_id: str
    model_version: str
    inference_time_ms: float = Field(ge=0.0)
    timestamp: datetime = Field(default_factory=datetime.now)


class EnsembleResult(BasePrediction):
    """Aggregated result from an ensemble of models."""

    strategy_used: EnsembleStrategyType
    models_responded: list[str] = Field(default_factory=list)
    models_failed: list[str] = Field(default_factory=list)
    individual_predictions: dict[str, BasePrediction] = Field(default_factory=dict)


# =============================================================================
# Training & Evaluation Results
# =============================================================================


class TrainResult(BaseModel):
    """Returned after a training run completes."""

    model_name: str
    model_version: str
    metrics: dict[str, float] = Field(default_factory=dict)
    epochs_completed: int = 0
    best_epoch: int | None = None
    artifact_path: str | None = None
    duration_seconds: float = 0.0
    provider: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)

    dataset_version: dict[str, Any] | None = None
    early_stopped: bool = False
    early_stopped_at_epoch: int | None = None
    precision_used: str | None = None
    gradient_checkpointing_used: bool = False
    distributed_world_size: int | None = None
    lr_found: float | None = None
    checkpoints_saved: int = 0
    preempted: bool = False


class EvalResult(BaseModel):
    """Returned after model evaluation."""

    model_name: str
    model_version: str
    metrics: dict[str, float] = Field(default_factory=dict)
    dataset_name: str | None = None
    dataset_size: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Feature Schema
# =============================================================================


class FeatureSchema(BaseModel):
    """Describes a single input feature for validation."""

    name: str
    dtype: Literal["float32", "float64", "int32", "int64", "bool", "string"]
    shape: tuple[int, ...] | None = None
    required: bool = True
    description: str = ""


# =============================================================================
# Model Metadata
# =============================================================================


class ModelMetadata(BaseModel):
    """Describes a registered model in the registry."""

    name: str
    version: str
    model_type: str
    framework: ModelFramework = ModelFramework.PYTORCH
    description: str = ""
    tags: dict[str, str] = Field(default_factory=dict)
    input_features: list[FeatureSchema] = Field(default_factory=list)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime | None = None
    model_size_bytes: int | None = None
    author: str | None = None


# =============================================================================
# Model Config — Base class, extended by each model
# =============================================================================


class ModelConfig(BaseModel):
    """Base config that every model extends with its hyperparameters.

    Example::

        class LightGBMConfig(ModelConfig):
            learning_rate: float = 0.01
            n_estimators: int = 1000
            max_depth: int = 8
    """

    model_config = ConfigDict(validate_assignment=True)

    version: str = "0.1.0"


# =============================================================================
# Config Registry Entry
# =============================================================================


class ConfigEntry(BaseModel):
    """A config version stored in the ConfigRegistry."""

    config: ModelConfig
    model_id: str
    phase: ConfigPhase = ConfigPhase.DEFAULT
    eval_result: EvalResult | None = None
    created_at: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Health Check
# =============================================================================


class HealthCheckResult(BaseModel):
    """Returned by model.health_check()."""

    status: Literal["healthy", "degraded", "unhealthy"]
    phase: ModelPhase
    uptime_seconds: float = 0.0
    last_inference_at: datetime | None = None
    inference_count: int = 0
    error_count: int = 0
    gpu_memory_used_mb: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Drift Detection
# =============================================================================


class DriftDetectionResult(BaseModel):
    """Result of drift detection for a single model."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    drift_type: DriftType
    drift_score: float = Field(ge=0.0, le=1.0, description="0.0 = no drift, 1.0 = severe drift")
    is_drifting: bool = False
    reference_window_size: int = Field(default=0, ge=0)
    current_window_size: int = Field(default=0, ge=0)
    metric_name: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    detected_at: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Model Health Report
# =============================================================================


class ModelHealthReport(BaseModel):
    """Aggregated health report for a model within an ensemble."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    model_version: str
    health_check: HealthCheckResult
    drift_results: list[DriftDetectionResult] = Field(default_factory=list)
    performance_trend: list[float] = Field(
        default_factory=list,
        description="Recent primary metric values (newest last)",
    )
    weight_in_ensemble: float | None = None
    recommendation: Literal["keep", "monitor", "prune"] = "keep"
    report_timestamp: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Ensemble Versioning
# =============================================================================


class EnsembleSnapshot(BaseModel):
    """Immutable snapshot of an ensemble configuration."""

    model_config = ConfigDict(frozen=True)

    version_id: str
    phase: EnsemblePhase = EnsemblePhase.DRAFT
    strategy_type: EnsembleStrategyType
    strategy_weights: dict[str, float] = Field(default_factory=dict)
    model_ids: list[str] = Field(default_factory=list)
    quorum: float = Field(ge=0.0, le=1.0, default=0.5)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    parent_version_id: str | None = None


class EnsembleABTest(BaseModel):
    """A/B test between two ensemble configurations."""

    model_config = ConfigDict(validate_assignment=True)

    test_id: str
    variant_a: EnsembleSnapshot
    variant_b: EnsembleSnapshot
    traffic_split: float = Field(
        ge=0.0,
        le=1.0,
        default=0.5,
        description="Fraction routed to variant A",
    )
    started_at: datetime = Field(default_factory=datetime.now)
    concluded_at: datetime | None = None
    winner: Literal["a", "b"] | None = None
    metrics_a: dict[str, float] = Field(default_factory=dict)
    metrics_b: dict[str, float] = Field(default_factory=dict)


# =============================================================================
# Auto-Pruning
# =============================================================================


class AutoPruneResult(BaseModel):
    """Result of auto-pruning evaluation on an ensemble."""

    model_config = ConfigDict(frozen=True)

    pruned_model_ids: list[str] = Field(default_factory=list)
    kept_model_ids: list[str] = Field(default_factory=list)
    reason: dict[str, str] = Field(
        default_factory=dict,
        description="model_id -> reason for pruning",
    )
    weight_threshold_used: float = 0.0
    evaluated_at: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Evolution Events
# =============================================================================


class EvolutionEvent(BaseModel):
    """Record of an ensemble strategy auto-evolution."""

    model_config = ConfigDict(frozen=True)

    from_strategy: EnsembleStrategyType
    to_strategy: EnsembleStrategyType
    trigger: EvolutionTrigger
    model_count: int = Field(ge=0)
    performance_data: dict[str, float] = Field(default_factory=dict)
    occurred_at: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Config Diffing
# =============================================================================


class ConfigDiff(BaseModel):
    """Result of comparing two ModelConfig instances.

    Fields:
        added: Keys present in config_b but not config_a.
        removed: Keys present in config_a but not config_b.
        changed: Keys present in both but with different values (field -> (old, new)).
    """

    model_config = ConfigDict(frozen=True)

    added: dict[str, Any] = Field(default_factory=dict)
    removed: dict[str, Any] = Field(default_factory=dict)
    changed: dict[str, tuple[Any, Any]] = Field(default_factory=dict)
