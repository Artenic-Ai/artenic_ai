"""Shared type definitions, enums, and constants for the Artenic AI platform."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

# =============================================================================
# Model Lifecycle
# =============================================================================


class ModelPhase(StrEnum):
    """Model lifecycle phases — managed by the platform."""

    CREATED = "created"
    WARMING_UP = "warming_up"
    READY = "ready"
    INFERENCE = "inference"
    TRAINING = "training"
    ERROR = "error"
    SHUTDOWN = "shutdown"


# =============================================================================
# Model Framework & Serialization
# =============================================================================


class ModelFramework(StrEnum):
    """Supported ML frameworks."""

    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    TRANSFORMERS = "transformers"
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    CATBOOST = "catboost"
    SKLEARN = "sklearn"
    ONNX = "onnx"
    CUSTOM = "custom"


class SerializationFormat(StrEnum):
    """Model serialization formats."""

    SAFETENSORS = "safetensors"
    ONNX = "onnx"
    TORCH = "torch"
    TORCHSCRIPT = "torchscript"
    PICKLE = "pickle"
    JOBLIB = "joblib"


# =============================================================================
# Ensemble
# =============================================================================


class EnsembleStrategyType(StrEnum):
    """Available ensemble combination strategies."""

    WEIGHTED_AVERAGE = "weighted_average"
    DYNAMIC_WEIGHTING = "dynamic_weighting"
    META_LEARNER = "meta_learner"
    MAJORITY_VOTING = "majority_voting"
    STACKING = "stacking"


class EnsemblePhase(StrEnum):
    """Ensemble lifecycle phase for versioning and promotion."""

    DRAFT = "draft"
    STAGING = "staging"
    CANARY = "canary"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class EvolutionTrigger(StrEnum):
    """What triggered an ensemble strategy auto-evolution."""

    MODEL_COUNT = "model_count"
    PERFORMANCE_THRESHOLD = "performance_threshold"
    MANUAL = "manual"


class DriftType(StrEnum):
    """Types of model drift detected by health monitoring."""

    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    PERFORMANCE_DRIFT = "performance_drift"


# =============================================================================
# Config Lifecycle
# =============================================================================


class ConfigPhase(StrEnum):
    """Config lifecycle phase — supports auto-evolution."""

    DEFAULT = "default"
    ACTIVE = "active"
    CANDIDATE = "candidate"
    PROMOTED = "promoted"
    RETIRED = "retired"


# =============================================================================
# Circuit Breaker
# =============================================================================


class CircuitBreakerState(StrEnum):
    """State of a circuit breaker protecting a model or service call."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# =============================================================================
# Type Aliases
# =============================================================================

type ModelId = str
type ModelVersion = str
type FeatureVector = dict[str, float]
type Metadata = dict[str, Any]
type EnsembleVersionId = str
type ABTestId = str
