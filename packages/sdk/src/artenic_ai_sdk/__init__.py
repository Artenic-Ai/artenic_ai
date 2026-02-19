"""Artenic AI SDK — Shared contracts for the Artenic AI platform.

Core exports::

    from artenic_ai_sdk import BaseModel, EnsembleManager
    from artenic_ai_sdk import BasePrediction, ModelConfig
    from artenic_ai_sdk import ConfigManager, ConfigRegistry
"""

from __future__ import annotations

# === Core contract ===
from artenic_ai_sdk.base_model import BaseModel

# === Client ===
from artenic_ai_sdk.client import PlatformClient

# === Config ===
from artenic_ai_sdk.config import ConfigManager, ConfigRegistry

# === Decorators ===
from artenic_ai_sdk.decorators import (
    cache_inference,
    circuit_breaker,
    log_lifecycle,
    rate_limit,
    retry,
    timeout,
    track_inference,
    validate_input,
)

# === Ensemble ===
from artenic_ai_sdk.ensemble import (
    DynamicWeighting,
    EnsembleManager,
    EnsembleStrategy,
    EvolutionPolicy,
    MajorityVoting,
    MetaLearner,
    Stacking,
    WeightedAverage,
)

# === Exceptions ===
from artenic_ai_sdk.exceptions import (
    AllModelsFailedError,
    ArtenicAIError,
    ArtenicTimeoutError,
    ArtifactCorruptedError,
    AuthenticationError,
    BudgetExceededError,
    CircuitBreakerOpenError,
    ConfigError,
    ConfigNotFoundError,
    ConfigValidationError,
    EnsembleError,
    FormatNotSupportedError,
    JobStuckError,
    ModelError,
    ModelInferenceError,
    ModelLoadError,
    ModelNotFoundError,
    ModelTrainingError,
    NoModelsRegisteredError,
    PlatformError,
    ProviderAuthError,
    ProviderError,
    ProviderQuotaError,
    ProviderTimeoutError,
    QuorumNotMetError,
    RateLimitError,
    SerializationError,
    ServiceUnavailableError,
    StrategyError,
    error_context,
)

# === Observability ===
from artenic_ai_sdk.observability import (
    MetricsCollector,
    ObservabilityMixin,
    StructuredLogger,
    correlation_context,
    get_trace_id,
)

# === Schemas ===
from artenic_ai_sdk.schemas import (
    AutoPruneResult,
    BasePrediction,
    ConfigDiff,
    ConfigEntry,
    DriftDetectionResult,
    EnsembleABTest,
    EnsembleResult,
    EnsembleSnapshot,
    EvalResult,
    EvolutionEvent,
    FeatureSchema,
    HealthCheckResult,
    ModelConfig,
    ModelHealthReport,
    ModelMetadata,
    TrainResult,
)

# === Serialization ===
from artenic_ai_sdk.serialization import ModelSerializer

# === Testing ===
from artenic_ai_sdk.testing import (
    MockModel,
    ModelTestCase,
    assert_latency_under,
    assert_prediction_stable,
    assert_throughput_above,
    create_test_features,
)

# === Training Intelligence ===
from artenic_ai_sdk.training import (
    CallbackRunner,
    TrainingCallback,
    TrainingConfig,
    TrainingContext,
    build_callbacks,
)

# === Types ===
from artenic_ai_sdk.types import (
    CircuitBreakerState,
    ConfigPhase,
    DriftType,
    EnsemblePhase,
    EnsembleStrategyType,
    EvolutionTrigger,
    ModelFramework,
    ModelPhase,
    SerializationFormat,
)

__version__ = "0.6.0"

__all__ = [
    # Exceptions
    "AllModelsFailedError",
    "ArtenicAIError",
    "ArtenicTimeoutError",
    "ArtifactCorruptedError",
    "AuthenticationError",
    "AutoPruneResult",
    # Core
    "BaseModel",
    "BasePrediction",
    "BudgetExceededError",
    # Training Intelligence
    "CallbackRunner",
    "CircuitBreakerOpenError",
    "CircuitBreakerState",
    "ConfigDiff",
    "ConfigEntry",
    "ConfigError",
    # Config
    "ConfigManager",
    "ConfigNotFoundError",
    "ConfigPhase",
    "ConfigRegistry",
    "ConfigValidationError",
    "DriftDetectionResult",
    "DriftType",
    # Ensemble
    "DynamicWeighting",
    "EnsembleABTest",
    "EnsembleError",
    "EnsembleManager",
    "EnsemblePhase",
    "EnsembleResult",
    "EnsembleSnapshot",
    "EnsembleStrategy",
    "EnsembleStrategyType",
    # Schemas
    "EvalResult",
    "EvolutionEvent",
    "EvolutionPolicy",
    "EvolutionTrigger",
    "FeatureSchema",
    "FormatNotSupportedError",
    "HealthCheckResult",
    "JobStuckError",
    "MajorityVoting",
    "MetaLearner",
    "MetricsCollector",
    # Testing
    "MockModel",
    "ModelConfig",
    "ModelError",
    "ModelFramework",
    "ModelHealthReport",
    "ModelInferenceError",
    "ModelLoadError",
    "ModelMetadata",
    "ModelNotFoundError",
    "ModelPhase",
    # Serialization
    "ModelSerializer",
    "ModelTestCase",
    "ModelTrainingError",
    "NoModelsRegisteredError",
    "ObservabilityMixin",
    # Client
    "PlatformClient",
    "PlatformError",
    "ProviderAuthError",
    "ProviderError",
    "ProviderQuotaError",
    "ProviderTimeoutError",
    "QuorumNotMetError",
    "RateLimitError",
    "SerializationError",
    "SerializationFormat",
    "ServiceUnavailableError",
    "Stacking",
    "StrategyError",
    "StructuredLogger",
    "TrainResult",
    "TrainingCallback",
    "TrainingConfig",
    "TrainingContext",
    "WeightedAverage",
    # Performance testing
    "assert_latency_under",
    "assert_prediction_stable",
    "assert_throughput_above",
    "build_callbacks",
    # Decorators
    "cache_inference",
    "circuit_breaker",
    # Observability
    "correlation_context",
    "create_test_features",
    # Exceptions utilities
    "error_context",
    "get_trace_id",
    "log_lifecycle",
    "rate_limit",
    "retry",
    "timeout",
    "track_inference",
    "validate_input",
]
