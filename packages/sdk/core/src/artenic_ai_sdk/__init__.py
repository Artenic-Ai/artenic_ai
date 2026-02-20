"""Artenic AI SDK — Shared contracts for the Artenic AI platform.

Core exports::

    from artenic_ai_sdk import BaseModel, BasePrediction, ModelConfig
    from artenic_ai_sdk import ConfigManager, ConfigRegistry
"""

from __future__ import annotations

# === Core contract ===
from artenic_ai_sdk.base_model import BaseModel

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

__version__ = "0.7.0"

# === Conditional re-exports (installed via extras) ===

# Ensemble (pip install artenic-ai-sdk[ensemble])
try:
    from artenic_ai_sdk_ensemble import (
        DynamicWeighting,
        EnsembleManager,
        EnsembleStrategy,
        EvolutionPolicy,
        MajorityVoting,
        MetaLearner,
        Stacking,
        WeightedAverage,
    )
except ImportError:  # pragma: no cover
    pass

# Training (pip install artenic-ai-sdk[training])
try:
    from artenic_ai_sdk_training import (
        CallbackRunner,
        TrainingCallback,
        TrainingConfig,
        TrainingContext,
        build_callbacks,
    )
except ImportError:  # pragma: no cover
    pass

# Client (pip install artenic-ai-sdk[client])
try:
    from artenic_ai_sdk_client import PlatformClient
except ImportError:  # pragma: no cover
    pass

_CORE_ALL = [
    # Core
    "BaseModel",
    # Config
    "ConfigManager",
    "ConfigRegistry",
    # Decorators
    "cache_inference",
    "circuit_breaker",
    "log_lifecycle",
    "rate_limit",
    "retry",
    "timeout",
    "track_inference",
    "validate_input",
    # Exceptions
    "AllModelsFailedError",
    "ArtenicAIError",
    "ArtenicTimeoutError",
    "ArtifactCorruptedError",
    "AuthenticationError",
    "BudgetExceededError",
    "CircuitBreakerOpenError",
    "ConfigError",
    "ConfigNotFoundError",
    "ConfigValidationError",
    "EnsembleError",
    "FormatNotSupportedError",
    "JobStuckError",
    "ModelError",
    "ModelInferenceError",
    "ModelLoadError",
    "ModelNotFoundError",
    "ModelTrainingError",
    "NoModelsRegisteredError",
    "PlatformError",
    "ProviderAuthError",
    "ProviderError",
    "ProviderQuotaError",
    "ProviderTimeoutError",
    "QuorumNotMetError",
    "RateLimitError",
    "SerializationError",
    "ServiceUnavailableError",
    "StrategyError",
    "error_context",
    # Observability
    "MetricsCollector",
    "ObservabilityMixin",
    "StructuredLogger",
    "correlation_context",
    "get_trace_id",
    # Schemas
    "AutoPruneResult",
    "BasePrediction",
    "ConfigDiff",
    "ConfigEntry",
    "DriftDetectionResult",
    "EnsembleABTest",
    "EnsembleResult",
    "EnsembleSnapshot",
    "EvalResult",
    "EvolutionEvent",
    "FeatureSchema",
    "HealthCheckResult",
    "ModelConfig",
    "ModelHealthReport",
    "ModelMetadata",
    "TrainResult",
    # Serialization
    "ModelSerializer",
    # Testing
    "MockModel",
    "ModelTestCase",
    "assert_latency_under",
    "assert_prediction_stable",
    "assert_throughput_above",
    "create_test_features",
    # Types
    "CircuitBreakerState",
    "ConfigPhase",
    "DriftType",
    "EnsemblePhase",
    "EnsembleStrategyType",
    "EvolutionTrigger",
    "ModelFramework",
    "ModelPhase",
    "SerializationFormat",
]

# Conditional exports — only in __all__ if the extra is installed
_CONDITIONAL = [
    "DynamicWeighting",
    "EnsembleManager",
    "EnsembleStrategy",
    "EvolutionPolicy",
    "MajorityVoting",
    "MetaLearner",
    "Stacking",
    "WeightedAverage",
    "CallbackRunner",
    "TrainingCallback",
    "TrainingConfig",
    "TrainingContext",
    "build_callbacks",
    "PlatformClient",
]

__all__ = _CORE_ALL + [name for name in _CONDITIONAL if name in dir()]
