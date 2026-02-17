"""Exception hierarchy for the Artenic AI platform.

All exceptions inherit from ArtenicAIError so callers can catch
platform-level errors with a single except clause.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator


class ArtenicAIError(Exception):
    """Base exception for all Artenic AI errors."""

    def __init__(self, message: str = "", *, details: dict[str, Any] | None = None) -> None:
        self.details = details or {}
        super().__init__(message)


# =============================================================================
# Model Errors
# =============================================================================


class ModelError(ArtenicAIError):
    """Base for model-related errors."""


class ModelNotFoundError(ModelError):
    """Raised when a model is not found in the registry."""


class ModelLoadError(ModelError):
    """Raised when a model fails to load from disk or remote storage."""


class ModelTrainingError(ModelError):
    """Raised when training fails or is interrupted."""


class ModelInferenceError(ModelError):
    """Raised when predict() fails."""


# =============================================================================
# Ensemble Errors
# =============================================================================


class EnsembleError(ArtenicAIError):
    """Base for ensemble-related errors."""


class NoModelsRegisteredError(EnsembleError):
    """Raised when predict() is called on an ensemble with no models."""


class StrategyError(EnsembleError):
    """Raised when the ensemble strategy fails to combine predictions."""


class QuorumNotMetError(EnsembleError):
    """Raised when not enough models responded within the timeout."""

    def __init__(
        self,
        message: str = "",
        *,
        required: int = 0,
        responded: int = 0,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.required = required
        self.responded = responded
        super().__init__(message, details=details)


class AllModelsFailedError(EnsembleError):
    """Raised when every model in the ensemble failed."""

    def __init__(
        self,
        message: str = "",
        *,
        failed_models: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.failed_models = failed_models or []
        super().__init__(message, details=details)


# =============================================================================
# Config Errors
# =============================================================================


class ConfigError(ArtenicAIError):
    """Base for configuration errors."""


class ConfigValidationError(ConfigError):
    """Raised when a config fails Pydantic validation."""


class ConfigNotFoundError(ConfigError):
    """Raised when a config file or registry entry is not found."""


# =============================================================================
# Serialization Errors
# =============================================================================


class SerializationError(ArtenicAIError):
    """Base for model serialization/deserialization errors."""


class FormatNotSupportedError(SerializationError):
    """Raised when an unsupported serialization format is requested."""


class ArtifactCorruptedError(SerializationError):
    """Raised when a model artifact fails integrity checks."""


# =============================================================================
# Platform Errors
# =============================================================================


class PlatformError(ArtenicAIError):
    """Base for platform-level errors."""


class ServiceUnavailableError(PlatformError):
    """Raised when a service (embedded or standalone) is unreachable."""


class AuthenticationError(PlatformError):
    """Raised when API key or JWT validation fails."""


class ArtenicTimeoutError(PlatformError):
    """Raised when an operation exceeds its deadline."""

    def __init__(
        self,
        message: str = "",
        *,
        timeout_seconds: float = 0.0,
        operation: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.operation = operation
        super().__init__(message, details=details)


class RateLimitError(PlatformError):
    """Raised when the platform returns HTTP 429."""

    def __init__(
        self,
        message: str = "",
        *,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(message, details=details)


class CircuitBreakerOpenError(PlatformError):
    """Raised when a circuit breaker is in OPEN state."""

    def __init__(
        self,
        message: str = "",
        *,
        open_since: float = 0.0,
        failure_count: int = 0,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.open_since = open_since
        self.failure_count = failure_count
        super().__init__(message, details=details)


# =============================================================================
# Provider Errors
# =============================================================================


class ProviderError(PlatformError):
    """Base for cloud provider errors."""


class ProviderAuthError(ProviderError):
    """Raised when authentication with a cloud provider fails."""


class ProviderQuotaError(ProviderError):
    """Raised when a resource quota is exceeded on a cloud provider."""


class ProviderTimeoutError(ProviderError):
    """Raised when a cloud provider operation times out."""


class JobStuckError(ProviderError):
    """Raised when a training job exceeds its maximum runtime."""

    def __init__(
        self,
        message: str = "",
        *,
        job_id: str = "",
        max_runtime_hours: float = 0.0,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.job_id = job_id
        self.max_runtime_hours = max_runtime_hours
        super().__init__(message, details=details)


# =============================================================================
# Budget Errors
# =============================================================================


class BudgetExceededError(PlatformError):
    """Raised when a budget limit is exceeded and dispatch is blocked."""

    def __init__(
        self,
        message: str = "Budget exceeded",
        *,
        spent_eur: float = 0.0,
        limit_eur: float = 0.0,
        scope: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.spent_eur = spent_eur
        self.limit_eur = limit_eur
        self.scope = scope
        super().__init__(message, details=details)


# =============================================================================
# Error Context Manager
# =============================================================================


@contextmanager
def error_context(**context: Any) -> Generator[None, None, None]:
    """Enrich ArtenicAIError exceptions with contextual metadata.

    Any ArtenicAIError raised inside the block will have its ``details``
    dict updated with the provided key-value pairs. Non-ArtenicAIError
    exceptions pass through unchanged.

    Example::

        with error_context(model_id="lgbm", operation="predict"):
            raise ModelInferenceError("timeout")
        # error.details == {"model_id": "lgbm", "operation": "predict"}
    """
    try:
        yield
    except ArtenicAIError as exc:
        exc.details.update(context)
        raise
