"""Tests for artenic_ai_sdk.exceptions — exception hierarchy."""

from __future__ import annotations

from typing import ClassVar

import pytest

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

# =============================================================================
# Hierarchy tests
# =============================================================================


class TestHierarchy:
    """Every exception should be catchable via ArtenicAIError."""

    ALL_EXCEPTIONS: ClassVar[list[type[ArtenicAIError]]] = [
        ModelError,
        ModelNotFoundError,
        ModelLoadError,
        ModelTrainingError,
        ModelInferenceError,
        EnsembleError,
        NoModelsRegisteredError,
        StrategyError,
        QuorumNotMetError,
        AllModelsFailedError,
        ConfigError,
        ConfigValidationError,
        ConfigNotFoundError,
        SerializationError,
        FormatNotSupportedError,
        ArtifactCorruptedError,
        PlatformError,
        ServiceUnavailableError,
        AuthenticationError,
        ArtenicTimeoutError,
        RateLimitError,
        CircuitBreakerOpenError,
        ProviderError,
        ProviderAuthError,
        ProviderQuotaError,
        ProviderTimeoutError,
        JobStuckError,
        BudgetExceededError,
    ]

    @pytest.mark.parametrize("exc_cls", ALL_EXCEPTIONS)
    def test_inherits_from_base(self, exc_cls: type[ArtenicAIError]) -> None:
        assert issubclass(exc_cls, ArtenicAIError)

    def test_model_errors_inherit_model_error(self) -> None:
        for cls in (ModelNotFoundError, ModelLoadError, ModelTrainingError, ModelInferenceError):
            assert issubclass(cls, ModelError)

    def test_ensemble_errors_inherit_ensemble_error(self) -> None:
        ensemble_errors = (
            NoModelsRegisteredError,
            StrategyError,
            QuorumNotMetError,
            AllModelsFailedError,
        )
        for cls in ensemble_errors:
            assert issubclass(cls, EnsembleError)

    def test_config_errors_inherit_config_error(self) -> None:
        for cls in (ConfigValidationError, ConfigNotFoundError):
            assert issubclass(cls, ConfigError)

    def test_serialization_errors_inherit_serialization_error(self) -> None:
        for cls in (FormatNotSupportedError, ArtifactCorruptedError):
            assert issubclass(cls, SerializationError)

    def test_provider_errors_inherit_provider_error(self) -> None:
        for cls in (ProviderAuthError, ProviderQuotaError, ProviderTimeoutError, JobStuckError):
            assert issubclass(cls, ProviderError)
            assert issubclass(cls, PlatformError)

    def test_budget_inherits_platform_error(self) -> None:
        assert issubclass(BudgetExceededError, PlatformError)


# =============================================================================
# Base error
# =============================================================================


class TestArtenicAIError:
    def test_message(self) -> None:
        err = ArtenicAIError("something went wrong")
        assert str(err) == "something went wrong"

    def test_details_default(self) -> None:
        err = ArtenicAIError("oops")
        assert err.details == {}

    def test_details_custom(self) -> None:
        err = ArtenicAIError("oops", details={"key": "val"})
        assert err.details == {"key": "val"}

    def test_catchable_as_exception(self) -> None:
        with pytest.raises(ArtenicAIError):
            raise ArtenicAIError("test")


# =============================================================================
# Rich exceptions (extra fields)
# =============================================================================


class TestQuorumNotMetError:
    def test_fields(self) -> None:
        err = QuorumNotMetError("quorum", required=3, responded=1)
        assert err.required == 3
        assert err.responded == 1
        assert str(err) == "quorum"

    def test_defaults(self) -> None:
        err = QuorumNotMetError()
        assert err.required == 0
        assert err.responded == 0


class TestAllModelsFailedError:
    def test_fields(self) -> None:
        err = AllModelsFailedError("all failed", failed_models=["a", "b"])
        assert err.failed_models == ["a", "b"]

    def test_defaults(self) -> None:
        err = AllModelsFailedError()
        assert err.failed_models == []


class TestArtenicTimeoutError:
    def test_fields(self) -> None:
        err = ArtenicTimeoutError("timeout", timeout_seconds=30.0, operation="predict")
        assert err.timeout_seconds == 30.0
        assert err.operation == "predict"

    def test_defaults(self) -> None:
        err = ArtenicTimeoutError()
        assert err.timeout_seconds == 0.0
        assert err.operation == ""


class TestRateLimitError:
    def test_fields(self) -> None:
        err = RateLimitError("429", retry_after=60.0)
        assert err.retry_after == 60.0

    def test_defaults(self) -> None:
        err = RateLimitError()
        assert err.retry_after is None


class TestCircuitBreakerOpenError:
    def test_fields(self) -> None:
        err = CircuitBreakerOpenError("open", open_since=1000.0, failure_count=5)
        assert err.open_since == 1000.0
        assert err.failure_count == 5

    def test_defaults(self) -> None:
        err = CircuitBreakerOpenError()
        assert err.open_since == 0.0
        assert err.failure_count == 0


class TestJobStuckError:
    def test_fields(self) -> None:
        err = JobStuckError("stuck", job_id="j-123", max_runtime_hours=24.0)
        assert err.job_id == "j-123"
        assert err.max_runtime_hours == 24.0

    def test_defaults(self) -> None:
        err = JobStuckError()
        assert err.job_id == ""
        assert err.max_runtime_hours == 0.0


class TestBudgetExceededError:
    def test_fields(self) -> None:
        err = BudgetExceededError(
            "over budget", spent_eur=150.0, limit_eur=100.0, scope="project-x"
        )
        assert err.spent_eur == 150.0
        assert err.limit_eur == 100.0
        assert err.scope == "project-x"

    def test_defaults(self) -> None:
        err = BudgetExceededError()
        assert str(err) == "Budget exceeded"
        assert err.spent_eur == 0.0
        assert err.limit_eur == 0.0
        assert err.scope == ""


# =============================================================================
# error_context
# =============================================================================


class TestErrorContext:
    def test_enriches_artenic_error(self) -> None:
        with (
            pytest.raises(ModelInferenceError) as exc_info,
            error_context(model_id="lgbm", operation="predict"),
        ):
            raise ModelInferenceError("timeout")
        assert exc_info.value.details["model_id"] == "lgbm"
        assert exc_info.value.details["operation"] == "predict"

    def test_passes_non_artenic_errors(self) -> None:
        with pytest.raises(ValueError, match="not artenic"), error_context(model_id="test"):
            raise ValueError("not artenic")

    def test_no_error(self) -> None:
        with error_context(model_id="test"):
            pass  # Should not raise

    def test_nested_contexts(self) -> None:
        with (
            pytest.raises(ModelInferenceError) as exc_info,
            error_context(outer="yes"),
            error_context(inner="yes"),
        ):
            raise ModelInferenceError("nested")
        assert exc_info.value.details["inner"] == "yes"
        assert exc_info.value.details["outer"] == "yes"

    def test_preserves_existing_details(self) -> None:
        with pytest.raises(ArtenicAIError) as exc_info, error_context(added="new"):
            raise ArtenicAIError("test", details={"existing": "value"})
        assert exc_info.value.details["existing"] == "value"
        assert exc_info.value.details["added"] == "new"
