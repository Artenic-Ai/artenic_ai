"""Tests for artenic_ai_sdk.decorators — retry, timeout, circuit_breaker, cache."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from artenic_ai_sdk.decorators import (
    CircuitBreaker,
    cache_inference,
    circuit_breaker,
    log_lifecycle,
    rate_limit,
    retry,
    timeout,
    track_inference,
    validate_input,
)
from artenic_ai_sdk.exceptions import CircuitBreakerOpenError, ModelInferenceError, RateLimitError
from artenic_ai_sdk.types import CircuitBreakerState

# =============================================================================
# track_inference
# =============================================================================


class _MockMetrics:
    """Fake metrics collector for testing decorators."""

    def __init__(self) -> None:
        self.records: list[tuple[float, str, bool]] = []

    def record(self, inference_time_ms: float, model_id: str, success: bool = True) -> None:
        self.records.append((inference_time_ms, model_id, success))


class TestTrackInference:
    @pytest.mark.asyncio
    async def test_records_success(self) -> None:
        metrics = _MockMetrics()

        class FakeModel:
            model_id = "test"
            _metrics = metrics

            @track_inference
            async def predict(self, features: dict) -> str:  # type: ignore[type-arg]
                return "ok"

        model = FakeModel()
        result = await model.predict({"a": 1.0})
        assert result == "ok"
        assert len(metrics.records) >= 1
        assert metrics.records[-1][2] is True

    @pytest.mark.asyncio
    async def test_records_failure(self) -> None:
        metrics = _MockMetrics()

        class FakeModel:
            model_id = "test"
            _metrics = metrics

            @track_inference
            async def predict(self, features: dict) -> str:  # type: ignore[type-arg]
                raise ValueError("bad")

        model = FakeModel()
        with pytest.raises(ValueError):
            await model.predict({"a": 1.0})
        assert metrics.records[-1][2] is False


# =============================================================================
# validate_input
# =============================================================================


class TestValidateInput:
    @pytest.mark.asyncio
    async def test_passes_when_keys_present(self) -> None:
        class FakeModel:
            @validate_input(required_keys=["a", "b"])
            async def predict(self, features: dict) -> str:  # type: ignore[type-arg]
                return "ok"

        model = FakeModel()
        result = await model.predict({"a": 1.0, "b": 2.0})
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_raises_when_keys_missing(self) -> None:
        class FakeModel:
            @validate_input(required_keys=["a", "b"])
            async def predict(self, features: dict) -> str:  # type: ignore[type-arg]
                return "ok"

        model = FakeModel()
        with pytest.raises(ModelInferenceError, match="Missing required input keys"):
            await model.predict({"a": 1.0})


# =============================================================================
# retry
# =============================================================================


class TestRetry:
    @pytest.mark.asyncio
    async def test_retries_on_failure(self) -> None:
        call_count = 0

        @retry(max_attempts=3, backoff_base=0.01)
        async def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return "success"

        result = await flaky()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_attempts(self) -> None:
        @retry(max_attempts=2, backoff_base=0.01)
        async def always_fail() -> None:
            raise ValueError("fail")

        with pytest.raises(ValueError):
            await always_fail()


# =============================================================================
# timeout
# =============================================================================


class TestTimeout:
    @pytest.mark.asyncio
    async def test_completes_within_timeout(self) -> None:
        @timeout(seconds=1.0)
        async def fast() -> str:
            return "ok"

        result = await fast()
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_raises_on_timeout(self) -> None:
        @timeout(seconds=0.01)
        async def slow() -> str:
            await asyncio.sleep(1.0)
            return "ok"

        with pytest.raises(ModelInferenceError, match="timed out"):
            await slow()


# =============================================================================
# CircuitBreaker class
# =============================================================================


class TestCircuitBreakerClass:
    def test_initial_state(self) -> None:
        cb = CircuitBreaker()
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0

    def test_opens_after_threshold(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

    def test_half_open_after_timeout(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        time.sleep(0.02)
        cb._check_state()  # Should transition to HALF_OPEN
        assert cb.state == CircuitBreakerState.HALF_OPEN

    def test_closes_on_success_from_half_open(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb._check_state()
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED

    def test_reopens_on_failure_from_half_open(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb._check_state()
        assert cb.state == CircuitBreakerState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

    def test_reset(self) -> None:
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        cb.reset()
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0

    def test_check_state_raises_when_open(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=999)
        cb.record_failure()
        with pytest.raises(CircuitBreakerOpenError):
            cb._check_state()


# =============================================================================
# circuit_breaker decorator
# =============================================================================


class TestCircuitBreakerDecorator:
    @pytest.mark.asyncio
    async def test_opens_after_failures(self) -> None:
        @circuit_breaker(failure_threshold=2, recovery_timeout=999)
        async def failing() -> None:
            raise ValueError("fail")

        for _ in range(2):
            with pytest.raises(ValueError):
                await failing()

        with pytest.raises(CircuitBreakerOpenError):
            await failing()

    @pytest.mark.asyncio
    async def test_exposes_state(self) -> None:
        @circuit_breaker(failure_threshold=5)
        async def fn() -> str:
            return "ok"

        assert fn._circuit_state() == CircuitBreakerState.CLOSED  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_reset(self) -> None:
        @circuit_breaker(failure_threshold=1, recovery_timeout=999)
        async def failing() -> None:
            raise ValueError("fail")

        with pytest.raises(ValueError):
            await failing()

        assert failing._circuit_state() == CircuitBreakerState.OPEN  # type: ignore[attr-defined]
        failing._circuit_reset()  # type: ignore[attr-defined]
        assert failing._circuit_state() == CircuitBreakerState.CLOSED  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_exposes_circuit_breaker_instance(self) -> None:
        @circuit_breaker(failure_threshold=5)
        async def fn() -> str:
            return "ok"

        cb = fn._circuit_breaker  # type: ignore[attr-defined]
        assert isinstance(cb, CircuitBreaker)


# =============================================================================
# cache_inference
# =============================================================================


class TestCacheInference:
    @pytest.mark.asyncio
    async def test_caches_result(self) -> None:
        call_count = 0

        class FakeModel:
            @cache_inference(maxsize=10, ttl_seconds=60)
            async def predict(self, features: dict) -> str:  # type: ignore[type-arg]
                nonlocal call_count
                call_count += 1
                return "result"

        model = FakeModel()
        r1 = await model.predict({"a": 1.0})
        r2 = await model.predict({"a": 1.0})
        assert r1 == r2
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_different_keys(self) -> None:
        call_count = 0

        class FakeModel:
            @cache_inference(maxsize=10, ttl_seconds=60)
            async def predict(self, features: dict) -> str:  # type: ignore[type-arg]
                nonlocal call_count
                call_count += 1
                return f"result_{call_count}"

        model = FakeModel()
        r1 = await model.predict({"a": 1.0})
        r2 = await model.predict({"b": 2.0})
        assert r1 != r2
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_cache_info(self) -> None:
        class FakeModel:
            @cache_inference(maxsize=10, ttl_seconds=60)
            async def predict(self, features: dict) -> str:  # type: ignore[type-arg]
                return "result"

        model = FakeModel()
        await model.predict({"a": 1.0})
        await model.predict({"a": 1.0})  # cache hit
        info = FakeModel.predict.cache_info()  # type: ignore[attr-defined]
        assert info["hits"] == 1
        assert info["misses"] == 1

    @pytest.mark.asyncio
    async def test_cache_clear(self) -> None:
        class FakeModel:
            @cache_inference(maxsize=10, ttl_seconds=60)
            async def predict(self, features: dict) -> str:  # type: ignore[type-arg]
                return "result"

        model = FakeModel()
        await model.predict({"a": 1.0})
        FakeModel.predict.cache_clear()  # type: ignore[attr-defined]
        info = FakeModel.predict.cache_info()  # type: ignore[attr-defined]
        assert info["size"] == 0

    @pytest.mark.asyncio
    async def test_evicts_lru(self) -> None:
        class FakeModel:
            @cache_inference(maxsize=2, ttl_seconds=60)
            async def predict(self, features: dict) -> str:  # type: ignore[type-arg]
                return "result"

        model = FakeModel()
        await model.predict({"a": 1})
        await model.predict({"b": 2})
        await model.predict({"c": 3})  # should evict {"a": 1}
        info = FakeModel.predict.cache_info()  # type: ignore[attr-defined]
        assert info["size"] == 2

    @pytest.mark.asyncio
    async def test_ttl_expiry(self) -> None:
        """Cache entries expire after TTL."""
        call_count = 0

        class FakeModel:
            @cache_inference(maxsize=10, ttl_seconds=0.01)
            async def predict(self, features: dict) -> str:  # type: ignore[type-arg]
                nonlocal call_count
                call_count += 1
                return f"result_{call_count}"

        model = FakeModel()
        r1 = await model.predict({"a": 1.0})
        await asyncio.sleep(0.02)  # Wait for TTL to expire
        r2 = await model.predict({"a": 1.0})
        assert r1 != r2  # Should have re-executed
        assert call_count == 2


# =============================================================================
# log_lifecycle
# =============================================================================


class TestLogLifecycle:
    @pytest.mark.asyncio
    async def test_logs_success(self) -> None:
        """log_lifecycle logs start and complete on success."""
        mock_logger = MagicMock()

        class FakeModel:
            model_id = "test-model"
            _logger = mock_logger

            @log_lifecycle
            async def warmup(self) -> str:
                return "done"

        model = FakeModel()
        result = await model.warmup()
        assert result == "done"

        # Logger should have been called for start and complete
        assert mock_logger.info.call_count == 2
        start_call = mock_logger.info.call_args_list[0]
        assert "started" in start_call[0][0]
        complete_call = mock_logger.info.call_args_list[1]
        assert "completed" in complete_call[0][0]

    @pytest.mark.asyncio
    async def test_logs_failure(self) -> None:
        """log_lifecycle logs start and error on exception."""
        mock_logger = MagicMock()

        class FakeModel:
            model_id = "test-model"
            _logger = mock_logger

            @log_lifecycle
            async def warmup(self) -> None:
                raise RuntimeError("oops")

        model = FakeModel()
        with pytest.raises(RuntimeError, match="oops"):
            await model.warmup()

        mock_logger.info.assert_called_once()  # start
        mock_logger.error.assert_called_once()  # error
        error_call = mock_logger.error.call_args
        assert "failed" in error_call[0][0]

    @pytest.mark.asyncio
    async def test_no_logger(self) -> None:
        """log_lifecycle works when _logger is None."""

        class FakeModel:
            model_id = "test"
            _logger = None

            @log_lifecycle
            async def warmup(self) -> str:
                return "done"

        model = FakeModel()
        result = await model.warmup()
        assert result == "done"

    @pytest.mark.asyncio
    async def test_no_logger_on_error(self) -> None:
        """log_lifecycle handles errors when _logger is None."""

        class FakeModel:
            model_id = "test"
            _logger = None

            @log_lifecycle
            async def warmup(self) -> None:
                raise ValueError("fail")

        model = FakeModel()
        with pytest.raises(ValueError):
            await model.warmup()


class TestCircuitBreakerDecoratorSuccess:
    @pytest.mark.asyncio
    async def test_success_path(self) -> None:
        """Circuit breaker passes through on success."""

        @circuit_breaker(failure_threshold=5)
        async def fn() -> str:
            return "ok"

        result = await fn()
        assert result == "ok"
        assert fn._circuit_state() == CircuitBreakerState.CLOSED  # type: ignore[attr-defined]


# =============================================================================
# rate_limit
# =============================================================================


class TestRateLimit:
    @pytest.mark.asyncio
    async def test_allows_under_limit(self) -> None:
        @rate_limit(max_calls=3, window_seconds=1.0)
        async def fn() -> str:
            return "ok"

        for _ in range(3):
            result = await fn()
            assert result == "ok"

    @pytest.mark.asyncio
    async def test_blocks_over_limit(self) -> None:
        @rate_limit(max_calls=2, window_seconds=10.0)
        async def fn() -> str:
            return "ok"

        await fn()
        await fn()
        with pytest.raises(RateLimitError, match="Rate limit exceeded"):
            await fn()

    @pytest.mark.asyncio
    async def test_resets_after_window(self) -> None:
        @rate_limit(max_calls=1, window_seconds=0.01)
        async def fn() -> str:
            return "ok"

        await fn()
        await asyncio.sleep(0.02)
        result = await fn()
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retry_after_is_positive(self) -> None:
        @rate_limit(max_calls=1, window_seconds=60.0)
        async def fn() -> str:
            return "ok"

        await fn()
        with pytest.raises(RateLimitError) as exc_info:
            await fn()
        assert exc_info.value.retry_after is not None
        assert exc_info.value.retry_after > 0

    @pytest.mark.asyncio
    async def test_exposes_calls_list(self) -> None:
        @rate_limit(max_calls=5, window_seconds=1.0)
        async def fn() -> str:
            return "ok"

        await fn()
        assert len(fn._rate_limit_calls) == 1  # type: ignore[attr-defined]
