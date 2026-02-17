"""Reusable decorators for model methods.

These decorators are used internally by the SDK (e.g. ObservabilityMixin
auto-wraps predict()) and can also be applied explicitly by model authors.
"""

from __future__ import annotations

import asyncio
import functools
import json as _json
import time
from collections.abc import Callable
from typing import Any

from artenic_ai_sdk.exceptions import (
    CircuitBreakerOpenError,
    ModelInferenceError,
    RateLimitError,
)
from artenic_ai_sdk.types import CircuitBreakerState

F = Callable[..., Any]


def track_inference(fn: F) -> F:
    """Record latency and error rate for an async predict method.

    Expects `self` to have `_metrics` (MetricsCollector) and
    `model_id` attributes — provided by ObservabilityMixin.
    """

    @functools.wraps(fn)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        success = True
        try:
            result = await fn(self, *args, **kwargs)
            return result
        except Exception:
            success = False
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            if hasattr(self, "_metrics") and self._metrics is not None:
                self._metrics.record(
                    inference_time_ms=elapsed_ms,
                    model_id=getattr(self, "model_id", "unknown"),
                    success=success,
                )

    return wrapper


def validate_input(*, required_keys: list[str] | None = None) -> Callable[[F], F]:
    """Validate that required keys are present in the input dict.

    Args:
        required_keys: List of keys that must be in the features dict.
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        async def wrapper(self: Any, features: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
            if required_keys:
                missing = [k for k in required_keys if k not in features]
                if missing:
                    raise ModelInferenceError(
                        f"Missing required input keys: {missing}",
                        details={
                            "missing_keys": missing,
                            "provided_keys": list(features.keys()),
                        },
                    )
            return await fn(self, features, *args, **kwargs)

        return wrapper

    return decorator


def retry(*, max_attempts: int = 3, backoff_base: float = 1.0) -> Callable[[F], F]:
    """Retry an async function with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts.
        backoff_base: Base delay in seconds (doubles each retry).
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return await fn(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        delay = backoff_base * (2**attempt)
                        await asyncio.sleep(delay)
            raise last_error  # type: ignore[misc]

        return wrapper

    return decorator


def timeout(*, seconds: float) -> Callable[[F], F]:
    """Enforce a timeout on an async function.

    Args:
        seconds: Maximum execution time in seconds.
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await asyncio.wait_for(fn(*args, **kwargs), timeout=seconds)
            except TimeoutError as e:
                raise ModelInferenceError(
                    f"{fn.__qualname__} timed out after {seconds}s",
                    details={"timeout_seconds": seconds},
                ) from e

        return wrapper

    return decorator


def log_lifecycle(fn: F) -> F:
    """Log lifecycle transitions (warmup, teardown, etc.).

    Expects `self` to have `_logger` (StructuredLogger) — provided
    by ObservabilityMixin.
    """

    @functools.wraps(fn)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        model_id = getattr(self, "model_id", "unknown")
        if hasattr(self, "_logger") and self._logger is not None:
            self._logger.info(
                f"{fn.__name__} started",
                model_id=model_id,
                event=f"lifecycle.{fn.__name__}.start",
            )
        try:
            result = await fn(self, *args, **kwargs)
            if hasattr(self, "_logger") and self._logger is not None:
                self._logger.info(
                    f"{fn.__name__} completed",
                    model_id=model_id,
                    event=f"lifecycle.{fn.__name__}.complete",
                )
            return result
        except Exception as e:
            if hasattr(self, "_logger") and self._logger is not None:
                self._logger.error(
                    f"{fn.__name__} failed: {e}",
                    model_id=model_id,
                    event=f"lifecycle.{fn.__name__}.error",
                    error=str(e),
                )
            raise

    return wrapper


# =============================================================================
# Circuit Breaker — class-based for testability
# =============================================================================


class CircuitBreaker:
    """Circuit breaker state machine.

    State transitions: CLOSED → OPEN (after threshold failures) →
    HALF_OPEN (after recovery_timeout) → CLOSED (on success) or OPEN (on failure).
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        expected_exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exceptions = expected_exceptions
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0

    @property
    def state(self) -> CircuitBreakerState:
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0

    def _check_state(self) -> None:
        """Check if we should transition from OPEN to HALF_OPEN or raise."""
        if self._state == CircuitBreakerState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                self._state = CircuitBreakerState.HALF_OPEN
            else:
                raise CircuitBreakerOpenError(
                    "Circuit breaker is OPEN",
                    open_since=self._last_failure_time,
                    failure_count=self._failure_count,
                )

    def record_success(self) -> None:
        """Record a successful call."""
        if self._state == CircuitBreakerState.HALF_OPEN:
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if (
            self._state == CircuitBreakerState.HALF_OPEN
            or self._failure_count >= self.failure_threshold
        ):
            self._state = CircuitBreakerState.OPEN


def circuit_breaker(
    *,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    expected_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """Circuit breaker decorator for async functions.

    Args:
        failure_threshold: Number of failures before opening the circuit.
        recovery_timeout: Seconds to wait before transitioning to HALF_OPEN.
        expected_exceptions: Exception types that count as failures.
    """

    def decorator(fn: F) -> F:
        cb = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exceptions=expected_exceptions,
        )

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            cb._check_state()
            try:
                result = await fn(*args, **kwargs)
                cb.record_success()
                return result
            except expected_exceptions:
                cb.record_failure()
                raise

        # Expose circuit breaker instance for testing
        wrapper._circuit_breaker = cb  # type: ignore[attr-defined]
        wrapper._circuit_state = lambda: cb.state  # type: ignore[attr-defined]
        wrapper._circuit_reset = cb.reset  # type: ignore[attr-defined]

        return wrapper

    return decorator


def cache_inference(
    *,
    maxsize: int = 128,
    ttl_seconds: float = 60.0,
) -> Callable[[F], F]:
    """Cache async inference results with TTL and LRU eviction.

    Args:
        maxsize: Maximum cache entries.
        ttl_seconds: Time-to-live for cache entries in seconds.
    """

    def decorator(fn: F) -> F:
        _cache: dict[str, tuple[float, Any]] = {}
        _access_order: list[str] = []
        _hits = 0
        _misses = 0

        def _make_key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
            # First positional arg after self is typically features dict
            features = args[1] if len(args) > 1 else kwargs.get("features", {})
            return _json.dumps(features, sort_keys=True, default=str)

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal _hits, _misses

            key = _make_key(args, kwargs)
            now = time.monotonic()

            # Check cache
            if key in _cache:
                ts, value = _cache[key]
                if now - ts < ttl_seconds:
                    _hits += 1
                    # Move to end of access order
                    if key in _access_order:
                        _access_order.remove(key)
                    _access_order.append(key)
                    return value
                else:
                    # Expired
                    del _cache[key]
                    if key in _access_order:
                        _access_order.remove(key)

            _misses += 1
            result = await fn(*args, **kwargs)

            # Evict LRU if at capacity
            while len(_cache) >= maxsize and _access_order:
                oldest = _access_order.pop(0)
                _cache.pop(oldest, None)

            _cache[key] = (now, result)
            _access_order.append(key)
            return result

        def cache_clear() -> None:
            nonlocal _hits, _misses
            _cache.clear()
            _access_order.clear()
            _hits = 0
            _misses = 0

        def cache_info() -> dict[str, int]:
            return {"hits": _hits, "misses": _misses, "size": len(_cache), "maxsize": maxsize}

        wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]
        wrapper.cache_info = cache_info  # type: ignore[attr-defined]

        return wrapper

    return decorator


# =============================================================================
# Rate Limiter — token bucket algorithm
# =============================================================================


def rate_limit(
    *,
    max_calls: int,
    window_seconds: float,
) -> Callable[[F], F]:
    """Rate-limit an async function using a token bucket.

    Args:
        max_calls: Maximum calls allowed per window.
        window_seconds: Window duration in seconds.
    """

    def decorator(fn: F) -> F:
        _lock = asyncio.Lock()
        _calls: list[float] = []

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with _lock:
                now = time.monotonic()
                # Evict expired timestamps
                cutoff = now - window_seconds
                while _calls and _calls[0] <= cutoff:
                    _calls.pop(0)

                if len(_calls) >= max_calls:
                    raise RateLimitError(
                        f"Rate limit exceeded: {max_calls} calls per {window_seconds}s",
                        retry_after=_calls[0] + window_seconds - now,
                    )
                _calls.append(now)

            return await fn(*args, **kwargs)

        wrapper._rate_limit_calls = _calls  # type: ignore[attr-defined]

        return wrapper

    return decorator
