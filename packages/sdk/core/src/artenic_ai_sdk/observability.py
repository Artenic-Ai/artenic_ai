"""Integrated observability: metrics collection, structured logging, and the
ObservabilityMixin that auto-instruments BaseModel subclasses.
"""

from __future__ import annotations

import bisect
import contextvars
import json
import logging
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

# =============================================================================
# Metrics Collection
# =============================================================================


@dataclass
class InferenceMetric:
    """Single inference measurement."""

    inference_time_ms: float
    model_id: str
    success: bool
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Collects inference metrics with rolling percentile computation.

    Uses a ring buffer for latencies and sorted insertion for
    efficient percentile calculation without numpy dependency.

    Thread-safe: all mutations are guarded by a lock.
    """

    def __init__(self, window_size: int = 1000) -> None:
        self._window_size = window_size
        self._latencies: list[float] = []
        self._recent: deque[InferenceMetric] = deque(maxlen=window_size)
        self._total_count: int = 0
        self._error_count: int = 0
        self._start_time: float = time.time()
        self._lock = threading.Lock()

    def record(
        self,
        inference_time_ms: float,
        model_id: str,
        success: bool = True,
    ) -> None:
        """Record a single inference measurement."""
        metric = InferenceMetric(
            inference_time_ms=inference_time_ms,
            model_id=model_id,
            success=success,
        )
        with self._lock:
            self._recent.append(metric)
            self._total_count += 1

            if not success:
                self._error_count += 1

            # Maintain sorted latency list for percentile computation
            if len(self._latencies) >= self._window_size:
                self._latencies.pop(0)
            bisect.insort(self._latencies, inference_time_ms)

    def get_summary(self) -> dict[str, Any]:
        """Return current metrics summary."""
        with self._lock:
            elapsed = time.time() - self._start_time
            return {
                "total_count": self._total_count,
                "error_count": self._error_count,
                "error_rate": self._error_count / max(self._total_count, 1),
                "p50_ms": self._percentile(50),
                "p95_ms": self._percentile(95),
                "p99_ms": self._percentile(99),
                "throughput_rps": self._total_count / max(elapsed, 0.001),
                "uptime_seconds": elapsed,
            }

    def _percentile(self, p: int) -> float:
        """Compute percentile from sorted latency list. Must hold lock."""
        if not self._latencies:
            return 0.0
        idx = int(len(self._latencies) * p / 100)
        idx = min(idx, len(self._latencies) - 1)
        return self._latencies[idx]

    def export_json(self) -> dict[str, Any]:
        """Export all metrics as a JSON-serializable dict.

        Returns:
            Complete metrics snapshot including summary and per-model breakdown.
        """
        with self._lock:
            per_model: dict[str, dict[str, int]] = {}
            for metric in self._recent:
                if metric.model_id not in per_model:
                    per_model[metric.model_id] = {"count": 0, "errors": 0}
                per_model[metric.model_id]["count"] += 1
                if not metric.success:
                    per_model[metric.model_id]["errors"] += 1

        return {
            "summary": self.get_summary(),
            "per_model": per_model,
        }

    def export_prometheus(self, prefix: str = "artenic_ai") -> str:
        """Export metrics in Prometheus text exposition format.

        Args:
            prefix: Metric name prefix.

        Returns:
            Prometheus-compatible text string.
        """
        summary = self.get_summary()
        lines = [
            f"# HELP {prefix}_inference_total Total inference count.",
            f"# TYPE {prefix}_inference_total counter",
            f"{prefix}_inference_total {summary['total_count']}",
            f"# HELP {prefix}_inference_errors_total Total inference errors.",
            f"# TYPE {prefix}_inference_errors_total counter",
            f"{prefix}_inference_errors_total {summary['error_count']}",
            f"# HELP {prefix}_inference_latency_p50_ms 50th percentile latency.",
            f"# TYPE {prefix}_inference_latency_p50_ms gauge",
            f"{prefix}_inference_latency_p50_ms {summary['p50_ms']}",
            f"# HELP {prefix}_inference_latency_p95_ms 95th percentile latency.",
            f"# TYPE {prefix}_inference_latency_p95_ms gauge",
            f"{prefix}_inference_latency_p95_ms {summary['p95_ms']}",
            f"# HELP {prefix}_inference_latency_p99_ms 99th percentile latency.",
            f"# TYPE {prefix}_inference_latency_p99_ms gauge",
            f"{prefix}_inference_latency_p99_ms {summary['p99_ms']}",
            f"# HELP {prefix}_throughput_rps Requests per second.",
            f"# TYPE {prefix}_throughput_rps gauge",
            f"{prefix}_throughput_rps {summary['throughput_rps']:.6f}",
        ]
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._latencies.clear()
            self._recent.clear()
            self._total_count = 0
            self._error_count = 0
            self._start_time = time.time()


# =============================================================================
# Trace Correlation
# =============================================================================

_trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "artenic_trace_id", default=None
)


@contextmanager
def correlation_context(trace_id: str | None = None) -> Generator[str, None, None]:
    """Context manager that sets a trace ID for log correlation.

    If no trace_id is provided, a new UUID is generated.
    The trace_id is propagated via contextvars so nested calls
    and async tasks within the block share the same ID.

    Args:
        trace_id: Explicit trace ID, or None for auto-generation.

    Yields:
        The active trace ID.
    """
    tid = trace_id or uuid.uuid4().hex
    token = _trace_id_var.set(tid)
    try:
        yield tid
    finally:
        _trace_id_var.reset(token)


def get_trace_id() -> str | None:
    """Return the current trace ID, or None if outside a correlation_context."""
    return _trace_id_var.get()


# =============================================================================
# Structured Logging
# =============================================================================


class StructuredLogger:
    """JSON-structured logger compatible with ELK, Datadog, etc.

    Automatically includes the current trace_id (from correlation_context)
    in every log entry for distributed tracing.

    Output format::

        {"timestamp": "...", "level": "INFO", "trace_id": "...",
         "model_id": "lgbm", "event": "predict.complete", "message": "..."}
    """

    def __init__(self, name: str = "artenic_ai") -> None:
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(_JsonFormatter())
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def _inject_trace(self, extra: dict[str, Any]) -> dict[str, Any]:
        """Inject trace_id from current correlation context."""
        tid = _trace_id_var.get()
        if tid is not None:
            extra.setdefault("trace_id", tid)
        return extra

    def info(self, message: str, **extra: Any) -> None:
        self._logger.info(message, extra={"structured": self._inject_trace(extra)})

    def warning(self, message: str, **extra: Any) -> None:
        self._logger.warning(message, extra={"structured": self._inject_trace(extra)})

    def error(self, message: str, **extra: Any) -> None:
        self._logger.error(message, extra={"structured": self._inject_trace(extra)})

    def debug(self, message: str, **extra: Any) -> None:
        self._logger.debug(message, extra={"structured": self._inject_trace(extra)})


class _JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        structured = getattr(record, "structured", {})
        entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            **structured,
        }
        return json.dumps(entry, default=str)


# =============================================================================
# Observability Mixin
# =============================================================================


class ObservabilityMixin:
    """Mixin that provides metrics and logging to BaseModel subclasses.

    When a class inherits from both BaseModel and ObservabilityMixin,
    it automatically gets:
    - self._metrics: MetricsCollector instance
    - self._logger: StructuredLogger instance
    - Automatic tracking of predict() calls via __init_subclass__
    """

    _metrics: MetricsCollector
    _logger: StructuredLogger
    _created_at: float

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Wrap predict() with tracking if the subclass defines it
        if "predict" in cls.__dict__:
            original = cls.__dict__["predict"]

            async def tracked_predict(self: Any, *args: Any, **kw: Any) -> Any:
                # Validate features if schema is declared
                if args and isinstance(args[0], dict) and hasattr(self, "validate_features"):
                    self.validate_features(args[0])

                start = time.perf_counter()
                success = True
                try:
                    result = await original(self, *args, **kw)
                    return result
                except Exception:
                    success = False
                    raise
                finally:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    self._metrics.record(
                        inference_time_ms=elapsed_ms,
                        model_id=getattr(self, "model_id", "unknown"),
                        success=success,
                    )
                    # Update inference tracking
                    self._inference_count += 1
                    self._last_inference_at = datetime.now()
                    if not success:
                        self._error_count += 1

            tracked_predict.__name__ = "predict"
            tracked_predict.__qualname__ = f"{cls.__qualname__}.predict"
            cls.predict = tracked_predict  # type: ignore[attr-defined]

    def _init_observability(self) -> None:
        """Initialize metrics and logger. Called by BaseModel.__init__."""
        self._metrics = MetricsCollector()
        self._logger = StructuredLogger(name=f"artenic_ai.{getattr(self, 'model_id', 'unknown')}")
        self._created_at = time.time()
