"""Tests for artenic_ai_sdk.observability — MetricsCollector, StructuredLogger."""

from __future__ import annotations

import json
import logging

from artenic_ai_sdk.observability import (
    MetricsCollector,
    StructuredLogger,
    _JsonFormatter,
    correlation_context,
    get_trace_id,
)


class TestMetricsCollector:
    def test_empty_summary(self) -> None:
        mc = MetricsCollector()
        summary = mc.get_summary()
        assert summary["total_count"] == 0
        assert summary["error_count"] == 0
        assert summary["p50_ms"] == 0.0

    def test_record_success(self) -> None:
        mc = MetricsCollector()
        mc.record(10.0, "model_a", success=True)
        mc.record(20.0, "model_a", success=True)
        summary = mc.get_summary()
        assert summary["total_count"] == 2
        assert summary["error_count"] == 0

    def test_record_failure(self) -> None:
        mc = MetricsCollector()
        mc.record(5.0, "model_a", success=False)
        summary = mc.get_summary()
        assert summary["error_count"] == 1
        assert summary["error_rate"] == 1.0

    def test_percentiles(self) -> None:
        mc = MetricsCollector()
        for i in range(100):
            mc.record(float(i), "model_a")
        summary = mc.get_summary()
        assert summary["p50_ms"] == 50.0
        assert summary["p95_ms"] == 95.0
        assert summary["p99_ms"] == 99.0

    def test_window_size(self) -> None:
        mc = MetricsCollector(window_size=10)
        for i in range(20):
            mc.record(float(i), "model_a")
        summary = mc.get_summary()
        assert summary["total_count"] == 20
        # Latencies list should be trimmed to window_size
        assert len(mc._latencies) <= 10

    def test_reset(self) -> None:
        mc = MetricsCollector()
        mc.record(10.0, "model_a")
        mc.reset()
        summary = mc.get_summary()
        assert summary["total_count"] == 0

    def test_throughput(self) -> None:
        mc = MetricsCollector()
        mc.record(1.0, "model_a")
        summary = mc.get_summary()
        assert summary["throughput_rps"] > 0
        assert summary["uptime_seconds"] >= 0


class TestStructuredLogger:
    def test_logger_creation(self) -> None:
        logger = StructuredLogger("test_logger")
        assert logger._logger.name == "test_logger"

    def test_info_log(self, caplog: logging.LogRecord) -> None:  # type: ignore[type-arg]
        logger = StructuredLogger("test_info")
        logger._logger.setLevel(logging.DEBUG)
        logger.info("test message", model_id="test")
        # The handler uses _JsonFormatter, so we verify the logger works without error

    def test_all_levels(self) -> None:
        logger = StructuredLogger("test_levels")
        logger._logger.setLevel(logging.DEBUG)
        logger.debug("debug msg")
        logger.info("info msg")
        logger.warning("warning msg")
        logger.error("error msg")


class TestMetricsExportJson:
    def test_empty(self) -> None:
        mc = MetricsCollector()
        data = mc.export_json()
        assert data["summary"]["total_count"] == 0
        assert data["per_model"] == {}

    def test_with_records(self) -> None:
        mc = MetricsCollector()
        mc.record(10.0, "model_a", success=True)
        mc.record(20.0, "model_a", success=False)
        mc.record(5.0, "model_b", success=True)
        data = mc.export_json()
        assert data["per_model"]["model_a"]["count"] == 2
        assert data["per_model"]["model_a"]["errors"] == 1
        assert data["per_model"]["model_b"]["count"] == 1
        assert data["per_model"]["model_b"]["errors"] == 0


class TestMetricsExportPrometheus:
    def test_format(self) -> None:
        mc = MetricsCollector()
        mc.record(10.0, "model_a")
        text = mc.export_prometheus()
        assert "artenic_ai_inference_total 1" in text
        assert "artenic_ai_inference_errors_total 0" in text
        assert "artenic_ai_inference_latency_p50_ms" in text
        assert "# HELP" in text
        assert "# TYPE" in text
        assert text.endswith("\n")

    def test_custom_prefix(self) -> None:
        mc = MetricsCollector()
        mc.record(1.0, "m")
        text = mc.export_prometheus(prefix="myapp")
        assert "myapp_inference_total" in text


class TestCorrelationContext:
    def test_auto_trace_id(self) -> None:
        assert get_trace_id() is None
        with correlation_context() as tid:
            assert tid is not None
            assert len(tid) == 32  # hex UUID
            assert get_trace_id() == tid
        assert get_trace_id() is None

    def test_explicit_trace_id(self) -> None:
        with correlation_context(trace_id="my-custom-trace") as tid:
            assert tid == "my-custom-trace"
            assert get_trace_id() == "my-custom-trace"

    def test_nested_contexts(self) -> None:
        with correlation_context(trace_id="outer") as outer:
            assert get_trace_id() == "outer"
            with correlation_context(trace_id="inner") as inner:
                assert get_trace_id() == "inner"
                assert inner == "inner"
            assert get_trace_id() == "outer"
            assert outer == "outer"

    def test_logger_includes_trace_id(self) -> None:
        logger = StructuredLogger("test_trace")
        logger._logger.setLevel(logging.DEBUG)

        # Without context — no trace_id injected
        logger.info("no trace")

        # With context — trace_id injected
        with correlation_context(trace_id="abc123"):
            logger.info("traced msg", model_id="test")


class TestJsonFormatter:
    def test_format(self) -> None:
        formatter = _JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        record.structured = {"model_id": "test"}  # type: ignore[attr-defined]
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "hello"
        assert parsed["level"] == "INFO"
        assert parsed["model_id"] == "test"
        assert "timestamp" in parsed
