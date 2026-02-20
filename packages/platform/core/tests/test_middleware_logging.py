"""Tests for artenic_ai_platform.middleware.logging — 100% coverage."""

from __future__ import annotations

import json
import logging

from artenic_ai_platform.middleware.correlation import correlation_id_var
from artenic_ai_platform.middleware.logging import (
    CorrelationIdFilter,
    PlatformJsonFormatter,
    setup_logging,
)

# ======================================================================
# CorrelationIdFilter
# ======================================================================


class TestCorrelationIdFilter:
    def test_injects_request_id(self) -> None:
        token = correlation_id_var.set("req-abc")
        try:
            f = CorrelationIdFilter()
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="hello",
                args=(),
                exc_info=None,
            )
            result = f.filter(record)
            assert result is True
            assert record.request_id == "req-abc"  # type: ignore[attr-defined]
        finally:
            correlation_id_var.reset(token)

    def test_empty_request_id_when_not_set(self) -> None:
        token = correlation_id_var.set("")
        try:
            f = CorrelationIdFilter()
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="test",
                args=(),
                exc_info=None,
            )
            f.filter(record)
            assert record.request_id == ""  # type: ignore[attr-defined]
        finally:
            correlation_id_var.reset(token)


# ======================================================================
# PlatformJsonFormatter
# ======================================================================


class TestPlatformJsonFormatter:
    def test_basic_format(self) -> None:
        fmt = PlatformJsonFormatter()
        record = logging.LogRecord(
            name="artenic_ai_platform.app",
            level=logging.INFO,
            pathname="app.py",
            lineno=42,
            msg="Server started",
            args=(),
            exc_info=None,
        )
        record.request_id = "rid-123"  # type: ignore[attr-defined]
        output = fmt.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["logger"] == "artenic_ai_platform.app"
        assert data["message"] == "Server started"
        assert data["request_id"] == "rid-123"
        assert data["exception"] is None
        assert "timestamp" in data

    def test_format_with_exception(self) -> None:
        fmt = PlatformJsonFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Something failed",
                args=(),
                exc_info=sys.exc_info(),
            )
        record.request_id = ""  # type: ignore[attr-defined]
        output = fmt.format(record)
        data = json.loads(output)

        assert data["level"] == "ERROR"
        assert data["exception"] is not None
        assert "ValueError" in data["exception"]
        assert "test error" in data["exception"]

    def test_format_without_request_id_attr(self) -> None:
        fmt = PlatformJsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="warning",
            args=(),
            exc_info=None,
        )
        # No request_id attribute set
        output = fmt.format(record)
        data = json.loads(output)
        assert data["request_id"] == ""

    def test_format_with_args(self) -> None:
        fmt = PlatformJsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Value is %d",
            args=(42,),
            exc_info=None,
        )
        output = fmt.format(record)
        data = json.loads(output)
        assert data["message"] == "Value is 42"

    def test_timestamp_format(self) -> None:
        fmt = PlatformJsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        output = fmt.format(record)
        data = json.loads(output)
        # ISO 8601 format with timezone
        assert "T" in data["timestamp"]
        assert "+" in data["timestamp"] or "Z" in data["timestamp"]


# ======================================================================
# setup_logging
# ======================================================================


class TestSetupLogging:
    def test_configures_root_logger(self) -> None:
        # Save original state
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        original_level = root.level

        try:
            setup_logging(level=logging.DEBUG)
            assert root.level == logging.DEBUG
            assert len(root.handlers) == 1

            handler = root.handlers[0]
            assert isinstance(handler.formatter, PlatformJsonFormatter)

            # Check filter is installed
            filter_types = [type(f) for f in handler.filters]
            assert CorrelationIdFilter in filter_types
        finally:
            # Restore original state
            root.handlers = original_handlers
            root.setLevel(original_level)

    def test_removes_existing_handlers(self) -> None:
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        original_level = root.level

        try:
            # Add extra handlers
            root.addHandler(logging.StreamHandler())
            root.addHandler(logging.StreamHandler())
            assert len(root.handlers) >= 2

            setup_logging()
            assert len(root.handlers) == 1
        finally:
            root.handlers = original_handlers
            root.setLevel(original_level)

    def test_default_level_is_info(self) -> None:
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        original_level = root.level

        try:
            setup_logging()
            assert root.level == logging.INFO
        finally:
            root.handlers = original_handlers
            root.setLevel(original_level)
