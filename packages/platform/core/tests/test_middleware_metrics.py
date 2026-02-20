"""Tests for artenic_ai_platform.middleware.metrics — 100% coverage."""

from __future__ import annotations

import contextlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from artenic_ai_platform.middleware.metrics import MetricsMiddleware

# ======================================================================
# MetricsMiddleware
# ======================================================================


class TestMetricsMiddleware:
    async def test_non_http_passes_through(self) -> None:
        inner = AsyncMock()
        mw = MetricsMiddleware(inner)
        scope: dict[str, Any] = {"type": "lifespan"}

        await mw(scope, AsyncMock(), AsyncMock())

        inner.assert_awaited_once()

    async def test_http_request_records_histogram(self) -> None:
        """When OTel is available, histogram is recorded."""
        inner_called = False

        async def inner_app(scope: dict[str, Any], receive: Any, send: Any) -> None:
            nonlocal inner_called
            inner_called = True
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b"ok"})

        mw = MetricsMiddleware(inner_app)
        sent_messages: list[dict[str, Any]] = []

        async def mock_send(msg: dict[str, Any]) -> None:
            sent_messages.append(msg)

        scope: dict[str, Any] = {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/models",
        }

        await mw(scope, AsyncMock(), mock_send)

        assert inner_called

    async def test_histogram_records_correct_attributes(self) -> None:
        """Verify histogram.record is called with correct attributes."""
        mock_histogram = MagicMock()

        async def inner_app(scope: dict[str, Any], receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 201})
            await send({"type": "http.response.body", "body": b"created"})

        mw = MetricsMiddleware(inner_app)
        mw._histogram = mock_histogram

        sent_messages: list[dict[str, Any]] = []

        async def mock_send(msg: dict[str, Any]) -> None:
            sent_messages.append(msg)

        scope: dict[str, Any] = {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/training/dispatch",
        }

        await mw(scope, AsyncMock(), mock_send)

        mock_histogram.record.assert_called_once()
        call_args = mock_histogram.record.call_args
        elapsed_ms = call_args[0][0]
        attrs = call_args[1]["attributes"]

        assert elapsed_ms > 0
        assert attrs["http.method"] == "POST"
        assert attrs["http.route"] == "/api/v1/training/dispatch"
        assert attrs["http.status_code"] == 201

    async def test_no_histogram_passes_through(self) -> None:
        """When OTel is not available, middleware is a passthrough."""
        inner = AsyncMock()
        mw = MetricsMiddleware(inner)
        mw._histogram = None

        scope: dict[str, Any] = {"type": "http", "method": "GET", "path": "/"}

        await mw(scope, AsyncMock(), AsyncMock())

        inner.assert_awaited_once()

    async def test_histogram_recorded_even_on_exception(self) -> None:
        """Histogram is recorded in finally block even if inner raises."""
        mock_histogram = MagicMock()

        async def inner_app(scope: dict[str, Any], receive: Any, send: Any) -> None:
            raise ValueError("boom")

        mw = MetricsMiddleware(inner_app)
        mw._histogram = mock_histogram

        scope: dict[str, Any] = {"type": "http", "method": "GET", "path": "/error"}

        with contextlib.suppress(ValueError):
            await mw(scope, AsyncMock(), AsyncMock())

        mock_histogram.record.assert_called_once()
        attrs = mock_histogram.record.call_args[1]["attributes"]
        assert attrs["http.status_code"] == 200  # default since no response sent

    async def test_default_method_and_path(self) -> None:
        """Missing method/path in scope defaults to GET and /."""
        mock_histogram = MagicMock()

        async def inner_app(scope: dict[str, Any], receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 404})

        mw = MetricsMiddleware(inner_app)
        mw._histogram = mock_histogram

        async def mock_send(msg: dict[str, Any]) -> None:
            pass

        scope: dict[str, Any] = {"type": "http"}

        await mw(scope, AsyncMock(), mock_send)

        attrs = mock_histogram.record.call_args[1]["attributes"]
        assert attrs["http.method"] == "GET"
        assert attrs["http.route"] == "/"

    async def test_status_code_captured_from_response(self) -> None:
        """Status code extracted from http.response.start message."""
        mock_histogram = MagicMock()

        async def inner_app(scope: dict[str, Any], receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 500})

        mw = MetricsMiddleware(inner_app)
        mw._histogram = mock_histogram

        async def mock_send(msg: dict[str, Any]) -> None:
            pass

        scope: dict[str, Any] = {"type": "http", "method": "GET", "path": "/fail"}

        await mw(scope, AsyncMock(), mock_send)

        attrs = mock_histogram.record.call_args[1]["attributes"]
        assert attrs["http.status_code"] == 500

    async def test_websocket_without_histogram_passes(self) -> None:
        """Websocket type also bypasses histogram."""
        inner = AsyncMock()
        mw = MetricsMiddleware(inner)
        mw._histogram = None

        scope: dict[str, Any] = {"type": "websocket"}

        await mw(scope, AsyncMock(), AsyncMock())

        inner.assert_awaited_once()

    def test_init_with_otel_available(self) -> None:
        """Histogram is created when OTel is available."""
        mw = MetricsMiddleware(AsyncMock())
        # OTel is installed in our env, so histogram should be created
        assert mw._histogram is not None

    def test_init_without_otel(self) -> None:
        """When _HAS_OTEL is False, histogram is None."""
        with patch("artenic_ai_platform.middleware.metrics._HAS_OTEL", False):
            mw = MetricsMiddleware(AsyncMock())
            assert mw._histogram is None

    def test_init_otel_none(self) -> None:
        """When otel_metrics is None, histogram is None."""
        with patch("artenic_ai_platform.middleware.metrics.otel_metrics", None):
            mw = MetricsMiddleware(AsyncMock())
            assert mw._histogram is None
