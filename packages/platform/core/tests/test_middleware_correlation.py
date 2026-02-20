"""Tests for artenic_ai_platform.middleware.correlation — 100% coverage."""

from __future__ import annotations

import contextlib
from typing import Any
from unittest.mock import AsyncMock

from artenic_ai_platform.middleware.correlation import (
    CorrelationIdMiddleware,
    _header_value,
    correlation_id_var,
    get_request_id,
)

# ======================================================================
# get_request_id + correlation_id_var
# ======================================================================


class TestGetRequestId:
    def test_default_is_empty_string(self) -> None:
        token = correlation_id_var.set("")
        try:
            assert get_request_id() == ""
        finally:
            correlation_id_var.reset(token)

    def test_returns_set_value(self) -> None:
        token = correlation_id_var.set("abc-123")
        try:
            assert get_request_id() == "abc-123"
        finally:
            correlation_id_var.reset(token)


# ======================================================================
# _header_value
# ======================================================================


class TestHeaderValue:
    def test_found(self) -> None:
        headers = [(b"x-request-id", b"my-id"), (b"content-type", b"text/html")]
        assert _header_value(headers, b"x-request-id") == "my-id"

    def test_not_found(self) -> None:
        headers = [(b"content-type", b"text/html")]
        assert _header_value(headers, b"x-request-id") is None

    def test_case_insensitive(self) -> None:
        headers = [(b"X-Request-ID", b"my-id")]
        assert _header_value(headers, b"x-request-id") == "my-id"

    def test_empty_headers(self) -> None:
        assert _header_value([], b"x-request-id") is None


# ======================================================================
# CorrelationIdMiddleware
# ======================================================================


class TestCorrelationIdMiddleware:
    async def test_non_http_scope_passes_through(self) -> None:
        inner = AsyncMock()
        mw = CorrelationIdMiddleware(inner)
        scope: dict[str, Any] = {"type": "lifespan"}

        await mw(scope, AsyncMock(), AsyncMock())

        inner.assert_awaited_once_with(scope, inner.call_args[0][1], inner.call_args[0][2])

    async def test_generates_uuid_when_no_header(self) -> None:
        captured_state: dict[str, Any] = {}
        captured_context_id: list[str] = []

        async def inner_app(scope: dict[str, Any], receive: Any, send: Any) -> None:
            captured_state.update(scope.get("state", {}))
            captured_context_id.append(get_request_id())

        mw = CorrelationIdMiddleware(inner_app)
        scope: dict[str, Any] = {"type": "http", "headers": []}

        await mw(scope, AsyncMock(), AsyncMock())

        assert "request_id" in captured_state
        assert len(captured_state["request_id"]) == 32  # UUID hex
        assert captured_context_id[0] == captured_state["request_id"]

    async def test_uses_provided_header(self) -> None:
        captured_id: list[str] = []

        async def inner_app(scope: dict[str, Any], receive: Any, send: Any) -> None:
            captured_id.append(get_request_id())

        mw = CorrelationIdMiddleware(inner_app)
        scope: dict[str, Any] = {
            "type": "http",
            "headers": [(b"x-request-id", b"custom-id-123")],
        }

        await mw(scope, AsyncMock(), AsyncMock())

        assert captured_id[0] == "custom-id-123"

    async def test_injects_header_in_response(self) -> None:
        sent_messages: list[dict[str, Any]] = []

        async def inner_app(scope: dict[str, Any], receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        async def mock_send(message: dict[str, Any]) -> None:
            sent_messages.append(message)

        mw = CorrelationIdMiddleware(inner_app)
        scope: dict[str, Any] = {
            "type": "http",
            "headers": [(b"x-request-id", b"test-id")],
        }

        await mw(scope, AsyncMock(), mock_send)

        start_msg = sent_messages[0]
        assert start_msg["type"] == "http.response.start"
        header_dict = dict(start_msg["headers"])
        assert header_dict[b"x-request-id"] == b"test-id"

    async def test_resets_context_var_after_request(self) -> None:
        token = correlation_id_var.set("before")

        async def inner_app(scope: dict[str, Any], receive: Any, send: Any) -> None:
            assert get_request_id() != "before"

        mw = CorrelationIdMiddleware(inner_app)
        scope: dict[str, Any] = {"type": "http", "headers": []}

        await mw(scope, AsyncMock(), AsyncMock())

        assert correlation_id_var.get() == "before"
        correlation_id_var.reset(token)

    async def test_resets_context_var_even_on_exception(self) -> None:
        token = correlation_id_var.set("initial")

        async def inner_app(scope: dict[str, Any], receive: Any, send: Any) -> None:
            raise ValueError("boom")

        mw = CorrelationIdMiddleware(inner_app)
        scope: dict[str, Any] = {"type": "http", "headers": []}

        with contextlib.suppress(ValueError):
            await mw(scope, AsyncMock(), AsyncMock())

        assert correlation_id_var.get() == "initial"
        correlation_id_var.reset(token)

    async def test_websocket_scope(self) -> None:
        captured_id: list[str] = []

        async def inner_app(scope: dict[str, Any], receive: Any, send: Any) -> None:
            captured_id.append(get_request_id())

        mw = CorrelationIdMiddleware(inner_app)
        scope: dict[str, Any] = {"type": "websocket", "headers": []}

        await mw(scope, AsyncMock(), AsyncMock())

        assert len(captured_id[0]) == 32  # UUID hex generated

    async def test_non_start_messages_not_modified(self) -> None:
        sent_messages: list[dict[str, Any]] = []

        async def inner_app(scope: dict[str, Any], receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"data"})

        async def mock_send(message: dict[str, Any]) -> None:
            sent_messages.append(message)

        mw = CorrelationIdMiddleware(inner_app)
        scope: dict[str, Any] = {"type": "http", "headers": []}

        await mw(scope, AsyncMock(), mock_send)

        body_msg = sent_messages[1]
        assert body_msg["type"] == "http.response.body"
        assert "headers" not in body_msg

    async def test_scope_state_created_if_missing(self) -> None:
        async def inner_app(scope: dict[str, Any], receive: Any, send: Any) -> None:
            pass

        mw = CorrelationIdMiddleware(inner_app)
        scope: dict[str, Any] = {"type": "http", "headers": []}
        # No "state" key in scope

        await mw(scope, AsyncMock(), AsyncMock())

        assert "state" in scope
        assert "request_id" in scope["state"]
