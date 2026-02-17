"""Tests for artenic_ai_platform.middleware.auth — 100% coverage."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

from artenic_ai_platform.middleware.auth import (
    EXEMPT_PATHS,
    EXEMPT_PREFIXES,
    AuthMiddleware,
    _header_value,
)

# ======================================================================
# _header_value
# ======================================================================


class TestHeaderValue:
    def test_found(self) -> None:
        headers = [(b"authorization", b"Bearer key123")]
        assert _header_value(headers, b"authorization") == "Bearer key123"

    def test_not_found(self) -> None:
        assert _header_value([], b"authorization") is None

    def test_case_insensitive(self) -> None:
        headers = [(b"Authorization", b"Bearer key")]
        assert _header_value(headers, b"authorization") == "Bearer key"


# ======================================================================
# Constants
# ======================================================================


class TestExemptPaths:
    def test_health_exempt(self) -> None:
        assert "/health" in EXEMPT_PATHS
        assert "/health/live" in EXEMPT_PATHS
        assert "/health/ready" in EXEMPT_PATHS

    def test_docs_exempt(self) -> None:
        assert "/docs" in EXEMPT_PATHS
        assert "/openapi.json" in EXEMPT_PATHS

    def test_root_exempt(self) -> None:
        assert "/" in EXEMPT_PATHS

    def test_assets_prefix(self) -> None:
        assert "/assets/" in EXEMPT_PREFIXES


# ======================================================================
# AuthMiddleware
# ======================================================================


class TestAuthMiddleware:
    async def test_non_http_passes_through(self) -> None:
        inner = AsyncMock()
        mw = AuthMiddleware(inner, api_key="secret")
        scope: dict[str, Any] = {"type": "lifespan"}

        await mw(scope, AsyncMock(), AsyncMock())

        inner.assert_awaited_once()

    async def test_dev_mode_no_key_passes_through(self) -> None:
        inner = AsyncMock()
        mw = AuthMiddleware(inner, api_key="")
        scope: dict[str, Any] = {
            "type": "http",
            "path": "/api/v1/models",
            "headers": [],
        }

        await mw(scope, AsyncMock(), AsyncMock())

        inner.assert_awaited_once()

    async def test_exempt_path_passes_through(self) -> None:
        inner = AsyncMock()
        mw = AuthMiddleware(inner, api_key="secret")

        for path in EXEMPT_PATHS:
            inner.reset_mock()
            scope: dict[str, Any] = {
                "type": "http",
                "path": path,
                "headers": [],
            }
            await mw(scope, AsyncMock(), AsyncMock())
            inner.assert_awaited_once()

    async def test_exempt_prefix_passes_through(self) -> None:
        inner = AsyncMock()
        mw = AuthMiddleware(inner, api_key="secret")
        scope: dict[str, Any] = {
            "type": "http",
            "path": "/assets/logo.png",
            "headers": [],
        }

        await mw(scope, AsyncMock(), AsyncMock())

        inner.assert_awaited_once()

    async def test_valid_bearer_token(self) -> None:
        inner = AsyncMock()
        mw = AuthMiddleware(inner, api_key="my-api-key")
        scope: dict[str, Any] = {
            "type": "http",
            "path": "/api/v1/models",
            "headers": [(b"authorization", b"Bearer my-api-key")],
        }

        await mw(scope, AsyncMock(), AsyncMock())

        inner.assert_awaited_once()

    async def test_invalid_bearer_token_returns_401(self) -> None:
        sent_messages: list[dict[str, Any]] = []

        async def mock_send(msg: dict[str, Any]) -> None:
            sent_messages.append(msg)

        mw = AuthMiddleware(AsyncMock(), api_key="correct-key")
        scope: dict[str, Any] = {
            "type": "http",
            "path": "/api/v1/models",
            "headers": [(b"authorization", b"Bearer wrong-key")],
        }

        await mw(scope, AsyncMock(), mock_send)

        assert sent_messages[0]["status"] == 401
        body = json.loads(sent_messages[1]["body"])
        assert body["error"]["code"] == "AUTHENTICATION_FAILED"

    async def test_missing_authorization_header_returns_401(self) -> None:
        sent_messages: list[dict[str, Any]] = []

        async def mock_send(msg: dict[str, Any]) -> None:
            sent_messages.append(msg)

        mw = AuthMiddleware(AsyncMock(), api_key="my-key")
        scope: dict[str, Any] = {
            "type": "http",
            "path": "/api/v1/models",
            "headers": [],
        }

        await mw(scope, AsyncMock(), mock_send)

        assert sent_messages[0]["status"] == 401

    async def test_non_bearer_auth_header_returns_401(self) -> None:
        sent_messages: list[dict[str, Any]] = []

        async def mock_send(msg: dict[str, Any]) -> None:
            sent_messages.append(msg)

        mw = AuthMiddleware(AsyncMock(), api_key="my-key")
        scope: dict[str, Any] = {
            "type": "http",
            "path": "/api/v1/models",
            "headers": [(b"authorization", b"Basic dXNlcjpwYXNz")],
        }

        await mw(scope, AsyncMock(), mock_send)

        assert sent_messages[0]["status"] == 401

    async def test_401_response_has_correct_content_type(self) -> None:
        sent_messages: list[dict[str, Any]] = []

        async def mock_send(msg: dict[str, Any]) -> None:
            sent_messages.append(msg)

        mw = AuthMiddleware(AsyncMock(), api_key="key")
        scope: dict[str, Any] = {
            "type": "http",
            "path": "/api/v1/models",
            "headers": [],
        }

        await mw(scope, AsyncMock(), mock_send)

        headers_dict = dict(sent_messages[0]["headers"])
        assert headers_dict[b"content-type"] == b"application/json"
        assert b"content-length" in headers_dict

    async def test_401_body_has_request_id(self) -> None:
        sent_messages: list[dict[str, Any]] = []

        async def mock_send(msg: dict[str, Any]) -> None:
            sent_messages.append(msg)

        mw = AuthMiddleware(AsyncMock(), api_key="key")
        scope: dict[str, Any] = {
            "type": "http",
            "path": "/api/v1/models",
            "headers": [],
        }

        await mw(scope, AsyncMock(), mock_send)

        body = json.loads(sent_messages[1]["body"])
        assert "request_id" in body["error"]

    async def test_websocket_with_no_key_passes(self) -> None:
        inner = AsyncMock()
        mw = AuthMiddleware(inner, api_key="")
        scope: dict[str, Any] = {"type": "websocket", "path": "/ws", "headers": []}

        await mw(scope, AsyncMock(), AsyncMock())

        inner.assert_awaited_once()

    async def test_default_path_when_missing(self) -> None:
        sent_messages: list[dict[str, Any]] = []

        async def mock_send(msg: dict[str, Any]) -> None:
            sent_messages.append(msg)

        mw = AuthMiddleware(AsyncMock(), api_key="key")
        scope: dict[str, Any] = {"type": "http", "headers": []}
        # No "path" key — defaults to "/"

        await mw(scope, AsyncMock(), mock_send)

        # "/" is in EXEMPT_PATHS, so it should pass through (no 401)
        # But inner was an AsyncMock — if exempt, inner is called
        # Actually "/" is exempt, so no 401 messages should be sent
        assert len(sent_messages) == 0
