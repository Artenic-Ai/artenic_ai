"""Tests for artenic_ai_platform.middleware.rate_limit — 100% coverage."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

from artenic_ai_platform.middleware.rate_limit import (
    EXEMPT_PATHS,
    RateLimitMiddleware,
    TokenBucket,
    _client_key,
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
        assert _header_value([], b"x-custom") is None

    def test_case_insensitive(self) -> None:
        headers = [(b"Authorization", b"Bearer key")]
        assert _header_value(headers, b"authorization") == "Bearer key"


# ======================================================================
# _client_key
# ======================================================================


class TestClientKey:
    def test_bearer_key(self) -> None:
        scope: dict[str, Any] = {
            "headers": [(b"authorization", b"Bearer my-key")],
            "client": ("1.2.3.4", 12345),
        }
        assert _client_key(scope) == "key:my-key"

    def test_ip_fallback(self) -> None:
        scope: dict[str, Any] = {
            "headers": [],
            "client": ("10.0.0.1", 8080),
        }
        assert _client_key(scope) == "ip:10.0.0.1"

    def test_no_client_tuple(self) -> None:
        scope: dict[str, Any] = {"headers": []}
        assert _client_key(scope) == "ip:unknown"

    def test_non_bearer_auth_header(self) -> None:
        scope: dict[str, Any] = {
            "headers": [(b"authorization", b"Basic abc")],
            "client": ("5.5.5.5", 80),
        }
        assert _client_key(scope) == "ip:5.5.5.5"


# ======================================================================
# TokenBucket
# ======================================================================


class TestTokenBucket:
    def test_initial_state(self) -> None:
        bucket = TokenBucket(capacity=10, rate=1.0)
        assert bucket.tokens == 10.0
        assert bucket.capacity == 10
        assert bucket.rate == 1.0

    def test_consume_success(self) -> None:
        bucket = TokenBucket(capacity=5, rate=1.0)
        assert bucket.consume() is True
        assert bucket.tokens < 5.0

    def test_consume_until_empty(self) -> None:
        bucket = TokenBucket(capacity=3, rate=0.0)  # no refill
        assert bucket.consume() is True
        assert bucket.consume() is True
        assert bucket.consume() is True
        assert bucket.consume() is False

    def test_retry_after_with_tokens(self) -> None:
        bucket = TokenBucket(capacity=5, rate=1.0)
        assert bucket.retry_after() == 0.0

    def test_retry_after_when_empty(self) -> None:
        bucket = TokenBucket(capacity=1, rate=1.0)
        bucket.tokens = 0.0
        retry = bucket.retry_after()
        assert retry > 0.0
        assert retry <= 1.0

    def test_refill_over_time(self) -> None:
        bucket = TokenBucket(capacity=10, rate=100.0)
        # Drain all tokens
        for _ in range(10):
            bucket.consume()
        # With rate=100/s, tokens refill quickly
        # Force time passage by manipulating last_refill
        bucket.last_refill -= 1.0  # simulate 1 second passing
        assert bucket.consume() is True

    def test_tokens_capped_at_capacity(self) -> None:
        bucket = TokenBucket(capacity=5, rate=100.0)
        bucket.last_refill -= 100.0  # lots of time passed
        bucket.consume()  # triggers refill
        # tokens should be capped at capacity - 1 (after consume)
        assert bucket.tokens <= bucket.capacity


# ======================================================================
# RateLimitMiddleware
# ======================================================================


class TestRateLimitMiddleware:
    async def test_non_http_passes_through(self) -> None:
        inner = AsyncMock()
        mw = RateLimitMiddleware(inner, per_minute=60, burst=10)
        scope: dict[str, Any] = {"type": "lifespan"}

        await mw(scope, AsyncMock(), AsyncMock())

        inner.assert_awaited_once()

    async def test_exempt_path_passes_through(self) -> None:
        inner = AsyncMock()
        mw = RateLimitMiddleware(inner, per_minute=60, burst=10)

        for path in EXEMPT_PATHS:
            inner.reset_mock()
            scope: dict[str, Any] = {
                "type": "http",
                "path": path,
                "headers": [],
                "client": ("1.2.3.4", 80),
            }
            await mw(scope, AsyncMock(), AsyncMock())
            inner.assert_awaited_once()

    async def test_request_allowed_under_quota(self) -> None:
        inner = AsyncMock()
        mw = RateLimitMiddleware(inner, per_minute=60, burst=10)
        scope: dict[str, Any] = {
            "type": "http",
            "path": "/api/v1/models",
            "headers": [],
            "client": ("1.2.3.4", 80),
        }

        await mw(scope, AsyncMock(), AsyncMock())

        inner.assert_awaited_once()

    async def test_request_blocked_over_quota(self) -> None:
        sent_messages: list[dict[str, Any]] = []

        async def mock_send(msg: dict[str, Any]) -> None:
            sent_messages.append(msg)

        mw = RateLimitMiddleware(AsyncMock(), per_minute=60, burst=2)

        for _ in range(2):
            scope: dict[str, Any] = {
                "type": "http",
                "path": "/api/v1/models",
                "headers": [],
                "client": ("1.2.3.4", 80),
            }
            await mw(scope, AsyncMock(), AsyncMock())

        # 3rd request should be blocked
        scope = {
            "type": "http",
            "path": "/api/v1/models",
            "headers": [],
            "client": ("1.2.3.4", 80),
        }
        await mw(scope, AsyncMock(), mock_send)

        assert sent_messages[0]["status"] == 429
        body = json.loads(sent_messages[1]["body"])
        assert body["error"]["code"] == "RATE_LIMIT_EXCEEDED"
        assert "retry_after_seconds" in body["error"]["details"]

    async def test_429_has_retry_after_header(self) -> None:
        sent_messages: list[dict[str, Any]] = []

        async def mock_send(msg: dict[str, Any]) -> None:
            sent_messages.append(msg)

        mw = RateLimitMiddleware(AsyncMock(), per_minute=60, burst=1)

        # First request uses the token
        scope: dict[str, Any] = {
            "type": "http",
            "path": "/api/v1/test",
            "headers": [(b"authorization", b"Bearer uniq-key")],
            "client": ("2.2.2.2", 80),
        }
        await mw(scope, AsyncMock(), AsyncMock())

        # Second request gets 429
        await mw(scope, AsyncMock(), mock_send)

        headers_dict = dict(sent_messages[0]["headers"])
        assert b"retry-after" in headers_dict

    async def test_per_client_isolation(self) -> None:
        inner = AsyncMock()
        mw = RateLimitMiddleware(inner, per_minute=60, burst=1)

        # Client A uses its token
        scope_a: dict[str, Any] = {
            "type": "http",
            "path": "/api/v1/test",
            "headers": [],
            "client": ("10.0.0.1", 80),
        }
        await mw(scope_a, AsyncMock(), AsyncMock())

        # Client B should still have its own bucket
        scope_b: dict[str, Any] = {
            "type": "http",
            "path": "/api/v1/test",
            "headers": [],
            "client": ("10.0.0.2", 80),
        }
        await mw(scope_b, AsyncMock(), AsyncMock())

        assert inner.await_count == 2

    async def test_websocket_rate_limited(self) -> None:
        sent_messages: list[dict[str, Any]] = []

        async def mock_send(msg: dict[str, Any]) -> None:
            sent_messages.append(msg)

        mw = RateLimitMiddleware(AsyncMock(), per_minute=60, burst=1)

        scope: dict[str, Any] = {
            "type": "websocket",
            "path": "/ws",
            "headers": [],
            "client": ("3.3.3.3", 80),
        }
        await mw(scope, AsyncMock(), AsyncMock())
        await mw(scope, AsyncMock(), mock_send)

        assert sent_messages[0]["status"] == 429

    async def test_bucket_reuse_for_same_client(self) -> None:
        mw = RateLimitMiddleware(AsyncMock(), per_minute=60, burst=5)

        scope: dict[str, Any] = {
            "type": "http",
            "path": "/api/v1/test",
            "headers": [],
            "client": ("4.4.4.4", 80),
        }
        await mw(scope, AsyncMock(), AsyncMock())
        await mw(scope, AsyncMock(), AsyncMock())

        # Same bucket reused
        assert len(mw._buckets) == 1
