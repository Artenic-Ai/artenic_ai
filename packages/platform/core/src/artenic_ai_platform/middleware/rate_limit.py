"""Token-bucket rate-limiting middleware (pure ASGI).

Each client (identified by API key or IP address) gets an independent
token bucket.  When the bucket is empty the middleware returns HTTP 429
with a ``Retry-After`` header.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from artenic_ai_platform.middleware.correlation import get_request_id

EXEMPT_PATHS: set[str] = {
    "/health",
    "/health/live",
    "/health/ready",
    "/docs",
    "/openapi.json",
    "/",
}

_Scope = dict[str, Any]
_Receive = Any
_Send = Any


@dataclass
class TokenBucket:
    """A simple token-bucket rate limiter.

    Parameters
    ----------
    capacity:
        Maximum number of tokens the bucket can hold (burst size).
    rate:
        Tokens added per second.
    """

    capacity: int
    rate: float
    tokens: float = field(init=False)
    last_refill: float = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = float(self.capacity)
        self.last_refill = time.monotonic()

    def consume(self) -> bool:
        """Attempt to consume one token.

        Returns ``True`` if the request is allowed, ``False`` if the
        bucket is empty.
        """
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.rate,
        )
        self.last_refill = now

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    def retry_after(self) -> float:
        """Seconds until at least one token is available."""
        if self.tokens >= 1.0:
            return 0.0
        return (1.0 - self.tokens) / self.rate


def _header_value(
    headers: list[tuple[bytes, bytes]],
    name: bytes,
) -> str | None:
    for key, val in headers:
        if key.lower() == name:
            return val.decode("latin-1")
    return None


def _client_key(scope: _Scope) -> str:
    """Derive a per-client key from the ASGI scope."""
    headers: list[tuple[bytes, bytes]] = scope.get(
        "headers",
        [],
    )
    auth = _header_value(headers, b"authorization")
    if auth and auth.lower().startswith("bearer "):
        return f"key:{auth[7:]}"

    # Fallback to client IP -------------------------------------------
    client: tuple[str, int] | None = scope.get("client")
    ip = client[0] if client else "unknown"
    return f"ip:{ip}"


class RateLimitMiddleware:
    """Pure ASGI middleware implementing per-client rate limiting.

    Parameters
    ----------
    app:
        The inner ASGI application.
    per_minute:
        Sustained requests per minute allowed per client.
    burst:
        Maximum burst size (token bucket capacity).
    """

    def __init__(
        self,
        app: Any,
        per_minute: int,
        burst: int,
    ) -> None:
        self.app = app
        self.per_minute = per_minute
        self.burst = burst
        self._rate: float = per_minute / 60.0
        self._buckets: dict[str, TokenBucket] = {}

    def _get_bucket(self, key: str) -> TokenBucket:
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = TokenBucket(
                capacity=self.burst,
                rate=self._rate,
            )
            self._buckets[key] = bucket
        return bucket

    async def __call__(
        self,
        scope: _Scope,
        receive: _Receive,
        send: _Send,
    ) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "/")
        if path in EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        key = _client_key(scope)
        bucket = self._get_bucket(key)

        if bucket.consume():
            await self.app(scope, receive, send)
            return

        retry = bucket.retry_after()
        await self._send_429(send, retry)

    @staticmethod
    async def _send_429(send: _Send, retry_after: float) -> None:
        body = json.dumps(
            {
                "error": {
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": "Too many requests.",
                    "request_id": get_request_id(),
                    "details": {
                        "retry_after_seconds": round(
                            retry_after,
                            2,
                        ),
                    },
                },
            },
        ).encode()

        retry_str = str(int(retry_after) + 1)

        await send(
            {
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    (b"content-type", b"application/json"),
                    (
                        b"content-length",
                        str(len(body)).encode(),
                    ),
                    (b"retry-after", retry_str.encode()),
                ],
            },
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
            },
        )
