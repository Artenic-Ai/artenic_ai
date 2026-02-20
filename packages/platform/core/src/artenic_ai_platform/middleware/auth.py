"""API-key authentication middleware (pure ASGI).

Validates ``Authorization: Bearer <key>`` against a pre-shared API key
using constant-time comparison.  Paths in :data:`EXEMPT_PATHS` and
prefixes in :data:`EXEMPT_PREFIXES` are allowed without credentials.
"""

from __future__ import annotations

import hmac
import json
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

EXEMPT_PREFIXES: tuple[str, ...] = ("/assets/",)

_Scope = dict[str, Any]
_Receive = Any
_Send = Any


def _header_value(
    headers: list[tuple[bytes, bytes]],
    name: bytes,
) -> str | None:
    for key, val in headers:
        if key.lower() == name:
            return val.decode("latin-1")
    return None


class AuthMiddleware:
    """Pure ASGI middleware for Bearer-token authentication.

    Parameters
    ----------
    app:
        The inner ASGI application.
    api_key:
        The expected API key.  When empty the middleware is disabled
        (development / testing mode) and all requests pass through.
    """

    def __init__(self, app: Any, api_key: str) -> None:
        self.app = app
        self.api_key = api_key

    async def __call__(
        self,
        scope: _Scope,
        receive: _Receive,
        send: _Send,
    ) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        # Dev mode — no key configured ---------------------------------
        if not self.api_key:
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "/")

        if path in EXEMPT_PATHS or any(path.startswith(p) for p in EXEMPT_PREFIXES):
            await self.app(scope, receive, send)
            return

        # Extract Bearer token -----------------------------------------
        headers: list[tuple[bytes, bytes]] = scope.get(
            "headers",
            [],
        )
        auth_header = _header_value(headers, b"authorization")
        token = ""
        if auth_header and auth_header.lower().startswith("bearer "):
            token = auth_header[7:]

        if hmac.compare_digest(token, self.api_key):
            await self.app(scope, receive, send)
            return

        # Reject -------------------------------------------------------
        await self._send_401(send)

    @staticmethod
    async def _send_401(send: _Send) -> None:
        body = json.dumps(
            {
                "error": {
                    "code": "AUTHENTICATION_FAILED",
                    "message": "Invalid or missing API key.",
                    "request_id": get_request_id(),
                },
            },
        ).encode()

        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (
                        b"content-length",
                        str(len(body)).encode(),
                    ),
                ],
            },
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
            },
        )
