"""Correlation-ID middleware for request tracing.

Implements a pure ASGI middleware that reads or generates a unique
request identifier and propagates it through a :class:`~contextvars.ContextVar`.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from typing import Any

correlation_id_var: ContextVar[str] = ContextVar(
    "correlation_id",
    default="",
)


def get_request_id() -> str:
    """Return the current correlation / request ID."""
    return correlation_id_var.get()


# -- internal helpers -----------------------------------------------------

_Scope = dict[str, Any]
_Receive = Any  # ASGI Receive callable
_Send = Any  # ASGI Send callable


def _header_value(
    headers: list[tuple[bytes, bytes]],
    name: bytes,
) -> str | None:
    """Extract the first value for *name* from raw ASGI headers."""
    for key, val in headers:
        if key.lower() == name:
            return val.decode("latin-1")
    return None


class CorrelationIdMiddleware:
    """Pure ASGI middleware that manages ``x-request-id``.

    * Reads the incoming ``x-request-id`` header; falls back to a new
      UUID-4 if the header is absent.
    * Stores the ID in ``scope["state"]["request_id"]`` and in the
      module-level :data:`correlation_id_var` context variable.
    * Injects the same ``x-request-id`` into every response.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(
        self,
        scope: _Scope,
        receive: _Receive,
        send: _Send,
    ) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        headers: list[tuple[bytes, bytes]] = scope.get(
            "headers",
            [],
        )
        request_id = _header_value(headers, b"x-request-id") or uuid.uuid4().hex

        # Store in scope state -----------------------------------------
        state: dict[str, Any] = scope.setdefault("state", {})
        state["request_id"] = request_id

        # Store in ContextVar ------------------------------------------
        token = correlation_id_var.set(request_id)

        async def _send_with_id(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                raw_headers: list[tuple[bytes, bytes]] = list(
                    message.get("headers", []),
                )
                raw_headers.append(
                    (b"x-request-id", request_id.encode("latin-1")),
                )
                message["headers"] = raw_headers
            await send(message)

        try:
            await self.app(scope, receive, _send_with_id)
        finally:
            correlation_id_var.reset(token)
