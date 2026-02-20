"""HTTP request-duration metrics middleware (pure ASGI).

When OpenTelemetry is available a histogram named
``http.server.request.duration`` is recorded for every HTTP request.
If the library is not installed the middleware is a transparent
pass-through.
"""

from __future__ import annotations

import time
from typing import Any

_Scope = dict[str, Any]
_Receive = Any
_Send = Any

# -- optional OpenTelemetry import ------------------------------------

try:
    from opentelemetry import metrics as otel_metrics

    _HAS_OTEL = True
except ImportError:  # pragma: no cover
    _HAS_OTEL = False
    otel_metrics = None  # type: ignore[assignment]


class MetricsMiddleware:
    """Pure ASGI middleware that records HTTP request duration.

    If ``opentelemetry`` is not installed the middleware simply
    forwards every request without any overhead.

    Parameters
    ----------
    app:
        The inner ASGI application.
    """

    def __init__(self, app: Any) -> None:
        self.app = app
        self._histogram: Any | None = None

        if _HAS_OTEL and otel_metrics is not None:
            meter = otel_metrics.get_meter(__name__)
            self._histogram = meter.create_histogram(
                name="http.server.request.duration",
                unit="ms",
                description=("Duration of inbound HTTP requests in milliseconds."),
            )

    async def __call__(
        self,
        scope: _Scope,
        receive: _Receive,
        send: _Send,
    ) -> None:
        if scope["type"] != "http" or self._histogram is None:
            await self.app(scope, receive, send)
            return

        method: str = scope.get("method", "GET")
        route: str = scope.get("path", "/")
        status_code: int = 200

        async def _capture_status(message: dict[str, Any]) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 200)
            await send(message)

        start = time.perf_counter()
        try:
            await self.app(scope, receive, _capture_status)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1_000
            self._histogram.record(
                elapsed_ms,
                attributes={
                    "http.method": method,
                    "http.route": route,
                    "http.status_code": status_code,
                },
            )
