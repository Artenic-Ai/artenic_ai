"""Centralised error handling for the Artenic AI platform.

Provides:

* :class:`CatchAllErrorMiddleware` — a pure ASGI middleware that
  catches unhandled exceptions and returns a generic 500 JSON body.
* :func:`register_error_handlers` — wires up FastAPI exception handlers
  for SDK exceptions, Starlette ``HTTPException``, validation errors,
  and a generic fallback.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from artenic_ai_platform.middleware.correlation import get_request_id
from artenic_ai_sdk.exceptions import (
    ArtenicAIError,
    ArtenicTimeoutError,
    AuthenticationError,
    CircuitBreakerOpenError,
    ConfigValidationError,
    ModelInferenceError,
    ModelLoadError,
    ModelNotFoundError,
    ProviderError,
    ProviderQuotaError,
    RateLimitError,
    ServiceUnavailableError,
)

if TYPE_CHECKING:
    from fastapi import FastAPI
    from starlette.requests import Request

logger = logging.getLogger(__name__)

_Scope = dict[str, Any]
_Receive = Any
_Send = Any

# -- status code mapping --------------------------------------------------

_STATUS_MAP: dict[type[ArtenicAIError], int] = {
    ModelNotFoundError: 404,
    ConfigValidationError: 422,
    AuthenticationError: 401,
    RateLimitError: 429,
    ServiceUnavailableError: 503,
    CircuitBreakerOpenError: 503,
    ArtenicTimeoutError: 504,
    ModelLoadError: 503,
    ModelInferenceError: 500,
    ProviderQuotaError: 429,
    ProviderError: 502,
}

# -- helpers ---------------------------------------------------------------

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _error_code_from_class(cls: type[Exception]) -> str:
    """Convert e.g. ``ModelNotFoundError`` to ``MODEL_NOT_FOUND``.

    The trailing ``Error`` suffix is stripped before conversion.
    """
    name = cls.__name__
    if name.endswith("Error"):
        name = name[: -len("Error")]
    return _CAMEL_RE.sub("_", name).upper()


def _error_body(
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "request_id": get_request_id(),
            "details": details or {},
        },
    }


# ======================================================================
# Pure ASGI catch-all middleware
# ======================================================================


class CatchAllErrorMiddleware:
    """Wraps the ASGI application and catches any unhandled exception.

    Returns a minimal ``500 Internal Server Error`` JSON response so
    the client always receives a well-formed error payload.
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

        try:
            await self.app(scope, receive, send)
        except Exception:
            logger.exception("Unhandled exception in ASGI application")
            body = json.dumps(
                _error_body(
                    "INTERNAL_SERVER_ERROR",
                    "An unexpected error occurred.",
                ),
            ).encode()

            await send(
                {
                    "type": "http.response.start",
                    "status": 500,
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


# ======================================================================
# FastAPI exception handlers
# ======================================================================


async def handle_artenic_error(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Map any :class:`ArtenicAIError` subclass to a JSON response.

    Walks the exception's MRO to find the most specific HTTP status
    code registered in :data:`_STATUS_MAP`.
    """
    assert isinstance(exc, ArtenicAIError)
    status = 500
    for cls in type(exc).__mro__:
        if cls in _STATUS_MAP:
            status = _STATUS_MAP[cls]
            break

    code = _error_code_from_class(type(exc))
    return JSONResponse(
        status_code=status,
        content=_error_body(code, str(exc), exc.details),
    )


async def handle_http_exception(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle Starlette :class:`HTTPException`."""
    assert isinstance(exc, HTTPException)
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_body(
            "HTTP_ERROR",
            str(exc.detail),
        ),
    )


async def handle_validation_error(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle Pydantic / FastAPI request validation errors."""
    assert isinstance(exc, RequestValidationError)
    return JSONResponse(
        status_code=422,
        content=_error_body(
            "VALIDATION_ERROR",
            "Request validation failed.",
            {"errors": exc.errors()},
        ),
    )


async def handle_generic_error(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Last-resort handler for completely unexpected exceptions."""
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content=_error_body(
            "INTERNAL_SERVER_ERROR",
            "An unexpected error occurred.",
        ),
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register all exception handlers on the given FastAPI app."""
    app.add_exception_handler(
        ArtenicAIError,
        handle_artenic_error,
    )
    app.add_exception_handler(
        HTTPException,
        handle_http_exception,
    )
    app.add_exception_handler(
        RequestValidationError,
        handle_validation_error,
    )
    app.add_exception_handler(
        Exception,
        handle_generic_error,
    )
