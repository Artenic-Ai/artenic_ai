"""Tests for artenic_ai_platform.middleware.errors — 100% coverage."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from starlette.exceptions import HTTPException

from artenic_ai_platform.middleware.errors import (
    CatchAllErrorMiddleware,
    _error_body,
    _error_code_from_class,
    handle_artenic_error,
    handle_generic_error,
    handle_http_exception,
    handle_validation_error,
    register_error_handlers,
)
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

# ======================================================================
# _error_code_from_class
# ======================================================================


class TestErrorCodeFromClass:
    def test_strips_error_suffix(self) -> None:
        assert _error_code_from_class(ModelNotFoundError) == "MODEL_NOT_FOUND"

    def test_camel_to_snake(self) -> None:
        assert _error_code_from_class(ArtenicTimeoutError) == "ARTENIC_TIMEOUT"

    def test_no_error_suffix(self) -> None:
        class SomeException(Exception):  # noqa: N818
            pass

        assert _error_code_from_class(SomeException) == "SOME_EXCEPTION"

    def test_single_word_error(self) -> None:
        class CustomError(Exception):
            pass

        assert _error_code_from_class(CustomError) == "CUSTOM"

    def test_authentication_error(self) -> None:
        assert _error_code_from_class(AuthenticationError) == "AUTHENTICATION"

    def test_rate_limit_error(self) -> None:
        assert _error_code_from_class(RateLimitError) == "RATE_LIMIT"

    def test_provider_quota_error(self) -> None:
        assert _error_code_from_class(ProviderQuotaError) == "PROVIDER_QUOTA"

    def test_config_validation_error(self) -> None:
        assert _error_code_from_class(ConfigValidationError) == "CONFIG_VALIDATION"

    def test_service_unavailable_error(self) -> None:
        assert _error_code_from_class(ServiceUnavailableError) == "SERVICE_UNAVAILABLE"

    def test_circuit_breaker_open_error(self) -> None:
        assert _error_code_from_class(CircuitBreakerOpenError) == "CIRCUIT_BREAKER_OPEN"

    def test_model_load_error(self) -> None:
        assert _error_code_from_class(ModelLoadError) == "MODEL_LOAD"

    def test_model_inference_error(self) -> None:
        assert _error_code_from_class(ModelInferenceError) == "MODEL_INFERENCE"

    def test_provider_error(self) -> None:
        assert _error_code_from_class(ProviderError) == "PROVIDER"


# ======================================================================
# _error_body
# ======================================================================


class TestErrorBody:
    def test_basic_body(self) -> None:
        body = _error_body("TEST_CODE", "test message")
        assert body["error"]["code"] == "TEST_CODE"
        assert body["error"]["message"] == "test message"
        assert body["error"]["details"] == {}
        assert "request_id" in body["error"]

    def test_with_details(self) -> None:
        details = {"field": "name", "reason": "required"}
        body = _error_body("VALIDATION", "invalid", details)
        assert body["error"]["details"] == details

    def test_none_details_becomes_empty_dict(self) -> None:
        body = _error_body("CODE", "msg", None)
        assert body["error"]["details"] == {}


# ======================================================================
# CatchAllErrorMiddleware
# ======================================================================


class TestCatchAllErrorMiddleware:
    async def test_non_http_passes_through(self) -> None:
        inner = AsyncMock()
        mw = CatchAllErrorMiddleware(inner)
        scope: dict[str, Any] = {"type": "lifespan"}

        await mw(scope, AsyncMock(), AsyncMock())

        inner.assert_awaited_once()

    async def test_successful_request_passes_through(self) -> None:
        inner = AsyncMock()
        mw = CatchAllErrorMiddleware(inner)
        scope: dict[str, Any] = {"type": "http"}

        await mw(scope, AsyncMock(), AsyncMock())

        inner.assert_awaited_once()

    async def test_exception_returns_500(self) -> None:
        async def inner_app(scope: Any, receive: Any, send: Any) -> None:
            raise RuntimeError("unexpected")

        mw = CatchAllErrorMiddleware(inner_app)
        sent_messages: list[dict[str, Any]] = []

        async def mock_send(msg: dict[str, Any]) -> None:
            sent_messages.append(msg)

        scope: dict[str, Any] = {"type": "http"}

        await mw(scope, AsyncMock(), mock_send)

        assert sent_messages[0]["status"] == 500
        headers_dict = dict(sent_messages[0]["headers"])
        assert headers_dict[b"content-type"] == b"application/json"

        body = json.loads(sent_messages[1]["body"])
        assert body["error"]["code"] == "INTERNAL_SERVER_ERROR"
        assert body["error"]["message"] == "An unexpected error occurred."

    async def test_websocket_exception_returns_500(self) -> None:
        async def inner_app(scope: Any, receive: Any, send: Any) -> None:
            raise ValueError("ws error")

        mw = CatchAllErrorMiddleware(inner_app)
        sent_messages: list[dict[str, Any]] = []

        async def mock_send(msg: dict[str, Any]) -> None:
            sent_messages.append(msg)

        scope: dict[str, Any] = {"type": "websocket"}

        await mw(scope, AsyncMock(), mock_send)

        assert sent_messages[0]["status"] == 500


# ======================================================================
# handle_artenic_error
# ======================================================================


class TestHandleArtenicError:
    async def test_model_not_found_404(self) -> None:
        exc = ModelNotFoundError("not found")
        resp = await handle_artenic_error(MagicMock(), exc)
        assert resp.status_code == 404
        body = json.loads(resp.body)
        assert body["error"]["code"] == "MODEL_NOT_FOUND"

    async def test_config_validation_422(self) -> None:
        exc = ConfigValidationError("bad config")
        resp = await handle_artenic_error(MagicMock(), exc)
        assert resp.status_code == 422

    async def test_authentication_401(self) -> None:
        exc = AuthenticationError("bad auth")
        resp = await handle_artenic_error(MagicMock(), exc)
        assert resp.status_code == 401

    async def test_rate_limit_429(self) -> None:
        exc = RateLimitError("too fast")
        resp = await handle_artenic_error(MagicMock(), exc)
        assert resp.status_code == 429

    async def test_service_unavailable_503(self) -> None:
        exc = ServiceUnavailableError("down")
        resp = await handle_artenic_error(MagicMock(), exc)
        assert resp.status_code == 503

    async def test_circuit_breaker_503(self) -> None:
        exc = CircuitBreakerOpenError("open")
        resp = await handle_artenic_error(MagicMock(), exc)
        assert resp.status_code == 503

    async def test_timeout_504(self) -> None:
        exc = ArtenicTimeoutError("timeout")
        resp = await handle_artenic_error(MagicMock(), exc)
        assert resp.status_code == 504

    async def test_model_load_503(self) -> None:
        exc = ModelLoadError("load fail")
        resp = await handle_artenic_error(MagicMock(), exc)
        assert resp.status_code == 503

    async def test_model_inference_500(self) -> None:
        exc = ModelInferenceError("inference fail")
        resp = await handle_artenic_error(MagicMock(), exc)
        assert resp.status_code == 500

    async def test_provider_quota_429(self) -> None:
        exc = ProviderQuotaError("quota exceeded")
        resp = await handle_artenic_error(MagicMock(), exc)
        assert resp.status_code == 429

    async def test_provider_error_502(self) -> None:
        exc = ProviderError("provider fail")
        resp = await handle_artenic_error(MagicMock(), exc)
        assert resp.status_code == 502

    async def test_generic_artenic_error_500(self) -> None:
        exc = ArtenicAIError("generic")
        resp = await handle_artenic_error(MagicMock(), exc)
        assert resp.status_code == 500

    async def test_response_includes_details(self) -> None:
        exc = ModelNotFoundError("not found", details={"model_id": "abc"})
        resp = await handle_artenic_error(MagicMock(), exc)
        body = json.loads(resp.body)
        assert body["error"]["details"]["model_id"] == "abc"

    async def test_response_includes_message(self) -> None:
        exc = ModelNotFoundError("Model 'xyz' does not exist")
        resp = await handle_artenic_error(MagicMock(), exc)
        body = json.loads(resp.body)
        assert "xyz" in body["error"]["message"]


# ======================================================================
# handle_http_exception
# ======================================================================


class TestHandleHttpException:
    async def test_404(self) -> None:
        exc = HTTPException(status_code=404, detail="not found")
        resp = await handle_http_exception(MagicMock(), exc)
        assert resp.status_code == 404
        body = json.loads(resp.body)
        assert body["error"]["code"] == "HTTP_ERROR"
        assert body["error"]["message"] == "not found"

    async def test_403(self) -> None:
        exc = HTTPException(status_code=403, detail="forbidden")
        resp = await handle_http_exception(MagicMock(), exc)
        assert resp.status_code == 403


# ======================================================================
# handle_validation_error
# ======================================================================


class TestHandleValidationError:
    async def test_returns_422(self) -> None:
        from fastapi.exceptions import RequestValidationError

        exc = RequestValidationError(
            errors=[{"loc": ["body", "name"], "msg": "required", "type": "missing"}]
        )
        resp = await handle_validation_error(MagicMock(), exc)
        assert resp.status_code == 422
        body = json.loads(resp.body)
        assert body["error"]["code"] == "VALIDATION_ERROR"
        assert body["error"]["message"] == "Request validation failed."
        assert "errors" in body["error"]["details"]


# ======================================================================
# handle_generic_error
# ======================================================================


class TestHandleGenericError:
    async def test_returns_500(self) -> None:
        exc = RuntimeError("something broke")
        resp = await handle_generic_error(MagicMock(), exc)
        assert resp.status_code == 500
        body = json.loads(resp.body)
        assert body["error"]["code"] == "INTERNAL_SERVER_ERROR"


# ======================================================================
# register_error_handlers
# ======================================================================


class TestRegisterErrorHandlers:
    def test_registers_4_handlers(self) -> None:
        app = MagicMock()
        register_error_handlers(app)
        assert app.add_exception_handler.call_count == 4
