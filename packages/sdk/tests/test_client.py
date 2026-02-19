"""Tests for artenic_ai_sdk.client — PlatformClient (mock httpx)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from artenic_ai_sdk.client import PlatformClient
from artenic_ai_sdk.exceptions import (
    AuthenticationError,
    PlatformError,
    RateLimitError,
    ServiceUnavailableError,
)
from artenic_ai_sdk.schemas import ModelConfig, ModelMetadata
from artenic_ai_sdk.types import ModelFramework


def _mock_response(
    status_code: int = 200,
    json_data: Any = None,
    headers: dict[str, str] | None = None,
    text: str = "",
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.headers = headers or {}
    resp.text = text
    return resp


class TestPlatformClient:
    @pytest.mark.asyncio
    async def test_not_initialized_raises(self) -> None:
        client = PlatformClient()
        with pytest.raises(PlatformError, match="not initialized"):
            await client._request("GET", "/test")

    @pytest.mark.asyncio
    async def test_auth_error(self) -> None:
        client = PlatformClient()
        client._client = AsyncMock()
        client._client.request = AsyncMock(return_value=_mock_response(status_code=401))

        with pytest.raises(AuthenticationError):
            await client._request("GET", "/test")

    @pytest.mark.asyncio
    async def test_rate_limit(self) -> None:
        client = PlatformClient()
        client._client = AsyncMock()
        client._client.request = AsyncMock(
            return_value=_mock_response(status_code=429, headers={"Retry-After": "1.0"})
        )

        with pytest.raises(RateLimitError):
            await client._request("GET", "/test", _max_retries=1)

    @pytest.mark.asyncio
    async def test_service_unavailable_retries(self) -> None:
        client = PlatformClient()
        client._client = AsyncMock()
        client._client.request = AsyncMock(return_value=_mock_response(status_code=503))

        with pytest.raises(ServiceUnavailableError):
            await client._request("GET", "/test", _max_retries=1)

    @pytest.mark.asyncio
    async def test_generic_error(self) -> None:
        client = PlatformClient()
        client._client = AsyncMock()
        client._client.request = AsyncMock(
            return_value=_mock_response(status_code=500, text="Internal Server Error")
        )

        with pytest.raises(PlatformError, match="500"):
            await client._request("GET", "/test")

    @pytest.mark.asyncio
    async def test_success(self) -> None:
        client = PlatformClient()
        client._client = AsyncMock()
        client._client.request = AsyncMock(return_value=_mock_response(json_data={"ok": True}))

        result = await client._request("GET", "/test")
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_register_model(self) -> None:
        client = PlatformClient()
        client._client = AsyncMock()
        client._client.request = AsyncMock(
            return_value=_mock_response(json_data={"model_id": "abc123"})
        )

        metadata = ModelMetadata(
            name="test", version="1.0", model_type="test", framework=ModelFramework.PYTORCH
        )
        model_id = await client.register_model(metadata)
        assert model_id == "abc123"

    @pytest.mark.asyncio
    async def test_list_models(self) -> None:
        client = PlatformClient()
        client._client = AsyncMock()
        client._client.request = AsyncMock(
            return_value=_mock_response(
                json_data=[
                    {"name": "m1", "version": "1.0", "model_type": "t", "framework": "pytorch"}
                ]
            )
        )

        models = await client.list_models()
        assert len(models) == 1
        assert models[0].name == "m1"

    @pytest.mark.asyncio
    async def test_dispatch_training(self) -> None:
        client = PlatformClient()
        client._client = AsyncMock()
        client._client.request = AsyncMock(
            return_value=_mock_response(json_data={"job_id": "job-001"})
        )

        job_id = await client.dispatch_training(
            service="my-service",
            model="lgbm",
            provider="ovh",
            config=ModelConfig(),
        )
        assert job_id == "job-001"

    @pytest.mark.asyncio
    async def test_predict(self) -> None:
        client = PlatformClient()
        client._client = AsyncMock()
        client._client.request = AsyncMock(
            return_value=_mock_response(
                json_data={
                    "confidence": 0.9,
                    "model_id": "m1",
                    "model_version": "1.0",
                    "inference_time_ms": 5.0,
                }
            )
        )

        pred = await client.predict("my-service", {"features": {"a": 1.0}})
        assert pred.confidence == 0.9

    @pytest.mark.asyncio
    async def test_predict_batch(self) -> None:
        client = PlatformClient()
        client._client = AsyncMock()
        client._client.request = AsyncMock(
            return_value=_mock_response(
                json_data=[
                    {
                        "confidence": 0.9,
                        "model_id": "m1",
                        "model_version": "1.0",
                        "inference_time_ms": 5.0,
                    },
                    {
                        "confidence": 0.8,
                        "model_id": "m1",
                        "model_version": "1.0",
                        "inference_time_ms": 6.0,
                    },
                ]
            )
        )

        results = await client.predict_batch("my-service", [{"a": 1.0}, {"a": 2.0}])
        assert len(results) == 2
        assert results[0].confidence == 0.9
        assert results[1].confidence == 0.8

    @pytest.mark.asyncio
    async def test_connection_error_retries(self) -> None:
        client = PlatformClient()
        client._client = AsyncMock()
        client._client.request = AsyncMock(side_effect=ConnectionError("refused"))

        with pytest.raises(ServiceUnavailableError):
            await client._request("GET", "/test", _max_retries=1, _backoff_base=0.01)

    @pytest.mark.asyncio
    async def test_get_model(self) -> None:
        client = PlatformClient()
        client._client = AsyncMock()
        client._client.request = AsyncMock(
            return_value=_mock_response(
                json_data={
                    "name": "m1",
                    "version": "1.0",
                    "model_type": "t",
                    "framework": "pytorch",
                }
            )
        )
        model = await client.get_model("m1")
        assert model.name == "m1"

    @pytest.mark.asyncio
    async def test_promote_model(self) -> None:
        client = PlatformClient()
        client._client = AsyncMock()
        client._client.request = AsyncMock(return_value=_mock_response(json_data={}))
        await client.promote_model("m1", "2.0")
        client._client.request.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_training_status(self) -> None:
        client = PlatformClient()
        client._client = AsyncMock()
        client._client.request = AsyncMock(
            return_value=_mock_response(json_data={"status": "running", "progress": 0.5})
        )
        result = await client.get_training_status("job-001")
        assert result["status"] == "running"

    @pytest.mark.asyncio
    async def test_health_check(self) -> None:
        client = PlatformClient()
        client._client = AsyncMock()
        client._client.request = AsyncMock(
            return_value=_mock_response(json_data={"status": "healthy"})
        )
        result = await client.health_check()
        assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_get_metrics(self) -> None:
        client = PlatformClient()
        client._client = AsyncMock()
        client._client.request = AsyncMock(return_value=_mock_response(json_data={"requests": 100}))
        result = await client.get_metrics()
        assert result["requests"] == 100

    @pytest.mark.asyncio
    async def test_aenter_with_api_key(self) -> None:
        """Test __aenter__ with API key creates httpx client with auth header."""
        mock_httpx = MagicMock()
        mock_async_client = AsyncMock()
        mock_httpx.AsyncClient.return_value = mock_async_client
        mock_httpx.Timeout.return_value = MagicMock()

        client = PlatformClient(api_key="test-key-123")
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = await client.__aenter__()
        assert result is client
        # Check that Authorization header was included
        call_kwargs = mock_httpx.AsyncClient.call_args
        headers = call_kwargs[1]["headers"]
        assert headers["Authorization"] == "Bearer test-key-123"

    @pytest.mark.asyncio
    async def test_aenter_without_api_key(self) -> None:
        """Test __aenter__ without API key creates httpx client without auth."""
        mock_httpx = MagicMock()
        mock_async_client = AsyncMock()
        mock_httpx.AsyncClient.return_value = mock_async_client
        mock_httpx.Timeout.return_value = MagicMock()

        client = PlatformClient()
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = await client.__aenter__()
        assert result is client
        call_kwargs = mock_httpx.AsyncClient.call_args
        headers = call_kwargs[1]["headers"]
        assert "Authorization" not in headers

    @pytest.mark.asyncio
    async def test_aexit_closes_client(self) -> None:
        """Test __aexit__ closes the httpx client."""
        client = PlatformClient()
        mock_client = AsyncMock()
        client._client = mock_client
        await client.__aexit__(None, None, None)
        mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_aexit_no_client(self) -> None:
        """Test __aexit__ when client is None (no-op)."""
        client = PlatformClient()
        client._client = None
        await client.__aexit__(None, None, None)  # Should not raise

    @pytest.mark.asyncio
    async def test_connection_error_retries_multiple(self) -> None:
        """Test retry with backoff on connection error (multi-attempt)."""
        client = PlatformClient()
        client._client = AsyncMock()
        # Fail twice then succeed
        client._client.request = AsyncMock(
            side_effect=[
                ConnectionError("refused"),
                ConnectionError("refused"),
                _mock_response(json_data={"ok": True}),
            ]
        )
        result = await client._request("GET", "/test", _max_retries=3, _backoff_base=0.01)
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_rate_limit_retries(self) -> None:
        """Test 429 retry with Retry-After header."""
        client = PlatformClient()
        client._client = AsyncMock()
        client._client.request = AsyncMock(
            side_effect=[
                _mock_response(status_code=429, headers={"Retry-After": "0.01"}),
                _mock_response(json_data={"ok": True}),
            ]
        )
        result = await client._request("GET", "/test", _max_retries=2, _backoff_base=0.01)
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_service_unavailable_retries_then_succeeds(self) -> None:
        """Test 503 retry with backoff then success."""
        client = PlatformClient()
        client._client = AsyncMock()
        client._client.request = AsyncMock(
            side_effect=[
                _mock_response(status_code=503),
                _mock_response(json_data={"ok": True}),
            ]
        )
        result = await client._request("GET", "/test", _max_retries=2, _backoff_base=0.01)
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_request_fallthrough_raises(self) -> None:
        """Test that exhausted retries with no explicit error falls through."""
        client = PlatformClient()
        client._client = AsyncMock()
        # Return 503 every time
        client._client.request = AsyncMock(return_value=_mock_response(status_code=503))
        with pytest.raises(ServiceUnavailableError):
            await client._request("GET", "/test", _max_retries=2, _backoff_base=0.01)

    @pytest.mark.asyncio
    async def test_max_retries_zero_raises(self) -> None:
        """_max_retries < 1 raises PlatformError."""
        client = PlatformClient()
        client._client = AsyncMock()
        with pytest.raises(PlatformError, match="_max_retries must be >= 1"):
            await client._request("GET", "/test", _max_retries=0)


class TestPlatformClientImportError:
    @pytest.mark.asyncio
    async def test_httpx_not_installed(self) -> None:
        with patch.dict("sys.modules", {"httpx": None}):
            client = PlatformClient()
            with pytest.raises(PlatformError, match="httpx not installed"):
                async with client:
                    pass
