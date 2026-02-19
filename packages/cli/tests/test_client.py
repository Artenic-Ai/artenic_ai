"""Tests for _client module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from artenic_ai_cli._async import run_async
from artenic_ai_cli._client import ApiClient
from artenic_ai_sdk.exceptions import (
    AuthenticationError,
    PlatformError,
    RateLimitError,
    ServiceUnavailableError,
)


def _mock_response(
    status_code: int = 200,
    json_data: object = None,
    text: str = "",
    headers: dict[str, str] | None = None,
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data if json_data is not None else {}
    resp.text = text
    resp.headers = headers or {}
    return resp


class TestApiClient:
    def test_get(self) -> None:
        async def _test() -> object:
            api = ApiClient("http://test:9000")
            async with api:
                api._client = AsyncMock()  # type: ignore[assignment]
                api._client.request = AsyncMock(return_value=_mock_response(json_data={"ok": True}))
                return await api.get("/health")

        assert run_async(_test()) == {"ok": True}

    def test_post(self) -> None:
        async def _test() -> object:
            api = ApiClient("http://test:9000")
            async with api:
                api._client = AsyncMock()  # type: ignore[assignment]
                api._client.request = AsyncMock(return_value=_mock_response(json_data={"id": "1"}))
                return await api.post("/api/v1/models", json={"name": "m"})

        assert run_async(_test()) == {"id": "1"}

    def test_put(self) -> None:
        async def _test() -> object:
            api = ApiClient("http://test:9000")
            async with api:
                api._client = AsyncMock()  # type: ignore[assignment]
                api._client.request = AsyncMock(
                    return_value=_mock_response(json_data={"updated": True})
                )
                return await api.put("/api/v1/models/1", json={"name": "n"})

        assert run_async(_test()) == {"updated": True}

    def test_delete(self) -> None:
        async def _test() -> object:
            api = ApiClient("http://test:9000")
            async with api:
                api._client = AsyncMock()  # type: ignore[assignment]
                api._client.request = AsyncMock(return_value=_mock_response(status_code=204))
                return await api.delete("/api/v1/models/1")

        assert run_async(_test()) is None

    def test_auth_header(self) -> None:
        async def _test() -> None:
            api = ApiClient("http://test:9000", api_key="secret")
            async with api:
                assert api._client is not None
                assert api._client.headers["Authorization"] == "Bearer secret"  # type: ignore[union-attr]

        run_async(_test())

    def test_no_auth_header_when_empty(self) -> None:
        async def _test() -> None:
            api = ApiClient("http://test:9000", api_key="")
            async with api:
                assert "Authorization" not in api._client.headers  # type: ignore[union-attr]

        run_async(_test())

    def test_401_raises_auth_error(self) -> None:
        async def _test() -> None:
            api = ApiClient("http://test:9000")
            async with api:
                api._client = AsyncMock()  # type: ignore[assignment]
                api._client.request = AsyncMock(return_value=_mock_response(status_code=401))
                await api.get("/x")

        with pytest.raises(AuthenticationError):
            run_async(_test())

    def test_429_raises_rate_limit(self) -> None:
        async def _test() -> None:
            api = ApiClient("http://test:9000")
            async with api:
                api._client = AsyncMock()  # type: ignore[assignment]
                api._client.request = AsyncMock(
                    return_value=_mock_response(status_code=429, headers={"Retry-After": "5"})
                )
                await api.get("/x")

        with pytest.raises(RateLimitError) as exc_info:
            run_async(_test())
        assert exc_info.value.retry_after == 5.0

    def test_4xx_raises_platform_error(self) -> None:
        async def _test() -> None:
            api = ApiClient("http://test:9000")
            async with api:
                api._client = AsyncMock()  # type: ignore[assignment]
                api._client.request = AsyncMock(
                    return_value=_mock_response(status_code=422, text="Validation error")
                )
                await api.get("/x")

        with pytest.raises(PlatformError, match="422"):
            run_async(_test())

    def test_5xx_raises_platform_error(self) -> None:
        async def _test() -> None:
            api = ApiClient("http://test:9000")
            async with api:
                api._client = AsyncMock()  # type: ignore[assignment]
                api._client.request = AsyncMock(
                    return_value=_mock_response(status_code=500, text="Internal")
                )
                await api.get("/x")

        with pytest.raises(PlatformError, match="500"):
            run_async(_test())

    def test_connect_error(self) -> None:
        async def _test() -> None:
            api = ApiClient("http://test:9000")
            async with api:
                api._client = AsyncMock()  # type: ignore[assignment]
                api._client.request = AsyncMock(side_effect=httpx.ConnectError("refused"))
                await api.get("/x")

        with pytest.raises(ServiceUnavailableError, match="Cannot connect"):
            run_async(_test())

    def test_timeout_error(self) -> None:
        async def _test() -> None:
            api = ApiClient("http://test:9000", timeout=5.0)
            async with api:
                api._client = AsyncMock()  # type: ignore[assignment]
                api._client.request = AsyncMock(side_effect=httpx.ReadTimeout("timeout"))
                await api.get("/x")

        with pytest.raises(ServiceUnavailableError, match="timed out"):
            run_async(_test())

    def test_not_initialized(self) -> None:
        async def _test() -> None:
            api = ApiClient("http://test:9000")
            await api.get("/x")

        with pytest.raises(PlatformError, match="not initialized"):
            run_async(_test())

    def test_aexit_no_client(self) -> None:
        async def _test() -> None:
            api = ApiClient("http://test:9000")
            await api.__aexit__(None, None, None)

        run_async(_test())

    def test_trailing_slash_stripped(self) -> None:
        api = ApiClient("http://test:9000/")
        assert api._base_url == "http://test:9000"

    def test_204_returns_none(self) -> None:
        async def _test() -> object:
            api = ApiClient("http://test:9000")
            async with api:
                api._client = AsyncMock()  # type: ignore[assignment]
                api._client.request = AsyncMock(return_value=_mock_response(status_code=204))
                return await api.post("/api/v1/models/1/promote", json={})

        assert run_async(_test()) is None

    def test_429_default_retry_after(self) -> None:
        async def _test() -> None:
            api = ApiClient("http://test:9000")
            async with api:
                api._client = AsyncMock()  # type: ignore[assignment]
                api._client.request = AsyncMock(return_value=_mock_response(status_code=429))
                await api.get("/x")

        with pytest.raises(RateLimitError) as exc_info:
            run_async(_test())
        assert exc_info.value.retry_after == 1.0

    def test_invalid_json_response(self) -> None:
        async def _test() -> object:
            api = ApiClient("http://test:9000")
            async with api:
                api._client = AsyncMock()  # type: ignore[assignment]
                resp = _mock_response(status_code=200)
                resp.json.side_effect = ValueError("No JSON")
                api._client.request = AsyncMock(return_value=resp)
                return await api.get("/x")

        with pytest.raises(PlatformError, match="Invalid JSON"):
            run_async(_test())
