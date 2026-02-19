"""Thin async HTTP client for the Artenic AI platform API."""

from __future__ import annotations

from typing import Any

import httpx

from artenic_ai_sdk.exceptions import (
    AuthenticationError,
    PlatformError,
    RateLimitError,
    ServiceUnavailableError,
)


class ApiClient:
    """Async HTTP client wrapping httpx for all platform endpoints.

    Unlike the SDK's ``PlatformClient`` (which covers 10 endpoints), this
    client exposes generic ``get``/``post``/``put``/``delete`` methods so
    the CLI can call all 37 platform endpoints without per-method wrappers.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:9000",
        api_key: str = "",
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> ApiClient:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(self._timeout),
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()

    async def get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        """HTTP GET."""
        return await self._request("GET", path, params=params)

    async def post(self, path: str, *, json: Any = None) -> Any:
        """HTTP POST."""
        return await self._request("POST", path, json=json)

    async def put(self, path: str, *, json: Any = None) -> Any:
        """HTTP PUT."""
        return await self._request("PUT", path, json=json)

    async def patch(self, path: str, *, json: Any = None) -> Any:
        """HTTP PATCH."""
        return await self._request("PATCH", path, json=json)

    async def delete(self, path: str) -> Any:
        """HTTP DELETE."""
        return await self._request("DELETE", path)

    async def upload_file(self, path: str, filename: str, data: bytes, mime_type: str) -> Any:
        """Upload a file via multipart/form-data."""
        if not self._client:
            raise PlatformError("Client not initialized. Use 'async with' context.")
        response = await self._client.post(path, files={"file": (filename, data, mime_type)})
        if response.status_code >= 400:
            raise PlatformError(f"Platform error {response.status_code}: {response.text}")
        return response.json()

    async def download_bytes(self, path: str) -> bytes:
        """Download raw bytes from an endpoint."""
        if not self._client:
            raise PlatformError("Client not initialized. Use 'async with' context.")
        response = await self._client.get(path)
        if response.status_code >= 400:
            raise PlatformError(f"Platform error {response.status_code}: {response.text}")
        return response.content

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Send request and map HTTP errors to SDK exceptions."""
        if not self._client:
            raise PlatformError("Client not initialized. Use 'async with' context.")

        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            raise ServiceUnavailableError(f"Cannot connect to {self._base_url}") from exc
        except httpx.TimeoutException as exc:
            raise ServiceUnavailableError(f"Request timed out after {self._timeout}s") from exc

        if response.status_code == 401:
            raise AuthenticationError("Invalid or missing API key")

        if response.status_code == 429:
            retry_after = float(response.headers.get("Retry-After", "1"))
            raise RateLimitError(
                f"Rate limited on {method} {path}",
                retry_after=retry_after,
            )

        if response.status_code == 204:
            return None

        if response.status_code >= 400:
            raise PlatformError(f"Platform error {response.status_code}: {response.text}")

        try:
            return response.json()
        except Exception as exc:
            raise PlatformError(f"Invalid JSON in response from {method} {path}: {exc}") from exc
