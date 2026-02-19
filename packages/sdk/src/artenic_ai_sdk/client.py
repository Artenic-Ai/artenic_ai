"""Async HTTP client for communicating with the artenic_ai platform.

Provides typed methods for model registry, training dispatch,
inference, and health monitoring through the gateway (port 9000).
"""

from __future__ import annotations

import asyncio
from typing import Any

from artenic_ai_sdk.exceptions import (
    AuthenticationError,
    PlatformError,
    RateLimitError,
    ServiceUnavailableError,
)
from artenic_ai_sdk.schemas import BasePrediction, ModelConfig, ModelMetadata


class PlatformClient:
    """Async client for the artenic_ai platform gateway.

    Uses httpx internally for connection pooling, retries, and timeouts.

    Example::

        async with PlatformClient(base_url="http://localhost:9000") as client:
            models = await client.list_models()
            result = await client.predict("my-service", {"features": ...})
    """

    def __init__(
        self,
        base_url: str = "http://localhost:9000",
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._client: Any = None

    async def __aenter__(self) -> PlatformClient:
        try:
            import httpx

            headers: dict[str, str] = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=httpx.Timeout(self._timeout),
            )
        except ImportError as e:
            raise PlatformError(
                "httpx not installed. Install with: pip install artenic-ai-sdk[client]"
            ) from e
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()

    # =========================================================================
    # Registry
    # =========================================================================

    async def register_model(self, metadata: ModelMetadata) -> str:
        """Register a model with the platform.

        Returns:
            The registered model ID.
        """
        response = await self._post(
            "/api/v1/models",
            json=metadata.model_dump(mode="json"),
        )
        return response["model_id"]  # type: ignore[no-any-return]

    async def get_model(self, model_id: str) -> ModelMetadata:
        """Get model metadata by ID."""
        data = await self._get(f"/api/v1/models/{model_id}")
        return ModelMetadata.model_validate(data)

    async def list_models(self) -> list[ModelMetadata]:
        """List all registered models."""
        data = await self._get("/api/v1/models")
        return [ModelMetadata.model_validate(m) for m in data]

    async def promote_model(self, model_id: str, version: str) -> None:
        """Promote a model version to production."""
        await self._post(
            f"/api/v1/models/{model_id}/promote",
            json={"version": version},
        )

    # =========================================================================
    # Training
    # =========================================================================

    async def dispatch_training(
        self,
        service: str,
        model: str,
        provider: str,
        config: ModelConfig,
    ) -> str:
        """Dispatch a training job to a remote provider.

        Args:
            service: Target service (e.g. 'my-service').
            model: Model name within the service.
            provider: Training provider (e.g. 'ovh', 'gcp').
            config: Training configuration.

        Returns:
            Job ID for tracking.
        """
        response = await self._post(
            "/api/v1/training/dispatch",
            json={
                "service": service,
                "model": model,
                "provider": provider,
                "config": config.model_dump(mode="json"),
            },
        )
        return response["job_id"]  # type: ignore[no-any-return]

    async def get_training_status(self, job_id: str) -> dict[str, Any]:
        """Check training job status.

        Returns:
            Job status dict with progress, metrics, etc.
        """
        return await self._get(f"/api/v1/training/{job_id}")  # type: ignore[no-any-return]

    # =========================================================================
    # Inference
    # =========================================================================

    async def predict(
        self,
        service: str,
        data: dict[str, Any],
    ) -> BasePrediction:
        """Send an inference request to a service.

        Args:
            service: Target service (e.g. 'my-service').
            data: Input data for the model.

        Returns:
            Prediction result.
        """
        response = await self._post(
            f"/api/v1/services/{service}/predict",
            json=data,
        )
        return BasePrediction.model_validate(response)

    async def predict_batch(
        self,
        service: str,
        batch: list[dict[str, Any]],
    ) -> list[BasePrediction]:
        """Send a batch inference request to a service.

        Args:
            service: Target service (e.g. 'my-service').
            batch: List of input data dicts.

        Returns:
            List of prediction results.
        """
        response = await self._post(
            f"/api/v1/services/{service}/predict_batch",
            json={"batch": batch},
        )
        return [BasePrediction.model_validate(r) for r in response]

    # =========================================================================
    # Health
    # =========================================================================

    async def health_check(self) -> dict[str, Any]:
        """Check platform health."""
        return await self._get("/health")  # type: ignore[no-any-return]

    async def get_metrics(self) -> dict[str, Any]:
        """Get platform-wide metrics."""
        return await self._get("/api/v1/metrics")  # type: ignore[no-any-return]

    # =========================================================================
    # HTTP helpers
    # =========================================================================

    async def _get(self, path: str) -> Any:
        return await self._request("GET", path)

    async def _post(self, path: str, json: Any = None) -> Any:
        return await self._request("POST", path, json=json)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        _max_retries: int = 3,
        _backoff_base: float = 1.0,
        **kwargs: Any,
    ) -> Any:
        if not self._client:
            raise PlatformError("Client not initialized. Use 'async with' context.")
        if _max_retries < 1:
            raise PlatformError("_max_retries must be >= 1")

        last_error: Exception | None = None

        for attempt in range(_max_retries):
            try:
                response = await self._client.request(method, path, **kwargs)
            except Exception as e:
                last_error = ServiceUnavailableError(f"Request failed: {method} {path}: {e}")
                last_error.__cause__ = e
                if attempt < _max_retries - 1:
                    await asyncio.sleep(_backoff_base * (2**attempt))
                    continue
                raise last_error from e

            if response.status_code == 401:
                raise AuthenticationError("Invalid or missing API key")

            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", _backoff_base))
                if attempt < _max_retries - 1:
                    await asyncio.sleep(retry_after)
                    continue
                raise RateLimitError(
                    f"Rate limited on {method} {path}",
                    retry_after=retry_after,
                )

            if response.status_code == 503:
                last_error = ServiceUnavailableError(f"Service unavailable: {path}")
                if attempt < _max_retries - 1:
                    await asyncio.sleep(_backoff_base * (2**attempt))
                    continue
                raise last_error

            if response.status_code >= 400:
                raise PlatformError(f"Platform error {response.status_code}: {response.text}")

            return response.json()

        # Unreachable: every loop iteration either raises or returns.
        raise AssertionError("Unreachable")  # pragma: no cover
