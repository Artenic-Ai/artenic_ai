"""Vast.ai connector — GPU compute marketplace.

Relies on ``httpx`` (MIT licence, already a platform dependency).
Vast.ai is compute-only (no storage).  All data is fetched from the
Vast.ai API using the user's own API key.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from artenic_ai_platform.providers_hub.connectors.base import (
    ConnectorContext,
    ProviderConnector,
)
from artenic_ai_platform.providers_hub.schemas import (
    ComputeInstance,
    ConnectionTestResult,
    ProviderRegion,
    StorageOption,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import — httpx is an optional dependency
# ---------------------------------------------------------------------------
try:
    import httpx

    _HAS_HTTPX = True
except ImportError:  # pragma: no cover
    _HAS_HTTPX = False  # pragma: no cover
    httpx = None  # type: ignore[assignment]  # pragma: no cover


def _require_httpx() -> None:
    if not _HAS_HTTPX:
        msg = (
            "The 'httpx' package is required for the Vast.ai connector.  "
            "Install it with:  pip install httpx"
        )
        raise ImportError(msg)


_BASE_URL = "https://console.vast.ai/api/v0"


class VastaiConnector(ProviderConnector):
    """Connector for Vast.ai GPU marketplace."""

    def __init__(self, provider_id: str = "vastai") -> None:
        self._provider_id = provider_id

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _headers(self, ctx: ConnectorContext) -> dict[str, str]:
        return {"Authorization": f"Bearer {ctx.credentials.get('api_key', '')}"}

    async def _get(self, url: str, headers: dict[str, str]) -> Any:
        _require_httpx()

        def _do() -> Any:
            resp = httpx.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json()

        return await asyncio.to_thread(_do)

    # ------------------------------------------------------------------
    # test_connection
    # ------------------------------------------------------------------

    async def test_connection(self, ctx: ConnectorContext) -> ConnectionTestResult:
        t0 = time.monotonic()
        try:
            url = f"{_BASE_URL}/users/current/"
            await self._get(url, self._headers(ctx))
            elapsed_ms = (time.monotonic() - t0) * 1000
            return ConnectionTestResult(
                success=True,
                message="Connected to Vast.ai API",
                latency_ms=round(elapsed_ms, 1),
            )
        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.warning("[vastai] Connection test failed: %s", exc)
            return ConnectionTestResult(
                success=False,
                message=str(exc),
                latency_ms=round(elapsed_ms, 1),
            )

    # ------------------------------------------------------------------
    # list_storage_options (compute-only — always empty)
    # ------------------------------------------------------------------

    async def list_storage_options(self, ctx: ConnectorContext) -> list[StorageOption]:
        return []

    # ------------------------------------------------------------------
    # list_compute_instances
    # ------------------------------------------------------------------

    async def list_compute_instances(
        self,
        ctx: ConnectorContext,
        *,
        region: str | None = None,
        gpu_only: bool = False,
    ) -> list[ComputeInstance]:
        url = f"{_BASE_URL}/bundles/"
        try:
            data = await self._get(url, self._headers(ctx))
        except Exception:
            logger.warning("[vastai] Failed to list bundles", exc_info=True)
            return []

        instances: list[ComputeInstance] = []
        offers: list[dict[str, Any]] = data.get("offers", [])
        for offer in offers:
            gpu_name: str = offer.get("gpu_name", "GPU")
            num_gpus: int = offer.get("num_gpus", 1)
            cpu_cores: int = offer.get("cpu_cores_effective", 0) or offer.get("cpu_cores", 0)
            ram_gb: float = round(offer.get("cpu_ram", 0) / 1024, 1) if offer.get("cpu_ram") else 0
            disk_gb: float = float(offer.get("disk_space", 0) or 0)

            inst = ComputeInstance(
                provider_id=self._provider_id,
                name=f"{gpu_name} x{num_gpus}",
                vcpus=cpu_cores,
                memory_gb=ram_gb,
                disk_gb=disk_gb,
                gpu_type=gpu_name,
                gpu_count=num_gpus,
                region=region or offer.get("geolocation", ""),
                available=bool(offer.get("rentable", True)),
            )
            instances.append(inst)

        if gpu_only:
            instances = [i for i in instances if i.gpu_count > 0]

        return instances

    # ------------------------------------------------------------------
    # list_regions (marketplace — no fixed regions)
    # ------------------------------------------------------------------

    async def list_regions(self, ctx: ConnectorContext) -> list[ProviderRegion]:
        return []
