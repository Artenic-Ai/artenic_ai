"""Vast.ai public catalog fetcher — console.vast.ai/api/v0/bundles."""

from __future__ import annotations

import asyncio
import logging

from artenic_ai_platform_providers.hub.public_catalog.base import CatalogFetcher
from artenic_ai_platform_providers.hub.schemas import (
    CatalogComputeFlavor,
    CatalogStorageTier,
)

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_BUNDLES_URL = "https://console.vast.ai/api/v0/bundles/"


class VastaiCatalogFetcher(CatalogFetcher):
    """Fetch public GPU marketplace offers from Vast.ai (no auth)."""

    async def fetch_compute_catalog(self) -> list[CatalogComputeFlavor]:
        data = await self._get()
        offers: list[dict[str, object]] = data.get("offers", [])  # type: ignore[assignment]
        result: list[CatalogComputeFlavor] = []
        for o in offers:
            gpu_name = str(o.get("gpu_name", "GPU"))
            num_gpus = int(o.get("num_gpus", 1) or 1)  # type: ignore[call-overload]
            vcpus = int(  # type: ignore[call-overload]
                o.get("cpu_cores_effective", 0) or o.get("cpu_cores", 0) or 0,
            )
            ram_mb = float(o.get("cpu_ram", 0) or 0)  # type: ignore[arg-type]
            disk = float(o.get("disk_space", 0) or 0)  # type: ignore[arg-type]
            dph = o.get("dph_total")
            price: float | None = float(str(dph)) if dph is not None else None
            geo = str(o.get("geolocation", ""))
            result.append(
                CatalogComputeFlavor(
                    provider_id="vastai",
                    name=f"{gpu_name} x{num_gpus}",
                    vcpus=vcpus,
                    memory_gb=round(ram_mb / 1024, 1) if ram_mb else 0.0,
                    disk_gb=disk,
                    gpu_type=gpu_name,
                    gpu_count=num_gpus,
                    price_per_hour=price,
                    currency="USD",
                    region=geo,
                    category="gpu",
                )
            )
        return result

    async def fetch_storage_catalog(self) -> list[CatalogStorageTier]:
        return []  # Vast.ai is compute-only

    # ------------------------------------------------------------------

    async def _get(self) -> dict[str, object]:
        if httpx is None:
            msg = "httpx is required for Vast.ai catalog.  pip install httpx"
            raise ImportError(msg)
        resp = await asyncio.to_thread(httpx.get, _BUNDLES_URL, timeout=30.0)
        resp.raise_for_status()
        result: dict[str, object] = resp.json()
        return result
