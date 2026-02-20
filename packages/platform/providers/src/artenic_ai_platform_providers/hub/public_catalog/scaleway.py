"""Scaleway public catalog fetcher — live API or static fallback."""

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

_PRODUCTS_URL = "https://api.scaleway.com/instance/v1/zones/fr-par-1/products/servers"

# Static fallback pricing data (from Scaleway public pricing page, Feb 2025).
_STATIC_COMPUTE: list[dict[str, object]] = [
    {"name": "DEV1-S", "vcpus": 2, "memory_gb": 2.0, "price": 0.01, "cat": "general"},
    {"name": "DEV1-M", "vcpus": 3, "memory_gb": 4.0, "price": 0.02, "cat": "general"},
    {"name": "DEV1-L", "vcpus": 4, "memory_gb": 8.0, "price": 0.04, "cat": "general"},
    {"name": "DEV1-XL", "vcpus": 4, "memory_gb": 12.0, "price": 0.06, "cat": "general"},
    {"name": "GP1-S", "vcpus": 8, "memory_gb": 32.0, "price": 0.11, "cat": "general"},
    {"name": "GP1-M", "vcpus": 16, "memory_gb": 64.0, "price": 0.22, "cat": "general"},
    {"name": "GP1-L", "vcpus": 32, "memory_gb": 128.0, "price": 0.44, "cat": "general"},
    {"name": "GP1-XL", "vcpus": 48, "memory_gb": 256.0, "price": 0.88, "cat": "general"},
    {
        "name": "RENDER-S",
        "vcpus": 10,
        "memory_gb": 45.0,
        "price": 1.0,
        "gpu": "P100",
        "gpus": 1,
        "cat": "gpu",
    },
    {
        "name": "GPU-3070-S",
        "vcpus": 8,
        "memory_gb": 16.0,
        "price": 0.65,
        "gpu": "RTX 3070",
        "gpus": 1,
        "cat": "gpu",
    },
]

_STATIC_STORAGE: list[dict[str, object]] = [
    {"name": "Object Storage — Standard", "price": 0.01, "type": "object_storage"},
    {"name": "Object Storage — Glacier", "price": 0.002, "type": "object_storage"},
]


class ScalewayCatalogFetcher(CatalogFetcher):
    """Fetch public pricing from Scaleway (live or static fallback)."""

    def __init__(self) -> None:
        self._is_live: bool | None = None

    def supports_live_catalog(self) -> bool:
        return self._is_live is True

    async def fetch_compute_catalog(self) -> list[CatalogComputeFlavor]:
        live = await self._try_live_compute()
        if live is not None:
            self._is_live = True
            return live
        self._is_live = False
        return self._static_compute()

    async def fetch_storage_catalog(self) -> list[CatalogStorageTier]:
        return self._static_storage()

    # ------------------------------------------------------------------
    # Live fetch
    # ------------------------------------------------------------------

    async def _try_live_compute(self) -> list[CatalogComputeFlavor] | None:
        """Attempt to fetch from Scaleway public products API."""
        if httpx is None:
            return None
        try:
            resp = await asyncio.to_thread(
                httpx.get,
                _PRODUCTS_URL,
                timeout=15.0,
            )
            if resp.status_code == 401:
                logger.info("[scaleway-catalog] API requires auth, using static fallback")
                return None
            resp.raise_for_status()
            data: dict[str, object] = resp.json()
            servers: dict[str, dict[str, object]] = data.get("servers", {})  # type: ignore[assignment]
            result: list[CatalogComputeFlavor] = []
            for name, info in servers.items():
                if not isinstance(info, dict):
                    continue
                ncpus = int(info.get("ncpus", 0) or 0)  # type: ignore[call-overload]
                ram_bytes = int(info.get("ram", 0) or 0)  # type: ignore[call-overload]
                memory_gb = round(ram_bytes / (1024**3), 1)
                gpu_count = int(info.get("gpu", 0) or 0)  # type: ignore[call-overload]
                arch = str(info.get("arch", ""))
                gpu_type = arch.upper() if gpu_count > 0 and arch else None
                per_hour = info.get("hourly_price")
                price: float | None = None
                if per_hour is not None:
                    price = float(str(per_hour)) / 100_000  # micro-cents
                result.append(
                    CatalogComputeFlavor(
                        provider_id="scaleway",
                        name=name,
                        vcpus=ncpus,
                        memory_gb=memory_gb,
                        gpu_type=gpu_type,
                        gpu_count=gpu_count,
                        price_per_hour=price,
                        currency="EUR",
                        category="gpu" if gpu_count > 0 else "general",
                    )
                )
            return result
        except Exception:
            logger.warning("[scaleway-catalog] Live fetch failed, using static", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Static fallback
    # ------------------------------------------------------------------

    def _static_compute(self) -> list[CatalogComputeFlavor]:
        return [
            CatalogComputeFlavor(
                provider_id="scaleway",
                name=str(d["name"]),
                vcpus=int(d["vcpus"]),  # type: ignore[call-overload]
                memory_gb=float(d["memory_gb"]),  # type: ignore[arg-type]
                gpu_type=str(d["gpu"]) if "gpu" in d else None,
                gpu_count=int(d["gpus"]) if "gpus" in d else 0,  # type: ignore[call-overload]
                price_per_hour=float(d["price"]),  # type: ignore[arg-type]
                currency="EUR",
                category=str(d.get("cat", "general")),
            )
            for d in _STATIC_COMPUTE
        ]

    def _static_storage(self) -> list[CatalogStorageTier]:
        return [
            CatalogStorageTier(
                provider_id="scaleway",
                name=str(d["name"]),
                type=str(d.get("type", "object_storage")),
                price_per_gb_month=float(d["price"]),  # type: ignore[arg-type]
                currency="EUR",
            )
            for d in _STATIC_STORAGE
        ]
