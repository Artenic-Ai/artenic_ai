"""Infomaniak public catalog fetcher — static pricing data."""

from __future__ import annotations

from artenic_ai_platform_providers.hub.public_catalog.base import CatalogFetcher
from artenic_ai_platform_providers.hub.schemas import (
    CatalogComputeFlavor,
    CatalogStorageTier,
)

# Static pricing from Infomaniak public website (Feb 2025).
_STATIC_COMPUTE: list[dict[str, object]] = [
    {
        "name": "a1-ram2-disk20-perf1",
        "vcpus": 1,
        "memory_gb": 2.0,
        "disk_gb": 20,
        "price": 0.0074,
        "cat": "general",
    },
    {
        "name": "a2-ram4-disk50-perf1",
        "vcpus": 2,
        "memory_gb": 4.0,
        "disk_gb": 50,
        "price": 0.0164,
        "cat": "general",
    },
    {
        "name": "a4-ram8-disk100-perf1",
        "vcpus": 4,
        "memory_gb": 8.0,
        "disk_gb": 100,
        "price": 0.0345,
        "cat": "general",
    },
    {
        "name": "a8-ram16-disk200-perf1",
        "vcpus": 8,
        "memory_gb": 16.0,
        "disk_gb": 200,
        "price": 0.0706,
        "cat": "general",
    },
    {
        "name": "a16-ram32-disk400-perf1",
        "vcpus": 16,
        "memory_gb": 32.0,
        "disk_gb": 400,
        "price": 0.1429,
        "cat": "general",
    },
    {
        "name": "a32-ram64-disk800-perf1",
        "vcpus": 32,
        "memory_gb": 64.0,
        "disk_gb": 800,
        "price": 0.2874,
        "cat": "general",
    },
]

_STATIC_STORAGE: list[dict[str, object]] = [
    {"name": "Object Storage — Standard", "price": 0.01, "type": "object_storage"},
]


class InfomaniakCatalogFetcher(CatalogFetcher):
    """Static catalog data for Infomaniak (no public pricing API)."""

    def supports_live_catalog(self) -> bool:
        return False

    async def fetch_compute_catalog(self) -> list[CatalogComputeFlavor]:
        return [
            CatalogComputeFlavor(
                provider_id="infomaniak",
                name=str(d["name"]),
                vcpus=int(d["vcpus"]),  # type: ignore[call-overload]
                memory_gb=float(d["memory_gb"]),  # type: ignore[arg-type]
                disk_gb=float(d.get("disk_gb", 0)),  # type: ignore[arg-type]
                price_per_hour=float(d["price"]),  # type: ignore[arg-type]
                currency="EUR",
                category=str(d.get("cat", "general")),
            )
            for d in _STATIC_COMPUTE
        ]

    async def fetch_storage_catalog(self) -> list[CatalogStorageTier]:
        return [
            CatalogStorageTier(
                provider_id="infomaniak",
                name=str(d["name"]),
                type=str(d.get("type", "object_storage")),
                price_per_gb_month=float(d["price"]),  # type: ignore[arg-type]
                currency="EUR",
            )
            for d in _STATIC_STORAGE
        ]
