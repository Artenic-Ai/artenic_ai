"""OVH public catalog fetcher — api.ovh.com/v1/cloud/price."""

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

_PRICING_URL = "https://api.ovh.com/v1/cloud/price"

# Known GPU model strings in OVH flavor names (longest first).
_KNOWN_GPUS = ("a100", "l40s", "l40", "v100", "h100", "l4", "t4")


def _parse_gpu(name: str) -> tuple[str | None, int]:
    """Extract GPU type and count from an OVH flavor name."""
    lower = name.lower()
    gpu_type: str | None = None
    for gpu in _KNOWN_GPUS:
        if gpu in lower:
            gpu_type = gpu.upper()
            break
    if gpu_type is None and not lower.startswith("gpu"):
        return None, 0
    if gpu_type is None:
        gpu_type = "GPU"
    count = 1
    for part in lower.split("-"):
        if part.startswith("x") and part[1:].isdigit():
            count = int(part[1:])
            break
    return gpu_type, count


def _category_from_name(name: str) -> str:
    """Derive instance category from OVH flavor name prefix."""
    lower = name.lower()
    if lower.startswith("gpu") or any(g in lower for g in _KNOWN_GPUS):
        return "gpu"
    if lower.startswith("r"):
        return "memory"
    if lower.startswith("c"):
        return "compute"
    if lower.startswith("i"):
        return "storage"
    return "general"


class OvhCatalogFetcher(CatalogFetcher):
    """Fetch public pricing from OVH API (no auth required)."""

    async def fetch_compute_catalog(self) -> list[CatalogComputeFlavor]:
        data = await self._get()
        flavors: list[dict[str, object]] = data.get("flavorHourly", [])  # type: ignore[assignment]
        result: list[CatalogComputeFlavor] = []
        for f in flavors:
            name = str(f.get("flavorName", ""))
            if not name:
                continue
            gpu_type, gpu_count = _parse_gpu(name)
            vcpus = int(f.get("vcpus", 0) or 0)  # type: ignore[call-overload]
            ram_mb = int(f.get("ram", 0) or 0)  # type: ignore[call-overload]
            disk_gb = float(f.get("disk", 0) or 0)  # type: ignore[arg-type]
            price_info = f.get("price", {})
            price_val: float | None = None
            if isinstance(price_info, dict):
                raw = price_info.get("value")
                if raw is not None:
                    price_val = float(raw)
            result.append(
                CatalogComputeFlavor(
                    provider_id="ovh",
                    name=name,
                    vcpus=vcpus,
                    memory_gb=round(ram_mb / 1024, 1) if ram_mb else 0.0,
                    disk_gb=disk_gb,
                    gpu_type=gpu_type,
                    gpu_count=gpu_count,
                    price_per_hour=price_val,
                    currency="EUR",
                    category=_category_from_name(name),
                )
            )
        return result

    async def fetch_storage_catalog(self) -> list[CatalogStorageTier]:
        data = await self._get()
        storage_list: list[dict[str, object]] = data.get("objectStorage", [])  # type: ignore[assignment]
        result: list[CatalogStorageTier] = []
        for s in storage_list:
            name = str(s.get("name", "Object Storage"))
            price_info = s.get("price", {})
            price_val: float | None = None
            if isinstance(price_info, dict):
                raw = price_info.get("value")
                if raw is not None:
                    price_val = float(raw)
            region = str(s.get("region", ""))
            result.append(
                CatalogStorageTier(
                    provider_id="ovh",
                    name=name,
                    type="object_storage",
                    price_per_gb_month=price_val,
                    currency="EUR",
                    region=region,
                )
            )
        return result

    # ------------------------------------------------------------------

    async def _get(self) -> dict[str, object]:
        if httpx is None:
            msg = "httpx is required for OVH catalog.  pip install httpx"
            raise ImportError(msg)
        resp = await asyncio.to_thread(httpx.get, _PRICING_URL, timeout=30.0)
        resp.raise_for_status()
        result: dict[str, object] = resp.json()
        return result
