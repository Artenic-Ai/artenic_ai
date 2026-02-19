"""Azure public catalog fetcher — prices.azure.com retail API."""

from __future__ import annotations

import asyncio
import logging
import re

from artenic_ai_platform.providers_hub.public_catalog.base import CatalogFetcher
from artenic_ai_platform.providers_hub.schemas import (
    CatalogComputeFlavor,
    CatalogStorageTier,
)

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_BASE_URL = "https://prices.azure.com/api/retail/prices"
_FILTER = (
    "serviceName eq 'Virtual Machines'"
    " and priceType eq 'Consumption'"
    " and contains(productName, 'Windows') eq false"
)
_MAX_PAGES = 5

# GPU family patterns → GPU type
_GPU_FAMILIES: dict[str, str] = {
    "NC": "T4",
    "ND": "A100",
    "NV": "V100",
}


def _parse_gpu_from_sku(sku: str) -> tuple[str | None, int]:
    """Derive GPU type and approximate count from Azure SKU name."""
    upper = sku.upper()
    for prefix, gpu in _GPU_FAMILIES.items():
        if f"STANDARD_{prefix}" in upper or upper.startswith(prefix):
            # Approximate count from trailing digits
            m = re.search(r"(\d+)", sku.split("_")[-1] if "_" in sku else sku)
            count = max(1, int(m.group(1)) // 6) if m else 1
            return gpu, count
    return None, 0


def _category_from_sku(sku: str) -> str:
    upper = sku.upper()
    if any(f"_{p}" in upper for p in ("NC", "ND", "NV")):
        return "gpu"
    if "_D" in upper:
        return "general"
    if "_E" in upper:
        return "memory"
    if "_F" in upper:
        return "compute"
    if "_L" in upper:
        return "storage"
    return "general"


class AzureCatalogFetcher(CatalogFetcher):
    """Fetch public VM pricing from Azure Retail Prices API."""

    async def fetch_compute_catalog(self) -> list[CatalogComputeFlavor]:
        items = await self._fetch_all_pages()
        seen: set[str] = set()
        result: list[CatalogComputeFlavor] = []
        for item in items:
            sku = str(item.get("armSkuName", ""))
            if not sku or sku in seen:
                continue
            seen.add(sku)
            raw_price = item.get("retailPrice")
            region = str(item.get("armRegionName", ""))
            gpu_type, gpu_count = _parse_gpu_from_sku(sku)
            # Azure doesn't expose vCPU/RAM in pricing API; use sku name
            vcpus = _extract_vcpu_hint(sku)
            price_val = float(str(raw_price)) if raw_price is not None else None
            result.append(
                CatalogComputeFlavor(
                    provider_id="azure",
                    name=sku,
                    vcpus=vcpus,
                    gpu_type=gpu_type,
                    gpu_count=gpu_count,
                    price_per_hour=price_val,
                    currency="USD",
                    region=region,
                    category=_category_from_sku(sku),
                )
            )
        return result

    async def fetch_storage_catalog(self) -> list[CatalogStorageTier]:
        return []  # Azure storage pricing is complex; skip for v1

    # ------------------------------------------------------------------

    async def _fetch_all_pages(self) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        url: str | None = f"{_BASE_URL}?$filter={_FILTER}"
        page = 0
        while url and page < _MAX_PAGES:
            data = await self._get(url)
            page_items: list[dict[str, object]] = data.get("Items", [])  # type: ignore[assignment]
            items.extend(page_items)
            url = data.get("NextPageLink")  # type: ignore[assignment]
            page += 1
        return items

    async def _get(self, url: str) -> dict[str, object]:
        if httpx is None:
            msg = "httpx is required for Azure catalog.  pip install httpx"
            raise ImportError(msg)
        resp = await asyncio.to_thread(httpx.get, url, timeout=30.0)
        resp.raise_for_status()
        result: dict[str, object] = resp.json()
        return result


def _extract_vcpu_hint(sku: str) -> int:
    """Best-effort vCPU count from Azure SKU name (e.g. Standard_D4s_v5 → 4)."""
    m = re.search(r"_\w*?(\d+)", sku)
    return int(m.group(1)) if m else 0
