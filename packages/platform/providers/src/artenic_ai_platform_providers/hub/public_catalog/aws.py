"""AWS public catalog fetcher — bulk pricing JSON from pricing.us-east-1.amazonaws.com."""

from __future__ import annotations

import asyncio
import logging
import re

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

_INDEX_URL = (
    "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/eu-west-1/index.json"
)

# Instance family prefix → GPU type
_GPU_FAMILIES: dict[str, str] = {
    "p2": "K80",
    "p3": "V100",
    "p4d": "A100",
    "p4de": "A100",
    "p5": "H100",
    "g4dn": "T4",
    "g4ad": "Radeon",
    "g5": "A10G",
    "g6": "L4",
    "g6e": "L4",
    "inf1": "Inferentia",
    "inf2": "Inferentia2",
    "trn1": "Trainium",
}


def _parse_instance_family(instance_type: str) -> str:
    """Extract family prefix from an AWS instance type (e.g. 'p4d.24xlarge' → 'p4d')."""
    return instance_type.split(".")[0] if "." in instance_type else instance_type


def _gpu_for_family(family: str) -> tuple[str | None, int]:
    gpu = _GPU_FAMILIES.get(family)
    if gpu is None:
        return None, 0
    return gpu, 1  # count is approximate


def _category_from_family(family: str) -> str:
    if family in _GPU_FAMILIES:
        return "gpu"
    first = family.rstrip("0123456789")
    if first in ("c", "hpc"):
        return "compute"
    if first in ("r", "x", "z"):
        return "memory"
    if first in ("i", "d", "h"):
        return "storage"
    return "general"


class AwsCatalogFetcher(CatalogFetcher):
    """Fetch public EC2 pricing from AWS bulk pricing JSON."""

    def __init__(self, region_url: str = _INDEX_URL) -> None:
        self._url = region_url

    async def fetch_compute_catalog(self) -> list[CatalogComputeFlavor]:
        data = await self._get()
        products: dict[str, dict[str, object]] = data.get("products", {})  # type: ignore[assignment]
        terms: dict[str, dict[str, object]] = data.get("terms", {})  # type: ignore[assignment]
        on_demand: dict[str, dict[str, object]] = terms.get("OnDemand", {})  # type: ignore[assignment]

        result: list[CatalogComputeFlavor] = []
        seen: set[str] = set()

        for sku, product in products.items():
            attrs: dict[str, str] = product.get("attributes", {})  # type: ignore[assignment]
            instance_type = attrs.get("instanceType", "")
            if not instance_type or instance_type in seen:
                continue
            # Filter: Linux, Shared tenancy, current generation
            if attrs.get("operatingSystem") != "Linux":
                continue
            if attrs.get("tenancy") != "Shared":
                continue

            seen.add(instance_type)
            family = _parse_instance_family(instance_type)
            gpu_type, gpu_count = _gpu_for_family(family)
            vcpus = int(attrs.get("vcpu", "0").replace(",", "") or 0)
            mem_str = attrs.get("memory", "0 GiB").split()[0].replace(",", "")
            memory_gb = float(mem_str) if mem_str else 0.0
            storage_str = attrs.get("storage", "")
            disk_gb = _parse_storage(storage_str)

            # Extract On-Demand price
            price = _extract_on_demand_price(sku, on_demand)

            result.append(
                CatalogComputeFlavor(
                    provider_id="aws",
                    name=instance_type,
                    vcpus=vcpus,
                    memory_gb=memory_gb,
                    disk_gb=disk_gb,
                    gpu_type=gpu_type,
                    gpu_count=gpu_count,
                    price_per_hour=price,
                    currency="USD",
                    region=attrs.get("location", ""),
                    category=_category_from_family(family),
                )
            )
        return result

    async def fetch_storage_catalog(self) -> list[CatalogStorageTier]:
        return []  # S3 pricing is separate; skip for v1

    # ------------------------------------------------------------------

    async def _get(self) -> dict[str, object]:
        if httpx is None:
            msg = "httpx is required for AWS catalog.  pip install httpx"
            raise ImportError(msg)
        resp = await asyncio.to_thread(
            httpx.get,
            self._url,
            timeout=60.0,
            follow_redirects=True,
        )
        resp.raise_for_status()
        result: dict[str, object] = resp.json()
        return result


def _extract_on_demand_price(
    sku: str,
    on_demand: dict[str, dict[str, object]],
) -> float | None:
    """Find the hourly USD price for a given SKU in OnDemand terms."""
    sku_terms = on_demand.get(sku, {})
    for _offer_id, offer in sku_terms.items():
        if not isinstance(offer, dict):
            continue
        dims: dict[str, dict[str, object]] = offer.get("priceDimensions", {})  # type: ignore[assignment,unused-ignore]
        for _dim_id, dim in dims.items():
            usd = dim.get("pricePerUnit", {})
            if isinstance(usd, dict):
                raw = usd.get("USD")
                if raw is not None:
                    return float(raw)
    return None


def _parse_storage(storage_str: str) -> float:
    """Best-effort parse of AWS storage attribute (e.g. '2 x 900 NVMe SSD')."""
    if not storage_str or storage_str.lower() in ("ebs only", "ebs-only", ""):
        return 0.0
    m = re.match(r"(\d+)\s*x\s*(\d+)", storage_str)
    if m:
        return float(int(m.group(1)) * int(m.group(2)))
    m2 = re.match(r"(\d+)", storage_str)
    return float(m2.group(1)) if m2 else 0.0
