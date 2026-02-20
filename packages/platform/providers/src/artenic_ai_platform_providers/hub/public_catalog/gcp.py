"""GCP public catalog fetcher — unofficial pricing calculator JSON."""

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

_PRICELIST_URL = "https://cloudpricingcalculator.appspot.com/static/data/pricelist.json"

# GCP machine type families
_MACHINE_FAMILIES: dict[str, str] = {
    "n1-standard": "general",
    "n2-standard": "general",
    "n2d-standard": "general",
    "n4-standard": "general",
    "e2-standard": "general",
    "e2-medium": "general",
    "e2-small": "general",
    "e2-micro": "general",
    "c2-standard": "compute",
    "c2d-standard": "compute",
    "c3-standard": "compute",
    "c4-standard": "compute",
    "m1-megamem": "memory",
    "m1-ultramem": "memory",
    "m2-megamem": "memory",
    "m2-ultramem": "memory",
    "m3-megamem": "memory",
    "m3-ultramem": "memory",
    "n2-highmem": "memory",
    "n2d-highmem": "memory",
    "n1-highmem": "memory",
    "n1-highcpu": "compute",
    "n2-highcpu": "compute",
    "a2-highgpu": "gpu",
    "a2-megagpu": "gpu",
    "a3-highgpu": "gpu",
    "g2-standard": "gpu",
}

_GPU_TYPES: dict[str, str] = {
    "a2-highgpu": "A100",
    "a2-megagpu": "A100",
    "a3-highgpu": "H100",
    "g2-standard": "L4",
}


class GcpCatalogFetcher(CatalogFetcher):
    """Fetch public pricing from GCP pricing calculator JSON."""

    async def fetch_compute_catalog(self) -> list[CatalogComputeFlavor]:
        data = await self._get()
        price_list: dict[str, object] = data.get("gcp_price_list", {})  # type: ignore[assignment]
        result: list[CatalogComputeFlavor] = []
        seen: set[str] = set()

        for key, value in price_list.items():
            if not isinstance(value, dict):
                continue
            # Match keys like CP-COMPUTEENGINE-VMIMAGE-N1-STANDARD-1
            if "VMIMAGE" not in key.upper():
                continue
            # Extract machine type from key
            machine_type = _extract_machine_type(key)
            if not machine_type or machine_type in seen:
                continue
            seen.add(machine_type)

            # Extract pricing (us region as default)
            price = value.get("us")
            if price is None:
                continue

            # Parse specs from machine type name
            vcpus, memory_gb = _parse_machine_specs(machine_type)
            family = _get_family(machine_type)
            category = _MACHINE_FAMILIES.get(family, "general")
            gpu_type = _GPU_TYPES.get(family)
            gpu_count = _gpu_count_from_name(machine_type) if gpu_type else 0

            result.append(
                CatalogComputeFlavor(
                    provider_id="gcp",
                    name=machine_type,
                    vcpus=vcpus,
                    memory_gb=memory_gb,
                    gpu_type=gpu_type,
                    gpu_count=gpu_count,
                    price_per_hour=float(price),
                    currency="USD",
                    category=category,
                )
            )
        return result

    async def fetch_storage_catalog(self) -> list[CatalogStorageTier]:
        data = await self._get()
        price_list: dict[str, object] = data.get("gcp_price_list", {})  # type: ignore[assignment]
        result: list[CatalogStorageTier] = []

        for key, value in price_list.items():
            if not isinstance(value, dict):
                continue
            upper = key.upper()
            if "STORAGE" not in upper or ("NEARLINE" not in upper and "STANDARD" not in upper):
                continue
            if "CLOUD-STORAGE" not in upper and "GCS" not in upper:
                continue
            price = value.get("us")
            if price is None:
                continue
            tier_name = "Standard" if "STANDARD" in upper else "Nearline"
            result.append(
                CatalogStorageTier(
                    provider_id="gcp",
                    name=f"Cloud Storage — {tier_name}",
                    type="object_storage",
                    price_per_gb_month=float(price),
                    currency="USD",
                )
            )
        return result

    # ------------------------------------------------------------------

    async def _get(self) -> dict[str, object]:
        if httpx is None:
            msg = "httpx is required for GCP catalog.  pip install httpx"
            raise ImportError(msg)
        resp = await asyncio.to_thread(
            httpx.get,
            _PRICELIST_URL,
            timeout=30.0,
            follow_redirects=True,
        )
        resp.raise_for_status()
        result: dict[str, object] = resp.json()
        return result


def _extract_machine_type(key: str) -> str:
    """Convert pricing key to machine type name (lowercase)."""
    parts = key.upper().split("-")
    try:
        idx = parts.index("VMIMAGE")
    except ValueError:
        return ""
    return "-".join(parts[idx + 1 :]).lower()


def _get_family(machine_type: str) -> str:
    """e.g. 'n1-standard-4' → 'n1-standard'."""
    parts = machine_type.split("-")
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    return machine_type


def _parse_machine_specs(machine_type: str) -> tuple[int, float]:
    """Best-effort vCPU and memory from a machine type name."""
    m = re.search(r"-(\d+)$", machine_type)
    vcpus = int(m.group(1)) if m else 0
    # Rough memory estimates by family
    family = _get_family(machine_type)
    mem_per_vcpu = 3.75  # default (n1-standard)
    if "highmem" in family or "megamem" in family or "ultramem" in family:
        mem_per_vcpu = 6.5
    elif "highcpu" in family:
        mem_per_vcpu = 0.9
    return vcpus, round(vcpus * mem_per_vcpu, 1)


def _gpu_count_from_name(machine_type: str) -> int:
    """Extract GPU count hint from machine type (e.g. a2-highgpu-4g → 4)."""
    m = re.search(r"-(\d+)g?$", machine_type)
    return int(m.group(1)) if m else 1
