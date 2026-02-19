"""Scaleway connector — uses the Scaleway HTTP API directly.

Relies on ``httpx`` (MIT licence, already a platform dependency).
All data is fetched from the Scaleway API using the user's own credentials.
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
            "The 'httpx' package is required for the Scaleway connector.  "
            "Install it with:  pip install httpx"
        )
        raise ImportError(msg)


_BASE_URL = "https://api.scaleway.com"

# Static region list — Scaleway does not expose a public regions endpoint
_SCALEWAY_ZONES = (
    ("fr-par-1", "Paris 1"),
    ("fr-par-2", "Paris 2"),
    ("fr-par-3", "Paris 3"),
    ("nl-ams-1", "Amsterdam 1"),
    ("nl-ams-2", "Amsterdam 2"),
    ("nl-ams-3", "Amsterdam 3"),
    ("pl-waw-1", "Warsaw 1"),
    ("pl-waw-2", "Warsaw 2"),
)


class ScalewayConnector(ProviderConnector):
    """Connector for Scaleway cloud."""

    def __init__(self, provider_id: str = "scaleway") -> None:
        self._provider_id = provider_id

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _headers(self, ctx: ConnectorContext) -> dict[str, str]:
        return {"X-Auth-Token": ctx.credentials.get("secret_key", "")}

    def _zone(self, ctx: ConnectorContext) -> str:
        return ctx.config.get("zone", "fr-par-1")

    def _region(self, ctx: ConnectorContext) -> str:
        """Derive region from zone (e.g. 'fr-par-1' → 'fr-par')."""
        zone = self._zone(ctx)
        parts = zone.rsplit("-", 1)
        return parts[0] if len(parts) == 2 else zone

    async def _get(self, url: str, headers: dict[str, str]) -> Any:
        """Perform a blocking httpx GET in a thread."""
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
            zone = self._zone(ctx)
            url = f"{_BASE_URL}/instance/v1/zones/{zone}/servers?per_page=1"
            await self._get(url, self._headers(ctx))
            elapsed_ms = (time.monotonic() - t0) * 1000
            return ConnectionTestResult(
                success=True,
                message="Connected to Scaleway API",
                latency_ms=round(elapsed_ms, 1),
            )
        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.warning("[scaleway] Connection test failed: %s", exc)
            return ConnectionTestResult(
                success=False,
                message=str(exc),
                latency_ms=round(elapsed_ms, 1),
            )

    # ------------------------------------------------------------------
    # list_storage_options
    # ------------------------------------------------------------------

    async def list_storage_options(self, ctx: ConnectorContext) -> list[StorageOption]:
        region = self._region(ctx)
        url = f"{_BASE_URL}/object/v1/regions/{region}/buckets"
        try:
            data = await self._get(url, self._headers(ctx))
        except Exception:
            logger.warning("[scaleway] Failed to list buckets", exc_info=True)
            return []

        result: list[StorageOption] = []
        for bucket in data.get("buckets", []):
            result.append(
                StorageOption(
                    provider_id=self._provider_id,
                    name=bucket.get("name", ""),
                    type="object_storage",
                    region=region,
                )
            )
        return result

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
        zone = self._zone(ctx)
        url = f"{_BASE_URL}/instance/v1/zones/{zone}/products/servers"
        try:
            data = await self._get(url, self._headers(ctx))
        except Exception:
            logger.warning("[scaleway] Failed to list server types", exc_info=True)
            return []

        instances: list[ComputeInstance] = []
        servers: dict[str, Any] = data.get("servers", {})
        for name, info in servers.items():
            ncpus: int = info.get("ncpus", 0)
            ram_bytes: int = info.get("ram", 0)
            memory_gb = round(ram_bytes / (1024**3), 1) if ram_bytes else 0.0

            gpu_count: int = info.get("gpu", 0) or 0
            gpu_type: str | None = None
            if gpu_count > 0:
                gpu_type = info.get("arch", "GPU").upper()

            inst = ComputeInstance(
                provider_id=self._provider_id,
                name=name,
                vcpus=ncpus,
                memory_gb=memory_gb,
                gpu_type=gpu_type,
                gpu_count=gpu_count,
                region=region or zone,
                available=True,
            )
            instances.append(inst)

        if gpu_only:
            instances = [i for i in instances if i.gpu_count > 0]

        return instances

    # ------------------------------------------------------------------
    # list_regions
    # ------------------------------------------------------------------

    async def list_regions(self, ctx: ConnectorContext) -> list[ProviderRegion]:
        return [
            ProviderRegion(provider_id=self._provider_id, id=zid, name=zname)
            for zid, zname in _SCALEWAY_ZONES
        ]
