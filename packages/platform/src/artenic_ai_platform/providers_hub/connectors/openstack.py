"""OpenStack connector — works with any OpenStack-based cloud.

Used by OVH Public Cloud, Infomaniak, and any private OpenStack
deployment.  Relies on ``openstacksdk`` (Apache 2.0).  All data is
fetched live from the provider's API using the user's own credentials.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import openstack
import openstack.connection

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


# Known GPU model strings in flavor names.
# Ordered longest-first so "l40s" matches before "l4".
_KNOWN_GPUS = ("a100", "l40s", "l40", "v100", "h100", "l4", "t4")


class OpenStackConnector(ProviderConnector):
    """Connector for any OpenStack-based cloud provider."""

    def __init__(self, provider_id: str = "openstack") -> None:
        self._provider_id = provider_id

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_connection(self, ctx: ConnectorContext) -> Any:
        """Create an ``openstack.Connection`` from user credentials."""
        creds = ctx.credentials  # pragma: no cover
        cfg = ctx.config  # pragma: no cover
        auth_url = cfg.get("auth_url") or creds.get("auth_url", "")  # pragma: no cover
        return openstack.connect(  # pragma: no cover
            auth_url=auth_url,
            project_id=creds.get("project_id", ""),
            username=creds.get("username", ""),
            password=creds.get("password", ""),
            region_name=cfg.get("region", ""),
            user_domain_name=cfg.get("user_domain_name", "Default"),
            project_domain_name=cfg.get("project_domain_name", "Default"),
        )

    # ------------------------------------------------------------------
    # test_connection
    # ------------------------------------------------------------------

    async def test_connection(self, ctx: ConnectorContext) -> ConnectionTestResult:
        """Connect and perform a lightweight API call."""
        t0 = time.monotonic()
        try:
            conn = await asyncio.to_thread(self._build_connection, ctx)
            try:
                flavors = await asyncio.to_thread(lambda: list(conn.compute.flavors()))
                elapsed_ms = (time.monotonic() - t0) * 1000
                return ConnectionTestResult(
                    success=True,
                    message=f"Connected — {len(flavors)} flavors available",
                    latency_ms=round(elapsed_ms, 1),
                )
            finally:
                await asyncio.to_thread(conn.close)
        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.warning("[openstack] Connection test failed: %s", exc)
            return ConnectionTestResult(
                success=False,
                message=str(exc),
                latency_ms=round(elapsed_ms, 1),
            )

    # ------------------------------------------------------------------
    # list_storage_options
    # ------------------------------------------------------------------

    async def list_storage_options(
        self,
        ctx: ConnectorContext,
    ) -> list[StorageOption]:
        """List Swift containers for the user's project."""
        conn = await asyncio.to_thread(self._build_connection, ctx)
        try:
            raw: list[Any] = await asyncio.to_thread(lambda: list(conn.object_store.containers()))
            result: list[StorageOption] = []
            for c in raw:
                result.append(
                    StorageOption(
                        provider_id=self._provider_id,
                        name=c.name,
                        type="object_storage",
                        region=ctx.config.get("region", ""),
                        bytes_used=getattr(c, "bytes_used", None) or getattr(c, "bytes", None),
                        object_count=getattr(c, "count", None) or getattr(c, "object_count", None),
                    )
                )
            return result
        finally:
            await asyncio.to_thread(conn.close)

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
        """List Nova flavors for the user's project."""
        conn = await asyncio.to_thread(self._build_connection, ctx)
        try:
            raw: list[Any] = await asyncio.to_thread(
                lambda: list(conn.compute.flavors(details=True))
            )
            instances: list[ComputeInstance] = []
            for flv in raw:
                name: str = flv.name
                vcpus: int = flv.vcpus or 0
                ram_mb: int = flv.ram or 0
                disk_gb: float = float(flv.disk or 0)
                memory_gb = round(ram_mb / 1024.0, 1)

                gpu_type, gpu_count = _parse_gpu_info(name)

                inst = ComputeInstance(
                    provider_id=self._provider_id,
                    name=name,
                    vcpus=vcpus,
                    memory_gb=memory_gb,
                    disk_gb=disk_gb,
                    gpu_type=gpu_type,
                    gpu_count=gpu_count,
                    region=region or ctx.config.get("region", ""),
                    available=True,
                )
                instances.append(inst)

            if gpu_only:
                instances = [i for i in instances if i.gpu_count > 0]

            return instances
        finally:
            await asyncio.to_thread(conn.close)

    # ------------------------------------------------------------------
    # list_regions
    # ------------------------------------------------------------------

    async def list_regions(
        self,
        ctx: ConnectorContext,
    ) -> list[ProviderRegion]:
        """List regions from Keystone."""
        conn = await asyncio.to_thread(self._build_connection, ctx)
        try:
            raw: list[Any] = await asyncio.to_thread(lambda: list(conn.identity.regions()))
            return [
                ProviderRegion(
                    provider_id=self._provider_id,
                    id=r.id,
                    name=getattr(r, "description", "") or r.id,
                )
                for r in raw
            ]
        finally:
            await asyncio.to_thread(conn.close)


# ---------------------------------------------------------------------------
# GPU parsing helpers
# ---------------------------------------------------------------------------


def _parse_gpu_info(flavor_name: str) -> tuple[str | None, int]:
    """Extract GPU type and count from an OpenStack flavor name."""
    name_lower = flavor_name.lower()

    # Check if it's a GPU flavor
    gpu_type: str | None = None
    for gpu in _KNOWN_GPUS:
        if gpu in name_lower:
            gpu_type = gpu.upper()
            break

    if gpu_type is None and not name_lower.startswith("gpu"):
        return None, 0

    if gpu_type is None:
        gpu_type = "GPU"

    # Parse count from "-xN" suffix
    gpu_count = 1
    parts = name_lower.split("-")
    for part in parts:
        if part.startswith("x") and part[1:].isdigit():
            gpu_count = int(part[1:])
            break

    return gpu_type, gpu_count
