"""Azure connector — Microsoft Azure (Blob Storage + Virtual Machines).

Relies on ``azure-identity``, ``azure-mgmt-compute``,
``azure-mgmt-subscription``, and ``azure-storage-blob``
(MIT, optional dependencies).  All data is fetched from the Azure API
using the user's own service principal credentials.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from artenic_ai_platform_providers.hub.connectors.base import (
    ConnectorContext,
    ProviderConnector,
)
from artenic_ai_platform_providers.hub.schemas import (
    ComputeInstance,
    ConnectionTestResult,
    ProviderRegion,
    StorageOption,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — azure packages are optional dependencies
# ---------------------------------------------------------------------------
try:
    from azure.identity import ClientSecretCredential  # pragma: no cover
    from azure.mgmt.compute import ComputeManagementClient  # pragma: no cover
    from azure.mgmt.subscription import SubscriptionClient  # pragma: no cover
    from azure.storage.blob import BlobServiceClient  # pragma: no cover

    _HAS_AZURE = True  # pragma: no cover
except ImportError:
    _HAS_AZURE = False
    ClientSecretCredential = None
    ComputeManagementClient = None
    SubscriptionClient = None
    BlobServiceClient = None


def _require_azure() -> None:
    if not _HAS_AZURE:
        msg = (
            "The Azure SDK packages are required for the Azure connector.  "
            "Install them with:  pip install azure-identity azure-mgmt-compute "
            "azure-mgmt-subscription azure-storage-blob"
        )
        raise ImportError(msg)


# Known GPU VM sizes
_GPU_SIZES: dict[str, str] = {
    "NC": "T4",
    "ND": "A100",
    "NV": "M60",
    "NCas_T4": "T4",
    "NCads_A100": "A100",
    "NDs_A100": "A100",
    "NDm_A100": "A100",
}


def _detect_gpu(vm_name: str) -> tuple[str | None, int]:
    """Detect GPU type and count from Azure VM size name.

    Azure GPU VM sizes follow the pattern ``Standard_N{family}...``
    (e.g. Standard_NC6, Standard_ND40rs_v2, Standard_NV12).
    We extract the part after ``Standard_`` and check if it starts
    with a known GPU family prefix.
    """
    parts = vm_name.split("_", 1)
    # Expect "Standard_NC6" → ["Standard", "NC6"]
    suffix = parts[1] if len(parts) == 2 else vm_name
    suffix_upper = suffix.upper()
    for prefix, gpu_type in _GPU_SIZES.items():
        if suffix_upper.startswith(prefix.upper()):
            return gpu_type, 1
    return None, 0


class AzureConnector(ProviderConnector):
    """Connector for Microsoft Azure."""

    def __init__(self, provider_id: str = "azure") -> None:
        self._provider_id = provider_id

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _credential(self, ctx: ConnectorContext) -> Any:
        """Build Azure ClientSecretCredential from user credentials."""
        _require_azure()
        return ClientSecretCredential(  # pragma: no cover
            tenant_id=ctx.credentials.get("tenant_id", ""),
            client_id=ctx.credentials.get("client_id", ""),
            client_secret=ctx.credentials.get("client_secret", ""),
        )

    def _subscription_id(self, ctx: ConnectorContext) -> str:
        return ctx.credentials.get("subscription_id", "")

    def _region(self, ctx: ConnectorContext) -> str:
        return ctx.config.get("region", "westeurope")

    # ------------------------------------------------------------------
    # test_connection
    # ------------------------------------------------------------------

    async def test_connection(self, ctx: ConnectorContext) -> ConnectionTestResult:
        t0 = time.monotonic()
        try:
            cred = self._credential(ctx)
            sub_id = self._subscription_id(ctx)
            region = self._region(ctx)

            def _test() -> int:
                client = ComputeManagementClient(cred, sub_id)
                sizes = list(client.virtual_machine_sizes.list(location=region))
                return len(sizes)

            count = await asyncio.to_thread(_test)
            elapsed_ms = (time.monotonic() - t0) * 1000
            return ConnectionTestResult(
                success=True,
                message=f"Connected — {count} VM sizes available in {region}",
                latency_ms=round(elapsed_ms, 1),
            )
        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.warning("[azure] Connection test failed: %s", exc)
            return ConnectionTestResult(
                success=False,
                message=str(exc),
                latency_ms=round(elapsed_ms, 1),
            )

    # ------------------------------------------------------------------
    # list_storage_options
    # ------------------------------------------------------------------

    async def list_storage_options(self, ctx: ConnectorContext) -> list[StorageOption]:
        # Blob storage listing requires a storage account connection string
        # or account URL — for now we list containers from config if provided
        account_url = ctx.config.get("storage_account_url", "")
        if not account_url:
            return []

        try:
            cred = self._credential(ctx)
        except Exception:
            logger.warning("[azure] Failed to build credentials for storage", exc_info=True)
            return []

        def _list() -> list[dict[str, Any]]:
            client = BlobServiceClient(account_url, credential=cred)
            return [{"name": c["name"]} for c in client.list_containers()]

        try:
            containers = await asyncio.to_thread(_list)
        except Exception:
            logger.warning("[azure] Failed to list blob containers", exc_info=True)
            return []

        return [
            StorageOption(
                provider_id=self._provider_id,
                name=c["name"],
                type="blob_storage",
                region=self._region(ctx),
            )
            for c in containers
        ]

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
        try:
            cred = self._credential(ctx)
        except Exception:
            logger.warning("[azure] Failed to build credentials for compute", exc_info=True)
            return []
        sub_id = self._subscription_id(ctx)
        location = region or self._region(ctx)

        def _list() -> list[Any]:
            client = ComputeManagementClient(cred, sub_id)
            return list(client.virtual_machine_sizes.list(location=location))

        try:
            sizes = await asyncio.to_thread(_list)
        except Exception:
            logger.warning("[azure] Failed to list VM sizes", exc_info=True)
            return []

        instances: list[ComputeInstance] = []
        for sz in sizes:
            name: str = sz.name
            vcpus: int = sz.number_of_cores or 0
            memory_mb: int = sz.memory_in_mb or 0
            memory_gb = round(memory_mb / 1024, 1)
            disk_gb = float(sz.resource_disk_size_in_mb or 0) / 1024

            gpu_type, gpu_count = _detect_gpu(name)

            inst = ComputeInstance(
                provider_id=self._provider_id,
                name=name,
                vcpus=vcpus,
                memory_gb=memory_gb,
                disk_gb=round(disk_gb, 1),
                gpu_type=gpu_type,
                gpu_count=gpu_count,
                region=location,
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
        try:
            cred = self._credential(ctx)
        except Exception:
            logger.warning("[azure] Failed to build credentials for regions", exc_info=True)
            return []
        sub_id = self._subscription_id(ctx)

        def _list() -> list[Any]:
            client = SubscriptionClient(cred)
            return list(client.subscriptions.list_locations(sub_id))

        try:
            locations = await asyncio.to_thread(_list)
        except Exception:
            logger.warning("[azure] Failed to list regions", exc_info=True)
            return []

        return [
            ProviderRegion(
                provider_id=self._provider_id,
                id=loc.name or "",
                name=getattr(loc, "display_name", "") or loc.name or "",
            )
            for loc in locations
        ]
