"""GCP connector — Google Cloud Platform (GCS + Compute Engine).

Relies on ``google-cloud-storage`` and ``google-cloud-compute``
(Apache 2.0, optional dependencies).  All data is fetched from the
Google Cloud API using the user's own service account credentials.
"""

from __future__ import annotations

import asyncio
import json
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
# Lazy imports — google-cloud packages are optional dependencies
# ---------------------------------------------------------------------------
try:
    from google.cloud import compute_v1  # pragma: no cover
    from google.cloud import storage as gcs  # pragma: no cover
    from google.oauth2 import service_account  # pragma: no cover

    _HAS_GCP = True  # pragma: no cover
except ImportError:
    _HAS_GCP = False
    compute_v1 = None
    gcs = None
    service_account = None  # type: ignore[assignment]


def _require_gcp() -> None:
    if not _HAS_GCP:
        msg = (
            "The 'google-cloud-storage' and 'google-cloud-compute' packages "
            "are required for the GCP connector.  Install them with:  "
            "pip install google-cloud-storage google-cloud-compute"
        )
        raise ImportError(msg)


# Known GPU accelerator types
_GPU_TYPES: dict[str, str] = {
    "nvidia-tesla-t4": "T4",
    "nvidia-tesla-v100": "V100",
    "nvidia-tesla-a100": "A100",
    "nvidia-a100-80gb": "A100-80G",
    "nvidia-l4": "L4",
    "nvidia-h100-80gb": "H100",
}


class GcpConnector(ProviderConnector):
    """Connector for Google Cloud Platform."""

    def __init__(self, provider_id: str = "gcp") -> None:
        self._provider_id = provider_id

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _credentials(self, ctx: ConnectorContext) -> Any:
        """Build google.oauth2 credentials from the service account JSON."""
        _require_gcp()
        raw = ctx.credentials.get("credentials_json", "{}")  # pragma: no cover
        info = json.loads(raw)  # pragma: no cover
        return service_account.Credentials.from_service_account_info(info)  # type: ignore[no-untyped-call]  # pragma: no cover

    def _project(self, ctx: ConnectorContext) -> str:
        return ctx.credentials.get("project_id", "")

    def _zone(self, ctx: ConnectorContext) -> str:
        return ctx.config.get("zone", "europe-west1-b")

    # ------------------------------------------------------------------
    # test_connection
    # ------------------------------------------------------------------

    async def test_connection(self, ctx: ConnectorContext) -> ConnectionTestResult:
        t0 = time.monotonic()
        try:
            creds = self._credentials(ctx)
            project = self._project(ctx)
            zone = self._zone(ctx)

            def _test() -> int:
                client = compute_v1.MachineTypesClient(credentials=creds)
                request = compute_v1.ListMachineTypesRequest(
                    project=project,
                    zone=zone,
                    max_results=1,
                )
                page = client.list(request=request)
                return len(list(page))

            count = await asyncio.to_thread(_test)
            elapsed_ms = (time.monotonic() - t0) * 1000
            return ConnectionTestResult(
                success=True,
                message=f"Connected — {count} machine type(s) found",
                latency_ms=round(elapsed_ms, 1),
            )
        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.warning("[gcp] Connection test failed: %s", exc)
            return ConnectionTestResult(
                success=False,
                message=str(exc),
                latency_ms=round(elapsed_ms, 1),
            )

    # ------------------------------------------------------------------
    # list_storage_options
    # ------------------------------------------------------------------

    async def list_storage_options(self, ctx: ConnectorContext) -> list[StorageOption]:
        try:
            creds = self._credentials(ctx)
        except Exception:
            logger.warning("[gcp] Failed to build credentials for GCS", exc_info=True)
            return []
        project = self._project(ctx)

        def _list() -> list[Any]:
            client = gcs.Client(project=project, credentials=creds)
            return list(client.list_buckets())

        try:
            buckets = await asyncio.to_thread(_list)
        except Exception:
            logger.warning("[gcp] Failed to list GCS buckets", exc_info=True)
            return []

        return [
            StorageOption(
                provider_id=self._provider_id,
                name=b.name,
                type="gcs",
                region=getattr(b, "location", "") or "",
            )
            for b in buckets
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
            creds = self._credentials(ctx)
        except Exception:
            logger.warning("[gcp] Failed to build credentials for compute", exc_info=True)
            return []
        project = self._project(ctx)
        zone = self._zone(ctx)

        def _list() -> list[Any]:
            client = compute_v1.MachineTypesClient(credentials=creds)
            request = compute_v1.ListMachineTypesRequest(project=project, zone=zone)
            return list(client.list(request=request))

        try:
            machine_types = await asyncio.to_thread(_list)
        except Exception:
            logger.warning("[gcp] Failed to list machine types", exc_info=True)
            return []

        instances: list[ComputeInstance] = []
        for mt in machine_types:
            name: str = mt.name
            vcpus: int = mt.guest_cpus or 0
            memory_mb: int = mt.memory_mb or 0
            memory_gb = round(memory_mb / 1024, 1)

            gpu_count = 0
            gpu_type: str | None = None
            accelerators = getattr(mt, "accelerators", []) or []
            for acc in accelerators:
                gpu_count += getattr(acc, "guest_accelerator_count", 0)
                raw_type = getattr(acc, "guest_accelerator_type", "")
                gpu_type = _GPU_TYPES.get(raw_type, raw_type)

            # Detect GPU from name (e.g. a2-highgpu-1g)
            if gpu_count == 0 and name.startswith("a2-"):
                gpu_type = "A100"
                gpu_count = 1

            inst = ComputeInstance(
                provider_id=self._provider_id,
                name=name,
                vcpus=vcpus,
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
        try:
            creds = self._credentials(ctx)
        except Exception:
            logger.warning("[gcp] Failed to build credentials for regions", exc_info=True)
            return []
        project = self._project(ctx)

        def _list() -> list[Any]:
            client = compute_v1.RegionsClient(credentials=creds)
            request = compute_v1.ListRegionsRequest(project=project)
            return list(client.list(request=request))

        try:
            regions = await asyncio.to_thread(_list)
        except Exception:
            logger.warning("[gcp] Failed to list regions", exc_info=True)
            return []

        return [
            ProviderRegion(
                provider_id=self._provider_id,
                id=r.name,
                name=getattr(r, "description", "") or r.name,
            )
            for r in regions
        ]
