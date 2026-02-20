"""AWS connector — Amazon Web Services (S3 + EC2).

Relies on ``boto3`` (Apache 2.0, optional dependency).
All data is fetched from the AWS API using the user's own credentials.
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
# Lazy import — boto3 is an optional dependency
# ---------------------------------------------------------------------------
try:
    import boto3  # pragma: no cover

    _HAS_BOTO3 = True  # pragma: no cover
except ImportError:
    _HAS_BOTO3 = False
    boto3 = None


def _require_boto3() -> None:
    if not _HAS_BOTO3:
        msg = (
            "The 'boto3' package is required for the AWS connector.  "
            "Install it with:  pip install boto3"
        )
        raise ImportError(msg)


# Known GPU instance families
_GPU_FAMILIES: dict[str, str] = {
    "p2": "K80",
    "p3": "V100",
    "p4d": "A100",
    "p4de": "A100",
    "p5": "H100",
    "g4dn": "T4",
    "g4ad": "Radeon Pro V520",
    "g5": "A10G",
    "g5g": "T4G",
    "g6": "L4",
    "g6e": "L40S",
    "inf1": "Inferentia",
    "inf2": "Inferentia2",
    "trn1": "Trainium",
    "trn1n": "Trainium",
    "dl1": "Gaudi",
}


class AwsConnector(ProviderConnector):
    """Connector for Amazon Web Services."""

    def __init__(self, provider_id: str = "aws") -> None:
        self._provider_id = provider_id

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _session(self, ctx: ConnectorContext) -> Any:
        """Create a boto3 Session from user credentials."""
        _require_boto3()
        return boto3.Session(  # pragma: no cover
            aws_access_key_id=ctx.credentials.get("access_key_id", ""),
            aws_secret_access_key=ctx.credentials.get("secret_access_key", ""),
            region_name=ctx.config.get("region", "eu-west-1"),
        )

    # ------------------------------------------------------------------
    # test_connection
    # ------------------------------------------------------------------

    async def test_connection(self, ctx: ConnectorContext) -> ConnectionTestResult:
        t0 = time.monotonic()
        try:
            session = self._session(ctx)

            def _test() -> Any:
                sts = session.client("sts")
                return sts.get_caller_identity()

            identity: dict[str, Any] = await asyncio.to_thread(_test)
            elapsed_ms = (time.monotonic() - t0) * 1000
            account = identity.get("Account", "unknown")
            return ConnectionTestResult(
                success=True,
                message=f"Connected — AWS account {account}",
                latency_ms=round(elapsed_ms, 1),
            )
        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.warning("[aws] Connection test failed: %s", exc)
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
            session = self._session(ctx)
        except Exception:
            logger.warning("[aws] Failed to create session for S3", exc_info=True)
            return []

        def _list() -> Any:
            s3 = session.client("s3")
            resp = s3.list_buckets()
            return resp.get("Buckets", [])

        try:
            buckets = await asyncio.to_thread(_list)
        except Exception:
            logger.warning("[aws] Failed to list S3 buckets", exc_info=True)
            return []

        return [
            StorageOption(
                provider_id=self._provider_id,
                name=b.get("Name", ""),
                type="s3",
                region=ctx.config.get("region", ""),
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
            session = self._session(ctx)
        except Exception:
            logger.warning("[aws] Failed to create session for EC2", exc_info=True)
            return []

        def _list() -> list[dict[str, Any]]:
            ec2 = session.client("ec2")
            paginator = ec2.get_paginator("describe_instance_types")
            result: list[dict[str, Any]] = []
            for page in paginator.paginate():
                result.extend(page.get("InstanceTypes", []))
            return result

        try:
            types = await asyncio.to_thread(_list)
        except Exception:
            logger.warning("[aws] Failed to list EC2 instance types", exc_info=True)
            return []

        instances: list[ComputeInstance] = []
        for it in types:
            name: str = it.get("InstanceType", "")
            vcpus: int = it.get("VCpuInfo", {}).get("DefaultVCpus", 0)
            mem_mb: int = it.get("MemoryInfo", {}).get("SizeInMiB", 0)
            memory_gb = round(mem_mb / 1024, 1)

            gpu_info = it.get("GpuInfo", {})
            gpus = gpu_info.get("Gpus", [])
            gpu_count = sum(g.get("Count", 0) for g in gpus)
            gpu_type = gpus[0].get("Name") if gpus else None

            # Fallback: detect GPU family from instance type name
            if gpu_count == 0:
                family = name.split(".")[0] if "." in name else ""
                if family in _GPU_FAMILIES:
                    gpu_type = _GPU_FAMILIES[family]
                    gpu_count = 1

            inst = ComputeInstance(
                provider_id=self._provider_id,
                name=name,
                vcpus=vcpus,
                memory_gb=memory_gb,
                gpu_type=gpu_type,
                gpu_count=gpu_count,
                region=region or ctx.config.get("region", ""),
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
            session = self._session(ctx)
        except Exception:
            logger.warning("[aws] Failed to create session for regions", exc_info=True)
            return []

        def _list() -> Any:
            ec2 = session.client("ec2")
            resp = ec2.describe_regions(AllRegions=True)
            return resp.get("Regions", [])

        try:
            regions = await asyncio.to_thread(_list)
        except Exception:
            logger.warning("[aws] Failed to list regions", exc_info=True)
            return []

        return [
            ProviderRegion(
                provider_id=self._provider_id,
                id=r.get("RegionName", ""),
                name=r.get("RegionName", ""),
            )
            for r in regions
        ]
