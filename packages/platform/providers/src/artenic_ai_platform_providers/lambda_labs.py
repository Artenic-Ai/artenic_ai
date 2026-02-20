"""Lambda Labs Cloud training provider.

Uses the Lambda Labs Cloud API via ``httpx`` (async HTTP client) to provision
GPU instances, upload training code via SSH/SCP, monitor execution, and
collect artifacts.

All operations are fully async — no blocking SDK calls required.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
import uuid
from dataclasses import dataclass
from typing import Any

from artenic_ai_platform_providers.base import (
    CloudJobStatus,
    InstanceType,
    JobStatus,
    TrainingSpec,
)
from artenic_ai_platform_providers.cloud_base import CloudProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import of httpx - it is an optional dependency
# ---------------------------------------------------------------------------
try:
    import httpx  # pragma: no cover

    _HTTPX_AVAILABLE = True  # pragma: no cover
except ImportError:  # pragma: no cover
    _HTTPX_AVAILABLE = False  # pragma: no cover
    httpx = None  # type: ignore[assignment]  # pragma: no cover


def _require_httpx() -> None:
    """Raise a clear error when the HTTP client library is missing."""
    if not _HTTPX_AVAILABLE:
        raise ImportError(
            "The 'httpx' package is required for LambdaLabsProvider.  "
            "Install it with:  pip install httpx"
        )


_BASE_URL = "https://cloud.lambdalabs.com/api/v1"


class LambdaLabsProvider(CloudProvider):
    """Lambda Labs Cloud training provider.

    Lambda Labs is a GPU-cloud provider — all instances include GPUs.

    Parameters
    ----------
    api_key:
        Lambda Labs API key (Bearer token).
    ssh_key_name:
        Name of a pre-registered SSH key in the Lambda Labs account,
        required for instance access and SCP-based code upload.
    instance_type:
        Default GPU instance type (default ``gpu_1x_a10``).
    region:
        Default region (default ``us-tx-3``).
    """

    def __init__(
        self,
        api_key: str,
        ssh_key_name: str,
        instance_type: str = "gpu_1x_a10",
        region: str = "us-tx-3",
    ) -> None:
        _require_httpx()
        super().__init__()

        self._api_key = api_key
        self._ssh_key_name = ssh_key_name
        self._default_instance_type = instance_type
        self._default_region = region

        # Initialised lazily in _connect()
        self._client: httpx.AsyncClient | None = None

        # Internal job tracking: job_id -> state
        self._jobs: dict[str, _LambdaJobState] = {}

    # ------------------------------------------------------------------
    # Provider identity
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "lambda_labs"

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def _connect(self) -> None:
        """Create an ``httpx.AsyncClient`` with the Lambda Labs auth header."""
        logger.info("[lambda_labs] Connecting to Lambda Labs API")
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

        # Verify connectivity by listing instance types
        resp = await self._client.get("/instance-types")
        resp.raise_for_status()
        logger.info("[lambda_labs] Connected to Lambda Labs API")

    async def _disconnect(self) -> None:
        """Close the httpx client."""
        logger.info("[lambda_labs] Disconnecting from Lambda Labs API")
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Instance listing
    # ------------------------------------------------------------------

    async def _list_instances(
        self,
        region: str | None = None,
        gpu_only: bool = False,
    ) -> list[InstanceType]:
        """Fetch available GPU instance types and pricing from Lambda Labs.

        Lambda Labs is a GPU-only provider, so the ``gpu_only`` parameter
        is effectively always True — all returned instances have GPUs.
        """
        assert self._client is not None

        resp = await self._client.get("/instance-types")
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()

        # The API returns {"data": {"instance_type_name": { ... }, ...}}
        instance_types_map: dict[str, Any] = data.get("data", {})
        instances: list[InstanceType] = []

        for type_name, type_info in instance_types_map.items():
            specs: dict[str, Any] = type_info.get("instance_type", {})
            regions_available: list[dict[str, Any]] = type_info.get(
                "regions_with_capacity_available", []
            )

            # Filter by region if specified
            available_region_names = [r.get("name", "") for r in regions_available]
            target_region = region or self._default_region
            is_available = not region or target_region in available_region_names

            # Parse GPU information from specs
            gpu_desc: str = specs.get("description", "")
            gpu_count = int(specs.get("specs", {}).get("gpus", 0))
            gpu_type = _parse_gpu_type_from_description(gpu_desc, type_name)

            # Parse resource specs
            vcpus = int(specs.get("specs", {}).get("vcpus", 0))
            ram_gb = int(specs.get("specs", {}).get("memory_gib", 0))

            # Parse pricing (Lambda returns USD cents per hour)
            price_cents = specs.get("price_cents_per_hour")
            price_per_hour = float(price_cents) / 100.0 if price_cents is not None else 0.0

            instance = InstanceType(
                name=type_name,
                vcpus=vcpus,
                memory_gb=float(ram_gb),
                gpu_type=gpu_type,
                gpu_count=gpu_count,
                price_per_hour_eur=price_per_hour,  # Lambda prices in USD
                spot_price_per_hour_eur=None,
                region=target_region if is_available else None,
                available=is_available and len(regions_available) > 0,
            )
            instances.append(instance)

        # Lambda Labs is GPU-only, but filter anyway for interface consistency
        if gpu_only:
            instances = [i for i in instances if i.gpu_count > 0]

        logger.info(
            "[lambda_labs] Found %d instance types (gpu_only=%s)",
            len(instances),
            gpu_only,
        )
        return instances

    # ------------------------------------------------------------------
    # Code upload
    # ------------------------------------------------------------------

    async def _upload_code(self, spec: TrainingSpec) -> str:
        """Defer code upload — Lambda Labs instances need SSH/SCP after boot.

        The code path is stored in the job state and will be uploaded via
        SCP once the instance is provisioned and has an IP address.
        """
        code_path = spec.config.get("code_path", "")

        if code_path:
            logger.info(
                "[lambda_labs] Code upload deferred - will SCP to instance "
                "after provisioning (path=%s)",
                code_path,
            )
        else:
            logger.info(
                "[lambda_labs] No code_path specified - code assumed pre-deployed on instance"
            )

        return f"deferred://{code_path}"

    # ------------------------------------------------------------------
    # Provisioning
    # ------------------------------------------------------------------

    async def _provision_and_start(self, spec: TrainingSpec) -> str:
        """Launch a Lambda Labs GPU instance."""
        assert self._client is not None

        job_id = f"lambda-{uuid.uuid4().hex[:12]}"
        instance_type = spec.instance_type or self._default_instance_type
        target_region = spec.region or self._default_region

        launch_payload: dict[str, Any] = {
            "region_name": target_region,
            "instance_type_name": instance_type,
            "ssh_key_names": [self._ssh_key_name],
            "name": f"artenic-{job_id}",
        }

        # Optional: attach file systems
        file_system_names = spec.config.get("file_system_names")
        if file_system_names:
            launch_payload["file_system_names"] = file_system_names

        logger.info(
            "[lambda_labs] Launching instance: type=%s region=%s",
            instance_type,
            target_region,
        )

        resp = await self._client.post(
            "/instance-operations/launch",
            json=launch_payload,
        )

        if resp.status_code not in (200, 201):
            body = resp.text
            logger.error(
                "[lambda_labs] Instance launch failed: %s %s",
                resp.status_code,
                body,
            )
            raise RuntimeError(f"Lambda Labs instance launch failed ({resp.status_code}): {body}")

        resp_data: dict[str, Any] = resp.json()
        instance_ids: list[str] = resp_data.get("data", {}).get("instance_ids", [])

        if not instance_ids:
            raise RuntimeError(
                f"Lambda Labs launch returned no instance IDs.  Response: {resp_data}"
            )

        instance_id = instance_ids[0]
        logger.info("[lambda_labs] Instance launched: id=%s", instance_id)

        # Track job state
        self._jobs[job_id] = _LambdaJobState(
            instance_id=instance_id,
            region=target_region,
            created_at=time.time(),
            spec=spec,
            instance_type=instance_type,
        )

        # If there is code to upload, wait for the instance to become active
        # and then SCP the code
        code_path = spec.config.get("code_path")
        if code_path:
            _upload_task = asyncio.ensure_future(
                self._deferred_code_upload(job_id, instance_id, code_path)
            )
            self._background_tasks = getattr(self, "_background_tasks", [])
            self._background_tasks.append(_upload_task)

        return job_id

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def _poll_provider(self, provider_job_id: str) -> CloudJobStatus:
        """Query Lambda Labs for the current instance status."""
        assert self._client is not None

        state = self._jobs.get(provider_job_id)
        if state is None:
            return CloudJobStatus(
                provider_job_id=provider_job_id,
                status=JobStatus.FAILED,
                error=f"Unknown job {provider_job_id}",
            )

        try:
            resp = await self._client.get(
                f"/instances/{state.instance_id}",
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.error(
                "[lambda_labs] Failed to fetch instance %s: %s",
                state.instance_id,
                exc,
            )
            return CloudJobStatus(
                provider_job_id=provider_job_id,
                status=JobStatus.FAILED,
                error=f"Cannot reach instance: {exc}",
            )

        resp_data: dict[str, Any] = resp.json()
        instance_data: dict[str, Any] = resp_data.get("data", {})
        ll_status: str = instance_data.get("status", "unknown")
        elapsed = time.time() - state.created_at

        # Store the IP address for later SSH access
        ip = instance_data.get("ip")
        if ip and not state.ip_address:
            state.ip_address = ip
            logger.info(
                "[lambda_labs] Instance %s has IP %s",
                state.instance_id,
                ip,
            )

        # Map Lambda Labs statuses to JobStatus
        status_map: dict[str, JobStatus] = {
            "booting": JobStatus.PENDING,
            "active": JobStatus.RUNNING,
            "unhealthy": JobStatus.FAILED,
            "terminated": JobStatus.COMPLETED,
        }
        job_status = status_map.get(ll_status, JobStatus.PENDING)

        return CloudJobStatus(
            provider_job_id=provider_job_id,
            status=job_status,
            duration_seconds=elapsed,
            cost_eur=self._estimate_cost(state, elapsed),
        )

    # ------------------------------------------------------------------
    # Artifact collection
    # ------------------------------------------------------------------

    async def _collect_artifacts(
        self,
        provider_job_id: str,
        status: CloudJobStatus,
    ) -> str | None:
        """Download training artifacts from the instance via SCP."""
        state = self._jobs.get(provider_job_id)
        if state is None:
            logger.warning(
                "[lambda_labs] Cannot collect artifacts - unknown job %s",
                provider_job_id,
            )
            return None

        if not state.ip_address:
            logger.warning(
                "[lambda_labs] No IP address for instance %s - cannot download artifacts",
                state.instance_id,
            )
            return None

        import os
        import tempfile

        local_dir = os.path.join(
            tempfile.gettempdir(),
            "artenic-artifacts",
            provider_job_id,
        )
        os.makedirs(local_dir, exist_ok=True)

        artifacts_path = state.spec.config.get("artifacts_path", "/home/ubuntu/artifacts")

        try:
            await asyncio.to_thread(
                subprocess.run,
                [
                    "scp",
                    "-o",
                    "StrictHostKeyChecking=no",
                    "-o",
                    "UserKnownHostsFile=/dev/null",
                    "-o",
                    "ConnectTimeout=15",
                    "-r",
                    f"ubuntu@{state.ip_address}:{artifacts_path}/",
                    local_dir,
                ],
                check=True,
                capture_output=True,
                timeout=300,
            )
            logger.info("[lambda_labs] Artifacts downloaded to %s", local_dir)
            return local_dir
        except Exception as exc:
            logger.warning("[lambda_labs] SCP artifact download failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def _cleanup_compute(self, provider_job_id: str) -> None:
        """Terminate the Lambda Labs instance."""
        assert self._client is not None

        state = self._jobs.get(provider_job_id)
        if state is None:
            logger.warning(
                "[lambda_labs] Cannot cleanup - unknown job %s",
                provider_job_id,
            )
            return

        logger.info(
            "[lambda_labs] Terminating instance %s for job %s",
            state.instance_id,
            provider_job_id,
        )

        try:
            resp = await self._client.post(
                "/instance-operations/terminate",
                json={"instance_ids": [state.instance_id]},
            )
            if resp.status_code in (200, 202, 204):
                logger.info("[lambda_labs] Instance %s terminated", state.instance_id)
            else:
                logger.warning(
                    "[lambda_labs] Terminate returned %d: %s",
                    resp.status_code,
                    resp.text,
                )
        except Exception as exc:
            logger.error(
                "[lambda_labs] Failed to terminate instance %s: %s",
                state.instance_id,
                exc,
            )
            raise

        # Remove from local tracking
        self._jobs.pop(provider_job_id, None)

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    async def _cancel_provider_job(self, provider_job_id: str) -> None:
        """Cancel a running job by terminating the instance.

        Lambda Labs does not have a separate "stop" action — termination
        is the only way to cancel.
        """
        assert self._client is not None

        state = self._jobs.get(provider_job_id)
        if state is None:
            logger.warning(
                "[lambda_labs] Cannot cancel - unknown job %s",
                provider_job_id,
            )
            return

        logger.info(
            "[lambda_labs] Cancelling job %s - terminating instance %s",
            provider_job_id,
            state.instance_id,
        )

        try:
            resp = await self._client.post(
                "/instance-operations/terminate",
                json={"instance_ids": [state.instance_id]},
            )
            if resp.status_code not in (200, 202, 204):
                logger.warning(
                    "[lambda_labs] Terminate returned %d: %s",
                    resp.status_code,
                    resp.text,
                )
        except Exception as exc:
            logger.error(
                "[lambda_labs] Failed to terminate instance %s: %s",
                state.instance_id,
                exc,
            )
            raise

    # ==================================================================
    # Private helpers
    # ==================================================================

    async def _deferred_code_upload(
        self,
        job_id: str,
        instance_id: str,
        code_path: str,
    ) -> None:
        """Wait for the instance to get an IP and upload code via SCP.

        This runs as a background coroutine after instance launch.
        """
        assert self._client is not None

        logger.info(
            "[lambda_labs] Waiting for instance %s to become active for code upload",
            instance_id,
        )

        ip_address: str | None = None
        deadline = time.time() + 300.0  # 5 minutes timeout

        while time.time() < deadline:
            try:
                resp = await self._client.get(f"/instances/{instance_id}")
                if resp.status_code == 200:
                    data = resp.json().get("data", {})
                    status = data.get("status", "")
                    ip_address = data.get("ip")

                    if status == "active" and ip_address:
                        break
                    if status in ("terminated", "unhealthy"):
                        logger.error(
                            "[lambda_labs] Instance %s is %s - aborting code upload",
                            instance_id,
                            status,
                        )
                        return
            except Exception as exc:
                logger.debug("[lambda_labs] Polling instance for IP failed: %s", exc)

            await asyncio.sleep(10.0)

        if not ip_address:
            logger.error(
                "[lambda_labs] Timed out waiting for instance %s IP address",
                instance_id,
            )
            return

        # Update stored IP
        state = self._jobs.get(job_id)
        if state is not None:
            state.ip_address = ip_address

        # Wait for SSH readiness
        await self._wait_for_ssh(ip_address)

        # Upload code via SCP
        logger.info(
            "[lambda_labs] Uploading code from %s to %s",
            code_path,
            ip_address,
        )

        try:
            # Ensure target directory exists
            await asyncio.to_thread(
                subprocess.run,
                [
                    "ssh",
                    "-o",
                    "StrictHostKeyChecking=no",
                    "-o",
                    "UserKnownHostsFile=/dev/null",
                    f"ubuntu@{ip_address}",
                    "mkdir -p /home/ubuntu/training",
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )

            # Upload code
            await asyncio.to_thread(
                subprocess.run,
                [
                    "scp",
                    "-o",
                    "StrictHostKeyChecking=no",
                    "-o",
                    "UserKnownHostsFile=/dev/null",
                    "-r",
                    code_path,
                    f"ubuntu@{ip_address}:/home/ubuntu/training/",
                ],
                check=True,
                capture_output=True,
                timeout=300,
            )
            logger.info("[lambda_labs] Code uploaded successfully to %s", ip_address)
        except Exception as exc:
            logger.error(
                "[lambda_labs] SCP code upload to %s failed: %s",
                ip_address,
                exc,
            )

    async def _wait_for_ssh(
        self,
        ip_address: str,
        timeout: float = 180.0,
        interval: float = 5.0,
    ) -> None:
        """Wait until the instance accepts SSH connections."""
        deadline = time.time() + timeout
        logger.info("[lambda_labs] Waiting for SSH on %s ...", ip_address)

        while time.time() < deadline:
            try:
                result = await asyncio.to_thread(
                    subprocess.run,
                    [
                        "ssh",
                        "-o",
                        "StrictHostKeyChecking=no",
                        "-o",
                        "UserKnownHostsFile=/dev/null",
                        "-o",
                        "ConnectTimeout=5",
                        "-o",
                        "BatchMode=yes",
                        f"ubuntu@{ip_address}",
                        "true",
                    ],
                    capture_output=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    logger.info("[lambda_labs] SSH ready on %s", ip_address)
                    return
            except Exception:
                pass
            await asyncio.sleep(interval)

        raise TimeoutError(f"SSH not available on {ip_address} after {timeout}s")

    def _estimate_cost(
        self,
        state: _LambdaJobState,
        elapsed_seconds: float,
    ) -> float | None:
        """Rough cost estimate based on hourly price.

        Lambda Labs bills by the hour. Accurate billing comes from the
        Lambda Labs dashboard.
        """
        if state.hourly_price is not None:
            hours = elapsed_seconds / 3600.0
            return round(hours * state.hourly_price, 4)
        return None


# ---------------------------------------------------------------------------
# Internal state tracking
# ---------------------------------------------------------------------------


@dataclass
class _LambdaJobState:
    """Tracks the state of a training job running on Lambda Labs."""

    instance_id: str
    region: str
    created_at: float
    spec: TrainingSpec
    instance_type: str
    ip_address: str | None = None
    hourly_price: float | None = None


# ---------------------------------------------------------------------------
# GPU type parsing helper
# ---------------------------------------------------------------------------


def _parse_gpu_type_from_description(
    description: str,
    type_name: str,
) -> str | None:
    """Extract GPU model from the instance description or type name.

    Examples
    --------
    >>> _parse_gpu_type_from_description("1x NVIDIA A10 (24 GB)", "gpu_1x_a10")
    'A10'
    >>> _parse_gpu_type_from_description("8x NVIDIA H100 SXM", "gpu_8x_h100_sxm5")
    'H100'
    """
    known_gpus = [
        "GH200",
        "H100",
        "H200",
        "A100",
        "A10",
        "A6000",
        "RTX 6000",
        "RTX 4090",
        "RTX 3090",
        "L40S",
        "L40",
    ]

    # Check description first (most reliable)
    description_upper = description.upper()
    for gpu in known_gpus:
        if gpu.upper() in description_upper:
            return gpu

    # Fall back to type name parsing
    name_upper = type_name.upper()
    for gpu in known_gpus:
        # Normalise for comparison (remove spaces)
        gpu_normalised = gpu.upper().replace(" ", "")
        if gpu_normalised in name_upper.replace("_", "").replace("-", ""):
            return gpu

    # Lambda Labs is GPU-only, so return a generic fallback
    return "GPU"
