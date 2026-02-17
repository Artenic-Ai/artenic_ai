"""Scaleway Cloud training provider.

Uses the Scaleway REST API via ``httpx`` (async HTTP client) to provision
compute instances, upload training code to Scaleway Object Storage
(S3-compatible), monitor execution, and collect artifacts.

All operations are fully async — no blocking SDK calls required.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from artenic_ai_platform.providers.base import (
    CloudJobStatus,
    InstanceType,
    JobStatus,
    TrainingSpec,
)
from artenic_ai_platform.providers.cloud_base import CloudProvider

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
            "The 'httpx' package is required for ScalewayProvider.  "
            "Install it with:  pip install httpx"
        )


# ---------------------------------------------------------------------------
# Known GPU instance type prefixes on Scaleway
# ---------------------------------------------------------------------------
_GPU_TYPE_PREFIXES: dict[str, str] = {
    "RENDER-S": "P100",
    "GPU-3070-S": "RTX 3070",
    "H100-1-80G": "H100",
    "H100-2-80G": "H100",
    "L4-1-24G": "L4",
    "L40S-1-48G": "L40S",
}

_BASE_URL = "https://api.scaleway.com"
_OBJECT_STORAGE_BASE = "https://s3.{zone}.scw.cloud"


class ScalewayProvider(CloudProvider):
    """Scaleway Cloud training provider.

    Parameters
    ----------
    secret_key:
        Scaleway secret key (used as API authentication token).
    access_key:
        Scaleway access key (used for Object Storage / S3-compatible API).
    project_id:
        Scaleway project identifier.
    zone:
        Scaleway availability zone (default ``fr-par-1``).
    instance_type:
        Default commercial type for instances (default ``DEV1-S``).
    image_id:
        Scaleway image UUID to use when creating instances.
    """

    def __init__(
        self,
        secret_key: str,
        access_key: str,
        project_id: str,
        zone: str = "fr-par-1",
        instance_type: str = "DEV1-S",
        image_id: str = "",
    ) -> None:
        _require_httpx()
        super().__init__()

        self._secret_key = secret_key
        self._access_key = access_key
        self._project_id = project_id
        self._zone = zone
        self._default_instance_type = instance_type
        self._image_id = image_id

        # Initialised lazily in _connect()
        self._client: httpx.AsyncClient | None = None

        # Internal job tracking: job_id -> state
        self._jobs: dict[str, _ScalewayJobState] = {}

    # ------------------------------------------------------------------
    # Provider identity
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "scaleway"

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def _connect(self) -> None:
        """Create an ``httpx.AsyncClient`` with the Scaleway auth header."""
        logger.info("[scaleway] Connecting to Scaleway API")
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            headers={
                "X-Auth-Token": self._secret_key,
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

        # Verify connectivity by listing zones
        resp = await self._client.get(
            f"/instance/v1/zones/{self._zone}/servers",
            params={"per_page": 1},
        )
        resp.raise_for_status()
        logger.info("[scaleway] Connected to Scaleway API (zone=%s)", self._zone)

    async def _disconnect(self) -> None:
        """Close the httpx client."""
        logger.info("[scaleway] Disconnecting from Scaleway API")
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
        """Fetch commercial types and pricing from the Scaleway API."""
        assert self._client is not None

        target_zone = region or self._zone
        resp = await self._client.get(
            f"/instance/v1/zones/{target_zone}/products/servers",
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()

        servers_map: dict[str, Any] = data.get("servers", {})
        instances: list[InstanceType] = []

        for name, info in servers_map.items():
            gpu_type, gpu_count = _detect_gpu(name, info)

            # Extract hourly price (Scaleway returns per-hour pricing in EUR)
            hourly_price = float(info.get("hourly_price", 0))

            # Extract resource details
            ncpus = int(info.get("ncpus", 0))
            ram_bytes = int(info.get("ram", 0))
            memory_gb = round(ram_bytes / (1024**3), 1) if ram_bytes else 0.0

            instance = InstanceType(
                name=name,
                vcpus=ncpus,
                memory_gb=memory_gb,
                gpu_type=gpu_type,
                gpu_count=gpu_count,
                price_per_hour_eur=hourly_price,
                spot_price_per_hour_eur=None,
                region=target_zone,
                available=True,
            )
            instances.append(instance)

        if gpu_only:
            instances = [i for i in instances if i.gpu_count > 0]

        logger.info(
            "[scaleway] Found %d instance types (gpu_only=%s, zone=%s)",
            len(instances),
            gpu_only,
            target_zone,
        )
        return instances

    # ------------------------------------------------------------------
    # Code upload
    # ------------------------------------------------------------------

    async def _upload_code(self, spec: TrainingSpec) -> str:
        """Upload training code to Scaleway Object Storage (S3-compatible).

        Uses pre-signed PUT requests via httpx to upload files without
        requiring boto3 or a dedicated S3 SDK.
        """
        assert self._client is not None

        code_path = spec.config.get("code_path")
        bucket = spec.config.get("s3_bucket", f"artenic-training-{self._project_id[:8]}")
        object_key = (
            f"artenic-training/{spec.service}/{spec.model}/{uuid.uuid4().hex[:12]}/code.tar.gz"
        )

        if not code_path:
            logger.info(
                "[scaleway] No code_path in spec - skipping upload "
                "(code assumed pre-deployed on image)"
            )
            return f"image://{self._image_id}"

        # Upload via Scaleway S3-compatible endpoint
        s3_region = self._zone.rsplit("-", 1)[0]  # e.g. "fr-par-1" -> "fr-par"
        s3_host = f"s3.{s3_region}.scw.cloud"
        upload_url = f"https://{s3_host}/{bucket}/{object_key}"

        try:
            from pathlib import Path

            # Read code archive or create a tarball
            code = Path(code_path)
            if code.is_file():
                content = code.read_bytes()
            else:
                import io
                import tarfile

                buf = io.BytesIO()
                with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                    tar.add(str(code), arcname=".")
                content = buf.getvalue()

            # Simple PUT to S3-compatible endpoint with Scaleway token auth
            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as s3_client:
                resp = await s3_client.put(
                    upload_url,
                    content=content,
                    headers={
                        "X-Auth-Token": self._secret_key,
                        "Content-Type": "application/gzip",
                    },
                )
                resp.raise_for_status()

            uri = f"s3://{bucket}/{object_key}"
            logger.info("[scaleway] Code uploaded to %s", uri)
            return uri

        except Exception as exc:
            logger.error("[scaleway] Code upload failed: %s", exc)
            raise RuntimeError(f"Failed to upload code to Scaleway Object Storage: {exc}") from exc

    # ------------------------------------------------------------------
    # Provisioning
    # ------------------------------------------------------------------

    async def _provision_and_start(self, spec: TrainingSpec) -> str:
        """Create a Scaleway instance and power it on."""
        assert self._client is not None

        job_id = f"scw-{uuid.uuid4().hex[:12]}"
        commercial_type = spec.instance_type or self._default_instance_type
        image_id = spec.config.get("image_id", self._image_id)
        target_zone = spec.region or self._zone

        if not image_id:
            raise ValueError(
                "An image_id is required to provision a Scaleway instance.  "
                "Pass it via the constructor or spec.config['image_id']."
            )

        # Build the server creation payload
        server_payload: dict[str, Any] = {
            "name": f"artenic-{job_id}",
            "commercial_type": commercial_type,
            "image": image_id,
            "project": self._project_id,
            "tags": [
                f"artenic-job-id:{job_id}",
                f"artenic-service:{spec.service}",
                f"artenic-model:{spec.model}",
                "managed-by:artenic-ai-platform",
            ],
        }

        logger.info(
            "[scaleway] Creating server: type=%s image=%s zone=%s",
            commercial_type,
            image_id,
            target_zone,
        )

        # Step 1: Create the server
        resp = await self._client.post(
            f"/instance/v1/zones/{target_zone}/servers",
            json=server_payload,
        )
        if resp.status_code not in (200, 201):
            body = resp.text
            logger.error("[scaleway] Server creation failed: %s %s", resp.status_code, body)
            raise RuntimeError(f"Scaleway server creation failed ({resp.status_code}): {body}")

        server_data: dict[str, Any] = resp.json().get("server", {})
        server_id: str = server_data.get("id", "")
        logger.info(
            "[scaleway] Server created: id=%s name=%s state=%s",
            server_id,
            server_data.get("name"),
            server_data.get("state"),
        )

        # Step 2: Power on the server
        action_resp = await self._client.post(
            f"/instance/v1/zones/{target_zone}/servers/{server_id}/action",
            json={"action": "poweron"},
        )
        if action_resp.status_code not in (200, 202):
            body = action_resp.text
            logger.error("[scaleway] Power-on failed: %s %s", action_resp.status_code, body)
            # Attempt cleanup of the created server
            await self._delete_server(target_zone, server_id)
            raise RuntimeError(
                f"Scaleway server power-on failed ({action_resp.status_code}): {body}"
            )

        logger.info("[scaleway] Server %s powered on", server_id)

        # Track job state
        self._jobs[job_id] = _ScalewayJobState(
            server_id=server_id,
            zone=target_zone,
            created_at=time.time(),
            spec=spec,
            commercial_type=commercial_type,
        )

        return job_id

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def _poll_provider(self, provider_job_id: str) -> CloudJobStatus:
        """Query Scaleway for the current server status."""
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
                f"/instance/v1/zones/{state.zone}/servers/{state.server_id}",
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.error(
                "[scaleway] Failed to fetch server %s: %s",
                state.server_id,
                exc,
            )
            return CloudJobStatus(
                provider_job_id=provider_job_id,
                status=JobStatus.FAILED,
                error=f"Cannot reach server: {exc}",
            )

        server_data: dict[str, Any] = resp.json().get("server", {})
        scw_state: str = server_data.get("state", "unknown")
        elapsed = time.time() - state.created_at

        # Map Scaleway server states to JobStatus
        status_map: dict[str, JobStatus] = {
            "starting": JobStatus.PENDING,
            "running": JobStatus.RUNNING,
            "stopping": JobStatus.COMPLETED,
            "stopped": JobStatus.COMPLETED,
            "stopped in place": JobStatus.COMPLETED,
            "locked": JobStatus.FAILED,
        }
        job_status = status_map.get(scw_state, JobStatus.PENDING)

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
        """Return the S3 URI where artifacts are expected.

        The training script on the instance is responsible for uploading
        artifacts to Scaleway Object Storage.  We return the expected
        URI so downstream consumers know where to look.
        """
        state = self._jobs.get(provider_job_id)
        if state is None:
            logger.warning(
                "[scaleway] Cannot collect artifacts - unknown job %s",
                provider_job_id,
            )
            return None

        bucket = state.spec.config.get(
            "s3_bucket",
            f"artenic-training-{self._project_id[:8]}",
        )
        uri = (
            f"s3://{bucket}/artenic-training/"
            f"{state.spec.service}/{state.spec.model}/"
            f"{provider_job_id}/artifacts"
        )
        logger.info("[scaleway] Artifacts expected at %s", uri)
        return uri

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def _cleanup_compute(self, provider_job_id: str) -> None:
        """Terminate and delete the Scaleway server."""
        assert self._client is not None

        state = self._jobs.get(provider_job_id)
        if state is None:
            logger.warning("[scaleway] Cannot cleanup - unknown job %s", provider_job_id)
            return

        logger.info(
            "[scaleway] Terminating server %s for job %s",
            state.server_id,
            provider_job_id,
        )

        # Step 1: Send terminate action
        try:
            action_resp = await self._client.post(
                f"/instance/v1/zones/{state.zone}/servers/{state.server_id}/action",
                json={"action": "terminate"},
            )
            if action_resp.status_code not in (200, 202, 204):
                logger.warning(
                    "[scaleway] Terminate action returned %d - attempting delete",
                    action_resp.status_code,
                )
        except Exception as exc:
            logger.warning("[scaleway] Terminate action failed: %s", exc)

        # Step 2: Delete the server resource
        await self._delete_server(state.zone, state.server_id)

        # Remove from local tracking
        self._jobs.pop(provider_job_id, None)

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    async def _cancel_provider_job(self, provider_job_id: str) -> None:
        """Cancel a running job by powering off the server."""
        assert self._client is not None

        state = self._jobs.get(provider_job_id)
        if state is None:
            logger.warning("[scaleway] Cannot cancel - unknown job %s", provider_job_id)
            return

        logger.info(
            "[scaleway] Cancelling job %s - powering off server %s",
            provider_job_id,
            state.server_id,
        )

        try:
            resp = await self._client.post(
                f"/instance/v1/zones/{state.zone}/servers/{state.server_id}/action",
                json={"action": "poweroff"},
            )
            if resp.status_code not in (200, 202):
                logger.warning(
                    "[scaleway] Power-off returned %d: %s",
                    resp.status_code,
                    resp.text,
                )
        except Exception as exc:
            logger.error(
                "[scaleway] Failed to power off server %s: %s",
                state.server_id,
                exc,
            )
            raise

    # ==================================================================
    # Private helpers
    # ==================================================================

    async def _delete_server(self, zone: str, server_id: str) -> None:
        """Delete a Scaleway server by ID."""
        assert self._client is not None

        try:
            resp = await self._client.delete(
                f"/instance/v1/zones/{zone}/servers/{server_id}",
            )
            if resp.status_code in (200, 204):
                logger.info("[scaleway] Server %s deleted", server_id)
            else:
                logger.warning(
                    "[scaleway] Server deletion returned %d: %s",
                    resp.status_code,
                    resp.text,
                )
        except Exception as exc:
            logger.error(
                "[scaleway] Failed to delete server %s: %s",
                server_id,
                exc,
            )
            raise

    def _estimate_cost(
        self,
        state: _ScalewayJobState,
        elapsed_seconds: float,
    ) -> float | None:
        """Rough cost estimate based on hourly price.

        Accurate billing comes from the Scaleway console.
        """
        if state.hourly_price is not None:
            hours = elapsed_seconds / 3600.0
            return round(hours * state.hourly_price, 4)
        return None


# ---------------------------------------------------------------------------
# Internal state tracking
# ---------------------------------------------------------------------------


@dataclass
class _ScalewayJobState:
    """Tracks the state of a training job running on Scaleway."""

    server_id: str
    zone: str
    created_at: float
    spec: TrainingSpec
    commercial_type: str
    hourly_price: float | None = None


# ---------------------------------------------------------------------------
# GPU detection helpers
# ---------------------------------------------------------------------------


def _detect_gpu(
    commercial_type: str,
    info: dict[str, Any],
) -> tuple[str | None, int]:
    """Detect GPU type and count from a Scaleway commercial type.

    Returns
    -------
    gpu_type
        Human-readable GPU model name, or None if not a GPU instance.
    gpu_count
        Number of GPUs, or 0 if not a GPU instance.
    """
    upper_name = commercial_type.upper()

    # Check against known GPU type prefixes
    for prefix, gpu_model in _GPU_TYPE_PREFIXES.items():
        if upper_name.startswith(prefix.upper()):
            # Parse count from the name (e.g. "H100-2-80G" -> 2)
            gpu_count = _parse_gpu_count_from_name(commercial_type)
            return gpu_model, max(gpu_count, 1)

    # Check if the API response includes GPU information
    gpu_info: dict[str, Any] | None = info.get("gpu")
    if gpu_info and int(gpu_info.get("count", 0)) > 0:
        return (
            str(gpu_info.get("type", "GPU")),
            int(gpu_info.get("count", 1)),
        )

    # Heuristic: look for GPU-related keywords in the name
    name_lower = commercial_type.lower()
    if any(kw in name_lower for kw in ("gpu", "render", "h100", "l4", "l40")):
        return "GPU", 1

    return None, 0


def _parse_gpu_count_from_name(name: str) -> int:
    """Extract GPU count from a commercial type name.

    Examples
    --------
    >>> _parse_gpu_count_from_name("H100-2-80G")
    2
    >>> _parse_gpu_count_from_name("GPU-3070-S")
    1
    """
    parts = name.split("-")
    for part in parts:
        # Look for a standalone digit that could be a count
        # Ignore parts that look like memory specs (e.g. "80G")
        if part.isdigit():
            count = int(part)
            if 1 <= count <= 16:
                return count
    return 1
