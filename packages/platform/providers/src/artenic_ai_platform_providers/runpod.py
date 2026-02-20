"""RunPod GPU cloud training provider.

Uses the RunPod GraphQL API via ``httpx`` to provision on-demand GPU pods,
monitor execution, and clean up resources.  No blocking SDK is used;
all network calls go through an ``httpx.AsyncClient``.
"""

from __future__ import annotations

import logging
import shlex
import time
import uuid
from typing import Any

import httpx

from artenic_ai_platform_providers.base import (
    CloudJobStatus,
    InstanceType,
    JobStatus,
    TrainingSpec,
)
from artenic_ai_platform_providers.cloud_base import CloudProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GraphQL query / mutation fragments
# ---------------------------------------------------------------------------

_GQL_DEPLOY_POD = """
mutation podFindAndDeployOnDemand($input: PodFindAndDeployOnDemandInput!) {
  podFindAndDeployOnDemand(input: $input) {
    id
    name
    desiredStatus
    imageName
    machineId
    machine {
      gpuDisplayName
    }
  }
}
"""

_GQL_GET_POD = """
query pod($input: PodFilter!) {
  pod(input: $input) {
    id
    name
    desiredStatus
    runtime {
      uptimeInSeconds
      gpus {
        id
        gpuUtilPercent
        memoryUtilPercent
      }
    }
    latestTelemetry
    machine {
      gpuDisplayName
    }
  }
}
"""

_GQL_TERMINATE_POD = """
mutation podTerminate($input: PodTerminateInput!) {
  podTerminate(input: $input)
}
"""

_GQL_GPU_TYPES = """
query gpuTypes {
  gpuTypes {
    id
    displayName
    memoryInGb
    secureCloud
    communityCloud
    lowestPrice(input: {gpuCount: 1}) {
      minimumBidPrice
      uninterruptablePrice
    }
  }
}
"""


class _RunPodJobState:
    """Internal bookkeeping for a running RunPod pod."""

    __slots__ = (
        "created_at",
        "hourly_price",
        "pod_id",
        "spec",
    )

    def __init__(
        self,
        pod_id: str,
        created_at: float,
        spec: TrainingSpec,
        hourly_price: float | None = None,
    ) -> None:
        self.pod_id = pod_id
        self.created_at = created_at
        self.spec = spec
        self.hourly_price = hourly_price


class RunPodProvider(CloudProvider):
    """RunPod GPU cloud training provider.

    Parameters
    ----------
    api_key:
        RunPod API key.
    gpu_type:
        GPU type identifier used when deploying pods (default
        ``"NVIDIA RTX A6000"``).
    docker_image:
        Docker image to run on the pod (default
        ``"runpod/pytorch:latest"``).
    cloud_type:
        Which RunPod cloud segment to use.  One of ``"COMMUNITY"``,
        ``"SECURE"``, or ``"ALL"`` (default ``"ALL"``).
    """

    _VALID_CLOUD_TYPES: frozenset[str] = frozenset({"COMMUNITY", "SECURE", "ALL"})

    def __init__(
        self,
        api_key: str,
        gpu_type: str = "NVIDIA RTX A6000",
        docker_image: str = "runpod/pytorch:latest",
        cloud_type: str = "ALL",
    ) -> None:
        super().__init__()

        if not api_key:
            raise ValueError("RunPod API key must not be empty")
        if cloud_type not in self._VALID_CLOUD_TYPES:
            raise ValueError(
                f"cloud_type must be one of {sorted(self._VALID_CLOUD_TYPES)}, got {cloud_type!r}"
            )

        self._api_key = api_key
        self._gpu_type = gpu_type
        self._docker_image = docker_image
        self._cloud_type = cloud_type
        self._base_url = f"https://api.runpod.io/graphql?api_key={api_key}"

        self._client: httpx.AsyncClient | None = None
        self._jobs: dict[str, _RunPodJobState] = {}

    # ------------------------------------------------------------------
    # Provider identity
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "runpod"

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def _connect(self) -> None:
        """Create an ``httpx.AsyncClient`` and verify API connectivity."""
        logger.info("[runpod] Connecting to RunPod GraphQL API")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=15.0),
            headers={"Content-Type": "application/json"},
        )

        # Lightweight connectivity check — list GPU types
        try:
            resp = await self._gql({"query": _GQL_GPU_TYPES})
            gpu_types = resp.get("data", {}).get("gpuTypes") or []
            logger.info("[runpod] Connected — %d GPU types available", len(gpu_types))
        except Exception as exc:
            await self._client.aclose()
            self._client = None
            raise ConnectionError(f"Failed to verify RunPod API connectivity: {exc}") from exc

    async def _disconnect(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            logger.info("[runpod] Disconnecting from RunPod API")
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
        """Query available GPU types and pricing from RunPod.

        RunPod is a GPU-only platform so ``gpu_only`` is effectively
        always ``True``.  The *region* parameter is currently unused
        because RunPod auto-places pods, but is accepted for interface
        compatibility.
        """
        resp = await self._gql({"query": _GQL_GPU_TYPES})
        data = resp.get("data", {})
        gpu_types: list[dict[str, Any]] = data.get("gpuTypes") or []

        instances: list[InstanceType] = []
        for gt in gpu_types:
            display_name: str = gt.get("displayName", gt.get("id", "unknown"))
            memory_gb: float = float(gt.get("memoryInGb", 0))

            # Filter by cloud type
            secure_available: bool = bool(gt.get("secureCloud"))
            community_available: bool = bool(gt.get("communityCloud"))

            if self._cloud_type == "SECURE" and not secure_available:
                continue
            if self._cloud_type == "COMMUNITY" and not community_available:
                continue

            # Pricing — RunPod returns lowest price info
            lowest = gt.get("lowestPrice") or {}
            on_demand_price = float(lowest.get("uninterruptablePrice") or 0)
            spot_price = float(lowest.get("minimumBidPrice") or 0)

            instances.append(
                InstanceType(
                    name=gt.get("id", display_name),
                    vcpus=0,  # RunPod does not expose vCPU counts per GPU type
                    memory_gb=memory_gb,
                    gpu_type=display_name,
                    gpu_count=1,
                    price_per_hour_eur=on_demand_price,
                    spot_price_per_hour_eur=spot_price if spot_price > 0 else None,
                    region=region,
                    available=True,
                )
            )

        logger.info(
            "[runpod] Found %d GPU types (cloud_type=%s)",
            len(instances),
            self._cloud_type,
        )
        return instances

    # ------------------------------------------------------------------
    # Code upload
    # ------------------------------------------------------------------

    async def _upload_code(self, spec: TrainingSpec) -> str:
        """Embed training code into docker args / environment variables.

        RunPod pods are ephemeral containers so there is no separate
        "upload" step — the training command and configuration are passed
        as docker arguments when the pod is created.
        """
        training_command = spec.config.get("training_command", "python train.py")
        code_uri = spec.config.get("code_uri", "")

        if code_uri:
            logger.info(
                "[runpod] Code URI configured: %s — pod will pull at boot",
                code_uri,
            )
            return str(code_uri)

        logger.info(
            "[runpod] No code_uri — training command will be passed as docker args: %s",
            training_command,
        )
        return f"docker://{training_command}"

    # ------------------------------------------------------------------
    # Provisioning
    # ------------------------------------------------------------------

    async def _provision_and_start(self, spec: TrainingSpec) -> str:
        """Deploy an on-demand GPU pod via the RunPod GraphQL API."""
        job_id = f"runpod-{uuid.uuid4().hex[:12]}"

        gpu_type_id = spec.instance_type or self._gpu_type
        docker_image = spec.config.get("docker_image", self._docker_image)
        volume_gb = int(spec.config.get("volume_gb", 20))
        container_disk_gb = int(spec.config.get("container_disk_gb", 20))
        gpu_count = int(spec.config.get("gpu_count", 1))

        # Build the docker start command
        training_command = spec.config.get("training_command", "python train.py")
        code_uri = spec.config.get("code_uri", "")

        # Build an onstart script that pulls code (if URI given) then runs training
        docker_args = self._build_docker_args(
            training_command=training_command,
            code_uri=code_uri,
            env_vars=spec.config.get("env", {}),
            job_id=job_id,
        )

        # Cloud type filter
        cloud_type = self._cloud_type
        allowed_ids = spec.config.get("allowed_gpu_ids")

        deploy_input: dict[str, Any] = {
            "name": f"artenic-{job_id}",
            "imageName": docker_image,
            "gpuTypeId": gpu_type_id,
            "cloudType": cloud_type,
            "dockerArgs": docker_args,
            "volumeInGb": volume_gb,
            "containerDiskInGb": container_disk_gb,
            "gpuCount": gpu_count,
            "supportPublicIp": True,
        }

        # Optional: constrain to specific GPU IDs
        if allowed_ids:
            deploy_input["allowedGpuIds"] = allowed_ids

        # Environment variables
        env_list = self._build_env_list(spec.config.get("env", {}))
        if env_list:
            deploy_input["env"] = env_list

        logger.info(
            "[runpod] Deploying pod: gpu=%s image=%s cloud_type=%s gpuCount=%d",
            gpu_type_id,
            docker_image,
            cloud_type,
            gpu_count,
        )

        resp = await self._gql(
            {
                "query": _GQL_DEPLOY_POD,
                "variables": {"input": deploy_input},
            }
        )

        pod_data = resp.get("data", {}).get("podFindAndDeployOnDemand")
        if pod_data is None:
            errors = resp.get("errors", [])
            error_msg = "; ".join(e.get("message", str(e)) for e in errors) if errors else str(resp)
            raise RuntimeError(f"RunPod pod deployment failed: {error_msg}")

        pod_id: str = pod_data["id"]
        logger.info(
            "[runpod] Pod deployed: id=%s name=%s desiredStatus=%s",
            pod_id,
            pod_data.get("name"),
            pod_data.get("desiredStatus"),
        )

        self._jobs[job_id] = _RunPodJobState(
            pod_id=pod_id,
            created_at=time.time(),
            spec=spec,
        )

        return job_id

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def _poll_provider(self, provider_job_id: str) -> CloudJobStatus:
        """Query the current status of a RunPod pod."""
        state = self._jobs.get(provider_job_id)
        if state is None:
            return CloudJobStatus(
                provider_job_id=provider_job_id,
                status=JobStatus.FAILED,
                error=f"Unknown job {provider_job_id}",
            )

        try:
            resp = await self._gql(
                {
                    "query": _GQL_GET_POD,
                    "variables": {"input": {"podId": state.pod_id}},
                }
            )
        except Exception as exc:
            logger.error("[runpod] Failed to poll pod %s: %s", state.pod_id, exc)
            return CloudJobStatus(
                provider_job_id=provider_job_id,
                status=JobStatus.FAILED,
                error=f"Cannot reach RunPod API: {exc}",
            )

        pod = resp.get("data", {}).get("pod")
        if pod is None:
            return CloudJobStatus(
                provider_job_id=provider_job_id,
                status=JobStatus.FAILED,
                error="Pod no longer exists",
            )

        elapsed = time.time() - state.created_at
        desired_status: str = pod.get("desiredStatus", "").upper()
        runtime: dict[str, Any] | None = pod.get("runtime")

        # Map RunPod status to our JobStatus
        job_status = self._map_pod_status(desired_status, runtime)

        # Extract GPU metrics if available
        metrics: dict[str, Any] | None = None
        if runtime and runtime.get("gpus"):
            metrics = {
                "uptime_seconds": runtime.get("uptimeInSeconds", 0),
                "gpus": runtime["gpus"],
            }

        return CloudJobStatus(
            provider_job_id=provider_job_id,
            status=job_status,
            metrics=metrics,
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
        """Collect artifacts from the pod.

        RunPod pods are ephemeral — artifacts should be pushed to
        external storage (S3, HuggingFace, etc.) as part of the training
        script.  This hook returns the configured artifact URI if set.
        """
        state = self._jobs.get(provider_job_id)
        if state is None:
            logger.warning(
                "[runpod] Cannot collect artifacts — unknown job %s",
                provider_job_id,
            )
            return None

        artifacts_uri = state.spec.config.get("artifacts_uri")
        if artifacts_uri:
            logger.info("[runpod] Artifacts expected at %s", artifacts_uri)
            return str(artifacts_uri)

        logger.info(
            "[runpod] No artifacts_uri configured for job %s — "
            "artifacts should be pushed by the training script",
            provider_job_id,
        )
        return None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def _cleanup_compute(self, provider_job_id: str) -> None:
        """Terminate and delete a RunPod pod."""
        state = self._jobs.get(provider_job_id)
        if state is None:
            logger.warning("[runpod] Cannot cleanup — unknown job %s", provider_job_id)
            return

        logger.info(
            "[runpod] Terminating pod %s for job %s",
            state.pod_id,
            provider_job_id,
        )

        try:
            resp = await self._gql(
                {
                    "query": _GQL_TERMINATE_POD,
                    "variables": {"input": {"podId": state.pod_id}},
                }
            )
            errors = resp.get("errors")
            if errors:
                error_msgs = "; ".join(e.get("message", str(e)) for e in errors)
                logger.error(
                    "[runpod] Terminate pod %s returned errors: %s",
                    state.pod_id,
                    error_msgs,
                )
                raise RuntimeError(f"Failed to terminate pod {state.pod_id}: {error_msgs}")
            logger.info("[runpod] Pod %s terminated successfully", state.pod_id)
        except httpx.HTTPError as exc:
            logger.error(
                "[runpod] HTTP error terminating pod %s: %s",
                state.pod_id,
                exc,
            )
            raise

        self._jobs.pop(provider_job_id, None)

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    async def _cancel_provider_job(self, provider_job_id: str) -> None:
        """Cancel a running job by terminating the pod."""
        state = self._jobs.get(provider_job_id)
        if state is None:
            logger.warning("[runpod] Cannot cancel — unknown job %s", provider_job_id)
            return

        logger.info(
            "[runpod] Cancelling job %s — terminating pod %s",
            provider_job_id,
            state.pod_id,
        )

        try:
            await self._gql(
                {
                    "query": _GQL_TERMINATE_POD,
                    "variables": {"input": {"podId": state.pod_id}},
                }
            )
            logger.info("[runpod] Pod %s terminated for cancellation", state.pod_id)
        except Exception as exc:
            logger.error(
                "[runpod] Failed to terminate pod %s during cancellation: %s",
                state.pod_id,
                exc,
            )
            raise

    # ==================================================================
    # Private helpers
    # ==================================================================

    async def _gql(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute a GraphQL request against the RunPod API.

        Raises
        ------
        httpx.HTTPStatusError
            If the HTTP response status code indicates an error.
        RuntimeError
            If the client has not been initialised.
        """
        if self._client is None:
            raise RuntimeError("RunPod client is not connected — call ensure_connected() first")

        response = await self._client.post(
            self._base_url,
            json=payload,
        )
        response.raise_for_status()

        data: dict[str, Any] = response.json()

        # Log GraphQL-level errors (distinct from HTTP errors)
        if "errors" in data:
            for err in data["errors"]:
                logger.warning("[runpod] GraphQL error: %s", err.get("message"))

        return data

    @staticmethod
    def _map_pod_status(
        desired_status: str,
        runtime: dict[str, Any] | None,
    ) -> JobStatus:
        """Map RunPod pod status fields to a ``JobStatus``."""
        if desired_status in ("RUNNING",):
            if runtime is not None:
                return JobStatus.RUNNING
            # Pod is desired RUNNING but has no runtime yet — still starting
            return JobStatus.PENDING
        if desired_status in ("EXITED", "TERMINATED"):
            return JobStatus.COMPLETED
        if desired_status in ("ERROR",):
            return JobStatus.FAILED
        if desired_status in ("STOPPED",):
            return JobStatus.CANCELLED
        # Catch-all for unknown states
        if runtime is not None:
            return JobStatus.RUNNING
        return JobStatus.PENDING

    @staticmethod
    def _build_docker_args(
        training_command: str,
        code_uri: str,
        env_vars: dict[str, str],
        job_id: str,
    ) -> str:
        """Build the docker startup command string.

        If a ``code_uri`` is provided (e.g. a git repo or S3 path), the
        startup script will pull the code before running the training
        command.
        """
        parts: list[str] = ["#!/bin/bash", "set -euo pipefail"]

        # Marker for observability
        parts.append(f'echo "artenic job {job_id} starting"')

        # Export environment variables
        for key, value in env_vars.items():
            parts.append(f"export {shlex.quote(key)}={shlex.quote(str(value))}")

        # Pull code if URI is provided
        if code_uri:
            if code_uri.startswith("git://") or code_uri.startswith("https://"):
                parts.append(f"git clone {shlex.quote(code_uri)} /workspace/code")
                parts.append("cd /workspace/code")
            elif code_uri.startswith("s3://"):
                parts.append(f"aws s3 sync {shlex.quote(code_uri)} /workspace/code")
                parts.append("cd /workspace/code")

        # Install requirements if present
        parts.append("if [ -f requirements.txt ]; then pip install --quiet -r requirements.txt; fi")

        # Run the training
        parts.append(training_command)

        return " && ".join(parts)

    @staticmethod
    def _build_env_list(
        env_vars: dict[str, str],
    ) -> list[dict[str, str]]:
        """Convert a flat dict of env vars to RunPod's expected format."""
        return [{"key": key, "value": str(value)} for key, value in env_vars.items()]

    @staticmethod
    def _estimate_cost(
        state: _RunPodJobState,
        elapsed_seconds: float,
    ) -> float | None:
        """Rough cost estimate based on elapsed time and hourly price."""
        if state.hourly_price is not None:
            hours = elapsed_seconds / 3600.0
            return round(hours * state.hourly_price, 4)
        return None
