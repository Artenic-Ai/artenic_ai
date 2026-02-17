"""Vast.ai GPU marketplace training provider.

Uses the Vast.ai REST API via ``httpx`` to search for GPU offers,
provision instances, monitor execution, and clean up resources.  No
blocking SDK is used; all network calls go through an
``httpx.AsyncClient``.
"""

from __future__ import annotations

import json
import logging
import shlex
import time
import uuid
from typing import Any

import httpx

from artenic_ai_platform.providers.base import (
    CloudJobStatus,
    InstanceType,
    JobStatus,
    TrainingSpec,
)
from artenic_ai_platform.providers.cloud_base import CloudProvider

logger = logging.getLogger(__name__)

_BASE_URL = "https://console.vast.ai/api/v0"


class _VastJobState:
    """Internal bookkeeping for a running Vast.ai instance."""

    __slots__ = (
        "created_at",
        "hourly_price",
        "instance_id",
        "spec",
    )

    def __init__(
        self,
        instance_id: int,
        created_at: float,
        spec: TrainingSpec,
        hourly_price: float | None = None,
    ) -> None:
        self.instance_id = instance_id
        self.created_at = created_at
        self.spec = spec
        self.hourly_price = hourly_price


class VastAIProvider(CloudProvider):
    """Vast.ai GPU marketplace training provider.

    Parameters
    ----------
    api_key:
        Vast.ai API key.
    docker_image:
        Docker image to run on the instance (default
        ``"pytorch/pytorch:latest"``).
    max_price_per_hour:
        Maximum acceptable price per GPU-hour in USD (default ``2.0``).
    min_reliability:
        Minimum host reliability score in [0, 1] (default ``0.95``).
    disk_gb:
        Disk space to allocate per instance in GB (default ``50``).
    """

    def __init__(
        self,
        api_key: str,
        docker_image: str = "pytorch/pytorch:latest",
        max_price_per_hour: float = 2.0,
        min_reliability: float = 0.95,
        disk_gb: int = 50,
    ) -> None:
        super().__init__()

        if not api_key:
            raise ValueError("Vast.ai API key must not be empty")
        if max_price_per_hour <= 0:
            raise ValueError("max_price_per_hour must be positive")
        if not 0.0 <= min_reliability <= 1.0:
            raise ValueError("min_reliability must be between 0.0 and 1.0")

        self._api_key = api_key
        self._docker_image = docker_image
        self._max_price_per_hour = max_price_per_hour
        self._min_reliability = min_reliability
        self._disk_gb = disk_gb

        self._client: httpx.AsyncClient | None = None
        self._jobs: dict[str, _VastJobState] = {}

    # ------------------------------------------------------------------
    # Provider identity
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "vastai"

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def _connect(self) -> None:
        """Create an ``httpx.AsyncClient`` and verify API connectivity."""
        logger.info("[vastai] Connecting to Vast.ai REST API")
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            timeout=httpx.Timeout(60.0, connect=15.0),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )

        # Lightweight connectivity check — query current user info
        try:
            resp = await self._client.get("/users/current/")
            resp.raise_for_status()
            user_data: dict[str, Any] = resp.json()
            username = user_data.get("username", "unknown")
            balance = user_data.get("balance", "N/A")
            logger.info(
                "[vastai] Connected as %s (balance: $%s)",
                username,
                balance,
            )
        except httpx.HTTPStatusError as exc:
            await self._client.aclose()
            self._client = None
            raise ConnectionError(
                f"Failed to verify Vast.ai API connectivity "
                f"(HTTP {exc.response.status_code}): {exc}"
            ) from exc
        except Exception as exc:
            await self._client.aclose()
            self._client = None
            raise ConnectionError(f"Failed to verify Vast.ai API connectivity: {exc}") from exc

    async def _disconnect(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            logger.info("[vastai] Disconnecting from Vast.ai API")
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
        """Query available GPU offers from the Vast.ai marketplace.

        Vast.ai is a GPU marketplace so results are effectively always
        GPU instances.  The ``gpu_only`` flag is accepted for interface
        compatibility but does not change behaviour.
        """
        params: dict[str, Any] = {
            "q": json.dumps(
                {
                    "verified": {"eq": True},
                    "external": {"eq": False},
                    "rentable": {"eq": True},
                    "dph_total": {"lte": self._max_price_per_hour},
                    "reliability2": {"gte": self._min_reliability},
                    "type": "on-demand",
                }
            ),
            "order": "score-",
            "limit": "100",
        }

        # Optional region filter
        if region:
            q_parsed: dict[str, Any] = json.loads(str(params["q"]))
            q_parsed["geolocation"] = {"eq": region}
            params["q"] = json.dumps(q_parsed)

        resp = await self._request("GET", "/bundles/", params=params)
        offers: list[dict[str, Any]] = resp.get("offers", [])

        instances: list[InstanceType] = []
        for offer in offers:
            gpu_name: str = offer.get("gpu_name", "unknown")
            num_gpus: int = int(offer.get("num_gpus", 1))
            dph_total: float = float(offer.get("dph_total", 0))
            cpu_cores: int = int(offer.get("cpu_cores_effective", 0))
            cpu_ram_gb: float = round(float(offer.get("cpu_ram", 0)) / 1024.0, 1)
            reliability: float = float(offer.get("reliability2", 0))
            geolocation: str | None = offer.get("geolocation")

            instances.append(
                InstanceType(
                    name=f"vast-{offer.get('id', 'unknown')}",
                    vcpus=cpu_cores,
                    memory_gb=cpu_ram_gb,
                    gpu_type=gpu_name,
                    gpu_count=num_gpus,
                    price_per_hour_eur=dph_total,
                    spot_price_per_hour_eur=None,  # Vast.ai uses on-demand bidding
                    region=geolocation or region,
                    available=reliability >= self._min_reliability,
                )
            )

        logger.info(
            "[vastai] Found %d offers (max_price=$%.2f, min_reliability=%.2f)",
            len(instances),
            self._max_price_per_hour,
            self._min_reliability,
        )
        return instances

    # ------------------------------------------------------------------
    # Code upload
    # ------------------------------------------------------------------

    async def _upload_code(self, spec: TrainingSpec) -> str:
        """Embed training code into the docker onstart script.

        Vast.ai instances run Docker containers.  Code is either pulled
        from a URI at boot (git clone, S3 sync, etc.) or the training
        command is executed directly inside the container.
        """
        training_command = spec.config.get("training_command", "python train.py")
        code_uri = spec.config.get("code_uri", "")

        if code_uri:
            logger.info(
                "[vastai] Code URI configured: %s — instance will pull at boot",
                code_uri,
            )
            return str(code_uri)

        logger.info(
            "[vastai] No code_uri — training command will be passed as onstart script: %s",
            training_command,
        )
        return f"onstart://{training_command}"

    # ------------------------------------------------------------------
    # Provisioning
    # ------------------------------------------------------------------

    async def _provision_and_start(self, spec: TrainingSpec) -> str:
        """Search for a matching offer and create an instance."""
        job_id = f"vast-{uuid.uuid4().hex[:12]}"

        docker_image = spec.config.get("docker_image", self._docker_image)
        disk_gb = int(spec.config.get("disk_gb", self._disk_gb))
        gpu_name_filter = spec.instance_type  # e.g. "RTX_4090"

        # --- Step 1: Search for matching offers ---
        search_query: dict[str, Any] = {
            "verified": {"eq": True},
            "external": {"eq": False},
            "rentable": {"eq": True},
            "dph_total": {"lte": self._max_price_per_hour},
            "reliability2": {"gte": self._min_reliability},
            "disk_space": {"gte": disk_gb},
            "type": "on-demand",
        }

        if gpu_name_filter:
            search_query["gpu_name"] = {"eq": gpu_name_filter}

        if spec.region:
            search_query["geolocation"] = {"eq": spec.region}

        params: dict[str, Any] = {
            "q": json.dumps(search_query),
            "order": "dph_total",
            "limit": "10",
        }

        logger.info(
            "[vastai] Searching for offers: gpu=%s max_price=$%.2f",
            gpu_name_filter or "any",
            self._max_price_per_hour,
        )

        search_resp = await self._request("GET", "/bundles/", params=params)
        offers: list[dict[str, Any]] = search_resp.get("offers", [])

        if not offers:
            raise RuntimeError(
                f"No Vast.ai offers found matching criteria: "
                f"gpu={gpu_name_filter!r}, max_price=${self._max_price_per_hour:.2f}, "
                f"min_reliability={self._min_reliability:.2f}, "
                f"disk_gb={disk_gb}"
            )

        # Pick the best offer (cheapest that meets requirements)
        best_offer = offers[0]
        offer_id: int = int(best_offer["id"])
        offer_price: float = float(best_offer.get("dph_total", 0))
        offer_gpu: str = best_offer.get("gpu_name", "unknown")

        logger.info(
            "[vastai] Selected offer %d: %s at $%.3f/hr",
            offer_id,
            offer_gpu,
            offer_price,
        )

        # --- Step 2: Create instance from the selected offer ---
        onstart_script = self._build_onstart_script(
            training_command=spec.config.get("training_command", "python train.py"),
            code_uri=spec.config.get("code_uri", ""),
            env_vars=spec.config.get("env", {}),
            job_id=job_id,
        )

        create_payload: dict[str, Any] = {
            "client_id": "me",
            "image": docker_image,
            "disk": disk_gb,
            "onstart": onstart_script,
            "runtype": "args",
            "label": f"artenic-{job_id}",
        }

        # Include custom docker args if specified
        docker_args = spec.config.get("docker_args")
        if docker_args:
            create_payload["args"] = docker_args

        # Pass environment variables
        env_vars = spec.config.get("env", {})
        if env_vars:
            env_str = " ".join(
                f"-e {shlex.quote(k)}={shlex.quote(str(v))}" for k, v in env_vars.items()
            )
            create_payload["extra_env"] = env_str

        create_resp = await self._request(
            "PUT",
            f"/asks/{offer_id}/",
            json_body=create_payload,
        )

        if not create_resp.get("success"):
            error_msg = create_resp.get("msg", str(create_resp))
            raise RuntimeError(
                f"Vast.ai instance creation failed for offer {offer_id}: {error_msg}"
            )

        instance_id = int(create_resp.get("new_contract", offer_id))
        logger.info(
            "[vastai] Instance %d created from offer %d (%s at $%.3f/hr)",
            instance_id,
            offer_id,
            offer_gpu,
            offer_price,
        )

        self._jobs[job_id] = _VastJobState(
            instance_id=instance_id,
            created_at=time.time(),
            spec=spec,
            hourly_price=offer_price,
        )

        return job_id

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def _poll_provider(self, provider_job_id: str) -> CloudJobStatus:
        """Query the current status of a Vast.ai instance."""
        state = self._jobs.get(provider_job_id)
        if state is None:
            return CloudJobStatus(
                provider_job_id=provider_job_id,
                status=JobStatus.FAILED,
                error=f"Unknown job {provider_job_id}",
            )

        try:
            resp = await self._request("GET", f"/instances/{state.instance_id}/")
        except Exception as exc:
            logger.error(
                "[vastai] Failed to poll instance %d: %s",
                state.instance_id,
                exc,
            )
            return CloudJobStatus(
                provider_job_id=provider_job_id,
                status=JobStatus.FAILED,
                error=f"Cannot reach Vast.ai API: {exc}",
            )

        # The instance data might be nested under "instances" or at top level
        instance_data: dict[str, Any] = resp
        if "instances" in resp:
            instances_list = resp["instances"]
            if isinstance(instances_list, list) and instances_list:
                instance_data = instances_list[0]

        actual_status: str = str(instance_data.get("actual_status", "")).lower()
        elapsed = time.time() - state.created_at

        # Map Vast.ai status to our JobStatus
        job_status = self._map_instance_status(actual_status)

        # Extract metrics
        metrics: dict[str, Any] | None = None
        gpu_util = instance_data.get("gpu_util")
        gpu_temp = instance_data.get("gpu_temp")
        if gpu_util is not None or gpu_temp is not None:
            metrics = {}
            if gpu_util is not None:
                metrics["gpu_utilization_percent"] = gpu_util
            if gpu_temp is not None:
                metrics["gpu_temperature_c"] = gpu_temp

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
        """Collect artifacts from the instance.

        Vast.ai instances are ephemeral — artifacts should be pushed to
        external storage as part of the training script.  This hook
        returns the configured artifact URI if set.
        """
        state = self._jobs.get(provider_job_id)
        if state is None:
            logger.warning(
                "[vastai] Cannot collect artifacts — unknown job %s",
                provider_job_id,
            )
            return None

        artifacts_uri = state.spec.config.get("artifacts_uri")
        if artifacts_uri:
            logger.info("[vastai] Artifacts expected at %s", artifacts_uri)
            return str(artifacts_uri)

        logger.info(
            "[vastai] No artifacts_uri configured for job %s — "
            "artifacts should be pushed by the training script",
            provider_job_id,
        )
        return None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def _cleanup_compute(self, provider_job_id: str) -> None:
        """Delete a Vast.ai instance."""
        state = self._jobs.get(provider_job_id)
        if state is None:
            logger.warning("[vastai] Cannot cleanup — unknown job %s", provider_job_id)
            return

        logger.info(
            "[vastai] Deleting instance %d for job %s",
            state.instance_id,
            provider_job_id,
        )

        try:
            await self._request("DELETE", f"/instances/{state.instance_id}/")
            logger.info(
                "[vastai] Instance %d deleted successfully",
                state.instance_id,
            )
        except httpx.HTTPStatusError as exc:
            # 404 means already deleted — treat as success
            if exc.response.status_code == 404:
                logger.info(
                    "[vastai] Instance %d already deleted",
                    state.instance_id,
                )
            else:
                logger.error(
                    "[vastai] HTTP error deleting instance %d: %s",
                    state.instance_id,
                    exc,
                )
                raise
        except Exception as exc:
            logger.error(
                "[vastai] Failed to delete instance %d: %s",
                state.instance_id,
                exc,
            )
            raise

        self._jobs.pop(provider_job_id, None)

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    async def _cancel_provider_job(self, provider_job_id: str) -> None:
        """Cancel a running job by stopping then deleting the instance."""
        state = self._jobs.get(provider_job_id)
        if state is None:
            logger.warning("[vastai] Cannot cancel — unknown job %s", provider_job_id)
            return

        logger.info(
            "[vastai] Cancelling job %s — stopping instance %d",
            provider_job_id,
            state.instance_id,
        )

        # Step 1: Stop the instance gracefully
        try:
            await self._request(
                "PUT",
                f"/instances/{state.instance_id}/",
                json_body={"status": "stopped"},
            )
            logger.info(
                "[vastai] Instance %d stop request sent",
                state.instance_id,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.info(
                    "[vastai] Instance %d already gone during cancel",
                    state.instance_id,
                )
                return
            logger.warning(
                "[vastai] Failed to stop instance %d (HTTP %d), proceeding to delete",
                state.instance_id,
                exc.response.status_code,
            )
        except Exception as exc:
            logger.warning(
                "[vastai] Failed to stop instance %d: %s, proceeding to delete",
                state.instance_id,
                exc,
            )

        # Step 2: Delete the instance
        try:
            await self._request("DELETE", f"/instances/{state.instance_id}/")
            logger.info(
                "[vastai] Instance %d deleted after cancellation",
                state.instance_id,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.info(
                    "[vastai] Instance %d already deleted",
                    state.instance_id,
                )
            else:
                logger.error(
                    "[vastai] HTTP error deleting instance %d: %s",
                    state.instance_id,
                    exc,
                )
                raise
        except Exception as exc:
            logger.error(
                "[vastai] Failed to delete instance %d during cancel: %s",
                state.instance_id,
                exc,
            )
            raise

    # ==================================================================
    # Private helpers
    # ==================================================================

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute an HTTP request against the Vast.ai REST API.

        Raises
        ------
        httpx.HTTPStatusError
            If the HTTP response indicates an error.
        RuntimeError
            If the client has not been initialised.
        """
        if self._client is None:
            raise RuntimeError("Vast.ai client is not connected — call ensure_connected() first")

        response = await self._client.request(
            method,
            path,
            params=params,
            json=json_body,
        )
        response.raise_for_status()

        # Some endpoints return empty bodies on success (DELETE)
        if not response.content:
            return {}

        result: dict[str, Any] = response.json()
        return result

    @staticmethod
    def _map_instance_status(actual_status: str) -> JobStatus:
        """Map a Vast.ai ``actual_status`` string to a ``JobStatus``."""
        if actual_status in ("running",):
            return JobStatus.RUNNING
        if actual_status in ("loading", "creating", "pulling", "starting"):
            return JobStatus.PENDING
        if actual_status in ("exited",):
            return JobStatus.COMPLETED
        if actual_status in ("error",):
            return JobStatus.FAILED
        if actual_status in ("stopped", "stopping"):
            return JobStatus.CANCELLED
        # Catch-all
        if actual_status:
            logger.warning("[vastai] Unknown instance status: %s", actual_status)
        return JobStatus.PENDING

    @staticmethod
    def _build_onstart_script(
        training_command: str,
        code_uri: str,
        env_vars: dict[str, str],
        job_id: str,
    ) -> str:
        """Build the onstart bash script for the Vast.ai instance.

        This script runs when the Docker container starts.  It pulls
        code if a URI is provided, installs requirements, and kicks off
        the training command.
        """
        parts: list[str] = [
            "#!/bin/bash",
            "set -euo pipefail",
            f'echo "artenic job {job_id} starting"',
        ]

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

        # Signal completion
        parts.append(f'echo "artenic job {job_id} finished with exit code $?"')

        return "\n".join(parts)

    @staticmethod
    def _estimate_cost(
        state: _VastJobState,
        elapsed_seconds: float,
    ) -> float | None:
        """Rough cost estimate based on elapsed time and hourly price."""
        if state.hourly_price is not None:
            hours = elapsed_seconds / 3600.0
            return round(hours * state.hourly_price, 4)
        return None
