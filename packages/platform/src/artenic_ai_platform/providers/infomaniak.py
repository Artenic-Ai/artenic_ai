"""Infomaniak Cloud training provider.

Uses the OpenStack SDK (``openstacksdk``) to provision compute instances on
Infomaniak Public Cloud, upload/download training code and artifacts via
Swift object storage, and query available flavors.  All blocking SDK calls
are dispatched via :func:`asyncio.to_thread` so the provider stays fully
async.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import shlex
import tarfile
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import openstack
import openstack.connection

from artenic_ai_platform.providers.base import (
    CloudJobStatus,
    InstanceType,
    JobStatus,
    TrainingSpec,
)
from artenic_ai_platform.providers.cloud_base import CloudProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cloud-init user-data template
# ---------------------------------------------------------------------------

_CLOUD_INIT_TEMPLATE = """\
#!/bin/bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

# ---- Signal start --------------------------------------------------------
echo '{{"event":"training_start","ts":"'$(date -u +%%Y-%%m-%%dT%%H:%%M:%%SZ)'"}}' \
    >> /var/log/artenic-training.log

# ---- Install base dependencies -------------------------------------------
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv jq > /dev/null 2>&1

# ---- Download training code ----------------------------------------------
{download_block}

# ---- Execute training -----------------------------------------------------
cd /opt/artenic-training
if [ -f requirements.txt ]; then
    python3 -m pip install --quiet -r requirements.txt
fi
{training_command}

# ---- Signal completion ----------------------------------------------------
echo '{{"event":"training_done","ts":"'$(date -u +%%Y-%%m-%%dT%%H:%%M:%%SZ)'","exit_code":'$?'}}' \
    >> /var/log/artenic-training.log

# ---- Upload artifacts -----------------------------------------------------
{upload_block}

echo '{{"event":"artifacts_uploaded","ts":"'$(date -u +%%Y-%%m-%%dT%%H:%%M:%%SZ)'"}}' \
    >> /var/log/artenic-training.log
"""

# Known GPU model identifiers that may appear in Infomaniak flavor names.
_INFOMANIAK_KNOWN_GPUS = ["a100", "v100", "t4", "l4", "l40s", "h100"]


class InfomaniakProvider(CloudProvider):
    """Infomaniak Public Cloud training provider (OpenStack-based).

    Parameters
    ----------
    auth_url:
        Keystone identity endpoint.  Defaults to the Infomaniak public
        endpoint ``https://api.pub1.infomaniak.cloud/identity/v3``.
    username:
        OpenStack username.
    password:
        OpenStack password.
    project_id:
        Infomaniak Public Cloud project ID.
    region:
        Infomaniak region code (default ``dc3-a``).
    flavor:
        Default Nova flavor name for compute instances.
    network_id:
        UUID of the Neutron network to attach instances to.
    image_id:
        UUID of the Glance image to boot from.
    container_name:
        Swift container used for code and artifact storage.
    """

    def __init__(
        self,
        auth_url: str = "https://api.pub1.infomaniak.cloud/identity/v3",
        username: str = "",
        password: str = "",
        project_id: str = "",
        region: str = "dc3-a",
        flavor: str | None = None,
        network_id: str | None = None,
        image_id: str | None = None,
        container_name: str = "artenic-training",
    ) -> None:
        super().__init__()

        self._auth_url = auth_url
        self._username = username
        self._password = password
        self._project_id = project_id
        self._region = region
        self._flavor = flavor
        self._network_id = network_id
        self._image_id = image_id
        self._container_name = container_name

        # Initialised lazily in _connect()
        self._conn: openstack.connection.Connection | None = None
        # Internal job tracking: job_id -> _InfomaniakJobState
        self._jobs: dict[str, _InfomaniakJobState] = {}

    # ------------------------------------------------------------------
    # Provider identity
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "infomaniak"

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def _connect(self) -> None:
        """Authenticate with Infomaniak Public Cloud via the OpenStack SDK."""
        logger.info(
            "[infomaniak] Connecting to Infomaniak Public Cloud (region=%s)",
            self._region,
        )

        def _do_connect() -> openstack.connection.Connection:
            return openstack.connect(
                auth_url=self._auth_url,
                project_id=self._project_id,
                username=self._username,
                password=self._password,
                region_name=self._region,
                user_domain_name="Default",
                project_domain_name="Default",
            )

        self._conn = await asyncio.to_thread(_do_connect)

        # Verify connectivity with a lightweight call
        flavors = await asyncio.to_thread(
            lambda: list(self._conn.compute.flavors())  # type: ignore[union-attr]
        )
        logger.info(
            "[infomaniak] Connected -- %d flavors available in %s",
            len(flavors),
            self._region,
        )

    async def _disconnect(self) -> None:
        """Close the OpenStack connection."""
        logger.info("[infomaniak] Disconnecting from Infomaniak Public Cloud")
        if self._conn is not None:
            await asyncio.to_thread(self._conn.close)
            self._conn = None

    # ------------------------------------------------------------------
    # Instance listing
    # ------------------------------------------------------------------

    async def _list_instances(
        self,
        region: str | None = None,
        gpu_only: bool = False,
    ) -> list[InstanceType]:
        """Query Infomaniak flavors via the OpenStack Compute API."""
        assert self._conn is not None

        flavors_raw: list[Any] = await asyncio.to_thread(
            lambda: list(self._conn.compute.flavors(details=True))  # type: ignore[union-attr]
        )

        instances: list[InstanceType] = []
        for flv in flavors_raw:
            name: str = flv.name
            vcpus: int = flv.vcpus or 0
            ram_mb: int = flv.ram or 0
            memory_gb = round(ram_mb / 1024.0, 1)

            # Detect GPU from flavor name or extra-specs
            gpu_type: str | None = None
            gpu_count = 0
            name_lower = name.lower()
            if "gpu" in name_lower:
                gpu_count = _parse_gpu_count_from_name(name)
                gpu_type = _parse_gpu_type_from_name(name)

            # Infomaniak does not expose a public pricing API at the
            # OpenStack level; pricing must be looked up externally.
            instance = InstanceType(
                name=name,
                vcpus=vcpus,
                memory_gb=memory_gb,
                gpu_type=gpu_type,
                gpu_count=gpu_count,
                price_per_hour_eur=0.0,
                spot_price_per_hour_eur=None,
                region=region or self._region,
                available=True,
            )
            instances.append(instance)

        if gpu_only:
            instances = [i for i in instances if i.gpu_count > 0]

        logger.info(
            "[infomaniak] Found %d instance types (gpu_only=%s)",
            len(instances),
            gpu_only,
        )
        return instances

    # ------------------------------------------------------------------
    # Code upload (Swift object storage)
    # ------------------------------------------------------------------

    async def _upload_code(self, spec: TrainingSpec) -> str:
        """Package and upload training code to Swift object storage."""
        assert self._conn is not None

        code_path = spec.config.get("code_path", ".")
        job_key = f"{spec.service}/{spec.model}/{uuid.uuid4().hex[:12]}"
        object_name = f"code/{job_key}/training.tar.gz"

        # Package the code into a tarball
        tar_path = await asyncio.to_thread(self._create_tarball, code_path)

        try:
            # Ensure the container exists
            await asyncio.to_thread(
                self._conn.object_store.create_container,
                self._container_name,
            )

            # Upload the tarball
            with open(tar_path, "rb") as fobj:
                data = fobj.read()

            await asyncio.to_thread(
                self._conn.object_store.upload_object,
                self._container_name,
                object_name,
                data=data,
            )

            uri = f"swift://{self._container_name}/{object_name}"
            logger.info("[infomaniak] Code uploaded to %s", uri)
            return uri
        finally:
            # Clean up temporary tarball
            if os.path.exists(tar_path):
                os.remove(tar_path)

    # ------------------------------------------------------------------
    # Provisioning
    # ------------------------------------------------------------------

    async def _provision_and_start(self, spec: TrainingSpec) -> str:
        """Create a Nova server and start training via cloud-init."""
        assert self._conn is not None

        job_id = f"infomaniak-{uuid.uuid4().hex[:12]}"
        flavor_name = spec.instance_type or self._flavor or "a2-ram4-disk20"
        image_id = spec.config.get("image_id", self._image_id)
        network_id = spec.config.get("network_id", self._network_id)

        if image_id is None:
            raise ValueError(
                "An image_id must be provided either in the constructor "
                "or in spec.config['image_id']"
            )

        # Build cloud-init user-data
        user_data = self._build_user_data(spec, job_id)
        user_data_b64 = base64.b64encode(user_data.encode()).decode()

        # Build server creation kwargs
        server_kwargs: dict[str, Any] = {
            "name": f"artenic-{job_id}",
            "flavor_id": flavor_name,
            "image_id": image_id,
            "user_data": user_data_b64,
            "metadata": {
                "artenic-job-id": job_id,
                "artenic-service": spec.service,
                "artenic-model": spec.model,
                "managed-by": "artenic-ai-platform",
            },
        }

        if network_id is not None:
            server_kwargs["networks"] = [{"uuid": network_id}]

        logger.info(
            "[infomaniak] Creating server: flavor=%s image=%s region=%s",
            flavor_name,
            image_id,
            spec.region or self._region,
        )

        server = await asyncio.to_thread(
            self._conn.compute.create_server,
            **server_kwargs,
        )

        server_id: str = server.id
        logger.info(
            "[infomaniak] Server %s created -- status=%s",
            server_id,
            server.status,
        )

        # Store job state
        self._jobs[job_id] = _InfomaniakJobState(
            server_id=server_id,
            created_at=time.time(),
            spec=spec,
        )

        return job_id

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def _poll_provider(self, provider_job_id: str) -> CloudJobStatus:
        """Check Nova server status."""
        assert self._conn is not None

        state = self._jobs.get(provider_job_id)
        if state is None:
            return CloudJobStatus(
                provider_job_id=provider_job_id,
                status=JobStatus.FAILED,
                error=f"Unknown job {provider_job_id}",
            )

        try:
            server = await asyncio.to_thread(
                self._conn.compute.get_server,
                state.server_id,
            )
        except Exception as exc:
            logger.error(
                "[infomaniak] Failed to fetch server %s: %s",
                state.server_id,
                exc,
            )
            return CloudJobStatus(
                provider_job_id=provider_job_id,
                status=JobStatus.FAILED,
                error=f"Cannot reach server: {exc}",
            )

        elapsed = time.time() - state.created_at
        os_status: str = (server.status or "UNKNOWN").upper()

        # Map OpenStack server status to our JobStatus
        if os_status == "BUILD":
            job_status = JobStatus.PENDING
        elif os_status == "ACTIVE":
            job_status = JobStatus.RUNNING
        elif os_status in ("SHUTOFF", "STOPPED"):
            job_status = JobStatus.COMPLETED
        elif os_status in ("ERROR", "DELETED"):
            job_status = JobStatus.FAILED
        else:
            job_status = JobStatus.RUNNING

        return CloudJobStatus(
            provider_job_id=provider_job_id,
            status=job_status,
            duration_seconds=elapsed,
            cost_eur=self._estimate_cost(state, elapsed),
        )

    # ------------------------------------------------------------------
    # Artifact collection (Swift download)
    # ------------------------------------------------------------------

    async def _collect_artifacts(
        self,
        provider_job_id: str,
        status: CloudJobStatus,
    ) -> str | None:
        """Download training artifacts from Swift object storage."""
        assert self._conn is not None

        state = self._jobs.get(provider_job_id)
        if state is None:
            logger.warning(
                "[infomaniak] Cannot collect artifacts -- unknown job %s",
                provider_job_id,
            )
            return None

        artifact_object = (
            f"artifacts/{state.spec.service}/{state.spec.model}/{provider_job_id}/artifacts.tar.gz"
        )

        local_dir = os.path.join(
            tempfile.gettempdir(),
            "artenic-artifacts",
            provider_job_id,
        )
        os.makedirs(local_dir, exist_ok=True)
        local_tar = os.path.join(local_dir, "artifacts.tar.gz")

        try:
            # Download from Swift
            data = await asyncio.to_thread(
                self._conn.object_store.download_object,
                artifact_object,
                container=self._container_name,
            )

            with open(local_tar, "wb") as fobj:
                fobj.write(data)

            # Extract the tarball
            await asyncio.to_thread(self._extract_tarball, local_tar, local_dir)
            logger.info("[infomaniak] Artifacts downloaded to %s", local_dir)
            return local_dir
        except Exception as exc:
            logger.warning("[infomaniak] Artifact download failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def _cleanup_compute(self, provider_job_id: str) -> None:
        """Delete the Nova server."""
        assert self._conn is not None

        state = self._jobs.get(provider_job_id)
        if state is None:
            logger.warning(
                "[infomaniak] Cannot cleanup -- unknown job %s",
                provider_job_id,
            )
            return

        try:
            logger.info("[infomaniak] Deleting server %s", state.server_id)
            await asyncio.to_thread(
                self._conn.compute.delete_server,
                state.server_id,
            )
            logger.info("[infomaniak] Server %s deleted successfully", state.server_id)
        except Exception as exc:
            logger.error(
                "[infomaniak] Failed to delete server %s: %s",
                state.server_id,
                exc,
            )
            raise

        # Remove from local tracking
        self._jobs.pop(provider_job_id, None)

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    async def _cancel_provider_job(self, provider_job_id: str) -> None:
        """Cancel a running job by stopping and deleting the server."""
        assert self._conn is not None

        state = self._jobs.get(provider_job_id)
        if state is None:
            logger.warning(
                "[infomaniak] Cannot cancel -- unknown job %s",
                provider_job_id,
            )
            return

        try:
            logger.info(
                "[infomaniak] Cancelling job %s -- stopping server %s",
                provider_job_id,
                state.server_id,
            )
            await asyncio.to_thread(
                self._conn.compute.stop_server,
                state.server_id,
            )
        except Exception as exc:
            logger.warning(
                "[infomaniak] Graceful stop failed for server %s, will force delete: %s",
                state.server_id,
                exc,
            )

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _build_user_data(self, spec: TrainingSpec, job_id: str) -> str:
        """Build the cloud-init bash script for the training run."""
        training_command = spec.config.get(
            "training_command",
            "python3 train.py",
        )

        # Download block: use swift CLI to fetch code
        download_block = (
            "mkdir -p /opt/artenic-training\n"
            "apt-get install -y -qq python3-swiftclient > /dev/null 2>&1\n"
            f"swift download {shlex.quote(self._container_name)} "
            f"--prefix code/{spec.service}/{spec.model}/ "
            f"--output-dir /opt/artenic-training/"
        )

        # Upload block: push artifacts back to Swift
        upload_block = (
            "if [ -d /opt/artenic-training/artifacts ]; then\n"
            f"    cd /opt/artenic-training/artifacts && "
            f"tar czf /tmp/artifacts.tar.gz . && "
            f"swift upload {shlex.quote(self._container_name)} "
            f"/tmp/artifacts.tar.gz "
            f"--object-name artifacts/{spec.service}/{spec.model}/"
            f"{job_id}/artifacts.tar.gz\n"
            "fi"
        )

        # Pass additional environment variables from spec.config
        env_vars = spec.config.get("env", {})
        if env_vars:
            env_block = ""
            for key, value in env_vars.items():
                env_block += f"export {shlex.quote(key)}={shlex.quote(str(value))}\n"
            training_command = env_block + training_command

        return _CLOUD_INIT_TEMPLATE.format(
            download_block=download_block,
            training_command=training_command,
            upload_block=upload_block,
        )

    @staticmethod
    def _create_tarball(code_path: str) -> str:
        """Create a gzipped tarball of the code directory.

        Returns the path to the temporary tarball.
        """
        tmp = tempfile.mktemp(suffix=".tar.gz")
        base = Path(code_path)

        with tarfile.open(tmp, "w:gz") as tar:
            if base.is_file():
                tar.add(str(base), arcname=base.name)
            else:
                for child in base.iterdir():
                    tar.add(str(child), arcname=child.name)

        return tmp

    @staticmethod
    def _extract_tarball(tar_path: str, dest_dir: str) -> None:
        """Extract a gzipped tarball into *dest_dir*."""
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=dest_dir, filter="data")

    def _estimate_cost(
        self,
        state: _InfomaniakJobState,
        elapsed_seconds: float,
    ) -> float | None:
        """Rough cost estimate based on cached hourly price."""
        if state.hourly_price is not None:
            hours = elapsed_seconds / 3600.0
            return round(hours * state.hourly_price, 4)
        return None


# ---------------------------------------------------------------------------
# Internal state tracking
# ---------------------------------------------------------------------------


class _InfomaniakJobState:
    """Tracks the state of a training job running on Infomaniak."""

    __slots__ = (
        "created_at",
        "hourly_price",
        "server_id",
        "spec",
    )

    def __init__(
        self,
        server_id: str,
        created_at: float,
        spec: TrainingSpec,
        hourly_price: float | None = None,
    ) -> None:
        self.server_id = server_id
        self.created_at = created_at
        self.spec = spec
        self.hourly_price = hourly_price


# ---------------------------------------------------------------------------
# GPU parsing helpers (Infomaniak-specific flavor naming conventions)
# ---------------------------------------------------------------------------


def _parse_gpu_count_from_name(flavor_name: str) -> int:
    """Extract GPU count from an Infomaniak flavor name.

    Looks for ``-xN`` suffixes (e.g. ``gpu-a100-x2``).
    """
    parts = flavor_name.lower().split("-")
    for part in parts:
        if part.startswith("x") and part[1:].isdigit():
            return int(part[1:])

    # Default: if the name contains "gpu", assume at least 1
    if "gpu" in flavor_name.lower():
        return 1
    return 0


def _parse_gpu_type_from_name(flavor_name: str) -> str | None:
    """Extract the GPU model from an Infomaniak flavor name."""
    name_lower = flavor_name.lower()
    for gpu in _INFOMANIAK_KNOWN_GPUS:
        if gpu in name_lower:
            return gpu.upper()
    return "GPU"  # generic fallback
