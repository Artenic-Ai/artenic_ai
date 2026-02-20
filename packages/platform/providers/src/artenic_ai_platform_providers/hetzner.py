"""Hetzner Cloud training provider.

Uses the ``hcloud`` Python SDK to provision dedicated and shared servers on
Hetzner Cloud, upload training code, monitor execution, and collect
artifacts.  All blocking SDK calls are dispatched via
:func:`asyncio.to_thread` so the provider stays fully async.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from artenic_ai_platform_providers.base import (
    CloudJobStatus,
    InstanceType,
    JobStatus,
    TrainingSpec,
)
from artenic_ai_platform_providers.cloud_base import CloudProvider

if TYPE_CHECKING:
    from hcloud.server_types.domain import ServerType
    from hcloud.servers.domain import Server
    from hcloud.ssh_keys.domain import SSHKey

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import of hcloud - it is an optional dependency
# ---------------------------------------------------------------------------
try:
    import hcloud  # pragma: no cover

    _HCLOUD_AVAILABLE = True  # pragma: no cover
except ImportError:
    _HCLOUD_AVAILABLE = False
    hcloud = None


def _require_hcloud() -> None:
    """Raise a clear error when the SDK is missing."""
    if not _HCLOUD_AVAILABLE:
        raise ImportError(
            "The 'hcloud' package is required for HetznerProvider.  "
            "Install it with:  pip install hcloud"
        )


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

# ---- S3 download (if configured) -----------------------------------------
{s3_download_block}

# ---- Execute training -----------------------------------------------------
cd /opt/artenic-training
if [ -f requirements.txt ]; then
    python3 -m pip install --quiet -r requirements.txt
fi
{training_command}

# ---- Signal completion ----------------------------------------------------
echo '{{"event":"training_done","ts":"'$(date -u +%%Y-%%m-%%dT%%H:%%M:%%SZ)'","exit_code":'$?'}}' \
    >> /var/log/artenic-training.log

# ---- Upload artifacts (if S3 configured) ----------------------------------
{s3_upload_block}

echo '{{"event":"artifacts_uploaded","ts":"'$(date -u +%%Y-%%m-%%dT%%H:%%M:%%SZ)'"}}' \
    >> /var/log/artenic-training.log
"""


class HetznerProvider(CloudProvider):
    """Hetzner Cloud training provider.

    Parameters
    ----------
    api_token:
        Hetzner Cloud API token.
    location:
        Hetzner datacenter location code (default ``fsn1``).
    ssh_key_name:
        Name of a pre-registered SSH key in the Hetzner project, used for
        emergency debug access and for SCP-based code upload.
    s3_endpoint:
        S3-compatible endpoint for Hetzner Object Storage (optional).
    s3_access_key:
        Access key for the S3-compatible endpoint (optional).
    s3_secret_key:
        Secret key for the S3-compatible endpoint (optional).
    s3_bucket:
        Bucket name for code and artifact storage (optional).
    """

    def __init__(
        self,
        api_token: str,
        location: str = "fsn1",
        ssh_key_name: str | None = None,
        s3_endpoint: str | None = None,
        s3_access_key: str | None = None,
        s3_secret_key: str | None = None,
        s3_bucket: str | None = None,
    ) -> None:
        _require_hcloud()
        super().__init__()

        self._api_token = api_token
        self._location = location
        self._ssh_key_name = ssh_key_name

        # S3-compatible object storage settings
        self._s3_endpoint = s3_endpoint
        self._s3_access_key = s3_access_key
        self._s3_secret_key = s3_secret_key
        self._s3_bucket = s3_bucket

        # Initialised lazily in _connect()
        self._client: hcloud.Client | None = None

        # Mapping from internal job ID to Hetzner server ID
        self._jobs: dict[str, _HetznerJobState] = {}

    # ------------------------------------------------------------------
    # Provider identity
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "hetzner"

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def _connect(self) -> None:
        """Create the ``hcloud.Client`` and verify connectivity."""
        logger.info("[hetzner] Connecting to Hetzner Cloud API")
        self._client = await asyncio.to_thread(hcloud.Client, token=self._api_token)
        # Verify token by listing datacenters (lightweight call)
        assert self._client is not None
        datacenters = await asyncio.to_thread(self._client.datacenters.get_all)
        logger.info(
            "[hetzner] Connected - %d datacenters available",
            len(datacenters),
        )

    async def _disconnect(self) -> None:
        """No persistent connection to close, but reset state."""
        logger.info("[hetzner] Disconnecting from Hetzner Cloud API")
        self._client = None

    # ------------------------------------------------------------------
    # Instance listing
    # ------------------------------------------------------------------

    async def _list_instances(
        self,
        region: str | None = None,
        gpu_only: bool = False,
    ) -> list[InstanceType]:
        """Fetch real server types and pricing from the Hetzner API."""
        assert self._client is not None

        server_types: list[ServerType] = await asyncio.to_thread(self._client.server_types.get_all)

        instances: list[InstanceType] = []
        for st in server_types:
            # Extract price for the requested location (or default location)
            target_location = region or self._location
            price_hourly = 0.0
            if st.prices:
                for price_entry in st.prices:
                    loc = price_entry.get("location", "")
                    if loc == target_location:
                        # Hetzner returns gross/net; prefer net
                        price_hourly = float(price_entry.get("price_hourly", {}).get("net", "0"))
                        break
                else:
                    # Fallback: use the first available price
                    first = st.prices[0] if st.prices else {}
                    price_hourly = float(first.get("price_hourly", {}).get("net", "0"))

            # Detect GPU from the server type description or name
            gpu_type: str | None = None
            gpu_count = 0
            description_lower = (st.description or "").lower()
            name_lower = st.name.lower()
            if "gpu" in name_lower or "gpu" in description_lower:
                gpu_count = _parse_gpu_count(st)
                gpu_type = _parse_gpu_type(st)

            instance = InstanceType(
                name=st.name,
                vcpus=st.cores,
                memory_gb=st.memory,
                gpu_type=gpu_type,
                gpu_count=gpu_count,
                price_per_hour_eur=price_hourly,
                spot_price_per_hour_eur=None,  # Hetzner has no spot market
                region=target_location,
                available=True,
            )
            instances.append(instance)

        if gpu_only:
            instances = [i for i in instances if i.gpu_count > 0]

        logger.info(
            "[hetzner] Found %d instance types (gpu_only=%s)",
            len(instances),
            gpu_only,
        )
        return instances

    # ------------------------------------------------------------------
    # Code upload
    # ------------------------------------------------------------------

    async def _upload_code(self, spec: TrainingSpec) -> str:
        """Upload training code to S3-compatible storage or via SCP."""
        code_path = spec.config.get("code_path", ".")
        job_key = f"artenic-training/{spec.service}/{spec.model}/{uuid.uuid4().hex[:12]}"

        if self._s3_configured:
            uri = await self._s3_upload_directory(code_path, job_key)
            logger.info("[hetzner] Code uploaded to %s", uri)
            return uri

        # Fallback: tar the code and SCP later during provisioning
        logger.info(
            "[hetzner] No S3 configured - code will be uploaded via SCP during provisioning"
        )
        return f"local://{code_path}"

    # ------------------------------------------------------------------
    # Provisioning
    # ------------------------------------------------------------------

    async def _provision_and_start(self, spec: TrainingSpec) -> str:
        """Create a Hetzner server and start training via cloud-init."""
        assert self._client is not None

        job_id = f"hetzner-{uuid.uuid4().hex[:12]}"
        instance_type = spec.instance_type or "cx22"
        image_name = spec.config.get("image", "ubuntu-22.04")

        # Resolve the server type
        server_type = await asyncio.to_thread(self._client.server_types.get_by_name, instance_type)
        if server_type is None:
            raise ValueError(f"Unknown Hetzner server type: {instance_type!r}")

        # Resolve the image
        image = await asyncio.to_thread(
            self._client.images.get_by_name_and_architecture,
            image_name,
            server_type.architecture,
        )
        if image is None:
            raise ValueError(
                f"Image {image_name!r} not found for architecture {server_type.architecture!r}"
            )

        # Resolve SSH key
        ssh_keys: list[SSHKey] = []
        if self._ssh_key_name:
            ssh_key = await asyncio.to_thread(self._client.ssh_keys.get_by_name, self._ssh_key_name)
            if ssh_key is not None:
                ssh_keys.append(ssh_key)
            else:
                logger.warning(
                    "[hetzner] SSH key %r not found - proceeding without",
                    self._ssh_key_name,
                )

        # Build cloud-init user-data script
        user_data = self._build_user_data(spec, job_id)

        # Resolve location
        location = await asyncio.to_thread(
            self._client.locations.get_by_name,
            spec.region or self._location,
        )

        # Labels for tracking
        labels = {
            "artenic-job-id": job_id,
            "artenic-service": spec.service,
            "artenic-model": spec.model,
            "managed-by": "artenic-ai-platform",
        }

        logger.info(
            "[hetzner] Creating server: type=%s image=%s location=%s",
            instance_type,
            image_name,
            location.name if location else self._location,
        )

        create_kwargs: dict[str, Any] = {
            "name": f"artenic-{job_id}",
            "server_type": server_type,
            "image": image,
            "ssh_keys": ssh_keys,
            "user_data": user_data,
            "labels": labels,
        }
        if location is not None:
            create_kwargs["location"] = location

        response = await asyncio.to_thread(self._client.servers.create, **create_kwargs)

        server = response.server
        logger.info(
            "[hetzner] Server %s (id=%d) created - status=%s, ip=%s",
            server.name,
            server.id,
            server.status,
            server.public_net.ipv4.ip if server.public_net.ipv4 else "N/A",
        )

        # If we need to SCP code (no S3), do it after server is ready
        code_path = spec.config.get("code_path")
        if code_path and not self._s3_configured and ssh_keys:
            # Wait for the server to be reachable
            await self._wait_for_ssh(server)
            await self._scp_upload(server, code_path)

        # Store job state
        self._jobs[job_id] = _HetznerJobState(
            server_id=server.id,
            server_name=server.name,
            created_at=time.time(),
            spec=spec,
        )

        return job_id

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def _poll_provider(self, provider_job_id: str) -> CloudJobStatus:
        """Check server status and parse training log for metrics."""
        assert self._client is not None

        state = self._jobs.get(provider_job_id)
        if state is None:
            return CloudJobStatus(
                provider_job_id=provider_job_id,
                status=JobStatus.FAILED,
                error=f"Unknown job {provider_job_id}",
            )

        # Fetch current server state
        try:
            server: Server = await asyncio.to_thread(
                self._client.servers.get_by_id, state.server_id
            )
        except Exception as exc:
            logger.error(
                "[hetzner] Failed to fetch server %d: %s",
                state.server_id,
                exc,
            )
            return CloudJobStatus(
                provider_job_id=provider_job_id,
                status=JobStatus.FAILED,
                error=f"Cannot reach server: {exc}",
            )

        if server is None:
            return CloudJobStatus(
                provider_job_id=provider_job_id,
                status=JobStatus.FAILED,
                error="Server no longer exists",
            )

        # Map Hetzner server status to our JobStatus
        elapsed = time.time() - state.created_at
        hetzner_status = server.status

        if hetzner_status == "initializing":
            job_status = JobStatus.PENDING
        elif hetzner_status in ("starting", "running"):
            # Server is running - try to read the training log
            metrics, training_done, error_msg = await self._parse_training_log(server)
            if training_done:
                job_status = JobStatus.COMPLETED if error_msg is None else JobStatus.FAILED
            else:
                job_status = JobStatus.RUNNING

            return CloudJobStatus(
                provider_job_id=provider_job_id,
                status=job_status,
                metrics=metrics,
                error=error_msg,
                duration_seconds=elapsed,
                cost_eur=self._estimate_cost(state, elapsed),
            )
        elif hetzner_status in ("stopping", "off", "deleting"):
            job_status = JobStatus.COMPLETED
        else:
            job_status = JobStatus.FAILED

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
        """Download training artifacts from the server or S3."""
        assert self._client is not None

        state = self._jobs.get(provider_job_id)
        if state is None:
            logger.warning(
                "[hetzner] Cannot collect artifacts - unknown job %s",
                provider_job_id,
            )
            return None

        # If S3 is configured, artifacts should already be uploaded by
        # the cloud-init script - just return the URI
        if self._s3_configured:
            uri = (
                f"s3://{self._s3_bucket}/artenic-training/"
                f"{state.spec.service}/{state.spec.model}/"
                f"{provider_job_id}/artifacts"
            )
            logger.info("[hetzner] Artifacts expected at %s", uri)
            return uri

        # Otherwise, SCP the artifacts directory down from the server
        try:
            server: Server = await asyncio.to_thread(
                self._client.servers.get_by_id, state.server_id
            )
        except Exception:
            logger.warning(
                "[hetzner] Server %d not reachable for artifact download",
                state.server_id,
            )
            return None

        if server is None or server.public_net.ipv4 is None:
            return None

        ip = server.public_net.ipv4.ip
        local_dir = os.path.join(
            tempfile.gettempdir(),
            "artenic-artifacts",
            provider_job_id,
        )
        os.makedirs(local_dir, exist_ok=True)

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
                    f"root@{ip}:/opt/artenic-training/artifacts/",
                    local_dir,
                ],
                check=True,
                capture_output=True,
                timeout=300,
            )
            logger.info("[hetzner] Artifacts downloaded to %s", local_dir)
            return local_dir
        except Exception as exc:
            logger.warning("[hetzner] SCP artifact download failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def _cleanup_compute(self, provider_job_id: str) -> None:
        """Delete the Hetzner server."""
        assert self._client is not None

        state = self._jobs.get(provider_job_id)
        if state is None:
            logger.warning("[hetzner] Cannot cleanup - unknown job %s", provider_job_id)
            return

        try:
            server = await asyncio.to_thread(self._client.servers.get_by_id, state.server_id)
            if server is not None:
                logger.info(
                    "[hetzner] Deleting server %s (id=%d)",
                    server.name,
                    server.id,
                )
                await asyncio.to_thread(self._client.servers.delete, server)
                logger.info(
                    "[hetzner] Server %d deleted successfully",
                    state.server_id,
                )
        except Exception as exc:
            logger.error(
                "[hetzner] Failed to delete server %d: %s",
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
        """Cancel a running job by deleting the server."""
        assert self._client is not None

        state = self._jobs.get(provider_job_id)
        if state is None:
            logger.warning("[hetzner] Cannot cancel - unknown job %s", provider_job_id)
            return

        try:
            server = await asyncio.to_thread(self._client.servers.get_by_id, state.server_id)
            if server is not None:
                logger.info(
                    "[hetzner] Cancelling job %s - shutting down server %d",
                    provider_job_id,
                    state.server_id,
                )
                # Attempt a graceful shutdown first
                try:
                    await asyncio.to_thread(self._client.servers.shutdown, server)
                except Exception:
                    # Force power-off if shutdown fails
                    await asyncio.to_thread(self._client.servers.power_off, server)
        except Exception as exc:
            logger.error(
                "[hetzner] Failed to cancel server %d: %s",
                state.server_id,
                exc,
            )
            raise

    # ==================================================================
    # Private helpers
    # ==================================================================

    @property
    def _s3_configured(self) -> bool:
        """Return True if S3 object storage is configured."""
        return bool(
            self._s3_endpoint and self._s3_access_key and self._s3_secret_key and self._s3_bucket
        )

    def _build_user_data(self, spec: TrainingSpec, job_id: str) -> str:
        """Build the cloud-init bash script for the training run."""
        training_command = spec.config.get(
            "training_command",
            "python3 train.py",
        )

        # S3 download block
        if self._s3_configured:
            s3_env = self._s3_env_exports()
            s3_download_block = (
                f"{s3_env}\n"
                f"mkdir -p /opt/artenic-training\n"
                f"apt-get install -y -qq s3cmd > /dev/null 2>&1\n"
                f"s3cmd --host={shlex.quote(self._s3_endpoint or '')} "
                f"--host-bucket='%s.{self._s3_endpoint}' "
                f"--access_key=$AWS_ACCESS_KEY_ID "
                f"--secret_key=$AWS_SECRET_ACCESS_KEY "
                f"sync s3://{self._s3_bucket}/artenic-training/"
                f"{spec.service}/{spec.model}/ "
                f"/opt/artenic-training/"
            )
        else:
            s3_download_block = (
                "# No S3 configured - code expected to be uploaded via SCP\n"
                "mkdir -p /opt/artenic-training"
            )

        # S3 upload block
        if self._s3_configured:
            s3_env = self._s3_env_exports()
            s3_upload_block = (
                f"{s3_env}\n"
                f"s3cmd --host={shlex.quote(self._s3_endpoint or '')} "
                f"--host-bucket='%s.{self._s3_endpoint}' "
                f"--access_key=$AWS_ACCESS_KEY_ID "
                f"--secret_key=$AWS_SECRET_ACCESS_KEY "
                f"sync /opt/artenic-training/artifacts/ "
                f"s3://{self._s3_bucket}/artenic-training/"
                f"{spec.service}/{spec.model}/{job_id}/artifacts/"
            )
        else:
            s3_upload_block = "# No S3 configured - artifacts stay on server"

        # Pass additional environment variables from spec.config
        env_block = ""
        env_vars = spec.config.get("env", {})
        if env_vars:
            for key, value in env_vars.items():
                env_block += f"export {shlex.quote(key)}={shlex.quote(str(value))}\n"
            training_command = env_block + training_command

        return _CLOUD_INIT_TEMPLATE.format(
            s3_download_block=s3_download_block,
            training_command=training_command,
            s3_upload_block=s3_upload_block,
        )

    def _s3_env_exports(self) -> str:
        """Return shell export lines for S3 credentials."""
        return (
            f"export AWS_ACCESS_KEY_ID={shlex.quote(self._s3_access_key or '')}\n"
            f"export AWS_SECRET_ACCESS_KEY={shlex.quote(self._s3_secret_key or '')}"
        )

    async def _wait_for_ssh(
        self,
        server: Any,
        timeout: float = 180.0,
        interval: float = 5.0,
    ) -> None:
        """Wait until the server accepts SSH connections."""
        if server.public_net.ipv4 is None:
            raise RuntimeError("Server has no public IPv4 address")

        ip = server.public_net.ipv4.ip
        deadline = time.time() + timeout
        logger.info("[hetzner] Waiting for SSH on %s ...", ip)

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
                        f"root@{ip}",
                        "true",
                    ],
                    capture_output=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    logger.info("[hetzner] SSH ready on %s", ip)
                    return
            except Exception:
                pass
            await asyncio.sleep(interval)

        raise TimeoutError(f"SSH not available on {ip} after {timeout}s")

    async def _scp_upload(self, server: Any, code_path: str) -> None:
        """Upload code directory to server via SCP."""
        if server.public_net.ipv4 is None:
            raise RuntimeError("Server has no public IPv4 address")

        ip = server.public_net.ipv4.ip
        logger.info("[hetzner] Uploading code from %s to %s", code_path, ip)

        # Ensure target directory exists
        await asyncio.to_thread(
            subprocess.run,
            [
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                f"root@{ip}",
                "mkdir -p /opt/artenic-training",
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
                f"root@{ip}:/opt/artenic-training/",
            ],
            check=True,
            capture_output=True,
            timeout=300,
        )
        logger.info("[hetzner] Code uploaded successfully")

    async def _parse_training_log(
        self,
        server: Any,
    ) -> tuple[dict[str, Any] | None, bool, str | None]:
        """SSH into the server and read the training log.

        Returns
        -------
        metrics
            Parsed metrics dict, or None.
        training_done
            True if the training_done event was found in the log.
        error
            Error message if the training exited non-zero, else None.
        """
        if server.public_net.ipv4 is None:
            return None, False, None

        ip = server.public_net.ipv4.ip

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
                    "ConnectTimeout=10",
                    "-o",
                    "BatchMode=yes",
                    f"root@{ip}",
                    "cat /var/log/artenic-training.log 2>/dev/null || true",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception:
            return None, False, None

        if result.returncode != 0:
            return None, False, None

        log_output = result.stdout.strip()
        if not log_output:
            return None, False, None

        # Parse structured JSON log lines
        metrics: dict[str, Any] = {}
        training_done = False
        error_msg: str | None = None

        for line in log_output.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if event.get("event") == "training_done":
                training_done = True
                exit_code = event.get("exit_code", 0)
                if exit_code != 0:
                    error_msg = f"Training exited with code {exit_code}"
            elif event.get("event") == "metrics":
                metrics.update(event.get("data", {}))
            elif event.get("event") == "artifacts_uploaded":
                metrics["artifacts_uploaded"] = True

        return (metrics if metrics else None), training_done, error_msg

    async def _s3_upload_directory(
        self,
        local_path: str,
        remote_key: str,
    ) -> str:
        """Upload a local directory to S3-compatible storage via boto3."""
        try:
            import boto3
        except ImportError as err:
            raise ImportError(
                "The 'boto3' package is required for S3 uploads.  "
                "Install it with:  pip install boto3"
            ) from err

        def _upload() -> str:
            s3 = boto3.client(
                "s3",
                endpoint_url=self._s3_endpoint,
                aws_access_key_id=self._s3_access_key,
                aws_secret_access_key=self._s3_secret_key,
            )

            base = Path(local_path)
            if base.is_file():
                key = f"{remote_key}/{base.name}"
                s3.upload_file(str(base), self._s3_bucket, key)
            else:
                for file_path in base.rglob("*"):
                    if file_path.is_file():
                        rel = file_path.relative_to(base)
                        key = f"{remote_key}/{rel.as_posix()}"
                        s3.upload_file(str(file_path), self._s3_bucket, key)

            return f"s3://{self._s3_bucket}/{remote_key}"

        return await asyncio.to_thread(_upload)

    def _estimate_cost(
        self,
        state: _HetznerJobState,
        elapsed_seconds: float,
    ) -> float | None:
        """Estimate the running cost based on instance type pricing.

        This is a rough estimate; accurate billing comes from the Hetzner
        invoice.
        """
        if state.spec.instance_type is None:
            return None

        # We cache pricing on the job state to avoid extra API calls
        if state.hourly_price is not None:
            hours = elapsed_seconds / 3600.0
            return round(hours * state.hourly_price, 4)

        return None


# ---------------------------------------------------------------------------
# Internal state tracking
# ---------------------------------------------------------------------------


class _HetznerJobState:
    """Tracks the state of a training job running on Hetzner Cloud."""

    __slots__ = (
        "created_at",
        "hourly_price",
        "server_id",
        "server_name",
        "spec",
    )

    def __init__(
        self,
        server_id: int,
        server_name: str,
        created_at: float,
        spec: TrainingSpec,
        hourly_price: float | None = None,
    ) -> None:
        self.server_id = server_id
        self.server_name = server_name
        self.created_at = created_at
        self.spec = spec
        self.hourly_price = hourly_price


# ---------------------------------------------------------------------------
# GPU parsing helpers
# ---------------------------------------------------------------------------


def _parse_gpu_count(server_type: Any) -> int:
    """Attempt to extract GPU count from a Hetzner ServerType.

    Hetzner does not expose a dedicated GPU count field, so we parse the
    server type name or description heuristically.
    """
    name = server_type.name.lower()
    desc = (server_type.description or "").lower()

    # Examples: "gpu-a100-80gb-x2" → 2 GPUs
    for text in (name, desc):
        for part in text.split("-"):
            if part.startswith("x") and part[1:].isdigit():
                return int(part[1:])

    # Default: if the name contains "gpu", assume at least 1
    if "gpu" in name or "gpu" in desc:
        return 1
    return 0


def _parse_gpu_type(server_type: Any) -> str | None:
    """Attempt to extract the GPU model from server type name/description."""
    name = server_type.name.lower()
    desc = (server_type.description or "").lower()

    known_gpus = ["a100", "a10", "h100", "l40s", "l40", "l4", "t4", "v100"]
    for text in (name, desc):
        for gpu in known_gpus:
            if gpu in text:
                return gpu.upper()

    return "GPU"  # generic fallback
