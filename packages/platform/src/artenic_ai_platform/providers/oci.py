"""Oracle Cloud Infrastructure (OCI) training provider.

Uses the OCI Python SDK to provision Compute instances with optional GPU
accelerators, upload training code and artifacts via Object Storage, and
monitor job lifecycle through instance metadata and status files.

All blocking SDK calls are dispatched via :func:`asyncio.to_thread` so the
provider stays fully async.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shutil
import tarfile
import tempfile
import time
import uuid
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
# Lazy import of the OCI SDK -- it is an optional dependency
# ---------------------------------------------------------------------------

try:
    import oci  # pragma: no cover
    from oci.exceptions import ServiceError  # pragma: no cover

    _HAS_OCI = True  # pragma: no cover
except ImportError:
    oci = None
    ServiceError = Exception
    _HAS_OCI = False


def _require_oci() -> None:
    """Raise a clear error when the OCI SDK is missing."""
    if not _HAS_OCI:
        raise ImportError(
            "The 'oci' package is required for OCIProvider.  Install it with:  pip install oci"
        )


# ---------------------------------------------------------------------------
# Approximate USD -> EUR conversion
# ---------------------------------------------------------------------------

_USD_TO_EUR = 0.92

# ---------------------------------------------------------------------------
# Known OCI GPU shapes and their GPU metadata
# ---------------------------------------------------------------------------

_GPU_SHAPES: dict[str, dict[str, Any]] = {
    "VM.GPU2.1": {"gpu_type": "P100", "gpu_count": 1},
    "VM.GPU3.1": {"gpu_type": "V100", "gpu_count": 1},
    "VM.GPU3.2": {"gpu_type": "V100", "gpu_count": 2},
    "VM.GPU3.4": {"gpu_type": "V100", "gpu_count": 4},
    "BM.GPU2.2": {"gpu_type": "P100", "gpu_count": 2},
    "BM.GPU3.8": {"gpu_type": "V100", "gpu_count": 8},
    "BM.GPU4.8": {"gpu_type": "A100", "gpu_count": 8},
    "BM.GPU.A10.4": {"gpu_type": "A10", "gpu_count": 4},
    "BM.GPU.A100-v2.8": {"gpu_type": "A100-80GB", "gpu_count": 8},
    "VM.GPU.A10.1": {"gpu_type": "A10", "gpu_count": 1},
    "VM.GPU.A10.2": {"gpu_type": "A10", "gpu_count": 2},
}

# ---------------------------------------------------------------------------
# Cloud-init user-data template
# ---------------------------------------------------------------------------

_CLOUD_INIT_TEMPLATE = """\
#!/bin/bash
set -euo pipefail

echo "=== Artenic AI Training Bootstrap ==="
echo "Job ID: {job_id}"

# Install OCI CLI if not present
if ! command -v oci &>/dev/null; then
    pip install oci-cli --quiet 2>/dev/null || true
fi

{env_lines}
export ARTENIC_JOB_ID="{job_id}"

# Download training code from Object Storage
mkdir -p /opt/artenic/training
cd /opt/artenic/training
oci os object get \\
    --namespace "{namespace}" \\
    --bucket-name "{bucket}" \\
    --name "{code_object}" \\
    --file code.tar.gz \\
    --auth instance_principal 2>/dev/null || true
tar xzf code.tar.gz 2>/dev/null || true

# Signal that training is starting
echo '{{"status":"running","started_at":"'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}}' > /tmp/status.json
oci os object put \\
    --namespace "{namespace}" \\
    --bucket-name "{bucket}" \\
    --name "artifacts/{job_id}/status.json" \\
    --file /tmp/status.json \\
    --force \\
    --auth instance_principal 2>/dev/null || true

# Run training
{train_command} 2>&1 | tee /opt/artenic/training.log
TRAIN_EXIT=$?

# Upload artifacts
if [ -d /opt/artenic/training/output ]; then
    for f in /opt/artenic/training/output/*; do
        [ -e "$f" ] || continue
        oci os object put \\
            --namespace "{namespace}" \\
            --bucket-name "{bucket}" \\
            --name "artifacts/{job_id}/output/$(basename $f)" \\
            --file "$f" \\
            --force \\
            --auth instance_principal 2>/dev/null || true
    done
fi

oci os object put \\
    --namespace "{namespace}" \\
    --bucket-name "{bucket}" \\
    --name "artifacts/{job_id}/training.log" \\
    --file /opt/artenic/training.log \\
    --force \\
    --auth instance_principal 2>/dev/null || true

if [ $TRAIN_EXIT -eq 0 ]; then
    STATUS="completed"
else
    STATUS="failed"
fi

FINISHED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
cat > /tmp/status.json <<EOJSON
{{"status":"$STATUS","finished_at":"$FINISHED","exit_code":$TRAIN_EXIT}}
EOJSON
oci os object put \\
    --namespace "{namespace}" \\
    --bucket-name "{bucket}" \\
    --name "artifacts/{job_id}/status.json" \\
    --file /tmp/status.json \\
    --force \\
    --auth instance_principal 2>/dev/null || true

echo "=== Training finished with exit code $TRAIN_EXIT ==="
"""


# ---------------------------------------------------------------------------
# Internal job tracking
# ---------------------------------------------------------------------------


class _OCIJobState:
    """Tracks the state of a training job running on OCI."""

    __slots__ = (
        "code_uri",
        "created_at",
        "instance_id",
        "instance_name",
        "output_prefix",
        "spec",
    )

    def __init__(
        self,
        instance_id: str,
        instance_name: str,
        created_at: float,
        spec: TrainingSpec,
        code_uri: str,
        output_prefix: str,
    ) -> None:
        self.instance_id = instance_id
        self.instance_name = instance_name
        self.created_at = created_at
        self.spec = spec
        self.code_uri = code_uri
        self.output_prefix = output_prefix


# ===========================================================================
# Provider
# ===========================================================================


class OCIProvider(CloudProvider):
    """Oracle Cloud Infrastructure training provider.

    Provisions OCI Compute instances (optionally GPU-enabled), uploads
    training code to OCI Object Storage, and monitors training via instance
    lifecycle state and status files stored in Object Storage.

    Parameters
    ----------
    compartment_id:
        OCID of the OCI compartment in which resources are created.
    config_file:
        Path to the OCI SDK config file.  Set to ``None`` to use instance
        principal authentication instead.
    config_profile:
        Profile name within the OCI config file (default ``"DEFAULT"``).
    region:
        OCI region identifier (e.g. ``"eu-frankfurt-1"``).
    bucket_name:
        Object Storage bucket for code upload and artifact storage.
    namespace:
        Object Storage namespace.  When ``None`` the provider will query
        the namespace automatically during ``_connect()``.
    shape:
        Default Compute shape when the training spec does not specify one.
    subnet_id:
        OCID of the VCN subnet in which instances are launched.
    image_id:
        OCID of the Compute image (e.g. an Oracle Linux GPU image).
    availability_domain:
        Availability domain for instance placement (e.g.
        ``"Uxxx:EU-FRANKFURT-1-AD-1"``).  When ``None`` the first AD in
        the compartment is used.
    ssh_public_key:
        Optional SSH public key injected into launched instances for debug
        access.
    """

    def __init__(
        self,
        *,
        compartment_id: str,
        config_file: str | None = "~/.oci/config",
        config_profile: str = "DEFAULT",
        region: str = "eu-frankfurt-1",
        bucket_name: str,
        namespace: str | None = None,
        shape: str = "VM.Standard.E4.Flex",
        subnet_id: str,
        image_id: str,
        availability_domain: str | None = None,
        ssh_public_key: str | None = None,
    ) -> None:
        _require_oci()
        super().__init__()

        self._compartment_id = compartment_id
        self._config_file = config_file
        self._config_profile = config_profile
        self._region = region
        self._bucket_name = bucket_name
        self._namespace = namespace
        self._shape = shape
        self._subnet_id = subnet_id
        self._image_id = image_id
        self._availability_domain = availability_domain
        self._ssh_public_key = ssh_public_key

        # SDK clients -- initialised in _connect()
        self._compute_client: Any = None
        self._object_storage_client: Any = None
        self._oci_config: dict[str, Any] = {}

        # Job tracking: provider_job_id -> _OCIJobState
        self._jobs: dict[str, _OCIJobState] = {}

    # ------------------------------------------------------------------
    # Property
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "oci"

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def _connect(self) -> None:
        """Authenticate with OCI and initialise SDK clients.

        Supports two authentication modes:

        1. **Config file** (default) -- reads ``~/.oci/config`` (or the path
           given via *config_file*).
        2. **Instance principal** -- when *config_file* is ``None``, the
           provider assumes it is running on an OCI instance and uses
           instance-principal authentication.
        """

        def _create_clients() -> tuple[Any, Any, dict[str, Any]]:
            if self._config_file is not None:
                config = oci.config.from_file(
                    file_location=self._config_file,
                    profile_name=self._config_profile,
                )
                if self._region:
                    config["region"] = self._region
                oci.config.validate_config(config)
                compute = oci.core.ComputeClient(config)
                object_storage = oci.object_storage.ObjectStorageClient(config)
            else:
                # Instance principal authentication
                signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
                config = {"region": self._region}
                compute = oci.core.ComputeClient(
                    config={},
                    signer=signer,
                )
                object_storage = oci.object_storage.ObjectStorageClient(
                    config={},
                    signer=signer,
                )

            return compute, object_storage, config

        (
            self._compute_client,
            self._object_storage_client,
            self._oci_config,
        ) = await asyncio.to_thread(_create_clients)

        # Resolve the Object Storage namespace if not provided.
        if self._namespace is None:
            self._namespace = await asyncio.to_thread(
                lambda: self._object_storage_client.get_namespace().data
            )

        logger.info(
            "[oci] Connected to OCI region=%s compartment=%s namespace=%s",
            self._region,
            self._compartment_id,
            self._namespace,
        )

    async def _disconnect(self) -> None:
        """Release SDK clients."""
        self._compute_client = None
        self._object_storage_client = None
        self._oci_config = {}
        logger.info("[oci] Disconnected from OCI")

    # ------------------------------------------------------------------
    # Instance listing & pricing
    # ------------------------------------------------------------------

    async def _list_instances(
        self,
        region: str | None = None,
        gpu_only: bool = False,
    ) -> list[InstanceType]:
        """Query OCI Compute for available shapes and approximate pricing.

        Uses ``list_shapes`` to enumerate shapes in the compartment.  GPU
        metadata is resolved from the well-known ``_GPU_SHAPES`` mapping
        as well as by inspecting the shape's ``gpu_description`` and
        ``gpus`` fields returned by the API.
        """
        target_region = region or self._region
        compartment = self._compartment_id

        def _fetch_shapes() -> list[dict[str, Any]]:
            results: list[dict[str, Any]] = []
            response = self._compute_client.list_shapes(compartment)
            for shape in response.data:
                results.append(_shape_to_dict(shape))
            # Handle pagination
            while response.has_next_page:
                response = self._compute_client.list_shapes(
                    compartment,
                    page=response.next_page,
                )
                for shape in response.data:
                    results.append(_shape_to_dict(shape))
            return results

        raw_shapes = await asyncio.to_thread(_fetch_shapes)
        logger.info(
            "[oci] Fetched %d shapes from compartment %s",
            len(raw_shapes),
            compartment,
        )

        instances: list[InstanceType] = []
        for shape_dict in raw_shapes:
            name: str = shape_dict["shape"]
            ocpus: int = shape_dict.get("ocpus", 0)
            memory_gb: float = shape_dict.get("memory_in_gbs", 0.0)

            # GPU information -- check the SDK response first, fall back
            # to the static map.
            gpu_count: int = shape_dict.get("gpus", 0)
            gpu_description: str = shape_dict.get("gpu_description", "")

            gpu_type: str | None = None
            if gpu_count > 0 and gpu_description:
                gpu_type = gpu_description
            elif name in _GPU_SHAPES:
                gpu_info = _GPU_SHAPES[name]
                gpu_type = gpu_info["gpu_type"]
                gpu_count = gpu_info["gpu_count"]
            else:
                # Heuristic: any shape name containing ".GPU" is GPU-capable
                for known_name, known_info in _GPU_SHAPES.items():
                    if name.startswith(known_name.rsplit(".", 1)[0] + "."):
                        gpu_type = known_info["gpu_type"]
                        if gpu_count == 0:
                            gpu_count = known_info["gpu_count"]
                        break

            if gpu_only and gpu_count == 0:
                continue

            # OCI does not expose a public pricing API in the SDK.
            # We approximate using the billing_type field or default to 0.
            price_eur = 0.0

            # vCPU equivalence: OCI counts OCPUs (1 OCPU = 2 vCPUs)
            vcpus = ocpus * 2

            instances.append(
                InstanceType(
                    name=name,
                    vcpus=vcpus,
                    memory_gb=round(memory_gb, 2),
                    gpu_type=gpu_type,
                    gpu_count=gpu_count,
                    price_per_hour_eur=price_eur,
                    spot_price_per_hour_eur=None,
                    region=target_region,
                    available=True,
                )
            )

        logger.info(
            "[oci] Returned %d instance types (gpu_only=%s)",
            len(instances),
            gpu_only,
        )
        return instances

    # ------------------------------------------------------------------
    # Code upload
    # ------------------------------------------------------------------

    async def _upload_code(self, spec: TrainingSpec) -> str:
        """Package the training code directory and upload to Object Storage.

        The tarball is uploaded to
        ``<bucket>/training/<service>/<model>/<short_uuid>/code.tar.gz``.

        Returns the Object Storage object name of the uploaded archive.
        """
        job_prefix = f"{spec.service}/{spec.model}/{uuid.uuid4().hex[:8]}"
        object_name = f"training/{job_prefix}/code.tar.gz"
        source_dir = spec.config.get("source_dir", ".")
        namespace = self._namespace or ""

        def _package_and_upload() -> str:
            tmp_dir = tempfile.mkdtemp(prefix="artenic_oci_")
            try:
                archive_path = os.path.join(tmp_dir, "code.tar.gz")
                with tarfile.open(archive_path, "w:gz") as tar:
                    tar.add(source_dir, arcname=".")

                with open(archive_path, "rb") as fh:
                    self._object_storage_client.put_object(
                        namespace_name=namespace,
                        bucket_name=self._bucket_name,
                        object_name=object_name,
                        put_object_body=fh,
                    )
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            return object_name

        obj_name = await asyncio.to_thread(_package_and_upload)
        uri = f"oci://{self._bucket_name}@{namespace}/{obj_name}"
        logger.info("[oci] Uploaded code to %s", uri)
        return uri

    # ------------------------------------------------------------------
    # Provision & start
    # ------------------------------------------------------------------

    async def _provision_and_start(self, spec: TrainingSpec) -> str:
        """Launch an OCI Compute instance and begin training via cloud-init.

        Returns a provider job ID (format ``oci-<short_uuid>``).
        """
        provider_job_id = f"oci-{uuid.uuid4().hex[:8]}"
        shape = spec.instance_type or self._shape
        namespace = self._namespace or ""

        # Build the cloud-init user-data script
        code_object = f"training/{spec.service}/{spec.model}/{provider_job_id}/code.tar.gz"
        user_data_script = self._build_user_data(spec, provider_job_id, namespace)
        user_data_b64 = base64.b64encode(user_data_script.encode()).decode()

        # Resolve availability domain
        availability_domain = self._availability_domain
        if availability_domain is None:
            availability_domain = await self._get_first_ad()

        def _launch() -> str:
            launch_details: dict[str, Any] = {
                "compartmentId": self._compartment_id,
                "availabilityDomain": availability_domain,
                "shape": shape,
                "displayName": f"artenic-{provider_job_id}",
                "sourceDetails": {
                    "sourceType": "image",
                    "imageId": self._image_id,
                },
                "createVnicDetails": {
                    "subnetId": self._subnet_id,
                    "assignPublicIp": True,
                },
                "metadata": {
                    "user_data": user_data_b64,
                    "artenic_job_id": provider_job_id,
                    "artenic_service": spec.service,
                    "artenic_model": spec.model,
                },
                "freeformTags": {
                    "artenic-job-id": provider_job_id,
                    "artenic-service": spec.service,
                    "artenic-model": spec.model,
                    "managed-by": "artenic-ai-platform",
                },
            }

            # Add SSH key if configured
            if self._ssh_public_key:
                launch_details["metadata"]["ssh_authorized_keys"] = self._ssh_public_key

            # Flex shapes require shape_config for OCPUs and memory
            if "Flex" in shape:
                ocpus = spec.config.get("ocpus", 4)
                memory_in_gbs = spec.config.get("memory_in_gbs", ocpus * 16)
                launch_details["shapeConfig"] = {
                    "ocpus": ocpus,
                    "memoryInGBs": memory_in_gbs,
                }

            # Use preemptible config if spot is requested
            if spec.is_spot:
                launch_details["preemptibleInstanceConfig"] = {
                    "preemptionAction": {
                        "type": "TERMINATE",
                        "preserveBootVolume": False,
                    }
                }

            response = self._compute_client.launch_instance(
                oci.core.models.LaunchInstanceDetails(**launch_details)
            )
            instance_id: str = response.data.id
            return instance_id

        instance_id = await asyncio.to_thread(_launch)

        # Store job state
        output_prefix = f"artifacts/{provider_job_id}"
        self._jobs[provider_job_id] = _OCIJobState(
            instance_id=instance_id,
            instance_name=f"artenic-{provider_job_id}",
            created_at=time.time(),
            spec=spec,
            code_uri=code_object,
            output_prefix=output_prefix,
        )

        logger.info(
            "[oci] Launched instance %s (shape=%s, spot=%s) for job %s",
            instance_id,
            shape,
            spec.is_spot,
            provider_job_id,
        )
        return provider_job_id

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def _poll_provider(self, provider_job_id: str) -> CloudJobStatus:
        """Check the OCI instance lifecycle state and Object Storage status file."""
        state = self._jobs.get(provider_job_id)
        if state is None:
            return CloudJobStatus(
                provider_job_id=provider_job_id,
                status=JobStatus.FAILED,
                error=f"No instance tracked for job {provider_job_id}",
            )

        def _get_instance() -> dict[str, Any]:
            response = self._compute_client.get_instance(state.instance_id)
            inst = response.data
            return {
                "lifecycle_state": inst.lifecycle_state,
                "time_created": str(inst.time_created) if inst.time_created else None,
            }

        try:
            instance_info = await asyncio.to_thread(_get_instance)
        except Exception as exc:
            error_str = str(exc)
            if "404" in error_str or "NotAuthorizedOrNotFound" in error_str:
                return CloudJobStatus(
                    provider_job_id=provider_job_id,
                    status=JobStatus.FAILED,
                    error="Instance no longer exists",
                )
            raise

        lifecycle_state: str = instance_info.get("lifecycle_state", "UNKNOWN")

        # Try to read the status file from Object Storage
        obj_status = await self._read_status_file(provider_job_id)

        elapsed = time.time() - state.created_at

        if obj_status:
            raw_status = obj_status.get("status", "")
            if raw_status == "completed":
                return CloudJobStatus(
                    provider_job_id=provider_job_id,
                    status=JobStatus.COMPLETED,
                    artifacts_uri=(
                        f"oci://{self._bucket_name}@{self._namespace}/{state.output_prefix}/output/"
                    ),
                    duration_seconds=elapsed,
                )
            if raw_status == "failed":
                return CloudJobStatus(
                    provider_job_id=provider_job_id,
                    status=JobStatus.FAILED,
                    error=(f"Training exited with code {obj_status.get('exit_code', '?')}"),
                    duration_seconds=elapsed,
                )
            if raw_status == "running":
                return CloudJobStatus(
                    provider_job_id=provider_job_id,
                    status=JobStatus.RUNNING,
                    duration_seconds=elapsed,
                )

        # Fall back to OCI instance lifecycle state mapping
        state_map: dict[str, JobStatus] = {
            "PROVISIONING": JobStatus.PENDING,
            "STARTING": JobStatus.PENDING,
            "RUNNING": JobStatus.RUNNING,
            "STOPPING": JobStatus.RUNNING,
            "STOPPED": JobStatus.FAILED,
            "TERMINATING": JobStatus.RUNNING,
            "TERMINATED": JobStatus.FAILED,
            "MOVING": JobStatus.RUNNING,
            "CREATING_IMAGE": JobStatus.RUNNING,
        }

        # A preemptible instance that is terminated may have been preempted
        mapped = state_map.get(lifecycle_state, JobStatus.PENDING)
        if lifecycle_state == "TERMINATED" and state.spec.is_spot:
            mapped = JobStatus.PREEMPTED

        return CloudJobStatus(
            provider_job_id=provider_job_id,
            status=mapped,
            duration_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    # Artifact collection
    # ------------------------------------------------------------------

    async def _collect_artifacts(
        self,
        provider_job_id: str,
        status: CloudJobStatus,
    ) -> str | None:
        """Download training artifacts from Object Storage.

        Returns the local directory containing downloaded artifacts, or
        ``None`` if nothing was found.
        """
        state = self._jobs.get(provider_job_id)
        if state is None:
            logger.warning(
                "[oci] No state tracked for job %s, cannot collect artifacts",
                provider_job_id,
            )
            return None

        prefix = f"{state.output_prefix}/output/"
        namespace = self._namespace or ""

        def _download() -> str | None:
            response = self._object_storage_client.list_objects(
                namespace_name=namespace,
                bucket_name=self._bucket_name,
                prefix=prefix,
            )
            objects = response.data.objects
            if not objects:
                return None

            local_dir = os.path.join(
                tempfile.gettempdir(),
                "artenic_artifacts",
                provider_job_id,
            )
            os.makedirs(local_dir, exist_ok=True)

            for obj_summary in objects:
                obj_name: str = obj_summary.name
                relative = obj_name[len(prefix) :]
                if not relative:
                    continue
                local_path = os.path.join(local_dir, relative)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                get_response = self._object_storage_client.get_object(
                    namespace_name=namespace,
                    bucket_name=self._bucket_name,
                    object_name=obj_name,
                )
                with open(local_path, "wb") as fh:
                    for chunk in get_response.data.raw.stream(1024 * 1024, decode_content=False):
                        fh.write(chunk)

            return local_dir

        local_dir = await asyncio.to_thread(_download)
        if local_dir:
            logger.info("[oci] Artifacts downloaded to %s", local_dir)
        else:
            logger.info("[oci] No artifacts found for job %s", provider_job_id)
        return local_dir

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def _cleanup_compute(self, provider_job_id: str) -> None:
        """Terminate the OCI Compute instance associated with a job."""
        state = self._jobs.pop(provider_job_id, None)
        if state is None:
            logger.debug("[oci] No instance to clean up for job %s", provider_job_id)
            return

        def _terminate() -> None:
            try:
                self._compute_client.terminate_instance(
                    state.instance_id,
                    preserve_boot_volume=False,
                )
            except ServiceError as exc:
                # Instance may already be terminated.
                if exc.status != 404:
                    raise

        await asyncio.to_thread(_terminate)
        logger.info(
            "[oci] Terminated instance %s for job %s",
            state.instance_id,
            provider_job_id,
        )

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    async def _cancel_provider_job(self, provider_job_id: str) -> None:
        """Cancel a running job by terminating the underlying instance."""
        state = self._jobs.get(provider_job_id)
        if state is None:
            logger.warning(
                "[oci] No instance found for job %s during cancellation",
                provider_job_id,
            )
            return

        def _terminate() -> None:
            try:
                self._compute_client.terminate_instance(
                    state.instance_id,
                    preserve_boot_volume=False,
                )
            except ServiceError as exc:
                if exc.status != 404:
                    raise

        await asyncio.to_thread(_terminate)
        logger.info(
            "[oci] Cancelled job %s (instance %s)",
            provider_job_id,
            state.instance_id,
        )

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _build_user_data(
        self,
        spec: TrainingSpec,
        job_id: str,
        namespace: str,
    ) -> str:
        """Generate the cloud-init user-data bash script."""
        train_command = spec.config.get("train_command", "python train.py")
        env_vars = spec.config.get("env", {})
        env_lines = "\n".join(f'export {k}="{v}"' for k, v in env_vars.items())

        code_object = f"training/{spec.service}/{spec.model}/{job_id}/code.tar.gz"

        return _CLOUD_INIT_TEMPLATE.format(
            job_id=job_id,
            namespace=namespace,
            bucket=self._bucket_name,
            code_object=code_object,
            train_command=train_command,
            env_lines=env_lines,
        )

    async def _get_first_ad(self) -> str:
        """Return the first availability domain in the compartment."""

        def _list_ads() -> str:
            response = oci.identity.IdentityClient(self._oci_config).list_availability_domains(
                self._compartment_id
            )
            if not response.data:
                raise RuntimeError(
                    f"No availability domains found in compartment {self._compartment_id}"
                )
            return str(response.data[0].name)

        return await asyncio.to_thread(_list_ads)

    async def _read_status_file(self, provider_job_id: str) -> dict[str, Any] | None:
        """Attempt to read the status.json from Object Storage."""
        namespace = self._namespace or ""
        object_name = f"artifacts/{provider_job_id}/status.json"

        def _read() -> dict[str, Any] | None:
            try:
                response = self._object_storage_client.get_object(
                    namespace_name=namespace,
                    bucket_name=self._bucket_name,
                    object_name=object_name,
                )
                body = response.data.content.decode()
                result: dict[str, Any] = json.loads(body)
                return result
            except ServiceError as exc:
                if exc.status == 404:
                    return None
                raise
            except (json.JSONDecodeError, AttributeError):
                return None

        return await asyncio.to_thread(_read)


# ======================================================================
# Module-level helpers
# ======================================================================


def _shape_to_dict(shape: Any) -> dict[str, Any]:
    """Convert an OCI Shape SDK object to a plain dict.

    Extracts the fields we care about and gracefully handles attributes
    that may be absent in older SDK versions.
    """
    result: dict[str, Any] = {
        "shape": getattr(shape, "shape", ""),
    }

    # OCPUs -- may be float for flex shapes
    ocpus = getattr(shape, "ocpus", None)
    if ocpus is not None:
        result["ocpus"] = int(ocpus)
    else:
        result["ocpus"] = 0

    # Memory
    memory = getattr(shape, "memory_in_gbs", None)
    if memory is not None:
        result["memory_in_gbs"] = float(memory)
    else:
        result["memory_in_gbs"] = 0.0

    # GPU
    gpus = getattr(shape, "gpus", None)
    if gpus is not None:
        result["gpus"] = int(gpus)
    else:
        result["gpus"] = 0

    gpu_description = getattr(shape, "gpu_description", None)
    if gpu_description is not None:
        result["gpu_description"] = str(gpu_description)
    else:
        result["gpu_description"] = ""

    return result
