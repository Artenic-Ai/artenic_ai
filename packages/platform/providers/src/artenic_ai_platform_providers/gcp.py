"""Google Cloud Platform training provider.

Uses the Compute Engine and Cloud Storage SDKs to provision GPU-enabled VMs,
upload training code, monitor jobs, and collect artifacts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import tempfile
import time
import uuid
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
# Lazy imports for optional GCP SDK packages
# ---------------------------------------------------------------------------

try:
    from google.cloud import compute_v1  # pragma: no cover

    _HAS_COMPUTE = True  # pragma: no cover
except ImportError:
    compute_v1 = None
    _HAS_COMPUTE = False

try:
    from google.cloud import storage as gcs_storage  # pragma: no cover

    _HAS_STORAGE = True  # pragma: no cover
except ImportError:
    gcs_storage = None
    _HAS_STORAGE = False

try:
    from google.oauth2 import service_account as _sa  # pragma: no cover

    _HAS_AUTH = True  # pragma: no cover
except ImportError:  # pragma: no cover
    _sa = None  # type: ignore[assignment]  # pragma: no cover
    _HAS_AUTH = False  # pragma: no cover

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GPU_MAP: dict[str, dict[str, Any]] = {
    "nvidia-tesla-a100": {
        "gpu_type": "A100",
        "memory_gb": 40,
    },
    "nvidia-a100-80gb": {
        "gpu_type": "A100-80GB",
        "memory_gb": 80,
    },
    "nvidia-tesla-v100": {
        "gpu_type": "V100",
        "memory_gb": 16,
    },
    "nvidia-tesla-t4": {
        "gpu_type": "T4",
        "memory_gb": 16,
    },
    "nvidia-l4": {
        "gpu_type": "L4",
        "memory_gb": 24,
    },
    "nvidia-h100-80gb": {
        "gpu_type": "H100",
        "memory_gb": 80,
    },
}

_DEFAULT_BOOT_IMAGE = (
    "projects/deeplearning-platform-release/global/images/family/pytorch-latest-gpu-debian-11-py310"
)

_STARTUP_SCRIPT_TEMPLATE = """\
#!/bin/bash
set -euo pipefail

echo "===== ARTENIC TRAINING VM STARTUP ====="

# Install gsutil if not present (should be on DL images already)
command -v gsutil >/dev/null 2>&1 || apt-get install -y google-cloud-sdk

# Download the training package
mkdir -p /opt/artenic/training
gsutil -m cp -r "{code_uri}/*" /opt/artenic/training/

cd /opt/artenic/training

# Install requirements if present
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

# Run training
echo "===== TRAINING START ====="
python {entry_point} {extra_args} 2>&1 | tee /opt/artenic/training/train.log
TRAIN_EXIT=$?

# Upload results
gsutil -m cp -r /opt/artenic/training/output/* "{output_uri}/" || true
gsutil cp /opt/artenic/training/train.log "{output_uri}/train.log" || true

if [ $TRAIN_EXIT -eq 0 ]; then
    echo "===== TRAINING COMPLETED ====="
else
    echo "===== TRAINING FAILED (exit $TRAIN_EXIT) ====="
fi
"""


class GCPProvider(CloudProvider):
    """Google Cloud Platform training provider.

    Provisions Compute Engine VMs with GPU accelerators, uploads training
    code to Cloud Storage, and monitors training via serial console output.
    """

    def __init__(
        self,
        project_id: str,
        credentials_path: str | None = None,
        region: str = "europe-west1",
        zone: str = "europe-west1-b",
        bucket_name: str | None = None,
        default_machine_type: str = "n1-standard-8",
        ssh_key_path: str | None = None,
    ) -> None:
        super().__init__()
        self._project_id = project_id
        self._credentials_path = credentials_path
        self._region = region
        self._zone = zone
        self._bucket_name = bucket_name or f"artenic-training-{project_id}"
        self._default_machine_type = default_machine_type
        self._ssh_key_path = ssh_key_path

        # SDK clients (initialised in _connect)
        self._instances_client: Any = None
        self._machine_types_client: Any = None
        self._accelerator_types_client: Any = None
        self._zone_operations_client: Any = None
        self._storage_client: Any = None

        # Tracking
        self._job_metadata: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Property
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "gcp"

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def _connect(self) -> None:
        """Initialise Compute Engine and Cloud Storage clients."""
        if not _HAS_COMPUTE:
            raise ImportError(
                "google-cloud-compute is not installed. "
                "Install it with: pip install google-cloud-compute"
            )
        if not _HAS_STORAGE:
            raise ImportError(
                "google-cloud-storage is not installed. "
                "Install it with: pip install google-cloud-storage"
            )

        credentials = None
        if self._credentials_path and _HAS_AUTH:
            credentials = _sa.Credentials.from_service_account_file(  # type: ignore[no-untyped-call]
                self._credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            logger.info(
                "[gcp] Using service account credentials from %s",
                self._credentials_path,
            )
        else:
            logger.info("[gcp] Using application default credentials")

        def _init_clients() -> None:
            kwargs: dict[str, Any] = {}
            if credentials is not None:
                kwargs["credentials"] = credentials

            self._instances_client = compute_v1.InstancesClient(**kwargs)
            self._machine_types_client = compute_v1.MachineTypesClient(**kwargs)
            self._accelerator_types_client = compute_v1.AcceleratorTypesClient(
                **kwargs,
            )
            self._zone_operations_client = compute_v1.ZoneOperationsClient(**kwargs)
            self._storage_client = gcs_storage.Client(
                project=self._project_id,
                credentials=credentials,
            )

        await asyncio.to_thread(_init_clients)
        logger.info(
            "[gcp] Connected to project=%s region=%s zone=%s",
            self._project_id,
            self._region,
            self._zone,
        )

    async def _disconnect(self) -> None:
        """Close SDK transports."""
        try:
            if self._instances_client is not None:
                transport = getattr(self._instances_client, "_transport", None)
                if transport and hasattr(transport, "close"):
                    transport.close()
            if self._storage_client is not None:
                self._storage_client.close()
        except Exception:
            logger.debug("[gcp] Error closing clients", exc_info=True)
        finally:
            self._instances_client = None
            self._machine_types_client = None
            self._accelerator_types_client = None
            self._zone_operations_client = None
            self._storage_client = None
            logger.info("[gcp] Disconnected")

    # ------------------------------------------------------------------
    # Instance listing
    # ------------------------------------------------------------------

    async def _list_instances(
        self,
        region: str | None = None,
        gpu_only: bool = False,
    ) -> list[InstanceType]:
        """List available machine types and GPU accelerators in the zone."""
        zone = self._zone
        if region and region != self._region:
            # Derive zone from region (pick first zone, "-b" suffix)
            zone = f"{region}-b"

        effective_region = region or self._region

        # Fetch machine types
        def _fetch_machine_types() -> list[Any]:
            request = compute_v1.ListMachineTypesRequest(
                project=self._project_id,
                zone=zone,
            )
            results = []
            for mt in self._machine_types_client.list(request=request):
                results.append(mt)
            return results

        # Fetch accelerator types
        def _fetch_accelerator_types() -> list[Any]:
            request = compute_v1.ListAcceleratorTypesRequest(
                project=self._project_id,
                zone=zone,
            )
            results = []
            for at in self._accelerator_types_client.list(request=request):
                results.append(at)
            return results

        machine_types, accelerator_types = await asyncio.gather(
            asyncio.to_thread(_fetch_machine_types),
            asyncio.to_thread(_fetch_accelerator_types),
        )

        instances: list[InstanceType] = []

        # Build GPU accelerator map for this zone
        gpu_accelerators: list[InstanceType] = []
        for at in accelerator_types:
            acc_name: str = at.name
            gpu_info = _GPU_MAP.get(acc_name)
            if gpu_info is None:
                # Check partial matches
                for key, info in _GPU_MAP.items():
                    if key in acc_name:
                        gpu_info = info
                        break
            if gpu_info is not None:
                gpu_accelerators.append(
                    InstanceType(
                        name=acc_name,
                        vcpus=0,
                        memory_gb=0.0,
                        gpu_type=gpu_info["gpu_type"],
                        gpu_count=at.maximum_cards_per_instance
                        if hasattr(at, "maximum_cards_per_instance")
                        else 1,
                        region=effective_region,
                    )
                )

        # Map machine types to InstanceType
        if not gpu_only:
            for mt in machine_types:
                vcpus = mt.guest_cpus if hasattr(mt, "guest_cpus") else 0
                memory_mb = mt.memory_mb if hasattr(mt, "memory_mb") else 0
                instances.append(
                    InstanceType(
                        name=mt.name,
                        vcpus=vcpus,
                        memory_gb=round(memory_mb / 1024, 2),
                        region=effective_region,
                        available=not mt.deprecated if hasattr(mt, "deprecated") else True,
                    )
                )

        # Add GPU-capable combinations (A2 / G2 families have built-in GPUs)
        for mt in machine_types:
            mt_name: str = mt.name
            # A2 family: built-in A100 GPUs
            if mt_name.startswith("a2-"):
                gpu_count = 0
                gpu_type = "A100"
                if "highgpu-1g" in mt_name:
                    gpu_count = 1
                elif "highgpu-2g" in mt_name:
                    gpu_count = 2
                elif "highgpu-4g" in mt_name:
                    gpu_count = 4
                elif "highgpu-8g" in mt_name:
                    gpu_count = 8
                elif "megagpu-16g" in mt_name:
                    gpu_count = 16
                    gpu_type = "A100"
                elif "ultragpu" in mt_name:
                    gpu_count = 8
                    gpu_type = "A100-80GB"
                if gpu_count > 0:
                    vcpus = mt.guest_cpus if hasattr(mt, "guest_cpus") else 0
                    memory_mb = mt.memory_mb if hasattr(mt, "memory_mb") else 0
                    instances.append(
                        InstanceType(
                            name=mt_name,
                            vcpus=vcpus,
                            memory_gb=round(memory_mb / 1024, 2),
                            gpu_type=gpu_type,
                            gpu_count=gpu_count,
                            region=effective_region,
                            available=True,
                        )
                    )
            # G2 family: built-in L4 GPUs
            elif mt_name.startswith("g2-"):
                gpu_count = 0
                # Extract the numeric suffix (e.g. "g2-standard-48" → "48")
                suffix = mt_name.rsplit("-", 1)[-1]
                if suffix in ("4", "8", "12", "16", "32"):
                    gpu_count = 1
                elif suffix == "24":
                    gpu_count = 2
                elif suffix == "48":
                    gpu_count = 4
                elif suffix == "96":
                    gpu_count = 8
                if gpu_count > 0:
                    vcpus = mt.guest_cpus if hasattr(mt, "guest_cpus") else 0
                    memory_mb = mt.memory_mb if hasattr(mt, "memory_mb") else 0
                    instances.append(
                        InstanceType(
                            name=mt_name,
                            vcpus=vcpus,
                            memory_gb=round(memory_mb / 1024, 2),
                            gpu_type="L4",
                            gpu_count=gpu_count,
                            region=effective_region,
                            available=True,
                        )
                    )

        # Also add standalone GPU accelerator entries
        instances.extend(gpu_accelerators)

        if gpu_only:
            instances = [i for i in instances if i.gpu_count > 0]

        logger.info(
            "[gcp] Listed %d instance types in %s (gpu_only=%s)",
            len(instances),
            effective_region,
            gpu_only,
        )
        return instances

    # ------------------------------------------------------------------
    # Upload code to GCS
    # ------------------------------------------------------------------

    async def _upload_code(self, spec: TrainingSpec) -> str:
        """Upload training code to a GCS bucket. Returns ``gs://`` URI."""
        job_id = spec.config.get("job_id", uuid.uuid4().hex[:12])
        code_path = spec.config.get("code_path", ".")
        prefix = f"training/{spec.service}/{spec.model}/{job_id}/code"

        def _upload() -> str:
            bucket = self._storage_client.bucket(self._bucket_name)

            # Create bucket if it doesn't exist
            if not bucket.exists():
                logger.info("[gcp] Creating bucket %s", self._bucket_name)
                bucket = self._storage_client.create_bucket(
                    self._bucket_name,
                    location=self._region,
                )

            source = pathlib.Path(code_path)
            if source.is_dir():
                for file_path in source.rglob("*"):
                    if file_path.is_file() and not _should_skip(file_path):
                        relative = file_path.relative_to(source)
                        blob_name = f"{prefix}/{relative}"
                        blob = bucket.blob(blob_name)
                        blob.upload_from_filename(str(file_path))
                        logger.debug("[gcp] Uploaded %s -> %s", file_path, blob_name)
            elif source.is_file():
                blob = bucket.blob(f"{prefix}/{source.name}")
                blob.upload_from_filename(str(source))
            else:
                raise FileNotFoundError(f"Code path not found: {code_path}")

            return f"gs://{self._bucket_name}/{prefix}"

        code_uri = await asyncio.to_thread(_upload)
        logger.info("[gcp] Uploaded training code to %s", code_uri)

        # Store for later use
        self._job_metadata.setdefault(job_id, {})["code_uri"] = code_uri
        self._job_metadata[job_id]["spec"] = spec

        return code_uri

    # ------------------------------------------------------------------
    # Provision and start VM
    # ------------------------------------------------------------------

    async def _provision_and_start(self, spec: TrainingSpec) -> str:
        """Create a Compute Engine VM with optional GPU and start training."""
        job_id = spec.config.get("job_id", uuid.uuid4().hex[:12])
        instance_name = f"artenic-train-{job_id}"
        machine_type = spec.instance_type or self._default_machine_type
        zone = spec.config.get("zone", self._zone)

        metadata = self._job_metadata.get(job_id, {})
        code_uri = metadata.get(
            "code_uri",
            f"gs://{self._bucket_name}/training/{spec.service}/{spec.model}/{job_id}/code",
        )
        output_uri = (
            f"gs://{self._bucket_name}/training/{spec.service}/{spec.model}/{job_id}/output"
        )

        entry_point = spec.config.get("entry_point", "train.py")
        extra_args = spec.config.get("extra_args", "")
        gpu_type = spec.config.get("gpu_type")
        gpu_count = spec.config.get("gpu_count", 1)
        boot_disk_size_gb = spec.config.get("boot_disk_size_gb", 200)
        boot_image = spec.config.get("boot_image", _DEFAULT_BOOT_IMAGE)
        preemptible = spec.is_spot

        startup_script = _STARTUP_SCRIPT_TEMPLATE.format(
            code_uri=code_uri,
            output_uri=output_uri,
            entry_point=entry_point,
            extra_args=extra_args,
        )

        def _create_instance() -> str:
            machine_type_url = f"zones/{zone}/machineTypes/{machine_type}"

            # Boot disk
            disk = compute_v1.AttachedDisk(
                auto_delete=True,
                boot=True,
                initialize_params=compute_v1.AttachedDiskInitializeParams(
                    disk_size_gb=boot_disk_size_gb,
                    source_image=boot_image,
                    disk_type=f"zones/{zone}/diskTypes/pd-ssd",
                ),
            )

            # Network
            network_interface = compute_v1.NetworkInterface(
                name="global/networks/default",
                access_configs=[
                    compute_v1.AccessConfig(
                        name="External NAT",
                        type_="ONE_TO_ONE_NAT",
                    )
                ],
            )

            # Metadata (startup script + labels)
            metadata_items = [
                compute_v1.Items(key="startup-script", value=startup_script),
                compute_v1.Items(key="artenic-job-id", value=job_id),
                compute_v1.Items(key="artenic-service", value=spec.service),
                compute_v1.Items(key="artenic-model", value=spec.model),
            ]

            instance_metadata = compute_v1.Metadata(items=metadata_items)

            # Scheduling
            scheduling = compute_v1.Scheduling(
                preemptible=preemptible,
                on_host_maintenance="TERMINATE" if (gpu_type or preemptible) else "MIGRATE",
                automatic_restart=not preemptible,
            )

            # Service account scopes
            service_account = compute_v1.ServiceAccount(
                email="default",
                scopes=[
                    "https://www.googleapis.com/auth/cloud-platform",
                ],
            )

            # Build instance resource
            instance_resource = compute_v1.Instance(
                name=instance_name,
                machine_type=machine_type_url,
                disks=[disk],
                network_interfaces=[network_interface],
                metadata=instance_metadata,
                scheduling=scheduling,
                service_accounts=[service_account],
                labels={
                    "artenic-job": job_id,
                    "artenic-service": spec.service.lower().replace("_", "-"),
                    "purpose": "training",
                },
            )

            # Attach GPU accelerators if requested
            if gpu_type:
                accelerator_type_url = f"zones/{zone}/acceleratorTypes/{gpu_type}"
                instance_resource.guest_accelerators = [
                    compute_v1.AcceleratorConfig(
                        accelerator_type=accelerator_type_url,
                        accelerator_count=gpu_count,
                    )
                ]

            # Insert (create) the instance
            request = compute_v1.InsertInstanceRequest(
                project=self._project_id,
                zone=zone,
                instance_resource=instance_resource,
            )

            operation = self._instances_client.insert(request=request)

            # Wait for the operation to complete
            _wait_for_zone_operation(
                self._zone_operations_client,
                self._project_id,
                zone,
                operation.name,
            )

            return instance_name

        instance_name = await asyncio.to_thread(_create_instance)

        # Store job metadata
        self._job_metadata[job_id] = {
            **self._job_metadata.get(job_id, {}),
            "instance_name": instance_name,
            "zone": zone,
            "machine_type": machine_type,
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "code_uri": code_uri,
            "output_uri": output_uri,
            "start_time": time.time(),
            "spec": spec,
        }

        logger.info(
            "[gcp] Provisioned VM %s (machine=%s, gpu=%s x%d, spot=%s)",
            instance_name,
            machine_type,
            gpu_type or "none",
            gpu_count if gpu_type else 0,
            preemptible,
        )
        return str(job_id)

    # ------------------------------------------------------------------
    # Poll status
    # ------------------------------------------------------------------

    async def _poll_provider(self, provider_job_id: str) -> CloudJobStatus:
        """Check VM status and parse serial output for metrics."""
        metadata = self._job_metadata.get(provider_job_id)
        if metadata is None:
            return CloudJobStatus(
                provider_job_id=provider_job_id,
                status=JobStatus.FAILED,
                error=f"No metadata found for job {provider_job_id}",
            )

        instance_name = metadata["instance_name"]
        zone = metadata["zone"]
        start_time = metadata.get("start_time", time.time())

        def _get_status() -> tuple[str, dict[str, Any]]:
            request = compute_v1.GetInstanceRequest(
                project=self._project_id,
                zone=zone,
                instance=instance_name,
            )
            try:
                instance = self._instances_client.get(request=request)
            except Exception as exc:
                error_msg = str(exc)
                if "404" in error_msg or "not found" in error_msg.lower():
                    return "TERMINATED", {}
                raise

            vm_status = instance.status  # RUNNING, TERMINATED, STAGING, etc.

            # Try to read serial console output for metrics
            metrics: dict[str, Any] = {}
            try:
                serial_request = compute_v1.GetSerialPortOutputInstanceRequest(
                    project=self._project_id,
                    zone=zone,
                    instance=instance_name,
                    port=1,
                )
                serial_output = self._instances_client.get_serial_port_output(
                    request=serial_request,
                )
                contents: str = serial_output.contents or ""
                metrics = _parse_serial_metrics(contents)
            except Exception:
                logger.debug(
                    "[gcp] Could not read serial console for %s",
                    instance_name,
                )

            return vm_status, metrics

        vm_status, metrics = await asyncio.to_thread(_get_status)

        duration = time.time() - start_time

        # Map GCE VM status to JobStatus
        if vm_status in ("PROVISIONING", "STAGING"):
            job_status = JobStatus.PENDING
        elif vm_status == "RUNNING":
            # Check if training completed from serial output
            if metrics.get("training_completed"):
                job_status = JobStatus.COMPLETED
            elif metrics.get("training_failed"):
                job_status = JobStatus.FAILED
            else:
                job_status = JobStatus.RUNNING
        elif vm_status in ("STOPPING", "STOPPED", "TERMINATED"):
            if metrics.get("training_completed"):
                job_status = JobStatus.COMPLETED
            elif metrics.get("training_failed"):
                job_status = JobStatus.FAILED
            else:
                # VM stopped without clear training status -- could be preemption
                is_spot = metadata.get("spec") and metadata["spec"].is_spot
                job_status = JobStatus.PREEMPTED if is_spot else JobStatus.FAILED
        elif vm_status == "SUSPENDED":
            job_status = JobStatus.PREEMPTED
        else:
            job_status = JobStatus.RUNNING

        return CloudJobStatus(
            provider_job_id=provider_job_id,
            status=job_status,
            metrics=metrics if metrics else None,
            duration_seconds=duration,
            error=metrics.get("error_message"),
        )

    # ------------------------------------------------------------------
    # Collect artifacts
    # ------------------------------------------------------------------

    async def _collect_artifacts(
        self,
        provider_job_id: str,
        status: CloudJobStatus,
    ) -> str | None:
        """Download training artifacts from GCS."""
        metadata = self._job_metadata.get(provider_job_id)
        if metadata is None:
            logger.warning(
                "[gcp] No metadata for job %s, cannot collect artifacts",
                provider_job_id,
            )
            return None

        output_uri: str = metadata.get("output_uri", "")
        if not output_uri.startswith("gs://"):
            return None

        # Parse bucket and prefix from gs:// URI
        parts = output_uri.replace("gs://", "").split("/", 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

        local_dir = pathlib.Path(tempfile.gettempdir()) / "artenic" / "artifacts" / provider_job_id

        def _download() -> str:
            local_dir.mkdir(parents=True, exist_ok=True)
            bucket = self._storage_client.bucket(bucket_name)
            blobs = list(bucket.list_blobs(prefix=prefix))

            if not blobs:
                logger.info("[gcp] No artifacts found at %s", output_uri)
                return output_uri  # Return remote URI if nothing to download

            for blob in blobs:
                relative_path = blob.name
                if prefix:
                    relative_path = blob.name[len(prefix) :].lstrip("/")
                if not relative_path:
                    continue

                local_path = local_dir / relative_path
                local_path.parent.mkdir(parents=True, exist_ok=True)
                blob.download_to_filename(str(local_path))
                logger.debug("[gcp] Downloaded %s -> %s", blob.name, local_path)

            logger.info(
                "[gcp] Downloaded %d artifacts to %s",
                len(blobs),
                local_dir,
            )
            return str(local_dir)

        local_path = await asyncio.to_thread(_download)
        return local_path

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def _cleanup_compute(self, provider_job_id: str) -> None:
        """Delete the VM instance for a completed or failed job."""
        metadata = self._job_metadata.get(provider_job_id)
        if metadata is None:
            logger.warning("[gcp] No metadata for job %s, nothing to clean up", provider_job_id)
            return

        instance_name = metadata["instance_name"]
        zone = metadata["zone"]

        def _delete() -> None:
            request = compute_v1.DeleteInstanceRequest(
                project=self._project_id,
                zone=zone,
                instance=instance_name,
            )
            try:
                operation = self._instances_client.delete(request=request)
                _wait_for_zone_operation(
                    self._zone_operations_client,
                    self._project_id,
                    zone,
                    operation.name,
                )
                logger.info("[gcp] Deleted VM %s in %s", instance_name, zone)
            except Exception as exc:
                if "404" in str(exc) or "not found" in str(exc).lower():
                    logger.info("[gcp] VM %s already deleted", instance_name)
                else:
                    raise

        await asyncio.to_thread(_delete)

        # Clean up local metadata
        self._job_metadata.pop(provider_job_id, None)

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    async def _cancel_provider_job(self, provider_job_id: str) -> None:
        """Stop and delete the VM for a cancelled job."""
        metadata = self._job_metadata.get(provider_job_id)
        if metadata is None:
            logger.warning("[gcp] No metadata for job %s, cannot cancel", provider_job_id)
            return

        instance_name = metadata["instance_name"]
        zone = metadata["zone"]

        def _stop_and_delete() -> None:
            # First try to stop the instance gracefully
            try:
                stop_request = compute_v1.StopInstanceRequest(
                    project=self._project_id,
                    zone=zone,
                    instance=instance_name,
                )
                operation = self._instances_client.stop(request=stop_request)
                _wait_for_zone_operation(
                    self._zone_operations_client,
                    self._project_id,
                    zone,
                    operation.name,
                    timeout_seconds=120,
                )
                logger.info("[gcp] Stopped VM %s", instance_name)
            except Exception as exc:
                logger.warning(
                    "[gcp] Failed to stop VM %s: %s",
                    instance_name,
                    exc,
                )

            # Then delete
            try:
                delete_request = compute_v1.DeleteInstanceRequest(
                    project=self._project_id,
                    zone=zone,
                    instance=instance_name,
                )
                operation = self._instances_client.delete(request=delete_request)
                _wait_for_zone_operation(
                    self._zone_operations_client,
                    self._project_id,
                    zone,
                    operation.name,
                )
                logger.info("[gcp] Deleted VM %s after cancellation", instance_name)
            except Exception as exc:
                if "404" in str(exc) or "not found" in str(exc).lower():
                    logger.info("[gcp] VM %s already deleted", instance_name)
                else:
                    raise

        await asyncio.to_thread(_stop_and_delete)


# ======================================================================
# Helpers (module-level)
# ======================================================================


def _wait_for_zone_operation(
    operations_client: Any,
    project: str,
    zone: str,
    operation_name: str,
    timeout_seconds: int = 300,
) -> None:
    """Block until a zonal operation completes or times out."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        result = operations_client.get(
            project=project,
            zone=zone,
            operation=operation_name,
        )
        if result.status == compute_v1.Operation.Status.DONE:
            if result.error:
                errors = "; ".join(e.message for e in (result.error.errors or []))
                raise RuntimeError(f"GCE operation {operation_name} failed: {errors}")
            return
        time.sleep(2)
    raise TimeoutError(f"GCE operation {operation_name} timed out after {timeout_seconds}s")


def _parse_serial_metrics(serial_output: str) -> dict[str, Any]:
    """Extract training metrics from serial console output.

    Looks for lines like ``METRIC: key=value`` and the sentinel lines
    ``===== TRAINING COMPLETED =====`` / ``===== TRAINING FAILED =====``.
    """
    metrics: dict[str, Any] = {}

    if "===== TRAINING COMPLETED =====" in serial_output:
        metrics["training_completed"] = True
    if "===== TRAINING FAILED" in serial_output:
        metrics["training_failed"] = True

    for line in serial_output.splitlines():
        line = line.strip()
        if line.startswith("METRIC:"):
            # Format: METRIC: key=value
            try:
                kv = line[len("METRIC:") :].strip()
                key, value = kv.split("=", 1)
                key = key.strip()
                value = value.strip()
                # Try to parse as number
                try:
                    metrics[key] = float(value)
                except ValueError:
                    metrics[key] = value
            except ValueError:
                continue

        # Also look for JSON-formatted metrics
        if line.startswith('{"metrics":') or line.startswith("{'metrics':"):
            try:
                parsed = json.loads(line.replace("'", '"'))
                if isinstance(parsed.get("metrics"), dict):
                    metrics.update(parsed["metrics"])
            except (json.JSONDecodeError, AttributeError):
                continue

    return metrics


def _should_skip(file_path: pathlib.Path) -> bool:
    """Return True for files that should not be uploaded."""
    skip_patterns = {
        "__pycache__",
        ".git",
        ".venv",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".tox",
        ".egg-info",
    }
    parts = file_path.parts
    for part in parts:
        if part in skip_patterns:
            return True
    skip_suffixes = {".pyc", ".pyo", ".egg", ".whl"}
    return file_path.suffix in skip_suffixes
