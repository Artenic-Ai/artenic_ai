"""CoreWeave cloud training provider.

CoreWeave is a Kubernetes-native GPU cloud.  This provider uses the
standard Kubernetes Python SDK to create Jobs, ConfigMaps, and PVCs on a
CoreWeave cluster.  All blocking SDK calls are dispatched via
:func:`asyncio.to_thread` so the provider stays fully async.
"""

from __future__ import annotations

import asyncio
import logging
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
# Lazy import of the kubernetes SDK - it is an optional dependency
# ---------------------------------------------------------------------------
try:
    from kubernetes import client as k8s_client  # pragma: no cover
    from kubernetes import config as k8s_config  # pragma: no cover
    from kubernetes.client.rest import ApiException  # pragma: no cover

    _HAS_K8S = True  # pragma: no cover
except ImportError:
    _HAS_K8S = False
    k8s_client = None
    k8s_config = None
    ApiException = Exception

# ---------------------------------------------------------------------------
# Approximate hourly pricing (EUR) for well-known CoreWeave GPU types.
# These are rough estimates; real billing comes from the CoreWeave invoice.
# ---------------------------------------------------------------------------
_COREWEAVE_GPU_PRICES: dict[str, float] = {
    "A100_PCIE_80GB": 2.21,
    "A100_SXM4_80GB": 2.39,
    "A100_PCIE_40GB": 2.06,
    "A40": 1.28,
    "RTX_A6000": 1.28,
    "RTX_A5000": 0.77,
    "RTX_A4000": 0.61,
    "H100_SXM5_80GB": 4.25,
    "H100_PCIE_80GB": 4.10,
    "A100X": 2.21,
}


def _require_k8s() -> None:
    """Raise a clear error when the SDK is missing."""
    if not _HAS_K8S:
        raise ImportError(
            "The 'kubernetes' package is required for CoreWeaveProvider.  "
            "Install it with:  pip install kubernetes"
        )


class CoreWeaveProvider(CloudProvider):
    """CoreWeave (Kubernetes-native GPU cloud) training provider.

    Parameters
    ----------
    kubeconfig_path:
        Path to a kubeconfig file.  When *None* the provider tries
        :func:`kubernetes.config.load_incluster_config` first, then falls
        back to the default kubeconfig.
    namespace:
        Kubernetes namespace for all resources (default ``"default"``).
    training_image:
        Container image used for the training Job.
    storage_class:
        StorageClass for PVCs that hold artifacts (default
        ``"shared-hdd-ord1"``).
    node_selector:
        Dictionary of label selectors applied to the Job pod spec to
        target a specific GPU type (e.g.
        ``{"gpu.nvidia.com/class": "A100_PCIE_80GB"}``).
    """

    def __init__(
        self,
        *,
        kubeconfig_path: str | None = None,
        namespace: str = "default",
        training_image: str,
        storage_class: str = "shared-hdd-ord1",
        node_selector: dict[str, str] | None = None,
    ) -> None:
        _require_k8s()
        super().__init__()

        self._kubeconfig_path = kubeconfig_path
        self._namespace = namespace
        self._training_image = training_image
        self._storage_class = storage_class
        self._node_selector: dict[str, str] = node_selector or {}

        # SDK clients - initialised in _connect()
        self._core_api: Any = None
        self._batch_api: Any = None
        self._api_client: Any = None

        # Internal job tracking: provider_job_id -> _CoreWeaveJobState
        self._jobs: dict[str, _CoreWeaveJobState] = {}

    # ------------------------------------------------------------------
    # Provider identity
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "coreweave"

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def _connect(self) -> None:
        """Load kubeconfig and create CoreV1Api + BatchV1Api clients."""

        def _load() -> tuple[Any, Any, Any]:
            if self._kubeconfig_path:
                k8s_config.load_kube_config(config_file=self._kubeconfig_path)
            else:
                try:
                    k8s_config.load_incluster_config()
                except k8s_config.ConfigException:
                    k8s_config.load_kube_config()

            api_client = k8s_client.ApiClient()
            core = k8s_client.CoreV1Api(api_client)
            batch = k8s_client.BatchV1Api(api_client)
            return core, batch, api_client

        self._core_api, self._batch_api, self._api_client = await asyncio.to_thread(_load)
        logger.info(
            "[coreweave] Connected to Kubernetes cluster (namespace=%s)",
            self._namespace,
        )

    async def _disconnect(self) -> None:
        """Close the API client and release references."""
        if self._api_client is not None:
            try:
                await asyncio.to_thread(self._api_client.close)
            except Exception:
                logger.debug("[coreweave] Error closing API client", exc_info=True)
        self._core_api = None
        self._batch_api = None
        self._api_client = None
        logger.info("[coreweave] Disconnected from Kubernetes cluster")

    # ------------------------------------------------------------------
    # Code upload
    # ------------------------------------------------------------------

    async def _upload_code(self, spec: TrainingSpec) -> str:
        """Create a ConfigMap containing training configuration.

        For larger payloads the training image is expected to pull code
        from a registry or object store.  The ConfigMap carries
        lightweight config / entrypoint overrides.
        """
        assert self._core_api is not None

        job_key = f"artenic-code-{uuid.uuid4().hex[:10]}"
        code_data: dict[str, str] = {
            "service": spec.service,
            "model": spec.model,
            "config.json": _safe_json(spec.config),
        }

        # If a small inline script is provided, include it.
        inline_code = spec.config.get("inline_code", "")
        if inline_code:
            code_data["train.py"] = str(inline_code)

        configmap = k8s_client.V1ConfigMap(
            metadata=k8s_client.V1ObjectMeta(
                name=job_key,
                namespace=self._namespace,
                labels={
                    "app.kubernetes.io/managed-by": "artenic-ai-platform",
                    "artenic.ai/service": spec.service,
                },
            ),
            data=code_data,
        )

        await asyncio.to_thread(
            self._core_api.create_namespaced_config_map,
            namespace=self._namespace,
            body=configmap,
        )
        logger.info("[coreweave] Created ConfigMap %s", job_key)
        return job_key

    # ------------------------------------------------------------------
    # Provisioning
    # ------------------------------------------------------------------

    async def _provision_and_start(self, spec: TrainingSpec) -> str:
        """Create a Kubernetes Job with CoreWeave GPU node selectors."""
        assert self._batch_api is not None

        provider_job_id = f"cw-{uuid.uuid4().hex[:10]}"
        job_name = f"artenic-{provider_job_id}"

        # Determine GPU count from spec config (default 1)
        gpu_count = int(spec.config.get("gpu_count", 1))

        # Build resource requirements
        resources = k8s_client.V1ResourceRequirements(
            requests={"nvidia.com/gpu": str(gpu_count)},
            limits={"nvidia.com/gpu": str(gpu_count)},
        )

        # Merge provider-level node selector with any spec overrides
        node_selector: dict[str, str] = dict(self._node_selector)
        spec_selectors = spec.config.get("node_selector")
        if isinstance(spec_selectors, dict):
            node_selector.update(spec_selectors)

        # Training command
        train_command = spec.config.get("train_command", "python train.py")
        command = ["/bin/sh", "-c", train_command]

        # Environment variables
        env_list: list[Any] = [
            k8s_client.V1EnvVar(name="ARTENIC_JOB_ID", value=provider_job_id),
            k8s_client.V1EnvVar(name="ARTENIC_SERVICE", value=spec.service),
            k8s_client.V1EnvVar(name="ARTENIC_MODEL", value=spec.model),
        ]
        for key, value in spec.config.get("env", {}).items():
            env_list.append(k8s_client.V1EnvVar(name=str(key), value=str(value)))

        # PVC for artifacts
        pvc_name = f"artenic-artifacts-{provider_job_id}"
        pvc_size = spec.config.get("artifact_storage_size", "50Gi")
        await self._create_pvc(pvc_name, str(pvc_size))

        volume_mounts = [
            k8s_client.V1VolumeMount(
                name="artifacts",
                mount_path="/artifacts",
            ),
        ]

        # Optionally mount the code ConfigMap
        code_configmap = spec.config.get("_code_configmap")
        if code_configmap:
            volume_mounts.append(
                k8s_client.V1VolumeMount(
                    name="code",
                    mount_path="/opt/artenic/code",
                    read_only=True,
                ),
            )

        volumes: list[Any] = [
            k8s_client.V1Volume(
                name="artifacts",
                persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
                    claim_name=pvc_name,
                ),
            ),
        ]
        if code_configmap:
            volumes.append(
                k8s_client.V1Volume(
                    name="code",
                    config_map=k8s_client.V1ConfigMapVolumeSource(
                        name=str(code_configmap),
                    ),
                ),
            )

        # Max runtime as activeDeadlineSeconds
        active_deadline = int(spec.max_runtime_hours * 3600)

        container = k8s_client.V1Container(
            name="training",
            image=self._training_image,
            command=command,
            env=env_list,
            resources=resources,
            volume_mounts=volume_mounts,
        )

        pod_spec = k8s_client.V1PodSpec(
            containers=[container],
            volumes=volumes,
            restart_policy="Never",
            node_selector=node_selector if node_selector else None,
        )

        template = k8s_client.V1PodTemplateSpec(
            metadata=k8s_client.V1ObjectMeta(
                labels={
                    "app.kubernetes.io/managed-by": "artenic-ai-platform",
                    "artenic.ai/job-id": provider_job_id,
                    "artenic.ai/service": spec.service,
                },
            ),
            spec=pod_spec,
        )

        job_spec = k8s_client.V1JobSpec(
            template=template,
            backoff_limit=0,
            active_deadline_seconds=active_deadline,
        )

        job_body = k8s_client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=k8s_client.V1ObjectMeta(
                name=job_name,
                namespace=self._namespace,
                labels={
                    "app.kubernetes.io/managed-by": "artenic-ai-platform",
                    "artenic.ai/job-id": provider_job_id,
                    "artenic.ai/service": spec.service,
                    "artenic.ai/model": spec.model,
                },
            ),
            spec=job_spec,
        )

        await asyncio.to_thread(
            self._batch_api.create_namespaced_job,
            namespace=self._namespace,
            body=job_body,
        )

        self._jobs[provider_job_id] = _CoreWeaveJobState(
            job_name=job_name,
            pvc_name=pvc_name,
            created_at=time.time(),
            spec=spec,
            gpu_count=gpu_count,
        )

        logger.info(
            "[coreweave] Created Job %s (gpu_count=%d, node_selector=%s)",
            job_name,
            gpu_count,
            node_selector,
        )
        return provider_job_id

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def _poll_provider(self, provider_job_id: str) -> CloudJobStatus:
        """Read Job conditions to determine status."""
        assert self._batch_api is not None

        state = self._jobs.get(provider_job_id)
        if state is None:
            return CloudJobStatus(
                provider_job_id=provider_job_id,
                status=JobStatus.FAILED,
                error=f"Unknown job {provider_job_id}",
            )

        try:
            job: Any = await asyncio.to_thread(
                self._batch_api.read_namespaced_job,
                name=state.job_name,
                namespace=self._namespace,
            )
        except ApiException as exc:
            logger.error(
                "[coreweave] Failed to read Job %s: %s",
                state.job_name,
                exc,
            )
            return CloudJobStatus(
                provider_job_id=provider_job_id,
                status=JobStatus.FAILED,
                error=f"Cannot read Job: {exc.reason}",
            )

        elapsed = time.time() - state.created_at
        job_status = self._parse_job_status(job)

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
        """Read pod logs and report the PVC path for artifacts."""
        assert self._core_api is not None

        state = self._jobs.get(provider_job_id)
        if state is None:
            logger.warning(
                "[coreweave] Cannot collect artifacts - unknown job %s",
                provider_job_id,
            )
            return None

        # Attempt to retrieve pod logs for the job
        try:
            pods: Any = await asyncio.to_thread(
                self._core_api.list_namespaced_pod,
                namespace=self._namespace,
                label_selector=f"artenic.ai/job-id={provider_job_id}",
            )
            for pod in pods.items:
                try:
                    log: str = await asyncio.to_thread(
                        self._core_api.read_namespaced_pod_log,
                        name=pod.metadata.name,
                        namespace=self._namespace,
                        tail_lines=500,
                    )
                    logger.info(
                        "[coreweave] Collected %d bytes of logs from pod %s",
                        len(log),
                        pod.metadata.name,
                    )
                except ApiException:
                    logger.debug(
                        "[coreweave] Could not read logs for pod %s",
                        pod.metadata.name,
                    )
        except ApiException:
            logger.debug("[coreweave] Could not list pods for job %s", provider_job_id)

        # The artifacts live on the PVC
        artifact_uri = f"pvc://{self._namespace}/{state.pvc_name}/artifacts"
        logger.info("[coreweave] Artifacts at %s", artifact_uri)
        return artifact_uri

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def _cleanup_compute(self, provider_job_id: str) -> None:
        """Delete the Kubernetes Job with Foreground propagation."""
        assert self._batch_api is not None

        state = self._jobs.get(provider_job_id)
        if state is None:
            logger.warning(
                "[coreweave] Cannot cleanup - unknown job %s",
                provider_job_id,
            )
            return

        try:
            await asyncio.to_thread(
                self._batch_api.delete_namespaced_job,
                name=state.job_name,
                namespace=self._namespace,
                body=k8s_client.V1DeleteOptions(
                    propagation_policy="Foreground",
                ),
            )
            logger.info("[coreweave] Deleted Job %s", state.job_name)
        except ApiException as exc:
            if exc.status == 404:
                logger.debug("[coreweave] Job %s already deleted", state.job_name)
            else:
                logger.error(
                    "[coreweave] Failed to delete Job %s: %s",
                    state.job_name,
                    exc,
                )
                raise

        self._jobs.pop(provider_job_id, None)

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    async def _cancel_provider_job(self, provider_job_id: str) -> None:
        """Cancel a running job by deleting the Job resource."""
        assert self._batch_api is not None

        state = self._jobs.get(provider_job_id)
        if state is None:
            logger.warning(
                "[coreweave] Cannot cancel - unknown job %s",
                provider_job_id,
            )
            return

        try:
            await asyncio.to_thread(
                self._batch_api.delete_namespaced_job,
                name=state.job_name,
                namespace=self._namespace,
                body=k8s_client.V1DeleteOptions(
                    propagation_policy="Foreground",
                ),
            )
            logger.info(
                "[coreweave] Cancelled Job %s for job %s",
                state.job_name,
                provider_job_id,
            )
        except ApiException as exc:
            if exc.status == 404:
                logger.debug("[coreweave] Job %s already gone", state.job_name)
            else:
                logger.error(
                    "[coreweave] Failed to cancel Job %s: %s",
                    state.job_name,
                    exc,
                )
                raise

    # ------------------------------------------------------------------
    # Instance listing
    # ------------------------------------------------------------------

    async def _list_instances(
        self,
        region: str | None = None,
        gpu_only: bool = False,
    ) -> list[InstanceType]:
        """List cluster nodes and extract allocatable GPU resources.

        CoreWeave nodes carry labels such as ``gpu.nvidia.com/class``
        and ``topology.kubernetes.io/region`` that we use to build
        :class:`InstanceType` entries with pricing information.
        """
        assert self._core_api is not None

        nodes: Any = await asyncio.to_thread(
            self._core_api.list_node,
        )

        instances: list[InstanceType] = []
        for node in nodes.items:
            labels: dict[str, str] = node.metadata.labels or {}
            allocatable: dict[str, str] = node.status.allocatable or {}

            # Region filtering
            node_region = labels.get(
                "topology.kubernetes.io/region",
                labels.get("topology.kubernetes.io/zone", ""),
            )
            if region and node_region and not node_region.startswith(region):
                continue

            # GPU info from CoreWeave-specific labels
            gpu_class = labels.get("gpu.nvidia.com/class", "")
            gpu_count_str = allocatable.get("nvidia.com/gpu", "0")
            try:
                gpu_count = int(gpu_count_str)
            except ValueError:
                gpu_count = 0

            if gpu_only and gpu_count == 0:
                continue

            # CPU and memory
            cpu_str = allocatable.get("cpu", "0")
            vcpus = _parse_k8s_cpu(cpu_str)

            memory_str = allocatable.get("memory", "0")
            memory_gb = _parse_k8s_memory_gb(memory_str)

            # Pricing from our lookup table
            price_eur = _COREWEAVE_GPU_PRICES.get(gpu_class, 0.0)
            if gpu_count > 1:
                price_eur *= gpu_count

            instance_name = node.metadata.name
            instances.append(
                InstanceType(
                    name=instance_name,
                    vcpus=vcpus,
                    memory_gb=memory_gb,
                    gpu_type=gpu_class if gpu_class else None,
                    gpu_count=gpu_count,
                    price_per_hour_eur=round(price_eur, 4),
                    spot_price_per_hour_eur=None,
                    region=node_region or None,
                    available=_is_node_ready(node),
                )
            )

        logger.info(
            "[coreweave] Found %d nodes (gpu_only=%s, region=%s)",
            len(instances),
            gpu_only,
            region,
        )
        return instances

    # ==================================================================
    # Private helpers
    # ==================================================================

    async def _create_pvc(self, name: str, size: str) -> None:
        """Create a PVC for training artifacts."""
        assert self._core_api is not None

        pvc = k8s_client.V1PersistentVolumeClaim(
            metadata=k8s_client.V1ObjectMeta(
                name=name,
                namespace=self._namespace,
                labels={
                    "app.kubernetes.io/managed-by": "artenic-ai-platform",
                },
            ),
            spec=k8s_client.V1PersistentVolumeClaimSpec(
                access_modes=["ReadWriteOnce"],
                storage_class_name=self._storage_class,
                resources=k8s_client.V1VolumeResourceRequirements(
                    requests={"storage": size},
                ),
            ),
        )

        try:
            await asyncio.to_thread(
                self._core_api.create_namespaced_persistent_volume_claim,
                namespace=self._namespace,
                body=pvc,
            )
            logger.info("[coreweave] Created PVC %s (%s)", name, size)
        except ApiException as exc:
            if exc.status == 409:
                logger.debug("[coreweave] PVC %s already exists", name)
            else:
                raise

    @staticmethod
    def _parse_job_status(job: Any) -> JobStatus:
        """Derive :class:`JobStatus` from Kubernetes Job conditions."""
        status = job.status
        if status is None:
            return JobStatus.PENDING

        # Check conditions first (Complete / Failed)
        for condition in status.conditions or []:
            if condition.type == "Complete" and condition.status == "True":
                return JobStatus.COMPLETED
            if condition.type == "Failed" and condition.status == "True":
                reason = condition.reason or ""
                if "DeadlineExceeded" in reason:
                    return JobStatus.FAILED
                return JobStatus.FAILED

        # Active pods mean the job is running
        if status.active and status.active > 0:
            return JobStatus.RUNNING

        # Succeeded count
        if status.succeeded and status.succeeded > 0:
            return JobStatus.COMPLETED

        # Failed count
        if status.failed and status.failed > 0:
            return JobStatus.FAILED

        return JobStatus.PENDING

    @staticmethod
    def _estimate_cost(
        state: _CoreWeaveJobState,
        elapsed_seconds: float,
    ) -> float | None:
        """Estimate running cost from GPU pricing table."""
        node_selector = state.spec.config.get("node_selector", {})
        gpu_class = node_selector.get("gpu.nvidia.com/class", "")
        per_gpu = _COREWEAVE_GPU_PRICES.get(gpu_class)
        if per_gpu is None:
            return None
        hours = elapsed_seconds / 3600.0
        return round(hours * per_gpu * state.gpu_count, 4)


# ---------------------------------------------------------------------------
# Internal state tracking
# ---------------------------------------------------------------------------


class _CoreWeaveJobState:
    """Tracks the state of a training job running on CoreWeave."""

    __slots__ = (
        "created_at",
        "gpu_count",
        "job_name",
        "pvc_name",
        "spec",
    )

    def __init__(
        self,
        job_name: str,
        pvc_name: str,
        created_at: float,
        spec: TrainingSpec,
        gpu_count: int = 1,
    ) -> None:
        self.job_name = job_name
        self.pvc_name = pvc_name
        self.created_at = created_at
        self.spec = spec
        self.gpu_count = gpu_count


# ---------------------------------------------------------------------------
# Kubernetes resource parsing helpers
# ---------------------------------------------------------------------------


def _parse_k8s_cpu(cpu_str: str) -> int:
    """Parse a Kubernetes CPU quantity string into an integer vCPU count.

    Examples: ``"4"`` -> 4, ``"4000m"`` -> 4, ``"500m"`` -> 1
    """
    cpu_str = cpu_str.strip()
    if cpu_str.endswith("m"):
        return max(1, int(cpu_str[:-1]) // 1000)
    try:
        return int(cpu_str)
    except ValueError:
        try:
            return max(1, int(float(cpu_str)))
        except ValueError:
            return 0


def _parse_k8s_memory_gb(memory_str: str) -> float:
    """Parse a Kubernetes memory quantity string into GB.

    Examples: ``"16Gi"`` -> 16.0, ``"16384Mi"`` -> 16.0, ``"17179869184"`` -> 16.0
    """
    memory_str = memory_str.strip()
    if memory_str.endswith("Ki"):
        return round(int(memory_str[:-2]) / (1024 * 1024), 2)
    if memory_str.endswith("Mi"):
        return round(int(memory_str[:-2]) / 1024, 2)
    if memory_str.endswith("Gi"):
        return round(int(memory_str[:-2]), 2)
    if memory_str.endswith("Ti"):
        return round(int(memory_str[:-2]) * 1024, 2)
    # Plain bytes
    try:
        return round(int(memory_str) / (1024**3), 2)
    except ValueError:
        return 0.0


def _is_node_ready(node: Any) -> bool:
    """Return True if the node has a Ready=True condition."""
    if node.status is None or node.status.conditions is None:
        return False
    for condition in node.status.conditions:
        if condition.type == "Ready":
            return bool(condition.status == "True")
    return False


def _safe_json(data: dict[str, Any]) -> str:
    """Serialise a dict to JSON, handling non-serialisable values."""
    import json

    try:
        return json.dumps(data, default=str)
    except (TypeError, ValueError):
        return "{}"
