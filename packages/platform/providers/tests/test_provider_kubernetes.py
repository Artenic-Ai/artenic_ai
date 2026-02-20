"""Tests for artenic_ai_platform_providers.kubernetes — KubernetesProvider."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from artenic_ai_platform_providers.base import (
    CloudJobStatus,
    JobStatus,
    TrainingSpec,
)

_MODULE = "artenic_ai_platform_providers.kubernetes"


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _spec(**overrides) -> TrainingSpec:
    defaults = {
        "service": "training",
        "model": "llama-7b",
        "provider": "kubernetes",
        "config": {},
    }
    defaults.update(overrides)
    return TrainingSpec(**defaults)


def _make_k8s_mock():
    """Build a mock kubernetes SDK module hierarchy."""
    mock_k8s_client = MagicMock()
    mock_k8s_config = MagicMock()
    for attr in (
        "V1ConfigMap",
        "V1ObjectMeta",
        "V1Job",
        "V1JobSpec",
        "V1PodTemplateSpec",
        "V1PodSpec",
        "V1Container",
        "V1EnvVar",
        "V1ResourceRequirements",
        "V1VolumeMount",
        "V1Volume",
        "V1PersistentVolumeClaimVolumeSource",
        "V1ConfigMapVolumeSource",
        "V1PersistentVolumeClaim",
        "V1PersistentVolumeClaimSpec",
        "V1VolumeResourceRequirements",
        "V1DeleteOptions",
        "ApiClient",
        "CoreV1Api",
        "BatchV1Api",
    ):
        setattr(mock_k8s_client, attr, MagicMock())
    mock_k8s_config.ConfigException = type("ConfigException", (Exception,), {})
    return mock_k8s_client, mock_k8s_config


class _FakeApiError(Exception):
    """Stand-in for kubernetes.client.rest.ApiException."""

    def __init__(self, status=500, reason="error"):
        self.status = status
        self.reason = reason
        super().__init__(reason)


# ---------------------------------------------------------------
# Module-level patches
# ---------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_k8s():
    mock_client, mock_config = _make_k8s_mock()
    with (
        patch(f"{_MODULE}._HAS_K8S", True),
        patch(f"{_MODULE}.k8s_client", mock_client),
        patch(f"{_MODULE}.k8s_config", mock_config),
        patch(f"{_MODULE}.ApiException", _FakeApiError),
    ):
        yield mock_client, mock_config


@pytest.fixture()
def provider(_patch_k8s):
    from artenic_ai_platform_providers.kubernetes import (
        KubernetesProvider,
    )

    return KubernetesProvider(
        training_image="registry.example.com/train:latest",
        namespace="artenic-training",
    )


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------


class TestKubernetesInit:
    def test_provider_name(self, provider):
        assert provider.provider_name == "kubernetes"

    def test_init_stores_fields(self, provider):
        assert provider._namespace == "artenic-training"
        assert provider._training_image == "registry.example.com/train:latest"
        assert provider._storage_class == "standard"
        assert provider._connected is False
        assert provider._node_selector == {}

    def test_init_with_node_selector(self, _patch_k8s):
        from artenic_ai_platform_providers.kubernetes import (
            KubernetesProvider,
        )

        p = KubernetesProvider(
            training_image="img",
            node_selector={"arch": "amd64"},
        )
        assert p._node_selector == {"arch": "amd64"}

    def test_init_raises_without_k8s(self):
        with patch(f"{_MODULE}._HAS_K8S", False), pytest.raises(ImportError, match="kubernetes"):
            from artenic_ai_platform_providers.kubernetes import (
                KubernetesProvider,
            )

            KubernetesProvider(training_image="img")


class TestKubernetesConnect:
    @pytest.mark.asyncio
    async def test_connect_default_kubeconfig(self, provider, _patch_k8s):
        mock_client, _mock_config = _patch_k8s
        mock_client.ApiClient.return_value = MagicMock()
        mock_client.CoreV1Api.return_value = MagicMock()
        mock_client.BatchV1Api.return_value = MagicMock()

        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            await provider._connect()
        assert provider._core_api is not None
        assert provider._batch_api is not None

    @pytest.mark.asyncio
    async def test_connect_with_kubeconfig_path(self, _patch_k8s):
        mock_client, mock_config = _patch_k8s
        mock_client.ApiClient.return_value = MagicMock()
        mock_client.CoreV1Api.return_value = MagicMock()
        mock_client.BatchV1Api.return_value = MagicMock()

        from artenic_ai_platform_providers.kubernetes import (
            KubernetesProvider,
        )

        p = KubernetesProvider(
            training_image="img",
            kubeconfig_path="/etc/kube/config",
        )
        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            await p._connect()
        mock_config.load_kube_config.assert_called_once_with(config_file="/etc/kube/config")

    @pytest.mark.asyncio
    async def test_connect_falls_back_to_kube_config(self, provider, _patch_k8s):
        mock_client, mock_config = _patch_k8s
        mock_config.load_incluster_config.side_effect = mock_config.ConfigException(
            "not in cluster"
        )
        mock_client.ApiClient.return_value = MagicMock()
        mock_client.CoreV1Api.return_value = MagicMock()
        mock_client.BatchV1Api.return_value = MagicMock()

        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            await provider._connect()
        mock_config.load_kube_config.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect(self, provider):
        mock_api = MagicMock()
        provider._api_client = mock_api
        provider._core_api = MagicMock()
        provider._batch_api = MagicMock()

        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            await provider._disconnect()
        assert provider._core_api is None
        assert provider._batch_api is None
        assert provider._api_client is None
        mock_api.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_handles_close_error(self, provider):
        mock_api = MagicMock()
        mock_api.close.side_effect = RuntimeError("oops")
        provider._api_client = mock_api

        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            await provider._disconnect()
        assert provider._api_client is None

    @pytest.mark.asyncio
    async def test_disconnect_noop(self, provider):
        provider._api_client = None
        await provider._disconnect()


class TestKubernetesUploadCode:
    @pytest.mark.asyncio
    async def test_upload_creates_configmap(self, provider):
        mock_core = MagicMock()
        provider._core_api = mock_core
        spec = _spec()

        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            result = await provider._upload_code(spec)
        assert result.startswith("artenic-code-")
        mock_core.create_namespaced_config_map.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_includes_inline_code(self, provider):
        mock_core = MagicMock()
        provider._core_api = mock_core
        spec = _spec(config={"inline_code": "import torch"})

        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            await provider._upload_code(spec)
        mock_core.create_namespaced_config_map.assert_called_once()


class TestKubernetesProvision:
    @pytest.mark.asyncio
    async def test_provision_creates_job_and_pvc(self, provider):
        mock_batch = MagicMock()
        mock_core = MagicMock()
        provider._batch_api = mock_batch
        provider._core_api = mock_core

        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            job_id = await provider._provision_and_start(_spec())

        assert job_id.startswith("k8s-")
        assert job_id in provider._jobs
        mock_batch.create_namespaced_job.assert_called_once()
        mock_core.create_namespaced_persistent_volume_claim.assert_called_once()

    @pytest.mark.asyncio
    async def test_provision_with_gpu_and_node_selector(self, provider):
        mock_batch = MagicMock()
        mock_core = MagicMock()
        provider._batch_api = mock_batch
        provider._core_api = mock_core

        spec = _spec(
            config={
                "gpu_count": 2,
                "node_selector": {"zone": "a"},
            }
        )
        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            job_id = await provider._provision_and_start(spec)
        assert job_id in provider._jobs

    @pytest.mark.asyncio
    async def test_provision_pvc_conflict_ok(self, provider):
        mock_batch = MagicMock()
        mock_core = MagicMock()
        mock_core.create_namespaced_persistent_volume_claim.side_effect = _FakeApiError(
            status=409, reason="already exists"
        )
        provider._batch_api = mock_batch
        provider._core_api = mock_core

        with (
            patch(
                "asyncio.to_thread",
                new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
            ),
            patch(
                f"{_MODULE}.ApiException",
                _FakeApiError,
            ),
        ):
            job_id = await provider._provision_and_start(_spec())
        assert job_id.startswith("k8s-")


class TestKubernetesPoll:
    @pytest.mark.asyncio
    async def test_poll_unknown_job(self, provider):
        provider._batch_api = MagicMock()
        result = await provider._poll_provider("nope")
        assert result.status == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_poll_running_job(self, provider):
        from artenic_ai_platform_providers.kubernetes import (
            _K8sJobState,
        )

        provider._jobs["j1"] = _K8sJobState(
            job_name="artenic-j1",
            pvc_name="pvc-j1",
            created_at=time.time() - 120,
            spec=_spec(),
        )
        mock_job = MagicMock()
        mock_job.status.conditions = []
        mock_job.status.active = 2
        mock_job.status.succeeded = 0
        mock_job.status.failed = 0
        mock_batch = MagicMock()
        mock_batch.read_namespaced_job.return_value = mock_job
        provider._batch_api = mock_batch

        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            result = await provider._poll_provider("j1")
        assert result.status == JobStatus.RUNNING
        # Kubernetes provider reports no cost
        assert result.cost_eur is None

    @pytest.mark.asyncio
    async def test_poll_completed_job(self, provider):
        from artenic_ai_platform_providers.kubernetes import (
            _K8sJobState,
        )

        provider._jobs["j2"] = _K8sJobState(
            job_name="artenic-j2",
            pvc_name="pvc-j2",
            created_at=time.time() - 600,
            spec=_spec(),
        )
        cond = MagicMock()
        cond.type = "Complete"
        cond.status = "True"
        mock_job = MagicMock()
        mock_job.status.conditions = [cond]
        mock_batch = MagicMock()
        mock_batch.read_namespaced_job.return_value = mock_job
        provider._batch_api = mock_batch

        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            result = await provider._poll_provider("j2")
        assert result.status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_poll_failed_job(self, provider):
        from artenic_ai_platform_providers.kubernetes import (
            _K8sJobState,
        )

        provider._jobs["j3"] = _K8sJobState(
            job_name="artenic-j3",
            pvc_name="pvc-j3",
            created_at=time.time(),
            spec=_spec(),
        )
        cond = MagicMock()
        cond.type = "Failed"
        cond.status = "True"
        mock_job = MagicMock()
        mock_job.status.conditions = [cond]
        mock_batch = MagicMock()
        mock_batch.read_namespaced_job.return_value = mock_job
        provider._batch_api = mock_batch

        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            result = await provider._poll_provider("j3")
        assert result.status == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_poll_pending_no_status(self, provider):
        from artenic_ai_platform_providers.kubernetes import (
            _K8sJobState,
        )

        provider._jobs["j4"] = _K8sJobState(
            job_name="artenic-j4",
            pvc_name="pvc-j4",
            created_at=time.time(),
            spec=_spec(),
        )
        mock_job = MagicMock()
        mock_job.status = None
        mock_batch = MagicMock()
        mock_batch.read_namespaced_job.return_value = mock_job
        provider._batch_api = mock_batch

        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            result = await provider._poll_provider("j4")
        assert result.status == JobStatus.PENDING

    @pytest.mark.asyncio
    async def test_poll_api_exception(self, provider):
        from artenic_ai_platform_providers.kubernetes import (
            _K8sJobState,
        )

        provider._jobs["j5"] = _K8sJobState(
            job_name="artenic-j5",
            pvc_name="pvc-j5",
            created_at=time.time(),
            spec=_spec(),
        )
        mock_batch = MagicMock()
        mock_batch.read_namespaced_job.side_effect = _FakeApiError(status=500, reason="internal")
        provider._batch_api = mock_batch

        with (
            patch(
                "asyncio.to_thread",
                new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
            ),
            patch(
                f"{_MODULE}.ApiException",
                _FakeApiError,
            ),
        ):
            result = await provider._poll_provider("j5")
        assert result.status == JobStatus.FAILED
        assert "Cannot read Job" in (result.error or "")


class TestKubernetesCollectArtifacts:
    @pytest.mark.asyncio
    async def test_collect_returns_pvc_uri(self, provider):
        from artenic_ai_platform_providers.kubernetes import (
            _K8sJobState,
        )

        provider._jobs["j1"] = _K8sJobState(
            job_name="artenic-j1",
            pvc_name="pvc-j1",
            created_at=time.time(),
            spec=_spec(),
        )
        mock_core = MagicMock()
        pods_result = MagicMock()
        pod = MagicMock()
        pod.metadata.name = "pod-1"
        pods_result.items = [pod]
        mock_core.list_namespaced_pod.return_value = pods_result
        mock_core.read_namespaced_pod_log.return_value = "logs"
        provider._core_api = mock_core

        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            result = await provider._collect_artifacts(
                "j1",
                CloudJobStatus(
                    provider_job_id="j1",
                    status=JobStatus.COMPLETED,
                ),
            )
        expected = "pvc://artenic-training/pvc-j1/artifacts"
        assert result == expected

    @pytest.mark.asyncio
    async def test_collect_unknown_job(self, provider):
        provider._core_api = MagicMock()
        result = await provider._collect_artifacts(
            "nope",
            CloudJobStatus(
                provider_job_id="nope",
                status=JobStatus.COMPLETED,
            ),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_collect_handles_pod_log_error(self, provider):
        from artenic_ai_platform_providers.kubernetes import (
            _K8sJobState,
        )

        provider._jobs["j2"] = _K8sJobState(
            job_name="artenic-j2",
            pvc_name="pvc-j2",
            created_at=time.time(),
            spec=_spec(),
        )
        mock_core = MagicMock()
        pods_result = MagicMock()
        pod = MagicMock()
        pod.metadata.name = "pod-2"
        pods_result.items = [pod]
        mock_core.list_namespaced_pod.return_value = pods_result
        mock_core.read_namespaced_pod_log.side_effect = _FakeApiError(
            status=404, reason="not found"
        )
        provider._core_api = mock_core

        with (
            patch(
                "asyncio.to_thread",
                new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
            ),
            patch(
                f"{_MODULE}.ApiException",
                _FakeApiError,
            ),
        ):
            result = await provider._collect_artifacts(
                "j2",
                CloudJobStatus(
                    provider_job_id="j2",
                    status=JobStatus.COMPLETED,
                ),
            )
        # Should still return the PVC URI
        assert result is not None
        assert "pvc-j2" in result


class TestKubernetesCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_deletes_job(self, provider):
        from artenic_ai_platform_providers.kubernetes import (
            _K8sJobState,
        )

        provider._jobs["j1"] = _K8sJobState(
            job_name="artenic-j1",
            pvc_name="pvc-j1",
            created_at=time.time(),
            spec=_spec(),
        )
        mock_batch = MagicMock()
        provider._batch_api = mock_batch

        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            await provider._cleanup_compute("j1")
        assert "j1" not in provider._jobs
        mock_batch.delete_namespaced_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_unknown_job(self, provider):
        provider._batch_api = MagicMock()
        await provider._cleanup_compute("nope")

    @pytest.mark.asyncio
    async def test_cleanup_404_is_ok(self, provider):
        from artenic_ai_platform_providers.kubernetes import (
            _K8sJobState,
        )

        provider._jobs["j2"] = _K8sJobState(
            job_name="artenic-j2",
            pvc_name="pvc-j2",
            created_at=time.time(),
            spec=_spec(),
        )
        mock_batch = MagicMock()
        mock_batch.delete_namespaced_job.side_effect = _FakeApiError(status=404, reason="not found")
        provider._batch_api = mock_batch

        with (
            patch(
                "asyncio.to_thread",
                new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
            ),
            patch(
                f"{_MODULE}.ApiException",
                _FakeApiError,
            ),
        ):
            await provider._cleanup_compute("j2")
        assert "j2" not in provider._jobs

    @pytest.mark.asyncio
    async def test_cleanup_non_404_raises(self, provider):
        from artenic_ai_platform_providers.kubernetes import (
            _K8sJobState,
        )

        provider._jobs["j3"] = _K8sJobState(
            job_name="artenic-j3",
            pvc_name="pvc-j3",
            created_at=time.time(),
            spec=_spec(),
        )
        mock_batch = MagicMock()
        mock_batch.delete_namespaced_job.side_effect = _FakeApiError(status=500, reason="internal")
        provider._batch_api = mock_batch

        with (
            patch(
                "asyncio.to_thread",
                new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
            ),
            patch(
                f"{_MODULE}.ApiException",
                _FakeApiError,
            ),
            pytest.raises(_FakeApiError),
        ):
            await provider._cleanup_compute("j3")


class TestKubernetesCancel:
    @pytest.mark.asyncio
    async def test_cancel_deletes_job(self, provider):
        from artenic_ai_platform_providers.kubernetes import (
            _K8sJobState,
        )

        provider._jobs["j1"] = _K8sJobState(
            job_name="artenic-j1",
            pvc_name="pvc-j1",
            created_at=time.time(),
            spec=_spec(),
        )
        mock_batch = MagicMock()
        provider._batch_api = mock_batch

        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            await provider._cancel_provider_job("j1")
        mock_batch.delete_namespaced_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_unknown_job(self, provider):
        provider._batch_api = MagicMock()
        await provider._cancel_provider_job("nope")

    @pytest.mark.asyncio
    async def test_cancel_404_is_ok(self, provider):
        from artenic_ai_platform_providers.kubernetes import (
            _K8sJobState,
        )

        provider._jobs["j2"] = _K8sJobState(
            job_name="artenic-j2",
            pvc_name="pvc-j2",
            created_at=time.time(),
            spec=_spec(),
        )
        mock_batch = MagicMock()
        mock_batch.delete_namespaced_job.side_effect = _FakeApiError(status=404, reason="gone")
        provider._batch_api = mock_batch

        with (
            patch(
                "asyncio.to_thread",
                new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
            ),
            patch(
                f"{_MODULE}.ApiException",
                _FakeApiError,
            ),
        ):
            await provider._cancel_provider_job("j2")

    @pytest.mark.asyncio
    async def test_cancel_non_404_raises(self, provider):
        from artenic_ai_platform_providers.kubernetes import (
            _K8sJobState,
        )

        provider._jobs["j3"] = _K8sJobState(
            job_name="artenic-j3",
            pvc_name="pvc-j3",
            created_at=time.time(),
            spec=_spec(),
        )
        mock_batch = MagicMock()
        mock_batch.delete_namespaced_job.side_effect = _FakeApiError(status=403, reason="forbidden")
        provider._batch_api = mock_batch

        with (
            patch(
                "asyncio.to_thread",
                new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
            ),
            patch(
                f"{_MODULE}.ApiException",
                _FakeApiError,
            ),
            pytest.raises(_FakeApiError),
        ):
            await provider._cancel_provider_job("j3")


class TestKubernetesListInstances:
    @pytest.mark.asyncio
    async def test_list_instances_parses_nodes(self, provider):
        node = MagicMock()
        node.metadata.name = "worker-01"
        node.metadata.labels = {
            "topology.kubernetes.io/region": "us-east-1",
            "nvidia.com/gpu.product": "Tesla-V100",
        }
        node.status.allocatable = {
            "cpu": "16",
            "memory": "64Gi",
            "nvidia.com/gpu": "2",
        }
        cond = MagicMock()
        cond.type = "Ready"
        cond.status = "True"
        node.status.conditions = [cond]

        nodes_result = MagicMock()
        nodes_result.items = [node]
        mock_core = MagicMock()
        mock_core.list_node.return_value = nodes_result
        provider._core_api = mock_core

        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            result = await provider._list_instances()
        assert len(result) == 1
        inst = result[0]
        assert inst.name == "worker-01"
        assert inst.gpu_type == "Tesla-V100"
        assert inst.gpu_count == 2
        assert inst.vcpus == 16
        assert inst.memory_gb == 64.0
        assert inst.price_per_hour_eur == 0.0
        assert inst.available is True
        assert inst.region == "us-east-1"

    @pytest.mark.asyncio
    async def test_list_instances_gpu_only_filter(self, provider):
        node = MagicMock()
        node.metadata.name = "cpu-node"
        node.metadata.labels = {}
        node.status.allocatable = {
            "cpu": "8",
            "memory": "32Gi",
        }
        node.status.conditions = []

        nodes_result = MagicMock()
        nodes_result.items = [node]
        mock_core = MagicMock()
        mock_core.list_node.return_value = nodes_result
        provider._core_api = mock_core

        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            result = await provider._list_instances(gpu_only=True)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_list_instances_region_filter(self, provider):
        node1 = MagicMock()
        node1.metadata.name = "east-node"
        node1.metadata.labels = {
            "topology.kubernetes.io/region": "us-east-1",
        }
        node1.status.allocatable = {"cpu": "4", "memory": "16Gi"}
        node1.status.conditions = []

        node2 = MagicMock()
        node2.metadata.name = "west-node"
        node2.metadata.labels = {
            "topology.kubernetes.io/region": "us-west-2",
        }
        node2.status.allocatable = {"cpu": "4", "memory": "16Gi"}
        node2.status.conditions = []

        nodes_result = MagicMock()
        nodes_result.items = [node1, node2]
        mock_core = MagicMock()
        mock_core.list_node.return_value = nodes_result
        provider._core_api = mock_core

        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            result = await provider._list_instances(region="us-east")
        assert len(result) == 1
        assert result[0].name == "east-node"


class TestKubernetesHelpers:
    def test_parse_k8s_cpu_plain(self):
        from artenic_ai_platform_providers.kubernetes import (
            _parse_k8s_cpu,
        )

        assert _parse_k8s_cpu("4") == 4

    def test_parse_k8s_cpu_millicores(self):
        from artenic_ai_platform_providers.kubernetes import (
            _parse_k8s_cpu,
        )

        assert _parse_k8s_cpu("4000m") == 4
        assert _parse_k8s_cpu("500m") == 1

    def test_parse_k8s_cpu_float(self):
        from artenic_ai_platform_providers.kubernetes import (
            _parse_k8s_cpu,
        )

        assert _parse_k8s_cpu("3.5") == 3

    def test_parse_k8s_cpu_invalid(self):
        from artenic_ai_platform_providers.kubernetes import (
            _parse_k8s_cpu,
        )

        assert _parse_k8s_cpu("abc") == 0

    def test_parse_k8s_memory_gi(self):
        from artenic_ai_platform_providers.kubernetes import (
            _parse_k8s_memory_gb,
        )

        assert _parse_k8s_memory_gb("16Gi") == 16.0

    def test_parse_k8s_memory_mi(self):
        from artenic_ai_platform_providers.kubernetes import (
            _parse_k8s_memory_gb,
        )

        assert _parse_k8s_memory_gb("16384Mi") == 16.0

    def test_parse_k8s_memory_bytes(self):
        from artenic_ai_platform_providers.kubernetes import (
            _parse_k8s_memory_gb,
        )

        assert _parse_k8s_memory_gb("17179869184") == 16.0

    def test_parse_k8s_memory_ti(self):
        from artenic_ai_platform_providers.kubernetes import (
            _parse_k8s_memory_gb,
        )

        assert _parse_k8s_memory_gb("1Ti") == 1024.0

    def test_parse_k8s_memory_invalid(self):
        from artenic_ai_platform_providers.kubernetes import (
            _parse_k8s_memory_gb,
        )

        assert _parse_k8s_memory_gb("abc") == 0.0

    def test_is_node_ready_true(self):
        from artenic_ai_platform_providers.kubernetes import (
            _is_node_ready,
        )

        node = MagicMock()
        cond = MagicMock()
        cond.type = "Ready"
        cond.status = "True"
        node.status.conditions = [cond]
        assert _is_node_ready(node) is True

    def test_is_node_ready_false(self):
        from artenic_ai_platform_providers.kubernetes import (
            _is_node_ready,
        )

        node = MagicMock()
        cond = MagicMock()
        cond.type = "Ready"
        cond.status = "False"
        node.status.conditions = [cond]
        assert _is_node_ready(node) is False

    def test_is_node_ready_no_conditions(self):
        from artenic_ai_platform_providers.kubernetes import (
            _is_node_ready,
        )

        node = MagicMock()
        node.status.conditions = None
        assert _is_node_ready(node) is False

    def test_parse_job_status_complete(self):
        from artenic_ai_platform_providers.kubernetes import (
            _parse_job_status,
        )

        cond = MagicMock()
        cond.type = "Complete"
        cond.status = "True"
        job = MagicMock()
        job.status.conditions = [cond]
        assert _parse_job_status(job) == JobStatus.COMPLETED

    def test_parse_job_status_failed(self):
        from artenic_ai_platform_providers.kubernetes import (
            _parse_job_status,
        )

        cond = MagicMock()
        cond.type = "Failed"
        cond.status = "True"
        job = MagicMock()
        job.status.conditions = [cond]
        assert _parse_job_status(job) == JobStatus.FAILED

    def test_parse_job_status_active(self):
        from artenic_ai_platform_providers.kubernetes import (
            _parse_job_status,
        )

        job = MagicMock()
        job.status.conditions = []
        job.status.active = 1
        job.status.succeeded = 0
        job.status.failed = 0
        assert _parse_job_status(job) == JobStatus.RUNNING

    def test_parse_job_status_none(self):
        from artenic_ai_platform_providers.kubernetes import (
            _parse_job_status,
        )

        job = MagicMock()
        job.status = None
        assert _parse_job_status(job) == JobStatus.PENDING

    def test_safe_json_valid(self):
        from artenic_ai_platform_providers.kubernetes import (
            _safe_json,
        )

        result = _safe_json({"key": "value"})
        assert '"key"' in result
        assert '"value"' in result

    def test_safe_json_non_serializable(self):
        from artenic_ai_platform_providers.kubernetes import (
            _safe_json,
        )

        result = _safe_json({"key": object()})
        assert isinstance(result, str)


# ======================================================================
# Additional tests for 100 % coverage
# ======================================================================


class TestKubernetesProvisionEnvVarsAndCodeConfigmap:
    """Cover lines 245, 262, 279: env vars, code configmap mount and volume."""

    @pytest.mark.asyncio
    async def test_provision_with_env_vars(self, provider):
        mock_batch = MagicMock()
        mock_core = MagicMock()
        provider._batch_api = mock_batch
        provider._core_api = mock_core

        spec = _spec(config={"env": {"LR": "0.001", "EPOCHS": "10"}})
        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            job_id = await provider._provision_and_start(spec)
        assert job_id in provider._jobs

    @pytest.mark.asyncio
    async def test_provision_with_code_configmap(self, provider):
        mock_batch = MagicMock()
        mock_core = MagicMock()
        provider._batch_api = mock_batch
        provider._core_api = mock_core

        spec = _spec(config={"_code_configmap": "artenic-code-abc123"})
        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            job_id = await provider._provision_and_start(spec)
        assert job_id in provider._jobs


class TestKubernetesCollectArtifactsPodListError:
    """Cover lines 452-453: ApiException when listing pods."""

    @pytest.mark.asyncio
    async def test_collect_artifacts_pod_list_error(self, provider):
        from artenic_ai_platform_providers.kubernetes import (
            _K8sJobState,
        )

        provider._jobs["j-list-err"] = _K8sJobState(
            job_name="artenic-j-list-err",
            pvc_name="pvc-j-list-err",
            created_at=time.time(),
            spec=_spec(),
        )
        mock_core = MagicMock()
        mock_core.list_namespaced_pod.side_effect = _FakeApiError(status=500, reason="internal")
        provider._core_api = mock_core

        with (
            patch(
                "asyncio.to_thread",
                new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
            ),
            patch(
                f"{_MODULE}.ApiException",
                _FakeApiError,
            ),
        ):
            result = await provider._collect_artifacts(
                "j-list-err",
                CloudJobStatus(
                    provider_job_id="j-list-err",
                    status=JobStatus.COMPLETED,
                ),
            )
        # Should still return the PVC URI even if pod listing fails
        assert result is not None
        assert "pvc-j-list-err" in result


class TestKubernetesGPUParseError:
    """Cover lines 580-581: GPU count parse ValueError."""

    @pytest.mark.asyncio
    async def test_list_instances_invalid_gpu_count(self, provider):
        node = MagicMock()
        node.metadata.name = "bad-gpu-node"
        node.metadata.labels = {}
        node.status.allocatable = {
            "cpu": "8",
            "memory": "32Gi",
            "nvidia.com/gpu": "invalid",
        }
        node.status.conditions = []

        nodes_result = MagicMock()
        nodes_result.items = [node]
        mock_core = MagicMock()
        mock_core.list_node.return_value = nodes_result
        provider._core_api = mock_core

        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            result = await provider._list_instances()
        assert len(result) == 1
        assert result[0].gpu_count == 0


class TestKubernetesPVCCreationError:
    """Cover line 662: PVC creation non-409 ApiException re-raises."""

    @pytest.mark.asyncio
    async def test_pvc_creation_non_409_raises(self, provider):
        mock_core = MagicMock()
        mock_core.create_namespaced_persistent_volume_claim.side_effect = _FakeApiError(
            status=500, reason="server error"
        )
        provider._core_api = mock_core

        with (
            patch(
                "asyncio.to_thread",
                new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
            ),
            patch(
                f"{_MODULE}.ApiException",
                _FakeApiError,
            ),
            pytest.raises(_FakeApiError),
        ):
            await provider._create_pvc("test-pvc", "20Gi")


class TestKubernetesMemoryKi:
    """Cover line 722: _parse_k8s_memory_gb with Ki suffix."""

    def test_parse_k8s_memory_ki(self):
        from artenic_ai_platform_providers.kubernetes import (
            _parse_k8s_memory_gb,
        )

        # 16 GiB = 16 * 1024 * 1024 Ki = 16777216 Ki
        assert _parse_k8s_memory_gb("16777216Ki") == 16.0


class TestKubernetesParseJobStatusSucceededAndFailed:
    """Cover lines 764-771: _parse_job_status with succeeded/failed counts."""

    def test_parse_job_status_succeeded_count(self):
        from artenic_ai_platform_providers.kubernetes import (
            _parse_job_status,
        )

        job = MagicMock()
        job.status.conditions = []
        job.status.active = 0
        job.status.succeeded = 1
        job.status.failed = 0
        assert _parse_job_status(job) == JobStatus.COMPLETED

    def test_parse_job_status_failed_count(self):
        from artenic_ai_platform_providers.kubernetes import (
            _parse_job_status,
        )

        job = MagicMock()
        job.status.conditions = []
        job.status.active = 0
        job.status.succeeded = 0
        job.status.failed = 1
        assert _parse_job_status(job) == JobStatus.FAILED

    def test_parse_job_status_no_conditions_no_counts(self):
        from artenic_ai_platform_providers.kubernetes import (
            _parse_job_status,
        )

        job = MagicMock()
        job.status.conditions = []
        job.status.active = 0
        job.status.succeeded = 0
        job.status.failed = 0
        assert _parse_job_status(job) == JobStatus.PENDING


class TestKubernetesSafeJsonError:
    """Cover lines 780-781: _safe_json TypeError/ValueError branch."""

    def test_safe_json_type_error(self):
        import json as _json

        from artenic_ai_platform_providers.kubernetes import (
            _safe_json,
        )

        # Patch json.dumps to raise TypeError
        original = _json.dumps
        try:
            _json.dumps = MagicMock(side_effect=TypeError("not serializable"))
            result = _safe_json({"key": "value"})
            assert result == "{}"
        finally:
            _json.dumps = original


class TestKubernetesIsNodeReadyNoReadyCondition:
    """Cover the case where no Ready condition exists at all."""

    def test_is_node_ready_no_ready_condition(self):
        from artenic_ai_platform_providers.kubernetes import (
            _is_node_ready,
        )

        node = MagicMock()
        cond = MagicMock()
        cond.type = "DiskPressure"
        cond.status = "False"
        node.status.conditions = [cond]
        assert _is_node_ready(node) is False

    def test_is_node_ready_no_status(self):
        from artenic_ai_platform_providers.kubernetes import (
            _is_node_ready,
        )

        node = MagicMock()
        node.status = None
        assert _is_node_ready(node) is False
