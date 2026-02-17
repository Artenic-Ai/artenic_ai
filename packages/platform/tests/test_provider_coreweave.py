"""Tests for artenic_ai_platform.providers.coreweave — CoreWeaveProvider."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from artenic_ai_platform.providers.base import (
    CloudJobStatus,
    JobStatus,
    TrainingSpec,
)

_MODULE = "artenic_ai_platform.providers.coreweave"


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _spec(**overrides) -> TrainingSpec:
    defaults = {
        "service": "training",
        "model": "llama-7b",
        "provider": "coreweave",
        "config": {},
    }
    defaults.update(overrides)
    return TrainingSpec(**defaults)


def _make_k8s_mock():
    """Build a mock kubernetes SDK module hierarchy."""
    mock_k8s_client = MagicMock()
    mock_k8s_config = MagicMock()
    # Ensure V1* constructors return MagicMock instances
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
    """Lightweight stand-in for kubernetes.client.rest.ApiException."""

    def __init__(self, status=500, reason="error"):
        self.status = status
        self.reason = reason
        super().__init__(reason)


# ---------------------------------------------------------------
# Module-level patches applied to every test
# ---------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_k8s():
    """Mock the kubernetes SDK at the module level."""
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
    from artenic_ai_platform.providers.coreweave import (
        CoreWeaveProvider,
    )

    return CoreWeaveProvider(
        training_image="registry.example.com/train:latest",
        namespace="test-ns",
    )


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------


class TestCoreWeaveInit:
    def test_provider_name(self, provider):
        assert provider.provider_name == "coreweave"

    def test_init_stores_fields(self, provider):
        assert provider._namespace == "test-ns"
        assert provider._training_image == "registry.example.com/train:latest"
        assert provider._storage_class == "shared-hdd-ord1"
        assert provider._connected is False

    def test_init_raises_without_k8s(self):
        with patch(f"{_MODULE}._HAS_K8S", False), pytest.raises(ImportError, match="kubernetes"):
            from artenic_ai_platform.providers.coreweave import (
                CoreWeaveProvider,
            )

            CoreWeaveProvider(
                training_image="img",
            )


class TestCoreWeaveConnect:
    @pytest.mark.asyncio
    async def test_connect_loads_kubeconfig(self, provider, _patch_k8s):
        mock_client, _mock_config = _patch_k8s
        mock_core = MagicMock()
        mock_batch = MagicMock()
        mock_api = MagicMock()
        mock_client.ApiClient.return_value = mock_api
        mock_client.CoreV1Api.return_value = mock_core
        mock_client.BatchV1Api.return_value = mock_batch

        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            await provider._connect()

        assert provider._core_api is mock_core
        assert provider._batch_api is mock_batch

    @pytest.mark.asyncio
    async def test_connect_with_kubeconfig_path(self, _patch_k8s):
        mock_client, mock_config = _patch_k8s
        mock_client.ApiClient.return_value = MagicMock()
        mock_client.CoreV1Api.return_value = MagicMock()
        mock_client.BatchV1Api.return_value = MagicMock()

        from artenic_ai_platform.providers.coreweave import (
            CoreWeaveProvider,
        )

        p = CoreWeaveProvider(
            training_image="img",
            kubeconfig_path="/path/to/config",
        )
        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            await p._connect()
        mock_config.load_kube_config.assert_called_once_with(config_file="/path/to/config")

    @pytest.mark.asyncio
    async def test_disconnect(self, provider):
        mock_api_client = MagicMock()
        provider._api_client = mock_api_client
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
        mock_api_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_handles_error(self, provider):
        mock_api_client = MagicMock()
        mock_api_client.close.side_effect = RuntimeError("oops")
        provider._api_client = mock_api_client

        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            await provider._disconnect()
        assert provider._api_client is None


class TestCoreWeaveUploadCode:
    @pytest.mark.asyncio
    async def test_upload_creates_configmap(self, provider, _patch_k8s):
        mock_core = MagicMock()
        provider._core_api = mock_core

        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            result = await provider._upload_code(_spec())

        assert result.startswith("artenic-code-")
        mock_core.create_namespaced_config_map.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_includes_inline_code(self, provider, _patch_k8s):
        mock_core = MagicMock()
        provider._core_api = mock_core
        spec = _spec(config={"inline_code": "print('hello')"})
        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            await provider._upload_code(spec)
        mock_core.create_namespaced_config_map.assert_called_once()


class TestCoreWeaveProvision:
    @pytest.mark.asyncio
    async def test_provision_creates_job(self, provider, _patch_k8s):
        mock_batch = MagicMock()
        mock_core = MagicMock()
        provider._batch_api = mock_batch
        provider._core_api = mock_core

        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            job_id = await provider._provision_and_start(_spec())

        assert job_id.startswith("cw-")
        assert job_id in provider._jobs
        mock_batch.create_namespaced_job.assert_called_once()
        # PVC should also have been created
        mock_core.create_namespaced_persistent_volume_claim.assert_called_once()

    @pytest.mark.asyncio
    async def test_provision_with_custom_gpu_count(self, provider, _patch_k8s):
        mock_batch = MagicMock()
        mock_core = MagicMock()
        provider._batch_api = mock_batch
        provider._core_api = mock_core
        spec = _spec(config={"gpu_count": 4})

        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            job_id = await provider._provision_and_start(spec)

        state = provider._jobs[job_id]
        assert state.gpu_count == 4


class TestCoreWeavePoll:
    @pytest.mark.asyncio
    async def test_poll_unknown_job(self, provider):
        provider._batch_api = MagicMock()
        result = await provider._poll_provider("nope")
        assert result.status == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_poll_running_job(self, provider):
        from artenic_ai_platform.providers.coreweave import (
            _CoreWeaveJobState,
        )

        provider._jobs["j1"] = _CoreWeaveJobState(
            job_name="artenic-j1",
            pvc_name="pvc-j1",
            created_at=time.time() - 60,
            spec=_spec(),
            gpu_count=1,
        )
        mock_job = MagicMock()
        mock_job.status.conditions = []
        mock_job.status.active = 1
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

    @pytest.mark.asyncio
    async def test_poll_completed_job(self, provider):
        from artenic_ai_platform.providers.coreweave import (
            _CoreWeaveJobState,
        )

        provider._jobs["j2"] = _CoreWeaveJobState(
            job_name="artenic-j2",
            pvc_name="pvc-j2",
            created_at=time.time() - 300,
            spec=_spec(),
            gpu_count=1,
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
    async def test_poll_api_exception(self, provider):
        from artenic_ai_platform.providers.coreweave import (
            _CoreWeaveJobState,
        )

        provider._jobs["j3"] = _CoreWeaveJobState(
            job_name="artenic-j3",
            pvc_name="pvc-j3",
            created_at=time.time(),
            spec=_spec(),
            gpu_count=1,
        )
        mock_batch = MagicMock()
        mock_batch.read_namespaced_job.side_effect = _FakeApiError(status=404, reason="not found")
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
            result = await provider._poll_provider("j3")
        assert result.status == JobStatus.FAILED
        assert "Cannot read Job" in (result.error or "")


class TestCoreWeaveCollectArtifacts:
    @pytest.mark.asyncio
    async def test_collect_returns_pvc_uri(self, provider):
        from artenic_ai_platform.providers.coreweave import (
            _CoreWeaveJobState,
        )

        provider._jobs["j1"] = _CoreWeaveJobState(
            job_name="artenic-j1",
            pvc_name="pvc-j1",
            created_at=time.time(),
            spec=_spec(),
            gpu_count=1,
        )
        mock_core = MagicMock()
        pods_result = MagicMock()
        pod = MagicMock()
        pod.metadata.name = "pod-1"
        pods_result.items = [pod]
        mock_core.list_namespaced_pod.return_value = pods_result
        mock_core.read_namespaced_pod_log.return_value = "log data"
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
        assert result == "pvc://test-ns/pvc-j1/artifacts"

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


class TestCoreWeaveCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_deletes_job(self, provider):
        from artenic_ai_platform.providers.coreweave import (
            _CoreWeaveJobState,
        )

        provider._jobs["j1"] = _CoreWeaveJobState(
            job_name="artenic-j1",
            pvc_name="pvc-j1",
            created_at=time.time(),
            spec=_spec(),
            gpu_count=1,
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
        from artenic_ai_platform.providers.coreweave import (
            _CoreWeaveJobState,
        )

        provider._jobs["j2"] = _CoreWeaveJobState(
            job_name="artenic-j2",
            pvc_name="pvc-j2",
            created_at=time.time(),
            spec=_spec(),
            gpu_count=1,
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
        from artenic_ai_platform.providers.coreweave import (
            _CoreWeaveJobState,
        )

        provider._jobs["j3"] = _CoreWeaveJobState(
            job_name="artenic-j3",
            pvc_name="pvc-j3",
            created_at=time.time(),
            spec=_spec(),
            gpu_count=1,
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


class TestCoreWeaveCancel:
    @pytest.mark.asyncio
    async def test_cancel_deletes_job(self, provider):
        from artenic_ai_platform.providers.coreweave import (
            _CoreWeaveJobState,
        )

        provider._jobs["j1"] = _CoreWeaveJobState(
            job_name="artenic-j1",
            pvc_name="pvc-j1",
            created_at=time.time(),
            spec=_spec(),
            gpu_count=1,
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
        from artenic_ai_platform.providers.coreweave import (
            _CoreWeaveJobState,
        )

        provider._jobs["j2"] = _CoreWeaveJobState(
            job_name="artenic-j2",
            pvc_name="pvc-j2",
            created_at=time.time(),
            spec=_spec(),
            gpu_count=1,
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
            await provider._cancel_provider_job("j2")


class TestCoreWeaveListInstances:
    @pytest.mark.asyncio
    async def test_list_instances_parses_nodes(self, provider):
        node = MagicMock()
        node.metadata.name = "gpu-node-01"
        node.metadata.labels = {
            "topology.kubernetes.io/region": "ORD1",
            "gpu.nvidia.com/class": "A100_PCIE_80GB",
        }
        node.status.allocatable = {
            "cpu": "32",
            "memory": "128Gi",
            "nvidia.com/gpu": "4",
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
        assert inst.name == "gpu-node-01"
        assert inst.gpu_type == "A100_PCIE_80GB"
        assert inst.gpu_count == 4
        assert inst.vcpus == 32
        assert inst.memory_gb == 128.0
        assert inst.available is True
        # Price should be per-GPU * count
        assert inst.price_per_hour_eur == round(2.21 * 4, 4)

    @pytest.mark.asyncio
    async def test_list_instances_gpu_only_filter(self, provider):
        node = MagicMock()
        node.metadata.name = "cpu-node"
        node.metadata.labels = {}
        node.status.allocatable = {
            "cpu": "16",
            "memory": "64Gi",
            "nvidia.com/gpu": "0",
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


class TestCoreWeaveEstimateCost:
    def test_estimate_with_known_gpu(self, provider):
        from artenic_ai_platform.providers.coreweave import (
            _CoreWeaveJobState,
        )

        state = _CoreWeaveJobState(
            job_name="j",
            pvc_name="p",
            created_at=0,
            spec=_spec(
                config={
                    "node_selector": {
                        "gpu.nvidia.com/class": "A100_PCIE_80GB",
                    },
                }
            ),
            gpu_count=2,
        )
        cost = provider._estimate_cost(state, 3600.0)
        assert cost == round(2.21 * 2, 4)

    def test_estimate_with_unknown_gpu(self, provider):
        from artenic_ai_platform.providers.coreweave import (
            _CoreWeaveJobState,
        )

        state = _CoreWeaveJobState(
            job_name="j",
            pvc_name="p",
            created_at=0,
            spec=_spec(config={}),
            gpu_count=1,
        )
        assert provider._estimate_cost(state, 3600.0) is None


# ---------------------------------------------------------------
# Additional tests for full coverage
# ---------------------------------------------------------------


class TestCoreWeaveConnectFallback:
    """Cover _connect fallback from in-cluster to default kubeconfig (lines 138-139)."""

    @pytest.mark.asyncio
    async def test_connect_incluster_fails_loads_kubeconfig(self, _patch_k8s):
        mock_client, mock_config = _patch_k8s
        mock_client.ApiClient.return_value = MagicMock()
        mock_client.CoreV1Api.return_value = MagicMock()
        mock_client.BatchV1Api.return_value = MagicMock()

        # Make load_incluster_config raise ConfigException
        mock_config.load_incluster_config.side_effect = mock_config.ConfigException(
            "not in cluster"
        )

        from artenic_ai_platform.providers.coreweave import CoreWeaveProvider

        p = CoreWeaveProvider(training_image="img", namespace="test-ns")

        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            await p._connect()

        # Should have fallen back to load_kube_config
        mock_config.load_kube_config.assert_called_once()


class TestCoreWeaveProvisionEdgeCases:
    """Cover provision edge cases: node_selector merge, env vars, code_configmap."""

    @pytest.mark.asyncio
    async def test_provision_with_spec_node_selector(self, provider, _patch_k8s):
        """spec.config node_selector merges with provider node_selector (line 233)."""
        mock_batch = MagicMock()
        mock_core = MagicMock()
        provider._batch_api = mock_batch
        provider._core_api = mock_core
        provider._node_selector = {"gpu.nvidia.com/class": "A100_PCIE_80GB"}

        spec = _spec(
            config={
                "node_selector": {"topology.kubernetes.io/region": "ORD1"},
            }
        )

        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            job_id = await provider._provision_and_start(spec)

        assert job_id in provider._jobs

    @pytest.mark.asyncio
    async def test_provision_with_env_vars(self, provider, _patch_k8s):
        """Extra env vars are added to the container (line 246)."""
        mock_batch = MagicMock()
        mock_core = MagicMock()
        provider._batch_api = mock_batch
        provider._core_api = mock_core

        spec = _spec(
            config={
                "env": {"LEARNING_RATE": "0.001", "BATCH_SIZE": "32"},
            }
        )

        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            job_id = await provider._provision_and_start(spec)

        assert job_id in provider._jobs

    @pytest.mark.asyncio
    async def test_provision_with_code_configmap(self, provider, _patch_k8s):
        """When _code_configmap is set, volume mount is added (lines 263, 280)."""
        mock_batch = MagicMock()
        mock_core = MagicMock()
        provider._batch_api = mock_batch
        provider._core_api = mock_core

        spec = _spec(
            config={
                "_code_configmap": "artenic-code-abc123",
            }
        )

        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
        ):
            job_id = await provider._provision_and_start(spec)

        assert job_id in provider._jobs


class TestCoreWeaveCollectArtifactsEdgeCases:
    """Cover artifact collection edge cases (lines 447-453)."""

    @pytest.mark.asyncio
    async def test_collect_pod_log_api_exception(self, provider, _patch_k8s):
        """ApiException reading pod logs is silently caught (lines 447-451)."""
        from artenic_ai_platform.providers.coreweave import _CoreWeaveJobState

        provider._jobs["j-log"] = _CoreWeaveJobState(
            job_name="artenic-j-log",
            pvc_name="pvc-j-log",
            created_at=time.time(),
            spec=_spec(),
            gpu_count=1,
        )
        mock_core = MagicMock()
        pods_result = MagicMock()
        pod = MagicMock()
        pod.metadata.name = "pod-log"
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
            patch(f"{_MODULE}.ApiException", _FakeApiError),
        ):
            result = await provider._collect_artifacts(
                "j-log",
                CloudJobStatus(provider_job_id="j-log", status=JobStatus.COMPLETED),
            )

        # Should still return the PVC URI
        assert result == "pvc://test-ns/pvc-j-log/artifacts"

    @pytest.mark.asyncio
    async def test_collect_list_pods_api_exception(self, provider, _patch_k8s):
        """ApiException listing pods is silently caught (lines 452-453)."""
        from artenic_ai_platform.providers.coreweave import _CoreWeaveJobState

        provider._jobs["j-pods"] = _CoreWeaveJobState(
            job_name="artenic-j-pods",
            pvc_name="pvc-j-pods",
            created_at=time.time(),
            spec=_spec(),
            gpu_count=1,
        )
        mock_core = MagicMock()
        mock_core.list_namespaced_pod.side_effect = _FakeApiError(status=500, reason="internal")
        provider._core_api = mock_core

        with (
            patch(
                "asyncio.to_thread",
                new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
            ),
            patch(f"{_MODULE}.ApiException", _FakeApiError),
        ):
            result = await provider._collect_artifacts(
                "j-pods",
                CloudJobStatus(provider_job_id="j-pods", status=JobStatus.COMPLETED),
            )

        assert result == "pvc://test-ns/pvc-j-pods/artifacts"


class TestCoreWeaveCancelNon404:
    """Cover cancel non-404 ApiException (lines 533-538)."""

    @pytest.mark.asyncio
    async def test_cancel_non_404_raises(self, provider, _patch_k8s):
        from artenic_ai_platform.providers.coreweave import _CoreWeaveJobState

        provider._jobs["j-c500"] = _CoreWeaveJobState(
            job_name="artenic-j-c500",
            pvc_name="pvc-j-c500",
            created_at=time.time(),
            spec=_spec(),
            gpu_count=1,
        )
        mock_batch = MagicMock()
        mock_batch.delete_namespaced_job.side_effect = _FakeApiError(status=500, reason="internal")
        provider._batch_api = mock_batch

        with (
            patch(
                "asyncio.to_thread",
                new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
            ),
            patch(f"{_MODULE}.ApiException", _FakeApiError),
            pytest.raises(_FakeApiError),
        ):
            await provider._cancel_provider_job("j-c500")


class TestCoreWeaveListInstancesEdgeCases:
    """Cover list_instances region filtering and GPU parsing edge cases."""

    @pytest.mark.asyncio
    async def test_list_instances_region_filter(self, provider):
        """Nodes in a non-matching region are excluded (line 572)."""
        node = MagicMock()
        node.metadata.name = "gpu-node-eu"
        node.metadata.labels = {
            "topology.kubernetes.io/region": "EU-1",
            "gpu.nvidia.com/class": "A100_PCIE_80GB",
        }
        node.status.allocatable = {
            "cpu": "32",
            "memory": "128Gi",
            "nvidia.com/gpu": "4",
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
            result = await provider._list_instances(region="ORD")

        # EU node should not match ORD region filter
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_list_instances_invalid_gpu_count(self, provider):
        """Invalid gpu_count string defaults to 0 (lines 579-580)."""
        node = MagicMock()
        node.metadata.name = "bad-gpu-node"
        node.metadata.labels = {
            "gpu.nvidia.com/class": "A100_PCIE_80GB",
        }
        node.status.allocatable = {
            "cpu": "16",
            "memory": "64Gi",
            "nvidia.com/gpu": "not-a-number",
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

    @pytest.mark.asyncio
    async def test_list_instances_no_gpu_class(self, provider):
        """Nodes without gpu class label have None gpu_type."""
        node = MagicMock()
        node.metadata.name = "no-gpu-node"
        node.metadata.labels = {}
        node.status.allocatable = {
            "cpu": "16",
            "memory": "64Gi",
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
        assert result[0].gpu_type is None
        assert result[0].gpu_count == 0

    @pytest.mark.asyncio
    async def test_list_instances_region_from_zone_label(self, provider):
        """Fallback to topology.kubernetes.io/zone when region label is absent."""
        node = MagicMock()
        node.metadata.name = "zone-node"
        node.metadata.labels = {
            "topology.kubernetes.io/zone": "ORD1-a",
        }
        node.status.allocatable = {
            "cpu": "8",
            "memory": "32Gi",
            "nvidia.com/gpu": "1",
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
            result = await provider._list_instances(region="ORD")

        # ORD1-a starts with ORD, so should match
        assert len(result) == 1
        assert result[0].region == "ORD1-a"


class TestCoreWeaveCreatePVC:
    """Cover _create_pvc exception handling (lines 652-656)."""

    @pytest.mark.asyncio
    async def test_create_pvc_conflict_409(self, provider, _patch_k8s):
        """409 conflict means PVC already exists and is silently ignored."""
        mock_core = MagicMock()
        mock_core.create_namespaced_persistent_volume_claim.side_effect = _FakeApiError(
            status=409, reason="AlreadyExists"
        )
        provider._core_api = mock_core

        with (
            patch(
                "asyncio.to_thread",
                new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
            ),
            patch(f"{_MODULE}.ApiException", _FakeApiError),
        ):
            await provider._create_pvc("pvc-test", "50Gi")

    @pytest.mark.asyncio
    async def test_create_pvc_other_error_raises(self, provider, _patch_k8s):
        """Non-409 ApiException is re-raised."""
        mock_core = MagicMock()
        mock_core.create_namespaced_persistent_volume_claim.side_effect = _FakeApiError(
            status=500, reason="internal"
        )
        provider._core_api = mock_core

        with (
            patch(
                "asyncio.to_thread",
                new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
            ),
            patch(f"{_MODULE}.ApiException", _FakeApiError),
            pytest.raises(_FakeApiError),
        ):
            await provider._create_pvc("pvc-test", "50Gi")


class TestCoreWeaveParseJobStatus:
    """Cover _parse_job_status edge cases (lines 663, 669-673, 680-687)."""

    def test_status_is_none(self, provider):
        """When job.status is None, return PENDING (line 663)."""
        job = MagicMock()
        job.status = None
        assert provider._parse_job_status(job) == JobStatus.PENDING

    def test_failed_condition_deadline_exceeded(self, provider):
        """Failed with DeadlineExceeded reason (lines 669-672)."""
        cond = MagicMock()
        cond.type = "Failed"
        cond.status = "True"
        cond.reason = "DeadlineExceeded"
        job = MagicMock()
        job.status.conditions = [cond]
        assert provider._parse_job_status(job) == JobStatus.FAILED

    def test_failed_condition_other_reason(self, provider):
        """Failed condition with a different reason (line 673)."""
        cond = MagicMock()
        cond.type = "Failed"
        cond.status = "True"
        cond.reason = "BackoffLimitExceeded"
        job = MagicMock()
        job.status.conditions = [cond]
        assert provider._parse_job_status(job) == JobStatus.FAILED

    def test_succeeded_count(self, provider):
        """Succeeded count > 0 means COMPLETED (lines 680-681)."""
        job = MagicMock()
        job.status.conditions = []
        job.status.active = 0
        job.status.succeeded = 1
        job.status.failed = 0
        assert provider._parse_job_status(job) == JobStatus.COMPLETED

    def test_failed_count(self, provider):
        """Failed count > 0 means FAILED (lines 684-685)."""
        job = MagicMock()
        job.status.conditions = []
        job.status.active = 0
        job.status.succeeded = 0
        job.status.failed = 2
        assert provider._parse_job_status(job) == JobStatus.FAILED

    def test_default_pending(self, provider):
        """No conditions, no active/succeeded/failed means PENDING (line 687)."""
        job = MagicMock()
        job.status.conditions = []
        job.status.active = 0
        job.status.succeeded = 0
        job.status.failed = 0
        assert provider._parse_job_status(job) == JobStatus.PENDING

    def test_failed_condition_no_reason(self, provider):
        """Failed condition with None reason."""
        cond = MagicMock()
        cond.type = "Failed"
        cond.status = "True"
        cond.reason = None
        job = MagicMock()
        job.status.conditions = [cond]
        assert provider._parse_job_status(job) == JobStatus.FAILED


class TestCoreWeaveParseK8sCpu:
    """Cover _parse_k8s_cpu edge cases (lines 747, 750-754)."""

    def test_millicpu(self):
        from artenic_ai_platform.providers.coreweave import _parse_k8s_cpu

        assert _parse_k8s_cpu("4000m") == 4
        assert _parse_k8s_cpu("500m") == 1

    def test_integer_string(self):
        from artenic_ai_platform.providers.coreweave import _parse_k8s_cpu

        assert _parse_k8s_cpu("8") == 8

    def test_float_string(self):
        """Float CPU value (line 752)."""
        from artenic_ai_platform.providers.coreweave import _parse_k8s_cpu

        assert _parse_k8s_cpu("3.5") == 3

    def test_invalid_string(self):
        """Completely invalid CPU value returns 0 (line 754)."""
        from artenic_ai_platform.providers.coreweave import _parse_k8s_cpu

        assert _parse_k8s_cpu("abc") == 0

    def test_whitespace(self):
        from artenic_ai_platform.providers.coreweave import _parse_k8s_cpu

        assert _parse_k8s_cpu("  16  ") == 16


class TestCoreWeaveParseK8sMemoryGb:
    """Cover _parse_k8s_memory_gb edge cases (lines 764, 766, 769-775)."""

    def test_ki_suffix(self):
        """Ki suffix (line 764)."""
        from artenic_ai_platform.providers.coreweave import _parse_k8s_memory_gb

        # 16 * 1024 * 1024 Ki = 16 GB
        assert _parse_k8s_memory_gb("16777216Ki") == 16.0

    def test_mi_suffix(self):
        """Mi suffix (line 766)."""
        from artenic_ai_platform.providers.coreweave import _parse_k8s_memory_gb

        assert _parse_k8s_memory_gb("16384Mi") == 16.0

    def test_gi_suffix(self):
        from artenic_ai_platform.providers.coreweave import _parse_k8s_memory_gb

        assert _parse_k8s_memory_gb("64Gi") == 64.0

    def test_ti_suffix(self):
        """Ti suffix (line 769)."""
        from artenic_ai_platform.providers.coreweave import _parse_k8s_memory_gb

        assert _parse_k8s_memory_gb("1Ti") == 1024.0

    def test_plain_bytes(self):
        """Plain bytes (lines 772-773)."""
        from artenic_ai_platform.providers.coreweave import _parse_k8s_memory_gb

        # 17179869184 bytes = 16 GB
        assert _parse_k8s_memory_gb("17179869184") == 16.0

    def test_invalid_string(self):
        """Invalid memory string returns 0.0 (lines 774-775)."""
        from artenic_ai_platform.providers.coreweave import _parse_k8s_memory_gb

        assert _parse_k8s_memory_gb("invalid") == 0.0


class TestCoreWeaveIsNodeReady:
    """Cover _is_node_ready edge cases (lines 781, 785)."""

    def test_node_status_none(self):
        """Node with None status returns False (line 781)."""
        from artenic_ai_platform.providers.coreweave import _is_node_ready

        node = MagicMock()
        node.status = None
        assert _is_node_ready(node) is False

    def test_node_conditions_none(self):
        """Node with None conditions returns False (line 781)."""
        from artenic_ai_platform.providers.coreweave import _is_node_ready

        node = MagicMock()
        node.status.conditions = None
        assert _is_node_ready(node) is False

    def test_no_ready_condition(self):
        """Node without Ready condition returns False (line 785)."""
        from artenic_ai_platform.providers.coreweave import _is_node_ready

        cond = MagicMock()
        cond.type = "DiskPressure"
        cond.status = "False"
        node = MagicMock()
        node.status.conditions = [cond]
        assert _is_node_ready(node) is False

    def test_ready_false(self):
        """Node with Ready=False returns False."""
        from artenic_ai_platform.providers.coreweave import _is_node_ready

        cond = MagicMock()
        cond.type = "Ready"
        cond.status = "False"
        node = MagicMock()
        node.status.conditions = [cond]
        assert _is_node_ready(node) is False

    def test_ready_true(self):
        """Node with Ready=True returns True."""
        from artenic_ai_platform.providers.coreweave import _is_node_ready

        cond = MagicMock()
        cond.type = "Ready"
        cond.status = "True"
        node = MagicMock()
        node.status.conditions = [cond]
        assert _is_node_ready(node) is True


class TestCoreWeaveSafeJson:
    """Cover _safe_json edge cases (lines 794-795)."""

    def test_safe_json_normal(self):
        from artenic_ai_platform.providers.coreweave import _safe_json

        result = _safe_json({"key": "value", "num": 42})
        assert '"key"' in result
        assert '"value"' in result

    def test_safe_json_non_serializable(self):
        """Non-serializable values use str() fallback."""
        from artenic_ai_platform.providers.coreweave import _safe_json

        result = _safe_json({"obj": object()})
        assert result != "{}"  # default=str should handle it

    def test_safe_json_error_returns_empty(self):
        """When JSON serialization truly fails, return '{}'."""
        import json

        from artenic_ai_platform.providers.coreweave import _safe_json

        with patch.object(
            json,
            "dumps",
            side_effect=TypeError("cannot serialize"),
        ):
            result = _safe_json({"key": "value"})
        assert result == "{}"
