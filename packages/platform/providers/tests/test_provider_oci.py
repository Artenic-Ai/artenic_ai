"""Tests for artenic_ai_platform_providers.oci — OCIProvider."""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from artenic_ai_platform_providers.base import (
    CloudJobStatus,
    JobStatus,
    TrainingSpec,
)

# ---------------------------------------------------------------------------
# Module path and shared patches
# ---------------------------------------------------------------------------

_MODULE = "artenic_ai_platform_providers.oci"


def _to_thread_patch():
    """Create a fresh asyncio.to_thread patch for each fixture call."""
    return patch(
        "asyncio.to_thread",
        new=AsyncMock(
            side_effect=lambda fn, *a, **kw: fn(*a, **kw),
        ),
    )


def _make_provider(**overrides):
    """Create an OCIProvider with mocked SDK availability."""
    from artenic_ai_platform_providers.oci import OCIProvider

    defaults = {
        "compartment_id": "ocid1.compartment.oc1..test",
        "config_file": "~/.oci/config",
        "config_profile": "DEFAULT",
        "region": "eu-frankfurt-1",
        "bucket_name": "artenic-bucket",
        "namespace": "testns",
        "shape": "VM.Standard.E4.Flex",
        "subnet_id": "ocid1.subnet.oc1..test",
        "image_id": "ocid1.image.oc1..test",
        "availability_domain": "AD-1",
    }
    defaults.update(overrides)
    return OCIProvider(**defaults)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def mock_oci():
    """Build a comprehensive MagicMock for the oci SDK."""
    m = MagicMock()
    # config helpers
    m.config.from_file.return_value = {
        "region": "eu-frankfurt-1",
    }
    m.config.validate_config.return_value = None
    # clients
    m.core.ComputeClient.return_value = MagicMock()
    m.object_storage.ObjectStorageClient.return_value = MagicMock()
    m.core.models.LaunchInstanceDetails = MagicMock(
        return_value=MagicMock(),
    )
    m.identity.IdentityClient.return_value = MagicMock()
    # auth
    m.auth.signers.InstancePrincipalsSecurityTokenSigner = MagicMock(
        return_value=MagicMock(),
    )
    return m


@pytest.fixture
def mock_service_error():
    """A mock ServiceError class."""
    cls = type(
        "ServiceError",
        (Exception,),
        {
            "__init__": lambda self, status, code, headers, message: (
                Exception.__init__(self, message),
                setattr(self, "status", status),
                setattr(self, "code", code),
            )[-1],
        },
    )
    return cls


@pytest.fixture
def oci_patches(mock_oci, mock_service_error):
    """Apply all module-level patches for the OCI SDK."""
    patches = [
        patch(f"{_MODULE}._HAS_OCI", True),
        patch(f"{_MODULE}.oci", mock_oci),
        patch(f"{_MODULE}.ServiceError", mock_service_error),
        _to_thread_patch(),
    ]
    for p in patches:
        p.start()
    yield {
        "oci": mock_oci,
        "ServiceError": mock_service_error,
    }
    for p in patches:
        p.stop()


# ======================================================================
# Tests
# ======================================================================


class TestOCIInit:
    def test_init_defaults(self, oci_patches):
        provider = _make_provider()
        assert provider.provider_name == "oci"
        assert provider._connected is False
        assert provider._compartment_id == "ocid1.compartment.oc1..test"
        assert provider._bucket_name == "artenic-bucket"

    def test_init_raises_without_oci(self):
        with patch(f"{_MODULE}._HAS_OCI", False), pytest.raises(ImportError, match="oci"):
            _make_provider()


class TestOCIProviderName:
    def test_provider_name(self, oci_patches):
        provider = _make_provider()
        assert provider.provider_name == "oci"


class TestOCIConnect:
    async def test_connect_with_config_file(self, oci_patches):
        provider = _make_provider()
        await provider._connect()

        oci_sdk = oci_patches["oci"]
        oci_sdk.config.from_file.assert_called_once_with(
            file_location="~/.oci/config",
            profile_name="DEFAULT",
        )
        oci_sdk.config.validate_config.assert_called_once()
        oci_sdk.core.ComputeClient.assert_called_once()
        oci_sdk.object_storage.ObjectStorageClient.assert_called_once()

        assert provider._compute_client is not None
        assert provider._object_storage_client is not None

    async def test_connect_with_instance_principal(self, oci_patches):
        provider = _make_provider(config_file=None)
        await provider._connect()

        oci_sdk = oci_patches["oci"]
        signer_cls = oci_sdk.auth.signers
        signer_cls.InstancePrincipalsSecurityTokenSigner.assert_called_once()

    async def test_connect_resolves_namespace(self, oci_patches):
        provider = _make_provider(namespace=None)

        oci_sdk = oci_patches["oci"]
        os_client = oci_sdk.object_storage.ObjectStorageClient.return_value
        ns_response = MagicMock()
        ns_response.data = "auto-namespace"
        os_client.get_namespace.return_value = ns_response

        await provider._connect()
        assert provider._namespace == "auto-namespace"


class TestOCIDisconnect:
    async def test_disconnect_clears_clients(self, oci_patches):
        provider = _make_provider()
        provider._compute_client = MagicMock()
        provider._object_storage_client = MagicMock()
        provider._oci_config = {"region": "eu-frankfurt-1"}

        await provider._disconnect()

        assert provider._compute_client is None
        assert provider._object_storage_client is None
        assert provider._oci_config == {}


class TestOCIListInstances:
    async def test_list_instances(self, oci_patches):
        provider = _make_provider()
        await provider._connect()

        # Mock shape objects
        shape1 = MagicMock()
        shape1.shape = "VM.Standard.E4.Flex"
        shape1.ocpus = 4
        shape1.memory_in_gbs = 64.0
        shape1.gpus = 0
        shape1.gpu_description = ""

        shape2 = MagicMock()
        shape2.shape = "BM.GPU4.8"
        shape2.ocpus = 64
        shape2.memory_in_gbs = 2048.0
        shape2.gpus = 8
        shape2.gpu_description = "A100"

        response = MagicMock()
        response.data = [shape1, shape2]
        response.has_next_page = False
        provider._compute_client.list_shapes.return_value = response

        instances = await provider._list_instances()

        assert len(instances) == 2
        names = [i.name for i in instances]
        assert "VM.Standard.E4.Flex" in names
        assert "BM.GPU4.8" in names

        gpu_inst = next(i for i in instances if i.name == "BM.GPU4.8")
        assert gpu_inst.gpu_type == "A100"
        assert gpu_inst.gpu_count == 8

    async def test_list_instances_gpu_only(self, oci_patches):
        provider = _make_provider()
        await provider._connect()

        shape_cpu = MagicMock()
        shape_cpu.shape = "VM.Standard.E4.Flex"
        shape_cpu.ocpus = 4
        shape_cpu.memory_in_gbs = 64.0
        shape_cpu.gpus = 0
        shape_cpu.gpu_description = ""

        shape_gpu = MagicMock()
        shape_gpu.shape = "VM.GPU3.1"
        shape_gpu.ocpus = 6
        shape_gpu.memory_in_gbs = 90.0
        shape_gpu.gpus = 0  # SDK returns 0 but _GPU_SHAPES knows it
        shape_gpu.gpu_description = ""

        response = MagicMock()
        response.data = [shape_cpu, shape_gpu]
        response.has_next_page = False
        provider._compute_client.list_shapes.return_value = response

        instances = await provider._list_instances(gpu_only=True)

        # Only the GPU shape should be returned
        names = [i.name for i in instances]
        assert "VM.Standard.E4.Flex" not in names
        assert "VM.GPU3.1" in names

    async def test_list_instances_pagination(self, oci_patches):
        provider = _make_provider()
        await provider._connect()

        shape1 = MagicMock()
        shape1.shape = "VM.Standard.E4.Flex"
        shape1.ocpus = 4
        shape1.memory_in_gbs = 64.0
        shape1.gpus = 0
        shape1.gpu_description = ""

        shape2 = MagicMock()
        shape2.shape = "VM.Standard.E3.Flex"
        shape2.ocpus = 2
        shape2.memory_in_gbs = 32.0
        shape2.gpus = 0
        shape2.gpu_description = ""

        page1 = MagicMock()
        page1.data = [shape1]
        page1.has_next_page = True
        page1.next_page = "page2token"

        page2 = MagicMock()
        page2.data = [shape2]
        page2.has_next_page = False

        provider._compute_client.list_shapes.side_effect = [page1, page2]

        instances = await provider._list_instances()
        assert len(instances) == 2


class TestOCIUploadCode:
    async def test_upload_code(self, oci_patches, tmp_path):
        provider = _make_provider()
        provider._object_storage_client = MagicMock()

        (tmp_path / "train.py").write_text("print('train')")

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="oci",
            config={"source_dir": str(tmp_path)},
        )
        uri = await provider._upload_code(spec)

        assert "artenic-bucket" in uri
        assert "testns" in uri
        provider._object_storage_client.put_object.assert_called_once()


class TestOCIProvisionAndStart:
    async def test_provision_and_start(self, oci_patches):
        provider = _make_provider()
        await provider._connect()

        launch_response = MagicMock()
        launch_response.data.id = "ocid1.instance.oc1..new"
        provider._compute_client.launch_instance.return_value = launch_response

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="oci",
        )
        job_id = await provider._provision_and_start(spec)

        assert job_id.startswith("oci-")
        assert job_id in provider._jobs
        state = provider._jobs[job_id]
        assert state.instance_id == "ocid1.instance.oc1..new"

    async def test_provision_flex_shape(self, oci_patches):
        provider = _make_provider(shape="VM.Standard.E4.Flex")
        await provider._connect()

        launch_response = MagicMock()
        launch_response.data.id = "ocid1.instance.oc1..flex"
        provider._compute_client.launch_instance.return_value = launch_response

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="oci",
            config={"ocpus": 8, "memory_in_gbs": 128},
        )
        job_id = await provider._provision_and_start(spec)
        assert job_id.startswith("oci-")

    async def test_provision_spot(self, oci_patches):
        provider = _make_provider()
        await provider._connect()

        launch_response = MagicMock()
        launch_response.data.id = "ocid1.instance.oc1..spot"
        provider._compute_client.launch_instance.return_value = launch_response

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="oci",
            is_spot=True,
        )
        job_id = await provider._provision_and_start(spec)
        assert job_id.startswith("oci-")

    async def test_provision_resolves_ad(self, oci_patches):
        provider = _make_provider(availability_domain=None)
        await provider._connect()

        # Mock AD lookup
        oci_sdk = oci_patches["oci"]
        ad_response = MagicMock()
        ad_obj = MagicMock()
        ad_obj.name = "Uxxx:EU-FRANKFURT-1-AD-1"
        ad_response.data = [ad_obj]
        identity_client = MagicMock()
        identity_client.list_availability_domains.return_value = ad_response
        oci_sdk.identity.IdentityClient.return_value = identity_client

        launch_response = MagicMock()
        launch_response.data.id = "ocid1.instance.oc1..ad"
        provider._compute_client.launch_instance.return_value = launch_response

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="oci",
        )
        job_id = await provider._provision_and_start(spec)
        assert job_id.startswith("oci-")


class TestOCIPollProvider:
    async def test_poll_running(self, oci_patches):
        from artenic_ai_platform_providers.oci import _OCIJobState

        provider = _make_provider()
        provider._compute_client = MagicMock()
        provider._object_storage_client = MagicMock()

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="oci",
        )
        state = _OCIJobState(
            instance_id="ocid1.instance.oc1..run",
            instance_name="artenic-oci-run",
            created_at=time.time() - 60,
            spec=spec,
            code_uri="training/nlp/bert/code.tar.gz",
            output_prefix="artifacts/oci-run",
        )
        provider._jobs["oci-run"] = state

        inst_response = MagicMock()
        inst_response.data.lifecycle_state = "RUNNING"
        inst_response.data.time_created = None
        provider._compute_client.get_instance.return_value = inst_response

        # No status file
        se = oci_patches["ServiceError"]
        provider._object_storage_client.get_object.side_effect = se(
            404,
            "NotFound",
            {},
            "not found",
        )

        status = await provider._poll_provider("oci-run")
        assert status.status == JobStatus.RUNNING

    async def test_poll_completed_from_status(self, oci_patches):
        from artenic_ai_platform_providers.oci import _OCIJobState

        provider = _make_provider()
        provider._compute_client = MagicMock()
        provider._object_storage_client = MagicMock()

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="oci",
        )
        state = _OCIJobState(
            instance_id="ocid1.instance.oc1..done",
            instance_name="artenic-oci-done",
            created_at=time.time() - 120,
            spec=spec,
            code_uri="training/nlp/bert/code.tar.gz",
            output_prefix="artifacts/oci-done",
        )
        provider._jobs["oci-done"] = state

        inst_response = MagicMock()
        inst_response.data.lifecycle_state = "RUNNING"
        inst_response.data.time_created = None
        provider._compute_client.get_instance.return_value = inst_response

        # Status file says completed
        obj_response = MagicMock()
        obj_response.data.content.decode.return_value = json.dumps(
            {"status": "completed"},
        )
        provider._object_storage_client.get_object.return_value = obj_response

        status = await provider._poll_provider("oci-done")
        assert status.status == JobStatus.COMPLETED

    async def test_poll_no_state(self, oci_patches):
        provider = _make_provider()
        status = await provider._poll_provider("unknown")
        assert status.status == JobStatus.FAILED
        assert "No instance tracked" in (status.error or "")

    async def test_poll_instance_not_found(self, oci_patches):
        from artenic_ai_platform_providers.oci import _OCIJobState

        provider = _make_provider()
        provider._compute_client = MagicMock()

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="oci",
        )
        state = _OCIJobState(
            instance_id="ocid1.instance.oc1..gone",
            instance_name="artenic-oci-gone",
            created_at=time.time() - 60,
            spec=spec,
            code_uri="training/nlp/bert/code.tar.gz",
            output_prefix="artifacts/oci-gone",
        )
        provider._jobs["oci-gone"] = state

        provider._compute_client.get_instance.side_effect = Exception("404 NotAuthorizedOrNotFound")

        status = await provider._poll_provider("oci-gone")
        assert status.status == JobStatus.FAILED

    async def test_poll_terminated_spot(self, oci_patches):
        from artenic_ai_platform_providers.oci import _OCIJobState

        provider = _make_provider()
        provider._compute_client = MagicMock()
        provider._object_storage_client = MagicMock()

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="oci",
            is_spot=True,
        )
        state = _OCIJobState(
            instance_id="ocid1.instance.oc1..spot",
            instance_name="artenic-oci-spot",
            created_at=time.time() - 60,
            spec=spec,
            code_uri="training/nlp/bert/code.tar.gz",
            output_prefix="artifacts/oci-spot",
        )
        provider._jobs["oci-spot"] = state

        inst_response = MagicMock()
        inst_response.data.lifecycle_state = "TERMINATED"
        inst_response.data.time_created = None
        provider._compute_client.get_instance.return_value = inst_response

        # No status file
        se = oci_patches["ServiceError"]
        provider._object_storage_client.get_object.side_effect = se(
            404,
            "NotFound",
            {},
            "not found",
        )

        status = await provider._poll_provider("oci-spot")
        assert status.status == JobStatus.PREEMPTED


class TestOCICollectArtifacts:
    async def test_collect_artifacts(self, oci_patches):
        from artenic_ai_platform_providers.oci import _OCIJobState

        provider = _make_provider()
        provider._object_storage_client = MagicMock()

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="oci",
        )
        state = _OCIJobState(
            instance_id="ocid1.instance.oc1..art",
            instance_name="artenic-oci-art",
            created_at=time.time(),
            spec=spec,
            code_uri="training/nlp/bert/code.tar.gz",
            output_prefix="artifacts/oci-art",
        )
        provider._jobs["oci-art"] = state

        # Mock list_objects
        obj_summary = MagicMock()
        obj_summary.name = "artifacts/oci-art/output/model.pt"
        list_response = MagicMock()
        list_response.data.objects = [obj_summary]
        provider._object_storage_client.list_objects.return_value = list_response

        # Mock get_object for download
        get_response = MagicMock()
        get_response.data.raw.stream.return_value = [b"model-data"]
        provider._object_storage_client.get_object.return_value = get_response

        dummy_status = CloudJobStatus(
            provider_job_id="oci-art",
            status=JobStatus.COMPLETED,
        )
        result = await provider._collect_artifacts("oci-art", dummy_status)
        assert result is not None

    async def test_collect_artifacts_no_state(self, oci_patches):
        provider = _make_provider()

        dummy_status = CloudJobStatus(
            provider_job_id="no-state",
            status=JobStatus.COMPLETED,
        )
        result = await provider._collect_artifacts(
            "no-state",
            dummy_status,
        )
        assert result is None

    async def test_collect_artifacts_empty(self, oci_patches):
        from artenic_ai_platform_providers.oci import _OCIJobState

        provider = _make_provider()
        provider._object_storage_client = MagicMock()

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="oci",
        )
        state = _OCIJobState(
            instance_id="ocid1.instance.oc1..empty",
            instance_name="artenic-oci-empty",
            created_at=time.time(),
            spec=spec,
            code_uri="training/nlp/bert/code.tar.gz",
            output_prefix="artifacts/oci-empty",
        )
        provider._jobs["oci-empty"] = state

        list_response = MagicMock()
        list_response.data.objects = []
        provider._object_storage_client.list_objects.return_value = list_response

        dummy_status = CloudJobStatus(
            provider_job_id="oci-empty",
            status=JobStatus.COMPLETED,
        )
        result = await provider._collect_artifacts(
            "oci-empty",
            dummy_status,
        )
        assert result is None


class TestOCICleanupCompute:
    async def test_cleanup_compute(self, oci_patches):
        from artenic_ai_platform_providers.oci import _OCIJobState

        provider = _make_provider()
        provider._compute_client = MagicMock()

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="oci",
        )
        state = _OCIJobState(
            instance_id="ocid1.instance.oc1..clean",
            instance_name="artenic-oci-clean",
            created_at=time.time(),
            spec=spec,
            code_uri="training/nlp/bert/code.tar.gz",
            output_prefix="artifacts/oci-clean",
        )
        provider._jobs["oci-clean"] = state

        await provider._cleanup_compute("oci-clean")

        provider._compute_client.terminate_instance.assert_called_once_with(
            "ocid1.instance.oc1..clean",
            preserve_boot_volume=False,
        )
        assert "oci-clean" not in provider._jobs

    async def test_cleanup_no_state(self, oci_patches):
        provider = _make_provider()
        # Should not raise
        await provider._cleanup_compute("missing")

    async def test_cleanup_already_terminated(self, oci_patches):
        from artenic_ai_platform_providers.oci import _OCIJobState

        provider = _make_provider()
        provider._compute_client = MagicMock()

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="oci",
        )
        state = _OCIJobState(
            instance_id="ocid1.instance.oc1..term",
            instance_name="artenic-oci-term",
            created_at=time.time(),
            spec=spec,
            code_uri="training/nlp/bert/code.tar.gz",
            output_prefix="artifacts/oci-term",
        )
        provider._jobs["oci-term"] = state

        se = oci_patches["ServiceError"]
        provider._compute_client.terminate_instance.side_effect = se(
            404,
            "NotFound",
            {},
            "not found",
        )

        # Should not raise — 404 is swallowed
        await provider._cleanup_compute("oci-term")


class TestOCICancelProviderJob:
    async def test_cancel_provider_job(self, oci_patches):
        from artenic_ai_platform_providers.oci import _OCIJobState

        provider = _make_provider()
        provider._compute_client = MagicMock()

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="oci",
        )
        state = _OCIJobState(
            instance_id="ocid1.instance.oc1..cancel",
            instance_name="artenic-oci-cancel",
            created_at=time.time(),
            spec=spec,
            code_uri="training/nlp/bert/code.tar.gz",
            output_prefix="artifacts/oci-cancel",
        )
        provider._jobs["oci-cancel"] = state

        await provider._cancel_provider_job("oci-cancel")

        provider._compute_client.terminate_instance.assert_called_once_with(
            "ocid1.instance.oc1..cancel",
            preserve_boot_volume=False,
        )

    async def test_cancel_no_state(self, oci_patches):
        provider = _make_provider()
        # Should not raise
        await provider._cancel_provider_job("ghost")

    async def test_cancel_already_terminated(self, oci_patches):
        from artenic_ai_platform_providers.oci import _OCIJobState

        provider = _make_provider()
        provider._compute_client = MagicMock()

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="oci",
        )
        state = _OCIJobState(
            instance_id="ocid1.instance.oc1..gone",
            instance_name="artenic-oci-gone",
            created_at=time.time(),
            spec=spec,
            code_uri="training/nlp/bert/code.tar.gz",
            output_prefix="artifacts/oci-gone",
        )
        provider._jobs["oci-gone"] = state

        se = oci_patches["ServiceError"]
        provider._compute_client.terminate_instance.side_effect = se(
            404,
            "NotFound",
            {},
            "not found",
        )

        # Should not raise
        await provider._cancel_provider_job("oci-gone")


# ======================================================================
# Additional tests for 100 % coverage
# ======================================================================


class TestOCIListInstancesGPUHeuristic:
    """Cover lines 424-427: GPU heuristic for shape names matching known prefix."""

    async def test_gpu_heuristic_prefix_match(self, oci_patches):
        provider = _make_provider()
        await provider._connect()

        # Shape name starts with a known GPU prefix but is not in _GPU_SHAPES directly.
        # E.g. "BM.GPU4.16" starts with "BM.GPU4." (from "BM.GPU4.8")
        shape_gpu = MagicMock()
        shape_gpu.shape = "BM.GPU4.16"
        shape_gpu.ocpus = 128
        shape_gpu.memory_in_gbs = 4096.0
        shape_gpu.gpus = 0  # SDK returns 0
        shape_gpu.gpu_description = ""  # No description

        response = MagicMock()
        response.data = [shape_gpu]
        response.has_next_page = False
        provider._compute_client.list_shapes.return_value = response

        instances = await provider._list_instances()
        assert len(instances) == 1
        # Should have resolved gpu_type from _GPU_SHAPES heuristic
        assert instances[0].gpu_type == "A100"
        assert instances[0].gpu_count == 8  # from BM.GPU4.8 fallback

    async def test_gpu_heuristic_with_existing_gpu_count(self, oci_patches):
        """Cover line 425-426: heuristic when gpu_count is already > 0."""
        provider = _make_provider()
        await provider._connect()

        shape_gpu = MagicMock()
        shape_gpu.shape = "BM.GPU4.16"
        shape_gpu.ocpus = 128
        shape_gpu.memory_in_gbs = 4096.0
        shape_gpu.gpus = 16  # SDK provides a count already
        shape_gpu.gpu_description = ""

        response = MagicMock()
        response.data = [shape_gpu]
        response.has_next_page = False
        provider._compute_client.list_shapes.return_value = response

        instances = await provider._list_instances()
        assert len(instances) == 1
        assert instances[0].gpu_type == "A100"
        # gpu_count stays at 16 since it was already nonzero
        assert instances[0].gpu_count == 16


class TestOCIProvisionSSHKey:
    """Cover line 554: ssh_public_key is set."""

    async def test_provision_with_ssh_key(self, oci_patches):
        provider = _make_provider(ssh_public_key="ssh-rsa AAAA...")
        await provider._connect()

        launch_response = MagicMock()
        launch_response.data.id = "ocid1.instance.oc1..sshkey"
        provider._compute_client.launch_instance.return_value = launch_response

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="oci",
        )
        job_id = await provider._provision_and_start(spec)
        assert job_id.startswith("oci-")


class TestOCIPollNon404Error:
    """Cover line 634: get_instance raises non-404 error (should re-raise)."""

    async def test_poll_unexpected_error_reraises(self, oci_patches):
        from artenic_ai_platform_providers.oci import _OCIJobState

        provider = _make_provider()
        provider._compute_client = MagicMock()

        spec = TrainingSpec(service="nlp", model="bert", provider="oci")
        state = _OCIJobState(
            instance_id="ocid1.instance.oc1..err",
            instance_name="artenic-oci-err",
            created_at=time.time() - 60,
            spec=spec,
            code_uri="training/nlp/bert/code.tar.gz",
            output_prefix="artifacts/oci-err",
        )
        provider._jobs["oci-err"] = state

        # Raise something that does NOT contain "404" or "NotAuthorizedOrNotFound"
        provider._compute_client.get_instance.side_effect = RuntimeError("connection refused")

        with pytest.raises(RuntimeError, match="connection refused"):
            await provider._poll_provider("oci-err")


class TestOCIPollFailedAndRunningFromStatus:
    """Cover lines 654-662: status 'failed' and 'running' from Object Storage."""

    async def test_poll_failed_from_status(self, oci_patches):
        from artenic_ai_platform_providers.oci import _OCIJobState

        provider = _make_provider()
        provider._compute_client = MagicMock()
        provider._object_storage_client = MagicMock()

        spec = TrainingSpec(service="nlp", model="bert", provider="oci")
        state = _OCIJobState(
            instance_id="ocid1.instance.oc1..fail",
            instance_name="artenic-oci-fail",
            created_at=time.time() - 120,
            spec=spec,
            code_uri="training/nlp/bert/code.tar.gz",
            output_prefix="artifacts/oci-fail",
        )
        provider._jobs["oci-fail"] = state

        inst_response = MagicMock()
        inst_response.data.lifecycle_state = "RUNNING"
        inst_response.data.time_created = None
        provider._compute_client.get_instance.return_value = inst_response

        obj_response = MagicMock()
        obj_response.data.content.decode.return_value = json.dumps(
            {"status": "failed", "exit_code": 1}
        )
        provider._object_storage_client.get_object.return_value = obj_response

        status = await provider._poll_provider("oci-fail")
        assert status.status == JobStatus.FAILED
        assert "1" in (status.error or "")

    async def test_poll_running_from_status(self, oci_patches):
        from artenic_ai_platform_providers.oci import _OCIJobState

        provider = _make_provider()
        provider._compute_client = MagicMock()
        provider._object_storage_client = MagicMock()

        spec = TrainingSpec(service="nlp", model="bert", provider="oci")
        state = _OCIJobState(
            instance_id="ocid1.instance.oc1..running",
            instance_name="artenic-oci-running",
            created_at=time.time() - 60,
            spec=spec,
            code_uri="training/nlp/bert/code.tar.gz",
            output_prefix="artifacts/oci-running",
        )
        provider._jobs["oci-running"] = state

        inst_response = MagicMock()
        inst_response.data.lifecycle_state = "RUNNING"
        inst_response.data.time_created = None
        provider._compute_client.get_instance.return_value = inst_response

        obj_response = MagicMock()
        obj_response.data.content.decode.return_value = json.dumps({"status": "running"})
        provider._object_storage_client.get_object.return_value = obj_response

        status = await provider._poll_provider("oci-running")
        assert status.status == JobStatus.RUNNING


class TestOCICollectArtifactsEmptyRelative:
    """Cover line 738: skipping artifact with empty relative path."""

    async def test_collect_artifacts_skips_prefix_only(self, oci_patches):
        from artenic_ai_platform_providers.oci import _OCIJobState

        provider = _make_provider()
        provider._object_storage_client = MagicMock()

        spec = TrainingSpec(service="nlp", model="bert", provider="oci")
        state = _OCIJobState(
            instance_id="ocid1.instance.oc1..skip",
            instance_name="artenic-oci-skip",
            created_at=time.time(),
            spec=spec,
            code_uri="training/nlp/bert/code.tar.gz",
            output_prefix="artifacts/oci-skip",
        )
        provider._jobs["oci-skip"] = state

        prefix = "artifacts/oci-skip/output/"

        # One object is the prefix itself (empty relative), one has actual content
        obj_prefix = MagicMock()
        obj_prefix.name = prefix
        obj_real = MagicMock()
        obj_real.name = f"{prefix}model.pt"

        list_response = MagicMock()
        list_response.data.objects = [obj_prefix, obj_real]
        provider._object_storage_client.list_objects.return_value = list_response

        get_response = MagicMock()
        get_response.data.raw.stream.return_value = [b"model-data"]
        provider._object_storage_client.get_object.return_value = get_response

        dummy_status = CloudJobStatus(
            provider_job_id="oci-skip",
            status=JobStatus.COMPLETED,
        )
        result = await provider._collect_artifacts("oci-skip", dummy_status)
        assert result is not None
        # Only 1 get_object call for model.pt, not the prefix-only key
        provider._object_storage_client.get_object.assert_called_once()


class TestOCICleanupNon404Raises:
    """Cover line 780: ServiceError with status != 404 should re-raise."""

    async def test_cleanup_non_404_raises(self, oci_patches):
        from artenic_ai_platform_providers.oci import _OCIJobState

        provider = _make_provider()
        provider._compute_client = MagicMock()

        spec = TrainingSpec(service="nlp", model="bert", provider="oci")
        state = _OCIJobState(
            instance_id="ocid1.instance.oc1..err",
            instance_name="artenic-oci-err",
            created_at=time.time(),
            spec=spec,
            code_uri="training/nlp/bert/code.tar.gz",
            output_prefix="artifacts/oci-err",
        )
        provider._jobs["oci-err"] = state

        se = oci_patches["ServiceError"]
        provider._compute_client.terminate_instance.side_effect = se(
            500, "InternalError", {}, "server error"
        )

        with pytest.raises(se):
            await provider._cleanup_compute("oci-err")


class TestOCICancelNon404Raises:
    """Cover line 811: cancel ServiceError with status != 404 should re-raise."""

    async def test_cancel_non_404_raises(self, oci_patches):
        from artenic_ai_platform_providers.oci import _OCIJobState

        provider = _make_provider()
        provider._compute_client = MagicMock()

        spec = TrainingSpec(service="nlp", model="bert", provider="oci")
        state = _OCIJobState(
            instance_id="ocid1.instance.oc1..cancel-err",
            instance_name="artenic-oci-cancel-err",
            created_at=time.time(),
            spec=spec,
            code_uri="training/nlp/bert/code.tar.gz",
            output_prefix="artifacts/oci-cancel-err",
        )
        provider._jobs["oci-cancel-err"] = state

        se = oci_patches["ServiceError"]
        provider._compute_client.terminate_instance.side_effect = se(
            403, "Forbidden", {}, "not authorized"
        )

        with pytest.raises(se):
            await provider._cancel_provider_job("oci-cancel-err")


class TestOCIGetFirstADNoADs:
    """Cover line 854: no availability domains found."""

    async def test_get_first_ad_no_ads(self, oci_patches):
        provider = _make_provider(availability_domain=None)
        await provider._connect()

        oci_sdk = oci_patches["oci"]
        ad_response = MagicMock()
        ad_response.data = []  # No ADs
        identity_client = MagicMock()
        identity_client.list_availability_domains.return_value = ad_response
        oci_sdk.identity.IdentityClient.return_value = identity_client

        with pytest.raises(RuntimeError, match="No availability domains"):
            await provider._get_first_ad()


class TestOCIReadStatusFileEdgeCases:
    """Cover lines 879-881: ServiceError non-404 re-raise, json decode error."""

    async def test_read_status_non_404_raises(self, oci_patches):
        provider = _make_provider()
        provider._object_storage_client = MagicMock()

        se = oci_patches["ServiceError"]
        provider._object_storage_client.get_object.side_effect = se(
            500, "InternalError", {}, "server error"
        )

        with pytest.raises(se):
            await provider._read_status_file("job-err")

    async def test_read_status_json_decode_error(self, oci_patches):
        provider = _make_provider()
        provider._object_storage_client = MagicMock()

        obj_response = MagicMock()
        obj_response.data.content.decode.return_value = "not valid json{{"
        provider._object_storage_client.get_object.return_value = obj_response

        result = await provider._read_status_file("job-badjson")
        assert result is None

    async def test_read_status_attribute_error(self, oci_patches):
        provider = _make_provider()
        provider._object_storage_client = MagicMock()

        obj_response = MagicMock()
        obj_response.data.content = None  # Will cause AttributeError on .decode()
        provider._object_storage_client.get_object.return_value = obj_response

        result = await provider._read_status_file("job-attr-err")
        assert result is None


class TestOCIShapeToDictMissingAttrs:
    """Cover lines 906, 913, 920, 926: _shape_to_dict with missing attributes."""

    def test_shape_to_dict_no_ocpus(self, oci_patches):
        from artenic_ai_platform_providers.oci import _shape_to_dict

        shape = MagicMock(spec=[])  # no attributes at all
        shape.shape = "VM.Test"
        # Remove specific attrs by using spec=[]
        result = _shape_to_dict(shape)
        assert result["ocpus"] == 0

    def test_shape_to_dict_no_memory(self, oci_patches):
        from artenic_ai_platform_providers.oci import _shape_to_dict

        shape = MagicMock(spec=[])
        shape.shape = "VM.Test"
        result = _shape_to_dict(shape)
        assert result["memory_in_gbs"] == 0.0

    def test_shape_to_dict_no_gpus(self, oci_patches):
        from artenic_ai_platform_providers.oci import _shape_to_dict

        shape = MagicMock(spec=[])
        shape.shape = "VM.Test"
        result = _shape_to_dict(shape)
        assert result["gpus"] == 0

    def test_shape_to_dict_no_gpu_description(self, oci_patches):
        from artenic_ai_platform_providers.oci import _shape_to_dict

        shape = MagicMock(spec=[])
        shape.shape = "VM.Test"
        result = _shape_to_dict(shape)
        assert result["gpu_description"] == ""

    def test_shape_to_dict_all_present(self, oci_patches):
        from artenic_ai_platform_providers.oci import _shape_to_dict

        shape = MagicMock()
        shape.shape = "BM.GPU4.8"
        shape.ocpus = 64.0
        shape.memory_in_gbs = 2048.0
        shape.gpus = 8
        shape.gpu_description = "A100"

        result = _shape_to_dict(shape)
        assert result["shape"] == "BM.GPU4.8"
        assert result["ocpus"] == 64
        assert result["memory_in_gbs"] == 2048.0
        assert result["gpus"] == 8
        assert result["gpu_description"] == "A100"
