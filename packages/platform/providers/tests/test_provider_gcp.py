"""Tests for artenic_ai_platform_providers.gcp — GCPProvider."""

from __future__ import annotations

import pathlib
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from artenic_ai_platform_providers.base import (
    CloudJobStatus,
    JobStatus,
    TrainingSpec,
)

# ---------------------------------------------------------------------------
# Module-level patches — the GCP SDK is not installed so we need to mock
# the module-level flag variables AND the SDK references.
# ---------------------------------------------------------------------------

_MODULE = "artenic_ai_platform_providers.gcp"


def _to_thread_patch():
    """Create a fresh asyncio.to_thread patch for each fixture call."""
    return patch(
        "asyncio.to_thread",
        new=AsyncMock(
            side_effect=lambda fn, *a, **kw: fn(*a, **kw),
        ),
    )


def _make_provider(**overrides):
    """Create a GCPProvider with mocked SDK availability."""
    from artenic_ai_platform_providers.gcp import GCPProvider

    defaults = {
        "project_id": "test-project",
        "credentials_path": None,
        "region": "europe-west1",
        "zone": "europe-west1-b",
        "bucket_name": "test-bucket",
    }
    defaults.update(overrides)
    return GCPProvider(**defaults)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def mock_compute_v1():
    """Build a comprehensive MagicMock for google.cloud.compute_v1."""
    m = MagicMock()
    # Operation status enum
    m.Operation.Status.DONE = "DONE"
    return m


@pytest.fixture
def mock_gcs_storage():
    return MagicMock()


@pytest.fixture
def mock_sa():
    return MagicMock()


@pytest.fixture
def gcp_patches(mock_compute_v1, mock_gcs_storage, mock_sa):
    """Apply all module-level patches for GCP SDK."""
    patches = [
        patch(f"{_MODULE}._HAS_COMPUTE", True),
        patch(f"{_MODULE}._HAS_STORAGE", True),
        patch(f"{_MODULE}._HAS_AUTH", True),
        patch(f"{_MODULE}.compute_v1", mock_compute_v1),
        patch(f"{_MODULE}.gcs_storage", mock_gcs_storage),
        patch(f"{_MODULE}._sa", mock_sa),
        _to_thread_patch(),
    ]
    for p in patches:
        p.start()
    yield {
        "compute_v1": mock_compute_v1,
        "gcs_storage": mock_gcs_storage,
        "sa": mock_sa,
    }
    for p in patches:
        p.stop()


# ======================================================================
# Tests
# ======================================================================


class TestGCPInit:
    def test_init_defaults(self, gcp_patches):
        provider = _make_provider()
        assert provider.provider_name == "gcp"
        assert provider._connected is False
        assert provider._project_id == "test-project"
        assert provider._bucket_name == "test-bucket"

    def test_init_auto_bucket_name(self, gcp_patches):
        provider = _make_provider(bucket_name=None)
        assert provider._bucket_name == "artenic-training-test-project"


class TestGCPProviderName:
    def test_provider_name(self, gcp_patches):
        provider = _make_provider()
        assert provider.provider_name == "gcp"


class TestGCPConnect:
    async def test_connect_with_default_creds(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        cv1 = gcp_patches["compute_v1"]
        cv1.InstancesClient.assert_called_once()
        cv1.MachineTypesClient.assert_called_once()
        cv1.AcceleratorTypesClient.assert_called_once()
        cv1.ZoneOperationsClient.assert_called_once()

        gcs = gcp_patches["gcs_storage"]
        gcs.Client.assert_called_once_with(
            project="test-project",
            credentials=None,
        )

    async def test_connect_with_service_account(self, gcp_patches):
        provider = _make_provider(credentials_path="/key.json")
        await provider._connect()

        sa = gcp_patches["sa"]
        sa.Credentials.from_service_account_file.assert_called_once()

    async def test_connect_raises_without_compute(self):
        with patch(f"{_MODULE}._HAS_COMPUTE", False), patch(f"{_MODULE}._HAS_STORAGE", True):
            provider = _make_provider()
            with pytest.raises(ImportError, match="google-cloud-compute"):
                await provider._connect()

    async def test_connect_raises_without_storage(self):
        with patch(f"{_MODULE}._HAS_COMPUTE", True), patch(f"{_MODULE}._HAS_STORAGE", False):
            provider = _make_provider()
            with pytest.raises(ImportError, match="google-cloud-storage"):
                await provider._connect()


class TestGCPDisconnect:
    async def test_disconnect_clears_clients(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        assert provider._instances_client is not None
        await provider._disconnect()

        assert provider._instances_client is None
        assert provider._storage_client is None

    async def test_disconnect_handles_error(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        # Simulate error during close
        transport = MagicMock()
        transport.close.side_effect = RuntimeError("close failed")
        provider._instances_client._transport = transport

        # Should not raise
        await provider._disconnect()
        assert provider._instances_client is None


class TestGCPListInstances:
    async def test_list_instances_basic(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        # Mock machine type
        mt = MagicMock()
        mt.name = "n1-standard-4"
        mt.guest_cpus = 4
        mt.memory_mb = 15360
        mt.deprecated = False

        provider._machine_types_client.list.return_value = [mt]
        provider._accelerator_types_client.list.return_value = []

        instances = await provider._list_instances()

        assert len(instances) >= 1
        assert instances[0].name == "n1-standard-4"
        assert instances[0].vcpus == 4

    async def test_list_instances_gpu_only(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        mt = MagicMock()
        mt.name = "n1-standard-4"
        mt.guest_cpus = 4
        mt.memory_mb = 15360
        mt.deprecated = False

        provider._machine_types_client.list.return_value = [mt]
        provider._accelerator_types_client.list.return_value = []

        instances = await provider._list_instances(gpu_only=True)
        # n1-standard-4 has no GPU, should be empty
        assert len(instances) == 0

    async def test_list_instances_with_a2_family(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        mt = MagicMock()
        mt.name = "a2-highgpu-1g"
        mt.guest_cpus = 12
        mt.memory_mb = 87040
        mt.deprecated = False

        provider._machine_types_client.list.return_value = [mt]
        provider._accelerator_types_client.list.return_value = []

        instances = await provider._list_instances(gpu_only=True)
        gpu_names = [i.name for i in instances]
        assert "a2-highgpu-1g" in gpu_names

    async def test_list_instances_with_accelerators(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        provider._machine_types_client.list.return_value = []

        at = MagicMock()
        at.name = "nvidia-tesla-a100"
        at.maximum_cards_per_instance = 4

        provider._accelerator_types_client.list.return_value = [at]

        instances = await provider._list_instances()
        gpu_entries = [i for i in instances if i.gpu_count > 0]
        assert len(gpu_entries) >= 1
        assert gpu_entries[0].gpu_type == "A100"


class TestGCPUploadCode:
    async def test_upload_code(self, gcp_patches, tmp_path):
        provider = _make_provider()
        await provider._connect()

        # Create a temp file to upload
        code_file = tmp_path / "train.py"
        code_file.write_text("print('hello')")

        mock_bucket = MagicMock()
        mock_bucket.exists.return_value = True
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        provider._storage_client.bucket.return_value = mock_bucket

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="gcp",
            config={
                "job_id": "test-job",
                "code_path": str(code_file),
            },
        )
        uri = await provider._upload_code(spec)

        assert uri.startswith("gs://test-bucket/training/")
        assert "test-job" in uri
        mock_blob.upload_from_filename.assert_called_once()

    async def test_upload_code_creates_bucket(self, gcp_patches, tmp_path):
        provider = _make_provider()
        await provider._connect()

        code_file = tmp_path / "train.py"
        code_file.write_text("print('hello')")

        mock_bucket = MagicMock()
        mock_bucket.exists.return_value = False
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        provider._storage_client.bucket.return_value = mock_bucket
        provider._storage_client.create_bucket.return_value = mock_bucket

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="gcp",
            config={
                "job_id": "test-job",
                "code_path": str(code_file),
            },
        )
        uri = await provider._upload_code(spec)
        assert uri.startswith("gs://")
        provider._storage_client.create_bucket.assert_called_once()


class TestGCPProvisionAndStart:
    async def test_provision_and_start(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        # Mock the insert operation
        mock_op = MagicMock()
        mock_op.name = "op-123"
        provider._instances_client.insert.return_value = mock_op

        # Mock _wait_for_zone_operation — ops_client.get returns DONE
        done_result = MagicMock()
        done_result.status = "DONE"
        done_result.error = None
        provider._zone_operations_client.get.return_value = done_result

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="gcp",
            config={"job_id": "test123"},
        )
        job_id = await provider._provision_and_start(spec)

        assert job_id == "test123"
        provider._instances_client.insert.assert_called_once()

    async def test_provision_with_gpu(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        mock_op = MagicMock()
        mock_op.name = "op-456"
        provider._instances_client.insert.return_value = mock_op

        done_result = MagicMock()
        done_result.status = "DONE"
        done_result.error = None
        provider._zone_operations_client.get.return_value = done_result

        spec = TrainingSpec(
            service="cv",
            model="resnet",
            provider="gcp",
            config={
                "job_id": "gpu-job",
                "gpu_type": "nvidia-tesla-a100",
                "gpu_count": 2,
            },
            is_spot=True,
        )
        job_id = await provider._provision_and_start(spec)
        assert job_id == "gpu-job"


class TestGCPPollProvider:
    async def test_poll_running(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        provider._job_metadata["job1"] = {
            "instance_name": "artenic-train-job1",
            "zone": "europe-west1-b",
            "start_time": time.time() - 60,
            "spec": TrainingSpec(
                service="nlp",
                model="bert",
                provider="gcp",
            ),
        }

        mock_instance = MagicMock()
        mock_instance.status = "RUNNING"
        provider._instances_client.get.return_value = mock_instance

        serial = MagicMock()
        serial.contents = "some output"
        provider._instances_client.get_serial_port_output.return_value = serial

        status = await provider._poll_provider("job1")
        assert status.status == JobStatus.RUNNING

    async def test_poll_completed(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        provider._job_metadata["job2"] = {
            "instance_name": "artenic-train-job2",
            "zone": "europe-west1-b",
            "start_time": time.time() - 120,
            "spec": TrainingSpec(
                service="nlp",
                model="bert",
                provider="gcp",
            ),
        }

        mock_instance = MagicMock()
        mock_instance.status = "RUNNING"
        provider._instances_client.get.return_value = mock_instance

        serial = MagicMock()
        serial.contents = "===== TRAINING COMPLETED =====\nMETRIC: loss=0.01\n"
        provider._instances_client.get_serial_port_output.return_value = serial

        status = await provider._poll_provider("job2")
        assert status.status == JobStatus.COMPLETED
        assert status.metrics is not None
        assert status.metrics.get("loss") == 0.01

    async def test_poll_no_metadata(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        status = await provider._poll_provider("missing-job")
        assert status.status == JobStatus.FAILED
        assert "No metadata" in (status.error or "")

    async def test_poll_terminated_vm_spot(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        provider._job_metadata["spot-job"] = {
            "instance_name": "artenic-train-spot",
            "zone": "europe-west1-b",
            "start_time": time.time() - 60,
            "spec": TrainingSpec(
                service="nlp",
                model="bert",
                provider="gcp",
                is_spot=True,
            ),
        }

        mock_instance = MagicMock()
        mock_instance.status = "TERMINATED"
        provider._instances_client.get.return_value = mock_instance

        serial = MagicMock()
        serial.contents = ""
        provider._instances_client.get_serial_port_output.return_value = serial

        status = await provider._poll_provider("spot-job")
        assert status.status == JobStatus.PREEMPTED

    async def test_poll_instance_not_found(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        provider._job_metadata["gone-job"] = {
            "instance_name": "artenic-train-gone",
            "zone": "europe-west1-b",
            "start_time": time.time() - 60,
            "spec": TrainingSpec(
                service="nlp",
                model="bert",
                provider="gcp",
            ),
        }

        provider._instances_client.get.side_effect = Exception("404 not found")

        status = await provider._poll_provider("gone-job")
        # TERMINATED path, non-spot -> FAILED
        assert status.status == JobStatus.FAILED


class TestGCPCollectArtifacts:
    async def test_collect_artifacts(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        provider._job_metadata["art-job"] = {
            "output_uri": "gs://test-bucket/training/nlp/bert/art-job/output",
        }

        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.name = "training/nlp/bert/art-job/output/model.pt"
        mock_bucket.list_blobs.return_value = [mock_blob]
        provider._storage_client.bucket.return_value = mock_bucket

        dummy_status = CloudJobStatus(
            provider_job_id="art-job",
            status=JobStatus.COMPLETED,
        )
        result = await provider._collect_artifacts("art-job", dummy_status)
        assert result is not None
        mock_blob.download_to_filename.assert_called_once()

    async def test_collect_artifacts_no_metadata(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        dummy_status = CloudJobStatus(
            provider_job_id="no-meta",
            status=JobStatus.COMPLETED,
        )
        result = await provider._collect_artifacts("no-meta", dummy_status)
        assert result is None

    async def test_collect_artifacts_no_blobs(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        provider._job_metadata["empty-job"] = {
            "output_uri": "gs://test-bucket/output/empty-job",
        }

        mock_bucket = MagicMock()
        mock_bucket.list_blobs.return_value = []
        provider._storage_client.bucket.return_value = mock_bucket

        dummy_status = CloudJobStatus(
            provider_job_id="empty-job",
            status=JobStatus.COMPLETED,
        )
        result = await provider._collect_artifacts(
            "empty-job",
            dummy_status,
        )
        # Should return output_uri when no blobs found
        assert result is not None


class TestGCPCleanupCompute:
    async def test_cleanup_compute(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        provider._job_metadata["clean-job"] = {
            "instance_name": "artenic-train-clean",
            "zone": "europe-west1-b",
        }

        mock_op = MagicMock()
        mock_op.name = "del-op"
        provider._instances_client.delete.return_value = mock_op

        done_result = MagicMock()
        done_result.status = "DONE"
        done_result.error = None
        provider._zone_operations_client.get.return_value = done_result

        await provider._cleanup_compute("clean-job")

        provider._instances_client.delete.assert_called_once()
        assert "clean-job" not in provider._job_metadata

    async def test_cleanup_compute_no_metadata(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        # Should not raise
        await provider._cleanup_compute("nonexistent")

    async def test_cleanup_already_deleted(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        provider._job_metadata["del-job"] = {
            "instance_name": "artenic-train-del",
            "zone": "europe-west1-b",
        }

        provider._instances_client.delete.side_effect = Exception("404 not found")

        # Should not raise — 404 is swallowed
        await provider._cleanup_compute("del-job")


class TestGCPCancelProviderJob:
    async def test_cancel_provider_job(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        provider._job_metadata["cancel-job"] = {
            "instance_name": "artenic-train-cancel",
            "zone": "europe-west1-b",
        }

        mock_op = MagicMock()
        mock_op.name = "stop-op"
        provider._instances_client.stop.return_value = mock_op
        provider._instances_client.delete.return_value = mock_op

        done_result = MagicMock()
        done_result.status = "DONE"
        done_result.error = None
        provider._zone_operations_client.get.return_value = done_result

        await provider._cancel_provider_job("cancel-job")

        provider._instances_client.stop.assert_called_once()
        provider._instances_client.delete.assert_called_once()

    async def test_cancel_no_metadata(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        # Should not raise
        await provider._cancel_provider_job("unknown")

    async def test_cancel_stop_fails(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        provider._job_metadata["fail-stop"] = {
            "instance_name": "artenic-train-fail",
            "zone": "europe-west1-b",
        }

        provider._instances_client.stop.side_effect = Exception("stop failed")
        mock_op = MagicMock()
        mock_op.name = "del-op"
        provider._instances_client.delete.return_value = mock_op

        done_result = MagicMock()
        done_result.status = "DONE"
        done_result.error = None
        provider._zone_operations_client.get.return_value = done_result

        # Should not raise; stop failure is caught, delete proceeds
        await provider._cancel_provider_job("fail-stop")
        provider._instances_client.delete.assert_called_once()


# ===================================================================
# Coverage: zone derivation from region (line 258)
# ===================================================================


class TestGCPListInstancesZoneDerivation:
    async def test_list_instances_with_different_region(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        mt = MagicMock()
        mt.name = "n1-standard-4"
        mt.guest_cpus = 4
        mt.memory_mb = 15360
        mt.deprecated = False

        provider._machine_types_client.list.return_value = [mt]
        provider._accelerator_types_client.list.return_value = []

        instances = await provider._list_instances(region="us-central1")
        assert len(instances) >= 1
        assert instances[0].region == "us-central1"


# ===================================================================
# Coverage: accelerator partial match (lines 298-301)
# ===================================================================


class TestGCPAcceleratorPartialMatch:
    async def test_accelerator_partial_match(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        provider._machine_types_client.list.return_value = []

        at = MagicMock()
        at.name = "nvidia-tesla-a100-special"
        at.maximum_cards_per_instance = 8

        provider._accelerator_types_client.list.return_value = [at]

        instances = await provider._list_instances()
        gpu_entries = [i for i in instances if i.gpu_count > 0]
        assert len(gpu_entries) >= 1
        assert gpu_entries[0].gpu_type == "A100"

    async def test_accelerator_unknown_type(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        provider._machine_types_client.list.return_value = []

        at = MagicMock()
        at.name = "unknown-accelerator"
        at.maximum_cards_per_instance = 1

        provider._accelerator_types_client.list.return_value = [at]

        instances = await provider._list_instances()
        # Unknown accelerator should not produce gpu entries
        gpu_entries = [i for i in instances if i.gpu_count > 0]
        assert len(gpu_entries) == 0


# ===================================================================
# Coverage: A2 family variants (lines 340-351)
# ===================================================================


class TestGCPA2FamilyVariants:
    async def test_a2_highgpu_2g(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        mt = MagicMock()
        mt.name = "a2-highgpu-2g"
        mt.guest_cpus = 24
        mt.memory_mb = 174080
        mt.deprecated = False

        provider._machine_types_client.list.return_value = [mt]
        provider._accelerator_types_client.list.return_value = []

        instances = await provider._list_instances(gpu_only=True)
        gpu_names = [i.name for i in instances]
        assert "a2-highgpu-2g" in gpu_names
        match = next(i for i in instances if i.name == "a2-highgpu-2g")
        assert match.gpu_count == 2
        assert match.gpu_type == "A100"

    async def test_a2_highgpu_4g(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        mt = MagicMock()
        mt.name = "a2-highgpu-4g"
        mt.guest_cpus = 48
        mt.memory_mb = 348160
        mt.deprecated = False

        provider._machine_types_client.list.return_value = [mt]
        provider._accelerator_types_client.list.return_value = []

        instances = await provider._list_instances(gpu_only=True)
        match = next(i for i in instances if i.name == "a2-highgpu-4g")
        assert match.gpu_count == 4

    async def test_a2_highgpu_8g(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        mt = MagicMock()
        mt.name = "a2-highgpu-8g"
        mt.guest_cpus = 96
        mt.memory_mb = 696320
        mt.deprecated = False

        provider._machine_types_client.list.return_value = [mt]
        provider._accelerator_types_client.list.return_value = []

        instances = await provider._list_instances(gpu_only=True)
        match = next(i for i in instances if i.name == "a2-highgpu-8g")
        assert match.gpu_count == 8

    async def test_a2_megagpu_16g(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        mt = MagicMock()
        mt.name = "a2-megagpu-16g"
        mt.guest_cpus = 96
        mt.memory_mb = 1392640
        mt.deprecated = False

        provider._machine_types_client.list.return_value = [mt]
        provider._accelerator_types_client.list.return_value = []

        instances = await provider._list_instances(gpu_only=True)
        match = next(i for i in instances if i.name == "a2-megagpu-16g")
        assert match.gpu_count == 16
        assert match.gpu_type == "A100"

    async def test_a2_ultragpu(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        mt = MagicMock()
        mt.name = "a2-ultragpu-8g"
        mt.guest_cpus = 96
        mt.memory_mb = 1392640
        mt.deprecated = False

        provider._machine_types_client.list.return_value = [mt]
        provider._accelerator_types_client.list.return_value = []

        instances = await provider._list_instances(gpu_only=True)
        match = next(i for i in instances if i.name == "a2-ultragpu-8g")
        assert match.gpu_count == 8
        assert match.gpu_type == "A100-80GB"

    async def test_a2_no_gpu_count_match(self, gcp_patches):
        """A2 server that doesn't match any known GPU pattern -> gpu_count=0."""
        provider = _make_provider()
        await provider._connect()

        mt = MagicMock()
        mt.name = "a2-standard-1"
        mt.guest_cpus = 4
        mt.memory_mb = 8192
        mt.deprecated = False

        provider._machine_types_client.list.return_value = [mt]
        provider._accelerator_types_client.list.return_value = []

        instances = await provider._list_instances(gpu_only=True)
        # gpu_count is 0 so it should be filtered out
        assert len(instances) == 0


# ===================================================================
# Coverage: G2 family GPU parsing (lines 368-387)
# ===================================================================


class TestGCPG2Family:
    async def test_g2_standard_4(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        mt = MagicMock()
        mt.name = "g2-standard-4"
        mt.guest_cpus = 4
        mt.memory_mb = 16384
        mt.deprecated = False

        provider._machine_types_client.list.return_value = [mt]
        provider._accelerator_types_client.list.return_value = []

        instances = await provider._list_instances(gpu_only=True)
        match = next(i for i in instances if i.name == "g2-standard-4")
        assert match.gpu_count == 1
        assert match.gpu_type == "L4"

    async def test_g2_standard_24(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        mt = MagicMock()
        mt.name = "g2-standard-24"
        mt.guest_cpus = 24
        mt.memory_mb = 98304
        mt.deprecated = False

        provider._machine_types_client.list.return_value = [mt]
        provider._accelerator_types_client.list.return_value = []

        instances = await provider._list_instances(gpu_only=True)
        match = next(i for i in instances if i.name == "g2-standard-24")
        assert match.gpu_count == 2
        assert match.gpu_type == "L4"

    async def test_g2_standard_32(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        mt = MagicMock()
        mt.name = "g2-standard-32"
        mt.guest_cpus = 32
        mt.memory_mb = 131072
        mt.deprecated = False

        provider._machine_types_client.list.return_value = [mt]
        provider._accelerator_types_client.list.return_value = []

        instances = await provider._list_instances(gpu_only=True)
        match = next(i for i in instances if i.name == "g2-standard-32")
        assert match.gpu_count == 1

    async def test_g2_standard_48(self, gcp_patches):
        """g2-standard-48 has 4 L4 GPUs."""
        provider = _make_provider()
        await provider._connect()

        mt = MagicMock()
        mt.name = "g2-standard-48"
        mt.guest_cpus = 48
        mt.memory_mb = 196608
        mt.deprecated = False

        provider._machine_types_client.list.return_value = [mt]
        provider._accelerator_types_client.list.return_value = []

        instances = await provider._list_instances(gpu_only=True)
        match = next(i for i in instances if i.name == "g2-standard-48")
        assert match.gpu_count == 4

    async def test_g2_standard_96(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        mt = MagicMock()
        mt.name = "g2-standard-96"
        mt.guest_cpus = 96
        mt.memory_mb = 393216
        mt.deprecated = False

        provider._machine_types_client.list.return_value = [mt]
        provider._accelerator_types_client.list.return_value = []

        instances = await provider._list_instances(gpu_only=True)
        match = next(i for i in instances if i.name == "g2-standard-96")
        assert match.gpu_count == 8

    async def test_g2_no_gpu_match(self, gcp_patches):
        """G2 server that doesn't match known patterns -> gpu_count=0."""
        provider = _make_provider()
        await provider._connect()

        mt = MagicMock()
        mt.name = "g2-custom-1"
        mt.guest_cpus = 1
        mt.memory_mb = 4096
        mt.deprecated = False

        provider._machine_types_client.list.return_value = [mt]
        provider._accelerator_types_client.list.return_value = []

        instances = await provider._list_instances(gpu_only=True)
        assert len(instances) == 0


# ===================================================================
# Coverage: upload_code directory walk (lines 436-442) and
# file not found (line 447)
# ===================================================================


class TestGCPUploadCodeDirectory:
    async def test_upload_code_directory(self, gcp_patches, tmp_path):
        provider = _make_provider()
        await provider._connect()

        # Create a directory with files
        sub = tmp_path / "project"
        sub.mkdir()
        (sub / "train.py").write_text("print('train')")
        (sub / "utils.py").write_text("pass")

        mock_bucket = MagicMock()
        mock_bucket.exists.return_value = True
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        provider._storage_client.bucket.return_value = mock_bucket

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="gcp",
            config={
                "job_id": "dir-job",
                "code_path": str(sub),
            },
        )
        uri = await provider._upload_code(spec)
        assert uri.startswith("gs://")
        assert mock_blob.upload_from_filename.call_count == 2

    async def test_upload_code_file_not_found(self, gcp_patches, tmp_path):
        provider = _make_provider()
        await provider._connect()

        mock_bucket = MagicMock()
        mock_bucket.exists.return_value = True
        provider._storage_client.bucket.return_value = mock_bucket

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="gcp",
            config={
                "job_id": "missing-job",
                "code_path": str(tmp_path / "nonexistent"),
            },
        )
        with pytest.raises(FileNotFoundError, match="Code path not found"):
            await provider._upload_code(spec)


# ===================================================================
# Coverage: poll PROVISIONING/STAGING (line 678), training_failed
# in RUNNING (line 684), TERMINATED with training_failed (line 689),
# TERMINATED non-spot (line 691-695), SUSPENDED (line 696-697),
# unknown VM status (line 698-699)
# ===================================================================


class TestGCPPollStatusBranches:
    def _setup_job_meta(self, provider, job_id, is_spot=False):
        provider._job_metadata[job_id] = {
            "instance_name": f"artenic-train-{job_id}",
            "zone": "europe-west1-b",
            "start_time": time.time() - 60,
            "spec": TrainingSpec(
                service="nlp",
                model="bert",
                provider="gcp",
                is_spot=is_spot,
            ),
        }

    async def test_poll_provisioning(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()
        self._setup_job_meta(provider, "prov-job")

        mock_instance = MagicMock()
        mock_instance.status = "PROVISIONING"
        provider._instances_client.get.return_value = mock_instance
        serial = MagicMock()
        serial.contents = ""
        provider._instances_client.get_serial_port_output.return_value = serial

        status = await provider._poll_provider("prov-job")
        assert status.status == JobStatus.PENDING

    async def test_poll_staging(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()
        self._setup_job_meta(provider, "stag-job")

        mock_instance = MagicMock()
        mock_instance.status = "STAGING"
        provider._instances_client.get.return_value = mock_instance
        serial = MagicMock()
        serial.contents = ""
        provider._instances_client.get_serial_port_output.return_value = serial

        status = await provider._poll_provider("stag-job")
        assert status.status == JobStatus.PENDING

    async def test_poll_running_training_failed(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()
        self._setup_job_meta(provider, "fail-run-job")

        mock_instance = MagicMock()
        mock_instance.status = "RUNNING"
        provider._instances_client.get.return_value = mock_instance
        serial = MagicMock()
        serial.contents = "===== TRAINING FAILED (exit 1) ====="
        provider._instances_client.get_serial_port_output.return_value = serial

        status = await provider._poll_provider("fail-run-job")
        assert status.status == JobStatus.FAILED

    async def test_poll_terminated_training_completed(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()
        self._setup_job_meta(provider, "term-done-job")

        mock_instance = MagicMock()
        mock_instance.status = "TERMINATED"
        provider._instances_client.get.return_value = mock_instance
        serial = MagicMock()
        serial.contents = "===== TRAINING COMPLETED ====="
        provider._instances_client.get_serial_port_output.return_value = serial

        status = await provider._poll_provider("term-done-job")
        assert status.status == JobStatus.COMPLETED

    async def test_poll_terminated_training_failed(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()
        self._setup_job_meta(provider, "term-fail-job")

        mock_instance = MagicMock()
        mock_instance.status = "TERMINATED"
        provider._instances_client.get.return_value = mock_instance
        serial = MagicMock()
        serial.contents = "===== TRAINING FAILED (exit 2) ====="
        provider._instances_client.get_serial_port_output.return_value = serial

        status = await provider._poll_provider("term-fail-job")
        assert status.status == JobStatus.FAILED

    async def test_poll_terminated_no_status_non_spot(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()
        self._setup_job_meta(provider, "term-nostat-job", is_spot=False)

        mock_instance = MagicMock()
        mock_instance.status = "TERMINATED"
        provider._instances_client.get.return_value = mock_instance
        serial = MagicMock()
        serial.contents = ""
        provider._instances_client.get_serial_port_output.return_value = serial

        status = await provider._poll_provider("term-nostat-job")
        assert status.status == JobStatus.FAILED

    async def test_poll_suspended(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()
        self._setup_job_meta(provider, "susp-job")

        mock_instance = MagicMock()
        mock_instance.status = "SUSPENDED"
        provider._instances_client.get.return_value = mock_instance
        serial = MagicMock()
        serial.contents = ""
        provider._instances_client.get_serial_port_output.return_value = serial

        status = await provider._poll_provider("susp-job")
        assert status.status == JobStatus.PREEMPTED

    async def test_poll_unknown_vm_status(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()
        self._setup_job_meta(provider, "unk-job")

        mock_instance = MagicMock()
        mock_instance.status = "REPAIRING"
        provider._instances_client.get.return_value = mock_instance
        serial = MagicMock()
        serial.contents = ""
        provider._instances_client.get_serial_port_output.return_value = serial

        status = await provider._poll_provider("unk-job")
        assert status.status == JobStatus.RUNNING

    async def test_poll_serial_console_error(self, gcp_patches):
        """Serial console read fails -- should still return status."""
        provider = _make_provider()
        await provider._connect()
        self._setup_job_meta(provider, "serial-err-job")

        mock_instance = MagicMock()
        mock_instance.status = "RUNNING"
        provider._instances_client.get.return_value = mock_instance
        provider._instances_client.get_serial_port_output.side_effect = Exception(
            "serial port error"
        )

        status = await provider._poll_provider("serial-err-job")
        assert status.status == JobStatus.RUNNING

    async def test_poll_instance_not_found_returns_terminated(self, gcp_patches):
        """404 when fetching instance -> TERMINATED -> FAILED (non-spot)."""
        provider = _make_provider()
        await provider._connect()
        self._setup_job_meta(provider, "gone404-job", is_spot=False)

        provider._instances_client.get.side_effect = Exception("404 not found")

        status = await provider._poll_provider("gone404-job")
        assert status.status == JobStatus.FAILED

    async def test_poll_instance_get_non_404_error_reraises(self, gcp_patches):
        """Non-404 error when fetching instance should re-raise."""
        provider = _make_provider()
        await provider._connect()
        self._setup_job_meta(provider, "err-reraise-job")

        provider._instances_client.get.side_effect = Exception("permission denied")

        with pytest.raises(Exception, match="permission denied"):
            await provider._poll_provider("err-reraise-job")


# ===================================================================
# Coverage: collect_artifacts no output_uri (line 729), blob with
# empty relative_path (line 752)
# ===================================================================


class TestGCPCollectArtifactsEdgeCases:
    async def test_collect_artifacts_no_output_uri(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        provider._job_metadata["no-uri-job"] = {
            "output_uri": "",
        }

        dummy_status = CloudJobStatus(
            provider_job_id="no-uri-job",
            status=JobStatus.COMPLETED,
        )
        result = await provider._collect_artifacts("no-uri-job", dummy_status)
        assert result is None

    async def test_collect_artifacts_non_gs_uri(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        provider._job_metadata["http-job"] = {
            "output_uri": "http://example.com/output",
        }

        dummy_status = CloudJobStatus(
            provider_job_id="http-job",
            status=JobStatus.COMPLETED,
        )
        result = await provider._collect_artifacts("http-job", dummy_status)
        assert result is None

    async def test_collect_artifacts_blob_with_empty_relative(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        provider._job_metadata["empty-rel-job"] = {
            "output_uri": "gs://test-bucket/training/output/",
        }

        mock_bucket = MagicMock()
        # Blob whose name equals the prefix -> empty relative_path -> skip
        mock_blob_empty = MagicMock()
        mock_blob_empty.name = "training/output/"
        # Blob with actual content
        mock_blob_real = MagicMock()
        mock_blob_real.name = "training/output/model.pt"
        mock_bucket.list_blobs.return_value = [mock_blob_empty, mock_blob_real]
        provider._storage_client.bucket.return_value = mock_bucket

        dummy_status = CloudJobStatus(
            provider_job_id="empty-rel-job",
            status=JobStatus.COMPLETED,
        )
        result = await provider._collect_artifacts("empty-rel-job", dummy_status)
        assert result is not None
        # Only the real blob should have been downloaded
        mock_blob_real.download_to_filename.assert_called_once()
        mock_blob_empty.download_to_filename.assert_not_called()


# ===================================================================
# Coverage: cleanup re-raise for non-404 errors (line 802)
# ===================================================================


class TestGCPCleanupRaises:
    async def test_cleanup_raises_non_404(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        provider._job_metadata["err-job"] = {
            "instance_name": "artenic-train-err",
            "zone": "europe-west1-b",
        }

        provider._instances_client.delete.side_effect = Exception("permission denied")

        with pytest.raises(Exception, match="permission denied"):
            await provider._cleanup_compute("err-job")


# ===================================================================
# Coverage: cancel delete with 404 (lines 862-866)
# ===================================================================


class TestGCPCancelDeleteEdgeCases:
    async def test_cancel_delete_404_swallowed(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        provider._job_metadata["del404-job"] = {
            "instance_name": "artenic-train-del404",
            "zone": "europe-west1-b",
        }

        mock_op = MagicMock()
        mock_op.name = "stop-op"
        provider._instances_client.stop.return_value = mock_op

        done_result = MagicMock()
        done_result.status = "DONE"
        done_result.error = None
        provider._zone_operations_client.get.return_value = done_result

        # Delete raises a 404
        provider._instances_client.delete.side_effect = Exception("404 not found")

        # Should not raise -- 404 is swallowed
        await provider._cancel_provider_job("del404-job")

    async def test_cancel_delete_non_404_raises(self, gcp_patches):
        provider = _make_provider()
        await provider._connect()

        provider._job_metadata["del-err-job"] = {
            "instance_name": "artenic-train-del-err",
            "zone": "europe-west1-b",
        }

        mock_op = MagicMock()
        mock_op.name = "stop-op"
        provider._instances_client.stop.return_value = mock_op

        done_result = MagicMock()
        done_result.status = "DONE"
        done_result.error = None
        provider._zone_operations_client.get.return_value = done_result

        # Delete raises a non-404 error
        provider._instances_client.delete.side_effect = Exception("quota exceeded")

        with pytest.raises(Exception, match="quota exceeded"):
            await provider._cancel_provider_job("del-err-job")


# ===================================================================
# Coverage: _wait_for_zone_operation error + timeout (lines 893-897)
# ===================================================================


class TestWaitForZoneOperation:
    def test_operation_error(self, gcp_patches):
        from artenic_ai_platform_providers.gcp import _wait_for_zone_operation

        mock_ops_client = MagicMock()
        cv1 = gcp_patches["compute_v1"]

        error_obj = MagicMock()
        err_entry = MagicMock()
        err_entry.message = "Quota exceeded"
        error_obj.errors = [err_entry]

        result = MagicMock()
        result.status = cv1.Operation.Status.DONE
        result.error = error_obj
        mock_ops_client.get.return_value = result

        with pytest.raises(RuntimeError, match="Quota exceeded"):
            _wait_for_zone_operation(mock_ops_client, "proj", "zone", "op-1")

    def test_operation_timeout(self, gcp_patches):
        from artenic_ai_platform_providers.gcp import _wait_for_zone_operation

        mock_ops_client = MagicMock()

        result = MagicMock()
        result.status = "PENDING"  # Never becomes DONE
        mock_ops_client.get.return_value = result

        with pytest.raises(TimeoutError, match="timed out"):
            _wait_for_zone_operation(
                mock_ops_client,
                "proj",
                "zone",
                "op-2",
                timeout_seconds=1,
            )


# ===================================================================
# Coverage: _parse_serial_metrics (lines 911, 925-928, 932-937)
# ===================================================================


class TestParseSerialMetrics:
    def test_training_failed_marker(self):
        from artenic_ai_platform_providers.gcp import _parse_serial_metrics

        metrics = _parse_serial_metrics("===== TRAINING FAILED (exit 1) =====")
        assert metrics.get("training_failed") is True

    def test_metric_kv_parsing(self):
        from artenic_ai_platform_providers.gcp import _parse_serial_metrics

        output = "METRIC: loss=0.05\nMETRIC: acc=0.95\n"
        metrics = _parse_serial_metrics(output)
        assert metrics["loss"] == 0.05
        assert metrics["acc"] == 0.95

    def test_metric_string_value(self):
        from artenic_ai_platform_providers.gcp import _parse_serial_metrics

        output = "METRIC: status=converged\n"
        metrics = _parse_serial_metrics(output)
        assert metrics["status"] == "converged"

    def test_metric_malformed_line(self):
        from artenic_ai_platform_providers.gcp import _parse_serial_metrics

        output = "METRIC: no-equals-sign\n"
        metrics = _parse_serial_metrics(output)
        # Should not crash, just skip
        assert "no-equals-sign" not in metrics

    def test_json_metrics_parsing(self):
        from artenic_ai_platform_providers.gcp import _parse_serial_metrics

        output = '{"metrics": {"epoch": 10, "loss": 0.01}}\n'
        metrics = _parse_serial_metrics(output)
        assert metrics["epoch"] == 10
        assert metrics["loss"] == 0.01

    def test_json_metrics_single_quotes(self):
        from artenic_ai_platform_providers.gcp import _parse_serial_metrics

        output = "{'metrics': {'epoch': 5}}\n"
        metrics = _parse_serial_metrics(output)
        assert metrics["epoch"] == 5

    def test_json_metrics_invalid(self):
        from artenic_ai_platform_providers.gcp import _parse_serial_metrics

        output = '{"metrics": invalid json}\n'
        metrics = _parse_serial_metrics(output)
        # Should not crash
        assert isinstance(metrics, dict)

    def test_json_metrics_non_dict(self):
        from artenic_ai_platform_providers.gcp import _parse_serial_metrics

        output = '{"metrics": "string_value"}\n'
        metrics = _parse_serial_metrics(output)
        # metrics is a string, not a dict, so update should be skipped
        assert "string_value" not in metrics.values()


# ===================================================================
# Coverage: _should_skip (lines 944-959)
# ===================================================================


class TestShouldSkip:
    def test_skip_pycache(self):
        from artenic_ai_platform_providers.gcp import _should_skip

        assert _should_skip(pathlib.Path("src/__pycache__/module.cpython-311.pyc")) is True

    def test_skip_git(self):
        from artenic_ai_platform_providers.gcp import _should_skip

        assert _should_skip(pathlib.Path(".git/objects/abc123")) is True

    def test_skip_venv(self):
        from artenic_ai_platform_providers.gcp import _should_skip

        assert _should_skip(pathlib.Path(".venv/lib/python3.11/site-packages/foo.py")) is True

    def test_skip_node_modules(self):
        from artenic_ai_platform_providers.gcp import _should_skip

        assert _should_skip(pathlib.Path("node_modules/pkg/index.js")) is True

    def test_skip_pyc_suffix(self):
        from artenic_ai_platform_providers.gcp import _should_skip

        assert _should_skip(pathlib.Path("module.pyc")) is True

    def test_skip_whl_suffix(self):
        from artenic_ai_platform_providers.gcp import _should_skip

        assert _should_skip(pathlib.Path("package.whl")) is True

    def test_skip_egg_suffix(self):
        from artenic_ai_platform_providers.gcp import _should_skip

        assert _should_skip(pathlib.Path("package.egg")) is True

    def test_skip_pyo_suffix(self):
        from artenic_ai_platform_providers.gcp import _should_skip

        assert _should_skip(pathlib.Path("module.pyo")) is True

    def test_skip_egg_info(self):
        from artenic_ai_platform_providers.gcp import _should_skip

        # The directory part must exactly match ".egg-info"
        assert _should_skip(pathlib.Path(".egg-info/PKG-INFO")) is True

    def test_skip_mypy_cache(self):
        from artenic_ai_platform_providers.gcp import _should_skip

        assert _should_skip(pathlib.Path(".mypy_cache/3.11/module.json")) is True

    def test_skip_pytest_cache(self):
        from artenic_ai_platform_providers.gcp import _should_skip

        assert _should_skip(pathlib.Path(".pytest_cache/v/cache/lastfailed")) is True

    def test_skip_tox(self):
        from artenic_ai_platform_providers.gcp import _should_skip

        assert _should_skip(pathlib.Path(".tox/py311/lib/foo.py")) is True

    def test_no_skip_normal(self):
        from artenic_ai_platform_providers.gcp import _should_skip

        assert _should_skip(pathlib.Path("src/train.py")) is False

    def test_no_skip_txt(self):
        from artenic_ai_platform_providers.gcp import _should_skip

        assert _should_skip(pathlib.Path("requirements.txt")) is False
