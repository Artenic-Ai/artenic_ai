"""Tests for artenic_ai_platform.providers.azure — AzureProvider."""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from artenic_ai_platform.providers.base import (
    CloudJobStatus,
    JobStatus,
    TrainingSpec,
)

# ---------------------------------------------------------------------------
# Module path and shared patches
# ---------------------------------------------------------------------------

_MODULE = "artenic_ai_platform.providers.azure"


def _to_thread_patch():
    """Create a fresh asyncio.to_thread patch for each fixture call."""
    return patch(
        "asyncio.to_thread",
        new=AsyncMock(
            side_effect=lambda fn, *a, **kw: fn(*a, **kw),
        ),
    )


def _make_provider(**overrides):
    """Create an AzureProvider with mocked SDK availability."""
    from artenic_ai_platform.providers.azure import AzureProvider

    defaults = {
        "subscription_id": "sub-test-123",
        "tenant_id": "tenant-test",
        "client_id": "client-test",
        "client_secret": "secret-test",
        "resource_group": "rg-artenic",
        "region": "westeurope",
        "storage_account": "artenicstore",
        "container_name": "artenic-training",
        "vm_size": "Standard_NC6s_v3",
    }
    defaults.update(overrides)
    return AzureProvider(**defaults)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def mock_identity():
    m = MagicMock()
    m.ClientSecretCredential = MagicMock(
        return_value=MagicMock(),
    )
    m.DefaultAzureCredential = MagicMock(
        return_value=MagicMock(),
    )
    return m


@pytest.fixture
def mock_compute_mgmt():
    m = MagicMock()
    m.ComputeManagementClient = MagicMock(return_value=MagicMock())
    return m


@pytest.fixture
def mock_storage():
    m = MagicMock()
    m.BlobServiceClient = MagicMock(return_value=MagicMock())
    return m


@pytest.fixture
def mock_network_mgmt():
    m = MagicMock()
    m.NetworkManagementClient = MagicMock(return_value=MagicMock())
    return m


@pytest.fixture
def azure_patches(
    mock_identity,
    mock_compute_mgmt,
    mock_storage,
    mock_network_mgmt,
):
    """Apply all module-level patches for the Azure SDK."""
    patches = [
        patch(f"{_MODULE}._HAS_IDENTITY", True),
        patch(f"{_MODULE}._HAS_COMPUTE", True),
        patch(f"{_MODULE}._HAS_STORAGE", True),
        patch(f"{_MODULE}._HAS_NETWORK", True),
        patch(
            f"{_MODULE}.ClientSecretCredential",
            mock_identity.ClientSecretCredential,
            create=True,
        ),
        patch(
            f"{_MODULE}.DefaultAzureCredential",
            mock_identity.DefaultAzureCredential,
            create=True,
        ),
        patch(
            f"{_MODULE}.ComputeManagementClient",
            mock_compute_mgmt.ComputeManagementClient,
            create=True,
        ),
        patch(
            f"{_MODULE}.BlobServiceClient",
            mock_storage.BlobServiceClient,
            create=True,
        ),
        patch(
            f"{_MODULE}.NetworkManagementClient",
            mock_network_mgmt.NetworkManagementClient,
            create=True,
        ),
        _to_thread_patch(),
    ]
    for p in patches:
        p.start()
    yield {
        "identity": mock_identity,
        "compute_mgmt": mock_compute_mgmt,
        "storage": mock_storage,
        "network_mgmt": mock_network_mgmt,
    }
    for p in patches:
        p.stop()


# ======================================================================
# Tests
# ======================================================================


class TestAzureInit:
    def test_init_defaults(self, azure_patches):
        provider = _make_provider()
        assert provider.provider_name == "azure"
        assert provider._connected is False
        assert provider._subscription_id == "sub-test-123"
        assert provider._resource_group == "rg-artenic"

    def test_init_raises_without_identity(self):
        with (
            patch(f"{_MODULE}._HAS_IDENTITY", False),
            patch(f"{_MODULE}._HAS_COMPUTE", True),
            patch(f"{_MODULE}._HAS_STORAGE", True),
            patch(f"{_MODULE}._HAS_NETWORK", True),
            pytest.raises(RuntimeError, match="azure-identity"),
        ):
            _make_provider()

    def test_init_raises_without_compute(self):
        with (
            patch(f"{_MODULE}._HAS_IDENTITY", True),
            patch(f"{_MODULE}._HAS_COMPUTE", False),
            patch(f"{_MODULE}._HAS_STORAGE", True),
            patch(f"{_MODULE}._HAS_NETWORK", True),
            pytest.raises(RuntimeError, match="azure-mgmt-compute"),
        ):
            _make_provider()

    def test_init_raises_without_storage(self):
        with (
            patch(f"{_MODULE}._HAS_IDENTITY", True),
            patch(f"{_MODULE}._HAS_COMPUTE", True),
            patch(f"{_MODULE}._HAS_STORAGE", False),
            patch(f"{_MODULE}._HAS_NETWORK", True),
            pytest.raises(RuntimeError, match="azure-mgmt-storage"),
        ):
            _make_provider()

    def test_init_raises_without_network(self):
        with (
            patch(f"{_MODULE}._HAS_IDENTITY", True),
            patch(f"{_MODULE}._HAS_COMPUTE", True),
            patch(f"{_MODULE}._HAS_STORAGE", True),
            patch(f"{_MODULE}._HAS_NETWORK", False),
            pytest.raises(RuntimeError, match="azure-mgmt-network"),
        ):
            _make_provider()


class TestAzureProviderName:
    def test_provider_name(self, azure_patches):
        provider = _make_provider()
        assert provider.provider_name == "azure"


class TestAzureConnect:
    async def test_connect_with_client_secret(self, azure_patches):
        provider = _make_provider()

        # Mock the container_client so get_container_properties works
        blob_svc = azure_patches["storage"].BlobServiceClient.return_value
        container_client = MagicMock()
        blob_svc.get_container_client.return_value = container_client

        await provider._connect()

        azure_patches["identity"].ClientSecretCredential.assert_called_once()
        assert provider._compute_client is not None
        assert provider._network_client is not None
        assert provider._blob_service_client is not None

    async def test_connect_with_default_credential(self, azure_patches):
        provider = _make_provider(
            client_id=None,
            client_secret=None,
            tenant_id=None,
        )

        blob_svc = azure_patches["storage"].BlobServiceClient.return_value
        container_client = MagicMock()
        blob_svc.get_container_client.return_value = container_client

        await provider._connect()

        azure_patches["identity"].DefaultAzureCredential.assert_called_once()

    async def test_connect_creates_container(self, azure_patches):
        provider = _make_provider()

        blob_svc = azure_patches["storage"].BlobServiceClient.return_value
        container_client = MagicMock()
        container_client.get_container_properties.side_effect = Exception("not found")
        blob_svc.get_container_client.return_value = container_client

        await provider._connect()

        container_client.create_container.assert_called_once()


class TestAzureDisconnect:
    async def test_disconnect_clears_clients(self, azure_patches):
        provider = _make_provider()
        provider._credential = MagicMock()
        provider._compute_client = MagicMock()
        provider._network_client = MagicMock()
        provider._blob_service_client = MagicMock()
        provider._container_client = MagicMock()

        await provider._disconnect()

        assert provider._credential is None
        assert provider._compute_client is None
        assert provider._network_client is None
        assert provider._blob_service_client is None
        assert provider._container_client is None

    async def test_disconnect_handles_close_error(self, azure_patches):
        provider = _make_provider()
        blob_mock = MagicMock()
        blob_mock.close.side_effect = RuntimeError("close failed")
        provider._blob_service_client = blob_mock
        provider._credential = MagicMock()
        provider._compute_client = MagicMock()
        provider._network_client = MagicMock()
        provider._container_client = MagicMock()

        # Should not raise
        await provider._disconnect()
        assert provider._blob_service_client is None


class TestAzureListInstances:
    async def test_list_instances(self, azure_patches):
        provider = _make_provider()

        # Mock compute client
        mock_size = MagicMock()
        mock_size.name = "Standard_NC6s_v3"
        mock_size.number_of_cores = 6
        mock_size.memory_in_mb = 114688
        mock_size.max_data_disk_count = 12
        mock_size.os_disk_size_in_mb = 1047552
        mock_size.resource_disk_size_in_mb = 344064

        provider._compute_client = MagicMock()
        provider._compute_client.virtual_machine_sizes.list.return_value = [
            mock_size,
        ]

        # Patch httpx pricing call to return empty data
        with patch(f"{_MODULE}.httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "Items": [],
                "NextPageLink": None,
            }
            mock_response.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(
                return_value=mock_client,
            )
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.AsyncClient.return_value = mock_client

            instances = await provider._list_instances()

        assert len(instances) >= 1
        assert instances[0].name == "Standard_NC6s_v3"
        assert instances[0].gpu_type == "V100"
        assert instances[0].gpu_count == 1


class TestAzureUploadCode:
    async def test_upload_code(self, azure_patches, tmp_path):
        provider = _make_provider()
        provider._container_client = MagicMock()

        mock_blob_client = MagicMock()
        provider._container_client.get_blob_client.return_value = mock_blob_client

        (tmp_path / "train.py").write_text("print('train')")

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="azure",
            config={"source_dir": str(tmp_path)},
        )
        code_prefix = await provider._upload_code(spec)

        assert code_prefix.startswith("training/")
        mock_blob_client.upload_blob.assert_called_once()


class TestAzureProvisionAndStart:
    async def test_provision_and_start(self, azure_patches):
        provider = _make_provider()
        provider._compute_client = MagicMock()
        provider._network_client = MagicMock()
        provider._container_client = MagicMock()

        # Mock IP creation
        ip_poller = MagicMock()
        ip_resource = MagicMock()
        ip_resource.id = "/subs/rg/providers/ip-id"
        ip_poller.result.return_value = ip_resource
        nw = provider._network_client
        nw.public_ip_addresses.begin_create_or_update.return_value = ip_poller

        # Mock VNet and Subnet
        vnet = MagicMock()
        vnet.name = "default-vnet"
        nw.virtual_networks.list.return_value = [vnet]
        subnet = MagicMock()
        subnet.id = "/subs/rg/providers/subnet-id"
        nw.subnets.list.return_value = [subnet]

        # Mock NIC creation
        nic_poller = MagicMock()
        nic_resource = MagicMock()
        nic_resource.id = "/subs/rg/providers/nic-id"
        nic_poller.result.return_value = nic_resource
        nw.network_interfaces.begin_create_or_update.return_value = nic_poller

        # Mock VM creation
        vm_poller = MagicMock()
        vm_result = MagicMock()
        vm_result.storage_profile.os_disk.name = "osdisk-1"
        vm_poller.result.return_value = vm_result
        cc = provider._compute_client
        cc.virtual_machines.begin_create_or_update.return_value = vm_poller

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="azure",
        )
        job_id = await provider._provision_and_start(spec)

        assert job_id.startswith("azure-")
        assert job_id in provider._jobs


class TestAzurePollProvider:
    async def test_poll_running(self, azure_patches):
        from artenic_ai_platform.providers.azure import _AzureJobState

        provider = _make_provider()
        provider._compute_client = MagicMock()
        provider._container_client = MagicMock()

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="azure",
        )
        state = _AzureJobState(
            vm_name="artenic-azure-abc",
            nic_name="artenic-azure-abc-nic",
            ip_name="artenic-azure-abc-ip",
            os_disk_name="osdisk",
            resource_group="rg-artenic",
            created_at=time.time() - 60,
            spec=spec,
            code_prefix="training/nlp/bert",
        )
        provider._jobs["azure-abc"] = state

        # Mock VM instance view
        vm = MagicMock()
        status_obj = MagicMock()
        status_obj.code = "PowerState/running"
        vm.instance_view.statuses = [status_obj]
        cc = provider._compute_client
        cc.virtual_machines.get.return_value = vm

        # No status.json in blob
        blob_client = MagicMock()
        blob_client.download_blob.side_effect = Exception("not found")
        provider._container_client.get_blob_client.return_value = blob_client

        status = await provider._poll_provider("azure-abc")
        assert status.status == JobStatus.RUNNING

    async def test_poll_completed_from_blob(self, azure_patches):
        from artenic_ai_platform.providers.azure import _AzureJobState

        provider = _make_provider()
        provider._compute_client = MagicMock()
        provider._container_client = MagicMock()

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="azure",
        )
        state = _AzureJobState(
            vm_name="artenic-azure-def",
            nic_name="artenic-azure-def-nic",
            ip_name="artenic-azure-def-ip",
            os_disk_name="osdisk",
            resource_group="rg-artenic",
            created_at=time.time() - 120,
            spec=spec,
            code_prefix="training/nlp/bert",
        )
        provider._jobs["azure-def"] = state

        vm = MagicMock()
        status_obj = MagicMock()
        status_obj.code = "PowerState/running"
        vm.instance_view.statuses = [status_obj]
        cc = provider._compute_client
        cc.virtual_machines.get.return_value = vm

        # status.json says completed
        blob_client = MagicMock()
        download_mock = MagicMock()
        download_mock.readall.return_value = json.dumps(
            {"status": "completed"},
        ).encode()
        blob_client.download_blob.return_value = download_mock
        provider._container_client.get_blob_client.return_value = blob_client

        status = await provider._poll_provider("azure-def")
        assert status.status == JobStatus.COMPLETED

    async def test_poll_no_state(self, azure_patches):
        provider = _make_provider()
        status = await provider._poll_provider("unknown")
        assert status.status == JobStatus.FAILED
        assert "No state" in (status.error or "")

    async def test_poll_vm_not_found(self, azure_patches):
        from artenic_ai_platform.providers.azure import _AzureJobState

        provider = _make_provider()
        provider._compute_client = MagicMock()

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="azure",
        )
        state = _AzureJobState(
            vm_name="gone-vm",
            nic_name="gone-nic",
            ip_name="gone-ip",
            os_disk_name="gone-disk",
            resource_group="rg-artenic",
            created_at=time.time() - 60,
            spec=spec,
            code_prefix="training/nlp/bert",
        )
        provider._jobs["azure-gone"] = state

        cc = provider._compute_client
        cc.virtual_machines.get.side_effect = Exception("ResourceNotFound")

        status = await provider._poll_provider("azure-gone")
        assert status.status == JobStatus.FAILED
        assert "no longer exists" in (status.error or "")

    async def test_poll_spot_deallocated(self, azure_patches):
        from artenic_ai_platform.providers.azure import _AzureJobState

        provider = _make_provider()
        provider._compute_client = MagicMock()
        provider._container_client = MagicMock()

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="azure",
            is_spot=True,
        )
        state = _AzureJobState(
            vm_name="spot-vm",
            nic_name="spot-nic",
            ip_name="spot-ip",
            os_disk_name="spot-disk",
            resource_group="rg-artenic",
            created_at=time.time() - 60,
            spec=spec,
            code_prefix="training/nlp/bert",
        )
        provider._jobs["azure-spot"] = state

        vm = MagicMock()
        status_obj = MagicMock()
        status_obj.code = "PowerState/deallocated"
        vm.instance_view.statuses = [status_obj]
        cc = provider._compute_client
        cc.virtual_machines.get.return_value = vm

        # No status blob
        blob_client = MagicMock()
        blob_client.download_blob.side_effect = Exception("not found")
        provider._container_client.get_blob_client.return_value = blob_client

        status = await provider._poll_provider("azure-spot")
        assert status.status == JobStatus.PREEMPTED


class TestAzureCollectArtifacts:
    async def test_collect_artifacts(self, azure_patches):
        provider = _make_provider()
        provider._container_client = MagicMock()

        mock_blob = MagicMock()
        mock_blob.name = "artifacts/job-a/output/model.pt"
        provider._container_client.list_blobs.return_value = [mock_blob]

        blob_client = MagicMock()
        download_mock = MagicMock()
        download_mock.readall.return_value = b"model-bytes"
        blob_client.download_blob.return_value = download_mock
        provider._container_client.get_blob_client.return_value = blob_client

        dummy_status = CloudJobStatus(
            provider_job_id="job-a",
            status=JobStatus.COMPLETED,
        )
        result = await provider._collect_artifacts("job-a", dummy_status)
        assert result is not None

    async def test_collect_artifacts_none(self, azure_patches):
        provider = _make_provider()
        provider._container_client = MagicMock()
        provider._container_client.list_blobs.return_value = []

        dummy_status = CloudJobStatus(
            provider_job_id="job-b",
            status=JobStatus.COMPLETED,
        )
        result = await provider._collect_artifacts("job-b", dummy_status)
        assert result is None


class TestAzureCleanupCompute:
    async def test_cleanup_compute(self, azure_patches):
        from artenic_ai_platform.providers.azure import _AzureJobState

        provider = _make_provider()
        provider._compute_client = MagicMock()
        provider._network_client = MagicMock()

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="azure",
        )
        state = _AzureJobState(
            vm_name="vm-clean",
            nic_name="nic-clean",
            ip_name="ip-clean",
            os_disk_name="disk-clean",
            resource_group="rg-artenic",
            created_at=time.time(),
            spec=spec,
            code_prefix="training/nlp/bert",
        )
        provider._jobs["job-clean"] = state

        # Mock pollers for delete operations
        for attr in [
            "virtual_machines",
            "disks",
        ]:
            mock_begin = getattr(
                provider._compute_client,
                attr,
            )
            poller = MagicMock()
            poller.result.return_value = None
            mock_begin.begin_delete.return_value = poller

        for attr in [
            "network_interfaces",
            "public_ip_addresses",
        ]:
            mock_begin = getattr(
                provider._network_client,
                attr,
            )
            poller = MagicMock()
            poller.result.return_value = None
            mock_begin.begin_delete.return_value = poller

        await provider._cleanup_compute("job-clean")
        assert "job-clean" not in provider._jobs

    async def test_cleanup_no_state(self, azure_patches):
        provider = _make_provider()
        # Should not raise
        await provider._cleanup_compute("missing")


class TestAzureCancelProviderJob:
    async def test_cancel_provider_job(self, azure_patches):
        from artenic_ai_platform.providers.azure import _AzureJobState

        provider = _make_provider()
        provider._compute_client = MagicMock()

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="azure",
        )
        state = _AzureJobState(
            vm_name="vm-cancel",
            nic_name="nic-cancel",
            ip_name="ip-cancel",
            os_disk_name="disk-cancel",
            resource_group="rg-artenic",
            created_at=time.time(),
            spec=spec,
            code_prefix="training/nlp/bert",
        )
        provider._jobs["job-cancel"] = state

        dealloc_poller = MagicMock()
        dealloc_poller.result.return_value = None
        cc = provider._compute_client
        cc.virtual_machines.begin_deallocate.return_value = dealloc_poller

        delete_poller = MagicMock()
        delete_poller.result.return_value = None
        cc.virtual_machines.begin_delete.return_value = delete_poller

        await provider._cancel_provider_job("job-cancel")

        cc.virtual_machines.begin_deallocate.assert_called_once()
        cc.virtual_machines.begin_delete.assert_called_once()

    async def test_cancel_no_state(self, azure_patches):
        provider = _make_provider()
        # Should not raise
        await provider._cancel_provider_job("unknown")

    async def test_cancel_vm_already_gone(self, azure_patches):
        from artenic_ai_platform.providers.azure import _AzureJobState

        provider = _make_provider()
        provider._compute_client = MagicMock()

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="azure",
        )
        state = _AzureJobState(
            vm_name="vm-gone",
            nic_name="nic-gone",
            ip_name="ip-gone",
            os_disk_name="disk-gone",
            resource_group="rg-artenic",
            created_at=time.time(),
            spec=spec,
            code_prefix="training/nlp/bert",
        )
        provider._jobs["job-gone"] = state

        cc = provider._compute_client
        cc.virtual_machines.begin_deallocate.side_effect = Exception("ResourceNotFound")
        delete_poller = MagicMock()
        delete_poller.result.return_value = None
        cc.virtual_machines.begin_delete.return_value = delete_poller

        # Should not raise
        await provider._cancel_provider_job("job-gone")


# ======================================================================
# Additional tests for full coverage
# ======================================================================


class TestAzureListInstancesGpuOnlyFilter:
    """Cover gpu_only filtering (line 449)."""

    async def test_list_instances_gpu_only_skips_non_gpu(self, azure_patches):
        provider = _make_provider()

        # Create a non-GPU VM size
        mock_size = MagicMock()
        mock_size.name = "Standard_D2s_v3"
        mock_size.number_of_cores = 2
        mock_size.memory_in_mb = 8192
        mock_size.max_data_disk_count = 4
        mock_size.os_disk_size_in_mb = 1047552
        mock_size.resource_disk_size_in_mb = 16384

        provider._compute_client = MagicMock()
        provider._compute_client.virtual_machine_sizes.list.return_value = [mock_size]

        with patch(f"{_MODULE}.httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"Items": [], "NextPageLink": None}
            mock_response.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.AsyncClient.return_value = mock_client

            instances = await provider._list_instances(gpu_only=True)

        assert len(instances) == 0


class TestAzureFetchRetailPricing:
    """Cover pricing parsing logic (lines 512-533, 539-540)."""

    async def test_pricing_windows_entries_skipped(self, azure_patches):
        """Windows VM pricing entries should be skipped."""
        provider = _make_provider()
        provider._compute_client = MagicMock()

        mock_size = MagicMock()
        mock_size.name = "Standard_NC6s_v3"
        mock_size.number_of_cores = 6
        mock_size.memory_in_mb = 114688
        mock_size.max_data_disk_count = 12
        mock_size.os_disk_size_in_mb = 1047552
        mock_size.resource_disk_size_in_mb = 344064
        provider._compute_client.virtual_machine_sizes.list.return_value = [mock_size]

        pricing_items = [
            {
                "armSkuName": "Standard_NC6s_v3",
                "retailPrice": 5.0,
                "skuName": "NC6s v3 Windows",
                "meterName": "NC6s v3",
            },
            {
                "armSkuName": "Standard_NC6s_v3",
                "retailPrice": 3.06,
                "skuName": "NC6s v3",
                "meterName": "NC6s v3",
            },
        ]

        with patch(f"{_MODULE}.httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "Items": pricing_items,
                "NextPageLink": None,
            }
            mock_response.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.AsyncClient.return_value = mock_client

            instances = await provider._list_instances()

        assert len(instances) == 1
        # Price should be the Linux price (3.06 USD * 0.92)
        assert instances[0].price_per_hour_eur == round(3.06 * 0.92, 6)

    async def test_pricing_spot_entries(self, azure_patches):
        """Spot pricing should be populated."""
        provider = _make_provider()
        provider._compute_client = MagicMock()

        mock_size = MagicMock()
        mock_size.name = "Standard_NC6s_v3"
        mock_size.number_of_cores = 6
        mock_size.memory_in_mb = 114688
        mock_size.max_data_disk_count = 12
        mock_size.os_disk_size_in_mb = 1047552
        mock_size.resource_disk_size_in_mb = 344064
        provider._compute_client.virtual_machine_sizes.list.return_value = [mock_size]

        pricing_items = [
            {
                "armSkuName": "Standard_NC6s_v3",
                "retailPrice": 3.06,
                "skuName": "NC6s v3",
                "meterName": "NC6s v3",
            },
            {
                "armSkuName": "Standard_NC6s_v3",
                "retailPrice": 0.92,
                "skuName": "NC6s v3 Spot",
                "meterName": "NC6s v3 Spot",
            },
        ]

        with patch(f"{_MODULE}.httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "Items": pricing_items,
                "NextPageLink": None,
            }
            mock_response.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.AsyncClient.return_value = mock_client

            instances = await provider._list_instances()

        assert len(instances) == 1
        assert instances[0].spot_price_per_hour_eur is not None
        assert instances[0].spot_price_per_hour_eur == round(0.92 * 0.92, 6)

    async def test_pricing_low_priority_skipped(self, azure_patches):
        """Low priority entries should be skipped for on-demand pricing."""
        provider = _make_provider()
        provider._compute_client = MagicMock()

        mock_size = MagicMock()
        mock_size.name = "Standard_NC6s_v3"
        mock_size.number_of_cores = 6
        mock_size.memory_in_mb = 114688
        mock_size.max_data_disk_count = 12
        mock_size.os_disk_size_in_mb = 1047552
        mock_size.resource_disk_size_in_mb = 344064
        provider._compute_client.virtual_machine_sizes.list.return_value = [mock_size]

        pricing_items = [
            {
                "armSkuName": "Standard_NC6s_v3",
                "retailPrice": 1.0,
                "skuName": "NC6s v3 Low Priority",
                "meterName": "NC6s v3 Low Priority",
            },
            {
                "armSkuName": "Standard_NC6s_v3",
                "retailPrice": 3.06,
                "skuName": "NC6s v3",
                "meterName": "NC6s v3",
            },
        ]

        with patch(f"{_MODULE}.httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "Items": pricing_items,
                "NextPageLink": None,
            }
            mock_response.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.AsyncClient.return_value = mock_client

            instances = await provider._list_instances()

        # On-demand should be 3.06, not 1.0 (low priority)
        assert instances[0].price_per_hour_eur == round(3.06 * 0.92, 6)

    async def test_pricing_fetch_exception(self, azure_patches):
        """When pricing API fails, prices default to 0."""
        provider = _make_provider()
        provider._compute_client = MagicMock()

        mock_size = MagicMock()
        mock_size.name = "Standard_NC6s_v3"
        mock_size.number_of_cores = 6
        mock_size.memory_in_mb = 114688
        mock_size.max_data_disk_count = 12
        mock_size.os_disk_size_in_mb = 1047552
        mock_size.resource_disk_size_in_mb = 344064
        provider._compute_client.virtual_machine_sizes.list.return_value = [mock_size]

        with patch(f"{_MODULE}.httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=Exception("network error"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.AsyncClient.return_value = mock_client

            instances = await provider._list_instances()

        assert len(instances) == 1
        assert instances[0].price_per_hour_eur == 0.0


class TestAzurePricingUnknownSku:
    """Cover line 514: pricing entry with armSkuName not in listed VMs is skipped."""

    async def test_pricing_unknown_sku_skipped(self, azure_patches):
        provider = _make_provider()
        provider._compute_client = MagicMock()

        mock_size = MagicMock()
        mock_size.name = "Standard_NC6s_v3"
        mock_size.number_of_cores = 6
        mock_size.memory_in_mb = 114688
        mock_size.max_data_disk_count = 12
        mock_size.os_disk_size_in_mb = 1047552
        mock_size.resource_disk_size_in_mb = 344064
        provider._compute_client.virtual_machine_sizes.list.return_value = [mock_size]

        pricing_items = [
            {
                "armSkuName": "Standard_UNKNOWN_VM",
                "retailPrice": 99.0,
                "skuName": "Unknown VM",
                "meterName": "Unknown",
            },
            {
                "armSkuName": "Standard_NC6s_v3",
                "retailPrice": 3.06,
                "skuName": "NC6s v3",
                "meterName": "NC6s v3",
            },
        ]

        with patch(f"{_MODULE}.httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "Items": pricing_items,
                "NextPageLink": None,
            }
            mock_response.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.AsyncClient.return_value = mock_client

            instances = await provider._list_instances()

        assert len(instances) == 1
        assert instances[0].name == "Standard_NC6s_v3"
        assert instances[0].price_per_hour_eur == round(3.06 * 0.92, 6)


class TestAzureProvisionNoVNets:
    """Cover no VNets/subnets errors (lines 649, 656)."""

    async def test_provision_no_vnets(self, azure_patches):
        provider = _make_provider()
        provider._compute_client = MagicMock()
        provider._network_client = MagicMock()
        provider._container_client = MagicMock()

        # Mock IP creation
        ip_poller = MagicMock()
        ip_resource = MagicMock()
        ip_resource.id = "/subs/rg/providers/ip-id"
        ip_poller.result.return_value = ip_resource
        nw = provider._network_client
        nw.public_ip_addresses.begin_create_or_update.return_value = ip_poller

        # No VNets
        nw.virtual_networks.list.return_value = []

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="azure",
        )
        with pytest.raises(RuntimeError, match="No virtual networks"):
            await provider._provision_and_start(spec)

    async def test_provision_no_subnets(self, azure_patches):
        provider = _make_provider()
        provider._compute_client = MagicMock()
        provider._network_client = MagicMock()
        provider._container_client = MagicMock()

        # Mock IP creation
        ip_poller = MagicMock()
        ip_resource = MagicMock()
        ip_resource.id = "/subs/rg/providers/ip-id"
        ip_poller.result.return_value = ip_resource
        nw = provider._network_client
        nw.public_ip_addresses.begin_create_or_update.return_value = ip_poller

        # VNet exists but no subnets
        vnet = MagicMock()
        vnet.name = "default-vnet"
        nw.virtual_networks.list.return_value = [vnet]
        nw.subnets.list.return_value = []

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="azure",
        )
        with pytest.raises(RuntimeError, match="No subnets"):
            await provider._provision_and_start(spec)


class TestAzureProvisionSSHKey:
    """Cover SSH key configuration (lines 688-691, 708)."""

    async def test_provision_with_ssh_key(self, azure_patches, tmp_path):
        provider = _make_provider(ssh_public_key_path=str(tmp_path / "id_rsa.pub"))
        provider._compute_client = MagicMock()
        provider._network_client = MagicMock()
        provider._container_client = MagicMock()

        # Write SSH key file
        ssh_key_file = tmp_path / "id_rsa.pub"
        ssh_key_file.write_text("ssh-rsa AAAA... user@host")

        # Mock IP creation
        ip_poller = MagicMock()
        ip_resource = MagicMock()
        ip_resource.id = "/subs/rg/providers/ip-id"
        ip_poller.result.return_value = ip_resource
        nw = provider._network_client
        nw.public_ip_addresses.begin_create_or_update.return_value = ip_poller

        # Mock VNet and Subnet
        vnet = MagicMock()
        vnet.name = "default-vnet"
        nw.virtual_networks.list.return_value = [vnet]
        subnet = MagicMock()
        subnet.id = "/subs/rg/providers/subnet-id"
        nw.subnets.list.return_value = [subnet]

        # Mock NIC creation
        nic_poller = MagicMock()
        nic_resource = MagicMock()
        nic_resource.id = "/subs/rg/providers/nic-id"
        nic_poller.result.return_value = nic_resource
        nw.network_interfaces.begin_create_or_update.return_value = nic_poller

        # Mock VM creation
        vm_poller = MagicMock()
        vm_result = MagicMock()
        vm_result.storage_profile.os_disk.name = "osdisk-ssh"
        vm_poller.result.return_value = vm_result
        cc = provider._compute_client
        cc.virtual_machines.begin_create_or_update.return_value = vm_poller

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="azure",
        )
        job_id = await provider._provision_and_start(spec)
        assert job_id.startswith("azure-")

        # Verify SSH key was in the VM parameters
        vm_params = cc.virtual_machines.begin_create_or_update.call_args[0][2]
        os_prof = vm_params["os_profile"]
        assert "linux_configuration" in os_prof
        assert os_prof["linux_configuration"]["disable_password_authentication"] is True


class TestAzureProvisionSpot:
    """Cover spot VM parameters (lines 753-755)."""

    async def test_provision_spot_vm(self, azure_patches):
        provider = _make_provider()
        provider._compute_client = MagicMock()
        provider._network_client = MagicMock()
        provider._container_client = MagicMock()

        # Mock IP creation
        ip_poller = MagicMock()
        ip_resource = MagicMock()
        ip_resource.id = "/subs/rg/providers/ip-id"
        ip_poller.result.return_value = ip_resource
        nw = provider._network_client
        nw.public_ip_addresses.begin_create_or_update.return_value = ip_poller

        # Mock VNet and Subnet
        vnet = MagicMock()
        vnet.name = "default-vnet"
        nw.virtual_networks.list.return_value = [vnet]
        subnet = MagicMock()
        subnet.id = "/subs/rg/providers/subnet-id"
        nw.subnets.list.return_value = [subnet]

        # Mock NIC creation
        nic_poller = MagicMock()
        nic_resource = MagicMock()
        nic_resource.id = "/subs/rg/providers/nic-id"
        nic_poller.result.return_value = nic_resource
        nw.network_interfaces.begin_create_or_update.return_value = nic_poller

        # Mock VM creation
        vm_poller = MagicMock()
        vm_result = MagicMock()
        vm_result.storage_profile.os_disk.name = "osdisk-spot"
        vm_poller.result.return_value = vm_result
        cc = provider._compute_client
        cc.virtual_machines.begin_create_or_update.return_value = vm_poller

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="azure",
            is_spot=True,
        )
        job_id = await provider._provision_and_start(spec)
        assert job_id.startswith("azure-")

        # Verify spot parameters
        vm_params = cc.virtual_machines.begin_create_or_update.call_args[0][2]
        assert vm_params["priority"] == "Spot"
        assert vm_params["eviction_policy"] == "Deallocate"
        assert vm_params["billing_profile"] == {"max_price": -1}


class TestAzurePollEdgeCases:
    """Cover poll VM status exception and blob status branches."""

    async def test_poll_vm_generic_error(self, azure_patches):
        """Non-ResourceNotFound errors return RUNNING with error message (lines 833-838)."""
        from artenic_ai_platform.providers.azure import _AzureJobState

        provider = _make_provider()
        provider._compute_client = MagicMock()

        spec = TrainingSpec(service="nlp", model="bert", provider="azure")
        state = _AzureJobState(
            vm_name="vm-err",
            nic_name="nic-err",
            ip_name="ip-err",
            os_disk_name="disk-err",
            resource_group="rg-artenic",
            created_at=time.time() - 60,
            spec=spec,
            code_prefix="training/nlp/bert",
        )
        provider._jobs["azure-err"] = state

        cc = provider._compute_client
        cc.virtual_machines.get.side_effect = Exception("timeout error")

        status = await provider._poll_provider("azure-err")
        assert status.status == JobStatus.RUNNING
        assert "Could not query VM" in (status.error or "")

    async def test_poll_blob_status_failed(self, azure_patches):
        """Blob status.json with 'failed' (lines 860-866)."""
        from artenic_ai_platform.providers.azure import _AzureJobState

        provider = _make_provider()
        provider._compute_client = MagicMock()
        provider._container_client = MagicMock()

        spec = TrainingSpec(service="nlp", model="bert", provider="azure")
        state = _AzureJobState(
            vm_name="vm-failed",
            nic_name="nic-failed",
            ip_name="ip-failed",
            os_disk_name="disk-failed",
            resource_group="rg-artenic",
            created_at=time.time() - 120,
            spec=spec,
            code_prefix="training/nlp/bert",
        )
        provider._jobs["azure-fail"] = state

        vm = MagicMock()
        status_obj = MagicMock()
        status_obj.code = "PowerState/running"
        vm.instance_view.statuses = [status_obj]
        cc = provider._compute_client
        cc.virtual_machines.get.return_value = vm

        # status.json says failed
        blob_client = MagicMock()
        download_mock = MagicMock()
        download_mock.readall.return_value = json.dumps(
            {"status": "failed", "exit_code": 1},
        ).encode()
        blob_client.download_blob.return_value = download_mock
        provider._container_client.get_blob_client.return_value = blob_client

        status = await provider._poll_provider("azure-fail")
        assert status.status == JobStatus.FAILED
        assert "exit_code" in (status.error or "") or "code" in (status.error or "")

    async def test_poll_blob_status_running(self, azure_patches):
        """Blob status.json with 'running' (lines 867-868)."""
        from artenic_ai_platform.providers.azure import _AzureJobState

        provider = _make_provider()
        provider._compute_client = MagicMock()
        provider._container_client = MagicMock()

        spec = TrainingSpec(service="nlp", model="bert", provider="azure")
        state = _AzureJobState(
            vm_name="vm-running",
            nic_name="nic-running",
            ip_name="ip-running",
            os_disk_name="disk-running",
            resource_group="rg-artenic",
            created_at=time.time() - 60,
            spec=spec,
            code_prefix="training/nlp/bert",
        )
        provider._jobs["azure-run"] = state

        vm = MagicMock()
        status_obj = MagicMock()
        status_obj.code = "PowerState/running"
        vm.instance_view.statuses = [status_obj]
        cc = provider._compute_client
        cc.virtual_machines.get.return_value = vm

        # status.json says running
        blob_client = MagicMock()
        download_mock = MagicMock()
        download_mock.readall.return_value = json.dumps(
            {"status": "running"},
        ).encode()
        blob_client.download_blob.return_value = download_mock
        provider._container_client.get_blob_client.return_value = blob_client

        status = await provider._poll_provider("azure-run")
        assert status.status == JobStatus.RUNNING


class TestAzureCollectArtifactsEdgeCases:
    """Cover artifact collection edge cases (line 946)."""

    async def test_collect_artifacts_skips_empty_relative(self, azure_patches):
        """Blobs with empty relative paths are skipped."""
        provider = _make_provider()
        provider._container_client = MagicMock()

        # A blob whose name equals the prefix (empty relative)
        mock_blob_empty = MagicMock()
        mock_blob_empty.name = "artifacts/job-x/output/"

        mock_blob_valid = MagicMock()
        mock_blob_valid.name = "artifacts/job-x/output/model.pt"

        provider._container_client.list_blobs.return_value = [
            mock_blob_empty,
            mock_blob_valid,
        ]

        blob_client = MagicMock()
        download_mock = MagicMock()
        download_mock.readall.return_value = b"model-bytes"
        blob_client.download_blob.return_value = download_mock
        provider._container_client.get_blob_client.return_value = blob_client

        dummy_status = CloudJobStatus(
            provider_job_id="job-x",
            status=JobStatus.COMPLETED,
        )
        result = await provider._collect_artifacts("job-x", dummy_status)
        assert result is not None
        # get_blob_client should only be called for the valid blob
        calls = provider._container_client.get_blob_client.call_args_list
        assert len(calls) == 1
        assert calls[0][0][0] == "artifacts/job-x/output/model.pt"


class TestAzureDeleteResource:
    """Cover _delete_resource edge cases (lines 1019-1024, 1029-1030)."""

    async def test_delete_resource_not_found(self, azure_patches):
        """ResourceNotFound is silently ignored."""
        provider = _make_provider()

        def _fake_delete():
            raise Exception("ResourceNotFound: The resource was not found")

        # Should not raise
        await provider._delete_resource("VM", _fake_delete)

    async def test_delete_resource_other_error_logged(self, azure_patches):
        """Non-ResourceNotFound errors are raised inside _do_delete, caught outside."""
        provider = _make_provider()

        def _fake_delete():
            raise RuntimeError("disk locked")

        # The outer try/except catches and logs the error
        await provider._delete_resource("OSDisk", _fake_delete)


class TestAzureCancelEdgeCases:
    """Cover cancel deallocation/delete non-ResourceNotFound (lines 1064, 1082-1087)."""

    async def test_cancel_deallocation_other_error(self, azure_patches):
        """Non-ResourceNotFound deallocation error is warned, not raised (line 1064)."""
        from artenic_ai_platform.providers.azure import _AzureJobState

        provider = _make_provider()
        provider._compute_client = MagicMock()

        spec = TrainingSpec(service="nlp", model="bert", provider="azure")
        state = _AzureJobState(
            vm_name="vm-dealloc",
            nic_name="nic-dealloc",
            ip_name="ip-dealloc",
            os_disk_name="disk-dealloc",
            resource_group="rg-artenic",
            created_at=time.time(),
            spec=spec,
            code_prefix="training/nlp/bert",
        )
        provider._jobs["job-dealloc"] = state

        cc = provider._compute_client
        # Deallocation raises non-ResourceNotFound
        cc.virtual_machines.begin_deallocate.side_effect = Exception("throttled")
        # Delete succeeds
        delete_poller = MagicMock()
        delete_poller.result.return_value = None
        cc.virtual_machines.begin_delete.return_value = delete_poller

        # Should not raise
        await provider._cancel_provider_job("job-dealloc")

    async def test_cancel_delete_other_error(self, azure_patches):
        """Non-ResourceNotFound delete error is warned, not raised (lines 1082-1087)."""
        from artenic_ai_platform.providers.azure import _AzureJobState

        provider = _make_provider()
        provider._compute_client = MagicMock()

        spec = TrainingSpec(service="nlp", model="bert", provider="azure")
        state = _AzureJobState(
            vm_name="vm-delerr",
            nic_name="nic-delerr",
            ip_name="ip-delerr",
            os_disk_name="disk-delerr",
            resource_group="rg-artenic",
            created_at=time.time(),
            spec=spec,
            code_prefix="training/nlp/bert",
        )
        provider._jobs["job-delerr"] = state

        cc = provider._compute_client
        # Deallocation succeeds
        dealloc_poller = MagicMock()
        dealloc_poller.result.return_value = None
        cc.virtual_machines.begin_deallocate.return_value = dealloc_poller
        # Delete raises non-ResourceNotFound
        cc.virtual_machines.begin_delete.side_effect = Exception("internal error")

        # Should not raise
        await provider._cancel_provider_job("job-delerr")

    async def test_cancel_delete_resource_not_found(self, azure_patches):
        """ResourceNotFound on delete is silently ignored (lines 1082-1085)."""
        from artenic_ai_platform.providers.azure import _AzureJobState

        provider = _make_provider()
        provider._compute_client = MagicMock()

        spec = TrainingSpec(service="nlp", model="bert", provider="azure")
        state = _AzureJobState(
            vm_name="vm-delnf",
            nic_name="nic-delnf",
            ip_name="ip-delnf",
            os_disk_name="disk-delnf",
            resource_group="rg-artenic",
            created_at=time.time(),
            spec=spec,
            code_prefix="training/nlp/bert",
        )
        provider._jobs["job-delnf"] = state

        cc = provider._compute_client
        dealloc_poller = MagicMock()
        dealloc_poller.result.return_value = None
        cc.virtual_machines.begin_deallocate.return_value = dealloc_poller
        cc.virtual_machines.begin_delete.side_effect = Exception("ResourceNotFound")

        # Should not raise
        await provider._cancel_provider_job("job-delnf")


class TestAzureParseGpuFromVmName:
    """Cover _parse_gpu_from_vm_name heuristic fallback (lines 1177-1180)."""

    def test_parse_gpu_known_family(self):
        from artenic_ai_platform.providers.azure import _parse_gpu_from_vm_name

        gpu_type, gpu_count = _parse_gpu_from_vm_name("Standard_NC6s_v3")
        assert gpu_type == "V100"
        assert gpu_count == 1

    def test_parse_gpu_heuristic_fallback(self):
        """Unknown NC/ND/NV size with 'standard_n' prefix returns ('GPU', 1)."""
        from artenic_ai_platform.providers.azure import _parse_gpu_from_vm_name

        gpu_type, gpu_count = _parse_gpu_from_vm_name("Standard_NC99_future_v9")
        assert gpu_type == "GPU"
        assert gpu_count == 1

    def test_parse_gpu_non_gpu_vm(self):
        """Non-GPU VM returns (None, 0)."""
        from artenic_ai_platform.providers.azure import _parse_gpu_from_vm_name

        gpu_type, gpu_count = _parse_gpu_from_vm_name("Standard_D2s_v3")
        assert gpu_type is None
        assert gpu_count == 0

    def test_parse_gpu_nd_family(self):
        """ND family VM uses heuristic."""
        from artenic_ai_platform.providers.azure import _parse_gpu_from_vm_name

        gpu_type, gpu_count = _parse_gpu_from_vm_name("Standard_ND99_future_v9")
        assert gpu_type == "GPU"
        assert gpu_count == 1

    def test_parse_gpu_nv_family(self):
        """NV family VM uses heuristic."""
        from artenic_ai_platform.providers.azure import _parse_gpu_from_vm_name

        gpu_type, gpu_count = _parse_gpu_from_vm_name("Standard_NV99_future_v9")
        assert gpu_type == "GPU"
        assert gpu_count == 1

    def test_parse_gpu_h100(self):
        """H100 VM."""
        from artenic_ai_platform.providers.azure import _parse_gpu_from_vm_name

        gpu_type, gpu_count = _parse_gpu_from_vm_name("Standard_ND96isr_H100_v5")
        assert gpu_type == "H100"
        assert gpu_count == 8
