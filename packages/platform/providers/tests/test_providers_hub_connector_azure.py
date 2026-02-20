"""Tests for the Azure connector."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from artenic_ai_platform_providers.hub.connectors.azure import (
    AzureConnector,
    _detect_gpu,
    _require_azure,
)
from artenic_ai_platform_providers.hub.connectors.base import ConnectorContext

CTX = ConnectorContext(
    credentials={
        "subscription_id": "sub-123",
        "tenant_id": "tenant-456",
        "client_id": "client-789",
        "client_secret": "secret-abc",
    },
    config={"region": "westeurope"},
)


# ======================================================================
# _require_azure
# ======================================================================


class TestRequireAzure:
    def test_raises_import_error(self) -> None:
        with pytest.raises(ImportError, match="Azure SDK"):
            _require_azure()


# ======================================================================
# _detect_gpu
# ======================================================================


class TestDetectGpu:
    def test_nc_series(self) -> None:
        gpu_type, count = _detect_gpu("Standard_NC6")
        assert gpu_type == "T4"
        assert count == 1

    def test_nd_series(self) -> None:
        gpu_type, count = _detect_gpu("Standard_ND40rs_v2")
        assert gpu_type == "A100"
        assert count == 1

    def test_no_gpu(self) -> None:
        gpu_type, count = _detect_gpu("Standard_D4s_v5")
        assert gpu_type is None
        assert count == 0


# ======================================================================
# test_connection
# ======================================================================


def _mock_vm_size(
    name: str = "Standard_D2s_v5",
    number_of_cores: int = 2,
    memory_in_mb: int = 8192,
    resource_disk_size_in_mb: int = 0,
) -> MagicMock:
    sz = MagicMock()
    sz.name = name
    sz.number_of_cores = number_of_cores
    sz.memory_in_mb = memory_in_mb
    sz.resource_disk_size_in_mb = resource_disk_size_in_mb
    return sz


def _mock_location(name: str, display_name: str = "") -> MagicMock:
    loc = MagicMock()
    loc.name = name
    loc.display_name = display_name
    return loc


class TestAzureTestConnection:
    async def test_success(self) -> None:
        connector = AzureConnector(provider_id="azure")
        mock_cred = MagicMock()
        mock_compute_client = MagicMock()
        mock_compute_client.virtual_machine_sizes.list.return_value = [
            _mock_vm_size(),
            _mock_vm_size("Standard_D4s_v5", 4, 16384),
        ]

        with (
            patch.object(connector, "_credential", return_value=mock_cred),
            patch(
                "artenic_ai_platform_providers.hub.connectors.azure.ComputeManagementClient",
                return_value=mock_compute_client,
            ),
        ):
            result = await connector.test_connection(CTX)

        assert result.success is True
        assert "2 VM sizes" in result.message
        assert result.latency_ms is not None

    async def test_failure(self) -> None:
        connector = AzureConnector(provider_id="azure")
        with patch.object(
            connector,
            "_credential",
            side_effect=Exception("auth error"),
        ):
            result = await connector.test_connection(CTX)
        assert result.success is False
        assert "auth error" in result.message


# ======================================================================
# list_storage_options
# ======================================================================


class TestAzureListStorage:
    async def test_no_storage_url_returns_empty(self) -> None:
        """Without storage_account_url config, returns empty."""
        connector = AzureConnector(provider_id="azure")
        mock_cred = MagicMock()
        with patch.object(connector, "_credential", return_value=mock_cred):
            result = await connector.list_storage_options(CTX)
        assert result == []

    async def test_returns_containers(self) -> None:
        connector = AzureConnector(provider_id="azure")
        mock_cred = MagicMock()
        mock_blob_client = MagicMock()
        mock_blob_client.list_containers.return_value = [
            {"name": "container-a"},
            {"name": "container-b"},
        ]

        ctx_with_storage = ConnectorContext(
            credentials=CTX.credentials,
            config={**CTX.config, "storage_account_url": "https://myaccount.blob.core.windows.net"},
        )

        with (
            patch.object(connector, "_credential", return_value=mock_cred),
            patch(
                "artenic_ai_platform_providers.hub.connectors.azure.BlobServiceClient",
                return_value=mock_blob_client,
            ),
        ):
            result = await connector.list_storage_options(ctx_with_storage)

        assert len(result) == 2
        assert result[0].name == "container-a"
        assert result[0].type == "blob_storage"

    async def test_api_error_returns_empty(self) -> None:
        """Credential OK but BlobServiceClient raises."""
        connector = AzureConnector(provider_id="azure")
        mock_cred = MagicMock()

        ctx_with_storage = ConnectorContext(
            credentials=CTX.credentials,
            config={**CTX.config, "storage_account_url": "https://myaccount.blob.core.windows.net"},
        )

        with (
            patch.object(connector, "_credential", return_value=mock_cred),
            patch(
                "artenic_ai_platform_providers.hub.connectors.azure.BlobServiceClient",
                side_effect=Exception("boom"),
            ),
        ):
            result = await connector.list_storage_options(ctx_with_storage)
        assert result == []

    async def test_credential_error_returns_empty(self) -> None:
        """_credential raises → returns empty."""
        connector = AzureConnector(provider_id="azure")
        ctx_with_storage = ConnectorContext(
            credentials=CTX.credentials,
            config={**CTX.config, "storage_account_url": "https://myaccount.blob.core.windows.net"},
        )
        with patch.object(connector, "_credential", side_effect=Exception("cred error")):
            result = await connector.list_storage_options(ctx_with_storage)
        assert result == []


# ======================================================================
# list_compute_instances
# ======================================================================


class TestAzureListCompute:
    async def test_returns_vm_sizes(self) -> None:
        connector = AzureConnector(provider_id="azure")
        mock_cred = MagicMock()
        mock_compute_client = MagicMock()
        mock_compute_client.virtual_machine_sizes.list.return_value = [
            _mock_vm_size("Standard_D2s_v5", 2, 8192, 0),
            _mock_vm_size("Standard_NC6", 6, 57344, 340 * 1024),
        ]

        with (
            patch.object(connector, "_credential", return_value=mock_cred),
            patch(
                "artenic_ai_platform_providers.hub.connectors.azure.ComputeManagementClient",
                return_value=mock_compute_client,
            ),
        ):
            result = await connector.list_compute_instances(CTX)

        assert len(result) == 2
        d2 = next(i for i in result if i.name == "Standard_D2s_v5")
        assert d2.vcpus == 2
        assert d2.gpu_count == 0

        nc = next(i for i in result if i.name == "Standard_NC6")
        assert nc.gpu_type == "T4"
        assert nc.gpu_count == 1

    async def test_gpu_only_filter(self) -> None:
        connector = AzureConnector(provider_id="azure")
        mock_cred = MagicMock()
        mock_compute_client = MagicMock()
        mock_compute_client.virtual_machine_sizes.list.return_value = [
            _mock_vm_size("Standard_D2s_v5", 2, 8192),
            _mock_vm_size("Standard_NC6", 6, 57344),
        ]

        with (
            patch.object(connector, "_credential", return_value=mock_cred),
            patch(
                "artenic_ai_platform_providers.hub.connectors.azure.ComputeManagementClient",
                return_value=mock_compute_client,
            ),
        ):
            result = await connector.list_compute_instances(CTX, gpu_only=True)

        assert len(result) == 1
        assert result[0].gpu_count > 0

    async def test_region_override(self) -> None:
        connector = AzureConnector(provider_id="azure")
        mock_cred = MagicMock()
        mock_compute_client = MagicMock()
        mock_compute_client.virtual_machine_sizes.list.return_value = [
            _mock_vm_size(),
        ]

        with (
            patch.object(connector, "_credential", return_value=mock_cred),
            patch(
                "artenic_ai_platform_providers.hub.connectors.azure.ComputeManagementClient",
                return_value=mock_compute_client,
            ),
        ):
            result = await connector.list_compute_instances(CTX, region="eastus")

        assert result[0].region == "eastus"

    async def test_error_returns_empty(self) -> None:
        connector = AzureConnector(provider_id="azure")
        with patch.object(connector, "_credential", side_effect=Exception("boom")):
            result = await connector.list_compute_instances(CTX)
        assert result == []

    async def test_api_error_returns_empty(self) -> None:
        """Credential OK but ComputeManagementClient raises."""
        connector = AzureConnector(provider_id="azure")
        mock_cred = MagicMock()
        with (
            patch.object(connector, "_credential", return_value=mock_cred),
            patch(
                "artenic_ai_platform_providers.hub.connectors.azure.ComputeManagementClient",
                side_effect=Exception("API error"),
            ),
        ):
            result = await connector.list_compute_instances(CTX)
        assert result == []


# ======================================================================
# list_regions
# ======================================================================


class TestAzureListRegions:
    async def test_returns_locations(self) -> None:
        connector = AzureConnector(provider_id="azure")
        mock_cred = MagicMock()
        mock_sub_client = MagicMock()
        mock_sub_client.subscriptions.list_locations.return_value = [
            _mock_location("westeurope", "West Europe"),
            _mock_location("eastus", "East US"),
        ]

        with (
            patch.object(connector, "_credential", return_value=mock_cred),
            patch(
                "artenic_ai_platform_providers.hub.connectors.azure.SubscriptionClient",
                return_value=mock_sub_client,
            ),
        ):
            result = await connector.list_regions(CTX)

        assert len(result) == 2
        assert result[0].id == "westeurope"
        assert result[0].name == "West Europe"

    async def test_error_returns_empty(self) -> None:
        connector = AzureConnector(provider_id="azure")
        with patch.object(connector, "_credential", side_effect=Exception("boom")):
            result = await connector.list_regions(CTX)
        assert result == []

    async def test_api_error_returns_empty(self) -> None:
        """Credential OK but SubscriptionClient raises."""
        connector = AzureConnector(provider_id="azure")
        mock_cred = MagicMock()
        with (
            patch.object(connector, "_credential", return_value=mock_cred),
            patch(
                "artenic_ai_platform_providers.hub.connectors.azure.SubscriptionClient",
                side_effect=Exception("API error"),
            ),
        ):
            result = await connector.list_regions(CTX)
        assert result == []


# ======================================================================
# Helper methods
# ======================================================================


class TestAzureHelpers:
    def test_subscription_id(self) -> None:
        connector = AzureConnector()
        assert connector._subscription_id(CTX) == "sub-123"

    def test_region_default(self) -> None:
        connector = AzureConnector()
        ctx = ConnectorContext(credentials={}, config={})
        assert connector._region(ctx) == "westeurope"


# ======================================================================
# _credential requires azure packages
# ======================================================================


class TestCredentialRequiresAzure:
    def test_raises_import_error(self) -> None:
        connector = AzureConnector()
        with pytest.raises(ImportError, match="Azure SDK"):
            connector._credential(CTX)
