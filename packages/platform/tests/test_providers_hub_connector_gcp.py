"""Tests for the GCP connector."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from artenic_ai_platform.providers_hub.connectors.base import ConnectorContext
from artenic_ai_platform.providers_hub.connectors.gcp import (
    _GPU_TYPES,
    GcpConnector,
    _require_gcp,
)

_SERVICE_ACCOUNT_JSON = '{"type":"service_account","project_id":"my-proj","private_key_id":"k1"}'

CTX = ConnectorContext(
    credentials={
        "project_id": "my-proj",
        "credentials_json": _SERVICE_ACCOUNT_JSON,
    },
    config={"zone": "europe-west1-b"},
)


# ======================================================================
# _require_gcp
# ======================================================================


class TestRequireGcp:
    def test_raises_import_error(self) -> None:
        with pytest.raises(ImportError, match="google-cloud"):
            _require_gcp()


# ======================================================================
# test_connection
# ======================================================================


def _mock_machine_type(
    name: str = "n1-standard-1",
    guest_cpus: int = 1,
    memory_mb: int = 3840,
    accelerators: list[Any] | None = None,
) -> MagicMock:
    mt = MagicMock()
    mt.name = name
    mt.guest_cpus = guest_cpus
    mt.memory_mb = memory_mb
    mt.accelerators = accelerators or []
    return mt


def _mock_region(name: str, description: str = "") -> MagicMock:
    r = MagicMock()
    r.name = name
    r.description = description
    return r


def _mock_bucket(name: str, location: str = "") -> MagicMock:
    b = MagicMock()
    b.name = name
    b.location = location
    return b


class TestGcpTestConnection:
    async def test_success(self) -> None:
        connector = GcpConnector(provider_id="gcp")
        mock_creds = MagicMock()
        mock_client = MagicMock()
        mock_client.list.return_value = [_mock_machine_type()]

        with (
            patch.object(connector, "_credentials", return_value=mock_creds),
            patch("artenic_ai_platform.providers_hub.connectors.gcp.compute_v1") as mock_compute,
        ):
            mock_compute.MachineTypesClient.return_value = mock_client
            mock_compute.ListMachineTypesRequest.return_value = MagicMock()
            result = await connector.test_connection(CTX)

        assert result.success is True
        assert "1 machine type" in result.message
        assert result.latency_ms is not None

    async def test_failure(self) -> None:
        connector = GcpConnector(provider_id="gcp")
        with patch.object(
            connector,
            "_credentials",
            side_effect=Exception("invalid json"),
        ):
            result = await connector.test_connection(CTX)
        assert result.success is False
        assert "invalid json" in result.message


# ======================================================================
# list_storage_options
# ======================================================================


class TestGcpListStorage:
    async def test_returns_buckets(self) -> None:
        connector = GcpConnector(provider_id="gcp")
        mock_creds = MagicMock()
        mock_gcs_client = MagicMock()
        mock_gcs_client.list_buckets.return_value = [
            _mock_bucket("data-bucket", "EU"),
            _mock_bucket("models-bucket", "US"),
        ]

        with (
            patch.object(connector, "_credentials", return_value=mock_creds),
            patch("artenic_ai_platform.providers_hub.connectors.gcp.gcs") as mock_gcs,
        ):
            mock_gcs.Client.return_value = mock_gcs_client
            result = await connector.list_storage_options(CTX)

        assert len(result) == 2
        assert result[0].name == "data-bucket"
        assert result[0].type == "gcs"
        assert result[0].region == "EU"

    async def test_error_returns_empty(self) -> None:
        connector = GcpConnector(provider_id="gcp")
        with patch.object(connector, "_credentials", side_effect=Exception("boom")):
            result = await connector.list_storage_options(CTX)
        assert result == []

    async def test_api_error_returns_empty(self) -> None:
        """Credentials OK but GCS client raises."""
        connector = GcpConnector(provider_id="gcp")
        mock_creds = MagicMock()
        with (
            patch.object(connector, "_credentials", return_value=mock_creds),
            patch("artenic_ai_platform.providers_hub.connectors.gcp.gcs") as mock_gcs,
        ):
            mock_gcs.Client.return_value.list_buckets.side_effect = Exception("API error")
            result = await connector.list_storage_options(CTX)
        assert result == []


# ======================================================================
# list_compute_instances
# ======================================================================


class TestGcpListCompute:
    async def test_returns_machine_types(self) -> None:
        connector = GcpConnector(provider_id="gcp")
        mock_creds = MagicMock()
        mock_client = MagicMock()

        acc = MagicMock()
        acc.guest_accelerator_count = 1
        acc.guest_accelerator_type = "nvidia-tesla-t4"

        mock_client.list.return_value = [
            _mock_machine_type("n1-standard-4", 4, 15360),
            _mock_machine_type("a2-highgpu-1g", 12, 87040, accelerators=[acc]),
        ]

        with (
            patch.object(connector, "_credentials", return_value=mock_creds),
            patch("artenic_ai_platform.providers_hub.connectors.gcp.compute_v1") as mock_compute,
        ):
            mock_compute.MachineTypesClient.return_value = mock_client
            mock_compute.ListMachineTypesRequest.return_value = MagicMock()
            result = await connector.list_compute_instances(CTX)

        assert len(result) == 2
        n1 = next(i for i in result if i.name == "n1-standard-4")
        assert n1.vcpus == 4
        assert n1.gpu_count == 0

        a2 = next(i for i in result if i.name == "a2-highgpu-1g")
        assert a2.gpu_type == "T4"
        assert a2.gpu_count == 1

    async def test_a2_family_fallback(self) -> None:
        """GPU detected from a2- prefix when no accelerators listed."""
        connector = GcpConnector(provider_id="gcp")
        mock_creds = MagicMock()
        mock_client = MagicMock()
        mock_client.list.return_value = [
            _mock_machine_type("a2-highgpu-1g", 12, 87040),
        ]

        with (
            patch.object(connector, "_credentials", return_value=mock_creds),
            patch("artenic_ai_platform.providers_hub.connectors.gcp.compute_v1") as mock_compute,
        ):
            mock_compute.MachineTypesClient.return_value = mock_client
            mock_compute.ListMachineTypesRequest.return_value = MagicMock()
            result = await connector.list_compute_instances(CTX)

        assert result[0].gpu_type == "A100"
        assert result[0].gpu_count == 1

    async def test_gpu_only_filter(self) -> None:
        connector = GcpConnector(provider_id="gcp")
        mock_creds = MagicMock()
        mock_client = MagicMock()
        mock_client.list.return_value = [
            _mock_machine_type("n1-standard-4", 4, 15360),
            _mock_machine_type("a2-highgpu-1g", 12, 87040),
        ]

        with (
            patch.object(connector, "_credentials", return_value=mock_creds),
            patch("artenic_ai_platform.providers_hub.connectors.gcp.compute_v1") as mock_compute,
        ):
            mock_compute.MachineTypesClient.return_value = mock_client
            mock_compute.ListMachineTypesRequest.return_value = MagicMock()
            result = await connector.list_compute_instances(CTX, gpu_only=True)

        assert len(result) == 1
        assert result[0].gpu_count > 0

    async def test_region_override(self) -> None:
        connector = GcpConnector(provider_id="gcp")
        mock_creds = MagicMock()
        mock_client = MagicMock()
        mock_client.list.return_value = [_mock_machine_type()]

        with (
            patch.object(connector, "_credentials", return_value=mock_creds),
            patch("artenic_ai_platform.providers_hub.connectors.gcp.compute_v1") as mock_compute,
        ):
            mock_compute.MachineTypesClient.return_value = mock_client
            mock_compute.ListMachineTypesRequest.return_value = MagicMock()
            result = await connector.list_compute_instances(CTX, region="us-central1-a")

        assert result[0].region == "us-central1-a"

    async def test_error_returns_empty(self) -> None:
        connector = GcpConnector(provider_id="gcp")
        with patch.object(connector, "_credentials", side_effect=Exception("boom")):
            result = await connector.list_compute_instances(CTX)
        assert result == []

    async def test_api_error_returns_empty(self) -> None:
        """Credentials OK but MachineTypesClient raises."""
        connector = GcpConnector(provider_id="gcp")
        mock_creds = MagicMock()
        with (
            patch.object(connector, "_credentials", return_value=mock_creds),
            patch("artenic_ai_platform.providers_hub.connectors.gcp.compute_v1") as mock_compute,
        ):
            mock_compute.MachineTypesClient.return_value.list.side_effect = Exception("API error")
            mock_compute.ListMachineTypesRequest.return_value = MagicMock()
            result = await connector.list_compute_instances(CTX)
        assert result == []


# ======================================================================
# list_regions
# ======================================================================


class TestGcpListRegions:
    async def test_returns_regions(self) -> None:
        connector = GcpConnector(provider_id="gcp")
        mock_creds = MagicMock()
        mock_client = MagicMock()
        mock_client.list.return_value = [
            _mock_region("europe-west1", "Belgium"),
            _mock_region("us-central1", "Iowa"),
        ]

        with (
            patch.object(connector, "_credentials", return_value=mock_creds),
            patch("artenic_ai_platform.providers_hub.connectors.gcp.compute_v1") as mock_compute,
        ):
            mock_compute.RegionsClient.return_value = mock_client
            mock_compute.ListRegionsRequest.return_value = MagicMock()
            result = await connector.list_regions(CTX)

        assert len(result) == 2
        assert result[0].id == "europe-west1"
        assert result[0].name == "Belgium"

    async def test_error_returns_empty(self) -> None:
        connector = GcpConnector(provider_id="gcp")
        with patch.object(connector, "_credentials", side_effect=Exception("boom")):
            result = await connector.list_regions(CTX)
        assert result == []

    async def test_api_error_returns_empty(self) -> None:
        """Credentials OK but RegionsClient raises."""
        connector = GcpConnector(provider_id="gcp")
        mock_creds = MagicMock()
        with (
            patch.object(connector, "_credentials", return_value=mock_creds),
            patch("artenic_ai_platform.providers_hub.connectors.gcp.compute_v1") as mock_compute,
        ):
            mock_compute.RegionsClient.return_value.list.side_effect = Exception("API error")
            mock_compute.ListRegionsRequest.return_value = MagicMock()
            result = await connector.list_regions(CTX)
        assert result == []


# ======================================================================
# GPU types dict
# ======================================================================


class TestGpuTypes:
    def test_known_types(self) -> None:
        assert "nvidia-tesla-t4" in _GPU_TYPES
        assert _GPU_TYPES["nvidia-tesla-t4"] == "T4"


# ======================================================================
# _credentials requires gcp packages
# ======================================================================


class TestCredentialsRequiresGcp:
    def test_raises_import_error(self) -> None:
        connector = GcpConnector()
        with pytest.raises(ImportError, match="google-cloud"):
            connector._credentials(CTX)


# ======================================================================
# Helper methods
# ======================================================================


class TestGcpHelpers:
    def test_project(self) -> None:
        connector = GcpConnector()
        assert connector._project(CTX) == "my-proj"

    def test_zone_default(self) -> None:
        connector = GcpConnector()
        ctx = ConnectorContext(credentials={}, config={})
        assert connector._zone(ctx) == "europe-west1-b"
