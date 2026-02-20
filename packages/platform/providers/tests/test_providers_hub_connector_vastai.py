"""Tests for the Vast.ai connector."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from artenic_ai_platform_providers.hub.connectors.base import ConnectorContext
from artenic_ai_platform_providers.hub.connectors.vastai import (
    VastaiConnector,
    _require_httpx,
)

CTX = ConnectorContext(
    credentials={"api_key": "vast-api-key-123"},
    config={},
)


# ======================================================================
# _require_httpx
# ======================================================================


class TestRequireHttpx:
    def test_raises_import_error(self) -> None:
        with (
            patch(
                "artenic_ai_platform_providers.hub.connectors.vastai._HAS_HTTPX",
                False,
            ),
            pytest.raises(ImportError, match="httpx"),
        ):
            _require_httpx()


# ======================================================================
# test_connection
# ======================================================================


class TestVastaiTestConnection:
    async def test_success(self) -> None:
        connector = VastaiConnector(provider_id="vastai")
        with patch.object(
            connector,
            "_get",
            return_value={"id": 12345, "username": "testuser"},
        ):
            result = await connector.test_connection(CTX)
        assert result.success is True
        assert "Connected" in result.message
        assert result.latency_ms is not None

    async def test_failure(self) -> None:
        connector = VastaiConnector(provider_id="vastai")
        with patch.object(
            connector,
            "_get",
            side_effect=Exception("auth error"),
        ):
            result = await connector.test_connection(CTX)
        assert result.success is False
        assert "auth error" in result.message


# ======================================================================
# list_storage_options (always empty — compute-only)
# ======================================================================


class TestVastaiListStorage:
    async def test_always_empty(self) -> None:
        connector = VastaiConnector(provider_id="vastai")
        result = await connector.list_storage_options(CTX)
        assert result == []


# ======================================================================
# list_compute_instances
# ======================================================================


class TestVastaiListCompute:
    async def test_returns_offers(self) -> None:
        connector = VastaiConnector(provider_id="vastai")
        with patch.object(
            connector,
            "_get",
            return_value={
                "offers": [
                    {
                        "gpu_name": "RTX 4090",
                        "num_gpus": 1,
                        "cpu_cores_effective": 8,
                        "cpu_ram": 32768,
                        "disk_space": 500,
                        "geolocation": "US",
                        "rentable": True,
                    },
                    {
                        "gpu_name": "A100",
                        "num_gpus": 4,
                        "cpu_cores": 32,
                        "cpu_ram": 131072,
                        "disk_space": 2000,
                        "geolocation": "EU",
                        "rentable": True,
                    },
                ]
            },
        ):
            result = await connector.list_compute_instances(CTX)
        assert len(result) == 2

        rtx = result[0]
        assert rtx.name == "RTX 4090 x1"
        assert rtx.gpu_type == "RTX 4090"
        assert rtx.gpu_count == 1
        assert rtx.vcpus == 8
        assert rtx.memory_gb == 32.0
        assert rtx.region == "US"

        a100 = result[1]
        assert a100.name == "A100 x4"
        assert a100.gpu_count == 4
        assert a100.vcpus == 32

    async def test_gpu_only_filter(self) -> None:
        connector = VastaiConnector(provider_id="vastai")
        with patch.object(
            connector,
            "_get",
            return_value={
                "offers": [
                    {
                        "gpu_name": "RTX 4090",
                        "num_gpus": 1,
                        "cpu_cores_effective": 8,
                        "cpu_ram": 32768,
                        "disk_space": 500,
                        "geolocation": "US",
                        "rentable": True,
                    },
                ]
            },
        ):
            result = await connector.list_compute_instances(CTX, gpu_only=True)
        assert len(result) == 1

    async def test_region_override(self) -> None:
        connector = VastaiConnector(provider_id="vastai")
        with patch.object(
            connector,
            "_get",
            return_value={
                "offers": [
                    {
                        "gpu_name": "RTX 4090",
                        "num_gpus": 1,
                        "cpu_cores_effective": 8,
                        "cpu_ram": 32768,
                        "disk_space": 500,
                        "geolocation": "US",
                        "rentable": True,
                    },
                ]
            },
        ):
            result = await connector.list_compute_instances(CTX, region="EU")
        assert result[0].region == "EU"

    async def test_error_returns_empty(self) -> None:
        connector = VastaiConnector(provider_id="vastai")
        with patch.object(connector, "_get", side_effect=Exception("boom")):
            result = await connector.list_compute_instances(CTX)
        assert result == []

    async def test_offer_missing_cpu_cores_effective(self) -> None:
        """Falls back to cpu_cores when cpu_cores_effective is 0."""
        connector = VastaiConnector(provider_id="vastai")
        with patch.object(
            connector,
            "_get",
            return_value={
                "offers": [
                    {
                        "gpu_name": "RTX 3090",
                        "num_gpus": 1,
                        "cpu_cores_effective": 0,
                        "cpu_cores": 16,
                        "cpu_ram": 65536,
                        "disk_space": 1000,
                        "geolocation": "US",
                        "rentable": True,
                    },
                ]
            },
        ):
            result = await connector.list_compute_instances(CTX)
        assert result[0].vcpus == 16

    async def test_offer_no_ram(self) -> None:
        """Handles missing cpu_ram gracefully."""
        connector = VastaiConnector(provider_id="vastai")
        with patch.object(
            connector,
            "_get",
            return_value={
                "offers": [
                    {
                        "gpu_name": "RTX 3090",
                        "num_gpus": 1,
                        "cpu_cores_effective": 4,
                        "disk_space": 0,
                        "geolocation": "",
                        "rentable": False,
                    },
                ]
            },
        ):
            result = await connector.list_compute_instances(CTX)
        assert result[0].memory_gb == 0
        assert result[0].available is False


# ======================================================================
# list_regions (always empty — marketplace)
# ======================================================================


class TestVastaiListRegions:
    async def test_always_empty(self) -> None:
        connector = VastaiConnector(provider_id="vastai")
        result = await connector.list_regions(CTX)
        assert result == []


# ======================================================================
# Helper methods
# ======================================================================


class TestVastaiHelpers:
    def test_headers(self) -> None:
        connector = VastaiConnector()
        headers = connector._headers(CTX)
        assert headers["Authorization"] == "Bearer vast-api-key-123"


# ======================================================================
# _get requires httpx
# ======================================================================


class TestGetRequiresHttpx:
    async def test_get_raises_import_error(self) -> None:
        connector = VastaiConnector()
        with (
            patch(
                "artenic_ai_platform_providers.hub.connectors.vastai._HAS_HTTPX",
                False,
            ),
            pytest.raises(ImportError, match="httpx"),
        ):
            await connector._get("https://example.com", {})

    async def test_get_success(self) -> None:
        """Exercise the real _get body with mocked httpx.get."""
        connector = VastaiConnector()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"id": 1}
        with patch("artenic_ai_platform_providers.hub.connectors.vastai.httpx") as mock_httpx:
            mock_httpx.get.return_value = mock_resp
            result = await connector._get("https://api.example.com/test", {"Auth": "key"})
        assert result == {"id": 1}
        mock_resp.raise_for_status.assert_called_once()
