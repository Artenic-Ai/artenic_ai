"""Tests for the Scaleway connector."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from artenic_ai_platform.providers_hub.connectors.base import ConnectorContext
from artenic_ai_platform.providers_hub.connectors.scaleway import (
    ScalewayConnector,
    _require_httpx,
)

CTX = ConnectorContext(
    credentials={
        "access_key": "SCWXXXXXXXXXXX",
        "secret_key": "secret-key-value",
        "project_id": "proj-123",
    },
    config={"zone": "fr-par-1"},
)


# ======================================================================
# _require_httpx
# ======================================================================


class TestRequireHttpx:
    def test_raises_import_error(self) -> None:
        with (
            patch(
                "artenic_ai_platform.providers_hub.connectors.scaleway._HAS_HTTPX",
                False,
            ),
            pytest.raises(ImportError, match="httpx"),
        ):
            _require_httpx()


# ======================================================================
# test_connection
# ======================================================================


class TestScalewayTestConnection:
    async def test_success(self) -> None:
        connector = ScalewayConnector(provider_id="scaleway")
        with patch.object(
            connector,
            "_get",
            return_value={"servers": []},
        ):
            result = await connector.test_connection(CTX)
        assert result.success is True
        assert "Connected" in result.message
        assert result.latency_ms is not None

    async def test_failure(self) -> None:
        connector = ScalewayConnector(provider_id="scaleway")
        with patch.object(
            connector,
            "_get",
            side_effect=Exception("auth error"),
        ):
            result = await connector.test_connection(CTX)
        assert result.success is False
        assert "auth error" in result.message


# ======================================================================
# list_storage_options
# ======================================================================


class TestScalewayListStorage:
    async def test_returns_buckets(self) -> None:
        connector = ScalewayConnector(provider_id="scaleway")
        with patch.object(
            connector,
            "_get",
            return_value={
                "buckets": [
                    {"name": "my-bucket-1"},
                    {"name": "my-bucket-2"},
                ]
            },
        ):
            result = await connector.list_storage_options(CTX)
        assert len(result) == 2
        assert result[0].name == "my-bucket-1"
        assert result[0].provider_id == "scaleway"
        assert result[0].region == "fr-par"

    async def test_empty(self) -> None:
        connector = ScalewayConnector(provider_id="scaleway")
        with patch.object(connector, "_get", return_value={"buckets": []}):
            result = await connector.list_storage_options(CTX)
        assert result == []

    async def test_error_returns_empty(self) -> None:
        connector = ScalewayConnector(provider_id="scaleway")
        with patch.object(connector, "_get", side_effect=Exception("boom")):
            result = await connector.list_storage_options(CTX)
        assert result == []


# ======================================================================
# list_compute_instances
# ======================================================================


class TestScalewayListCompute:
    async def test_returns_servers(self) -> None:
        connector = ScalewayConnector(provider_id="scaleway")
        with patch.object(
            connector,
            "_get",
            return_value={
                "servers": {
                    "DEV1-S": {
                        "ncpus": 2,
                        "ram": 2147483648,
                        "gpu": 0,
                    },
                    "GPU-3070-S": {
                        "ncpus": 8,
                        "ram": 17179869184,
                        "gpu": 1,
                        "arch": "ampere",
                    },
                }
            },
        ):
            result = await connector.list_compute_instances(CTX)
        assert len(result) == 2

        dev = next(i for i in result if i.name == "DEV1-S")
        assert dev.vcpus == 2
        assert dev.memory_gb == 2.0
        assert dev.gpu_count == 0

        gpu = next(i for i in result if i.name == "GPU-3070-S")
        assert gpu.gpu_count == 1
        assert gpu.gpu_type == "AMPERE"

    async def test_gpu_only_filter(self) -> None:
        connector = ScalewayConnector(provider_id="scaleway")
        with patch.object(
            connector,
            "_get",
            return_value={
                "servers": {
                    "DEV1-S": {"ncpus": 2, "ram": 2147483648, "gpu": 0},
                    "GPU-3070-S": {"ncpus": 8, "ram": 17179869184, "gpu": 1, "arch": "ampere"},
                }
            },
        ):
            result = await connector.list_compute_instances(CTX, gpu_only=True)
        assert len(result) == 1
        assert result[0].gpu_count > 0

    async def test_region_override(self) -> None:
        connector = ScalewayConnector(provider_id="scaleway")
        with patch.object(
            connector,
            "_get",
            return_value={"servers": {"DEV1-S": {"ncpus": 2, "ram": 2147483648, "gpu": 0}}},
        ):
            result = await connector.list_compute_instances(CTX, region="nl-ams-1")
        assert result[0].region == "nl-ams-1"

    async def test_error_returns_empty(self) -> None:
        connector = ScalewayConnector(provider_id="scaleway")
        with patch.object(connector, "_get", side_effect=Exception("boom")):
            result = await connector.list_compute_instances(CTX)
        assert result == []


# ======================================================================
# list_regions
# ======================================================================


class TestScalewayListRegions:
    async def test_returns_static_zones(self) -> None:
        connector = ScalewayConnector(provider_id="scaleway")
        result = await connector.list_regions(CTX)
        assert len(result) >= 6
        ids = [r.id for r in result]
        assert "fr-par-1" in ids
        assert "nl-ams-1" in ids


# ======================================================================
# Helper methods
# ======================================================================


class TestScalewayHelpers:
    def test_headers(self) -> None:
        connector = ScalewayConnector()
        headers = connector._headers(CTX)
        assert headers["X-Auth-Token"] == "secret-key-value"

    def test_zone_default(self) -> None:
        connector = ScalewayConnector()
        ctx = ConnectorContext(credentials={}, config={})
        assert connector._zone(ctx) == "fr-par-1"

    def test_region_derived(self) -> None:
        connector = ScalewayConnector()
        assert connector._region(CTX) == "fr-par"

    def test_region_fallback(self) -> None:
        connector = ScalewayConnector()
        ctx = ConnectorContext(credentials={}, config={"zone": "singlepart"})
        assert connector._region(ctx) == "singlepart"


# ======================================================================
# _get requires httpx
# ======================================================================


class TestGetRequiresHttpx:
    async def test_get_raises_import_error(self) -> None:
        connector = ScalewayConnector()
        with (
            patch(
                "artenic_ai_platform.providers_hub.connectors.scaleway._HAS_HTTPX",
                False,
            ),
            pytest.raises(ImportError, match="httpx"),
        ):
            await connector._get("https://example.com", {})

    async def test_get_success(self) -> None:
        """Exercise the real _get body with mocked httpx.get."""
        connector = ScalewayConnector()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": "ok"}
        with patch("artenic_ai_platform.providers_hub.connectors.scaleway.httpx") as mock_httpx:
            mock_httpx.get.return_value = mock_resp
            result = await connector._get("https://api.example.com/test", {"X-Auth": "key"})
        assert result == {"data": "ok"}
        mock_resp.raise_for_status.assert_called_once()
