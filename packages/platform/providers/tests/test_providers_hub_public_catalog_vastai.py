"""Tests for Vast.ai public catalog fetcher."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from artenic_ai_platform_providers.hub.public_catalog.vastai import VastaiCatalogFetcher


class TestVastaiCatalogCompute:
    async def test_parses_offers(self) -> None:
        fetcher = VastaiCatalogFetcher()
        mock_data = {
            "offers": [
                {
                    "gpu_name": "RTX 4090",
                    "num_gpus": 1,
                    "cpu_cores_effective": 8,
                    "cpu_ram": 32768,
                    "disk_space": 500,
                    "dph_total": 0.45,
                    "geolocation": "US",
                },
                {
                    "gpu_name": "A100",
                    "num_gpus": 4,
                    "cpu_cores_effective": 0,
                    "cpu_cores": 32,
                    "cpu_ram": 131072,
                    "disk_space": 2000,
                    "dph_total": 3.20,
                    "geolocation": "EU",
                },
            ],
        }
        with patch.object(fetcher, "_get", return_value=mock_data):
            result = await fetcher.fetch_compute_catalog()

        assert len(result) == 2
        assert result[0].name == "RTX 4090 x1"
        assert result[0].gpu_type == "RTX 4090"
        assert result[0].gpu_count == 1
        assert result[0].vcpus == 8
        assert result[0].memory_gb == 32.0
        assert result[0].price_per_hour == 0.45
        assert result[0].currency == "USD"
        assert result[0].region == "US"
        assert result[0].category == "gpu"

        assert result[1].name == "A100 x4"
        assert result[1].gpu_count == 4
        assert result[1].vcpus == 32

    async def test_empty_offers(self) -> None:
        fetcher = VastaiCatalogFetcher()
        with patch.object(fetcher, "_get", return_value={}):
            result = await fetcher.fetch_compute_catalog()
        assert result == []

    async def test_no_price(self) -> None:
        fetcher = VastaiCatalogFetcher()
        mock_data = {
            "offers": [
                {
                    "gpu_name": "T4",
                    "num_gpus": 1,
                    "cpu_cores_effective": 4,
                    "cpu_ram": 16384,
                    "disk_space": 100,
                    "geolocation": "US",
                },
            ],
        }
        with patch.object(fetcher, "_get", return_value=mock_data):
            result = await fetcher.fetch_compute_catalog()
        assert result[0].price_per_hour is None


class TestVastaiCatalogStorage:
    async def test_always_empty(self) -> None:
        fetcher = VastaiCatalogFetcher()
        result = await fetcher.fetch_storage_catalog()
        assert result == []


class TestVastaiGetRequiresHttpx:
    async def test_raises_import_error(self) -> None:
        fetcher = VastaiCatalogFetcher()
        with (
            patch("artenic_ai_platform_providers.hub.public_catalog.vastai.httpx", None),
            pytest.raises(ImportError, match="httpx"),
        ):
            await fetcher._get()

    async def test_get_success(self) -> None:
        fetcher = VastaiCatalogFetcher()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"offers": []}
        with patch("artenic_ai_platform_providers.hub.public_catalog.vastai.httpx") as mock_httpx:
            mock_httpx.get.return_value = mock_resp
            result = await fetcher._get()
        assert result == {"offers": []}
        mock_resp.raise_for_status.assert_called_once()
