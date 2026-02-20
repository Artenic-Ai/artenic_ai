"""Tests for OVH public catalog fetcher."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from artenic_ai_platform_providers.hub.public_catalog.ovh import (
    OvhCatalogFetcher,
    _category_from_name,
    _parse_gpu,
)


class TestParseGpu:
    def test_no_gpu(self) -> None:
        assert _parse_gpu("b2-30") == (None, 0)

    def test_gpu_prefix(self) -> None:
        assert _parse_gpu("gpu-b2-120") == ("GPU", 1)

    def test_a100(self) -> None:
        assert _parse_gpu("gpu-a100-80g") == ("A100", 1)

    def test_multi_gpu(self) -> None:
        assert _parse_gpu("gpu-a100-80g-x4") == ("A100", 4)

    def test_h100(self) -> None:
        assert _parse_gpu("gpu-h100-80g-x8") == ("H100", 8)


class TestCategoryFromName:
    def test_gpu(self) -> None:
        assert _category_from_name("gpu-a100-80g") == "gpu"

    def test_general(self) -> None:
        assert _category_from_name("b2-30") == "general"

    def test_memory(self) -> None:
        assert _category_from_name("r2-60") == "memory"

    def test_compute(self) -> None:
        assert _category_from_name("c2-30") == "compute"

    def test_storage(self) -> None:
        assert _category_from_name("i1-60") == "storage"


class TestOvhCatalogFetcherCompute:
    async def test_parses_flavors(self) -> None:
        fetcher = OvhCatalogFetcher()
        mock_data = {
            "flavorHourly": [
                {
                    "flavorName": "b2-30",
                    "vcpus": 8,
                    "ram": 30720,
                    "disk": 200,
                    "price": {"value": 0.1167},
                },
                {
                    "flavorName": "gpu-a100-80g",
                    "vcpus": 12,
                    "ram": 122880,
                    "disk": 400,
                    "price": {"value": 2.50},
                },
            ],
        }
        with patch.object(fetcher, "_get", return_value=mock_data):
            result = await fetcher.fetch_compute_catalog()

        assert len(result) == 2
        assert result[0].name == "b2-30"
        assert result[0].vcpus == 8
        assert result[0].memory_gb == 30.0
        assert result[0].price_per_hour == 0.1167
        assert result[0].gpu_count == 0
        assert result[0].category == "general"

        assert result[1].name == "gpu-a100-80g"
        assert result[1].gpu_type == "A100"
        assert result[1].gpu_count == 1
        assert result[1].price_per_hour == 2.50
        assert result[1].category == "gpu"

    async def test_empty_flavors(self) -> None:
        fetcher = OvhCatalogFetcher()
        with patch.object(fetcher, "_get", return_value={}):
            result = await fetcher.fetch_compute_catalog()
        assert result == []

    async def test_skips_empty_name(self) -> None:
        fetcher = OvhCatalogFetcher()
        with patch.object(
            fetcher,
            "_get",
            return_value={"flavorHourly": [{"flavorName": ""}]},
        ):
            result = await fetcher.fetch_compute_catalog()
        assert result == []

    async def test_price_none(self) -> None:
        fetcher = OvhCatalogFetcher()
        with patch.object(
            fetcher,
            "_get",
            return_value={"flavorHourly": [{"flavorName": "b2-30", "price": {}}]},
        ):
            result = await fetcher.fetch_compute_catalog()
        assert result[0].price_per_hour is None

    async def test_price_not_dict(self) -> None:
        fetcher = OvhCatalogFetcher()
        with patch.object(
            fetcher,
            "_get",
            return_value={"flavorHourly": [{"flavorName": "b2-30", "price": "bad"}]},
        ):
            result = await fetcher.fetch_compute_catalog()
        assert result[0].price_per_hour is None


class TestOvhCatalogFetcherStorage:
    async def test_parses_storage(self) -> None:
        fetcher = OvhCatalogFetcher()
        mock_data = {
            "objectStorage": [
                {"name": "Standard", "price": {"value": 0.01}, "region": "GRA"},
            ],
        }
        with patch.object(fetcher, "_get", return_value=mock_data):
            result = await fetcher.fetch_storage_catalog()
        assert len(result) == 1
        assert result[0].name == "Standard"
        assert result[0].price_per_gb_month == 0.01
        assert result[0].region == "GRA"

    async def test_empty_storage(self) -> None:
        fetcher = OvhCatalogFetcher()
        with patch.object(fetcher, "_get", return_value={}):
            result = await fetcher.fetch_storage_catalog()
        assert result == []


class TestOvhGetRequiresHttpx:
    async def test_raises_import_error(self) -> None:
        fetcher = OvhCatalogFetcher()
        with (
            patch("artenic_ai_platform_providers.hub.public_catalog.ovh.httpx", None),
            pytest.raises(ImportError, match="httpx"),
        ):
            await fetcher._get()

    async def test_get_success(self) -> None:
        fetcher = OvhCatalogFetcher()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"flavorHourly": []}
        with patch("artenic_ai_platform_providers.hub.public_catalog.ovh.httpx") as mock_httpx:
            mock_httpx.get.return_value = mock_resp
            result = await fetcher._get()
        assert result == {"flavorHourly": []}
        mock_resp.raise_for_status.assert_called_once()
