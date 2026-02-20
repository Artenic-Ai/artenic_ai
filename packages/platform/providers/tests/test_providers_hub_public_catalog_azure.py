"""Tests for Azure public catalog fetcher."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from artenic_ai_platform_providers.hub.public_catalog.azure import (
    AzureCatalogFetcher,
    _category_from_sku,
    _extract_vcpu_hint,
    _parse_gpu_from_sku,
)


class TestParseGpuFromSku:
    def test_nc_series(self) -> None:
        assert _parse_gpu_from_sku("Standard_NC6") == ("T4", 1)

    def test_nd_series(self) -> None:
        gpu, count = _parse_gpu_from_sku("Standard_ND40rs_v2")
        assert gpu == "A100"
        assert count >= 1

    def test_no_gpu(self) -> None:
        assert _parse_gpu_from_sku("Standard_D4s_v5") == (None, 0)


class TestCategoryFromSku:
    def test_gpu(self) -> None:
        assert _category_from_sku("Standard_NC6") == "gpu"

    def test_general(self) -> None:
        assert _category_from_sku("Standard_D4s_v5") == "general"

    def test_memory(self) -> None:
        assert _category_from_sku("Standard_E8s_v5") == "memory"

    def test_compute(self) -> None:
        assert _category_from_sku("Standard_F4s_v2") == "compute"

    def test_storage(self) -> None:
        assert _category_from_sku("Standard_L8s_v2") == "storage"

    def test_unknown(self) -> None:
        assert _category_from_sku("Custom_X1") == "general"


class TestExtractVcpuHint:
    def test_standard_d4(self) -> None:
        assert _extract_vcpu_hint("Standard_D4s_v5") == 4

    def test_no_match(self) -> None:
        assert _extract_vcpu_hint("Unknown") == 0


class TestAzureCatalogCompute:
    async def test_parses_items(self) -> None:
        fetcher = AzureCatalogFetcher()
        mock_data = {
            "Items": [
                {
                    "armSkuName": "Standard_D4s_v5",
                    "retailPrice": 0.192,
                    "armRegionName": "westeurope",
                },
                {
                    "armSkuName": "Standard_NC6",
                    "retailPrice": 0.90,
                    "armRegionName": "westeurope",
                },
            ],
            "NextPageLink": None,
        }
        with patch.object(fetcher, "_fetch_all_pages", return_value=mock_data["Items"]):
            result = await fetcher.fetch_compute_catalog()

        assert len(result) == 2
        d4 = next(r for r in result if r.name == "Standard_D4s_v5")
        assert d4.price_per_hour == 0.192
        assert d4.currency == "USD"
        assert d4.gpu_count == 0
        assert d4.vcpus == 4

        nc6 = next(r for r in result if r.name == "Standard_NC6")
        assert nc6.gpu_type == "T4"
        assert nc6.gpu_count >= 1
        assert nc6.category == "gpu"

    async def test_deduplicates(self) -> None:
        fetcher = AzureCatalogFetcher()
        items = [
            {"armSkuName": "Standard_D4s_v5", "retailPrice": 0.192, "armRegionName": "west"},
            {"armSkuName": "Standard_D4s_v5", "retailPrice": 0.192, "armRegionName": "east"},
        ]
        with patch.object(fetcher, "_fetch_all_pages", return_value=items):
            result = await fetcher.fetch_compute_catalog()
        assert len(result) == 1

    async def test_empty_sku(self) -> None:
        fetcher = AzureCatalogFetcher()
        with patch.object(
            fetcher,
            "_fetch_all_pages",
            return_value=[{"armSkuName": "", "retailPrice": 0}],
        ):
            result = await fetcher.fetch_compute_catalog()
        assert result == []


class TestAzureCatalogStorage:
    async def test_returns_empty(self) -> None:
        fetcher = AzureCatalogFetcher()
        result = await fetcher.fetch_storage_catalog()
        assert result == []


class TestAzurePagination:
    async def test_follows_pages(self) -> None:
        fetcher = AzureCatalogFetcher()
        page1 = {
            "Items": [{"armSkuName": "D1", "retailPrice": 0.1, "armRegionName": "west"}],
            "NextPageLink": "https://prices.azure.com/page2",
        }
        page2 = {
            "Items": [{"armSkuName": "D2", "retailPrice": 0.2, "armRegionName": "west"}],
            "NextPageLink": None,
        }
        with patch.object(fetcher, "_get", side_effect=[page1, page2]):
            items = await fetcher._fetch_all_pages()
        assert len(items) == 2

    async def test_max_pages_limit(self) -> None:
        fetcher = AzureCatalogFetcher()
        page = {
            "Items": [{"armSkuName": "D1", "retailPrice": 0.1, "armRegionName": "w"}],
            "NextPageLink": "https://prices.azure.com/next",
        }
        with patch.object(fetcher, "_get", return_value=page):
            items = await fetcher._fetch_all_pages()
        # Should stop after _MAX_PAGES (5)
        assert len(items) == 5


class TestAzureGetRequiresHttpx:
    async def test_raises_import_error(self) -> None:
        fetcher = AzureCatalogFetcher()
        with (
            patch("artenic_ai_platform_providers.hub.public_catalog.azure.httpx", None),
            pytest.raises(ImportError, match="httpx"),
        ):
            await fetcher._get("https://example.com")

    async def test_get_success(self) -> None:
        fetcher = AzureCatalogFetcher()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"Items": []}
        with patch("artenic_ai_platform_providers.hub.public_catalog.azure.httpx") as mock_httpx:
            mock_httpx.get.return_value = mock_resp
            result = await fetcher._get("https://example.com")
        assert result == {"Items": []}
        mock_resp.raise_for_status.assert_called_once()
