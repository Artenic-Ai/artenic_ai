"""Tests for Infomaniak public catalog fetcher (static data)."""

from __future__ import annotations

from artenic_ai_platform_providers.hub.public_catalog.infomaniak import (
    InfomaniakCatalogFetcher,
)


class TestInfomaniakCatalogCompute:
    async def test_returns_static_flavors(self) -> None:
        fetcher = InfomaniakCatalogFetcher()
        result = await fetcher.fetch_compute_catalog()
        assert len(result) >= 5
        assert result[0].provider_id == "infomaniak"
        assert result[0].currency == "EUR"
        assert result[0].price_per_hour is not None
        assert result[0].vcpus > 0

    async def test_category(self) -> None:
        fetcher = InfomaniakCatalogFetcher()
        result = await fetcher.fetch_compute_catalog()
        categories = {r.category for r in result}
        assert "general" in categories


class TestInfomaniakCatalogStorage:
    async def test_returns_static_storage(self) -> None:
        fetcher = InfomaniakCatalogFetcher()
        result = await fetcher.fetch_storage_catalog()
        assert len(result) >= 1
        assert result[0].provider_id == "infomaniak"
        assert result[0].price_per_gb_month is not None


class TestInfomaniakSupportsLive:
    def test_static_only(self) -> None:
        fetcher = InfomaniakCatalogFetcher()
        assert fetcher.supports_live_catalog() is False
