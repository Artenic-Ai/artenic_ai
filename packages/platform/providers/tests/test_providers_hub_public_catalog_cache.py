"""Tests for CatalogCache TTL cache."""

from __future__ import annotations

from unittest.mock import patch

from artenic_ai_platform_providers.hub.public_catalog.base import CatalogCache, CatalogFetcher


class TestCatalogCache:
    def test_get_miss_returns_none(self) -> None:
        cache = CatalogCache()
        assert cache.get("nonexistent") is None

    def test_set_and_get(self) -> None:
        cache = CatalogCache(ttl_seconds=60.0)
        cache.set("key1", {"data": 42})
        assert cache.get("key1") == {"data": 42}

    def test_expired_entry_returns_none(self) -> None:
        cache = CatalogCache(ttl_seconds=10.0)
        cache.set("key1", "value")
        # Simulate time passing beyond TTL
        with patch("artenic_ai_platform_providers.hub.public_catalog.base.time") as mock_time:
            mock_time.monotonic.return_value = 999_999.0
            assert cache.get("key1") is None

    def test_invalidate(self) -> None:
        cache = CatalogCache()
        cache.set("key1", "value")
        cache.invalidate("key1")
        assert cache.get("key1") is None

    def test_invalidate_missing_key(self) -> None:
        cache = CatalogCache()
        cache.invalidate("nonexistent")  # should not raise

    def test_clear(self) -> None:
        cache = CatalogCache()
        cache.set("k1", 1)
        cache.set("k2", 2)
        cache.clear()
        assert cache.get("k1") is None
        assert cache.get("k2") is None


class TestCatalogFetcherBase:
    def test_supports_live_catalog_default(self) -> None:
        """Default supports_live_catalog() returns True."""
        from artenic_ai_platform_providers.hub.public_catalog.ovh import (
            OvhCatalogFetcher,
        )

        fetcher: CatalogFetcher = OvhCatalogFetcher()
        assert fetcher.supports_live_catalog() is True
