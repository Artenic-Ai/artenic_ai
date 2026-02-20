"""Abstract base for public catalog fetchers and in-memory TTL cache."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from artenic_ai_platform_providers.hub.schemas import (
        CatalogComputeFlavor,
        CatalogStorageTier,
    )


# ---------------------------------------------------------------------------
# Fetcher ABC
# ---------------------------------------------------------------------------


class CatalogFetcher(ABC):
    """Abstract base for public catalog fetchers (no auth needed)."""

    @abstractmethod
    async def fetch_compute_catalog(self) -> list[CatalogComputeFlavor]:
        """Fetch public compute flavor/pricing catalog."""

    @abstractmethod
    async def fetch_storage_catalog(self) -> list[CatalogStorageTier]:
        """Fetch public storage tier/pricing catalog."""

    def supports_live_catalog(self) -> bool:
        """Whether this fetcher retrieves live data from a public API."""
        return True


# ---------------------------------------------------------------------------
# In-memory TTL cache
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    data: Any
    expires_at: float


class CatalogCache:
    """Simple in-memory TTL cache for public catalog data."""

    def __init__(self, ttl_seconds: float = 3600.0) -> None:
        self._ttl = ttl_seconds
        self._store: dict[str, _CacheEntry] = {}

    def get(self, key: str) -> Any | None:
        """Return cached value or *None* if missing / expired."""
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.monotonic() > entry.expires_at:
            del self._store[key]
            return None
        return entry.data

    def set(self, key: str, data: Any) -> None:
        """Store *data* under *key* with the configured TTL."""
        self._store[key] = _CacheEntry(
            data=data,
            expires_at=time.monotonic() + self._ttl,
        )

    def invalidate(self, key: str) -> None:
        """Remove a single cache entry."""
        self._store.pop(key, None)

    def clear(self) -> None:
        """Drop all cached entries."""
        self._store.clear()
