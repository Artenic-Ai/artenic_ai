"""Tests for the public catalog REST API endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from artenic_ai_platform.app import _lifespan, create_app
from artenic_ai_platform_providers.hub.schemas import (
    CatalogComputeFlavor,
    CatalogStorageTier,
)
from artenic_ai_platform_providers.hub.service import _clear_catalog_cache
from artenic_ai_platform.settings import PlatformSettings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from fastapi import FastAPI

BASE = "/api/v1/providers"


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
async def app_with_lifespan(tmp_path: Path) -> AsyncGenerator[FastAPI, None]:
    settings = PlatformSettings(
        database_url="sqlite+aiosqlite://",
        api_key="",
        secret_key="test-secret",
        otel_enabled=False,
        dataset={"storage": {"local_dir": str(tmp_path / "datasets")}},
    )
    app = create_app(settings)
    async with _lifespan(app):
        yield app


@pytest.fixture
async def client(app_with_lifespan: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app_with_lifespan)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        follow_redirects=True,
    ) as c:
        yield c


@pytest.fixture(autouse=True)
def _clear_catalog() -> None:
    _clear_catalog_cache()


# ======================================================================
# Mock helpers
# ======================================================================

_MOCK_COMPUTE = [
    CatalogComputeFlavor(
        provider_id="ovh",
        name="b2-30",
        vcpus=8,
        memory_gb=30.0,
        price_per_hour=0.12,
        category="general",
    ),
]

_MOCK_STORAGE = [
    CatalogStorageTier(
        provider_id="ovh",
        name="Standard",
        price_per_gb_month=0.01,
    ),
]


def _mock_fetcher() -> MagicMock:
    """Build a mock CatalogFetcher with sync supports_live_catalog."""
    m = MagicMock()
    m.fetch_compute_catalog = AsyncMock(return_value=_MOCK_COMPUTE)
    m.fetch_storage_catalog = AsyncMock(return_value=_MOCK_STORAGE)
    m.supports_live_catalog.return_value = True
    return m


# ======================================================================
# GET /api/v1/providers/{id}/catalog
# ======================================================================


class TestGetProviderCatalog:
    async def test_returns_catalog(self, client: AsyncClient) -> None:
        with patch(
            "artenic_ai_platform_providers.hub.service._get_catalog_fetcher",
            return_value=_mock_fetcher(),
        ):
            resp = await client.get(f"{BASE}/ovh/catalog")
        assert resp.status_code == 200
        data = resp.json()
        assert data["provider_id"] == "ovh"
        assert len(data["compute"]) == 1
        assert data["compute"][0]["name"] == "b2-30"
        assert len(data["storage"]) == 1

    async def test_unknown_provider_404(self, client: AsyncClient) -> None:
        resp = await client.get(f"{BASE}/nonexistent/catalog")
        assert resp.status_code == 404


# ======================================================================
# GET /api/v1/providers/{id}/catalog/compute
# ======================================================================


class TestGetProviderCatalogCompute:
    async def test_returns_compute(self, client: AsyncClient) -> None:
        with patch(
            "artenic_ai_platform_providers.hub.service._get_catalog_fetcher",
            return_value=_mock_fetcher(),
        ):
            resp = await client.get(f"{BASE}/ovh/catalog/compute")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "b2-30"

    async def test_gpu_only_filters(self, client: AsyncClient) -> None:
        with patch(
            "artenic_ai_platform_providers.hub.service._get_catalog_fetcher",
            return_value=_mock_fetcher(),
        ):
            resp = await client.get(f"{BASE}/ovh/catalog/compute?gpu_only=true")
        assert resp.status_code == 200
        # No GPU flavors in mock → empty
        assert resp.json() == []

    async def test_unknown_provider_404(self, client: AsyncClient) -> None:
        resp = await client.get(f"{BASE}/nonexistent/catalog/compute")
        assert resp.status_code == 404


# ======================================================================
# GET /api/v1/providers/{id}/catalog/storage
# ======================================================================


class TestGetProviderCatalogStorage:
    async def test_returns_storage(self, client: AsyncClient) -> None:
        with patch(
            "artenic_ai_platform_providers.hub.service._get_catalog_fetcher",
            return_value=_mock_fetcher(),
        ):
            resp = await client.get(f"{BASE}/ovh/catalog/storage")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "Standard"

    async def test_unknown_provider_404(self, client: AsyncClient) -> None:
        resp = await client.get(f"{BASE}/nonexistent/catalog/storage")
        assert resp.status_code == 404


# ======================================================================
# GET /api/v1/providers/catalog/compute (aggregate)
# ======================================================================


class TestAllCatalogCompute:
    async def test_returns_aggregate(self, client: AsyncClient) -> None:
        with patch(
            "artenic_ai_platform_providers.hub.service._get_catalog_fetcher",
            return_value=_mock_fetcher(),
        ):
            resp = await client.get(f"{BASE}/catalog/compute")
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    async def test_gpu_only_param(self, client: AsyncClient) -> None:
        with patch(
            "artenic_ai_platform_providers.hub.service._get_catalog_fetcher",
            return_value=_mock_fetcher(),
        ):
            resp = await client.get(f"{BASE}/catalog/compute?gpu_only=true")
        assert resp.status_code == 200


# ======================================================================
# GET /api/v1/providers/catalog/storage (aggregate)
# ======================================================================


class TestAllCatalogStorage:
    async def test_returns_aggregate(self, client: AsyncClient) -> None:
        with patch(
            "artenic_ai_platform_providers.hub.service._get_catalog_fetcher",
            return_value=_mock_fetcher(),
        ):
            resp = await client.get(f"{BASE}/catalog/storage")
        assert resp.status_code == 200
        assert len(resp.json()) >= 1
