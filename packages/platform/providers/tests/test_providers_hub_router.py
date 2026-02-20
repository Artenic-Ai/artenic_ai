"""Tests for artenic_ai_platform_providers.hub.router — Provider Hub REST API."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from artenic_ai_platform.app import _lifespan, create_app
from artenic_ai_platform_providers.hub.schemas import (
    ComputeInstance,
    ConnectionTestResult,
    ProviderRegion,
    StorageOption,
)
from artenic_ai_platform.settings import PlatformSettings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from fastapi import FastAPI


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


# ======================================================================
# Helpers
# ======================================================================

BASE = "/api/v1/providers"


def _ovh_credentials() -> dict:
    return {
        "auth_url": "https://auth.cloud.ovh.net/v3",
        "username": "testuser",
        "password": "testpass",
        "project_id": "test-project-id",
    }


def _ovh_config() -> dict:
    return {"region": "GRA11", "user_domain_name": "Default"}


async def _configure_ovh(client: AsyncClient) -> dict:
    """Configure OVH and return the response body."""
    resp = await client.put(
        f"{BASE}/ovh/configure",
        json={"credentials": _ovh_credentials(), "config": _ovh_config()},
    )
    assert resp.status_code == 200
    return resp.json()


def _mock_test_success() -> AsyncMock:
    """Mock connector with successful test_connection."""
    mock = AsyncMock()
    mock.test_connection.return_value = ConnectionTestResult(
        success=True,
        message="Connected — 10 flavors available",
        latency_ms=42.0,
    )
    return mock


def _mock_test_failure() -> AsyncMock:
    """Mock connector with failing test_connection."""
    mock = AsyncMock()
    mock.test_connection.return_value = ConnectionTestResult(
        success=False,
        message="Auth failed",
        latency_ms=100.0,
    )
    return mock


# ======================================================================
# GET /api/v1/providers
# ======================================================================


class TestListProviders:
    async def test_returns_catalog(self, client: AsyncClient) -> None:
        resp = await client.get(BASE)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        ovh = next(p for p in data if p["id"] == "ovh")
        assert ovh["display_name"] == "OVH Public Cloud"
        assert ovh["enabled"] is False
        assert ovh["status"] == "unconfigured"
        assert len(ovh["capabilities"]) == 3

    async def test_reflects_enabled_state(self, client: AsyncClient) -> None:
        await _configure_ovh(client)
        with patch(
            "artenic_ai_platform_providers.hub.service._get_connector",
            return_value=_mock_test_success(),
        ):
            await client.post(f"{BASE}/ovh/enable")

        resp = await client.get(BASE)
        ovh = next(p for p in resp.json() if p["id"] == "ovh")
        assert ovh["enabled"] is True
        assert ovh["status"] == "connected"


# ======================================================================
# GET /api/v1/providers/{id}
# ======================================================================


class TestGetProvider:
    async def test_returns_detail(self, client: AsyncClient) -> None:
        resp = await client.get(f"{BASE}/ovh")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "ovh"
        assert data["connector_type"] == "openstack"
        assert len(data["credential_fields"]) >= 3
        assert len(data["config_fields"]) >= 2
        assert data["has_credentials"] is False

    async def test_unknown_provider_404(self, client: AsyncClient) -> None:
        resp = await client.get(f"{BASE}/nonexistent")
        assert resp.status_code == 404

    async def test_shows_credentials_flag(self, client: AsyncClient) -> None:
        await _configure_ovh(client)
        resp = await client.get(f"{BASE}/ovh")
        assert resp.json()["has_credentials"] is True


# ======================================================================
# PUT /api/v1/providers/{id}/configure
# ======================================================================


class TestConfigureProvider:
    async def test_configure_success(self, client: AsyncClient) -> None:
        data = await _configure_ovh(client)
        assert data["id"] == "ovh"
        assert data["status"] == "configured"
        assert data["has_credentials"] is True
        assert data["config"]["region"] == "GRA11"

    async def test_configure_unknown_404(self, client: AsyncClient) -> None:
        resp = await client.put(
            f"{BASE}/nonexistent/configure",
            json={"credentials": {}, "config": {}},
        )
        assert resp.status_code == 404

    async def test_reconfigure_updates(self, client: AsyncClient) -> None:
        await _configure_ovh(client)
        resp = await client.put(
            f"{BASE}/ovh/configure",
            json={
                "credentials": {**_ovh_credentials(), "username": "newuser"},
                "config": {"region": "SBG5"},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["config"]["region"] == "SBG5"
        assert data["status"] == "configured"

    async def test_empty_credentials_rejected(self, client: AsyncClient) -> None:
        """T7: Empty credentials should be rejected with 400."""
        resp = await client.put(
            f"{BASE}/ovh/configure",
            json={"credentials": {}, "config": {}},
        )
        assert resp.status_code == 400
        assert "Missing required" in resp.json()["error"]["message"]


# ======================================================================
# POST /api/v1/providers/{id}/enable
# ======================================================================


class TestEnableProvider:
    async def test_enable_success(self, client: AsyncClient) -> None:
        await _configure_ovh(client)
        with patch(
            "artenic_ai_platform_providers.hub.service._get_connector",
            return_value=_mock_test_success(),
        ):
            resp = await client.post(f"{BASE}/ovh/enable")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["status"] == "connected"

    async def test_enable_without_config_400(self, client: AsyncClient) -> None:
        resp = await client.post(f"{BASE}/ovh/enable")
        assert resp.status_code == 400

    async def test_enable_connection_failure_400(self, client: AsyncClient) -> None:
        await _configure_ovh(client)
        with patch(
            "artenic_ai_platform_providers.hub.service._get_connector",
            return_value=_mock_test_failure(),
        ):
            resp = await client.post(f"{BASE}/ovh/enable")
        assert resp.status_code == 400
        assert "Auth failed" in resp.json()["error"]["message"]


# ======================================================================
# POST /api/v1/providers/{id}/disable
# ======================================================================


class TestDisableProvider:
    async def test_disable_success(self, client: AsyncClient) -> None:
        await _configure_ovh(client)
        with patch(
            "artenic_ai_platform_providers.hub.service._get_connector",
            return_value=_mock_test_success(),
        ):
            await client.post(f"{BASE}/ovh/enable")

        resp = await client.post(f"{BASE}/ovh/disable")
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False

    async def test_disable_unconfigured_400(self, client: AsyncClient) -> None:
        resp = await client.post(f"{BASE}/ovh/disable")
        assert resp.status_code == 400


# ======================================================================
# POST /api/v1/providers/{id}/test
# ======================================================================


class TestTestProvider:
    async def test_test_success(self, client: AsyncClient) -> None:
        await _configure_ovh(client)
        with patch(
            "artenic_ai_platform_providers.hub.service._get_connector",
            return_value=_mock_test_success(),
        ):
            resp = await client.post(f"{BASE}/ovh/test")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["latency_ms"] is not None

    async def test_test_failure(self, client: AsyncClient) -> None:
        await _configure_ovh(client)
        with patch(
            "artenic_ai_platform_providers.hub.service._get_connector",
            return_value=_mock_test_failure(),
        ):
            resp = await client.post(f"{BASE}/ovh/test")
        assert resp.status_code == 200
        assert resp.json()["success"] is False

    async def test_test_unconfigured_400(self, client: AsyncClient) -> None:
        resp = await client.post(f"{BASE}/ovh/test")
        assert resp.status_code == 400

    async def test_test_updates_status(self, client: AsyncClient) -> None:
        await _configure_ovh(client)
        with patch(
            "artenic_ai_platform_providers.hub.service._get_connector",
            return_value=_mock_test_success(),
        ):
            await client.post(f"{BASE}/ovh/test")

        resp = await client.get(f"{BASE}/ovh")
        assert resp.json()["status"] == "connected"
        assert resp.json()["last_checked_at"] is not None


# ======================================================================
# DELETE /api/v1/providers/{id}
# ======================================================================


class TestDeleteProvider:
    async def test_delete_configured(self, client: AsyncClient) -> None:
        await _configure_ovh(client)
        resp = await client.delete(f"{BASE}/ovh")
        assert resp.status_code == 204

        resp = await client.get(f"{BASE}/ovh")
        assert resp.json()["has_credentials"] is False
        assert resp.json()["status"] == "unconfigured"

    async def test_delete_unconfigured_noop(self, client: AsyncClient) -> None:
        resp = await client.delete(f"{BASE}/ovh")
        assert resp.status_code == 204


# ======================================================================
# GET /api/v1/providers/{id}/storage
# ======================================================================


class TestProviderStorage:
    async def test_storage_active_provider(self, client: AsyncClient) -> None:
        await _configure_ovh(client)
        mock = _mock_test_success()
        mock.list_storage_options.return_value = [
            StorageOption(
                provider_id="ovh",
                name="my-container",
                type="object_storage",
                region="GRA11",
            ),
        ]
        with patch(
            "artenic_ai_platform_providers.hub.service._get_connector",
            return_value=mock,
        ):
            await client.post(f"{BASE}/ovh/enable")
            resp = await client.get(f"{BASE}/ovh/storage")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "my-container"

    async def test_storage_inactive_400(self, client: AsyncClient) -> None:
        resp = await client.get(f"{BASE}/ovh/storage")
        assert resp.status_code == 400


# ======================================================================
# GET /api/v1/providers/{id}/compute
# ======================================================================


class TestProviderCompute:
    async def test_compute_active_provider(self, client: AsyncClient) -> None:
        await _configure_ovh(client)
        mock = _mock_test_success()
        mock.list_compute_instances.return_value = [
            ComputeInstance(
                provider_id="ovh",
                name="b2-30",
                vcpus=8,
                memory_gb=30.0,
                disk_gb=200.0,
                region="GRA11",
            ),
            ComputeInstance(
                provider_id="ovh",
                name="gpu-a100-80g",
                vcpus=12,
                memory_gb=120.0,
                disk_gb=400.0,
                gpu_type="A100",
                gpu_count=1,
                region="GRA11",
            ),
        ]
        with patch(
            "artenic_ai_platform_providers.hub.service._get_connector",
            return_value=mock,
        ):
            await client.post(f"{BASE}/ovh/enable")
            resp = await client.get(f"{BASE}/ovh/compute")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    async def test_compute_gpu_only(self, client: AsyncClient) -> None:
        await _configure_ovh(client)
        mock = _mock_test_success()
        mock.list_compute_instances.return_value = [
            ComputeInstance(
                provider_id="ovh",
                name="gpu-a100-80g",
                vcpus=12,
                memory_gb=120.0,
                gpu_type="A100",
                gpu_count=1,
            ),
        ]
        with patch(
            "artenic_ai_platform_providers.hub.service._get_connector",
            return_value=mock,
        ):
            await client.post(f"{BASE}/ovh/enable")
            resp = await client.get(f"{BASE}/ovh/compute", params={"gpu_only": True})

        assert resp.status_code == 200
        assert len(resp.json()) == 1

    async def test_compute_inactive_400(self, client: AsyncClient) -> None:
        resp = await client.get(f"{BASE}/ovh/compute")
        assert resp.status_code == 400


# ======================================================================
# GET /api/v1/providers/{id}/regions
# ======================================================================


class TestProviderRegions:
    async def test_regions_active_provider(self, client: AsyncClient) -> None:
        await _configure_ovh(client)
        mock = _mock_test_success()
        mock.list_regions.return_value = [
            ProviderRegion(provider_id="ovh", id="GRA11", name="Gravelines"),
            ProviderRegion(provider_id="ovh", id="SBG5", name="Strasbourg"),
        ]
        with patch(
            "artenic_ai_platform_providers.hub.service._get_connector",
            return_value=mock,
        ):
            await client.post(f"{BASE}/ovh/enable")
            resp = await client.get(f"{BASE}/ovh/regions")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    async def test_regions_inactive_400(self, client: AsyncClient) -> None:
        resp = await client.get(f"{BASE}/ovh/regions")
        assert resp.status_code == 400


# ======================================================================
# GET /api/v1/providers/capabilities/storage
# ======================================================================


class TestAllStorage:
    async def test_no_active_providers(self, client: AsyncClient) -> None:
        resp = await client.get(f"{BASE}/capabilities/storage")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_aggregates_active_providers(self, client: AsyncClient) -> None:
        await _configure_ovh(client)
        mock = _mock_test_success()
        mock.list_storage_options.return_value = [
            StorageOption(provider_id="ovh", name="container-1"),
        ]
        with patch(
            "artenic_ai_platform_providers.hub.service._get_connector",
            return_value=mock,
        ):
            await client.post(f"{BASE}/ovh/enable")
            resp = await client.get(f"{BASE}/capabilities/storage")

        assert resp.status_code == 200
        assert len(resp.json()) == 1


# ======================================================================
# GET /api/v1/providers/capabilities/compute
# ======================================================================


class TestAllCompute:
    async def test_no_active_providers(self, client: AsyncClient) -> None:
        resp = await client.get(f"{BASE}/capabilities/compute")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_aggregates_active_providers(self, client: AsyncClient) -> None:
        await _configure_ovh(client)
        mock = _mock_test_success()
        mock.list_compute_instances.return_value = [
            ComputeInstance(provider_id="ovh", name="b2-30"),
        ]
        with patch(
            "artenic_ai_platform_providers.hub.service._get_connector",
            return_value=mock,
        ):
            await client.post(f"{BASE}/ovh/enable")
            resp = await client.get(f"{BASE}/capabilities/compute")

        assert resp.status_code == 200
        assert len(resp.json()) == 1
