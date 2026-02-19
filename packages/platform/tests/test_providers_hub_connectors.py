"""Tests for artenic_ai_platform.providers_hub.connectors — OpenStack connector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from artenic_ai_platform.providers_hub.connectors.base import ConnectorContext
from artenic_ai_platform.providers_hub.connectors.openstack import (
    OpenStackConnector,
    _parse_gpu_info,
)

# ======================================================================
# GPU parsing unit tests
# ======================================================================


class TestParseGpuInfo:
    def test_no_gpu(self) -> None:
        gpu_type, count = _parse_gpu_info("b2-30")
        assert gpu_type is None
        assert count == 0

    def test_gpu_prefix_only(self) -> None:
        gpu_type, count = _parse_gpu_info("gpu-b2-120")
        assert gpu_type == "GPU"
        assert count == 1

    def test_a100(self) -> None:
        gpu_type, count = _parse_gpu_info("gpu-a100-80g")
        assert gpu_type == "A100"
        assert count == 1

    def test_v100(self) -> None:
        gpu_type, count = _parse_gpu_info("gpu-v100-32g")
        assert gpu_type == "V100"
        assert count == 1

    def test_multi_gpu(self) -> None:
        gpu_type, count = _parse_gpu_info("gpu-a100-80g-x4")
        assert gpu_type == "A100"
        assert count == 4

    def test_t4(self) -> None:
        gpu_type, count = _parse_gpu_info("gpu-t4-small")
        assert gpu_type == "T4"
        assert count == 1

    def test_h100(self) -> None:
        gpu_type, count = _parse_gpu_info("gpu-h100-80g-x8")
        assert gpu_type == "H100"
        assert count == 8

    def test_l40s(self) -> None:
        gpu_type, count = _parse_gpu_info("gpu-l40s-48g")
        assert gpu_type == "L40S"
        assert count == 1

    def test_non_gpu_with_gpu_substring(self) -> None:
        """A flavor name containing 'a100' but not starting with 'gpu'."""
        gpu_type, count = _parse_gpu_info("a100-something")
        assert gpu_type == "A100"
        assert count == 1


# ======================================================================
# Mock OpenStack objects
# ======================================================================


@dataclass
class _MockFlavor:
    name: str
    vcpus: int = 4
    ram: int = 8192
    disk: int = 80


@dataclass
class _MockContainer:
    name: str
    bytes_used: int = 1024
    count: int = 5


@dataclass
class _MockRegion:
    id: str
    description: str = ""


def _make_mock_connection(
    *,
    flavors: list[Any] | None = None,
    containers: list[Any] | None = None,
    regions: list[Any] | None = None,
) -> MagicMock:
    """Build a mock openstack.Connection."""
    conn = MagicMock()

    # compute.flavors()
    if flavors is not None:
        conn.compute.flavors.return_value = iter(flavors)
    else:
        conn.compute.flavors.return_value = iter([])

    # object_store.containers()
    if containers is not None:
        conn.object_store.containers.return_value = iter(containers)
    else:
        conn.object_store.containers.return_value = iter([])

    # identity.regions()
    if regions is not None:
        conn.identity.regions.return_value = iter(regions)
    else:
        conn.identity.regions.return_value = iter([])

    conn.close = MagicMock()
    return conn


# ======================================================================
# Connector context
# ======================================================================

CTX = ConnectorContext(
    credentials={
        "auth_url": "https://auth.example.com/v3",
        "username": "user",
        "password": "pass",
        "project_id": "proj",
    },
    config={"region": "REG1", "user_domain_name": "Default"},
)


# ======================================================================
# test_connection
# ======================================================================


class TestOpenStackTestConnection:
    async def test_success(self) -> None:
        conn = _make_mock_connection(flavors=[_MockFlavor("b2-30")])
        connector = OpenStackConnector(provider_id="ovh")
        with patch.object(connector, "_build_connection", return_value=conn):
            result = await connector.test_connection(CTX)

        assert result.success is True
        assert "1 flavors" in result.message
        assert result.latency_ms is not None
        conn.close.assert_called_once()

    async def test_failure(self) -> None:
        connector = OpenStackConnector(provider_id="ovh")
        with patch.object(
            connector,
            "_build_connection",
            side_effect=Exception("auth error"),
        ):
            result = await connector.test_connection(CTX)

        assert result.success is False
        assert "auth error" in result.message


# ======================================================================
# list_storage_options
# ======================================================================


class TestOpenStackListStorage:
    async def test_returns_containers(self) -> None:
        conn = _make_mock_connection(
            containers=[
                _MockContainer("container-a", bytes_used=2048, count=10),
                _MockContainer("container-b"),
            ],
        )
        connector = OpenStackConnector(provider_id="ovh")
        with patch.object(connector, "_build_connection", return_value=conn):
            result = await connector.list_storage_options(CTX)

        assert len(result) == 2
        assert result[0].name == "container-a"
        assert result[0].provider_id == "ovh"
        assert result[0].bytes_used == 2048
        assert result[0].object_count == 10
        conn.close.assert_called_once()

    async def test_empty(self) -> None:
        conn = _make_mock_connection(containers=[])
        connector = OpenStackConnector(provider_id="ovh")
        with patch.object(connector, "_build_connection", return_value=conn):
            result = await connector.list_storage_options(CTX)
        assert result == []


# ======================================================================
# list_compute_instances
# ======================================================================


class TestOpenStackListCompute:
    async def test_returns_flavors(self) -> None:
        conn = _make_mock_connection(
            flavors=[
                _MockFlavor("b2-30", vcpus=8, ram=30720, disk=200),
                _MockFlavor("gpu-a100-80g", vcpus=12, ram=122880, disk=400),
            ],
        )
        connector = OpenStackConnector(provider_id="ovh")
        with patch.object(connector, "_build_connection", return_value=conn):
            result = await connector.list_compute_instances(CTX)

        assert len(result) == 2
        assert result[0].name == "b2-30"
        assert result[0].vcpus == 8
        assert result[0].memory_gb == 30.0
        assert result[0].gpu_count == 0
        assert result[1].gpu_type == "A100"
        assert result[1].gpu_count == 1

    async def test_gpu_only_filter(self) -> None:
        conn = _make_mock_connection(
            flavors=[
                _MockFlavor("b2-30"),
                _MockFlavor("gpu-a100-80g"),
            ],
        )
        connector = OpenStackConnector(provider_id="ovh")
        with patch.object(connector, "_build_connection", return_value=conn):
            result = await connector.list_compute_instances(CTX, gpu_only=True)
        assert len(result) == 1
        assert result[0].gpu_count > 0

    async def test_region_override(self) -> None:
        conn = _make_mock_connection(flavors=[_MockFlavor("b2-30")])
        connector = OpenStackConnector(provider_id="ovh")
        with patch.object(connector, "_build_connection", return_value=conn):
            result = await connector.list_compute_instances(CTX, region="SBG5")
        assert result[0].region == "SBG5"


# ======================================================================
# list_regions
# ======================================================================


class TestOpenStackListRegions:
    async def test_returns_regions(self) -> None:
        conn = _make_mock_connection(
            regions=[
                _MockRegion("GRA11", "Gravelines"),
                _MockRegion("SBG5", "Strasbourg"),
            ],
        )
        connector = OpenStackConnector(provider_id="ovh")
        with patch.object(connector, "_build_connection", return_value=conn):
            result = await connector.list_regions(CTX)

        assert len(result) == 2
        assert result[0].id == "GRA11"
        assert result[0].name == "Gravelines"

    async def test_empty(self) -> None:
        conn = _make_mock_connection(regions=[])
        connector = OpenStackConnector(provider_id="ovh")
        with patch.object(connector, "_build_connection", return_value=conn):
            result = await connector.list_regions(CTX)
        assert result == []


# ======================================================================
# Catalog tests
# ======================================================================


class TestCatalog:
    def test_ovh_in_catalog(self) -> None:
        from artenic_ai_platform.providers_hub.catalog import (
            PROVIDER_CATALOG,
            get_provider_definition,
            list_provider_definitions,
        )

        assert "ovh" in PROVIDER_CATALOG
        defn = get_provider_definition("ovh")
        assert defn is not None
        assert defn.connector_type == "openstack"
        assert len(defn.capabilities) == 2
        assert len(defn.credential_fields) == 4

        all_defns = list_provider_definitions()
        assert len(all_defns) >= 1

    def test_unknown_returns_none(self) -> None:
        from artenic_ai_platform.providers_hub.catalog import get_provider_definition

        assert get_provider_definition("nonexistent") is None


# ======================================================================
# _require_openstack — lines 42-47
# ======================================================================


class TestRequireOpenstack:
    def test_raises_import_error(self) -> None:
        from artenic_ai_platform.providers_hub.connectors.openstack import _require_openstack

        with pytest.raises(ImportError, match="openstacksdk"):
            _require_openstack()


# ======================================================================
# _build_connection calls _require_openstack — line 67
# ======================================================================


class TestBuildConnectionRequiresOpenstack:
    def test_raises_import_error(self) -> None:
        connector = OpenStackConnector(provider_id="ovh")
        with pytest.raises(ImportError, match="openstacksdk"):
            connector._build_connection(CTX)
