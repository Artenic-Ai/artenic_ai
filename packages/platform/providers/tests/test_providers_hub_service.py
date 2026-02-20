"""Tests for artenic_ai_platform_providers.hub.service — business logic."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from artenic_ai_platform.config.crypto import SecretManager
from artenic_ai_platform.db.engine import (
    create_async_engine,
    create_session_factory,
    create_tables,
)
from artenic_ai_platform.db.models import ProviderRecord
from artenic_ai_platform_providers.hub.schemas import (
    CatalogComputeFlavor,
    ComputeInstance,
    ConnectionTestResult,
    ProviderRegion,
    StorageOption,
)
from artenic_ai_platform_providers.hub.service import (
    ProviderService,
    _clear_catalog_cache,
    _clear_connector_cache,
    _get_catalog_fetcher,
    _get_connector,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncSession


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
async def session() -> AsyncGenerator[AsyncSession, None]:
    engine = create_async_engine("sqlite+aiosqlite://")
    factory = create_session_factory(engine)
    await create_tables(engine)
    async with factory() as session:
        yield session
    await engine.dispose()


@pytest.fixture
def secrets() -> SecretManager:
    return SecretManager("test-passphrase")


@pytest.fixture
def svc(session: AsyncSession, secrets: SecretManager) -> ProviderService:
    return ProviderService(session, secrets)


def _mock_connector_success() -> AsyncMock:
    mock = AsyncMock()
    mock.test_connection.return_value = ConnectionTestResult(
        success=True,
        message="OK",
        latency_ms=10.0,
    )
    mock.list_storage_options.return_value = [
        StorageOption(provider_id="ovh", name="data"),
    ]
    mock.list_compute_instances.return_value = [
        ComputeInstance(provider_id="ovh", name="b2-30", vcpus=8, memory_gb=30.0),
    ]
    mock.list_regions.return_value = [
        ProviderRegion(provider_id="ovh", id="GRA11", name="Gravelines"),
    ]
    return mock


def _mock_connector_failure() -> AsyncMock:
    mock = AsyncMock()
    mock.test_connection.return_value = ConnectionTestResult(
        success=False,
        message="Auth failed",
    )
    return mock


CREDS = {
    "auth_url": "https://auth.cloud.ovh.net/v3",
    "username": "user",
    "password": "pass",
    "project_id": "proj",
}
CONFIG = {"region": "GRA11"}


# ======================================================================
# list_providers
# ======================================================================


class TestListProviders:
    async def test_returns_catalog(self, svc: ProviderService) -> None:
        result = await svc.list_providers()
        assert len(result) >= 1
        ovh = next(p for p in result if p.id == "ovh")
        assert ovh.display_name == "OVH Public Cloud"
        assert ovh.enabled is False
        assert ovh.status == "unconfigured"

    async def test_reflects_db_state(
        self,
        svc: ProviderService,
        secrets: SecretManager,
    ) -> None:
        await svc.configure_provider("ovh", CREDS, CONFIG)
        result = await svc.list_providers()
        ovh = next(p for p in result if p.id == "ovh")
        assert ovh.status == "configured"


# ======================================================================
# get_provider
# ======================================================================


class TestGetProvider:
    async def test_returns_detail(self, svc: ProviderService) -> None:
        detail = await svc.get_provider("ovh")
        assert detail.id == "ovh"
        assert len(detail.credential_fields) >= 3
        assert detail.has_credentials is False

    async def test_unknown_raises(self, svc: ProviderService) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            await svc.get_provider("nonexistent")


# ======================================================================
# configure_provider
# ======================================================================


class TestConfigureProvider:
    async def test_creates_record(self, svc: ProviderService, session: AsyncSession) -> None:
        await svc.configure_provider("ovh", CREDS, CONFIG)
        rec = await session.get(ProviderRecord, "ovh")
        assert rec is not None
        assert rec.status == "configured"
        assert rec.config["region"] == "GRA11"

    async def test_credentials_encrypted(
        self,
        svc: ProviderService,
        session: AsyncSession,
        secrets: SecretManager,
    ) -> None:
        await svc.configure_provider("ovh", CREDS, CONFIG)
        rec = await session.get(ProviderRecord, "ovh")
        assert rec is not None
        # Credentials are encrypted — not plaintext
        assert "testpass" not in (rec.credentials or "")
        # But can be decrypted
        plaintext = secrets.decrypt(rec.credentials)
        assert json.loads(plaintext)["password"] == "pass"

    async def test_reconfigure_updates(self, svc: ProviderService) -> None:
        await svc.configure_provider("ovh", CREDS, CONFIG)
        await svc.configure_provider("ovh", {**CREDS, "username": "new"}, {"region": "SBG5"})
        detail = await svc.get_provider("ovh")
        assert detail.config["region"] == "SBG5"

    async def test_unknown_raises(self, svc: ProviderService) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            await svc.configure_provider("nonexistent", {}, {})


# ======================================================================
# enable_provider
# ======================================================================


class TestEnableProvider:
    async def test_enable_success(self, svc: ProviderService) -> None:
        await svc.configure_provider("ovh", CREDS, CONFIG)
        with patch(
            "artenic_ai_platform_providers.hub.service._get_connector",
            return_value=_mock_connector_success(),
        ):
            detail = await svc.enable_provider("ovh")
        assert detail.enabled is True
        assert detail.status == "connected"

    async def test_enable_unconfigured_raises(self, svc: ProviderService) -> None:
        with pytest.raises(ValueError, match="not configured"):
            await svc.enable_provider("ovh")

    async def test_enable_connection_failure_raises(self, svc: ProviderService) -> None:
        await svc.configure_provider("ovh", CREDS, CONFIG)
        with (
            patch(
                "artenic_ai_platform_providers.hub.service._get_connector",
                return_value=_mock_connector_failure(),
            ),
            pytest.raises(ValueError, match="Connection test failed"),
        ):
            await svc.enable_provider("ovh")


# ======================================================================
# disable_provider
# ======================================================================


class TestDisableProvider:
    async def test_disable_success(self, svc: ProviderService) -> None:
        await svc.configure_provider("ovh", CREDS, CONFIG)
        with patch(
            "artenic_ai_platform_providers.hub.service._get_connector",
            return_value=_mock_connector_success(),
        ):
            await svc.enable_provider("ovh")
        detail = await svc.disable_provider("ovh")
        assert detail.enabled is False

    async def test_disable_unconfigured_raises(self, svc: ProviderService) -> None:
        with pytest.raises(ValueError, match="not configured"):
            await svc.disable_provider("ovh")


# ======================================================================
# test_provider
# ======================================================================


class TestTestProvider:
    async def test_success_updates_status(self, svc: ProviderService) -> None:
        await svc.configure_provider("ovh", CREDS, CONFIG)
        with patch(
            "artenic_ai_platform_providers.hub.service._get_connector",
            return_value=_mock_connector_success(),
        ):
            result = await svc.test_provider("ovh")
        assert result.success is True
        detail = await svc.get_provider("ovh")
        assert detail.status == "connected"
        assert detail.last_checked_at is not None

    async def test_failure_updates_status(self, svc: ProviderService) -> None:
        await svc.configure_provider("ovh", CREDS, CONFIG)
        with patch(
            "artenic_ai_platform_providers.hub.service._get_connector",
            return_value=_mock_connector_failure(),
        ):
            result = await svc.test_provider("ovh")
        assert result.success is False
        detail = await svc.get_provider("ovh")
        assert detail.status == "error"

    async def test_unconfigured_raises(self, svc: ProviderService) -> None:
        with pytest.raises(ValueError, match="not configured"):
            await svc.test_provider("ovh")


# ======================================================================
# delete_provider_config
# ======================================================================


class TestDeleteProviderConfig:
    async def test_delete_removes_record(
        self,
        svc: ProviderService,
        session: AsyncSession,
    ) -> None:
        await svc.configure_provider("ovh", CREDS, CONFIG)
        await svc.delete_provider_config("ovh")
        rec = await session.get(ProviderRecord, "ovh")
        assert rec is None

    async def test_delete_unconfigured_noop(self, svc: ProviderService) -> None:
        # Should not raise
        await svc.delete_provider_config("ovh")


# ======================================================================
# Capability queries
# ======================================================================


class TestCapabilities:
    async def _setup_active_ovh(self, svc: ProviderService) -> None:
        await svc.configure_provider("ovh", CREDS, CONFIG)
        with patch(
            "artenic_ai_platform_providers.hub.service._get_connector",
            return_value=_mock_connector_success(),
        ):
            await svc.enable_provider("ovh")

    async def test_storage_for_provider(self, svc: ProviderService) -> None:
        await self._setup_active_ovh(svc)
        with patch(
            "artenic_ai_platform_providers.hub.service._get_connector",
            return_value=_mock_connector_success(),
        ):
            result = await svc.list_storage_for_provider("ovh")
        assert len(result) == 1
        assert result[0].name == "data"

    async def test_compute_for_provider(self, svc: ProviderService) -> None:
        await self._setup_active_ovh(svc)
        with patch(
            "artenic_ai_platform_providers.hub.service._get_connector",
            return_value=_mock_connector_success(),
        ):
            result = await svc.list_compute_for_provider("ovh")
        assert len(result) == 1

    async def test_regions_for_provider(self, svc: ProviderService) -> None:
        await self._setup_active_ovh(svc)
        with patch(
            "artenic_ai_platform_providers.hub.service._get_connector",
            return_value=_mock_connector_success(),
        ):
            result = await svc.list_regions_for_provider("ovh")
        assert len(result) == 1

    async def test_storage_inactive_raises(self, svc: ProviderService) -> None:
        with pytest.raises(ValueError, match="not active"):
            await svc.list_storage_for_provider("ovh")

    async def test_all_storage_empty(self, svc: ProviderService) -> None:
        result = await svc.list_all_storage_options()
        assert result == []

    async def test_all_compute_empty(self, svc: ProviderService) -> None:
        result = await svc.list_all_compute_instances()
        assert result == []

    async def test_all_storage_aggregates(self, svc: ProviderService) -> None:
        await self._setup_active_ovh(svc)
        with patch(
            "artenic_ai_platform_providers.hub.service._get_connector",
            return_value=_mock_connector_success(),
        ):
            result = await svc.list_all_storage_options()
        assert len(result) == 1

    async def test_all_compute_aggregates(self, svc: ProviderService) -> None:
        await self._setup_active_ovh(svc)
        with patch(
            "artenic_ai_platform_providers.hub.service._get_connector",
            return_value=_mock_connector_success(),
        ):
            result = await svc.list_all_compute_instances()
        assert len(result) == 1

    async def test_aggregation_tolerates_provider_error(self, svc: ProviderService) -> None:
        """T6: One provider failing should not break the aggregation."""
        await self._setup_active_ovh(svc)
        mock = AsyncMock()
        mock.list_storage_options.side_effect = RuntimeError("Boom")
        with patch(
            "artenic_ai_platform_providers.hub.service._get_connector",
            return_value=mock,
        ):
            result = await svc.list_all_storage_options()
        # The provider raised, so result should be empty (no other active providers)
        assert result == []


# ======================================================================
# T1: Decryption failure
# ======================================================================


class TestDecryptionFailure:
    async def test_decrypt_failure_returns_empty_credentials(self, session: AsyncSession) -> None:
        """T1: When decrypt fails, empty credentials are returned (with logging)."""
        good_secrets = SecretManager("good-passphrase")
        bad_secrets = SecretManager("bad-passphrase")

        # Configure with good secrets
        svc_good = ProviderService(session, good_secrets)
        await svc_good.configure_provider("ovh", CREDS, CONFIG)

        # Now use a service with bad secrets — decryption will fail
        svc_bad = ProviderService(session, bad_secrets)
        creds = svc_bad._decrypt_credentials(
            await session.get(ProviderRecord, "ovh")  # type: ignore[arg-type]
        )
        assert creds == {}


# ======================================================================
# T4: Unknown connector type
# ======================================================================


class TestGetConnector:
    def test_unknown_connector_type_raises(self) -> None:
        """T4: _get_connector raises ValueError for unknown connector types."""
        _clear_connector_cache()
        with pytest.raises(ValueError, match="Unknown connector type"):
            _get_connector("unknown_type", "some_id")


# ======================================================================
# T5: Enable when status is "error"
# ======================================================================


class TestEnableAfterError:
    async def test_enable_after_error_status_rejected(self, svc: ProviderService) -> None:
        """T5: After a failed test (status=error), enable should be rejected."""
        await svc.configure_provider("ovh", CREDS, CONFIG)
        # Run a failing test → status becomes "error"
        with patch(
            "artenic_ai_platform_providers.hub.service._get_connector",
            return_value=_mock_connector_failure(),
        ):
            await svc.test_provider("ovh")

        detail = await svc.get_provider("ovh")
        assert detail.status == "error"

        # Try to enable — should be rejected
        with pytest.raises(ValueError, match="Reconfigure credentials"):
            await svc.enable_provider("ovh")


# ======================================================================
# T7: Credential field validation
# ======================================================================


class TestCredentialValidation:
    async def test_empty_credentials_rejected(self, svc: ProviderService) -> None:
        """T7: Configuring with empty credentials should raise ValueError."""
        with pytest.raises(ValueError, match="Missing required credential fields"):
            await svc.configure_provider("ovh", {}, {})

    async def test_partial_credentials_rejected(self, svc: ProviderService) -> None:
        """T7: Missing required fields should raise ValueError listing them."""
        with pytest.raises(ValueError, match="password"):
            await svc.configure_provider("ovh", {"auth_url": "url", "username": "user"}, {})


# ======================================================================
# H1: disable_provider sets status to "disabled"
# ======================================================================


class TestDisableStatus:
    async def test_disable_sets_status_to_disabled(self, svc: ProviderService) -> None:
        """H1: After disabling, status should be 'disabled'."""
        await svc.configure_provider("ovh", CREDS, CONFIG)
        with patch(
            "artenic_ai_platform_providers.hub.service._get_connector",
            return_value=_mock_connector_success(),
        ):
            await svc.enable_provider("ovh")
        detail = await svc.disable_provider("ovh")
        assert detail.enabled is False
        assert detail.status == "disabled"

    async def test_reenable_after_disable(self, svc: ProviderService) -> None:
        """H1: Re-enabling after disable should work."""
        await svc.configure_provider("ovh", CREDS, CONFIG)
        with patch(
            "artenic_ai_platform_providers.hub.service._get_connector",
            return_value=_mock_connector_success(),
        ):
            await svc.enable_provider("ovh")
        await svc.disable_provider("ovh")
        with patch(
            "artenic_ai_platform_providers.hub.service._get_connector",
            return_value=_mock_connector_success(),
        ):
            detail = await svc.enable_provider("ovh")
        assert detail.enabled is True
        assert detail.status == "connected"


# ======================================================================
# Coverage: _get_connector cache hit (line 49)
# ======================================================================


class TestGetConnectorCacheHit:
    def test_second_call_returns_cached_connector(self) -> None:
        """Calling _get_connector twice with same args returns the cached instance."""
        _clear_connector_cache()
        first = _get_connector("openstack", "test-cache-id")
        second = _get_connector("openstack", "test-cache-id")
        assert first is second
        _clear_connector_cache()


# ======================================================================
# Coverage: _get_connector openstack branch (lines 52-56, 61-62)
# ======================================================================


class TestGetConnectorOpenStack:
    def test_openstack_connector_created(self) -> None:
        """_get_connector('openstack', ...) creates an OpenStackConnector."""
        from artenic_ai_platform_providers.hub.connectors.openstack import (
            OpenStackConnector,
        )

        _clear_connector_cache()
        connector = _get_connector("openstack", "test-os-id")
        assert isinstance(connector, OpenStackConnector)
        _clear_connector_cache()


class TestGetConnectorScaleway:
    def test_scaleway_connector_created(self) -> None:
        from artenic_ai_platform_providers.hub.connectors.scaleway import ScalewayConnector

        _clear_connector_cache()
        connector = _get_connector("scaleway", "test-scw-id")
        assert isinstance(connector, ScalewayConnector)
        _clear_connector_cache()


class TestGetConnectorVastai:
    def test_vastai_connector_created(self) -> None:
        from artenic_ai_platform_providers.hub.connectors.vastai import VastaiConnector

        _clear_connector_cache()
        connector = _get_connector("vastai", "test-vast-id")
        assert isinstance(connector, VastaiConnector)
        _clear_connector_cache()


class TestGetConnectorAws:
    def test_aws_connector_created(self) -> None:
        from artenic_ai_platform_providers.hub.connectors.aws import AwsConnector

        _clear_connector_cache()
        connector = _get_connector("aws", "test-aws-id")
        assert isinstance(connector, AwsConnector)
        _clear_connector_cache()


class TestGetConnectorGcp:
    def test_gcp_connector_created(self) -> None:
        from artenic_ai_platform_providers.hub.connectors.gcp import GcpConnector

        _clear_connector_cache()
        connector = _get_connector("gcp", "test-gcp-id")
        assert isinstance(connector, GcpConnector)
        _clear_connector_cache()


class TestGetConnectorAzure:
    def test_azure_connector_created(self) -> None:
        from artenic_ai_platform_providers.hub.connectors.azure import AzureConnector

        _clear_connector_cache()
        connector = _get_connector("azure", "test-azure-id")
        assert isinstance(connector, AzureConnector)
        _clear_connector_cache()


# ======================================================================
# Coverage: _test_connection with unknown provider_id (line 309)
# ======================================================================


class TestTestConnectionUnknownProvider:
    async def test_test_connection_unknown_provider(
        self,
        svc: ProviderService,
        session: AsyncSession,
    ) -> None:
        """_test_connection returns failure for a provider_id not in catalog."""
        # Insert a record with an id that is NOT in PROVIDER_CATALOG
        rec = ProviderRecord(
            id="nonexistent_provider_xyz",
            display_name="Ghost Provider",
            credentials="",
            config={},
            status="configured",
        )
        session.add(rec)
        await session.commit()
        await session.refresh(rec)

        result = await svc._test_connection("nonexistent_provider_xyz", rec)
        assert result.success is False
        assert "Unknown provider" in result.message


# ======================================================================
# Coverage: defn is None in list_all_storage_options /
#           list_all_compute_instances (lines 374, 399)
# ======================================================================


class TestAggregationSkipsUnknownProvider:
    async def test_list_all_storage_skips_unknown_provider(
        self,
        svc: ProviderService,
        session: AsyncSession,
    ) -> None:
        """list_all_storage_options skips providers not in the catalog."""
        rec = ProviderRecord(
            id="ghost_provider_storage",
            display_name="Ghost",
            credentials="",
            config={},
            status="connected",
            enabled=True,
        )
        session.add(rec)
        await session.commit()

        result = await svc.list_all_storage_options()
        # The ghost provider has no catalog entry (defn is None) → skipped
        assert result == []

    async def test_list_all_compute_skips_unknown_provider(
        self,
        svc: ProviderService,
        session: AsyncSession,
    ) -> None:
        """list_all_compute_instances skips providers not in the catalog."""
        rec = ProviderRecord(
            id="ghost_provider_compute",
            display_name="Ghost",
            credentials="",
            config={},
            status="connected",
            enabled=True,
        )
        session.add(rec)
        await session.commit()

        result = await svc.list_all_compute_instances()
        # The ghost provider has no catalog entry (defn is None) → skipped
        assert result == []


# ======================================================================
# Coverage: except Exception in list_all_compute_instances (lines 408-409)
# ======================================================================


class TestAggregationToleratesComputeError:
    async def test_compute_aggregation_tolerates_provider_error(
        self,
        svc: ProviderService,
    ) -> None:
        """One provider failing should not break compute aggregation."""
        await svc.configure_provider("ovh", CREDS, CONFIG)
        with patch(
            "artenic_ai_platform_providers.hub.service._get_connector",
            return_value=_mock_connector_success(),
        ):
            await svc.enable_provider("ovh")

        mock = AsyncMock()
        mock.list_compute_instances.side_effect = RuntimeError("Boom")
        with patch(
            "artenic_ai_platform_providers.hub.service._get_connector",
            return_value=mock,
        ):
            result = await svc.list_all_compute_instances()
        assert result == []


# ======================================================================
# Coverage: _decrypt_credentials with empty credentials (line 423)
# ======================================================================


class TestDecryptEmptyCredentials:
    async def test_empty_credentials_returns_empty_dict(
        self,
        svc: ProviderService,
        session: AsyncSession,
    ) -> None:
        """_decrypt_credentials returns {} when credentials is empty string."""
        rec = ProviderRecord(
            id="empty_creds_provider",
            display_name="Empty Creds",
            credentials="",
            config={},
            status="configured",
        )
        session.add(rec)
        await session.commit()
        await session.refresh(rec)

        result = svc._decrypt_credentials(rec)
        assert result == {}


# ======================================================================
# Coverage: _resolve_active_provider with unknown provider_id (lines 448-449)
# ======================================================================


class TestResolveActiveProviderUnknown:
    async def test_unknown_provider_raises_via_storage(
        self,
        svc: ProviderService,
    ) -> None:
        """list_storage_for_provider with unknown id raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            await svc.list_storage_for_provider("nonexistent_provider_xyz")


# ======================================================================
# Public catalog service tests
# ======================================================================


class TestGetCatalogFetcher:
    def test_ovh_fetcher(self) -> None:
        _clear_catalog_cache()
        from artenic_ai_platform_providers.hub.public_catalog.ovh import OvhCatalogFetcher

        fetcher = _get_catalog_fetcher("ovh")
        assert isinstance(fetcher, OvhCatalogFetcher)
        _clear_catalog_cache()

    def test_infomaniak_fetcher(self) -> None:
        _clear_catalog_cache()
        from artenic_ai_platform_providers.hub.public_catalog.infomaniak import (
            InfomaniakCatalogFetcher,
        )

        fetcher = _get_catalog_fetcher("infomaniak")
        assert isinstance(fetcher, InfomaniakCatalogFetcher)
        _clear_catalog_cache()

    def test_scaleway_fetcher(self) -> None:
        _clear_catalog_cache()
        from artenic_ai_platform_providers.hub.public_catalog.scaleway import (
            ScalewayCatalogFetcher,
        )

        fetcher = _get_catalog_fetcher("scaleway")
        assert isinstance(fetcher, ScalewayCatalogFetcher)
        _clear_catalog_cache()

    def test_vastai_fetcher(self) -> None:
        _clear_catalog_cache()
        from artenic_ai_platform_providers.hub.public_catalog.vastai import (
            VastaiCatalogFetcher,
        )

        fetcher = _get_catalog_fetcher("vastai")
        assert isinstance(fetcher, VastaiCatalogFetcher)
        _clear_catalog_cache()

    def test_aws_fetcher(self) -> None:
        _clear_catalog_cache()
        from artenic_ai_platform_providers.hub.public_catalog.aws import AwsCatalogFetcher

        fetcher = _get_catalog_fetcher("aws")
        assert isinstance(fetcher, AwsCatalogFetcher)
        _clear_catalog_cache()

    def test_gcp_fetcher(self) -> None:
        _clear_catalog_cache()
        from artenic_ai_platform_providers.hub.public_catalog.gcp import GcpCatalogFetcher

        fetcher = _get_catalog_fetcher("gcp")
        assert isinstance(fetcher, GcpCatalogFetcher)
        _clear_catalog_cache()

    def test_azure_fetcher(self) -> None:
        _clear_catalog_cache()
        from artenic_ai_platform_providers.hub.public_catalog.azure import AzureCatalogFetcher

        fetcher = _get_catalog_fetcher("azure")
        assert isinstance(fetcher, AzureCatalogFetcher)
        _clear_catalog_cache()

    def test_cache_hit(self) -> None:
        _clear_catalog_cache()
        first = _get_catalog_fetcher("ovh")
        second = _get_catalog_fetcher("ovh")
        assert first is second
        _clear_catalog_cache()

    def test_unknown_raises(self) -> None:
        _clear_catalog_cache()
        with pytest.raises(ValueError, match="No catalog fetcher"):
            _get_catalog_fetcher("nonexistent")
        _clear_catalog_cache()


class TestCatalogService:
    async def test_get_provider_catalog(self, svc: ProviderService) -> None:
        _clear_catalog_cache()
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_compute_catalog = AsyncMock(
            return_value=[
                CatalogComputeFlavor(provider_id="ovh", name="b2-30", vcpus=8, memory_gb=30.0),
            ]
        )
        mock_fetcher.fetch_storage_catalog = AsyncMock(return_value=[])
        mock_fetcher.supports_live_catalog.return_value = True
        with patch(
            "artenic_ai_platform_providers.hub.service._get_catalog_fetcher",
            return_value=mock_fetcher,
        ):
            result = await svc.get_provider_catalog("ovh")
        assert result.provider_id == "ovh"
        assert result.provider_name == "OVH Public Cloud"
        assert len(result.compute) == 1
        assert result.cached is False
        assert result.is_live is True
        _clear_catalog_cache()

    async def test_get_provider_catalog_cached(self, svc: ProviderService) -> None:
        _clear_catalog_cache()
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_compute_catalog = AsyncMock(
            return_value=[
                CatalogComputeFlavor(provider_id="ovh", name="b2-30"),
            ]
        )
        mock_fetcher.fetch_storage_catalog = AsyncMock(return_value=[])
        mock_fetcher.supports_live_catalog.return_value = True
        with patch(
            "artenic_ai_platform_providers.hub.service._get_catalog_fetcher",
            return_value=mock_fetcher,
        ):
            first = await svc.get_provider_catalog("ovh")
            second = await svc.get_provider_catalog("ovh")
        assert first.cached is False
        assert second.cached is True
        # Fetcher should only be called once (second call uses cache)
        assert mock_fetcher.fetch_compute_catalog.call_count == 1
        _clear_catalog_cache()

    async def test_get_provider_catalog_unknown_raises(self, svc: ProviderService) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            await svc.get_provider_catalog("nonexistent_xyz")

    async def test_get_catalog_compute_gpu_only(self, svc: ProviderService) -> None:
        _clear_catalog_cache()
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_compute_catalog = AsyncMock(
            return_value=[
                CatalogComputeFlavor(provider_id="ovh", name="b2-30"),
                CatalogComputeFlavor(
                    provider_id="ovh", name="gpu-a100", gpu_type="A100", gpu_count=1
                ),
            ]
        )
        mock_fetcher.fetch_storage_catalog = AsyncMock(return_value=[])
        mock_fetcher.supports_live_catalog.return_value = True
        with patch(
            "artenic_ai_platform_providers.hub.service._get_catalog_fetcher",
            return_value=mock_fetcher,
        ):
            result = await svc.get_catalog_compute("ovh", gpu_only=True)
        assert len(result) == 1
        assert result[0].gpu_type == "A100"
        _clear_catalog_cache()

    async def test_get_all_catalog_compute_tolerates_error(
        self,
        svc: ProviderService,
    ) -> None:
        _clear_catalog_cache()
        with patch(
            "artenic_ai_platform_providers.hub.service._get_catalog_fetcher",
            side_effect=RuntimeError("Boom"),
        ):
            result = await svc.get_all_catalog_compute()
        # All providers fail → empty list
        assert result == []
        _clear_catalog_cache()

    async def test_get_all_catalog_storage_tolerates_error(
        self,
        svc: ProviderService,
    ) -> None:
        _clear_catalog_cache()
        with patch(
            "artenic_ai_platform_providers.hub.service._get_catalog_fetcher",
            side_effect=RuntimeError("Boom"),
        ):
            result = await svc.get_all_catalog_storage()
        assert result == []
        _clear_catalog_cache()
