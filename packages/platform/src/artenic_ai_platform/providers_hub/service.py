"""Provider Hub business logic."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import select

from artenic_ai_platform.db.models import ProviderRecord
from artenic_ai_platform.providers_hub.catalog import (
    PROVIDER_CATALOG,
    get_provider_definition,
)
from artenic_ai_platform.providers_hub.connectors.base import ConnectorContext
from artenic_ai_platform.providers_hub.schemas import (
    CapabilityOut,
    CatalogComputeFlavor,
    CatalogResponse,
    CatalogStorageTier,
    ComputeInstance,
    ConfigFieldOut,
    ConnectionTestResult,
    CredentialFieldOut,
    ProviderDetail,
    ProviderRegion,
    ProviderSummary,
    StorageOption,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from artenic_ai_platform.config.crypto import SecretManager
    from artenic_ai_platform.providers_hub.connectors.base import ProviderConnector
    from artenic_ai_platform.providers_hub.public_catalog.base import CatalogFetcher

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connector registry  (connector_type -> ProviderConnector)
# ---------------------------------------------------------------------------

_CONNECTOR_CACHE: dict[str, ProviderConnector] = {}


def _get_connector(connector_type: str, provider_id: str) -> ProviderConnector:
    """Lazy-load and cache a connector by type."""
    cache_key = f"{connector_type}:{provider_id}"
    if cache_key in _CONNECTOR_CACHE:
        return _CONNECTOR_CACHE[cache_key]

    connector: ProviderConnector

    if connector_type == "openstack":
        from artenic_ai_platform.providers_hub.connectors.openstack import (
            OpenStackConnector,
        )

        connector = OpenStackConnector(provider_id=provider_id)
    elif connector_type == "scaleway":
        from artenic_ai_platform.providers_hub.connectors.scaleway import (
            ScalewayConnector,
        )

        connector = ScalewayConnector(provider_id=provider_id)
    elif connector_type == "vastai":
        from artenic_ai_platform.providers_hub.connectors.vastai import (
            VastaiConnector,
        )

        connector = VastaiConnector(provider_id=provider_id)
    elif connector_type == "aws":
        from artenic_ai_platform.providers_hub.connectors.aws import (
            AwsConnector,
        )

        connector = AwsConnector(provider_id=provider_id)
    elif connector_type == "gcp":
        from artenic_ai_platform.providers_hub.connectors.gcp import (
            GcpConnector,
        )

        connector = GcpConnector(provider_id=provider_id)
    elif connector_type == "azure":
        from artenic_ai_platform.providers_hub.connectors.azure import (
            AzureConnector,
        )

        connector = AzureConnector(provider_id=provider_id)
    else:
        msg = f"Unknown connector type: {connector_type}"
        raise ValueError(msg)

    _CONNECTOR_CACHE[cache_key] = connector
    return connector


def _evict_connector_cache(connector_type: str, provider_id: str) -> None:
    """Remove a cached connector entry (e.g. after reconfigure or delete)."""
    cache_key = f"{connector_type}:{provider_id}"
    _CONNECTOR_CACHE.pop(cache_key, None)


def _clear_connector_cache() -> None:
    """Clear the entire connector cache (useful for tests)."""
    _CONNECTOR_CACHE.clear()


# ---------------------------------------------------------------------------
# Public catalog fetcher registry
# ---------------------------------------------------------------------------

from artenic_ai_platform.providers_hub.public_catalog.base import CatalogCache  # noqa: E402

_CATALOG_CACHE = CatalogCache(ttl_seconds=3600.0)
_CATALOG_FETCHER_CACHE: dict[str, CatalogFetcher] = {}


def _get_catalog_fetcher(provider_id: str) -> CatalogFetcher:
    """Lazy-load and cache a catalog fetcher by provider ID."""
    if provider_id in _CATALOG_FETCHER_CACHE:
        return _CATALOG_FETCHER_CACHE[provider_id]

    fetcher: CatalogFetcher

    if provider_id == "ovh":
        from artenic_ai_platform.providers_hub.public_catalog.ovh import (
            OvhCatalogFetcher,
        )

        fetcher = OvhCatalogFetcher()
    elif provider_id == "infomaniak":
        from artenic_ai_platform.providers_hub.public_catalog.infomaniak import (
            InfomaniakCatalogFetcher,
        )

        fetcher = InfomaniakCatalogFetcher()
    elif provider_id == "scaleway":
        from artenic_ai_platform.providers_hub.public_catalog.scaleway import (
            ScalewayCatalogFetcher,
        )

        fetcher = ScalewayCatalogFetcher()
    elif provider_id == "vastai":
        from artenic_ai_platform.providers_hub.public_catalog.vastai import (
            VastaiCatalogFetcher,
        )

        fetcher = VastaiCatalogFetcher()
    elif provider_id == "aws":
        from artenic_ai_platform.providers_hub.public_catalog.aws import (
            AwsCatalogFetcher,
        )

        fetcher = AwsCatalogFetcher()
    elif provider_id == "gcp":
        from artenic_ai_platform.providers_hub.public_catalog.gcp import (
            GcpCatalogFetcher,
        )

        fetcher = GcpCatalogFetcher()
    elif provider_id == "azure":
        from artenic_ai_platform.providers_hub.public_catalog.azure import (
            AzureCatalogFetcher,
        )

        fetcher = AzureCatalogFetcher()
    else:
        msg = f"No catalog fetcher for provider: {provider_id}"
        raise ValueError(msg)

    _CATALOG_FETCHER_CACHE[provider_id] = fetcher
    return fetcher


def _clear_catalog_cache() -> None:
    """Clear the catalog cache and fetcher cache (useful for tests)."""
    _CATALOG_CACHE.clear()
    _CATALOG_FETCHER_CACHE.clear()


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ProviderService:
    """Provider Hub service — CRUD, testing, and capability queries."""

    def __init__(
        self,
        session: AsyncSession,
        secret_manager: SecretManager,
    ) -> None:
        self._session = session
        self._secrets = secret_manager

    # ------------------------------------------------------------------
    # Catalog sync — ensure all catalog providers have a DB record
    # ------------------------------------------------------------------

    async def _sync_catalog(self) -> dict[str, ProviderRecord]:
        """Create ProviderRecord rows for any catalog entry missing from DB.

        Returns a mapping of provider_id → ProviderRecord for all
        catalog providers.
        """
        rows = (await self._session.execute(select(ProviderRecord))).scalars().all()
        db_map: dict[str, ProviderRecord] = {r.id: r for r in rows}

        created = False
        for defn in PROVIDER_CATALOG.values():
            if defn.id not in db_map:
                rec = ProviderRecord(
                    id=defn.id,
                    display_name=defn.display_name,
                )
                self._session.add(rec)
                db_map[defn.id] = rec
                created = True

        if created:
            await self._session.flush()

        return db_map

    # ------------------------------------------------------------------
    # List providers (catalog merged with DB state)
    # ------------------------------------------------------------------

    async def list_providers(self) -> list[ProviderSummary]:
        """Return all known providers with their current state."""
        db_map = await self._sync_catalog()

        result: list[ProviderSummary] = []
        for defn in PROVIDER_CATALOG.values():
            rec = db_map.get(defn.id)
            result.append(
                ProviderSummary(
                    id=defn.id,
                    display_name=defn.display_name,
                    description=defn.description,
                    website=defn.website,
                    connector_type=defn.connector_type,
                    capabilities=[
                        CapabilityOut(type=c.type, name=c.name, description=c.description)
                        for c in defn.capabilities
                    ],
                    enabled=rec.enabled if rec else False,
                    status=rec.status if rec else "unconfigured",
                    status_message=rec.status_message if rec else "",
                )
            )
        return result

    # ------------------------------------------------------------------
    # Get provider detail
    # ------------------------------------------------------------------

    async def get_provider(self, provider_id: str) -> ProviderDetail:
        """Return full detail for one provider."""
        defn = get_provider_definition(provider_id)
        if defn is None:
            msg = f"Unknown provider: {provider_id}"
            raise ValueError(msg)

        rec = await self._session.get(ProviderRecord, provider_id)
        if rec is None:
            rec = ProviderRecord(id=defn.id, display_name=defn.display_name)
            self._session.add(rec)
            await self._session.flush()

        config: dict[str, str] = rec.config if rec.config else {}
        has_credentials = bool(rec.credentials)

        return ProviderDetail(
            id=defn.id,
            display_name=defn.display_name,
            description=defn.description,
            website=defn.website,
            connector_type=defn.connector_type,
            capabilities=[
                CapabilityOut(type=c.type, name=c.name, description=c.description)
                for c in defn.capabilities
            ],
            credential_fields=[
                CredentialFieldOut(
                    key=f.key,
                    label=f.label,
                    required=f.required,
                    secret=f.secret,
                    placeholder=f.placeholder,
                )
                for f in defn.credential_fields
            ],
            config_fields=[
                ConfigFieldOut(
                    key=f.key,
                    label=f.label,
                    default=f.default,
                    description=f.description,
                )
                for f in defn.config_fields
            ],
            config=config,
            has_credentials=has_credentials,
            enabled=rec.enabled,
            status=rec.status,
            status_message=rec.status_message,
            last_checked_at=rec.last_checked_at,
            created_at=rec.created_at,
            updated_at=rec.updated_at,
        )

    # ------------------------------------------------------------------
    # Configure provider
    # ------------------------------------------------------------------

    async def configure_provider(
        self,
        provider_id: str,
        credentials: dict[str, str],
        config: dict[str, str],
    ) -> ProviderDetail:
        """Store (or update) credentials and config for a provider."""
        defn = get_provider_definition(provider_id)
        if defn is None:
            msg = f"Unknown provider: {provider_id}"
            raise ValueError(msg)

        # H3: Validate that all required credential fields are present
        missing = [f.key for f in defn.credential_fields if f.required and f.key not in credentials]
        if missing:
            msg = f"Missing required credential fields: {', '.join(missing)}"
            raise ValueError(msg)

        encrypted = self._secrets.encrypt(json.dumps(credentials))

        rec = await self._session.get(ProviderRecord, provider_id)
        if rec is None:
            rec = ProviderRecord(
                id=provider_id,
                display_name=defn.display_name,
                credentials=encrypted,
                config=config,
                status="configured",
                status_message="Credentials saved — run a connection test to verify.",
            )
            self._session.add(rec)
        else:
            rec.credentials = encrypted
            rec.config = config
            rec.status = "configured"
            rec.status_message = "Credentials updated — run a connection test to verify."

        await self._session.commit()
        await self._session.refresh(rec)

        # C1: Evict stale connector from cache after reconfigure
        _evict_connector_cache(defn.connector_type, provider_id)

        return await self.get_provider(provider_id)

    # ------------------------------------------------------------------
    # Enable / disable
    # ------------------------------------------------------------------

    async def enable_provider(self, provider_id: str) -> ProviderDetail:
        """Enable a provider (requires successful connection test)."""
        rec = await self._session.get(ProviderRecord, provider_id)
        if rec is None:
            msg = f"Provider {provider_id} is not configured yet."
            raise ValueError(msg)

        # H1+H2: Accept "disabled" status; improve error message
        if rec.status not in ("connected", "configured", "disabled"):
            msg = (
                f"Cannot enable provider {provider_id}: "
                f"status is '{rec.status}'. "
                f"Reconfigure credentials or run a successful connection test."
            )
            raise ValueError(msg)

        # Test connection before enabling
        test_result = await self._test_connection(provider_id, rec)
        if not test_result.success:
            msg = f"Connection test failed: {test_result.message}"
            raise ValueError(msg)

        rec.enabled = True
        rec.status = "connected"
        rec.status_message = "Provider is active."
        await self._session.commit()
        await self._session.refresh(rec)
        return await self.get_provider(provider_id)

    async def disable_provider(self, provider_id: str) -> ProviderDetail:
        """Disable a provider."""
        rec = await self._session.get(ProviderRecord, provider_id)
        if rec is None:
            msg = f"Provider {provider_id} is not configured yet."
            raise ValueError(msg)

        rec.enabled = False
        rec.status = "disabled"  # H1: Update status on disable
        rec.status_message = "Provider disabled by user."
        await self._session.commit()
        await self._session.refresh(rec)
        return await self.get_provider(provider_id)

    # ------------------------------------------------------------------
    # Test connection
    # ------------------------------------------------------------------

    async def test_provider(self, provider_id: str) -> ConnectionTestResult:
        """Run a connection test for a configured provider."""
        rec = await self._session.get(ProviderRecord, provider_id)
        if rec is None:
            msg = f"Provider {provider_id} is not configured yet."
            raise ValueError(msg)

        result = await self._test_connection(provider_id, rec)

        rec.last_checked_at = datetime.now(UTC)
        if result.success:
            rec.status = "connected"
            rec.status_message = result.message
        else:
            rec.status = "error"
            rec.status_message = result.message

        await self._session.commit()
        await self._session.refresh(rec)
        return result

    async def _test_connection(
        self,
        provider_id: str,
        rec: ProviderRecord,
    ) -> ConnectionTestResult:
        """Internal helper: run the connector's test_connection."""
        defn = get_provider_definition(provider_id)
        if defn is None:
            return ConnectionTestResult(
                success=False,
                message=f"Unknown provider: {provider_id}",
            )

        credentials = self._decrypt_credentials(rec)
        config = rec.config if rec.config else {}
        connector = _get_connector(defn.connector_type, provider_id)
        ctx = ConnectorContext(credentials=credentials, config=config)
        return await connector.test_connection(ctx)

    # ------------------------------------------------------------------
    # Delete configuration
    # ------------------------------------------------------------------

    async def delete_provider_config(self, provider_id: str) -> None:
        """Remove all stored credentials and config for a provider."""
        rec = await self._session.get(ProviderRecord, provider_id)
        if rec is not None:
            # C1: Evict connector from cache on delete
            defn = get_provider_definition(provider_id)
            if defn is not None:
                _evict_connector_cache(defn.connector_type, provider_id)

            await self._session.delete(rec)
            await self._session.commit()

    # ------------------------------------------------------------------
    # Capability queries (live data from provider API)
    # ------------------------------------------------------------------

    async def list_storage_for_provider(
        self,
        provider_id: str,
    ) -> list[StorageOption]:
        """Fetch storage options from a single active provider."""
        _rec, connector, ctx = await self._resolve_active_provider(provider_id)
        return await connector.list_storage_options(ctx)

    async def list_compute_for_provider(
        self,
        provider_id: str,
        *,
        region: str | None = None,
        gpu_only: bool = False,
    ) -> list[ComputeInstance]:
        """Fetch compute instances from a single active provider."""
        _rec, connector, ctx = await self._resolve_active_provider(provider_id)
        return await connector.list_compute_instances(ctx, region=region, gpu_only=gpu_only)

    async def list_regions_for_provider(
        self,
        provider_id: str,
    ) -> list[ProviderRegion]:
        """Fetch regions from a single active provider."""
        _rec, connector, ctx = await self._resolve_active_provider(provider_id)
        return await connector.list_regions(ctx)

    async def list_all_storage_options(self) -> list[StorageOption]:
        """Aggregate storage options from all active providers."""
        result: list[StorageOption] = []
        for provider_id, rec in await self._get_active_providers():
            try:
                defn = get_provider_definition(provider_id)
                if defn is None:
                    continue
                connector = _get_connector(defn.connector_type, provider_id)
                ctx = self._make_ctx(rec)
                options = await connector.list_storage_options(ctx)
                result.extend(options)
            except Exception:
                logger.warning(
                    "Failed to list storage for provider %s",
                    provider_id,
                    exc_info=True,
                )
        return result

    async def list_all_compute_instances(
        self,
        *,
        region: str | None = None,
        gpu_only: bool = False,
    ) -> list[ComputeInstance]:
        """Aggregate compute instances from all active providers."""
        result: list[ComputeInstance] = []
        for provider_id, rec in await self._get_active_providers():
            try:
                defn = get_provider_definition(provider_id)
                if defn is None:
                    continue
                connector = _get_connector(defn.connector_type, provider_id)
                ctx = self._make_ctx(rec)
                instances = await connector.list_compute_instances(
                    ctx,
                    region=region,
                    gpu_only=gpu_only,
                )
                result.extend(instances)
            except Exception:
                logger.warning(
                    "Failed to list compute for provider %s",
                    provider_id,
                    exc_info=True,
                )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decrypt_credentials(self, rec: ProviderRecord) -> dict[str, str]:
        """Decrypt the JSON credentials blob."""
        if not rec.credentials:
            return {}
        plaintext = self._secrets.decrypt(rec.credentials)
        if plaintext == "***DECRYPTION_FAILED***":
            # C2: Log error instead of silently returning empty dict
            logger.error(
                "Failed to decrypt credentials for provider %s",
                rec.id,
            )
            return {}
        result: dict[str, str] = json.loads(plaintext)
        return result

    def _make_ctx(self, rec: ProviderRecord) -> ConnectorContext:
        """Build a ConnectorContext from a ProviderRecord."""
        return ConnectorContext(
            credentials=self._decrypt_credentials(rec),
            config=rec.config if rec.config else {},
        )

    async def _resolve_active_provider(
        self,
        provider_id: str,
    ) -> tuple[ProviderRecord, ProviderConnector, ConnectorContext]:
        """Fetch and validate an active provider, return (rec, connector, ctx)."""
        defn = get_provider_definition(provider_id)
        if defn is None:
            msg = f"Unknown provider: {provider_id}"
            raise ValueError(msg)

        rec = await self._session.get(ProviderRecord, provider_id)
        if rec is None or not rec.enabled:
            msg = f"Provider {provider_id} is not active."
            raise ValueError(msg)

        connector = _get_connector(defn.connector_type, provider_id)
        ctx = self._make_ctx(rec)
        return rec, connector, ctx

    async def _get_active_providers(self) -> list[tuple[str, ProviderRecord]]:
        """Return all enabled providers as (id, record) pairs."""
        rows = (
            (
                await self._session.execute(
                    select(ProviderRecord).where(ProviderRecord.enabled.is_(True))
                )
            )
            .scalars()
            .all()
        )
        return [(r.id, r) for r in rows]

    # ------------------------------------------------------------------
    # Public catalog (no credentials needed)
    # ------------------------------------------------------------------

    async def get_provider_catalog(self, provider_id: str) -> CatalogResponse:
        """Fetch public catalog (compute + storage) for a provider."""
        defn = get_provider_definition(provider_id)
        if defn is None:
            msg = f"Unknown provider: {provider_id}"
            raise ValueError(msg)

        cache_key = f"catalog:{provider_id}"
        cached = _CATALOG_CACHE.get(cache_key)
        if cached is not None:
            return CatalogResponse(
                provider_id=provider_id,
                provider_name=defn.display_name,
                compute=cached["compute"],
                storage=cached["storage"],
                is_live=cached["is_live"],
                cached=True,
                last_fetched_at=cached["fetched_at"],
            )

        fetcher = _get_catalog_fetcher(provider_id)
        compute = await fetcher.fetch_compute_catalog()
        storage = await fetcher.fetch_storage_catalog()
        now = datetime.now(UTC)

        _CATALOG_CACHE.set(
            cache_key,
            {
                "compute": compute,
                "storage": storage,
                "is_live": fetcher.supports_live_catalog(),
                "fetched_at": now,
            },
        )

        return CatalogResponse(
            provider_id=provider_id,
            provider_name=defn.display_name,
            compute=compute,
            storage=storage,
            is_live=fetcher.supports_live_catalog(),
            cached=False,
            last_fetched_at=now,
        )

    async def get_catalog_compute(
        self,
        provider_id: str,
        *,
        gpu_only: bool = False,
    ) -> list[CatalogComputeFlavor]:
        """Fetch just the compute catalog for a provider."""
        catalog = await self.get_provider_catalog(provider_id)
        flavors = catalog.compute
        if gpu_only:
            flavors = [f for f in flavors if f.gpu_count > 0]
        return flavors

    async def get_catalog_storage(
        self,
        provider_id: str,
    ) -> list[CatalogStorageTier]:
        """Fetch just the storage catalog for a provider."""
        catalog = await self.get_provider_catalog(provider_id)
        return catalog.storage

    async def get_all_catalog_compute(
        self,
        *,
        gpu_only: bool = False,
    ) -> list[CatalogComputeFlavor]:
        """Aggregate compute catalog from all providers."""
        result: list[CatalogComputeFlavor] = []
        for provider_id in PROVIDER_CATALOG:
            try:
                flavors = await self.get_catalog_compute(
                    provider_id,
                    gpu_only=gpu_only,
                )
                result.extend(flavors)
            except Exception:
                logger.warning(
                    "Failed to fetch catalog for provider %s",
                    provider_id,
                    exc_info=True,
                )
        return result

    async def get_all_catalog_storage(self) -> list[CatalogStorageTier]:
        """Aggregate storage catalog from all providers."""
        result: list[CatalogStorageTier] = []
        for provider_id in PROVIDER_CATALOG:
            try:
                storage = await self.get_catalog_storage(provider_id)
                result.extend(storage)
            except Exception:
                logger.warning(
                    "Failed to fetch storage catalog for provider %s",
                    provider_id,
                    exc_info=True,
                )
        return result
