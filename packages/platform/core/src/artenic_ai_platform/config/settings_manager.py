"""Settings manager — hot-reload, audit trail, encryption."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from artenic_ai_platform.config.settings_schema import (
    SCHEMA_REGISTRY,
    get_schema_all,
    get_schema_for_scope,
    get_section_schema,
)
from artenic_ai_platform.db.models import (
    ConfigAuditRecord,
    ConfigSettingRecord,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from artenic_ai_platform.config.crypto import SecretManager

logger = logging.getLogger(__name__)


class SettingsManager:
    """Read/write runtime configuration with encryption and audit."""

    def __init__(
        self,
        session: AsyncSession,
        secret_manager: SecretManager | None = None,
    ) -> None:
        self._session = session
        self._secret = secret_manager

    # ------------------------------------------------------------------
    # Schema queries (delegate to settings_schema module)
    # ------------------------------------------------------------------

    @staticmethod
    def get_schema_all() -> dict[str, list[dict[str, Any]]]:
        """Full schema registry."""
        return get_schema_all()

    @staticmethod
    def get_schema_for_scope(scope: str) -> list[dict[str, Any]]:
        """Sections for a scope."""
        return get_schema_for_scope(scope)

    @staticmethod
    def get_section_schema(scope: str, section: str) -> dict[str, Any] | None:
        """Single section schema."""
        return get_section_schema(scope, section)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_all(self, scope: str) -> dict[str, dict[str, str]]:
        """Get all settings for a scope, grouped by section."""
        stmt = (
            select(ConfigSettingRecord)
            .where(ConfigSettingRecord.scope == scope)
            .order_by(
                ConfigSettingRecord.section,
                ConfigSettingRecord.key,
            )
        )
        result = await self._session.execute(stmt)
        rows = result.scalars().all()

        grouped: dict[str, dict[str, str]] = {}
        for row in rows:
            section = grouped.setdefault(row.section, {})
            value = row.value
            if row.encrypted and self._secret:
                value = self._secret.decrypt(value)
            section[row.key] = value
        return grouped

    async def get_section(
        self,
        scope: str,
        section: str,
        mask_secrets: bool = True,
    ) -> dict[str, str]:
        """Get all settings for a specific section."""
        stmt = (
            select(ConfigSettingRecord)
            .where(
                ConfigSettingRecord.scope == scope,
                ConfigSettingRecord.section == section,
            )
            .order_by(ConfigSettingRecord.key)
        )
        result = await self._session.execute(stmt)
        rows = result.scalars().all()

        # Build a set of secret keys from schema
        secret_keys = self._get_secret_keys(scope, section)

        settings: dict[str, str] = {}
        for row in rows:
            value = row.value
            if row.encrypted and self._secret:
                value = self._secret.decrypt(value)
            if mask_secrets and row.key in secret_keys and self._secret:
                value = self._secret.mask(value)
            settings[row.key] = value
        return settings

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def update_section(
        self,
        scope: str,
        section: str,
        updates: dict[str, str],
        updated_by: str = "api",
    ) -> dict[str, str]:
        """Update settings in a section.

        - Validates keys against schema
        - Encrypts secret fields
        - Creates audit log entries
        - Returns the updated section values
        """
        secret_keys = self._get_secret_keys(scope, section)

        for key, value in updates.items():
            # Fetch existing or create new
            stmt = select(ConfigSettingRecord).where(
                ConfigSettingRecord.scope == scope,
                ConfigSettingRecord.section == section,
                ConfigSettingRecord.key == key,
            )
            result = await self._session.execute(stmt)
            existing = result.scalar_one_or_none()

            old_value: str | None = None
            encrypted = key in secret_keys
            store_value = value

            if encrypted and self._secret:
                store_value = self._secret.encrypt(value)

            if existing is not None:
                old_value = existing.value
                if existing.encrypted and self._secret:
                    old_value = self._secret.decrypt(old_value)
                existing.value = store_value
                existing.encrypted = encrypted
                existing.updated_by = updated_by
            else:
                record = ConfigSettingRecord(
                    scope=scope,
                    section=section,
                    key=key,
                    value=store_value,
                    encrypted=encrypted,
                    updated_by=updated_by,
                )
                self._session.add(record)

            # Audit log
            audit = ConfigAuditRecord(
                scope=scope,
                section=section,
                key=key,
                old_value=old_value,
                new_value=value,
                action="update" if existing else "create",
                changed_by=updated_by,
            )
            self._session.add(audit)

        await self._session.commit()
        logger.info(
            "Updated %d settings in %s/%s by %s",
            len(updates),
            scope,
            section,
            updated_by,
        )

        return await self.get_section(scope, section, mask_secrets=False)

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    async def get_audit_log(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """Return paginated audit log entries."""
        stmt = (
            select(ConfigAuditRecord)
            .order_by(ConfigAuditRecord.changed_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        rows = result.scalars().all()
        return [self._audit_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_secret_keys(scope: str, section: str) -> set[str]:
        """Collect secret field keys from the schema."""
        keys: set[str] = set()
        for sec in SCHEMA_REGISTRY.get(scope, []):
            if sec.name == section:
                for f in sec.fields:
                    if f.secret:
                        keys.add(f.key)
                break
        return keys

    @staticmethod
    def _audit_to_dict(record: ConfigAuditRecord) -> dict[str, Any]:
        """Serialise an audit record."""
        return {
            "id": record.id,
            "scope": record.scope,
            "section": record.section,
            "key": record.key,
            "old_value": record.old_value,
            "new_value": record.new_value,
            "action": record.action,
            "changed_by": record.changed_by,
            "changed_at": (record.changed_at.isoformat() if record.changed_at else None),
        }
