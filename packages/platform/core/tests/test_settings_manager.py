"""Tests for artenic_ai_platform.config.settings_manager — 100% coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from artenic_ai_platform.config.crypto import SecretManager
from artenic_ai_platform.config.settings_manager import SettingsManager
from artenic_ai_platform.db.models import Base

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
async def session() -> AsyncGenerator[AsyncSession, None]:
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as sess:
        yield sess
    await engine.dispose()


@pytest.fixture
def secret() -> SecretManager:
    return SecretManager("test-key")


# ======================================================================
# Schema delegation
# ======================================================================


class TestSchemaDelegation:
    def test_get_schema_all(self) -> None:
        result = SettingsManager.get_schema_all()
        assert "global" in result

    def test_get_schema_for_scope(self) -> None:
        result = SettingsManager.get_schema_for_scope("global")
        assert len(result) > 0

    def test_get_section_schema(self) -> None:
        result = SettingsManager.get_section_schema("global", "core")
        assert result is not None
        assert result["name"] == "core"


# ======================================================================
# get_all
# ======================================================================


class TestGetAll:
    async def test_empty_scope(self, session: AsyncSession) -> None:
        mgr = SettingsManager(session)
        result = await mgr.get_all("global")
        assert result == {}

    async def test_with_settings(self, session: AsyncSession, secret: SecretManager) -> None:
        mgr = SettingsManager(session, secret)
        await mgr.update_section("global", "core", {"host": "localhost"})

        result = await mgr.get_all("global")
        assert "core" in result
        assert result["core"]["host"] == "localhost"

    async def test_encrypted_value_decrypted_on_read(
        self, session: AsyncSession, secret: SecretManager
    ) -> None:
        mgr = SettingsManager(session, secret)
        # api_key is a secret field in core section
        await mgr.update_section("global", "core", {"api_key": "my-secret-key"})

        result = await mgr.get_all("global")
        assert result["core"]["api_key"] == "my-secret-key"

    async def test_encrypted_value_without_secret_manager(
        self, session: AsyncSession, secret: SecretManager
    ) -> None:
        # First store with encryption
        mgr_with = SettingsManager(session, secret)
        await mgr_with.update_section("global", "core", {"api_key": "my-key"})

        # Read without SecretManager — returns raw encrypted value
        mgr_without = SettingsManager(session)
        result = await mgr_without.get_all("global")
        # The value should be the encrypted form, not "my-key"
        assert result["core"]["api_key"] != "my-key"


# ======================================================================
# get_section
# ======================================================================


class TestGetSection:
    async def test_empty_section(self, session: AsyncSession) -> None:
        mgr = SettingsManager(session)
        result = await mgr.get_section("global", "core")
        assert result == {}

    async def test_mask_secrets(self, session: AsyncSession, secret: SecretManager) -> None:
        mgr = SettingsManager(session, secret)
        await mgr.update_section("global", "core", {"api_key": "super-secret"})

        result = await mgr.get_section("global", "core", mask_secrets=True)
        assert result["api_key"] != "super-secret"
        assert "***" in result["api_key"]

    async def test_no_mask(self, session: AsyncSession, secret: SecretManager) -> None:
        mgr = SettingsManager(session, secret)
        await mgr.update_section("global", "core", {"api_key": "visible"})

        result = await mgr.get_section("global", "core", mask_secrets=False)
        assert result["api_key"] == "visible"

    async def test_non_secret_field_not_masked(
        self, session: AsyncSession, secret: SecretManager
    ) -> None:
        mgr = SettingsManager(session, secret)
        await mgr.update_section("global", "core", {"host": "0.0.0.0"})

        result = await mgr.get_section("global", "core", mask_secrets=True)
        assert result["host"] == "0.0.0.0"

    async def test_mask_without_secret_manager(self, session: AsyncSession) -> None:
        mgr = SettingsManager(session)
        # Store a non-encrypted value in a secret field
        await mgr.update_section("global", "core", {"api_key": "plain"})

        result = await mgr.get_section("global", "core", mask_secrets=True)
        # Without secret_manager, no masking happens
        assert result["api_key"] == "plain"


# ======================================================================
# update_section
# ======================================================================


class TestUpdateSection:
    async def test_create_new_settings(self, session: AsyncSession, secret: SecretManager) -> None:
        mgr = SettingsManager(session, secret)
        result = await mgr.update_section(
            "global", "mlflow", {"mlflow_tracking_uri": "http://mlflow:5000"}
        )
        assert result["mlflow_tracking_uri"] == "http://mlflow:5000"

    async def test_update_existing_setting(
        self, session: AsyncSession, secret: SecretManager
    ) -> None:
        mgr = SettingsManager(session, secret)
        await mgr.update_section("global", "core", {"host": "0.0.0.0"})
        result = await mgr.update_section("global", "core", {"host": "127.0.0.1"})
        assert result["host"] == "127.0.0.1"

    async def test_update_encrypted_existing(
        self, session: AsyncSession, secret: SecretManager
    ) -> None:
        mgr = SettingsManager(session, secret)
        await mgr.update_section("global", "core", {"api_key": "old"})
        result = await mgr.update_section("global", "core", {"api_key": "new"})
        assert result["api_key"] == "new"

    async def test_returns_unmasked_values(
        self, session: AsyncSession, secret: SecretManager
    ) -> None:
        mgr = SettingsManager(session, secret)
        result = await mgr.update_section("global", "core", {"api_key": "abc123"})
        # update_section calls get_section(mask_secrets=False)
        assert result["api_key"] == "abc123"

    async def test_multiple_keys_at_once(
        self, session: AsyncSession, secret: SecretManager
    ) -> None:
        mgr = SettingsManager(session, secret)
        result = await mgr.update_section("global", "core", {"host": "localhost", "port": "9000"})
        assert result["host"] == "localhost"
        assert result["port"] == "9000"

    async def test_custom_updated_by(self, session: AsyncSession, secret: SecretManager) -> None:
        mgr = SettingsManager(session, secret)
        await mgr.update_section("global", "core", {"host": "test"}, updated_by="admin")
        audit = await mgr.get_audit_log()
        assert audit[0]["changed_by"] == "admin"


# ======================================================================
# get_audit_log
# ======================================================================


class TestGetAuditLog:
    async def test_empty_log(self, session: AsyncSession) -> None:
        mgr = SettingsManager(session)
        result = await mgr.get_audit_log()
        assert result == []

    async def test_log_after_create(self, session: AsyncSession, secret: SecretManager) -> None:
        mgr = SettingsManager(session, secret)
        await mgr.update_section("global", "core", {"host": "localhost"})

        log = await mgr.get_audit_log()
        assert len(log) == 1
        assert log[0]["action"] == "create"
        assert log[0]["key"] == "host"
        assert log[0]["old_value"] is None
        assert log[0]["new_value"] == "localhost"

    async def test_log_after_update(self, session: AsyncSession, secret: SecretManager) -> None:
        mgr = SettingsManager(session, secret)
        await mgr.update_section("global", "core", {"host": "old"})
        await mgr.update_section("global", "core", {"host": "new"})

        log = await mgr.get_audit_log()
        assert len(log) == 2
        actions = {e["action"] for e in log}
        assert "create" in actions
        assert "update" in actions
        update_entry = next(e for e in log if e["action"] == "update")
        assert update_entry["old_value"] == "old"
        assert update_entry["new_value"] == "new"

    async def test_pagination(self, session: AsyncSession, secret: SecretManager) -> None:
        mgr = SettingsManager(session, secret)
        for i in range(5):
            await mgr.update_section("global", "core", {"host": str(i)})

        page1 = await mgr.get_audit_log(limit=2, offset=0)
        assert len(page1) == 2

        page2 = await mgr.get_audit_log(limit=2, offset=2)
        assert len(page2) == 2

    async def test_audit_dict_fields(self, session: AsyncSession, secret: SecretManager) -> None:
        mgr = SettingsManager(session, secret)
        await mgr.update_section("global", "core", {"host": "x"})

        log = await mgr.get_audit_log()
        entry = log[0]
        assert "id" in entry
        assert "scope" in entry
        assert "section" in entry
        assert "key" in entry
        assert "changed_by" in entry
        assert "changed_at" in entry


# ======================================================================
# _get_secret_keys
# ======================================================================


class TestGetSecretKeys:
    def test_core_secret_keys(self) -> None:
        keys = SettingsManager._get_secret_keys("global", "core")
        assert "api_key" in keys
        assert "secret_key" in keys
        assert "database_url" in keys
        assert "host" not in keys

    def test_non_secret_section(self) -> None:
        keys = SettingsManager._get_secret_keys("global", "rate_limit")
        assert len(keys) == 0

    def test_unknown_scope(self) -> None:
        keys = SettingsManager._get_secret_keys("unknown", "core")
        assert len(keys) == 0

    def test_unknown_section(self) -> None:
        keys = SettingsManager._get_secret_keys("global", "nonexistent")
        assert len(keys) == 0
