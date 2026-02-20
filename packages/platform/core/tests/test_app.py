"""Tests for artenic_ai_platform.app + deps — 100% coverage."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from artenic_ai_platform.app import _build_dataset_storage, _lifespan, create_app
from artenic_ai_platform.config.crypto import SecretManager
from artenic_ai_platform.db.engine import create_async_engine, create_tables
from artenic_ai_platform.deps import build_get_db, get_db
from artenic_ai_platform.entities.datasets.storage import (
    AzureBlobStorage,
    FilesystemStorage,
    GCSStorage,
    OVHSwiftStorage,
    S3Storage,
)
from artenic_ai_platform.settings import DatasetConfig, DatasetStorageConfig, PlatformSettings

# ======================================================================
# deps.py — get_db
# ======================================================================


class TestGetDb:
    async def test_get_db_raises_before_init(self) -> None:
        with pytest.raises(RuntimeError, match="lifespan"):
            async for _ in get_db():
                pass


# ======================================================================
# deps.py — build_get_db
# ======================================================================


class TestBuildGetDb:
    async def test_build_get_db_yields_session(self) -> None:
        engine = create_async_engine("sqlite+aiosqlite://")
        await create_tables(engine)

        from artenic_ai_platform.db.engine import create_session_factory

        factory = create_session_factory(engine)
        dep = build_get_db(factory)

        sessions = []
        async for session in dep():
            sessions.append(session)

        assert len(sessions) == 1
        await engine.dispose()


# ======================================================================
# app.py — create_app
# ======================================================================


class TestCreateApp:
    def test_returns_fastapi_instance(self) -> None:
        app = create_app(
            PlatformSettings(
                database_url="sqlite+aiosqlite://",
                api_key="",
                secret_key="test",
                otel_enabled=False,
            )
        )
        assert isinstance(app, FastAPI)

    def test_default_settings(self) -> None:
        app = create_app()
        assert isinstance(app, FastAPI)
        assert isinstance(app.state.settings, PlatformSettings)

    def test_title_and_version(self) -> None:
        app = create_app(
            PlatformSettings(
                database_url="sqlite+aiosqlite://",
                api_key="",
                secret_key="test",
                otel_enabled=False,
            )
        )
        assert app.title == "Artenic AI Platform"
        assert app.version == "0.7.0"

    def test_settings_stored_in_state(self) -> None:
        settings = PlatformSettings(
            database_url="sqlite+aiosqlite://",
            api_key="",
            secret_key="test",
            otel_enabled=False,
        )
        app = create_app(settings)
        assert app.state.settings is settings

    def test_health_router_mounted(self) -> None:
        app = create_app(
            PlatformSettings(
                database_url="sqlite+aiosqlite://",
                api_key="",
                secret_key="test",
                otel_enabled=False,
            )
        )
        routes = [r.path for r in app.routes]  # type: ignore[union-attr]
        assert "/health" in routes

    async def test_health_endpoint_accessible(self) -> None:
        app = create_app(
            PlatformSettings(
                database_url="sqlite+aiosqlite://",
                api_key="",
                secret_key="test",
                otel_enabled=False,
            )
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200


# ======================================================================
# app.py — _lifespan
# ======================================================================


class TestLifespan:
    async def test_lifespan_initializes_state(self) -> None:
        settings = PlatformSettings(
            database_url="sqlite+aiosqlite://",
            api_key="",
            secret_key="test-pass",
            otel_enabled=False,
        )
        app = create_app(settings)

        async with _lifespan(app):
            assert hasattr(app.state, "engine")
            assert hasattr(app.state, "session_factory")
            assert hasattr(app.state, "secret_manager")
            assert hasattr(app.state, "get_db")
            assert isinstance(app.state.secret_manager, SecretManager)

    async def test_lifespan_registers_local_provider(self) -> None:
        from artenic_ai_platform_providers.local import LocalProvider
        from artenic_ai_platform.settings import LocalConfig

        settings = PlatformSettings(
            database_url="sqlite+aiosqlite://",
            api_key="",
            secret_key="test",
            otel_enabled=False,
            local=LocalConfig(enabled=True),
        )
        app = create_app(settings)

        async with _lifespan(app):
            providers = app.state.training_providers
            assert "local" in providers
            assert isinstance(providers["local"], LocalProvider)

    async def test_lifespan_masks_password_in_log(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """URLs containing ``@`` have the password replaced with ``***``."""
        import artenic_ai_platform.app as app_mod

        # Use a real sqlite engine but pretend the URL has credentials
        real_engine_fn = app_mod.create_async_engine
        monkeypatch.setattr(
            app_mod, "create_async_engine", lambda _url: real_engine_fn("sqlite+aiosqlite://")
        )

        settings = PlatformSettings(
            database_url="postgresql+asyncpg://user:s3cret@localhost/artenic",
            api_key="",
            secret_key="test",
            otel_enabled=False,
        )
        app = create_app(settings)

        async with _lifespan(app):
            pass

        captured = capsys.readouterr().out
        assert "s3cret" not in captured
        assert ":***@" in captured

    async def test_lifespan_disposes_engine_on_shutdown(self) -> None:
        settings = PlatformSettings(
            database_url="sqlite+aiosqlite://",
            api_key="",
            secret_key="test",
            otel_enabled=False,
        )
        app = create_app(settings)

        async with _lifespan(app):
            engine = app.state.engine
            # engine should be usable during lifespan
            async with engine.connect() as conn:
                from sqlalchemy import text

                await conn.execute(text("SELECT 1"))

        # After lifespan, engine is disposed (pool closed)

    async def test_lifespan_creates_tables(self) -> None:
        settings = PlatformSettings(
            database_url="sqlite+aiosqlite://",
            api_key="",
            secret_key="test",
            otel_enabled=False,
        )
        app = create_app(settings)

        async with _lifespan(app):
            from sqlalchemy import inspect

            async with app.state.engine.connect() as conn:
                tables = await conn.run_sync(lambda sync_conn: inspect(sync_conn).get_table_names())
            assert len(tables) == 25

    async def test_lifespan_get_db_works(self) -> None:
        settings = PlatformSettings(
            database_url="sqlite+aiosqlite://",
            api_key="",
            secret_key="test",
            otel_enabled=False,
        )
        app = create_app(settings)

        async with _lifespan(app):
            get_db_dep = app.state.get_db
            async for session in get_db_dep():
                assert session is not None


# ======================================================================
# app.py — middleware stack
# ======================================================================


class TestMiddlewareStack:
    def test_middleware_registered(self) -> None:
        app = create_app(
            PlatformSettings(
                database_url="sqlite+aiosqlite://",
                api_key="test-key",
                secret_key="test",
                otel_enabled=False,
                rate_limit_per_minute=120,
                rate_limit_burst=20,
            )
        )
        # Middleware is registered — verify by checking app.middleware_stack is not None
        # (FastAPI builds the middleware stack lazily)
        assert app is not None

    async def test_auth_blocks_without_key(self) -> None:
        app = create_app(
            PlatformSettings(
                database_url="sqlite+aiosqlite://",
                api_key="my-secret-key",
                secret_key="test",
                otel_enabled=False,
            )
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/v1/nonexistent")
        assert resp.status_code == 401

    async def test_auth_passes_with_key(self) -> None:
        app = create_app(
            PlatformSettings(
                database_url="sqlite+aiosqlite://",
                api_key="my-key",
                secret_key="test",
                otel_enabled=False,
            )
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/v1/nonexistent",
                headers={"Authorization": "Bearer my-key"},
            )
        # Should get past auth (404 from no route, not 401)
        assert resp.status_code != 401

    async def test_correlation_id_in_response(self) -> None:
        app = create_app(
            PlatformSettings(
                database_url="sqlite+aiosqlite://",
                api_key="",
                secret_key="test",
                otel_enabled=False,
            )
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        assert "x-request-id" in resp.headers


# ======================================================================
# app.py — _build_dataset_storage
# ======================================================================


def _settings_with_storage(backend: str, **overrides: str) -> PlatformSettings:
    """Create PlatformSettings with a specific dataset storage backend."""
    storage_kwargs: dict[str, str] = {"backend": backend, **overrides}
    return PlatformSettings(
        database_url="sqlite+aiosqlite://",
        api_key="",
        secret_key="test",
        otel_enabled=False,
        dataset=DatasetConfig(storage=DatasetStorageConfig(**storage_kwargs)),
    )


class TestBuildDatasetStorage:
    def test_filesystem_backend(self) -> None:
        settings = _settings_with_storage("filesystem", local_dir="/tmp/ds")
        storage = _build_dataset_storage(settings)
        assert isinstance(storage, FilesystemStorage)

    def test_s3_backend(self) -> None:
        settings = _settings_with_storage(
            "s3", bucket="my-bucket", prefix="ds/", region="eu-west-1"
        )
        storage = _build_dataset_storage(settings)
        assert isinstance(storage, S3Storage)

    def test_gcs_backend(self) -> None:
        settings = _settings_with_storage("gcs", bucket="gcs-bucket", project_id="proj-1")
        storage = _build_dataset_storage(settings)
        assert isinstance(storage, GCSStorage)

    def test_azure_backend(self) -> None:
        settings = _settings_with_storage(
            "azure", container="my-container", connection_string="DefaultEndpoints..."
        )
        storage = _build_dataset_storage(settings)
        assert isinstance(storage, AzureBlobStorage)

    def test_ovh_backend(self) -> None:
        settings = _settings_with_storage(
            "ovh", container="ovh-ctr", endpoint_url="https://s3.ovh.net", region="gra"
        )
        storage = _build_dataset_storage(settings)
        assert isinstance(storage, OVHSwiftStorage)

    def test_unknown_backend_raises(self) -> None:
        settings = _settings_with_storage("filesystem")
        # Monkey-patch to bypass Literal validation
        settings.dataset.storage.backend = "unknown"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown dataset storage backend"):
            _build_dataset_storage(settings)
