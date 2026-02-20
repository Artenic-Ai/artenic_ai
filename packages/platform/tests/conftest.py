"""Shared test fixtures for artenic_ai_platform."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from httpx import ASGITransport, AsyncClient

from artenic_ai_platform.app import create_app
from artenic_ai_platform.budget.service import BudgetManager
from artenic_ai_platform.config.crypto import SecretManager
from artenic_ai_platform.db.engine import (
    create_async_engine,
    create_session_factory,
    create_tables,
)
from artenic_ai_platform.deps import build_get_db
from artenic_ai_platform.entities.datasets.storage import FilesystemStorage
from artenic_ai_platform.providers.mock import MockProvider
from artenic_ai_platform.settings import PlatformSettings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from fastapi import FastAPI


@pytest.fixture
def settings() -> PlatformSettings:
    """Settings with aiosqlite in-memory database and auth disabled."""
    return PlatformSettings(
        database_url="sqlite+aiosqlite://",
        api_key="",
        secret_key="test-passphrase",
        otel_enabled=False,
    )


@pytest.fixture
def app(settings: PlatformSettings) -> FastAPI:
    """Create a FastAPI test app."""
    return create_app(settings)


@pytest.fixture
async def client(app: FastAPI, settings: PlatformSettings) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client wired to the test app with full app state."""
    # Manually initialize app state (same as lifespan)
    engine = create_async_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    await create_tables(engine)

    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.secret_manager = SecretManager(settings.secret_key)
    app.state.mlflow = None
    app.state.training_providers = {"mock": MockProvider()}
    app.state.settings = settings

    def _budget_factory(session: object) -> BudgetManager:
        return BudgetManager(
            session,  # type: ignore[arg-type]
            enforcement_mode=settings.budget.enforcement_mode,
            alert_threshold_pct=settings.budget.alert_threshold_pct,
        )

    app.state.budget_manager_factory = _budget_factory
    app.state.get_db = build_get_db(session_factory)
    app.state.dataset_storage = FilesystemStorage(base_dir="/tmp/test-datasets")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    await engine.dispose()
