"""FastAPI application factory with lifespan management."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.cors import CORSMiddleware

from artenic_ai_platform.ab_testing.router import router as ab_testing_router
from artenic_ai_platform.budget.router import router as budget_router
from artenic_ai_platform.budget.service import BudgetManager
from artenic_ai_platform.config.crypto import SecretManager
from artenic_ai_platform.db.engine import (
    create_async_engine,
    create_session_factory,
    create_tables,
)
from artenic_ai_platform.deps import build_get_db
from artenic_ai_platform.entities.datasets.router import router as entity_dataset_router
from artenic_ai_platform.entities.datasets.storage import (
    AzureBlobStorage,
    FilesystemStorage,
    GCSStorage,
    OVHSwiftStorage,
    S3Storage,
    StorageBackend,
)
from artenic_ai_platform.entities.ensembles.router import router as entity_ensemble_router
from artenic_ai_platform.entities.features.router import router as entity_feature_router
from artenic_ai_platform.entities.lineage.router import router as entity_lineage_router
from artenic_ai_platform.entities.models.router import router as entity_model_router
from artenic_ai_platform.entities.runs.router import router as entity_run_router
from artenic_ai_platform.events.event_bus import EventBus
from artenic_ai_platform.events.ws import router as ws_router
from artenic_ai_platform.health.monitor import HealthMonitor
from artenic_ai_platform.health.router import router as health_router
from artenic_ai_platform.inference.model_loader import ModelLoader
from artenic_ai_platform.inference.router import router as inference_router
from artenic_ai_platform.middleware.auth import AuthMiddleware
from artenic_ai_platform.middleware.correlation import CorrelationIdMiddleware
from artenic_ai_platform.middleware.errors import (
    CatchAllErrorMiddleware,
    register_error_handlers,
)
from artenic_ai_platform.middleware.logging import setup_logging
from artenic_ai_platform.middleware.metrics import MetricsMiddleware
from artenic_ai_platform.middleware.rate_limit import RateLimitMiddleware
from artenic_ai_platform.plugins.loader import discover_plugins
from artenic_ai_platform.routes.config import router as config_router
from artenic_ai_platform.settings import PlatformSettings
from artenic_ai_platform_providers.hub.router import router as providers_hub_router
from artenic_ai_platform_providers.mock import MockProvider
from artenic_ai_platform_training.router import router as training_router
from artenic_ai_platform_training.tracking.mlflow_client import MLflowTracker

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage startup / shutdown of platform components."""
    settings: PlatformSettings = app.state.settings

    # --- Startup --------------------------------------------------------
    engine = create_async_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    await create_tables(engine)

    secret_manager = SecretManager(settings.secret_key)

    # MLflow tracker (optional)
    mlflow = MLflowTracker(
        tracking_uri=settings.mlflow_tracking_uri,
        artifact_root=settings.mlflow_artifact_root,
    )
    await mlflow.setup()

    # Training providers (mock always available; real providers added if enabled)
    training_providers: dict[str, object] = {"mock": MockProvider()}

    if settings.local.enabled:
        from artenic_ai_platform_providers.local import LocalProvider

        training_providers["local"] = LocalProvider(
            work_dir=settings.local.work_dir,
            max_concurrent_jobs=settings.local.max_concurrent_jobs,
            default_timeout_hours=settings.local.default_timeout_hours,
            gpu_enabled=settings.local.gpu_enabled,
            python_executable=settings.local.python_executable,
        )

    # Event bus — shared async pub/sub
    event_bus = EventBus()

    # Health monitor — background loop
    health_monitor = HealthMonitor(
        session_factory,
        event_bus=event_bus,
        check_interval_seconds=getattr(settings.health, "check_interval_seconds", 60.0),
        drift_threshold=getattr(settings.health, "drift_threshold", 0.1),
    )

    # Discover and load model plugins
    plugin_registry = discover_plugins()
    model_loader = ModelLoader()
    await model_loader.load_from_registry(plugin_registry)

    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.secret_manager = secret_manager
    app.state.mlflow = mlflow
    app.state.training_providers = training_providers
    app.state.event_bus = event_bus
    app.state.health_monitor = health_monitor
    app.state.model_loader = model_loader

    # Budget manager factory — creates a BudgetManager per-request
    def _budget_factory(session: object) -> BudgetManager:
        return BudgetManager(
            session,  # type: ignore[arg-type]
            enforcement_mode=settings.budget.enforcement_mode,
            alert_threshold_pct=settings.budget.alert_threshold_pct,
        )

    app.state.budget_manager_factory = _budget_factory

    # Wire the get_db dependency to the real session factory
    app.state.get_db = build_get_db(session_factory)

    # Dataset storage
    app.state.dataset_storage = _build_dataset_storage(settings)

    # Start health monitor if enabled
    if getattr(settings.health, "enabled", True):
        health_monitor.start()

    db_url = str(settings.database_url)
    if "@" in db_url:
        db_url = db_url.split("@")[0].rsplit(":", 1)[0] + ":***@" + db_url.split("@")[1]
    logger.info("Platform started (database=%s)", db_url)

    yield

    # --- Shutdown -------------------------------------------------------
    await model_loader.teardown_all()
    health_monitor.stop()
    await engine.dispose()
    logger.info("Platform shut down")


def _build_dataset_storage(settings: PlatformSettings) -> StorageBackend:
    """Create the dataset storage backend based on configuration."""
    backend = settings.dataset.storage.backend
    if backend == "filesystem":
        return FilesystemStorage(base_dir=settings.dataset.storage.local_dir)
    if backend == "s3":
        return S3Storage(
            bucket=settings.dataset.storage.bucket,
            prefix=settings.dataset.storage.prefix,
            endpoint_url=settings.dataset.storage.endpoint_url,
            access_key=settings.dataset.storage.access_key,
            secret_key=settings.dataset.storage.secret_key,
            region=settings.dataset.storage.region,
        )
    if backend == "gcs":
        return GCSStorage(
            bucket=settings.dataset.storage.bucket,
            prefix=settings.dataset.storage.prefix,
            credentials_path=settings.dataset.storage.credentials_path,
            project_id=settings.dataset.storage.project_id,
        )
    if backend == "azure":
        return AzureBlobStorage(
            container=settings.dataset.storage.container,
            prefix=settings.dataset.storage.prefix,
            connection_string=settings.dataset.storage.connection_string,
        )
    if backend == "ovh":
        return OVHSwiftStorage(
            container=settings.dataset.storage.container,
            prefix=settings.dataset.storage.prefix,
            endpoint_url=settings.dataset.storage.endpoint_url,
            access_key=settings.dataset.storage.access_key,
            secret_key=settings.dataset.storage.secret_key,
            region=settings.dataset.storage.region,
        )
    msg = f"Unknown dataset storage backend: {backend}"
    raise ValueError(msg)


def create_app(settings: PlatformSettings | None = None) -> FastAPI:
    """Build and return the configured FastAPI application."""
    if settings is None:
        settings = PlatformSettings()

    setup_logging()

    app = FastAPI(
        title="Artenic AI Platform",
        version="0.7.0",
        lifespan=_lifespan,
    )
    app.state.settings = settings

    # ----- Middleware stack (outer → inner) ----------------------------
    # Order: Correlation → CatchAll → CORS → Auth → RateLimit → Metrics → GZip
    # Added in reverse because FastAPI/Starlette processes them LIFO.
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        per_minute=settings.rate_limit_per_minute,
        burst=settings.rate_limit_burst,
    )
    app.add_middleware(AuthMiddleware, api_key=settings.api_key)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(CatchAllErrorMiddleware)
    app.add_middleware(CorrelationIdMiddleware)

    # ----- Exception handlers -----------------------------------------
    register_error_handlers(app)

    # ----- Routers ---------------------------------------------------
    app.include_router(health_router)
    app.include_router(entity_dataset_router)
    app.include_router(entity_model_router)
    app.include_router(entity_run_router)
    app.include_router(config_router)
    app.include_router(training_router)
    app.include_router(budget_router)
    app.include_router(inference_router)
    app.include_router(entity_ensemble_router)
    app.include_router(entity_feature_router)
    app.include_router(entity_lineage_router)
    app.include_router(ab_testing_router)
    app.include_router(providers_hub_router)
    app.include_router(ws_router)

    return app
