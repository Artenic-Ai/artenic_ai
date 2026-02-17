# Artenic AI Platform

Central gateway, model registry, training orchestrator, and monitoring service for the Artenic AI platform.

The platform is the **core backend** of the monorepo — it depends on `artenic-ai-sdk` and exposes a FastAPI REST + WebSocket API consumed by the CLI, dashboard, and external integrations.

## Installation

```bash
pip install artenic-ai-platform

# With cloud provider SDKs
pip install artenic-ai-platform[gcp]        # Google Cloud
pip install artenic-ai-platform[aws]        # Amazon Web Services
pip install artenic-ai-platform[azure]      # Microsoft Azure
pip install artenic-ai-platform[openstack]  # OVH, Infomaniak
pip install artenic-ai-platform[k8s]        # Kubernetes, CoreWeave
```

## Quick Start

```bash
# Development mode (in-memory SQLite)
uv run python -m artenic_ai_platform

# Production (PostgreSQL)
export ARTENIC_DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/artenic
export ARTENIC_API_KEY=your-secret-api-key
export ARTENIC_SECRET_KEY=your-encryption-key
uv run python -m artenic_ai_platform
```

The server starts at `http://localhost:8000` with interactive docs at `/docs`.

## Architecture

```
packages/platform/src/artenic_ai_platform/
├── app.py                     # FastAPI app factory + lifespan
├── settings.py                # PlatformSettings (pydantic-settings)
├── deps.py                    # Dependency injection
├── __main__.py                # Entry point
│
├── config/                    # Runtime configuration
│   ├── crypto.py              # Fernet encryption (SecretManager)
│   ├── settings_schema.py     # Declarative field metadata
│   └── settings_manager.py    # Hot-reload, audit, encryption
│
├── db/                        # Database layer
│   ├── engine.py              # Async SQLAlchemy engine
│   └── models.py              # 17 ORM tables
│
├── middleware/                 # ASGI middleware stack
│   ├── correlation.py         # X-Request-ID propagation
│   ├── auth.py                # Bearer token authentication
│   ├── rate_limit.py          # Token bucket per-client
│   ├── metrics.py             # OpenTelemetry HTTP histograms
│   ├── errors.py              # Exception → HTTP status mapping
│   └── logging.py             # Structured JSON logging
│
├── registry/                  # Model registry
│   ├── service.py             # CRUD + promote + retire
│   └── router.py              # /api/v1/models/*
│
├── tracking/                  # Experiment tracking
│   └── mlflow_client.py       # Async MLflow wrapper
│
├── training/                  # Training orchestration
│   ├── service.py             # Dispatch + status + cancel
│   ├── router.py              # /api/v1/training/*
│   ├── job_poller.py          # Background polling loop
│   ├── spot_manager.py        # Preemption detection + retry
│   ├── cost_predictor.py      # Instance cost estimation
│   └── outcome_writer.py      # Training outcome persistence
│
├── providers/                 # Cloud provider integrations
│   ├── base.py                # TrainingProvider protocol
│   ├── cloud_base.py          # CloudProvider base class
│   ├── mock.py                # Dev/test provider
│   ├── local.py               # Local subprocess training
│   ├── gcp.py                 # Google Cloud
│   ├── aws.py                 # Amazon Web Services
│   ├── azure.py               # Microsoft Azure
│   ├── oci.py                 # Oracle Cloud
│   ├── ovh.py                 # OVH (OpenStack)
│   ├── infomaniak.py          # Infomaniak (OpenStack)
│   ├── hetzner.py             # Hetzner Cloud
│   ├── scaleway.py            # Scaleway
│   ├── lambda_labs.py         # Lambda Labs GPU
│   ├── runpod.py              # RunPod GPU
│   ├── vastai.py              # Vast.ai marketplace
│   ├── coreweave.py           # CoreWeave (Kubernetes)
│   └── kubernetes.py          # Generic Kubernetes
│
├── budget/                    # Budget governance
│   ├── service.py             # Multi-scope enforcement
│   ├── router.py              # /api/v1/budgets/*
│   └── alert_dispatcher.py    # Webhook alerts (HMAC-SHA256)
│
├── inference/                 # Inference gateway
│   ├── service.py             # Predict + batch + A/B routing
│   └── router.py              # /api/v1/services/{service}/*
│
├── ensemble/                  # Ensemble management
│   ├── service.py             # Versioned ensembles
│   └── router.py              # /api/v1/ensembles/*
│
├── ab_testing/                # A/B testing
│   ├── service.py             # Lifecycle + metric aggregation
│   └── router.py              # /api/v1/ab-tests/*
│
├── health/                    # Health monitoring
│   ├── monitor.py             # Background drift detection
│   └── router.py              # /health, /health/ready, /health/detailed
│
├── events/                    # Real-time events
│   ├── event_bus.py           # Async pub/sub
│   └── ws.py                  # WebSocket endpoint /ws
│
├── plugins/                   # Plugin system
│   └── loader.py              # Entry-point discovery
│
└── routes/                    # Additional routes
    └── config.py              # /api/v1/settings/*
```

## API Reference

### Health

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/health/ready` | Readiness probe (DB check) |
| `GET` | `/health/detailed` | Full component health |

### Model Registry

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/models` | Register a new model |
| `GET` | `/api/v1/models` | List all registered models |
| `GET` | `/api/v1/models/{model_id}` | Get model details |
| `POST` | `/api/v1/models/{model_id}/promote` | Promote model stage |
| `POST` | `/api/v1/models/{model_id}/retire` | Retire a model |

### Training

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/training/dispatch` | Dispatch training job |
| `GET` | `/api/v1/training/{job_id}` | Get job status |
| `POST` | `/api/v1/training/{job_id}/cancel` | Cancel running job |

### Inference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/services/{service}/predict` | Single prediction |
| `POST` | `/api/v1/services/{service}/predict_batch` | Batch predictions |

### Ensembles

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/ensembles` | Create ensemble |
| `GET` | `/api/v1/ensembles` | List ensembles |
| `GET` | `/api/v1/ensembles/{id}` | Get ensemble |
| `PUT` | `/api/v1/ensembles/{id}` | Update ensemble |
| `POST` | `/api/v1/ensembles/{id}/train` | Dispatch ensemble training |
| `GET` | `/api/v1/ensembles/{id}/versions` | Version history |

### A/B Testing

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/ab-tests` | Create A/B test |
| `GET` | `/api/v1/ab-tests` | List tests |
| `GET` | `/api/v1/ab-tests/{id}` | Get test details |
| `GET` | `/api/v1/ab-tests/{id}/results` | Aggregated results |
| `POST` | `/api/v1/ab-tests/{id}/conclude` | Conclude with winner |
| `POST` | `/api/v1/ab-tests/{id}/pause` | Pause test |
| `POST` | `/api/v1/ab-tests/{id}/resume` | Resume test |

### Budget Governance

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/budgets` | List budget rules |
| `POST` | `/api/v1/budgets` | Create budget rule |
| `PUT` | `/api/v1/budgets/{budget_id}` | Update budget |
| `GET` | `/api/v1/budgets/spending` | Current spending summary |

### Runtime Settings

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/settings/schema` | List configurable fields |
| `GET` | `/api/v1/settings/{scope}` | Get settings for scope |
| `GET` | `/api/v1/settings/{scope}/{section}` | Get section settings |
| `PUT` | `/api/v1/settings/{scope}/{section}` | Update section (hot-reload) |
| `GET` | `/api/v1/settings/audit` | Change audit log |

### WebSocket

| Protocol | Path | Description |
|----------|------|-------------|
| `WS` | `/ws?topics=training,health,...` | Real-time event stream |

Topics: `training`, `ensemble`, `health`, `lifecycle`, `config`, `ab_test`, `budget`

## Configuration

All settings are configured via environment variables with the `ARTENIC_` prefix:

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `ARTENIC_HOST` | `0.0.0.0` | Bind address |
| `ARTENIC_PORT` | `8000` | Bind port |
| `ARTENIC_DEBUG` | `false` | Debug mode |
| `ARTENIC_API_KEY` | `""` | API key (empty = auth disabled) |
| `ARTENIC_SECRET_KEY` | auto-generated | Fernet encryption key |
| `ARTENIC_DATABASE_URL` | `sqlite+aiosqlite://` | Database URL |

### MLflow

| Variable | Default | Description |
|----------|---------|-------------|
| `ARTENIC_MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URI |
| `ARTENIC_MLFLOW_ARTIFACT_ROOT` | `mlruns` | Artifact storage root |

### Cloud Providers

Each provider has an `ARTENIC_{PROVIDER}_ENABLED` toggle and provider-specific settings:

| Provider | Enable Variable | Key Settings |
|----------|----------------|-------------|
| GCP | `ARTENIC_GCP__ENABLED` | `project_id`, `credentials_path`, `region` |
| AWS | `ARTENIC_AWS__ENABLED` | `access_key_id`, `secret_access_key`, `region` |
| Azure | `ARTENIC_AZURE__ENABLED` | `subscription_id`, `tenant_id`, `client_id` |
| OCI | `ARTENIC_OCI__ENABLED` | `config_file`, `tenancy_ocid`, `compartment_id` |
| OVH | `ARTENIC_OVH__ENABLED` | `auth_url`, `username`, `password`, `project_id` |
| Infomaniak | `ARTENIC_INFOMANIAK__ENABLED` | `auth_url`, `username`, `password` |
| Hetzner | `ARTENIC_HETZNER__ENABLED` | `api_key`, `location` |
| Scaleway | `ARTENIC_SCALEWAY__ENABLED` | `secret_key`, `access_key`, `project_id` |
| Lambda Labs | `ARTENIC_LAMBDA_LABS__ENABLED` | `api_key`, `ssh_key_name` |
| RunPod | `ARTENIC_RUNPOD__ENABLED` | `api_key`, `gpu_type` |
| Vast.ai | `ARTENIC_VASTAI__ENABLED` | `api_key`, `max_price_per_hour` |
| CoreWeave | `ARTENIC_COREWEAVE__ENABLED` | `kubeconfig_path`, `namespace` |
| Kubernetes | `ARTENIC_KUBERNETES__ENABLED` | `kubeconfig_path`, `namespace` |

## Middleware Stack

The middleware executes in this order (outer → inner):

1. **Correlation ID** — `X-Request-ID` propagation + ContextVar
2. **Catch-All Errors** — Unhandled exception safety net
3. **CORS** — Cross-origin resource sharing
4. **Authentication** — Bearer token with `hmac.compare_digest`
5. **Rate Limiting** — Token bucket per-client
6. **Metrics** — OpenTelemetry HTTP request duration
7. **GZip** — Response compression

## Database Schema

17 tables managed by SQLAlchemy 2.0 with `Mapped[]` syntax:

| Table | Module | Purpose |
|-------|--------|---------|
| `artenic_models` | Registry | Model metadata and lifecycle stage |
| `promotions` | Registry | Immutable promotion audit trail |
| `training_jobs` | Training | Job status, config, costs, spot info |
| `budgets` | Budget | Scope/period budget limits |
| `budget_alerts` | Budget | Fired alert audit log |
| `training_outcomes` | Training | Denormalized outcomes for optimizer |
| `optimizer_recommendations` | Training | Optimizer decisions (stub) |
| `optimizer_training_samples` | Training | LTR samples (stub) |
| `ensembles` | Ensemble | Ensemble definition + strategy |
| `ensemble_versions` | Ensemble | Immutable version snapshots |
| `ensemble_jobs` | Ensemble | Aggregated training jobs |
| `ab_tests` | A/B Testing | Test definition + status + winner |
| `ab_test_metrics` | A/B Testing | Per-inference metric records |
| `model_health` | Health | Health metrics + drift scores |
| `config_settings` | Config | Runtime settings (encrypted) |
| `config_audit_log` | Config | Change audit trail |
| `config_overrides` | Config | Legacy compatibility |

## Development

```bash
# From monorepo root
uv sync --dev

# Run platform tests
uv run pytest packages/platform/tests/ -v

# Quality checks
uv run ruff check packages/platform/
uv run ruff format --check packages/platform/
uv run mypy packages/platform/

# Coverage (100% required)
uv run pytest packages/platform/tests/ \
  --cov=artenic_ai_platform --cov-fail-under=100
```

### Test Architecture

All tests use **aiosqlite** in-memory databases (no Docker required for CI). Cloud provider tests mock external SDKs via `unittest.mock`.

- **1362 tests**, 100% coverage
- mypy strict — 0 errors
- ruff lint + format — clean

## License

Apache License 2.0 — see [LICENSE](../../LICENSE) for details.
