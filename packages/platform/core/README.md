# Artenic AI Platform

Central gateway, model registry, training orchestrator, and monitoring service for the Artenic AI platform.

The platform is the **core backend** of the monorepo — it depends on `artenic-ai-sdk` and exposes a FastAPI REST + WebSocket API consumed by the CLI, dashboard, and external integrations.

## Installation

```bash
pip install artenic-ai-platform    # core gateway + registry + datasets

# Cloud provider SDKs are on the providers sub-package:
pip install artenic-ai-platform-providers            # base (includes openstacksdk)
pip install artenic-ai-platform-providers[gcp]       # Google Cloud
pip install artenic-ai-platform-providers[aws]       # Amazon Web Services
pip install artenic-ai-platform-providers[azure]     # Microsoft Azure
pip install artenic-ai-platform-providers[k8s]       # Kubernetes, CoreWeave

# Training orchestration:
pip install artenic-ai-platform-training             # TrainingManager + MLflow
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

The server starts at `http://localhost:9000` with interactive docs at `/docs`.

## Architecture

The platform is split into 3 packages:

```
packages/platform/
├── core/src/artenic_ai_platform/      # artenic-ai-platform (668 tests)
│   ├── app.py                         # FastAPI app factory + lifespan
│   ├── settings/                      # PlatformSettings (4 sub-modules)
│   ├── deps.py                        # Dependency injection
│   ├── __main__.py                    # Entry point
│   ├── config/                        # Runtime config (crypto, hot-reload, audit)
│   ├── db/                            # Database layer (engine + 7 model files)
│   ├── middleware/                     # ASGI stack (auth, rate limit, CORS, metrics)
│   ├── registry/                      # Model registry (CRUD + promote/retire)
│   ├── budget/                        # Budget governance (multi-scope enforcement)
│   ├── inference/                     # Inference gateway (predict + A/B routing)
│   ├── ensemble/                      # Ensemble management (versioned)
│   ├── ab_testing/                    # A/B testing (lifecycle + metrics)
│   ├── entities/                      # Generic entity CRUD (datasets, models, runs, …)
│   ├── health/                        # Health monitoring (drift detection)
│   ├── events/                        # EventBus + WebSocket
│   ├── plugins/                       # Entry-point discovery
│   └── routes/                        # Settings routes
│
├── providers/src/artenic_ai_platform_providers/  # artenic-ai-platform-providers (1107 tests)
│   ├── base.py                        # TrainingProvider protocol
│   ├── cloud_base.py                  # CloudProvider base class
│   ├── local.py, mock.py             # Local + dev providers
│   ├── gcp.py, aws.py, azure.py, …   # 16 cloud providers
│   └── hub/                           # Provider Hub REST API + public catalog
│       ├── router.py                  # 17 REST endpoints
│       ├── catalog.py                 # PROVIDER_CATALOG registry
│       └── public_catalog/            # 7 fetchers (OVH, AWS, GCP, Azure, …)
│
└── training/src/artenic_ai_platform_training/    # artenic-ai-platform-training (125 tests)
    ├── service.py                     # TrainingManager (dispatch + status + cancel)
    ├── router.py                      # /api/v1/training/*
    ├── job_poller.py                  # Background polling loop
    ├── spot_manager.py                # Preemption detection + retry
    ├── cost_predictor.py              # Instance cost estimation
    ├── outcome_writer.py              # Training outcome persistence
    └── tracking/
        └── mlflow_client.py           # Async MLflow wrapper
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

# Run all platform tests (core + providers + training — 1900 tests)
uv run pytest packages/platform/ -v

# Run only core tests (668 tests)
uv run pytest packages/platform/core/tests/ -v

# Quality checks
uv run ruff check packages/platform/
uv run ruff format --check packages/platform/
uv run mypy packages/platform/

# Coverage (100% required, all 3 sub-packages)
uv run pytest packages/platform/ \
  --cov=artenic_ai_platform \
  --cov=artenic_ai_platform_providers \
  --cov=artenic_ai_platform_training \
  --cov-fail-under=100
```

### Test Architecture

All tests use **aiosqlite** in-memory databases (no Docker required for CI). Cloud provider tests mock external SDKs via `unittest.mock`.

- **1900 tests** across 3 sub-packages, 100% coverage
- mypy strict — 0 errors
- ruff lint + format — clean

## License

Apache License 2.0 — see [LICENSE](../../../LICENSE) for details.
