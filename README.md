<p align="center">
  <strong>Artenic AI</strong><br>
  Open-Source ML Platform — Train, Serve & Scale AI Models
</p>

<p align="center">
  <a href="https://ai.artenic.ch"><strong>Live Demo &rarr; ai.artenic.ch</strong></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License">
  <a href="https://github.com/Artenic-Ai/artenic_ai/actions/workflows/ci.yml"><img src="https://github.com/Artenic-Ai/artenic_ai/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/coverage-100%25-brightgreen" alt="Coverage">
</p>

---

## Overview

**Artenic AI** is an open-source, self-hosted ML platform for training, serving, and monitoring
AI models at scale. It dispatches training jobs locally or to 16+ cloud providers, manages model lifecycle
with ensemble orchestration, enforces budgets, and serves predictions through a unified API gateway.

Try it now: **[ai.artenic.ch](https://ai.artenic.ch)** — full demo dashboard with realistic mock data.

## Architecture

```
                 ┌────────────┐
                 │  Dashboard │  React 19 + Vite + Tailwind
                 │            │  ai.artenic.ch
                 └─────┬──────┘
                       │ REST / WebSocket
                 ┌─────┴──────┐
                 │  Platform  │  FastAPI gateway
                 │            │  Registry, Orchestrator, Monitoring
                 └──┬───┬───┬─┘
          ┌─────────┘   │   └─────────┐
     ┌────┴────┐   ┌────┴────┐   ┌────┴─────┐
     │   SDK   │   │   CLI   │   │ Optimizer │
     │ (leaf)  │   │         │   │   (LTR)   │
     └─────────┘   └─────────┘   └──────────┘
```

## Project Structure

This is a [uv](https://docs.astral.sh/uv/) workspace monorepo:

```
artenic_ai/
├── packages/
│   ├── sdk/           # Shared SDK — BaseModel contract, schemas, ensemble management
│   ├── platform/      # Central platform — Gateway, Registry, Orchestrator, Monitoring
│   ├── cli/           # Command-line interface — manage platform via terminal
│   └── optimizer/     # Training optimizer — LTR-based instance selection (stub)
├── dashboard/         # React dashboard — ai.artenic.ch
├── pyproject.toml     # Workspace root configuration
├── justfile           # Development commands
└── docker-compose.dev.yml  # PostgreSQL + MLflow
```

### Package Status

| Package | Description | Status | Tests | Coverage |
|---------|-------------|--------|-------|----------|
| `sdk` | BaseModel contract, schemas, ensemble, serialization, decorators | **Complete** | 610 | 100% |
| `platform` | FastAPI gateway, registry, training, datasets, providers, public catalog | **Complete** | 1892 | 100% |
| `cli` | Command-line interface — 12 command groups, 60+ subcommands | **Complete** | 224 | 100% |
| `optimizer` | LTR-based training instance selection | Stub | — | — |
| `dashboard` | React admin UI — 11 pages, dark theme, demo mode | **Complete** | 68 | — |

### Dependency Graph

```
sdk (leaf — no internal deps)
 ├── platform (depends on sdk)
 ├── cli (depends on sdk)
 └── optimizer (depends on sdk)
```

## Key Features

### SDK (`packages/sdk/`)

- **BaseModel ABC** with 7-state lifecycle and Template Method pattern
- **Ensemble management** — 5 strategies (weighted average, dynamic weighting, meta-learner, majority voting, stacking)
- **8 training callbacks** — early stopping, checkpointing, LR finder, mixed precision, gradient checkpointing, data versioning, distributed training, data splitting
- **7+ decorators** — `@track_inference`, `@retry`, `@timeout`, `@circuit_breaker`, `@cache_inference`, `@rate_limit`, `@validate_input`
- **Model serialization** — 6 formats (safetensors, ONNX, PyTorch, TorchScript, pickle, joblib)
- **30+ typed exceptions** with rich context
- **PlatformClient** — async HTTP client with retry and rate limit handling

### Platform (`packages/platform/`)

- **FastAPI gateway** with full middleware stack (auth, rate limit, CORS, correlation ID, error handling, metrics)
- **Model registry** — CRUD, promote/retire lifecycle, MLflow sync
- **Training orchestrator** — dispatch locally or to 16 cloud providers, job polling, spot preemption handling
- **16 training providers** — Local (subprocess), GCP, AWS, Azure, OCI, OVH, Infomaniak, Hetzner, Scaleway, Lambda Labs, RunPod, Vast.ai, CoreWeave, Kubernetes, Mock
- **Budget governance** — multi-scope (global/service/provider), multi-period (daily/weekly/monthly), enforcement modes (block/warn)
- **A/B testing** — variant routing, metric aggregation, statistical analysis
- **Ensemble management** — versioned ensembles with training dispatch
- **Health monitoring** — background drift detection, error rate tracking, latency percentiles
- **Dataset management** — CRUD, file upload/download, versioning (SHA-256), auto-stats, tabular preview, lineage tracking
- **Storage abstraction** — filesystem (default), cloud stubs (S3, GCS, Azure, OVH) — user selects backend per dataset
- **Providers Hub** — 17 REST endpoints for cloud provider management (list, configure, enable/disable, test connection, storage/compute/regions capabilities, public catalog)
- **Public Catalog** — real-time pricing & flavors from 7 cloud providers (no auth required), in-memory TTL cache, static fallback for providers without public API
- **Event system** — async pub/sub EventBus + WebSocket real-time streaming
- **Settings hot-reload** — encrypted secrets, audit log, runtime configuration
- **Plugin system** — entry-point discovery for providers, strategies, services

### CLI (`packages/cli/`)

- **12 command groups** covering all platform API endpoints (models, training, inference, ensembles, A/B tests, budgets, datasets, providers, settings, health, config)
- **Rich terminal output** — tables and key-value dicts with `--json` machine-readable mode
- **TOML multi-profile configuration** (`~/.artenic/config.toml`) with precedence: CLI flags > env vars > TOML > defaults
- **Security hardening** — credential masking, TOML injection prevention, error message sanitization
- **Async HTTP client** (httpx) with structured error handling for all SDK/platform exceptions

### Dashboard (`dashboard/`) — [ai.artenic.ch](https://ai.artenic.ch)

- **11 pages** — Overview, Model Registry, Training Jobs, Datasets, Inference Playground, Ensembles, A/B Tests, Providers, Budgets, Settings, Health Monitoring
- **Dark theme** — professional Grafana/Vercel-inspired palette with semantic color tokens
- **Demo mode** — realistic mock data (8 models, 12 training jobs, 3 ensembles, 3 A/B tests, 4 budget rules, 16 providers, 7 catalog fetchers) — no backend required
- **Catalog tab** — public pricing & flavors per provider (always visible, Live/Static badges)
- **15 UI components** — DataTable with sort/pagination, Dialog with a11y, Charts (area, bar, line), Toast notifications
- **Stack** — React 19, Vite 7, Tailwind CSS 4, Recharts 3, TanStack React Query 5, TypeScript strict

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (package manager)
- [just](https://github.com/casey/just) (command runner, optional)
- Docker + Docker Compose (for PostgreSQL + MLflow)
- Node.js 22+ (for dashboard)

### Setup

```bash
# Clone
git clone https://github.com/Artenic-Ai/artenic_ai.git
cd artenic_ai

# Install all Python packages + dev dependencies
just setup
# or: uv sync --dev && uv run pre-commit install

# Start dev dependencies (PostgreSQL + MLflow)
just dev-up

# Run tests
just test

# Run quality checks
just check
```

### Run the Platform

```bash
# Start the platform server (in-memory SQLite for dev)
uv run python -m artenic_ai_platform

# Or with PostgreSQL
ARTENIC_DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/artenic \
  uv run python -m artenic_ai_platform
```

### Run the Dashboard

```bash
cd dashboard
npm install
npm run dev          # Dev server on http://localhost:5173
```

The dashboard runs in demo mode by default (`VITE_DEMO_MODE=true`) with realistic mock data.
To connect to a running platform, set `VITE_DEMO_MODE=false` and configure `VITE_API_URL`.

### Use the CLI

```bash
# Check platform health
artenic health check

# Register and manage models
artenic model register --name my-model --version 1.0 --type lightgbm
artenic model list

# Dispatch training
artenic training dispatch --service my-svc --model my-model --provider local

# Run predictions
artenic predict my-service --data '{"feature": 1.5}'

# Manage cloud providers
artenic provider list
artenic provider configure aws --credentials '{"access_key": "..."}'
artenic provider test aws
artenic provider enable aws

# Browse public pricing catalog
artenic provider catalog aws --gpu-only

# JSON output for scripting
artenic --json training list | jq '.[] | .job_id'
```

### Docker Deployment

```bash
# Build the production image
docker build -t artenic-ai-platform .

# Run with environment variables
docker run -p 9000:9000 \
  -e ARTENIC_API_KEY=your-api-key \
  -e ARTENIC_SECRET_KEY=your-secret-key \
  -e ARTENIC_DATABASE_URL=postgresql+asyncpg://user:pass@db:5432/artenic \
  artenic-ai-platform

# Or use docker compose for full stack (PostgreSQL + MLflow + Platform)
docker compose -f docker-compose.dev.yml up -d
docker run -p 9000:9000 --network artenic-ai-dev_default \
  -e ARTENIC_DATABASE_URL=postgresql+asyncpg://postgres:postgres@artenic-ai-dev-db:5432/artenic_ai \
  artenic-ai-platform
```

### Available Commands

Run `just` to see all available commands. Key ones:

| Command | Description |
|---------|-------------|
| `just setup` | Install deps + pre-commit hooks |
| `just check` | Lint + format check + type check |
| `just test` | Run all tests |
| `just test-cov` | Run tests with coverage report |
| `just test-sdk` | Run SDK tests only |
| `just test-platform` | Run platform tests only |
| `just test-cli` | Run CLI tests only |
| `just dev-up` | Start PostgreSQL + MLflow |
| `just dev-down` | Stop dev services |

## API Endpoints

The platform exposes the following API:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `GET` | `/health/ready` | Readiness check (DB connectivity) |
| `GET` | `/health/detailed` | Full component health |
| `POST` | `/api/v1/models` | Register a model |
| `GET` | `/api/v1/models` | List all models |
| `GET` | `/api/v1/models/{id}` | Get model details |
| `POST` | `/api/v1/models/{id}/promote` | Promote model stage |
| `POST` | `/api/v1/training/dispatch` | Dispatch training job |
| `GET` | `/api/v1/training/{job_id}` | Get training status |
| `POST` | `/api/v1/services/{service}/predict` | Run inference |
| `POST` | `/api/v1/services/{service}/predict_batch` | Batch inference |
| `POST` | `/api/v1/ensembles` | Create ensemble |
| `GET` | `/api/v1/ensembles` | List ensembles |
| `POST` | `/api/v1/ab-tests` | Create A/B test |
| `GET` | `/api/v1/ab-tests/{id}/results` | Get A/B test results |
| `GET` | `/api/v1/budgets` | List budgets |
| `POST` | `/api/v1/budgets` | Create budget rule |
| `GET` | `/api/v1/budgets/spending` | Current spending |
| `GET/PUT` | `/api/v1/settings/{scope}/{section}` | Runtime settings |
| `GET` | `/api/v1/datasets/storage-options` | Available storage backends |
| `POST` | `/api/v1/datasets` | Create dataset |
| `GET` | `/api/v1/datasets` | List datasets |
| `GET` | `/api/v1/datasets/{id}` | Dataset details |
| `PATCH` | `/api/v1/datasets/{id}` | Update metadata |
| `DELETE` | `/api/v1/datasets/{id}` | Delete dataset |
| `POST` | `/api/v1/datasets/{id}/files` | Upload file (multipart) |
| `GET` | `/api/v1/datasets/{id}/files` | List files |
| `GET` | `/api/v1/datasets/{id}/files/{name}` | Download file |
| `DELETE` | `/api/v1/datasets/{id}/files/{name}` | Delete file |
| `POST` | `/api/v1/datasets/{id}/versions` | Create version snapshot |
| `GET` | `/api/v1/datasets/{id}/versions` | List versions |
| `GET` | `/api/v1/datasets/{id}/stats` | Auto-computed statistics |
| `GET` | `/api/v1/datasets/{id}/preview` | Preview tabular data |
| `POST` | `/api/v1/datasets/{id}/lineage` | Add lineage link |
| `GET` | `/api/v1/datasets/{id}/lineage` | Get lineage |
| `GET` | `/api/v1/providers` | List all providers |
| `GET` | `/api/v1/providers/capabilities/storage` | All storage options |
| `GET` | `/api/v1/providers/capabilities/compute` | All compute instances |
| `GET` | `/api/v1/providers/{id}` | Provider details |
| `PUT` | `/api/v1/providers/{id}/configure` | Configure credentials |
| `POST` | `/api/v1/providers/{id}/enable` | Enable provider |
| `POST` | `/api/v1/providers/{id}/disable` | Disable provider |
| `POST` | `/api/v1/providers/{id}/test` | Test connection |
| `DELETE` | `/api/v1/providers/{id}` | Remove provider config |
| `GET` | `/api/v1/providers/{id}/storage` | Provider storage options |
| `GET` | `/api/v1/providers/{id}/compute` | Provider compute instances |
| `GET` | `/api/v1/providers/{id}/regions` | Provider regions |
| `GET` | `/api/v1/providers/catalog/compute` | Aggregate compute catalog (all providers) |
| `GET` | `/api/v1/providers/catalog/storage` | Aggregate storage catalog (all providers) |
| `GET` | `/api/v1/providers/{id}/catalog` | Provider full catalog (compute + storage) |
| `GET` | `/api/v1/providers/{id}/catalog/compute` | Provider compute flavors |
| `GET` | `/api/v1/providers/{id}/catalog/storage` | Provider storage tiers |
| `WS` | `/ws` | Real-time event stream |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

Copyright 2026 Artenic Cloud
