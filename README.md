<p align="center">
  <strong>Artenic AI</strong><br>
  Open-Source ML Platform — Train, Serve & Scale AI Models
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License">
  <img src="https://img.shields.io/badge/tests-1973%20passed-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/coverage-100%25-brightgreen" alt="Coverage">
</p>

---

## Overview

**Artenic AI** is an open-source, self-hosted ML platform for training, serving, and monitoring
AI models at scale. It dispatches training jobs locally or to 16+ cloud providers, manages model lifecycle
with ensemble orchestration, enforces budgets, and serves predictions through a unified API gateway.

## Architecture

```
                 ┌────────────┐
                 │  Dashboard │  React 19 + Vite + Tailwind
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
│   ├── cli/           # Command-line interface (stub)
│   └── optimizer/     # Training optimizer — LTR-based instance selection (stub)
├── dashboard/         # React dashboard (stub)
├── pyproject.toml     # Workspace root configuration
├── justfile           # Development commands
└── docker-compose.dev.yml  # PostgreSQL + MLflow
```

### Package Status

| Package | Description | Status | Tests | Coverage |
|---------|-------------|--------|-------|----------|
| `sdk` | BaseModel contract, schemas, ensemble, serialization, decorators | **Complete** | 611 | 100% |
| `platform` | FastAPI gateway, registry, training orchestrator, 15 providers | **Complete** | 1362 | 100% |
| `cli` | Command-line interface | Stub | — | — |
| `optimizer` | LTR-based training instance selection | Stub | — | — |
| `dashboard` | React admin UI | Stub | — | — |

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
- **Event system** — async pub/sub EventBus + WebSocket real-time streaming
- **Settings hot-reload** — encrypted secrets, audit log, runtime configuration
- **Plugin system** — entry-point discovery for providers, strategies, services

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
| `WS` | `/ws` | Real-time event stream |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

Copyright 2026 - Artenic_
