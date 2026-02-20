# Changelog

All notable changes to this project will be documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.7.0] — 2026-02-20

### Step 8 — Monorepo Refactoring: Fine-Grained Packages

#### Architecture — Package Split (4 → 9 packages)
- **SDK split** from 1 monolithic package into 4:
  - `artenic-ai-sdk` (core) — BaseModel contract, schemas, types, exceptions, decorators, observability, serialization, testing utilities, config management (332 tests)
  - `artenic-ai-sdk-ensemble` — EnsembleManager, 5 strategies (WeightedAverage, DynamicWeighting, MetaLearner, MajorityVoting, Stacking), evolution policy (101 tests)
  - `artenic-ai-sdk-training` — 8 training callbacks, checkpointing, LR finder, mixed precision, gradient checkpointing, data versioning, distributed training, data splitting (150 tests)
  - `artenic-ai-sdk-client` — PlatformClient async HTTP client (httpx) (27 tests)
- **Platform split** from 1 monolithic package into 3:
  - `artenic-ai-platform` (core) — FastAPI gateway, middleware, model registry, datasets, budget, inference, ensemble, A/B testing, health monitor, events, settings, plugins (668 tests)
  - `artenic-ai-platform-providers` — 16 cloud training providers, Provider Hub REST API, public catalog with 7 fetchers (1107 tests)
  - `artenic-ai-platform-training` — Training orchestration (TrainingManager), job polling, spot manager, cost predictor, outcome writer, MLflow tracking (125 tests)
- CLI and Optimizer unchanged

#### Installation — Extras System
- `pip install artenic-ai-sdk` installs core only (minimal dependencies)
- `pip install artenic-ai-sdk[ensemble]` adds ensemble strategies
- `pip install artenic-ai-sdk[training]` adds training callbacks + serialization
- `pip install artenic-ai-sdk[client]` adds HTTP platform client
- `pip install artenic-ai-sdk[all]` installs all SDK sub-packages
- SDK core provides conditional re-exports via `try/except ImportError`

#### Workspace Configuration
- `pyproject.toml` workspace members use glob patterns: `packages/sdk/*`, `packages/platform/*`
- 9 source paths in ruff, mypy, pytest, and coverage configs
- CI pipeline coverage flags updated for all sub-packages
- `justfile` gains 7 new granular test recipes

#### Legacy Cleanup
- Migrated `ModelHealthRecord` FK from `RegisteredModel` to `MLModel`
- Removed 6 legacy database tables
- Split `models.py` (510 lines) into 7 domain files under `db/models/`
- Split `settings.py` (510 lines) into 4 sub-modules under `settings/`

#### Generic Entity CRUD API
- New `entities/` module with `BaseService` pattern for datasets, models, ensembles, runs, features, lineage
- Replaces legacy per-entity modules with a unified ML entity CRUD API

#### Quality
- 2738 tests across all packages, 100% coverage
- mypy strict, ruff clean
- Dashboard: 68 tests, TypeScript strict

## [0.6.0] — 2026-02-19

### Step 7 — Provider Hub Public Catalog

#### Public Catalog — Pricing & Flavors (no authentication required)
- `public_catalog/` module with `CatalogFetcher` ABC and in-memory TTL cache (1h default)
- 7 catalog fetchers:
  - **OVH** — `api.ovh.com/v1/cloud/price` (compute flavors + storage tiers)
  - **Azure** — `prices.azure.com/api/retail/prices` (paginated OData, Linux VMs)
  - **AWS** — `pricing.us-east-1.amazonaws.com` bulk JSON (EC2 EU-West-1, On-Demand)
  - **GCP** — `cloudpricingcalculator.appspot.com` price list (Compute Engine + Cloud Storage)
  - **Vast.ai** — `console.vast.ai/api/v0/bundles` GPU marketplace offers
  - **Scaleway** — live products API with static fallback (10 instance types, 2 storage tiers)
  - **Infomaniak** — static catalog (14 instance types, 2 storage tiers)
- 3 new Pydantic models: `CatalogComputeFlavor`, `CatalogStorageTier`, `CatalogResponse`
- 5 new REST endpoints (17 total for Provider Hub):
  - `GET /api/v1/providers/catalog/compute` — aggregate compute catalog (all providers)
  - `GET /api/v1/providers/catalog/storage` — aggregate storage catalog (all providers)
  - `GET /api/v1/providers/{id}/catalog` — full provider catalog (compute + storage + metadata)
  - `GET /api/v1/providers/{id}/catalog/compute` — provider compute flavors (with `gpu_only` filter)
  - `GET /api/v1/providers/{id}/catalog/storage` — provider storage tiers

#### Dashboard — Catalog Tab
- "Catalog" tab on provider detail page (always visible, not gated by credentials)
- Compute flavors table: Flavor, vCPUs, Memory, GPU, Price/hr, Category
- Storage tiers table: Tier, Type, Price/GB/month, Description
- Live/Static and Cached status badges
- Mock catalog data for all 7 providers

#### Quality
- Platform: 1892 tests (338 new for providers + catalog), 100% coverage
- mypy strict — 0 errors across 85 source files
- ruff lint + format — clean
- Dashboard: tsc + build clean

---

## [0.5.0] — 2026-02-19

### Step 6 — Datasets — Full-Stack Feature (Platform + CLI + Dashboard)

#### Platform — Dataset Management API (`/api/v1/datasets`)
- 4 ORM models: `DatasetRecord`, `DatasetVersionRecord`, `DatasetFileRecord`, `DatasetLineageRecord` (21 tables total)
- `StorageBackend` abstraction with `FilesystemStorage` implementation + cloud stubs (S3, GCS, Azure, OVH)
- `DatasetService` — CRUD, file upload/download, versioning (SHA-256 hashes), auto-stats (CSV/JSON/JSONL record counts), tabular preview, lineage tracking
- 17 REST endpoints: storage options, CRUD, files (upload/list/download/delete), versions, stats, preview, lineage
- `DatasetConfig` + `DatasetStorageConfig` in platform settings
- `python-multipart` dependency for file upload support

#### CLI — Dataset Commands
- `artenic dataset` command group: list, create, get, update, delete
- File operations: upload, download, files
- Analytics: stats, preview (with `--limit`)
- Versioning: `version list`, `version create` (with `--message`)
- Lineage: view linked models and training jobs
- Rich table output + `--json` machine-readable mode

#### Dashboard — Datasets Section
- **Datasets List** page with DataTable (name, format, files, size, version, created), create dialog with storage backend selector
- **Dataset Detail** page with metadata, statistics, schema (tabular), preview, files, version history, lineage
- Sidebar navigation with Database icon
- React Query hooks (`useDatasets`, `useDatasetFiles`, `useDatasetVersions`, `useDatasetStats`, `useDatasetPreview`, `useDatasetLineage`)
- Mock data: 5 demo datasets (imagenet-mini, customer-churn, financial-ticks, nlp-reviews, audio-commands)

#### Security Audit
- Path traversal protection in `FilesystemStorage._resolve()` with `is_relative_to()` check
- Filename sanitization (`_sanitize_filename`) strips path components and unsafe characters
- Upload size limit enforcement from `settings.dataset.max_upload_size_mb`
- File extension allowlist from `settings.dataset.allowed_extensions`
- ON DELETE CASCADE on all dataset FK relationships
- Input validation: name length/blank, format, description limits via Pydantic `Field` validators
- Dataset existence checks on list-files, list-versions, get-lineage endpoints
- Pagination on list endpoints (offset/limit with max 1000)
- Size guards: stats/preview skip files > 100 MB, preview limit capped at 500
- Consistent JSON error responses via `HTTPException` (no raw `Response`)
- Validation error handler: sanitize non-serializable `ctx` values

#### Quality
- Platform: 1554 tests (148 dataset tests), 100% coverage
- CLI: 224 tests (32 dataset commands + 7 client methods), 100% coverage
- Dashboard: 68 tests, build clean
- mypy strict — 0 errors across SDK (26), Platform (76), CLI (18) source files
- ruff lint + format — clean
- SDK: 610 tests, 100% coverage

## [0.4.0] — 2026-02-18

### Step 5 — Dashboard (`dashboard/`) — [ai.artenic.ch](https://ai.artenic.ch)

#### Live Demo
- Deployed at **[ai.artenic.ch](https://ai.artenic.ch)** — full demo with realistic mock data, no backend required

#### Pages (9)
- **Overview** — KPI cards, recent activity feed, training status, budget spending chart
- **Model Registry** — list with search/filter, detail view with metadata + promotion history
- **Training Jobs** — list with status badges, detail view with provider info + logs
- **Inference Playground** — interactive JSON editor, model selector, response viewer
- **Ensembles** — list + detail with member models and strategy configuration
- **A/B Tests** — list + detail with variant metrics and statistical results
- **Budgets** — rules management, spending summary, provider spending chart (Recharts)
- **Settings** — schema-driven form with section navigation, runtime hot-reload
- **Health Monitoring** — model health table with drift/error/latency indicators

#### UI & Design
- Dark theme — professional Grafana/Vercel-inspired palette with semantic color tokens
- 15 UI components: Button, Badge, Card, DataTable (sort/pagination), Dialog (a11y: aria-modal, Escape, focus management), EmptyState, ErrorState, Input, JsonEditor, Spinner, StatusDot, Toast (success/error/warning/info, auto-dismiss), DetailRow
- 3 chart types via Recharts: AreaChart, BarChart, LineChart
- Lazy-loaded pages with React.lazy() + Suspense
- 404 catch-all route

#### Architecture
- **Demo mode** — `VITE_DEMO_MODE=true` at api-client level, transparent to hooks/components
- Mock data: 8 models, 12 training jobs, 3 ensembles, 3 A/B tests, 4 budget rules, activity feed, health reports
- TanStack React Query with query key factory pattern
- API client with demo/real mode toggle (same hooks work for both)
- Error states on all pages with retry capability

#### Stack
- React 19, Vite 7, Tailwind CSS 4 (`@theme` syntax), TypeScript 5.7 strict + `noUncheckedIndexedAccess`
- Recharts 3, TanStack React Query 5, Lucide React, React Router 7

#### Quality
- 51 tests (Vitest + @testing-library/react), 6 test suites
- `tsc --noEmit` strict — 0 errors
- Vite build — clean, no warnings
- CI: added `npm test` to dashboard job
- Security audit: 14 corrections applied (useMemo optimizations, shared components extraction, query key consistency, error states, accessibility fixes)

---

## [0.3.0] — 2026-02-17

### Step 4 — CLI (`packages/cli/`)

#### Commands
- 10 command groups covering all platform API endpoints
- `artenic health` — liveness, readiness, detailed health
- `artenic model` — register, list, get, promote, retire
- `artenic predict` / `artenic predict-batch` — single/batch inference
- `artenic training` — dispatch, list, status, cancel jobs
- `artenic budget` — create, list, update rules, check spending
- `artenic ensemble` — create, list, update, train, versions
- `artenic ab-test` — create, list, results, conclude, pause, resume
- `artenic settings` — schema, get, update, audit log
- `artenic config` — local CLI config (show, set, use-profile)

#### Features
- Rich terminal output (tables, dicts) + `--json` machine-readable mode
- TOML multi-profile configuration (`~/.artenic/config.toml`)
- Config precedence: CLI flags > env vars > TOML file > defaults
- Async HTTP client (httpx) with retry-friendly error handling
- Credential masking in output and error messages
- TOML injection prevention, config directory permissions (0o700)
- Input validation: JSON params, tag format, KEY=VALUE pairs

#### Quality
- 159 tests, 100% coverage
- mypy strict — 0 errors across 17 source files
- ruff lint + format — clean
- Security audit: 14 corrections applied (5 critical, 6 high, 3 medium)

---

## [0.2.0] — 2026-02-16

### Step 3 — Platform Service (`packages/platform/`)

#### Step 3.1 — Foundation
- `PlatformSettings` (pydantic-settings) with `ARTENIC_` prefix, all provider configs, nested models for budget/health/canary/ensemble/A/B/webhook/spot
- Async SQLAlchemy engine (`sqlite+aiosqlite` for dev, `postgresql+asyncpg` for prod) with session factory
- 17 ORM tables (initial): models, promotions, training jobs, budgets, alerts, outcomes, ensembles, versions, A/B tests, metrics, health, config settings, audit log
- `SecretManager` — Fernet symmetric encryption with SHA-256 key derivation, dev-mode fallback
- Middleware stack (outer → inner): Correlation ID, Catch-All Errors, CORS, Auth, Rate Limit, Metrics, GZip
- API key authentication with `hmac.compare_digest` constant-time comparison, exempt paths
- Token bucket rate limiting (per-client, configurable burst + rate)
- OpenTelemetry HTTP metrics (request duration histogram)
- Structured JSON logging with correlation ID injection
- Exception → HTTP status mapping (30+ SDK exceptions to proper 4xx/5xx)
- Health endpoints: `/health` (liveness), `/health/ready` (readiness), `/health/detailed` (component health)
- FastAPI app factory with async lifespan (engine, sessions, secrets, MLflow, event bus, health monitor)
- Entry point: `uv run python -m artenic_ai_platform`

#### Step 3.2 — Registry + MLflow + Settings + Plugins
- `ModelRegistry` — register, get, list, promote (stage transitions + audit trail), retire, get_best_model
- `MLflowTracker` — async wrapper with graceful degradation (all methods return `None` when unavailable)
- `SettingsManager` — get/update with encryption, audit log, hot-reload, EventBus publish
- `SettingsSchema` — declarative field metadata (secret flags, restart_required, choices, types)
- Settings API: `/api/v1/settings/{scope}/{section}` (CRUD + schema + audit log)
- `PluginLoader` — entry-point discovery for providers, strategies, services (graceful on import failure)

#### Step 3.3 — Training Orchestrator + Providers + Budget
- `TrainingManager` — dispatch (cost estimate → budget check → MLflow run → provider submit), status, cancel
- `TrainingProvider` protocol with `TrainingSpec`/`JobStatus` dataclasses
- `CloudProvider` base class — full lifecycle: package → upload → provision → execute → monitor → collect → cleanup
- **16 training providers**:
  - **Local**: subprocess-based local training (CPU/GPU)
  - **Hyperscalers**: GCP (Compute Engine + GCS), AWS (EC2 + S3), Azure (VMs + Blob), OCI (Compute + Object Storage)
  - **European**: OVH (OpenStack), Infomaniak (OpenStack), Hetzner (REST API), Scaleway (REST API)
  - **GPU Clouds**: Lambda Labs, RunPod, Vast.ai (marketplace), CoreWeave (Kubernetes)
  - **Generic**: Kubernetes (any cluster), Mock (dev/test)
- Dockerfile for production deployment (multi-stage, non-root, health check)
- `JobPoller` — background loop polling all running jobs, finalize completed/stuck jobs
- `SpotManager` — preemption detection heuristics, automatic retry with failover
- `CostPredictor` — catalog-based instance cost estimation with ranked recommendations
- `OutcomeWriter` — denormalized training outcome persistence for future optimizer integration
- `BudgetManager` — multi-scope (global/service/provider), multi-period (daily/weekly/monthly), enforcement modes (block/warn)
- Budget API: `/api/v1/budgets` (CRUD + spending summary)
- `AlertDispatcher` — webhook alerts with HMAC-SHA256 signing, exponential backoff retry

#### Step 3.4 — Inference + Ensemble + A/B + Health + Events
- `InferenceService` — predict/predict_batch with A/B test routing and health recording
- `PlatformEnsembleManager` — create, update (versioned snapshots), dispatch ensemble training, version history
- `ABTestManager` — create test (variant validation), weighted random selection, metric recording, results aggregation (mean/std/min/max/error_rate/latency), conclude/pause/resume lifecycle
- `HealthMonitor` — background loop with buffered observations, error rate / latency percentiles / confidence drift computation, persist to DB, publish alerts on degraded/unhealthy
- `EventBus` — in-memory async pub/sub with bounded queues (maxsize=256), dead subscriber cleanup
- WebSocket endpoint `/ws` — topic-based subscription, 30s heartbeat, auto-unsubscribe on disconnect
- Inference API: `/api/v1/services/{service}/predict`, `/api/v1/services/{service}/predict_batch`
- Ensemble API: `/api/v1/ensembles` (CRUD + train + versions)
- A/B Testing API: `/api/v1/ab-tests` (CRUD + results + conclude + pause + resume)

#### Quality
- 1406 tests, 100% coverage
- mypy strict mode — 0 errors across 72 source files
- ruff lint + format — clean
- CI pipeline: lint + type-check + test for both SDK and Platform
- Dockerfile for production deployment (multi-stage, non-root, health check)

---

## [0.1.0] — 2026-02-15

### Step 1 — Project Foundation
- Monorepo structure with uv workspace (sdk, platform, cli, optimizer, dashboard)
- Root pyproject.toml with shared dev dependencies (ruff, mypy, pytest)
- justfile with dev commands (setup, check, test, dev-up/down)
- docker-compose.dev.yml for PostgreSQL + MLflow
- Pre-commit hooks configuration
- CI/CD foundation
- Root README.md with project overview

### Step 2 — SDK Core (`packages/sdk/`)

#### Core Contract
- `BaseModel` ABC with Template Method pattern (warmup -> ready -> predict -> teardown)
- 7-state lifecycle (`ModelPhase`: CREATED -> WARMING_UP -> READY -> INFERENCE -> TRAINING -> ERROR -> SHUTDOWN)
- Automatic observability via `ObservabilityMixin`
- Feature validation with `FeatureSchema`

#### Schemas (Pydantic v2)
- `BasePrediction` with auto-generated `prediction_id` (UUID)
- `EnsembleResult`, `TrainResult`, `EvalResult`
- `ModelMetadata` with `model_size_bytes` and `author` fields
- `ModelConfig` with validate_assignment
- `HealthCheckResult`, `DriftDetectionResult`, `ModelHealthReport`
- `EnsembleSnapshot`, `EnsembleABTest`, `AutoPruneResult`, `EvolutionEvent`
- `ConfigEntry` with lifecycle phases

#### Types & Enums (9 StrEnums, 6 type aliases)
- `ModelFramework` — 10 frameworks (PyTorch, TensorFlow, JAX, Transformers, LightGBM, XGBoost, CatBoost, sklearn, ONNX, Custom)
- `SerializationFormat` — 6 formats (safetensors, ONNX, torch, torchscript, pickle, joblib)
- `EnsembleStrategyType` — 5 strategies (weighted_average, dynamic_weighting, meta_learner, majority_voting, stacking)
- `EnsemblePhase`, `ConfigPhase`, `CircuitBreakerState`, `DriftType`, `EvolutionTrigger`, `ModelPhase`

#### Exceptions (30+ custom exception classes)
- Model, Ensemble, Config, Serialization, Platform, Provider, Budget domains
- Rich context: `details` dict, domain-specific attributes (`retry_after`, `failure_count`, `spent_eur`)

#### Ensemble Management
- `EnsembleManager` — parallel inference, quorum, graceful degradation
- 5 strategies: WeightedAverage, DynamicWeighting, MetaLearner, MajorityVoting, Stacking
- Auto-evolution (strategy upgrades based on model count/performance)
- Ensemble versioning (snapshots, restore)
- A/B testing with traffic splitting
- Auto-pruning by weight threshold
- Health reporting with drift detection and recommendations

#### Serialization
- `ModelSerializer` — save/load with metadata.json + config.json
- 6 formats: safetensors, ONNX, PyTorch, TorchScript, Pickle, Joblib
- Lazy imports for optional dependencies
- `list_versions()` for model registry browsing

#### Training Intelligence (8 features, callback-based)
- Smart Data Splitting (holdout, k-fold, stratified, time-series)
- Data Versioning (SHA256 + xxhash fingerprinting)
- Mixed Precision (FP16/BF16, dynamic/static loss scaling)
- Gradient Checkpointing (auto-detection, memory estimation)
- Smart Checkpointing (best-model tracking, SIGTERM preemption handler)
- Early Stopping (patience, divergence detection)
- Learning Rate Finder (Leslie Smith method, gradient smoothing)
- Distributed Training (FSDP/DDP, sharding strategies)

#### Configuration Management
- `ConfigManager` — YAML load/save with Pydantic validation
- `ConfigRegistry` — version tracking, performance association, auto-evolution proposals
- Config lifecycle: DEFAULT -> ACTIVE -> CANDIDATE -> PROMOTED -> RETIRED

#### Observability
- `MetricsCollector` — thread-safe, rolling percentiles (p50, p95, p99)
- `StructuredLogger` — JSON logging (ELK/Datadog compatible)
- `ObservabilityMixin` — automatic predict() instrumentation

#### Decorators (7)
- `@track_inference`, `@validate_input`, `@retry`, `@timeout`
- `@log_lifecycle`, `@circuit_breaker`, `@cache_inference`

#### Client
- `PlatformClient` — async HTTP (httpx) with retry, rate limit handling
- Endpoints: registry, training dispatch, inference, health

#### Testing
- `MockModel` — configurable fake model
- `ModelTestCase` — lifecycle assertion suite
- `create_test_features()` — feature generation from schema

#### Quality
- 611 tests, 100% coverage
- mypy strict mode — 0 errors
- ruff lint + format — clean
- SDK README.md with full API reference

### Step 2.2 — SDK Optimisation (observability, config, decorators, testing, client)

#### Observability Enhancements
- `correlation_context()` — context manager for distributed trace ID propagation (contextvars)
- `get_trace_id()` — retrieve current trace ID from context
- `StructuredLogger` auto-injects trace_id into all log entries
- `MetricsCollector.export_prometheus()` — Prometheus text exposition format
- `MetricsCollector.export_json()` — complete metrics dict with per-model breakdown

#### Configuration Enhancements
- `ConfigManager.load_with_env()` — load YAML then override fields with env vars (`PREFIX_FIELD`)
- `ConfigManager.diff()` — compute added/removed/changed fields between two configs
- `ConfigDiff` — Pydantic schema for config comparison results

#### Exceptions Enhancements
- `error_context()` — context manager to enrich `ArtenicAIError` with metadata

#### Decorators Enhancements
- `@rate_limit` — token bucket rate limiting (max_calls, window_seconds, thread-safe)

#### Testing Enhancements
- `assert_latency_under()` — assert predict() latency below threshold
- `assert_throughput_above()` — assert minimum requests per second
- `assert_prediction_stable()` — assert prediction reproducibility across runs

#### Client Enhancements
- `PlatformClient.predict_batch()` — batch inference via single HTTP call
