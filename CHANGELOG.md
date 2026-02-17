# Changelog

All notable changes to this project will be documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] — 2026-02-16

### Step 3 — Platform Service (`packages/platform/`)

#### Step 3.1 — Foundation
- `PlatformSettings` (pydantic-settings) with `ARTENIC_` prefix, all provider configs, nested models for budget/health/canary/ensemble/A/B/webhook/spot
- Async SQLAlchemy engine (`sqlite+aiosqlite` for dev, `postgresql+asyncpg` for prod) with session factory
- 17 ORM tables: models, promotions, training jobs, budgets, alerts, outcomes, ensembles, versions, A/B tests, metrics, health, config settings, audit log
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
