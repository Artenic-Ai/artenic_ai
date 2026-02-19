# Artenic Optimizer — Roadmap

## Overview

The optimizer is built incrementally across 5 phases. Each phase is self-contained, tested, and deployable. Phases build on each other but earlier phases remain functional independently.

```
Phase 1          Phase 2          Phase 3          Phase 4          Phase 5
Foundation       LTR + Loop       Storage          Cost             Dashboard
+ Heuristic      + Platform       Optimizer        Analyzer         + CLI
v0.1.0           v0.2.0           v0.3.0           v0.4.0           v0.5.0
───────────────▶───────────────▶───────────────▶───────────────▶───────────────▶
```

---

## Phase 1 — Foundation + Heuristic Ranker (v0.1.0)

**Goal**: Establish the optimizer's core types, schemas, and a functional rule-based training advisor.

### Features

- [ ] `types.py` — Enums: `OptimizerMode`, `StorageTier`, `RecommendationType`, `OptimizationGoal`
- [ ] `schemas.py` — Pydantic v2 models: `WorkloadProfile`, `InstanceRecommendation`, `TrainingAdvice`, `StorageAdvice`, `CostReport`, `OptimizerStatus`
- [ ] `_internal/math_utils.py` — NDCG computation, min-max normalization, score aggregation
- [ ] `training_advisor/feature_extractor.py` — Workload + instance → ~25 feature vector
- [ ] `training_advisor/heuristic_ranker.py` — Rule-based ranking with configurable weights (cost 40%, GPU 30%, memory 20%, reliability 10%)
- [ ] `training_advisor/advisor.py` — Orchestrator (heuristic-only mode)
- [ ] `__init__.py` — Public API exports
- [ ] Full test suite for all Phase 1 modules (90%+ coverage)

### Files to Create

```
src/artenic_optimizer/
├── types.py
├── schemas.py
├── _internal/
│   ├── __init__.py
│   └── math_utils.py
└── training_advisor/
    ├── __init__.py
    ├── feature_extractor.py
    ├── heuristic_ranker.py
    └── advisor.py

tests/
├── conftest.py
├── test_types.py
├── test_schemas.py
├── test_internal/
│   └── test_math_utils.py
└── training_advisor/
    ├── test_feature_extractor.py
    ├── test_heuristic_ranker.py
    └── test_advisor.py
```

### Files to Modify

- `pyproject.toml` — add `scikit-learn>=1.5` dependency
- `__init__.py` — update with public exports

### Dependencies

- None (first phase)

### Success Criteria

- [ ] `ruff check` and `mypy` pass with zero errors
- [ ] Tests pass with 90%+ coverage
- [ ] `TrainingAdvisor.recommend(workload, instances)` returns ranked `InstanceRecommendation` list
- [ ] Heuristic scoring produces stable, deterministic rankings
- [ ] Feature extraction handles all `ModelFramework` variants

---

## Phase 2 — LTR + Feedback Loop + Platform Integration (v0.2.0)

**Goal**: Add the LightGBM LambdaMART ranker, close the feedback loop, and wire the optimizer into the platform API.

### Features

- [ ] `training_advisor/ltr_ranker.py` — LightGBM LambdaMART (`LGBMRanker`)
  - `train(samples)` — train on `OptimizerTrainingSampleRecord` data
  - `predict(workload, instances)` — rank instances
  - `evaluate()` — NDCG@5, NDCG@10, MRR
  - `save(path)` / `load(path)` — model persistence (joblib)
  - Auto-retrain when new samples > threshold
  - Minimum 50 samples to activate
- [ ] `training_advisor/feedback_collector.py` — Outcome → relevance labels
  - Relevance scale: 0 (failed), 1 (poor), 2 (adequate), 3 (good), 4 (optimal)
  - Computes labels from: success, cost ratio, duration ratio, metric improvement
  - Groups samples by workload hash (`query_id`)
  - Writes `OptimizerTrainingSampleRecord` to DB
- [ ] Advisor updated: supports `HEURISTIC`, `LTR`, `HYBRID` modes
  - Hybrid: LTR score * 0.7 + heuristic score * 0.3
  - Budget-aware filtering
  - Writes `OptimizerRecommendationRecord` to DB
- [ ] Platform integration:
  - `platform/optimizer/service.py` — thin service layer
  - `platform/optimizer/router.py` — `/api/v1/optimizer` endpoints (recommend, status, history, retrain, metrics)
  - `platform/settings.py` — `OptimizerConfig` sub-config
  - `platform/app.py` — wire optimizer router in lifespan

### Files to Create

```
src/artenic_optimizer/
└── training_advisor/
    ├── ltr_ranker.py
    └── feedback_collector.py

# In the platform package:
packages/platform/src/artenic_ai_platform/optimizer/
├── __init__.py
├── service.py
└── router.py
```

### Files to Modify

- `packages/optimizer/src/artenic_optimizer/training_advisor/advisor.py` — add LTR/hybrid modes
- `packages/platform/src/artenic_ai_platform/settings.py` — add `OptimizerConfig`
- `packages/platform/src/artenic_ai_platform/app.py` — import and wire router

### Dependencies

- Phase 1 (types, schemas, feature extractor, heuristic ranker)

### Success Criteria

- [ ] LTR ranker trains on synthetic data and produces valid rankings
- [ ] Feedback collector correctly maps outcomes to relevance labels 0-4
- [ ] Auto-retrain triggers when sample threshold exceeded
- [ ] `/api/v1/optimizer/recommend` returns ranked recommendations
- [ ] `/api/v1/optimizer/status` shows mode, sample count, model version
- [ ] `/api/v1/optimizer/metrics` returns NDCG@5, MRR when LTR is active
- [ ] All existing platform tests still pass
- [ ] 90%+ coverage on new code

---

## Phase 3 — Storage Optimizer (v0.3.0)

**Goal**: Analyze platform datasets and recommend optimal storage strategies.

### Features

- [ ] `storage_optimizer/usage_analyzer.py` — Dataset usage analysis
  - Query `DatasetRecord` + `DatasetFileRecord` for size/format distribution
  - Track access patterns from API logs (read/download frequency)
  - Identify growth trends (version-over-version size changes)
- [ ] `storage_optimizer/format_advisor.py` — Format recommendations
  - CSV > 100MB → Parquet (3-5x compression)
  - JSON (tabular) > 50MB → Parquet or JSONL (2-4x compression)
  - Small datasets < 10MB → keep as-is
  - Estimate compression ratio and read performance improvement
- [ ] `storage_optimizer/tiering_engine.py` — Hot/warm/cold classification
  - Hot: accessed in last 7 days or > 4x/month → local/SSD
  - Warm: accessed in last 30 days or 1-4x/month → S3/GCS standard
  - Cold: 90+ days since access, < 1x/month → Glacier/Archive
- [ ] `storage_optimizer/optimizer.py` — Orchestrator
  - `analyze(dataset_id)` → `StorageAdvice`
  - `analyze_all()` → bulk recommendations
  - Estimate cost savings from format conversion + tiering
- [ ] Platform endpoints:
  - `POST /api/v1/optimizer/storage/analyze` — analyze a specific dataset
  - `GET /api/v1/optimizer/storage/recommendations` — all recommendations

### Files to Create

```
src/artenic_optimizer/
└── storage_optimizer/
    ├── __init__.py
    ├── usage_analyzer.py
    ├── format_advisor.py
    ├── tiering_engine.py
    └── optimizer.py

tests/
└── storage_optimizer/
    ├── test_usage_analyzer.py
    ├── test_format_advisor.py
    ├── test_tiering_engine.py
    └── test_optimizer.py
```

### Dependencies

- Phase 1 (types, schemas)
- Platform datasets module (DatasetRecord, DatasetFileRecord)

### Success Criteria

- [ ] Usage analyzer correctly aggregates dataset metadata
- [ ] Format advisor recommends Parquet for large CSV/JSON datasets
- [ ] Tiering engine classifies datasets based on access patterns
- [ ] Storage optimizer produces actionable `StorageAdvice` with savings estimates
- [ ] Platform endpoints return valid responses
- [ ] 90%+ coverage

---

## Phase 4 — Cost Analyzer (v0.4.0)

**Goal**: Real-time multi-provider cost intelligence and spending insights.

### Features

- [ ] `cost_analyzer/provider_benchmark.py` — Provider comparison
  - Wrap `CostPredictor` with richer analysis
  - Side-by-side comparison: price/hour, GPU availability, region coverage
  - Pareto-optimal frontier: cost vs speed
  - GPU availability scoring per provider/region
- [ ] `cost_analyzer/spot_advisor.py` — Spot instance intelligence
  - Analyze historical preemption data from `TrainingJob.preempted` / `preemption_count`
  - Compute: avg savings %, preemption rate, expected total cost with retries
  - Recommend spot when: savings > 30% AND preemption rate < 20% AND checkpointing enabled
  - ROI: `spot_cost + (preemption_rate * retry_overhead)` vs `on_demand_cost`
- [ ] `cost_analyzer/historical_analyzer.py` — Spending trends
  - Trends by provider, instance type, time period
  - Cost efficiency: cost per metric improvement over time
  - Anomaly detection: flag unusually expensive runs
  - Savings report: actual spend vs optimizer-recommended spend
- [ ] `cost_analyzer/analyzer.py` — Orchestrator
  - `compare_providers(workload)` → `CostReport`
  - `spending_summary(period)` → historical analysis
  - `savings_opportunity()` → actionable recommendations
- [ ] Platform endpoints:
  - `GET /api/v1/optimizer/cost/compare` — provider comparison for a workload
  - `GET /api/v1/optimizer/cost/summary` — historical spending summary
  - `GET /api/v1/optimizer/cost/savings` — savings opportunities

### Files to Create

```
src/artenic_optimizer/
└── cost_analyzer/
    ├── __init__.py
    ├── provider_benchmark.py
    ├── spot_advisor.py
    ├── historical_analyzer.py
    └── analyzer.py

tests/
└── cost_analyzer/
    ├── test_provider_benchmark.py
    ├── test_spot_advisor.py
    ├── test_historical_analyzer.py
    └── test_analyzer.py
```

### Dependencies

- Phase 1 (types, schemas)
- Platform: `CostPredictor`, `TrainingJob`, `BudgetRecord`

### Success Criteria

- [ ] Provider benchmark returns Pareto-optimal recommendations
- [ ] Spot advisor correctly evaluates risk/reward based on historical data
- [ ] Historical analyzer identifies spending anomalies
- [ ] Savings report provides actionable data
- [ ] 90%+ coverage

---

## Phase 5 — Dashboard + CLI + Polish (v0.5.0)

**Goal**: Full visibility of the optimizer in the dashboard and CLI.

### Features

- [ ] Dashboard: new **Optimizer** page (read-only in demo mode)
  - Status card: mode (heuristic/LTR/hybrid), sample count, NDCG@5
  - Recommendation history: sortable table with outcome column
  - Provider comparison: bar chart (cost/hour by provider for a workload)
  - Storage recommendations: table with format/tiering suggestions
  - Cost trends: line chart (spending over time by provider)
  - Savings report: estimated savings vs actual
- [ ] Dashboard: mock data for demo mode (optimizer recommendations, cost data)
- [ ] CLI commands:
  - `artenic optimizer status` — show optimizer mode and metrics
  - `artenic optimizer recommend` — get recommendations for a workload
  - `artenic optimizer cost compare` — provider cost comparison
  - `artenic optimizer cost summary` — spending summary
  - `artenic optimizer storage analyze` — storage recommendations
  - `artenic optimizer retrain` — trigger LTR retrain
- [ ] Performance benchmarks: feature extraction throughput, ranking latency
- [ ] End-to-end integration test: dispatch → outcome → feedback → retrain → improved ranking

### Dependencies

- All previous phases
- Dashboard patterns (React Query hooks, mock handlers)
- CLI patterns (Click commands, Rich output)

### Success Criteria

- [ ] Dashboard optimizer page renders correctly in demo mode
- [ ] All CLI commands work and produce valid output
- [ ] Feature extraction: < 1ms per (workload, instance) pair
- [ ] Ranking: < 10ms for 100 instances
- [ ] End-to-end feedback loop verified with synthetic data
- [ ] All CI checks pass (lint, types, tests across Python 3.12 & 3.13, dashboard)

---

## Future Considerations (Post v0.5.0)

These are not planned but may be explored after the 5 phases are complete:

- **GPU Memory Estimator**: Static analysis of model architecture to predict peak GPU memory (inspired by DNNMem/xMem research)
- **Hyperparameter-Aware Recommendations**: Factor in hyperparameter search space size when recommending instances
- **Multi-Job Scheduling**: Optimize instance allocation across multiple concurrent training jobs
- **LoRA/QLoRA Advisor**: Recommend parameter-efficient fine-tuning when full fine-tuning cost exceeds threshold
- **Predictive Preemption**: ML model to predict spot instance preemption probability
- **Carbon-Aware Scheduling**: Factor in energy source and carbon intensity when selecting regions
- **Provider SLA Monitoring**: Track provider reliability, latency, and uptime over time
