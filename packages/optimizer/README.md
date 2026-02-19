# Artenic Optimizer

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](../../LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://python.org)
[![CI](https://github.com/Artenic-Ai/artenic_ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Artenic-Ai/artenic_ai/actions)

**Smart Agent for ML Resource Optimization** — an intelligent, self-improving optimization engine that helps you choose the best infrastructure for your ML workloads.

Part of the [Artenic AI Platform](https://github.com/Artenic-Ai/artenic_ai) monorepo.

---

## Vision

The Artenic Optimizer acts as an intelligent assistant integrated into the platform. Instead of manually choosing providers, instance types, and storage backends, the optimizer analyzes your workload and historical data to make recommendations that minimize cost, maximize performance, or find the optimal balance.

The optimizer is built on **three pillars**:

| Pillar | Role | Approach |
|--------|------|----------|
| **Training Advisor** | Select the best provider + instance for a training job | Learning-to-Rank (LightGBM LambdaMART) with heuristic fallback |
| **Storage Optimizer** | Recommend storage backends, formats, and tiering for datasets | Rule-based analysis of dataset access patterns and characteristics |
| **Cost Analyzer** | Real-time multi-provider cost intelligence and spending insights | Live pricing comparison, spot analysis, historical trend detection |

Results are visible in the platform dashboard (read-only in demo mode), providing a public showcase of the optimizer's capabilities without exposing private model data.

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Artenic AI Platform                          │
│                                                                     │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │   Training    │───▶│  OutcomeWriter   │───▶│  Training        │  │
│  │   Manager     │    │  (feedback loop) │    │  OutcomeRecord   │  │
│  └──────┬───────┘    └──────────────────┘    └────────┬─────────┘  │
│         │                                             │            │
│         │ dispatch request                            │ outcomes   │
│         ▼                                             ▼            │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    ARTENIC OPTIMIZER                          │  │
│  │                                                              │  │
│  │  ┌────────────────┐ ┌──────────────────┐ ┌───────────────┐  │  │
│  │  │   Training     │ │    Storage       │ │    Cost       │  │  │
│  │  │   Advisor      │ │    Optimizer     │ │    Analyzer   │  │  │
│  │  │                │ │                  │ │               │  │  │
│  │  │ FeatureExtract │ │ UsageAnalyzer    │ │ ProviderBench │  │  │
│  │  │ HeuristicRank  │ │ FormatAdvisor    │ │ SpotAdvisor   │  │  │
│  │  │ LTR Ranker     │ │ TieringEngine    │ │ HistAnalyzer  │  │  │
│  │  │ FeedbackCollect│ │                  │ │               │  │  │
│  │  └───────┬────────┘ └────────┬─────────┘ └───────┬───────┘  │  │
│  │          │                   │                    │          │  │
│  └──────────┼───────────────────┼────────────────────┼──────────┘  │
│             │                   │                    │              │
│             ▼                   ▼                    ▼              │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     /api/v1/optimizer                         │  │
│  └──────────────────────────────────────────────────────────────┘  │
│             │                   │                    │              │
└─────────────┼───────────────────┼────────────────────┼──────────────┘
              ▼                   ▼                    ▼
       ┌──────────────────────────────────────────────────────┐
       │              Dashboard (read-only demo)               │
       │  Optimizer page: status, recommendations, cost charts │
       └──────────────────────────────────────────────────────┘
```

### Data Flow

```
Training Job Dispatch
    │
    ├──▶ CostPredictor.predict_cost()        ← live provider pricing
    │         │
    │         ▼
    ├──▶ Optimizer.recommend()               ← ranked instance recommendations
    │         │
    │         ├── FeatureExtractor            ← workload + instance → feature vector
    │         ├── HeuristicRanker (Phase 1)   ← rule-based scoring
    │         └── LTR Ranker (Phase 2)        ← LightGBM LambdaMART
    │         │
    │         ▼
    │    OptimizerRecommendationRecord        ← persisted recommendation
    │
    ▼
Training Execution (on selected provider)
    │
    ▼
Job Completion
    │
    ├──▶ OutcomeWriter.write_outcome()        ← captures actual cost/duration
    │         │
    │         ▼
    │    TrainingOutcomeRecord                ← ground truth
    │         │
    │         ▼
    └──▶ FeedbackCollector.collect()          ← computes relevance label (0-4)
              │
              ▼
         OptimizerTrainingSampleRecord        ← LTR training data
              │
              ▼
         LTR Ranker auto-retrain              ← model improves over time
```

---

## Module Structure

```
packages/optimizer/
├── pyproject.toml
├── README.md
├── ROADMAP.md
├── src/artenic_optimizer/
│   ├── __init__.py                         # Public API exports
│   ├── types.py                            # Enums (OptimizerMode, StorageTier, etc.)
│   ├── schemas.py                          # Pydantic v2 request/response models
│   │
│   ├── training_advisor/                   # Pillar 1: Instance selection
│   │   ├── __init__.py
│   │   ├── feature_extractor.py            # Workload + instance → feature vector
│   │   ├── heuristic_ranker.py             # Rule-based fallback (< 50 samples)
│   │   ├── ltr_ranker.py                   # LightGBM LambdaMART ranker
│   │   ├── feedback_collector.py           # Outcome → relevance labels (0-4)
│   │   └── advisor.py                      # Orchestrator: heuristic / LTR / hybrid
│   │
│   ├── storage_optimizer/                  # Pillar 2: Storage intelligence
│   │   ├── __init__.py
│   │   ├── usage_analyzer.py               # Dataset access patterns, growth trends
│   │   ├── format_advisor.py               # CSV→Parquet recommendations
│   │   ├── tiering_engine.py               # Hot/warm/cold classification
│   │   └── optimizer.py                    # Orchestrator
│   │
│   ├── cost_analyzer/                      # Pillar 3: Cost intelligence
│   │   ├── __init__.py
│   │   ├── provider_benchmark.py           # Real-time provider comparison
│   │   ├── spot_advisor.py                 # Spot vs on-demand decision engine
│   │   ├── historical_analyzer.py          # Spending trends, savings detection
│   │   └── analyzer.py                     # Orchestrator
│   │
│   └── _internal/                          # Shared utilities
│       ├── __init__.py
│       └── math_utils.py                   # NDCG, normalization, scoring
│
└── tests/
    ├── conftest.py
    ├── test_types.py
    ├── test_schemas.py
    ├── training_advisor/
    │   ├── test_feature_extractor.py
    │   ├── test_heuristic_ranker.py
    │   ├── test_ltr_ranker.py
    │   ├── test_feedback_collector.py
    │   └── test_advisor.py
    ├── storage_optimizer/
    │   ├── test_usage_analyzer.py
    │   ├── test_format_advisor.py
    │   ├── test_tiering_engine.py
    │   └── test_optimizer.py
    ├── cost_analyzer/
    │   ├── test_provider_benchmark.py
    │   ├── test_spot_advisor.py
    │   ├── test_historical_analyzer.py
    │   └── test_analyzer.py
    └── test_internal/
        └── test_math_utils.py
```

---

## Pillar 1: Training Advisor

The Training Advisor is the core intelligence of the optimizer. Given a workload description and a list of available instances, it returns a ranked list of recommendations.

### Feature Extraction

The `FeatureExtractor` converts (workload, instance) pairs into numerical feature vectors for ranking:

**Workload features** (~10):
- `dataset_size_gb` — total dataset size
- `model_params_m` — model parameter count (millions)
- `num_epochs` — training epochs
- `batch_size` — batch size
- `framework_id` — encoded framework (PyTorch=0, TF=1, JAX=2, etc.)
- `gpu_required` — boolean flag
- `precision_id` — encoded precision (fp32=0, fp16=1, bf16=2)
- `num_workers` — data loading workers
- `distributed` — multi-node flag
- `estimated_hours` — user-provided estimate

**Instance features** (~8):
- `vcpus` — virtual CPU count
- `memory_gb` — RAM
- `gpu_count` — number of GPUs
- `gpu_type_id` — encoded GPU type (A100=0, H100=1, T4=2, etc.)
- `price_per_hour` — on-demand price (EUR)
- `spot_price_per_hour` — spot price (EUR, 0 if unavailable)
- `is_spot` — spot flag
- `provider_id` — encoded provider

**Cross features** (~6):
- `memory_per_param_ratio` — memory_gb / model_params_m
- `cost_per_gpu_hour` — price / gpu_count
- `gpu_memory_headroom` — estimated GPU memory margin
- `spot_discount_pct` — (on_demand - spot) / on_demand
- `vcpu_per_worker` — vcpus / num_workers
- `cost_efficiency_score` — composite heuristic score

### Heuristic Ranker (Phase 1)

Rule-based ranking used when fewer than 50 training samples are available:

1. **Filter**: Remove instances that don't meet minimum requirements (GPU, memory)
2. **Score**: Weighted combination:
   - Cost efficiency: 40%
   - GPU match: 30%
   - Memory headroom: 20%
   - Provider reliability: 10% (from historical success rate, defaults to 1.0)
3. **Sort**: Descending by score
4. **Return**: Top-N `InstanceRecommendation` objects

### LTR Ranker (Phase 2)

LightGBM LambdaMART model trained on historical training outcomes:

- **Algorithm**: LambdaMART (listwise Learning-to-Rank)
- **Library**: LightGBM (`lightgbm.LGBMRanker`)
- **Training data**: `OptimizerTrainingSampleRecord` entries grouped by `query_id`
- **Minimum samples**: 50 (configurable) before activation
- **Auto-retrain**: Triggered when new samples exceed threshold
- **Evaluation metrics**: NDCG@5, NDCG@10, MRR

### Feedback Loop

The `FeedbackCollector` converts training outcomes into LTR training samples:

**Relevance labels** (0-4 scale):
| Label | Meaning | Criteria |
|-------|---------|----------|
| 0 | Failed | Training job failed |
| 1 | Poor | Success but >2x predicted cost or >2x predicted duration |
| 2 | Adequate | Success, within 1.5-2x of predictions |
| 3 | Good | Success, within 1-1.5x of predictions |
| 4 | Optimal | Success, at or below predicted cost and duration |

### Advisor Modes

| Mode | Condition | Behavior |
|------|-----------|----------|
| `HEURISTIC` | < 50 samples | Pure rule-based ranking |
| `LTR` | >= 50 samples, trained model | Pure LightGBM ranking |
| `HYBRID` | >= 50 samples | LTR score * 0.7 + heuristic score * 0.3 |

---

## Pillar 2: Storage Optimizer

Analyzes platform datasets (`DatasetRecord`) to recommend optimal storage strategies.

### Usage Analyzer
- Queries dataset metadata: size, format, file count, version history
- Tracks access patterns: read/download frequency from API logs
- Identifies growth trends: version-over-version size changes

### Format Advisor
| Current Format | Size Threshold | Recommendation | Expected Benefit |
|---------------|----------------|----------------|-----------------|
| CSV | > 100 MB | Parquet | 3-5x compression, faster reads |
| JSON (tabular) | > 50 MB | Parquet or JSONL | 2-4x compression |
| Any | < 10 MB | Keep as-is | Conversion overhead not worth it |

### Tiering Engine

Classifies datasets into storage tiers:

| Tier | Criteria | Recommended Backend |
|------|----------|-------------------|
| **Hot** | Accessed in last 7 days, or > 4x/month | Local filesystem, fast SSD |
| **Warm** | Accessed in last 30 days, or 1-4x/month | S3/GCS standard |
| **Cold** | Not accessed in 90+ days, < 1x/month | S3 Glacier, Azure Archive |

---

## Pillar 3: Cost Analyzer

Real-time cost intelligence across all configured providers.

### Provider Benchmark
- Queries live pricing via `CostPredictor` for all enabled providers
- Side-by-side comparison: price/hour, GPU availability, region coverage
- Pareto-optimal frontier: identifies instances that are best for cost, speed, or balanced

### Spot Advisor
- Analyzes historical spot preemption data from `TrainingJob` records
- Recommends spot when: savings > 30% AND preemption rate < 20% AND checkpointing enabled
- ROI calculation: `spot_cost + (preemption_rate * retry_overhead)` vs `on_demand_cost`

### Historical Analyzer
- Spending trends by provider, instance type, time period
- Cost efficiency: cost per metric improvement over time
- Anomaly detection: flags unusually expensive training runs
- Savings report: actual spend vs what optimizer would have recommended

---

## Platform Integration

### Existing Infrastructure (Already Built)

The platform provides all the data structures the optimizer needs:

| Component | Location | Role |
|-----------|----------|------|
| `TrainingOutcomeRecord` | `platform/db/models.py` | Ground truth for completed jobs |
| `OptimizerRecommendationRecord` | `platform/db/models.py` | Persisted recommendations |
| `OptimizerTrainingSampleRecord` | `platform/db/models.py` | LTR training data |
| `CostPredictor` | `platform/training/cost_predictor.py` | Live provider pricing |
| `OutcomeWriter` | `platform/training/outcome_writer.py` | Feedback loop bridge |
| 16 provider configs | `platform/settings.py` | GCP, AWS, Azure, Lambda Labs, RunPod, etc. |

### API Endpoints (Planned)

```
GET  /api/v1/optimizer/status                    → Optimizer status and metrics
POST /api/v1/optimizer/recommend                 → Training instance recommendations
GET  /api/v1/optimizer/history                   → Recommendation history (paginated)
POST /api/v1/optimizer/storage/analyze           → Storage advice for a dataset
GET  /api/v1/optimizer/storage/recommendations   → All storage recommendations
GET  /api/v1/optimizer/cost/compare              → Provider cost comparison
GET  /api/v1/optimizer/cost/summary              → Historical spending summary
GET  /api/v1/optimizer/cost/savings              → Savings opportunities
POST /api/v1/optimizer/retrain                   → Trigger manual LTR retrain
GET  /api/v1/optimizer/metrics                   → NDCG, MRR, sample count
```

### Dashboard Page (Planned)

Read-only page in the demo dashboard:
- **Status card**: optimizer mode (heuristic/LTR/hybrid), sample count, model accuracy
- **Recommendation history**: table of past recommendations with outcomes
- **Provider comparison**: bar chart comparing cost/performance across providers
- **Storage recommendations**: table with format/tiering suggestions per dataset
- **Cost trends**: line chart showing spending over time
- **Savings report**: estimated vs actual savings

---

## Tech Stack

| Dependency | Version | Purpose |
|-----------|---------|---------|
| `artenic-ai-sdk` | workspace | Core abstractions, schemas, types |
| `lightgbm` | >= 4.5 | LambdaMART Learning-to-Rank model |
| `numpy` | >= 1.26 | Numerical operations, feature vectors |
| `scikit-learn` | >= 1.5 | NDCG scoring, train/test split, metrics |

---

## Development

### Install

```bash
uv sync --dev --package artenic-optimizer
```

### Test

```bash
pytest packages/optimizer/tests/ -v --cov=artenic_optimizer --cov-fail-under=90
```

### Lint & Type Check

```bash
ruff check packages/optimizer/
mypy packages/optimizer/
```

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the detailed phased implementation plan.

---

## Inspired By

The optimizer design draws from research and proven patterns in:

- **[SkyPilot](https://github.com/skypilot-org/skypilot)** — multi-cloud provider selection and spot instance management
- **[LightGBM LambdaMART](https://lightgbm.readthedocs.io/)** — state-of-the-art Learning-to-Rank algorithm
- **[DNNMem](https://www.microsoft.com/en-us/research/publication/estimating-gpu-memory-consumption-of-deep-neural-networks/)** — GPU memory estimation for neural networks
- **[Optuna](https://optuna.org/)** — hyperparameter optimization with pruning
- **[Ray Tune](https://docs.ray.io/en/latest/tune/)** — distributed hyperparameter search with cost-awareness

---

## License

Apache License 2.0 — see [LICENSE](../../LICENSE) for details.
