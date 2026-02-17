# Artenic AI SDK

Shared contracts, schemas, and ensemble management for the Artenic AI platform.

The SDK is the **leaf package** of the monorepo — it has no internal dependencies and is imported by `platform`, `cli`, and `optimizer`.

## Installation

```bash
pip install artenic-ai-sdk

# With optional dependencies
pip install artenic-ai-sdk[torch]    # PyTorch + safetensors
pip install artenic-ai-sdk[onnx]     # ONNX Runtime
pip install artenic-ai-sdk[sklearn]  # scikit-learn + joblib
pip install artenic-ai-sdk[client]   # httpx (platform client)
```

## Quick Start

### Define a Model

```python
from artenic_ai_sdk import BaseModel, BasePrediction, ModelConfig
from artenic_ai_sdk.types import ModelFramework

class MyModel(BaseModel):
    @property
    def model_id(self) -> str:
        return "my_classifier"

    @property
    def model_version(self) -> str:
        return "0.1.0"

    @property
    def model_type(self) -> str:
        return "lightgbm"

    @property
    def framework(self) -> ModelFramework:
        return ModelFramework.LIGHTGBM

    async def _do_warmup(self) -> None:
        self._model = load_my_model()

    async def predict(self, features: dict) -> BasePrediction:
        score = self._model.predict(features)
        return BasePrediction(
            confidence=score,
            model_id=self.model_id,
            model_version=self.model_version,
            inference_time_ms=1.0,
        )

    # ... implement remaining abstract methods
```

### Ensemble Inference

```python
from artenic_ai_sdk import EnsembleManager, WeightedAverage

manager = EnsembleManager(
    strategy=WeightedAverage(weights={"lgbm": 0.6, "xgb": 0.4}),
)
await manager.register("lgbm", lgbm_model)
await manager.register("xgb", xgb_model)

result = await manager.predict(features)
print(result.confidence, result.strategy_used)
```

### Decorators

```python
from artenic_ai_sdk import track_inference, retry, timeout, circuit_breaker

class MyModel(BaseModel):
    @track_inference
    @retry(max_attempts=3)
    @timeout(seconds=5.0)
    async def predict(self, features):
        ...
```

## API Reference

### Core

| Class | Description |
|-------|-------------|
| `BaseModel` | Abstract base class — the contract every model implements |
| `BasePrediction` | Generic prediction output with confidence, metadata, and tracking ID |
| `ModelConfig` | Base configuration (extended by each model) |
| `PlatformClient` | Async HTTP client for the platform gateway |
| `ModelSerializer` | Save/load model weights in multiple formats |

### Ensemble

| Class | Description |
|-------|-------------|
| `EnsembleManager` | Orchestrates parallel inference, strategy selection, versioning, A/B testing |
| `WeightedAverage` | Fixed-weight combination (Phase 1) |
| `DynamicWeighting` | Performance-adjusted weights with exponential decay (Phase 2) |
| `MetaLearner` | Attention-based meta-learner (Phase 3) |
| `MajorityVoting` | Majority vote for classification ensembles (Phase 4) |
| `Stacking` | Linear meta-model combining base predictions (Phase 5) |
| `EvolutionPolicy` | Auto-evolution configuration for strategy upgrades |

### Decorators

| Decorator | Description |
|-----------|-------------|
| `@track_inference` | Automatic latency + error tracking |
| `@validate_input` | Validate required feature keys |
| `@retry` | Retry with exponential backoff |
| `@timeout` | Async timeout enforcement |
| `@log_lifecycle` | Structured logging for lifecycle methods |
| `@circuit_breaker` | Circuit breaker pattern (closed/open/half-open) |
| `@cache_inference` | LRU + TTL caching for predictions |
| `@rate_limit` | Token bucket rate limiting |

### Supported ML Frameworks

| Framework | Enum Value | Use Case |
|-----------|-----------|----------|
| PyTorch | `PYTORCH` | Deep learning, T-Mamba |
| TensorFlow | `TENSORFLOW` | Deep learning |
| JAX | `JAX` | High-performance ML |
| Transformers | `TRANSFORMERS` | HuggingFace models |
| LightGBM | `LIGHTGBM` | Gradient boosting |
| XGBoost | `XGBOOST` | Gradient boosting |
| CatBoost | `CATBOOST` | Gradient boosting |
| scikit-learn | `SKLEARN` | Classical ML |
| ONNX | `ONNX` | Cross-framework interop |
| Custom | `CUSTOM` | User-defined frameworks |

### Serialization Formats

| Format | Enum Value | Dependency | Use Case |
|--------|-----------|------------|----------|
| SafeTensors | `SAFETENSORS` | `safetensors` | Default — safe, fast tensor serialization |
| ONNX | `ONNX` | `onnxruntime` | Cross-framework deployment |
| PyTorch | `TORCH` | `torch` | Native PyTorch checkpoint |
| TorchScript | `TORCHSCRIPT` | `torch` | Production PyTorch deployment |
| Pickle | `PICKLE` | stdlib | sklearn / XGBoost models |
| Joblib | `JOBLIB` | `joblib` | Efficient sklearn serialization |

### Ensemble Strategies

| Strategy | Enum Value | Models | Description |
|----------|-----------|--------|-------------|
| Weighted Average | `WEIGHTED_AVERAGE` | 1-2 | Fixed manual weights |
| Dynamic Weighting | `DYNAMIC_WEIGHTING` | 3-4 | Performance-adjusted with exponential decay |
| Meta-Learner | `META_LEARNER` | 4+ | Attention-based neural combiner |
| Majority Voting | `MAJORITY_VOTING` | 2+ | Classification vote with tie-breaking |
| Stacking | `STACKING` | 2+ | Linear meta-model (no torch dependency) |

### Observability

| Feature | Description |
|---------|-------------|
| `MetricsCollector` | Thread-safe latency tracking with rolling percentiles (p50/p95/p99) |
| `StructuredLogger` | JSON logging (ELK/Datadog compatible) with auto trace_id injection |
| `ObservabilityMixin` | Automatic predict() instrumentation |
| `correlation_context()` | Context manager for distributed trace ID propagation |
| `get_trace_id()` | Get the current trace ID from context |
| `MetricsCollector.export_prometheus()` | Export metrics in Prometheus text format |
| `MetricsCollector.export_json()` | Export metrics as JSON (per-model breakdown) |

### Configuration

| Feature | Description |
|---------|-------------|
| `ConfigManager.load()` | Load YAML config with Pydantic validation |
| `ConfigManager.load_with_env()` | Load YAML + override fields with env vars (`PREFIX_FIELD`) |
| `ConfigManager.diff()` | Compare two configs — added, removed, changed fields |
| `ConfigRegistry` | Version tracking, performance association, rollback, A/B testing |
| `ConfigDiff` | Schema for config comparison results |

### Exceptions

| Feature | Description |
|---------|-------------|
| 30+ exception classes | Model, Ensemble, Config, Serialization, Platform, Provider, Budget |
| `error_context()` | Context manager to enrich errors with metadata |

### Testing Utilities

| Feature | Description |
|---------|-------------|
| `MockModel` | Configurable fake model for testing |
| `ModelTestCase` | Lifecycle assertion suite |
| `create_test_features()` | Generate features from schema |
| `assert_latency_under()` | Assert predict() latency below threshold |
| `assert_throughput_above()` | Assert minimum requests per second |
| `assert_prediction_stable()` | Assert prediction reproducibility |

## Training Intelligence

The SDK includes a training callback system with:

- **EarlyStopping** — Stop training when validation metric plateaus
- **SmartCheckpointer** — Periodic saves, best-model tracking, spot preemption handling
- **LRFinder** — Automatic learning rate range discovery
- **MixedPrecision** — FP16/BF16 automatic mixed precision
- **GradientCheckpointing** — Memory-efficient training for large models
- **DataVersioning** — xxhash-based dataset fingerprinting
- **DistributedTraining** — Multi-GPU / multi-node coordination
- **DataSplitting** — Time-series aware train/val/test splits

## Development

```bash
# From monorepo root
uv sync --dev

# Run SDK tests
uv run pytest packages/sdk/tests/ -v

# Quality checks
uv run ruff check packages/sdk/
uv run ruff format --check packages/sdk/
uv run mypy packages/sdk/

# Coverage (100% required)
uv run pytest packages/sdk/tests/ --cov=artenic_ai_sdk --cov-fail-under=100
```

## License

Apache License 2.0 — see [LICENSE](../../LICENSE) for details.
