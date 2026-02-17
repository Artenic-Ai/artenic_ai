"""Tests for artenic_ai_sdk.testing — MockModel, ModelTestCase, create_test_features."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from artenic_ai_sdk.schemas import BasePrediction, FeatureSchema
from artenic_ai_sdk.testing import (
    MockModel,
    ModelTestCase,
    assert_latency_under,
    assert_prediction_stable,
    assert_throughput_above,
    create_test_features,
)
from artenic_ai_sdk.types import ModelFramework, ModelPhase

if TYPE_CHECKING:
    from pathlib import Path


# ===================================================================
# MockModel
# ===================================================================


class TestMockModel:
    async def test_properties(self) -> None:
        model = MockModel(mock_id="test_m", mock_version="1.0.0", mock_type="lgbm")
        assert model.model_id == "test_m"
        assert model.model_version == "1.0.0"
        assert model.model_type == "lgbm"
        assert model.framework == ModelFramework.PYTORCH

    async def test_predict_with_latency(self) -> None:
        """MockModel with positive latency triggers asyncio.sleep."""
        model = MockModel(mock_confidence=0.85, mock_latency_ms=10.0)
        await model.warmup()
        pred = await model.predict({"x": 1.0})
        assert isinstance(pred, BasePrediction)
        assert pred.confidence == 0.85
        assert pred.inference_time_ms == 10.0

    async def test_predict(self) -> None:
        model = MockModel(mock_confidence=0.9, mock_latency_ms=0.0)
        await model.warmup()
        pred = await model.predict({"a": 1.0, "b": 2.0})
        assert isinstance(pred, BasePrediction)
        assert pred.confidence == 0.9
        assert pred.model_id == "mock_model"
        assert "a" in pred.metadata["features_received"]
        assert "b" in pred.metadata["features_received"]

    async def test_predict_call_count(self) -> None:
        model = MockModel(mock_latency_ms=0.0)
        await model.warmup()
        assert model.predict_call_count == 0
        await model.predict({"x": 1})
        await model.predict({"x": 2})
        assert model.predict_call_count == 2

    async def test_predict_raises_on_mock_error(self) -> None:
        err = ValueError("mock failure")
        model = MockModel(mock_error=err, mock_latency_ms=0.0)
        await model.warmup()
        with pytest.raises(ValueError, match="mock failure"):
            await model.predict({"x": 1})

    async def test_warmup_and_phase(self) -> None:
        model = MockModel()
        assert model.phase == ModelPhase.CREATED
        await model.warmup()
        assert model.phase == ModelPhase.READY

    async def test_preprocess(self) -> None:
        model = MockModel()
        result = await model.preprocess({"raw": True})
        assert result == {"raw": True}

    async def test_train(self) -> None:
        from artenic_ai_sdk.schemas import ModelConfig

        model = MockModel(mock_id="trainer")
        result = await model.train([], ModelConfig())
        assert result.model_name == "trainer"
        assert result.metrics["accuracy"] == 0.9

    async def test_evaluate(self) -> None:
        model = MockModel(mock_id="evaluator")
        result = await model.evaluate([])
        assert result.model_name == "evaluator"
        assert result.metrics["accuracy"] == 0.85

    async def test_save_load(self, tmp_path: Path) -> None:
        model = MockModel(mock_id="saver", mock_version="2.0")
        saved = await model.save(tmp_path / "model_dir")
        assert saved.exists()
        assert (saved / "mock_model.txt").read_text(encoding="utf-8") == "saver:2.0"
        # Load should not raise
        await model.load(saved)

    async def test_get_metadata(self) -> None:
        model = MockModel(mock_id="meta_m", mock_version="3.0", mock_type="xgb")
        meta = model.get_metadata()
        assert meta.name == "meta_m"
        assert meta.version == "3.0"
        assert meta.model_type == "xgb"
        assert meta.framework == ModelFramework.PYTORCH

    async def test_teardown(self) -> None:
        model = MockModel()
        await model.warmup()
        await model.teardown()
        assert model.phase == ModelPhase.SHUTDOWN


# ===================================================================
# ModelTestCase
# ===================================================================


class TestModelTestCase:
    async def test_assert_predict_returns_valid_output(self) -> None:
        model = MockModel(mock_confidence=0.75, mock_latency_ms=0.0)
        await model.warmup()
        tc = ModelTestCase()
        pred = await tc.assert_predict_returns_valid_output(model, {"x": 1})
        assert pred.confidence == 0.75

    async def test_assert_warmup_transitions_to_ready(self) -> None:
        model = MockModel()
        tc = ModelTestCase()
        await tc.assert_warmup_transitions_to_ready(model)
        assert model.phase == ModelPhase.READY

    async def test_assert_save_load_roundtrip(self, tmp_path: Path) -> None:
        model = MockModel(mock_latency_ms=0.0)
        tc = ModelTestCase()
        await tc.assert_save_load_roundtrip(model, tmp_path)

    async def test_assert_health_check_healthy(self) -> None:
        model = MockModel()
        await model.warmup()
        tc = ModelTestCase()
        await tc.assert_health_check_healthy(model)

    async def test_assert_batch_matches_individual(self) -> None:
        model = MockModel(mock_confidence=0.8, mock_latency_ms=0.0)
        await model.warmup()
        tc = ModelTestCase()
        batch = [{"x": 1}, {"x": 2}, {"x": 3}]
        await tc.assert_batch_matches_individual(model, batch)

    async def test_assert_full_lifecycle(self) -> None:
        model = MockModel(mock_latency_ms=0.0)
        tc = ModelTestCase()
        await tc.assert_full_lifecycle(model, {"feature_a": 1.0})
        assert model.phase == ModelPhase.SHUTDOWN


# ===================================================================
# create_test_features
# ===================================================================


class TestCreateTestFeatures:
    def test_basic_dtypes(self) -> None:
        schema = [
            FeatureSchema(name="f1", dtype="float32", required=True),
            FeatureSchema(name="f2", dtype="int64", required=True),
            FeatureSchema(name="f3", dtype="bool", required=False),
            FeatureSchema(name="f4", dtype="string", required=False),
        ]
        features = create_test_features(schema)
        assert features["f1"] == 0.0
        assert features["f2"] == 0
        assert features["f3"] is False
        assert features["f4"] == "test"

    def test_int_dtype(self) -> None:
        schema = [
            FeatureSchema(name="x", dtype="int32", required=True),
            FeatureSchema(name="y", dtype="int64", required=True),
        ]
        features = create_test_features(schema)
        assert features["x"] == 0
        assert features["y"] == 0

    def test_empty_schema(self) -> None:
        features = create_test_features([])
        assert features == {}

    def test_all_float_types(self) -> None:
        schema = [
            FeatureSchema(name="a", dtype="float32", required=True),
            FeatureSchema(name="b", dtype="float64", required=True),
        ]
        features = create_test_features(schema)
        assert features["a"] == 0.0
        assert features["b"] == 0.0


# ===================================================================
# Performance Assertions
# ===================================================================


class TestAssertLatencyUnder:
    async def test_passes(self) -> None:
        model = MockModel(mock_latency_ms=0.0)
        await model.warmup()
        latency = await assert_latency_under(model, {"x": 1.0}, max_ms=1000.0)
        assert latency < 1000.0

    async def test_fails(self) -> None:
        model = MockModel(mock_latency_ms=50.0)
        await model.warmup()
        with pytest.raises(AssertionError, match="exceeds threshold"):
            await assert_latency_under(model, {"x": 1.0}, max_ms=0.001)


class TestAssertThroughputAbove:
    async def test_passes(self) -> None:
        model = MockModel(mock_latency_ms=0.0)
        await model.warmup()
        rps = await assert_throughput_above(model, {"x": 1.0}, n_calls=5, min_rps=1.0)
        assert rps >= 1.0

    async def test_fails(self) -> None:
        model = MockModel(mock_latency_ms=100.0)
        await model.warmup()
        with pytest.raises(AssertionError, match="below threshold"):
            await assert_throughput_above(model, {"x": 1.0}, n_calls=2, min_rps=1_000_000.0)


class TestAssertPredictionStable:
    async def test_stable(self) -> None:
        model = MockModel(mock_confidence=0.75, mock_latency_ms=0.0)
        await model.warmup()
        confidences = await assert_prediction_stable(model, {"x": 1.0}, n_runs=3)
        assert len(confidences) == 3
        assert all(c == 0.75 for c in confidences)

    async def test_unstable(self) -> None:
        """Mock a model that returns different confidences each call."""
        call_count = 0
        original_predict = MockModel.predict

        async def varying_predict(self, features):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            self._mock_confidence = 0.5 + call_count * 0.1
            return await original_predict(self, features)

        model = MockModel(mock_latency_ms=0.0)
        await model.warmup()
        model.predict = varying_predict.__get__(model)  # type: ignore[attr-defined]
        with pytest.raises(AssertionError, match="not stable"):
            await assert_prediction_stable(model, {"x": 1.0}, n_runs=3)


# ===================================================================
# Public API surface (__init__.py)
# ===================================================================


class TestPublicAPI:
    def test_core_imports(self) -> None:
        from artenic_ai_sdk import BaseModel, BasePrediction, ModelConfig

        assert BaseModel is not None
        assert BasePrediction is not None
        assert ModelConfig is not None

    def test_ensemble_imports(self) -> None:
        from artenic_ai_sdk import (
            DynamicWeighting,
            EnsembleManager,
            EnsembleStrategy,
            EvolutionPolicy,
            MetaLearner,
            WeightedAverage,
        )

        assert EnsembleManager is not None
        assert EnsembleStrategy is not None
        assert WeightedAverage is not None
        assert DynamicWeighting is not None
        assert MetaLearner is not None
        assert EvolutionPolicy is not None

    def test_config_imports(self) -> None:
        from artenic_ai_sdk import ConfigManager, ConfigRegistry

        assert ConfigManager is not None
        assert ConfigRegistry is not None

    def test_decorator_imports(self) -> None:
        from artenic_ai_sdk import (
            cache_inference,
            circuit_breaker,
            log_lifecycle,
            rate_limit,
            retry,
            timeout,
            track_inference,
            validate_input,
        )

        assert all(
            callable(f)
            for f in [
                track_inference,
                validate_input,
                retry,
                timeout,
                log_lifecycle,
                circuit_breaker,
                cache_inference,
                rate_limit,
            ]
        )

    def test_observability_imports(self) -> None:
        from artenic_ai_sdk import (
            MetricsCollector,
            ObservabilityMixin,
            StructuredLogger,
            correlation_context,
            get_trace_id,
        )

        assert MetricsCollector is not None
        assert StructuredLogger is not None
        assert ObservabilityMixin is not None
        assert callable(correlation_context)
        assert callable(get_trace_id)

    def test_testing_imports(self) -> None:
        from artenic_ai_sdk import (
            MockModel,
            ModelTestCase,
            assert_latency_under,
            assert_prediction_stable,
            assert_throughput_above,
            create_test_features,
        )

        assert MockModel is not None
        assert ModelTestCase is not None
        assert callable(create_test_features)
        assert callable(assert_latency_under)
        assert callable(assert_throughput_above)
        assert callable(assert_prediction_stable)

    def test_training_imports(self) -> None:
        from artenic_ai_sdk import (
            CallbackRunner,
            TrainingCallback,
            TrainingConfig,
            TrainingContext,
            build_callbacks,
        )

        assert TrainingConfig is not None
        assert TrainingCallback is not None
        assert TrainingContext is not None
        assert CallbackRunner is not None
        assert callable(build_callbacks)

    def test_types_imports(self) -> None:
        from artenic_ai_sdk import (
            CircuitBreakerState,
            ConfigPhase,
            DriftType,
            EnsemblePhase,
            EnsembleStrategyType,
            EvolutionTrigger,
            ModelFramework,
            ModelPhase,
            SerializationFormat,
        )

        assert ModelPhase.CREATED is not None
        assert ModelFramework.PYTORCH is not None
        assert SerializationFormat.SAFETENSORS is not None
        assert EnsembleStrategyType.WEIGHTED_AVERAGE is not None
        assert EnsemblePhase.DRAFT is not None
        assert EvolutionTrigger.MODEL_COUNT is not None
        assert DriftType.DATA_DRIFT is not None
        assert ConfigPhase.DEFAULT is not None
        assert CircuitBreakerState.CLOSED is not None

    def test_exception_imports(self) -> None:
        from artenic_ai_sdk import (
            ArtenicAIError,
            ConfigError,
            EnsembleError,
            ModelError,
            PlatformError,
            SerializationError,
            error_context,
        )

        assert issubclass(ModelError, ArtenicAIError)
        assert issubclass(EnsembleError, ArtenicAIError)
        assert issubclass(ConfigError, ArtenicAIError)
        assert issubclass(SerializationError, ArtenicAIError)
        assert issubclass(PlatformError, ArtenicAIError)
        assert callable(error_context)

    def test_schema_imports(self) -> None:
        from artenic_ai_sdk import (
            AutoPruneResult,
            ConfigDiff,
            ConfigEntry,
            DriftDetectionResult,
            EnsembleABTest,
            EnsembleResult,
            EnsembleSnapshot,
            EvalResult,
            EvolutionEvent,
            FeatureSchema,
            HealthCheckResult,
            ModelHealthReport,
            ModelMetadata,
            ModelSerializer,
            PlatformClient,
            TrainResult,
        )

        assert all(
            cls is not None
            for cls in [
                EnsembleResult,
                TrainResult,
                EvalResult,
                FeatureSchema,
                ModelMetadata,
                ConfigDiff,
                ConfigEntry,
                HealthCheckResult,
                DriftDetectionResult,
                ModelHealthReport,
                EnsembleSnapshot,
                EnsembleABTest,
                AutoPruneResult,
                EvolutionEvent,
                ModelSerializer,
                PlatformClient,
            ]
        )

    def test_no_trading_types(self) -> None:
        """Ensure trading-specific types were removed from v2."""
        import artenic_ai_sdk

        assert not hasattr(artenic_ai_sdk, "TradingPrediction")
        assert not hasattr(artenic_ai_sdk, "SignalDirection")
        assert not hasattr(artenic_ai_sdk, "MarketRegime")
        assert not hasattr(artenic_ai_sdk, "DynamicComposer")

    def test_version(self) -> None:
        import artenic_ai_sdk

        assert artenic_ai_sdk.__version__ == "0.1.0"

    def test_all_exports_are_importable(self) -> None:
        import artenic_ai_sdk

        for name in artenic_ai_sdk.__all__:
            assert hasattr(artenic_ai_sdk, name), f"{name} in __all__ but not importable"
