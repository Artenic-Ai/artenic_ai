"""Tests for artenic_ai_sdk.schemas — Pydantic models."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from artenic_ai_sdk.schemas import (
    AutoPruneResult,
    BasePrediction,
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
    ModelConfig,
    ModelHealthReport,
    ModelMetadata,
    TrainResult,
)
from artenic_ai_sdk.types import (
    ConfigPhase,
    DriftType,
    EnsemblePhase,
    EnsembleStrategyType,
    EvolutionTrigger,
    ModelFramework,
    ModelPhase,
)


class TestBasePrediction:
    def test_create_valid(self) -> None:
        pred = BasePrediction(
            confidence=0.95,
            model_id="test",
            model_version="0.1.0",
            inference_time_ms=12.5,
        )
        assert pred.confidence == 0.95
        assert pred.model_id == "test"
        assert isinstance(pred.timestamp, datetime)

    def test_prediction_id_auto_generated(self) -> None:
        pred = BasePrediction(
            confidence=0.5, model_id="x", model_version="1.0", inference_time_ms=1.0
        )
        assert pred.prediction_id  # non-empty
        # UUID format: 8-4-4-4-12
        assert len(pred.prediction_id.split("-")) == 5

    def test_prediction_id_unique(self) -> None:
        pred1 = BasePrediction(
            confidence=0.5, model_id="x", model_version="1.0", inference_time_ms=1.0
        )
        pred2 = BasePrediction(
            confidence=0.5, model_id="x", model_version="1.0", inference_time_ms=1.0
        )
        assert pred1.prediction_id != pred2.prediction_id

    def test_prediction_id_explicit(self) -> None:
        pred = BasePrediction(
            prediction_id="custom-id",
            confidence=0.5,
            model_id="x",
            model_version="1.0",
            inference_time_ms=1.0,
        )
        assert pred.prediction_id == "custom-id"

    def test_frozen(self) -> None:
        pred = BasePrediction(
            confidence=0.5, model_id="x", model_version="1.0", inference_time_ms=1.0
        )
        with pytest.raises(ValidationError):
            pred.confidence = 0.9  # type: ignore[misc]

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            BasePrediction(confidence=1.5, model_id="x", model_version="1.0", inference_time_ms=1.0)
        with pytest.raises(ValidationError):
            BasePrediction(
                confidence=-0.1, model_id="x", model_version="1.0", inference_time_ms=1.0
            )


class TestEnsembleResult:
    def test_create(self) -> None:
        result = EnsembleResult(
            confidence=0.8,
            model_id="ensemble",
            model_version="1.0",
            inference_time_ms=50.0,
            strategy_used=EnsembleStrategyType.WEIGHTED_AVERAGE,
            models_responded=["a", "b"],
        )
        assert result.strategy_used == EnsembleStrategyType.WEIGHTED_AVERAGE
        assert result.models_responded == ["a", "b"]
        assert result.models_failed == []


class TestTrainResult:
    def test_create_minimal(self) -> None:
        result = TrainResult(model_name="test", model_version="1.0")
        assert result.epochs_completed == 0
        assert result.early_stopped is False

    def test_create_full(self) -> None:
        result = TrainResult(
            model_name="test",
            model_version="1.0",
            metrics={"loss": 0.01, "accuracy": 0.99},
            epochs_completed=100,
            best_epoch=95,
            early_stopped=True,
            early_stopped_at_epoch=95,
            precision_used="fp16",
            gradient_checkpointing_used=True,
        )
        assert result.best_epoch == 95
        assert result.precision_used == "fp16"


class TestEvalResult:
    def test_create(self) -> None:
        result = EvalResult(
            model_name="test",
            model_version="1.0",
            metrics={"accuracy": 0.95},
            dataset_size=1000,
        )
        assert result.dataset_size == 1000


class TestFeatureSchema:
    def test_create(self) -> None:
        schema = FeatureSchema(name="price", dtype="float64")
        assert schema.required is True
        assert schema.shape is None

    def test_invalid_dtype(self) -> None:
        with pytest.raises(ValidationError):
            FeatureSchema(name="x", dtype="complex128")  # type: ignore[arg-type]


class TestModelMetadata:
    def test_defaults(self) -> None:
        meta = ModelMetadata(name="test", version="1.0", model_type="lgbm")
        assert meta.framework == ModelFramework.PYTORCH
        assert meta.tags == {}
        assert meta.input_features == []

    def test_model_size_bytes_default(self) -> None:
        meta = ModelMetadata(name="test", version="1.0", model_type="lgbm")
        assert meta.model_size_bytes is None

    def test_model_size_bytes_set(self) -> None:
        meta = ModelMetadata(
            name="test", version="1.0", model_type="lgbm", model_size_bytes=1024000
        )
        assert meta.model_size_bytes == 1024000

    def test_author_default(self) -> None:
        meta = ModelMetadata(name="test", version="1.0", model_type="lgbm")
        assert meta.author is None

    def test_author_set(self) -> None:
        meta = ModelMetadata(name="test", version="1.0", model_type="lgbm", author="artenic")
        assert meta.author == "artenic"


class TestModelConfig:
    def test_defaults(self) -> None:
        config = ModelConfig()
        assert config.version == "0.1.0"

    def test_validate_assignment(self) -> None:
        config = ModelConfig()
        config.version = "0.2.0"
        assert config.version == "0.2.0"

    def test_no_regime_overrides(self) -> None:
        """Verify regime_overrides have been removed."""
        config = ModelConfig()
        assert not hasattr(config, "regime_overrides")


class TestConfigEntry:
    def test_create(self) -> None:
        entry = ConfigEntry(config=ModelConfig(), model_id="test")
        assert entry.phase == ConfigPhase.DEFAULT
        assert entry.eval_result is None

    def test_no_regime_field(self) -> None:
        """Verify regime field has been removed."""
        entry = ConfigEntry(config=ModelConfig(), model_id="test")
        assert not hasattr(entry, "regime")


class TestHealthCheckResult:
    def test_healthy(self) -> None:
        result = HealthCheckResult(status="healthy", phase=ModelPhase.READY)
        assert result.inference_count == 0

    def test_invalid_status(self) -> None:
        with pytest.raises(ValidationError):
            HealthCheckResult(status="unknown", phase=ModelPhase.READY)  # type: ignore[arg-type]


class TestDriftDetectionResult:
    def test_create(self) -> None:
        result = DriftDetectionResult(
            model_id="test",
            drift_type=DriftType.DATA_DRIFT,
            drift_score=0.3,
        )
        assert result.is_drifting is False

    def test_score_bounds(self) -> None:
        with pytest.raises(ValidationError):
            DriftDetectionResult(model_id="test", drift_type=DriftType.DATA_DRIFT, drift_score=1.5)


class TestModelHealthReport:
    def test_create(self) -> None:
        health = HealthCheckResult(status="healthy", phase=ModelPhase.READY)
        report = ModelHealthReport(
            model_id="test",
            model_version="1.0",
            health_check=health,
        )
        assert report.recommendation == "keep"


class TestEnsembleSnapshot:
    def test_create(self) -> None:
        snap = EnsembleSnapshot(
            version_id="v1",
            strategy_type=EnsembleStrategyType.WEIGHTED_AVERAGE,
            model_ids=["a", "b"],
        )
        assert snap.phase == EnsemblePhase.DRAFT
        assert snap.quorum == 0.5

    def test_no_composer_regime_preferences(self) -> None:
        """Verify regime-based field has been removed."""
        snap = EnsembleSnapshot(
            version_id="v1",
            strategy_type=EnsembleStrategyType.WEIGHTED_AVERAGE,
        )
        assert not hasattr(snap, "composer_regime_preferences")


class TestEnsembleABTest:
    def test_create(self) -> None:
        snap_a = EnsembleSnapshot(
            version_id="a", strategy_type=EnsembleStrategyType.WEIGHTED_AVERAGE
        )
        snap_b = EnsembleSnapshot(
            version_id="b", strategy_type=EnsembleStrategyType.DYNAMIC_WEIGHTING
        )
        test = EnsembleABTest(test_id="t1", variant_a=snap_a, variant_b=snap_b)
        assert test.traffic_split == 0.5
        assert test.winner is None


class TestAutoPruneResult:
    def test_create(self) -> None:
        result = AutoPruneResult(
            pruned_model_ids=["bad_model"],
            kept_model_ids=["good_model"],
            reason={"bad_model": "low weight"},
        )
        assert result.weight_threshold_used == 0.0


class TestEvolutionEvent:
    def test_create(self) -> None:
        event = EvolutionEvent(
            from_strategy=EnsembleStrategyType.WEIGHTED_AVERAGE,
            to_strategy=EnsembleStrategyType.META_LEARNER,
            trigger=EvolutionTrigger.MODEL_COUNT,
            model_count=5,
        )
        assert event.model_count == 5


class TestConfigDiff:
    def test_create_empty(self) -> None:
        diff = ConfigDiff()
        assert diff.added == {}
        assert diff.removed == {}
        assert diff.changed == {}

    def test_create_with_values(self) -> None:
        diff = ConfigDiff(
            added={"new_field": 42},
            removed={"old_field": "gone"},
            changed={"version": ("1.0", "2.0")},
        )
        assert diff.added == {"new_field": 42}
        assert diff.removed == {"old_field": "gone"}
        assert diff.changed["version"] == ("1.0", "2.0")

    def test_frozen(self) -> None:
        diff = ConfigDiff()
        with pytest.raises(ValidationError):
            diff.added = {"x": 1}  # type: ignore[misc]


class TestNoTradingSchemas:
    """Verify trading-specific schemas are NOT present."""

    def test_no_trading_prediction(self) -> None:
        import artenic_ai_sdk.schemas as mod

        assert not hasattr(mod, "TradingPrediction")
