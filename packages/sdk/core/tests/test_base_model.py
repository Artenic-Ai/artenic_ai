"""Tests for artenic_ai_sdk.base_model — BaseModel ABC."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from artenic_ai_sdk.base_model import BaseModel
from artenic_ai_sdk.exceptions import ModelInferenceError, ModelLoadError
from artenic_ai_sdk.schemas import (
    BasePrediction,
    EvalResult,
    FeatureSchema,
    ModelConfig,
    ModelMetadata,
    TrainResult,
)
from artenic_ai_sdk.types import ModelFramework, ModelPhase, SerializationFormat

if TYPE_CHECKING:
    from pathlib import Path


# =============================================================================
# Concrete implementation for testing
# =============================================================================


class DummyModel(BaseModel):
    """Minimal concrete implementation for testing."""

    def __init__(self, *, fail_warmup: bool = False, fail_predict: bool = False) -> None:
        super().__init__()
        self._fail_warmup = fail_warmup
        self._fail_predict = fail_predict

    @property
    def model_id(self) -> str:
        return "dummy_model"

    @property
    def model_version(self) -> str:
        return "1.0.0"

    @property
    def model_type(self) -> str:
        return "dummy"

    @property
    def framework(self) -> ModelFramework:
        return ModelFramework.PYTORCH

    async def _do_warmup(self) -> None:
        if self._fail_warmup:
            raise RuntimeError("warmup failed")

    async def predict(self, features: dict[str, Any]) -> BasePrediction:
        if self._fail_predict:
            raise ValueError("predict failed")
        return BasePrediction(
            confidence=0.9,
            model_id=self.model_id,
            model_version=self.model_version,
            inference_time_ms=1.0,
        )

    async def preprocess(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        return raw_input

    async def train(self, dataset: Any, config: ModelConfig) -> TrainResult:
        return TrainResult(model_name=self.model_id, model_version=self.model_version)

    async def evaluate(self, dataset: Any) -> EvalResult:
        return EvalResult(model_name=self.model_id, model_version=self.model_version)

    async def save(
        self,
        path: Path,
        format: SerializationFormat = SerializationFormat.SAFETENSORS,
    ) -> Path:
        return path

    async def load(self, path: Path) -> None:
        pass

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name=self.model_id,
            version=self.model_version,
            model_type=self.model_type,
            framework=self.framework,
        )


# =============================================================================
# Tests
# =============================================================================


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_initial_phase(self) -> None:
        model = DummyModel()
        assert model.phase == ModelPhase.CREATED
        assert not model.is_ready

    @pytest.mark.asyncio
    async def test_warmup_transitions_to_ready(self) -> None:
        model = DummyModel()
        await model.warmup()
        assert model.phase == ModelPhase.READY
        assert model.is_ready

    @pytest.mark.asyncio
    async def test_warmup_failure_transitions_to_error(self) -> None:
        model = DummyModel(fail_warmup=True)
        with pytest.raises(ModelLoadError):
            await model.warmup()
        assert model.phase == ModelPhase.ERROR

    @pytest.mark.asyncio
    async def test_teardown_transitions_to_shutdown(self) -> None:
        model = DummyModel()
        await model.warmup()
        await model.teardown()
        assert model.phase == ModelPhase.SHUTDOWN

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        async with DummyModel() as model:
            assert model.phase == ModelPhase.READY
        assert model.phase == ModelPhase.SHUTDOWN


class TestPredict:
    @pytest.mark.asyncio
    async def test_predict_returns_prediction(self) -> None:
        model = DummyModel()
        await model.warmup()
        result = await model.predict({"a": 1.0})
        assert isinstance(result, BasePrediction)
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_predict_batch(self) -> None:
        model = DummyModel()
        await model.warmup()
        results = await model.predict_batch([{"a": 1.0}, {"b": 2.0}])
        assert len(results) == 2


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_healthy(self) -> None:
        model = DummyModel()
        await model.warmup()
        health = await model.health_check()
        assert health.status == "healthy"
        assert health.phase == ModelPhase.READY

    @pytest.mark.asyncio
    async def test_unhealthy_on_error_phase(self) -> None:
        model = DummyModel(fail_warmup=True)
        with pytest.raises(ModelLoadError):
            await model.warmup()
        health = await model.health_check()
        assert health.status == "unhealthy"

    @pytest.mark.asyncio
    async def test_inference_count_tracked(self) -> None:
        model = DummyModel()
        await model.warmup()
        await model.predict({"a": 1.0})
        await model.predict({"b": 2.0})
        health = await model.health_check()
        assert health.inference_count == 2


class TestFeatureValidation:
    @pytest.mark.asyncio
    async def test_no_schema_is_noop(self) -> None:
        model = DummyModel()
        model.validate_features({"anything": "goes"})  # No error

    @pytest.mark.asyncio
    async def test_missing_required_feature(self) -> None:
        class SchemaModel(DummyModel):
            def get_feature_schema(self) -> list[FeatureSchema]:
                return [FeatureSchema(name="price", dtype="float64")]

        model = SchemaModel()
        with pytest.raises(ModelInferenceError, match="Missing required"):
            model.validate_features({})

    @pytest.mark.asyncio
    async def test_wrong_dtype(self) -> None:
        class SchemaModel(DummyModel):
            def get_feature_schema(self) -> list[FeatureSchema]:
                return [FeatureSchema(name="price", dtype="float64")]

        model = SchemaModel()
        with pytest.raises(ModelInferenceError, match="expected float64"):
            model.validate_features({"price": "not_a_number"})


class TestConfig:
    @pytest.mark.asyncio
    async def test_get_config(self) -> None:
        config = ModelConfig(version="2.0.0")
        model = DummyModel()
        model._config = config
        assert model.get_config().version == "2.0.0"

    @pytest.mark.asyncio
    async def test_update_config(self) -> None:
        model = DummyModel()
        new_config = ModelConfig(version="3.0.0")
        await model.update_config(new_config)
        assert model.get_config().version == "3.0.0"


class TestMetadata:
    def test_get_metadata(self) -> None:
        model = DummyModel()
        meta = model.get_metadata()
        assert meta.name == "dummy_model"
        assert meta.version == "1.0.0"
        assert meta.framework == ModelFramework.PYTORCH

    def test_repr(self) -> None:
        model = DummyModel()
        r = repr(model)
        assert "DummyModel" in r
        assert "dummy_model" in r
        assert "created" in r


class TestTeardownError:
    @pytest.mark.asyncio
    async def test_teardown_error_logged(self) -> None:
        """Teardown errors are caught and logged without raising."""

        class FailTeardownModel(DummyModel):
            async def _do_teardown(self) -> None:
                raise RuntimeError("teardown boom")

        model = FailTeardownModel()
        await model.warmup()
        # Should NOT raise — error is caught and logged
        await model.teardown()
        assert model.phase == ModelPhase.SHUTDOWN


class TestDegradedHealth:
    @pytest.mark.asyncio
    async def test_degraded_status(self) -> None:
        """Health check returns 'degraded' when error rate > 0.1."""
        model = DummyModel()
        await model.warmup()
        # Manipulate metrics to simulate high error rate
        model._metrics._total_count = 10
        model._metrics._error_count = 2  # 20% error rate
        health = await model.health_check()
        assert health.status == "degraded"


class TestPostprocess:
    @pytest.mark.asyncio
    async def test_default_postprocess(self) -> None:
        """Default postprocess returns prediction unchanged."""
        model = DummyModel()
        pred = BasePrediction(
            confidence=0.8,
            model_id="test",
            model_version="1.0",
            inference_time_ms=1.0,
        )
        result = await model.postprocess(pred)
        assert result is pred


class TestValidateFeaturesSkip:
    def test_optional_feature_not_in_input(self) -> None:
        """validate_features skips optional features not in input."""

        class SchemaModel(DummyModel):
            def get_feature_schema(self) -> list[FeatureSchema]:
                return [
                    FeatureSchema(name="price", dtype="float64"),
                    FeatureSchema(name="volume", dtype="float64", required=False),
                ]

        model = SchemaModel()
        # Only providing "price" — "volume" is optional, should not error
        model.validate_features({"price": 1.0})
