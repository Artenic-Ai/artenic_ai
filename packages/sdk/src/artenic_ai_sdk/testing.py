"""Testing utilities for model authors and platform developers.

Provides MockModel for ensemble/platform testing, ModelTestCase for
contract validation, and helpers for generating test data.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from artenic_ai_sdk.base_model import BaseModel
from artenic_ai_sdk.schemas import (
    BasePrediction,
    EvalResult,
    FeatureSchema,
    ModelMetadata,
    TrainResult,
)
from artenic_ai_sdk.types import ModelFramework, SerializationFormat

if TYPE_CHECKING:
    from pathlib import Path

    from artenic_ai_sdk.schemas import ModelConfig


class MockModel(BaseModel):
    """Configurable mock model for testing ensembles and platform.

    Example::

        model = MockModel(
            mock_id="test_lgbm",
            mock_confidence=0.85,
            mock_latency_ms=10.0,
        )
        await model.warmup()
        prediction = await model.predict({"feature_a": 1.0})
    """

    def __init__(
        self,
        mock_id: str = "mock_model",
        mock_version: str = "0.1.0",
        mock_type: str = "mock",
        mock_confidence: float = 0.8,
        mock_latency_ms: float = 1.0,
        mock_error: Exception | None = None,
        config: ModelConfig | None = None,
    ) -> None:
        super().__init__(config=config)
        self._mock_id = mock_id
        self._mock_version = mock_version
        self._mock_type = mock_type
        self._mock_confidence = mock_confidence
        self._mock_latency_ms = mock_latency_ms
        self._mock_error = mock_error
        self._predict_call_count = 0

    @property
    def model_id(self) -> str:
        return self._mock_id

    @property
    def model_version(self) -> str:
        return self._mock_version

    @property
    def model_type(self) -> str:
        return self._mock_type

    @property
    def framework(self) -> ModelFramework:
        return ModelFramework.PYTORCH

    @property
    def predict_call_count(self) -> int:
        return self._predict_call_count

    async def _do_warmup(self) -> None:
        pass

    async def predict(self, features: dict[str, Any]) -> BasePrediction:
        self._predict_call_count += 1

        if self._mock_error:
            raise self._mock_error

        if self._mock_latency_ms > 0:
            await asyncio.sleep(self._mock_latency_ms / 1000)

        return BasePrediction(
            confidence=self._mock_confidence,
            metadata={"mock": True, "features_received": list(features.keys())},
            model_id=self._mock_id,
            model_version=self._mock_version,
            inference_time_ms=self._mock_latency_ms,
        )

    async def preprocess(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        return raw_input

    async def train(self, dataset: Any, config: ModelConfig) -> TrainResult:
        return TrainResult(
            model_name=self._mock_id,
            model_version=self._mock_version,
            metrics={"accuracy": 0.9},
            epochs_completed=10,
        )

    async def evaluate(self, dataset: Any) -> EvalResult:
        return EvalResult(
            model_name=self._mock_id,
            model_version=self._mock_version,
            metrics={"accuracy": 0.85},
        )

    async def save(
        self,
        path: Path,
        format: SerializationFormat = SerializationFormat.SAFETENSORS,
    ) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        (path / "mock_model.txt").write_text(
            f"{self._mock_id}:{self._mock_version}",
            encoding="utf-8",
        )
        return path

    async def load(self, path: Path) -> None:
        pass

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name=self._mock_id,
            version=self._mock_version,
            model_type=self._mock_type,
            framework=self.framework,
            description="Mock model for testing",
        )


class ModelTestCase:
    """Reusable assertions for validating BaseModel implementations.

    Example::

        async def test_my_model():
            model = MyModel()
            tc = ModelTestCase()
            await tc.assert_full_lifecycle(model, sample_features)
    """

    async def assert_predict_returns_valid_output(
        self,
        model: BaseModel,
        features: dict[str, Any],
    ) -> BasePrediction:
        """Verify predict() returns a valid BasePrediction."""
        result = await model.predict(features)
        assert isinstance(result, BasePrediction), (
            f"predict() must return BasePrediction, got {type(result)}"
        )
        assert 0.0 <= result.confidence <= 1.0, (
            f"confidence must be in [0, 1], got {result.confidence}"
        )
        assert result.model_id == model.model_id
        assert result.inference_time_ms >= 0
        return result

    async def assert_warmup_transitions_to_ready(
        self,
        model: BaseModel,
    ) -> None:
        """Verify warmup() transitions to READY phase."""
        from artenic_ai_sdk.types import ModelPhase

        assert model.phase == ModelPhase.CREATED
        await model.warmup()
        assert model.phase == ModelPhase.READY  # type: ignore[comparison-overlap]

    async def assert_save_load_roundtrip(
        self,
        model: BaseModel,
        tmp_path: Path,
    ) -> None:
        """Verify save() then load() works without errors."""
        saved_path = await model.save(tmp_path / "test_model")
        assert saved_path.exists()
        await model.load(saved_path)

    async def assert_health_check_healthy(
        self,
        model: BaseModel,
    ) -> None:
        """Verify health_check() returns healthy after warmup."""
        result = await model.health_check()
        assert result.status == "healthy"

    async def assert_batch_matches_individual(
        self,
        model: BaseModel,
        batch: list[dict[str, Any]],
    ) -> None:
        """Verify predict_batch() results match individual predict() calls."""
        batch_results = await model.predict_batch(batch)
        individual_results = [await model.predict(f) for f in batch]

        assert len(batch_results) == len(individual_results)
        for batch_r, indiv_r in zip(batch_results, individual_results, strict=True):
            assert abs(batch_r.confidence - indiv_r.confidence) < 1e-6

    async def assert_full_lifecycle(
        self,
        model: BaseModel,
        sample_features: dict[str, Any],
    ) -> None:
        """Run the full lifecycle: warmup -> predict -> health -> teardown."""
        await self.assert_warmup_transitions_to_ready(model)
        await self.assert_predict_returns_valid_output(model, sample_features)
        await self.assert_health_check_healthy(model)
        await model.teardown()

        from artenic_ai_sdk.types import ModelPhase

        assert model.phase == ModelPhase.SHUTDOWN


def create_test_features(schema: list[FeatureSchema]) -> dict[str, Any]:
    """Generate valid test features from a FeatureSchema list.

    Creates dummy values matching each feature's dtype.

    Args:
        schema: List of FeatureSchema definitions.

    Returns:
        Dictionary of feature_name -> dummy_value.
    """
    dtype_defaults: dict[str, Any] = {
        "float32": 0.0,
        "float64": 0.0,
        "int32": 0,
        "int64": 0,
        "bool": False,
        "string": "test",
    }

    features: dict[str, Any] = {}
    for feat in schema:
        features[feat.name] = dtype_defaults.get(feat.dtype, 0.0)

    return features


# =============================================================================
# Performance Assertion Helpers
# =============================================================================


async def assert_latency_under(
    model: BaseModel,
    features: dict[str, Any],
    max_ms: float,
) -> float:
    """Assert that a single predict() call completes under the given threshold.

    Args:
        model: The model to test.
        features: Input features dict.
        max_ms: Maximum allowed latency in milliseconds.

    Returns:
        Actual latency in milliseconds.
    """
    start = time.perf_counter()
    await model.predict(features)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms <= max_ms, f"Latency {elapsed_ms:.2f}ms exceeds threshold {max_ms:.2f}ms"
    return elapsed_ms


async def assert_throughput_above(
    model: BaseModel,
    features: dict[str, Any],
    n_calls: int,
    min_rps: float,
) -> float:
    """Assert that the model achieves at least ``min_rps`` requests per second.

    Args:
        model: The model to test.
        features: Input features dict.
        n_calls: Number of sequential predict() calls.
        min_rps: Minimum required requests per second.

    Returns:
        Actual throughput in requests per second.
    """
    start = time.perf_counter()
    for _ in range(n_calls):
        await model.predict(features)
    elapsed = time.perf_counter() - start
    actual_rps = n_calls / max(elapsed, 1e-9)
    assert actual_rps >= min_rps, (
        f"Throughput {actual_rps:.2f} rps below threshold {min_rps:.2f} rps"
    )
    return actual_rps


async def assert_prediction_stable(
    model: BaseModel,
    features: dict[str, Any],
    n_runs: int = 5,
) -> list[float]:
    """Assert that repeated predictions produce identical confidence values.

    Args:
        model: The model to test.
        features: Input features dict.
        n_runs: Number of repeated calls.

    Returns:
        List of confidence values from each run.
    """
    confidences: list[float] = []
    for _ in range(n_runs):
        pred = await model.predict(features)
        confidences.append(pred.confidence)
    first = confidences[0]
    assert all(c == first for c in confidences), f"Predictions are not stable: {confidences}"
    return confidences
