"""Tests for artenic_ai_sdk.ensemble — strategies, evolution, manager."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from artenic_ai_sdk.base_model import BaseModel
from artenic_ai_sdk_ensemble import (
    DynamicWeighting,
    EnsembleManager,
    EvolutionPolicy,
    MajorityVoting,
    MetaLearner,
    Stacking,
    WeightedAverage,
)
from artenic_ai_sdk.exceptions import (
    NoModelsRegisteredError,
    QuorumNotMetError,
    StrategyError,
)
from artenic_ai_sdk.schemas import (
    BasePrediction,
    EvalResult,
    ModelConfig,
    ModelMetadata,
)
from artenic_ai_sdk.types import (
    EnsembleStrategyType,
    ModelFramework,
)

# =============================================================================
# Helpers
# =============================================================================


def _pred(
    confidence: float,
    model_id: str = "m",
    latency: float = 1.0,
    metadata: dict[str, Any] | None = None,
) -> BasePrediction:
    """Create a test prediction."""
    return BasePrediction(
        confidence=confidence,
        model_id=model_id,
        model_version="1.0",
        inference_time_ms=latency,
        metadata=metadata or {},
    )


class DummyModel(BaseModel):
    """Minimal concrete model for testing."""

    def __init__(
        self,
        mid: str = "dummy",
        version: str = "1.0",
        confidence: float = 0.9,
        fail: bool = False,
    ) -> None:
        super().__init__()
        self._mid = mid
        self._version = version
        self._confidence = confidence
        self._fail = fail

    @property
    def model_id(self) -> str:
        return self._mid

    @property
    def model_version(self) -> str:
        return self._version

    @property
    def model_type(self) -> str:
        return "dummy"

    @property
    def framework(self) -> ModelFramework:
        return ModelFramework.CUSTOM

    async def _do_warmup(self) -> None:
        pass

    async def predict(self, features: dict[str, Any]) -> BasePrediction:
        if self._fail:
            raise RuntimeError("Model failed")
        return _pred(self._confidence, self._mid)

    async def preprocess(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        return raw_input

    async def train(self, dataset: Any, config: ModelConfig) -> Any:
        return None

    async def evaluate(self, dataset: Any) -> EvalResult:
        return EvalResult(model_name=self._mid, model_version=self._version)

    async def save(self, path: Any, format: Any = None) -> Any:
        return path

    async def load(self, path: Any) -> None:
        pass

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name=self._mid,
            version=self._version,
            model_type="dummy",
            framework=ModelFramework.CUSTOM,
        )


# =============================================================================
# WeightedAverage
# =============================================================================


class TestWeightedAverage:
    def test_strategy_type(self) -> None:
        s = WeightedAverage()
        assert s.strategy_type == EnsembleStrategyType.WEIGHTED_AVERAGE

    @pytest.mark.asyncio
    async def test_combine_equal_weights(self) -> None:
        s = WeightedAverage()
        preds = {"a": _pred(0.8, "a"), "b": _pred(0.6, "b")}
        result = await s.combine(preds)
        assert result.confidence == pytest.approx(0.7, abs=0.01)

    @pytest.mark.asyncio
    async def test_combine_explicit_weights(self) -> None:
        s = WeightedAverage(weights={"a": 0.7, "b": 0.3})
        preds = {"a": _pred(1.0, "a"), "b": _pred(0.0, "b")}
        result = await s.combine(preds)
        assert result.confidence == pytest.approx(0.7, abs=0.01)

    @pytest.mark.asyncio
    async def test_combine_empty_raises(self) -> None:
        s = WeightedAverage()
        with pytest.raises(StrategyError, match="No predictions"):
            await s.combine({})

    @pytest.mark.asyncio
    async def test_combine_all_zero_weights_raises(self) -> None:
        s = WeightedAverage(weights={"a": 0.0})
        preds = {"a": _pred(0.5, "a")}
        with pytest.raises(StrategyError, match="All weights are zero"):
            await s.combine(preds)

    def test_set_weights(self) -> None:
        s = WeightedAverage()
        s.set_weights({"a": 0.5, "b": 0.5})
        assert s.get_weights() == {"a": 0.5, "b": 0.5}

    @pytest.mark.asyncio
    async def test_update_weights_noop(self) -> None:
        s = WeightedAverage(weights={"a": 0.6})
        eval_r = EvalResult(model_name="a", model_version="1.0", metrics={"acc": 0.99})
        await s.update_weights("a", eval_r)
        assert s.get_weights() == {"a": 0.6}  # unchanged

    def test_import_state(self) -> None:
        s = WeightedAverage()
        s.import_state({"weights": {"x": 0.3, "y": 0.7}})
        assert s.get_weights() == {"x": 0.3, "y": 0.7}

    def test_export_state(self) -> None:
        s = WeightedAverage(weights={"a": 1.0})
        state = s.export_state()
        assert "weights" in state
        assert state["weights"] == {"a": 1.0}


# =============================================================================
# DynamicWeighting
# =============================================================================


class TestDynamicWeighting:
    def test_strategy_type(self) -> None:
        s = DynamicWeighting()
        assert s.strategy_type == EnsembleStrategyType.DYNAMIC_WEIGHTING

    @pytest.mark.asyncio
    async def test_combine_without_history(self) -> None:
        s = DynamicWeighting()
        preds = {"a": _pred(0.8, "a"), "b": _pred(0.6, "b")}
        result = await s.combine(preds)
        # Equal weights fallback (both 1.0)
        assert result.confidence == pytest.approx(0.7, abs=0.01)

    @pytest.mark.asyncio
    async def test_combine_empty_raises(self) -> None:
        s = DynamicWeighting()
        with pytest.raises(StrategyError, match="No predictions"):
            await s.combine({})

    @pytest.mark.asyncio
    async def test_update_weights_changes_weights(self) -> None:
        s = DynamicWeighting(performance_metric="acc")
        eval_a = EvalResult(model_name="a", model_version="1", metrics={"acc": 0.9})
        eval_b = EvalResult(model_name="b", model_version="1", metrics={"acc": 0.5})
        await s.update_weights("a", eval_a)
        await s.update_weights("b", eval_b)
        weights = s.get_weights()
        assert weights["a"] > weights["b"]

    @pytest.mark.asyncio
    async def test_exponential_decay(self) -> None:
        s = DynamicWeighting(performance_metric="acc", decay_factor=0.5)
        eval_good = EvalResult(model_name="a", model_version="1", metrics={"acc": 0.9})
        eval_bad = EvalResult(model_name="a", model_version="1", metrics={"acc": 0.1})
        # Old good, then bad -> recent bad should dominate
        await s.update_weights("a", eval_good)
        await s.update_weights("a", eval_bad)
        weights = s.get_weights()
        # With decay=0.5, recent scores matter more, so weight should be < 0.5
        assert weights["a"] < 0.5

    def test_import_state(self) -> None:
        s = DynamicWeighting()
        s.import_state({"weights": {"a": 0.8, "b": 0.2}})
        assert s.get_weights() == {"a": 0.8, "b": 0.2}


# =============================================================================
# MetaLearner
# =============================================================================


class TestMetaLearner:
    def test_strategy_type(self) -> None:
        s = MetaLearner()
        assert s.strategy_type == EnsembleStrategyType.META_LEARNER

    @pytest.mark.asyncio
    async def test_combine_fallback(self) -> None:
        s = MetaLearner()
        preds = {"a": _pred(0.8, "a"), "b": _pred(0.6, "b")}
        result = await s.combine(preds)
        assert result.metadata["meta_model_active"] is False
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_combine_empty_raises(self) -> None:
        s = MetaLearner()
        with pytest.raises(StrategyError, match="No predictions"):
            await s.combine({})

    @pytest.mark.asyncio
    async def test_update_weights(self) -> None:
        s = MetaLearner(performance_metric="acc")
        eval_r = EvalResult(
            model_name="a",
            model_version="1",
            metrics={"acc": 0.85},
        )
        await s.update_weights("a", eval_r)
        assert s.get_weights() == {"a": 0.85}

    def test_import_export_state(self) -> None:
        s = MetaLearner()
        s.import_state({"weights": {"a": 0.5}})
        state = s.export_state()
        assert state["weights"] == {"a": 0.5}

    def test_build_meta_model(self) -> None:
        s = MetaLearner(feature_dim=4)
        mock_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            s.build_meta_model(n_models=3)
        assert s._meta_model is not None
        assert s._meta_model.n_models == 3
        assert s._meta_model.feature_dim == 4

    def test_build_meta_model_custom_dim(self) -> None:
        s = MetaLearner(feature_dim=4)
        mock_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            s.build_meta_model(n_models=2, feature_dim=16)
        assert s._meta_model is not None
        assert s._meta_model.feature_dim == 16

    @pytest.mark.asyncio
    async def test_load_meta_model_no_torch(self) -> None:
        s = MetaLearner()
        with patch.dict("sys.modules", {"torch": None}):
            await s.load_meta_model("/fake/path")
        assert s._meta_model is None

    @pytest.mark.asyncio
    async def test_load_meta_model_with_torch(self) -> None:
        s = MetaLearner()
        mock_torch = MagicMock()
        mock_torch.load.return_value = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            await s.load_meta_model("/fake/path")
        assert s._meta_model is not None

    @pytest.mark.asyncio
    async def test_combine_with_meta_model(self) -> None:
        """Test MetaLearner combine when a meta model is active (mocked)."""
        s = MetaLearner(feature_dim=4)

        # Build a mock meta model
        mock_meta = MagicMock()
        mock_meta.n_models = 2
        mock_meta.feature_dim = 4

        # Mock torch tensor operations
        mock_torch = MagicMock()
        # forward returns a tensor-like object with [0] indexing and .tolist()
        mock_weights = MagicMock()
        mock_weights.__getitem__ = lambda self, idx: MagicMock(tolist=lambda: [0.6, 0.4])
        mock_meta.forward.return_value = mock_weights
        mock_torch.tensor.return_value = MagicMock()
        mock_torch.float32 = "float32"
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()

        s._meta_model = mock_meta

        preds = {"a": _pred(0.9, "a", 5.0), "b": _pred(0.7, "b", 3.0)}
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = await s.combine(preds)

        assert result.metadata["meta_model_active"] is True
        assert 0.0 <= result.confidence <= 1.0
        assert "weights" in result.metadata

    @pytest.mark.asyncio
    async def test_combine_with_meta_model_fewer_models_than_expected(self) -> None:
        """When fewer predictions than meta model expects, padding kicks in."""
        s = MetaLearner(feature_dim=4)

        mock_meta = MagicMock()
        mock_meta.n_models = 4  # expects 4 models
        mock_meta.feature_dim = 4

        mock_torch = MagicMock()
        mock_weights = MagicMock()
        mock_weights.__getitem__ = lambda self, idx: MagicMock(tolist=lambda: [0.3, 0.7, 0.0, 0.0])
        mock_meta.forward.return_value = mock_weights
        mock_torch.tensor.return_value = MagicMock()
        mock_torch.float32 = "float32"
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()

        s._meta_model = mock_meta

        # Only 2 predictions but meta model expects 4
        preds = {"a": _pred(0.9, "a", 5.0), "b": _pred(0.7, "b", 3.0)}
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = await s.combine(preds)

        assert result.metadata["meta_model_active"] is True

    @pytest.mark.asyncio
    async def test_combine_with_meta_model_no_torch_fallback(self) -> None:
        """When meta model exists but torch import fails, fallback."""
        s = MetaLearner()
        s._meta_model = MagicMock()  # fake meta model exists
        preds = {"a": _pred(0.8, "a"), "b": _pred(0.6, "b")}
        with patch.dict("sys.modules", {"torch": None}):
            result = await s.combine(preds)
        assert result.metadata["meta_model_active"] is False


# =============================================================================
# MajorityVoting
# =============================================================================


class TestMajorityVoting:
    @pytest.mark.asyncio
    async def test_strategy_type(self) -> None:
        s = MajorityVoting()
        assert s.strategy_type == EnsembleStrategyType.MAJORITY_VOTING

    @pytest.mark.asyncio
    async def test_combine_majority_class(self) -> None:
        preds = {
            "a": _pred(0.9, "a", metadata={"predicted_class": "buy"}),
            "b": _pred(0.8, "b", metadata={"predicted_class": "buy"}),
            "c": _pred(0.7, "c", metadata={"predicted_class": "sell"}),
        }
        s = MajorityVoting()
        result = await s.combine(preds)
        assert result.metadata["predicted_class"] == "buy"
        assert result.metadata["vote_counts"]["buy"] == 2
        assert result.metadata["vote_counts"]["sell"] == 1
        assert result.confidence == pytest.approx(2 / 3)

    @pytest.mark.asyncio
    async def test_combine_tie_break_by_confidence(self) -> None:
        """When two classes tie on vote count, highest avg confidence wins."""
        preds = {
            "a": _pred(0.9, "a", metadata={"predicted_class": "buy"}),
            "b": _pred(0.3, "b", metadata={"predicted_class": "sell"}),
        }
        s = MajorityVoting()
        result = await s.combine(preds)
        # Tie: 1 vote each. "buy" has higher confidence → wins
        assert result.metadata["predicted_class"] == "buy"

    @pytest.mark.asyncio
    async def test_combine_no_class_info_fallback(self) -> None:
        """When no predicted_class metadata, fallback to highest confidence."""
        preds = {
            "a": _pred(0.9, "a"),
            "b": _pred(0.6, "b"),
        }
        s = MajorityVoting()
        result = await s.combine(preds)
        assert result.metadata["fallback"] is True
        assert result.metadata["selected_model"] == "a"
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_combine_empty_raises(self) -> None:
        s = MajorityVoting()
        with pytest.raises(StrategyError, match="No predictions"):
            await s.combine({})

    @pytest.mark.asyncio
    async def test_update_weights(self) -> None:
        s = MajorityVoting()
        perf = EvalResult(model_name="a", model_version="1.0", metrics={"accuracy": 0.9})
        await s.update_weights("model_a", perf)
        weights = s.get_weights()
        assert "model_a" in weights
        assert weights["model_a"] == 1.0  # accuracy > 0.5 → correct

    @pytest.mark.asyncio
    async def test_update_weights_low_accuracy(self) -> None:
        s = MajorityVoting()
        perf = EvalResult(model_name="a", model_version="1.0", metrics={"accuracy": 0.3})
        await s.update_weights("model_a", perf)
        assert s.get_weights()["model_a"] == 0.0

    def test_export_import_state(self) -> None:
        s = MajorityVoting()
        s._weights = {"a": 0.8, "b": 0.6}
        s._correct_counts = {"a": 4, "b": 3}
        s._total_counts = {"a": 5, "b": 5}
        state = s.export_state()
        assert state["weights"] == {"a": 0.8, "b": 0.6}
        assert state["correct_counts"] == {"a": 4, "b": 3}

        s2 = MajorityVoting()
        s2.import_state(state)
        assert s2.get_weights() == {"a": 0.8, "b": 0.6}
        assert s2._correct_counts == {"a": 4, "b": 3}

    def test_import_state_empty(self) -> None:
        s = MajorityVoting()
        s.import_state({})
        assert s.get_weights() == {}


# =============================================================================
# Stacking
# =============================================================================


class TestStacking:
    @pytest.mark.asyncio
    async def test_strategy_type(self) -> None:
        s = Stacking()
        assert s.strategy_type == EnsembleStrategyType.STACKING

    @pytest.mark.asyncio
    async def test_combine_equal_weights_fallback(self) -> None:
        """Without fitted weights, uses equal combination."""
        preds = {
            "a": _pred(0.8, "a"),
            "b": _pred(0.6, "b"),
        }
        s = Stacking()
        result = await s.combine(preds)
        assert result.confidence == pytest.approx(0.7)
        assert result.metadata["fitted"] is False

    @pytest.mark.asyncio
    async def test_combine_with_learned_weights(self) -> None:
        s = Stacking()
        s._weights = {"a": 0.7, "b": 0.3}
        s._bias = 0.0
        preds = {
            "a": _pred(1.0, "a"),
            "b": _pred(0.0, "b"),
        }
        result = await s.combine(preds)
        assert result.confidence == pytest.approx(0.7)
        assert result.metadata["fitted"] is True

    @pytest.mark.asyncio
    async def test_combine_clamps_output(self) -> None:
        """Confidence should be clamped to [0, 1]."""
        s = Stacking()
        s._weights = {"a": 2.0}
        s._bias = 0.5
        preds = {"a": _pred(0.8, "a")}
        result = await s.combine(preds)
        assert result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_combine_empty_raises(self) -> None:
        s = Stacking()
        with pytest.raises(StrategyError, match="No predictions"):
            await s.combine({})

    @pytest.mark.asyncio
    async def test_update_weights(self) -> None:
        s = Stacking()
        perf = EvalResult(model_name="a", model_version="1.0", metrics={"accuracy": 0.85})
        await s.update_weights("model_a", perf)
        assert s.get_weights()["model_a"] == 0.85

    def test_fit_basic(self) -> None:
        """Fit on simple linear data: outcome = confidence."""
        s = Stacking()
        training_data = [
            {"predictions": {"a": 0.0}, "outcome": 0.0},
            {"predictions": {"a": 0.5}, "outcome": 0.5},
            {"predictions": {"a": 1.0}, "outcome": 1.0},
        ]
        s.fit(training_data)
        assert "a" in s.get_weights()
        # The weight for "a" should be close to 1.0 (perfect linear relationship)
        assert s.get_weights()["a"] == pytest.approx(1.0, abs=0.1)
        assert s._bias == pytest.approx(0.0, abs=0.1)

    def test_fit_insufficient_data(self) -> None:
        """Less than 2 data points: no fit."""
        s = Stacking()
        s.fit([{"predictions": {"a": 0.5}, "outcome": 0.5}])
        assert s.get_weights() == {}

    def test_fit_no_model_ids(self) -> None:
        """Training data with empty predictions: no fit."""
        s = Stacking()
        s.fit([{"predictions": {}, "outcome": 0.5}, {"predictions": {}, "outcome": 0.5}])
        assert s.get_weights() == {}

    def test_fit_singular_matrix(self) -> None:
        """Identical data points make singular matrix: no crash."""
        s = Stacking()
        training_data = [
            {"predictions": {"a": 0.5}, "outcome": 0.5},
            {"predictions": {"a": 0.5}, "outcome": 0.5},
        ]
        s.fit(training_data)
        # Should not crash — either fits or returns gracefully

    def test_export_import_state(self) -> None:
        s = Stacking()
        s._weights = {"a": 0.7, "b": 0.3}
        s._bias = 0.1
        state = s.export_state()
        assert state["weights"] == {"a": 0.7, "b": 0.3}
        assert state["bias"] == 0.1

        s2 = Stacking()
        s2.import_state(state)
        assert s2.get_weights() == {"a": 0.7, "b": 0.3}
        assert s2._bias == 0.1

    def test_import_state_empty(self) -> None:
        s = Stacking()
        s.import_state({})
        assert s.get_weights() == {}
        assert s._bias == 0.0

    def test_solve_linear_basic(self) -> None:
        """Test the linear solver with a simple system."""
        # 2x = 6 → x = 3
        result = Stacking._solve_linear([[2.0]], [6.0])
        assert result is not None
        assert result[0] == pytest.approx(3.0)

    def test_solve_linear_singular(self) -> None:
        """Singular matrix returns None."""
        result = Stacking._solve_linear([[0.0]], [1.0])
        assert result is None

    def test_solve_linear_2x2(self) -> None:
        """Test 2x2 system: x + y = 3, x - y = 1 → x=2, y=1."""
        result = Stacking._solve_linear([[1.0, 1.0], [1.0, -1.0]], [3.0, 1.0])
        assert result is not None
        assert result[0] == pytest.approx(2.0)
        assert result[1] == pytest.approx(1.0)


# =============================================================================
# Manager: build_strategy for new types
# =============================================================================


class TestManagerNewStrategies:
    def test_build_majority_voting(self) -> None:
        manager = EnsembleManager(strategy=WeightedAverage())
        s = manager._build_strategy(EnsembleStrategyType.MAJORITY_VOTING)
        assert isinstance(s, MajorityVoting)

    def test_build_stacking(self) -> None:
        manager = EnsembleManager(strategy=WeightedAverage())
        s = manager._build_strategy(EnsembleStrategyType.STACKING)
        assert isinstance(s, Stacking)

    def test_build_stacking_with_evolution_policy(self) -> None:
        policy = EvolutionPolicy(
            enabled=True,
            performance_metric="f1_score",
        )
        manager = EnsembleManager(strategy=WeightedAverage(), evolution_policy=policy)
        s = manager._build_strategy(EnsembleStrategyType.STACKING)
        assert isinstance(s, Stacking)
        assert s._metric == "f1_score"


# =============================================================================
# _AttentionGatingNetwork (mock-based, torch not installed)
# =============================================================================


class TestAttentionGatingNetwork:
    def test_init_and_properties(self) -> None:
        """Test _AttentionGatingNetwork init with mocked torch.nn."""
        mock_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            from artenic_ai_sdk_ensemble.strategies import _AttentionGatingNetwork

            net = _AttentionGatingNetwork(n_models=3, feature_dim=4, hidden_dim=16, num_heads=2)
        assert net.n_models == 3
        assert net.feature_dim == 4

    def test_forward(self) -> None:
        """Test forward pass with mocked torch.nn."""
        mock_torch = MagicMock()
        # attention returns (output, weights) tuple
        mock_attn_output = MagicMock()
        mock_torch.nn.MultiheadAttention.return_value.return_value = (mock_attn_output, MagicMock())
        with patch.dict("sys.modules", {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            from artenic_ai_sdk_ensemble.strategies import _AttentionGatingNetwork

            net = _AttentionGatingNetwork(n_models=2, feature_dim=4)

        mock_input = MagicMock()
        result = net.forward(mock_input)
        assert result is not None

    def test_parameters_iterator(self) -> None:
        """Test parameters() returns an iterable of params."""
        mock_torch = MagicMock()
        mock_torch.nn.Linear.return_value.parameters.return_value = iter([MagicMock()])
        mock_torch.nn.LayerNorm.return_value.parameters.return_value = iter([MagicMock()])
        mock_torch.nn.MultiheadAttention.return_value.parameters.return_value = iter([MagicMock()])
        with patch.dict("sys.modules", {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            from artenic_ai_sdk_ensemble.strategies import _AttentionGatingNetwork

            net = _AttentionGatingNetwork(n_models=2, feature_dim=4)

        params = list(net.parameters())
        assert len(params) > 0


# =============================================================================
# DynamicWeighting — additional coverage
# =============================================================================


class TestDynamicWeightingCoverage:
    @pytest.mark.asyncio
    async def test_combine_zero_weights(self) -> None:
        """When all explicit weights are zero, fallback to equal."""
        s = DynamicWeighting()
        # Manually set zero weights
        s._weights = {"a": 0.0, "b": 0.0}
        preds = {"a": _pred(0.8, "a"), "b": _pred(0.6, "b")}
        result = await s.combine(preds)
        # Should fallback to equal weights
        assert result.confidence == pytest.approx(0.7, abs=0.01)

    @pytest.mark.asyncio
    async def test_compute_weights_empty_scores(self) -> None:
        """Model with empty history gets min_weight."""
        from collections import deque

        s = DynamicWeighting(min_weight=0.1)
        s._history = {"a": deque(maxlen=20)}  # empty deque
        weights = s._compute_weights()
        assert weights["a"] == 0.1


# =============================================================================
# EvolutionPolicy
# =============================================================================


class TestEvolutionPolicy:
    def test_defaults(self) -> None:
        p = EvolutionPolicy()
        assert p.dynamic_weighting_threshold == 3
        assert p.meta_learner_threshold == 5
        assert p.min_observations == 10
        assert p.performance_metric == "accuracy"
        assert p.enabled is False

    def test_custom_values(self) -> None:
        p = EvolutionPolicy(
            dynamic_weighting_threshold=2,
            meta_learner_threshold=4,
            min_observations=5,
            performance_metric="f1",
            enabled=True,
        )
        assert p.dynamic_weighting_threshold == 2
        assert p.enabled is True


# =============================================================================
# EnsembleManager — Registration & Inference
# =============================================================================


class TestEnsembleManagerBasic:
    @pytest.mark.asyncio
    async def test_register_and_count(self) -> None:
        manager = EnsembleManager(strategy=WeightedAverage())
        model = DummyModel("m1")
        await manager.register("m1", model)
        assert manager.model_count == 1
        assert "m1" in manager.models

    @pytest.mark.asyncio
    async def test_unregister(self) -> None:
        manager = EnsembleManager(strategy=WeightedAverage())
        await manager.register("m1", DummyModel("m1"))
        await manager.unregister("m1")
        assert manager.model_count == 0

    @pytest.mark.asyncio
    async def test_predict_no_models_raises(self) -> None:
        manager = EnsembleManager(strategy=WeightedAverage())
        with pytest.raises(NoModelsRegisteredError):
            await manager.predict({"a": 1.0})

    @pytest.mark.asyncio
    async def test_predict_single_model(self) -> None:
        manager = EnsembleManager(strategy=WeightedAverage())
        await manager.register("m1", DummyModel("m1", confidence=0.8))
        result = await manager.predict({"a": 1.0})
        assert result.confidence == pytest.approx(0.8, abs=0.01)
        assert "m1" in result.models_responded

    @pytest.mark.asyncio
    async def test_predict_multiple_models(self) -> None:
        s = WeightedAverage(weights={"m1": 0.6, "m2": 0.4})
        manager = EnsembleManager(strategy=s)
        await manager.register("m1", DummyModel("m1", confidence=1.0))
        await manager.register("m2", DummyModel("m2", confidence=0.5))
        result = await manager.predict({"a": 1.0})
        assert result.strategy_used == EnsembleStrategyType.WEIGHTED_AVERAGE
        assert len(result.models_responded) == 2

    @pytest.mark.asyncio
    async def test_quorum_not_met(self) -> None:
        manager = EnsembleManager(strategy=WeightedAverage(), quorum=1.0)
        await manager.register("m1", DummyModel("m1", fail=True))
        with pytest.raises(QuorumNotMetError):
            await manager.predict({"a": 1.0})

    @pytest.mark.asyncio
    async def test_fallback_prediction(self) -> None:
        manager = EnsembleManager(strategy=WeightedAverage(), quorum=1.0)
        await manager.register("m1", DummyModel("m1", fail=True))
        fallback = _pred(0.5, "fallback")
        manager.set_fallback_prediction(fallback)
        result = await manager.predict({"a": 1.0})
        assert result.metadata.get("fallback") is True
        assert result.confidence == 0.5


# =============================================================================
# EnsembleManager — Strategy management
# =============================================================================


class TestEnsembleManagerStrategy:
    @pytest.mark.asyncio
    async def test_swap_strategy(self) -> None:
        manager = EnsembleManager(strategy=WeightedAverage(weights={"m1": 0.8}))
        new_s = DynamicWeighting()
        await manager.swap_strategy(new_s, transfer_weights=True)
        assert manager.strategy.strategy_type == EnsembleStrategyType.DYNAMIC_WEIGHTING
        # Weights should have been transferred
        assert new_s.get_weights().get("m1") == 0.8

    @pytest.mark.asyncio
    async def test_swap_strategy_no_transfer(self) -> None:
        manager = EnsembleManager(strategy=WeightedAverage(weights={"m1": 0.8}))
        new_s = DynamicWeighting()
        await manager.swap_strategy(new_s, transfer_weights=False)
        assert new_s.get_weights() == {}

    def test_get_weights(self) -> None:
        s = WeightedAverage(weights={"a": 0.5, "b": 0.5})
        manager = EnsembleManager(strategy=s)
        assert manager.get_weights() == {"a": 0.5, "b": 0.5}


# =============================================================================
# EnsembleManager — Evolution
# =============================================================================


class TestEnsembleManagerEvolution:
    @pytest.mark.asyncio
    async def test_no_evolution_without_policy(self) -> None:
        manager = EnsembleManager(strategy=WeightedAverage())
        event = await manager.check_evolution()
        assert event is None

    @pytest.mark.asyncio
    async def test_no_evolution_policy_disabled(self) -> None:
        policy = EvolutionPolicy(enabled=False)
        manager = EnsembleManager(strategy=WeightedAverage(), evolution_policy=policy)
        event = await manager.check_evolution()
        assert event is None

    @pytest.mark.asyncio
    async def test_evolution_wa_to_dw(self) -> None:
        policy = EvolutionPolicy(
            dynamic_weighting_threshold=2,
            min_observations=1,
            enabled=True,
        )
        manager = EnsembleManager(
            strategy=WeightedAverage(),
            evolution_policy=policy,
        )
        await manager.register("m1", DummyModel("m1"))
        await manager.register("m2", DummyModel("m2"))

        eval_r = EvalResult(model_name="m1", model_version="1", metrics={"accuracy": 0.9})
        event = await manager.update_model_performance("m1", eval_r)
        assert event is not None
        assert event.to_strategy == EnsembleStrategyType.DYNAMIC_WEIGHTING
        assert manager.strategy.strategy_type == EnsembleStrategyType.DYNAMIC_WEIGHTING

    @pytest.mark.asyncio
    async def test_evolution_dw_to_ml(self) -> None:
        policy = EvolutionPolicy(
            meta_learner_threshold=2,
            min_observations=1,
            enabled=True,
        )
        manager = EnsembleManager(
            strategy=DynamicWeighting(),
            evolution_policy=policy,
        )
        await manager.register("m1", DummyModel("m1"))
        await manager.register("m2", DummyModel("m2"))

        eval_r = EvalResult(model_name="m1", model_version="1", metrics={"accuracy": 0.9})
        event = await manager.update_model_performance("m1", eval_r)
        assert event is not None
        assert event.to_strategy == EnsembleStrategyType.META_LEARNER

    @pytest.mark.asyncio
    async def test_no_evolution_too_few_observations(self) -> None:
        policy = EvolutionPolicy(
            dynamic_weighting_threshold=1,
            min_observations=100,
            enabled=True,
        )
        manager = EnsembleManager(
            strategy=WeightedAverage(),
            evolution_policy=policy,
        )
        await manager.register("m1", DummyModel("m1"))
        event = await manager.check_evolution()
        assert event is None

    def test_evolution_log(self) -> None:
        manager = EnsembleManager(strategy=WeightedAverage())
        assert manager.evolution_log == []

    def test_evolution_policy_property(self) -> None:
        policy = EvolutionPolicy()
        manager = EnsembleManager(strategy=WeightedAverage(), evolution_policy=policy)
        assert manager.evolution_policy is policy


# =============================================================================
# EnsembleManager — Versioning
# =============================================================================


class TestEnsembleManagerVersioning:
    @pytest.mark.asyncio
    async def test_snapshot(self) -> None:
        manager = EnsembleManager(strategy=WeightedAverage(weights={"m1": 0.8}))
        await manager.register("m1", DummyModel("m1"))
        snap = await manager.snapshot(metadata={"note": "test"})
        assert snap.version_id is not None
        assert snap.strategy_type == EnsembleStrategyType.WEIGHTED_AVERAGE
        assert snap.strategy_weights == {"m1": 0.8}
        assert snap.model_ids == ["m1"]
        assert snap.metadata == {"note": "test"}
        assert snap.parent_version_id is None

    @pytest.mark.asyncio
    async def test_snapshot_parent_chain(self) -> None:
        manager = EnsembleManager(strategy=WeightedAverage())
        s1 = await manager.snapshot()
        s2 = await manager.snapshot()
        assert s2.parent_version_id == s1.version_id

    @pytest.mark.asyncio
    async def test_restore_snapshot(self) -> None:
        manager = EnsembleManager(strategy=WeightedAverage(weights={"m1": 0.8}))
        snap = await manager.snapshot()

        # Change strategy
        await manager.swap_strategy(DynamicWeighting())

        # Restore
        await manager.restore_snapshot(snap.version_id)
        assert manager.strategy.strategy_type == EnsembleStrategyType.WEIGHTED_AVERAGE

    @pytest.mark.asyncio
    async def test_restore_snapshot_not_found(self) -> None:
        manager = EnsembleManager(strategy=WeightedAverage())
        with pytest.raises(ValueError, match="Snapshot not found"):
            await manager.restore_snapshot("nonexistent")

    def test_version_history(self) -> None:
        manager = EnsembleManager(strategy=WeightedAverage())
        assert manager.get_version_history() == []


# =============================================================================
# EnsembleManager — Auto-Pruning
# =============================================================================


class TestEnsembleManagerPruning:
    @pytest.mark.asyncio
    async def test_auto_prune_no_weights(self) -> None:
        manager = EnsembleManager(strategy=WeightedAverage())
        await manager.register("m1", DummyModel("m1"))
        result = await manager.auto_prune()
        assert result.pruned_model_ids == []
        assert "m1" in result.kept_model_ids

    @pytest.mark.asyncio
    async def test_auto_prune_removes_low_weight(self) -> None:
        s = WeightedAverage(weights={"m1": 0.9, "m2": 0.8, "m3": 0.01})
        manager = EnsembleManager(strategy=s)
        await manager.register("m1", DummyModel("m1"))
        await manager.register("m2", DummyModel("m2"))
        await manager.register("m3", DummyModel("m3"))

        result = await manager.auto_prune(weight_threshold=0.05, min_models=2)
        assert "m3" in result.pruned_model_ids
        assert len(result.kept_model_ids) >= 2
        assert manager.model_count == 2

    @pytest.mark.asyncio
    async def test_auto_prune_keeps_minimum(self) -> None:
        s = WeightedAverage(weights={"m1": 0.01, "m2": 0.01})
        manager = EnsembleManager(strategy=s)
        await manager.register("m1", DummyModel("m1"))
        await manager.register("m2", DummyModel("m2"))

        result = await manager.auto_prune(weight_threshold=0.5, min_models=2)
        # Should keep both even though weights are below threshold
        assert result.pruned_model_ids == []
        assert len(result.kept_model_ids) == 2


# =============================================================================
# EnsembleManager — A/B Testing
# =============================================================================


class TestEnsembleManagerABTest:
    @pytest.mark.asyncio
    async def test_create_ab_test(self) -> None:
        manager = EnsembleManager(strategy=WeightedAverage())
        snap_a = await manager.snapshot()
        snap_b = await manager.snapshot()

        test = await manager.create_ensemble_ab_test("t1", snap_a, snap_b, 0.5)
        assert test.test_id == "t1"
        assert test.traffic_split == 0.5

    def test_select_ab_variant(self) -> None:
        manager = EnsembleManager(strategy=WeightedAverage())

        async def _setup() -> None:
            snap_a = await manager.snapshot()
            snap_b = await manager.snapshot()
            await manager.create_ensemble_ab_test("t1", snap_a, snap_b, 0.5)

        asyncio.get_event_loop().run_until_complete(_setup())

        assert manager.select_ab_variant("t1", 0.3) == "a"
        assert manager.select_ab_variant("t1", 0.7) == "b"

    def test_select_ab_variant_not_found(self) -> None:
        manager = EnsembleManager(strategy=WeightedAverage())
        with pytest.raises(KeyError, match="A/B test not found"):
            manager.select_ab_variant("nonexistent")

    @pytest.mark.asyncio
    async def test_conclude_ab_test(self) -> None:
        manager = EnsembleManager(strategy=WeightedAverage())
        snap_a = await manager.snapshot()

        await manager.swap_strategy(DynamicWeighting())
        snap_b = await manager.snapshot()

        await manager.create_ensemble_ab_test("t1", snap_a, snap_b, 0.5)
        concluded = await manager.conclude_ensemble_ab_test("t1", winner="a")
        assert concluded.winner == "a"
        assert manager.strategy.strategy_type == EnsembleStrategyType.WEIGHTED_AVERAGE

    @pytest.mark.asyncio
    async def test_conclude_ab_test_not_found(self) -> None:
        manager = EnsembleManager(strategy=WeightedAverage())
        with pytest.raises(KeyError, match="A/B test not found"):
            await manager.conclude_ensemble_ab_test("nonexistent", winner="a")


# =============================================================================
# EnsembleManager — Health Reports
# =============================================================================


class TestEnsembleManagerHealth:
    @pytest.mark.asyncio
    async def test_health_report_not_found(self) -> None:
        manager = EnsembleManager(strategy=WeightedAverage())
        report = await manager.get_model_health_report("nonexistent")
        assert report is None

    @pytest.mark.asyncio
    async def test_health_report_healthy(self) -> None:
        s = WeightedAverage(weights={"m1": 0.5})
        manager = EnsembleManager(strategy=s)
        model = DummyModel("m1")
        await manager.register("m1", model)
        await model.warmup()

        report = await manager.get_model_health_report("m1")
        assert report is not None
        assert report.model_id == "m1"
        assert report.recommendation == "keep"

    @pytest.mark.asyncio
    async def test_health_report_low_weight(self) -> None:
        s = WeightedAverage(weights={"m1": 0.01})
        manager = EnsembleManager(strategy=s)
        model = DummyModel("m1")
        await manager.register("m1", model)
        await model.warmup()

        report = await manager.get_model_health_report("m1")
        assert report is not None
        assert report.recommendation == "monitor"


# =============================================================================
# EnsembleManager — Model Timeout
# =============================================================================


class TestEnsembleManagerTimeout:
    @pytest.mark.asyncio
    async def test_set_model_timeout(self) -> None:
        manager = EnsembleManager(strategy=WeightedAverage())
        manager.set_model_timeout("m1", 5.0)
        assert manager._model_timeouts["m1"] == 5.0

    @pytest.mark.asyncio
    async def test_check_evolution_no_transition_needed(self) -> None:
        """Line 339: evolution check when already at META_LEARNER — no transition."""
        policy = EvolutionPolicy(
            min_observations=1,
            enabled=True,
        )
        manager = EnsembleManager(
            strategy=MetaLearner(),
            evolution_policy=policy,
        )
        await manager.register("m1", DummyModel("m1"))
        manager._performance_observations = 10
        event = await manager.check_evolution()
        assert event is None

    def test_build_strategy_unknown_type(self) -> None:
        """Line 372: _build_strategy with unknown strategy type raises."""
        manager = EnsembleManager(strategy=WeightedAverage())
        with pytest.raises(StrategyError, match="Unknown strategy type"):
            manager._build_strategy("nonexistent")  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_auto_prune_weight_references_unregistered_model(self) -> None:
        """Line 480: auto_prune skips weight entries not in _models."""
        s = WeightedAverage(weights={"m1": 0.9, "ghost": 0.8, "m2": 0.01})
        manager = EnsembleManager(strategy=s)
        await manager.register("m1", DummyModel("m1"))
        await manager.register("m2", DummyModel("m2"))
        # "ghost" is in weights but not registered
        result = await manager.auto_prune(weight_threshold=0.05, min_models=1)
        assert "ghost" not in result.kept_model_ids
        assert "ghost" not in result.pruned_model_ids

    @pytest.mark.asyncio
    async def test_auto_prune_keeps_above_threshold(self) -> None:
        """Line 487: model beyond min_models kept when weight >= threshold."""
        s = WeightedAverage(weights={"m1": 0.9, "m2": 0.8, "m3": 0.6})
        manager = EnsembleManager(strategy=s)
        await manager.register("m1", DummyModel("m1"))
        await manager.register("m2", DummyModel("m2"))
        await manager.register("m3", DummyModel("m3"))
        result = await manager.auto_prune(weight_threshold=0.05, min_models=2)
        # All 3 should be kept — m3 is above threshold
        assert len(result.kept_model_ids) == 3
        assert result.pruned_model_ids == []

    @pytest.mark.asyncio
    async def test_health_report_unhealthy_recommends_prune(self) -> None:
        """Line 618: unhealthy model gets 'prune' recommendation."""
        s = WeightedAverage(weights={"m1": 0.5})
        manager = EnsembleManager(strategy=s)
        model = DummyModel("m1", fail=False)
        await manager.register("m1", model)
        # Force model into ERROR phase (unhealthy)
        model._phase = __import__("artenic_ai_sdk.types", fromlist=["ModelPhase"]).ModelPhase.ERROR
        report = await manager.get_model_health_report("m1")
        assert report is not None
        assert report.recommendation == "prune"

    @pytest.mark.asyncio
    async def test_per_model_timeout(self) -> None:
        manager = EnsembleManager(strategy=WeightedAverage(), timeout_seconds=0.01)

        class SlowModel(DummyModel):
            async def predict(self, features: dict[str, Any]) -> BasePrediction:
                await asyncio.sleep(1.0)
                return _pred(0.9, "slow")

        await manager.register("slow", SlowModel("slow"))
        await manager.register("fast", DummyModel("fast"))

        # fast model should respond, slow should timeout
        manager.set_model_timeout("fast", 10.0)  # override for fast model
        result = await manager.predict({"a": 1.0})
        assert "fast" in result.models_responded
        assert "slow" in result.models_failed
