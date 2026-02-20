"""Ensemble combination strategies.

The ensemble system evolves incrementally:
    Phase 1 (1-2 models)  -> WeightedAverage (fixed weights)
    Phase 2 (3-4 models)  -> DynamicWeighting (performance-adjusted)
    Phase 3 (4+ models)   -> MetaLearner (attention-based)

Strategies are hot-swappable at runtime via EnsembleManager.swap_strategy().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Any

from artenic_ai_sdk.exceptions import StrategyError
from artenic_ai_sdk.schemas import BasePrediction, EvalResult
from artenic_ai_sdk.types import EnsembleStrategyType

# =============================================================================
# Strategy ABC
# =============================================================================


class EnsembleStrategy(ABC):
    """Base class for ensemble combination strategies.

    Strategies are stateful (they track weights) and hot-swappable.
    """

    @property
    @abstractmethod
    def strategy_type(self) -> EnsembleStrategyType:
        """Identifier for this strategy."""
        ...  # pragma: no cover

    @abstractmethod
    async def combine(
        self,
        predictions: dict[str, BasePrediction],
        context: dict[str, Any] | None = None,
    ) -> BasePrediction:
        """Combine multiple model predictions into a single result.

        Args:
            predictions: Mapping of model_id -> prediction.
            context: Optional context for strategy-specific logic.

        Returns:
            Combined prediction.
        """
        ...  # pragma: no cover

    @abstractmethod
    async def update_weights(
        self,
        model_id: str,
        performance: EvalResult,
    ) -> None:
        """Update internal weights after a model evaluation.

        Args:
            model_id: The model that was evaluated.
            performance: Evaluation results.
        """
        ...  # pragma: no cover

    @abstractmethod
    def get_weights(self) -> dict[str, float]:
        """Return current weight assignments."""
        ...  # pragma: no cover

    def export_state(self) -> dict[str, Any]:
        """Export internal state for transfer to another strategy.

        Default implementation exports weights. Subclasses may override
        to include additional state (history, meta-model params, etc.).
        """
        return {"weights": self.get_weights()}

    def import_state(self, state: dict[str, Any]) -> None:  # noqa: B027
        """Import state from another strategy.

        Used during hot-swap to preserve learned weights across strategy
        changes. Default is a no-op -- subclasses override as needed.
        """


# =============================================================================
# Concrete Strategies
# =============================================================================


class WeightedAverage(EnsembleStrategy):
    """Fixed-weight combination. Phase 1 (1-2 models).

    Simple weighted average of confidence scores. Weights are set
    manually and remain constant until explicitly changed.

    Example::

        strategy = WeightedAverage(weights={"lgbm": 0.6, "t_mamba": 0.4})
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self._weights: dict[str, float] = weights or {}

    @property
    def strategy_type(self) -> EnsembleStrategyType:
        return EnsembleStrategyType.WEIGHTED_AVERAGE

    async def combine(
        self,
        predictions: dict[str, BasePrediction],
        context: dict[str, Any] | None = None,
    ) -> BasePrediction:
        if not predictions:
            raise StrategyError("No predictions to combine")

        # Normalize weights to available models
        available = {k: v for k, v in self._weights.items() if k in predictions}
        if not available:
            # Equal weights fallback
            n = len(predictions)
            available = {k: 1.0 / n for k in predictions}

        total_weight = sum(available.values())
        if total_weight == 0:
            raise StrategyError("All weights are zero")

        # Weighted average of confidence
        combined_confidence = sum(
            predictions[mid].confidence * (w / total_weight) for mid, w in available.items()
        )

        # Use the prediction from the highest-weighted model as the base
        base_model_id = max(available, key=lambda k: available[k])
        base = predictions[base_model_id]

        return BasePrediction(
            confidence=combined_confidence,
            metadata={"weights": available, "strategy": "weighted_average"},
            model_id="ensemble",
            model_version=base.model_version,
            inference_time_ms=max(p.inference_time_ms for p in predictions.values()),
        )

    async def update_weights(
        self,
        model_id: str,
        performance: EvalResult,
    ) -> None:
        # Fixed weights -- no auto-update. Use set_weights() to change.
        pass

    def set_weights(self, weights: dict[str, float]) -> None:
        """Manually set weights."""
        self._weights = weights

    def get_weights(self) -> dict[str, float]:
        return dict(self._weights)

    def import_state(self, state: dict[str, Any]) -> None:
        """Import weights from a previous strategy."""
        weights = state.get("weights", {})
        if weights:
            self._weights = dict(weights)


class DynamicWeighting(EnsembleStrategy):
    """Performance-adjusted weights. Phase 2 (3-4 models).

    Weights are recalculated based on a rolling window of evaluation
    results. Recent performance is weighted more heavily via exponential
    decay.
    """

    def __init__(
        self,
        lookback_window: int = 20,
        performance_metric: str = "accuracy",
        decay_factor: float = 0.9,
        min_weight: float = 0.05,
    ) -> None:
        self._lookback = lookback_window
        self._metric = performance_metric
        self._decay = decay_factor
        self._min_weight = min_weight
        self._history: dict[str, deque[float]] = {}
        self._weights: dict[str, float] = {}

    @property
    def strategy_type(self) -> EnsembleStrategyType:
        return EnsembleStrategyType.DYNAMIC_WEIGHTING

    async def combine(
        self,
        predictions: dict[str, BasePrediction],
        context: dict[str, Any] | None = None,
    ) -> BasePrediction:
        if not predictions:
            raise StrategyError("No predictions to combine")

        # Use computed weights, fallback to equal
        available = {k: self._weights.get(k, 1.0) for k in predictions}
        total = sum(available.values())
        if total == 0:
            total = float(len(available))
            available = {k: 1.0 for k in available}

        combined_confidence = sum(
            predictions[mid].confidence * (w / total) for mid, w in available.items()
        )

        base_model_id = max(available, key=lambda k: available[k])
        base = predictions[base_model_id]

        return BasePrediction(
            confidence=combined_confidence,
            metadata={
                "weights": {k: round(v / total, 4) for k, v in available.items()},
                "strategy": "dynamic_weighting",
                "metric": self._metric,
            },
            model_id="ensemble",
            model_version=base.model_version,
            inference_time_ms=max(p.inference_time_ms for p in predictions.values()),
        )

    async def update_weights(
        self,
        model_id: str,
        performance: EvalResult,
    ) -> None:
        score = performance.metrics.get(self._metric, 0.0)

        if model_id not in self._history:
            self._history[model_id] = deque(maxlen=self._lookback)
        self._history[model_id].append(score)

        # Recompute all weights using exponential decay
        self._weights = self._compute_weights()

    def _compute_weights(self) -> dict[str, float]:
        weights: dict[str, float] = {}
        for model_id, scores in self._history.items():
            if not scores:
                weights[model_id] = self._min_weight
                continue

            # Exponential decay: most recent scores matter more
            weighted_sum = 0.0
            decay_sum = 0.0
            for i, score in enumerate(scores):
                decay = self._decay ** (len(scores) - 1 - i)
                weighted_sum += score * decay
                decay_sum += decay

            avg = weighted_sum / max(decay_sum, 1e-9)
            weights[model_id] = max(avg, self._min_weight)

        return weights

    def get_weights(self) -> dict[str, float]:
        return dict(self._weights)

    def import_state(self, state: dict[str, Any]) -> None:
        """Import weights from a previous strategy as initial seeds."""
        weights = state.get("weights", {})
        if weights:
            self._weights = dict(weights)
            # Initialize history from imported weights
            for model_id, w in weights.items():
                if model_id not in self._history:
                    self._history[model_id] = deque(maxlen=self._lookback)
                    self._history[model_id].append(w)


class _AttentionGatingNetwork:
    """Small attention network that learns per-model trust weights.

    Architecture: Linear -> ReLU -> LayerNorm -> MultiheadAttention -> Linear -> Softmax.
    Produces attention weights (batch, n_models) given model feature vectors.

    Uses lazy torch import -- only instantiated when a meta-model is built.
    """

    def __init__(
        self,
        n_models: int,
        feature_dim: int = 8,
        hidden_dim: int = 32,
        num_heads: int = 4,
    ) -> None:
        import torch.nn as nn

        self.n_models = n_models
        self.feature_dim = feature_dim

        self.input_proj = nn.Linear(feature_dim * n_models, hidden_dim)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.output_proj = nn.Linear(hidden_dim, n_models)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, model_features: Any) -> Any:
        """Compute attention weights from model feature vectors.

        Args:
            model_features: Tensor of shape (batch, n_models * feature_dim).

        Returns:
            Attention weights of shape (batch, n_models).
        """
        x = self.relu(self.input_proj(model_features))
        x = self.norm(x)
        # Self-attention over the hidden representation
        x_seq = x.unsqueeze(1)  # (batch, 1, hidden)
        attn_out, _ = self.attention(x_seq, x_seq, x_seq)
        attn_out = attn_out.squeeze(1)  # (batch, hidden)
        logits = self.output_proj(attn_out)  # (batch, n_models)
        return self.softmax(logits)

    def parameters(self) -> Any:
        """Return all learnable parameters (for optimizer)."""
        import itertools

        return itertools.chain(
            self.input_proj.parameters(),
            self.norm.parameters(),
            self.attention.parameters(),
            self.output_proj.parameters(),
        )


class MetaLearner(EnsembleStrategy):
    """Attention-based meta-learner. Phase 3 (4+ models).

    Learns which model to trust based on prediction context.
    Uses a small attention network trained on historical
    prediction-outcome pairs.

    Note: The actual neural network is loaded lazily -- this class
    provides the interface and falls back to performance-weighted
    combination when no trained meta-model is available.
    """

    def __init__(
        self,
        attention_heads: int = 4,
        performance_metric: str = "accuracy",
        feature_dim: int = 8,
    ) -> None:
        self._attention_heads = attention_heads
        self._metric = performance_metric
        self._feature_dim = feature_dim
        self._weights: dict[str, float] = {}
        self._meta_model: _AttentionGatingNetwork | None = None

    @property
    def strategy_type(self) -> EnsembleStrategyType:
        return EnsembleStrategyType.META_LEARNER

    async def combine(
        self,
        predictions: dict[str, BasePrediction],
        context: dict[str, Any] | None = None,
    ) -> BasePrediction:
        if not predictions:
            raise StrategyError("No predictions to combine")

        if self._meta_model is not None:
            return await self._combine_with_meta_model(predictions, context)

        # Fallback: weighted by tracked performance
        return await self._combine_fallback(predictions)

    async def _combine_with_meta_model(
        self,
        predictions: dict[str, BasePrediction],
        context: dict[str, Any] | None,
    ) -> BasePrediction:
        """Use the trained attention network to combine predictions."""
        try:
            import torch
        except ImportError:
            return await self._combine_fallback(predictions)

        model_ids = sorted(predictions.keys())

        # Build feature vector per model: [confidence, latency_norm, perf_weight, ...]
        features_list: list[float] = []
        max_latency = max(p.inference_time_ms for p in predictions.values()) or 1.0

        for mid in model_ids:
            pred = predictions[mid]
            feat = [
                pred.confidence,
                pred.inference_time_ms / max_latency,
                self._weights.get(mid, 0.5),
            ]
            # Pad to feature_dim
            while len(feat) < self._feature_dim:
                feat.append(0.0)
            feat = feat[: self._feature_dim]
            features_list.extend(feat)

        # Pad if fewer models than expected
        assert self._meta_model is not None  # for type checker
        expected_len = self._meta_model.n_models * self._meta_model.feature_dim
        while len(features_list) < expected_len:
            features_list.append(0.0)
        features_list = features_list[:expected_len]

        input_tensor = torch.tensor([features_list], dtype=torch.float32)

        with torch.no_grad():
            attention_weights = self._meta_model.forward(input_tensor)

        weights_np = attention_weights[0].tolist()

        # Map back to model_ids (truncate to actual models count)
        weight_map: dict[str, float] = {}
        combined_confidence = 0.0
        for i, mid in enumerate(model_ids):
            if i < len(weights_np):
                w: float = weights_np[i]
                weight_map[mid] = round(w, 4)
                combined_confidence += predictions[mid].confidence * w

        return BasePrediction(
            confidence=combined_confidence,
            metadata={
                "weights": weight_map,
                "strategy": "meta_learner",
                "meta_model_active": True,
            },
            model_id="ensemble",
            model_version="0.1.0",
            inference_time_ms=max(p.inference_time_ms for p in predictions.values()),
        )

    async def _combine_fallback(
        self,
        predictions: dict[str, BasePrediction],
    ) -> BasePrediction:
        """Fallback combination using tracked performance weights."""
        available = {k: self._weights.get(k, 1.0) for k in predictions}
        total = sum(available.values()) or float(len(available))

        combined_confidence = sum(
            predictions[mid].confidence * (available[mid] / total) for mid in available
        )

        return BasePrediction(
            confidence=combined_confidence,
            metadata={
                "weights": {k: round(v / total, 4) for k, v in available.items()},
                "strategy": "meta_learner",
                "meta_model_active": False,
            },
            model_id="ensemble",
            model_version="0.1.0",
            inference_time_ms=max(p.inference_time_ms for p in predictions.values()),
        )

    async def update_weights(
        self,
        model_id: str,
        performance: EvalResult,
    ) -> None:
        score = performance.metrics.get(self._metric, 0.0)
        self._weights[model_id] = score

    def build_meta_model(self, n_models: int, feature_dim: int | None = None) -> None:
        """Create a fresh attention gating network.

        Args:
            n_models: Number of models in the ensemble.
            feature_dim: Feature dimension per model. Uses instance default if None.
        """
        dim = feature_dim or self._feature_dim
        self._meta_model = _AttentionGatingNetwork(
            n_models=n_models,
            feature_dim=dim,
            num_heads=self._attention_heads,
        )

    async def load_meta_model(self, path: Any) -> None:
        """Load a trained meta-learner model from disk.

        Args:
            path: Path to the saved meta-model.
        """
        try:
            import torch

            self._meta_model = torch.load(path, weights_only=True)
        except ImportError:
            pass

    def get_weights(self) -> dict[str, float]:
        return dict(self._weights)

    def import_state(self, state: dict[str, Any]) -> None:
        """Import weights from a previous strategy as performance seeds."""
        weights = state.get("weights", {})
        if weights:
            self._weights = dict(weights)


# =============================================================================
# MajorityVoting — Phase 4 (classification ensembles)
# =============================================================================


class MajorityVoting(EnsembleStrategy):
    """Majority voting for classification ensembles.

    Each model's prediction should include ``metadata["predicted_class"]``.
    Falls back to highest-confidence model if no class info is available.

    Confidence is the fraction of models that voted for the winning class.
    """

    def __init__(self) -> None:
        self._weights: dict[str, float] = {}
        self._correct_counts: dict[str, int] = {}
        self._total_counts: dict[str, int] = {}

    @property
    def strategy_type(self) -> EnsembleStrategyType:
        return EnsembleStrategyType.MAJORITY_VOTING

    async def combine(
        self,
        predictions: dict[str, BasePrediction],
        context: dict[str, Any] | None = None,
    ) -> BasePrediction:
        if not predictions:
            raise StrategyError("No predictions to combine")

        # Extract predicted classes
        votes: dict[str, list[str]] = {}  # class -> list of model_ids
        for model_id, pred in predictions.items():
            cls = pred.metadata.get("predicted_class")
            if cls is not None:
                cls_str = str(cls)
                votes.setdefault(cls_str, []).append(model_id)

        if not votes:
            # Fallback: pick highest-confidence model
            best_mid = max(predictions, key=lambda k: predictions[k].confidence)
            base = predictions[best_mid]
            return BasePrediction(
                confidence=base.confidence,
                metadata={
                    "strategy": "majority_voting",
                    "fallback": True,
                    "selected_model": best_mid,
                },
                model_id="ensemble",
                model_version=base.model_version,
                inference_time_ms=max(p.inference_time_ms for p in predictions.values()),
            )

        # Find the majority class (tie-break by average confidence)
        def _vote_key(cls: str) -> tuple[int, float]:
            voter_ids = votes[cls]
            avg_conf = sum(predictions[m].confidence for m in voter_ids) / len(voter_ids)
            return (len(voter_ids), avg_conf)

        winning_class = max(votes, key=_vote_key)
        winning_voters = votes[winning_class]
        vote_fraction = len(winning_voters) / len(predictions)

        # Use the highest-confidence voter as the base
        best_voter = max(winning_voters, key=lambda m: predictions[m].confidence)
        base = predictions[best_voter]

        return BasePrediction(
            confidence=vote_fraction,
            metadata={
                "strategy": "majority_voting",
                "predicted_class": winning_class,
                "vote_counts": {cls: len(mids) for cls, mids in votes.items()},
                "total_voters": len(predictions),
            },
            model_id="ensemble",
            model_version=base.model_version,
            inference_time_ms=max(p.inference_time_ms for p in predictions.values()),
        )

    async def update_weights(
        self,
        model_id: str,
        performance: EvalResult,
    ) -> None:
        accuracy = performance.metrics.get("accuracy", 0.0)
        self._correct_counts[model_id] = self._correct_counts.get(model_id, 0) + (
            1 if accuracy > 0.5 else 0
        )
        self._total_counts[model_id] = self._total_counts.get(model_id, 0) + 1

        # Weight = accuracy rate
        total = self._total_counts[model_id]
        correct = self._correct_counts[model_id]
        self._weights[model_id] = correct / total if total > 0 else 0.0

    def get_weights(self) -> dict[str, float]:
        return dict(self._weights)

    def export_state(self) -> dict[str, Any]:
        return {
            "weights": self.get_weights(),
            "correct_counts": dict(self._correct_counts),
            "total_counts": dict(self._total_counts),
        }

    def import_state(self, state: dict[str, Any]) -> None:
        weights = state.get("weights", {})
        if weights:
            self._weights = dict(weights)
        self._correct_counts = dict(state.get("correct_counts", {}))
        self._total_counts = dict(state.get("total_counts", {}))


# =============================================================================
# Stacking — Phase 5 (meta-model)
# =============================================================================


class Stacking(EnsembleStrategy):
    """Stacking ensemble — meta-model combines base predictions.

    Uses a simple linear combination trained on historical predictions.
    No torch dependency — uses pure Python math (closed-form least squares).
    """

    def __init__(self, performance_metric: str = "accuracy") -> None:
        self._metric = performance_metric
        self._weights: dict[str, float] = {}
        self._bias: float = 0.0
        self._history: list[tuple[dict[str, float], float]] = []

    @property
    def strategy_type(self) -> EnsembleStrategyType:
        return EnsembleStrategyType.STACKING

    async def combine(
        self,
        predictions: dict[str, BasePrediction],
        context: dict[str, Any] | None = None,
    ) -> BasePrediction:
        if not predictions:
            raise StrategyError("No predictions to combine")

        model_ids = sorted(predictions.keys())

        if self._weights:
            # Use learned meta-weights
            combined = self._bias
            total_w = 0.0
            for mid in model_ids:
                w = self._weights.get(mid, 1.0 / len(model_ids))
                combined += predictions[mid].confidence * w
                total_w += abs(w)
            # Clamp to [0, 1]
            combined = max(0.0, min(1.0, combined))
        else:
            # Equal weights fallback
            combined = sum(predictions[m].confidence for m in model_ids) / len(model_ids)
            total_w = 1.0

        base_model_id = max(model_ids, key=lambda k: self._weights.get(k, 0.0))
        base = predictions[base_model_id]

        return BasePrediction(
            confidence=combined,
            metadata={
                "strategy": "stacking",
                "weights": {
                    k: round(self._weights.get(k, 1.0 / len(model_ids)), 4) for k in model_ids
                },
                "bias": round(self._bias, 4),
                "fitted": bool(self._weights),
            },
            model_id="ensemble",
            model_version=base.model_version,
            inference_time_ms=max(p.inference_time_ms for p in predictions.values()),
        )

    async def update_weights(
        self,
        model_id: str,
        performance: EvalResult,
    ) -> None:
        score = performance.metrics.get(self._metric, 0.0)
        self._weights[model_id] = score

    def fit(self, training_data: list[dict[str, Any]]) -> None:
        """Fit meta-weights on historical (predictions, outcome) pairs.

        Each entry in training_data should have:
            - "predictions": dict[str, float] — model_id -> confidence
            - "outcome": float — the actual outcome (0 or 1, or continuous)

        Uses closed-form least squares: w = (X^T X)^{-1} X^T y.
        Falls back to equal weights if not enough data or singular matrix.
        """
        if len(training_data) < 2:
            return

        # Collect model_ids from all entries
        all_model_ids: set[str] = set()
        for entry in training_data:
            all_model_ids.update(entry.get("predictions", {}).keys())
        model_ids = sorted(all_model_ids)

        if not model_ids:
            return

        n_samples = len(training_data)
        n_features = len(model_ids) + 1  # +1 for bias

        # Build design matrix and target vector
        design: list[list[float]] = []
        targets: list[float] = []
        for entry in training_data:
            preds = entry.get("predictions", {})
            row = [preds.get(mid, 0.0) for mid in model_ids]
            row.append(1.0)  # bias term
            design.append(row)
            targets.append(float(entry.get("outcome", 0.0)))

        # Normal equations: (X^T X) w = X^T y
        xtx = [
            [sum(design[k][i] * design[k][j] for k in range(n_samples)) for j in range(n_features)]
            for i in range(n_features)
        ]
        xty = [sum(design[k][i] * targets[k] for k in range(n_samples)) for i in range(n_features)]

        # Solve via Gaussian elimination
        w = self._solve_linear(xtx, xty)
        if w is None:
            return

        for i, mid in enumerate(model_ids):
            self._weights[mid] = w[i]
        self._bias = w[-1]

        # Store history for export
        self._history = [
            (entry.get("predictions", {}), float(entry.get("outcome", 0.0)))
            for entry in training_data
        ]

    @staticmethod
    def _solve_linear(
        coefficients: list[list[float]],
        rhs: list[float],
    ) -> list[float] | None:
        """Solve a linear system via Gaussian elimination with partial pivoting."""
        n = len(rhs)
        # Augmented matrix
        aug = [[*coefficients[i][:], rhs[i]] for i in range(n)]

        for col in range(n):
            # Partial pivoting
            max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
            if abs(aug[max_row][col]) < 1e-12:
                return None  # Singular
            aug[col], aug[max_row] = aug[max_row], aug[col]

            # Eliminate below
            for row in range(col + 1, n):
                factor = aug[row][col] / aug[col][col]
                for j in range(col, n + 1):
                    aug[row][j] -= factor * aug[col][j]

        # Back substitution
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = sum(aug[i][j] * x[j] for j in range(i + 1, n))
            x[i] = (aug[i][n] - s) / aug[i][i]

        return x

    def get_weights(self) -> dict[str, float]:
        return dict(self._weights)

    def export_state(self) -> dict[str, Any]:
        return {
            "weights": self.get_weights(),
            "bias": self._bias,
            "history_size": len(self._history),
        }

    def import_state(self, state: dict[str, Any]) -> None:
        weights = state.get("weights", {})
        if weights:
            self._weights = dict(weights)
        self._bias = float(state.get("bias", 0.0))
