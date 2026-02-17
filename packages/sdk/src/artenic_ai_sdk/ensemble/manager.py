"""EnsembleManager — orchestrates an ensemble of BaseModel instances.

Handles model registration, parallel inference, strategy selection,
versioning, A/B testing, auto-pruning, and health reporting.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any, Literal

from artenic_ai_sdk.exceptions import (
    NoModelsRegisteredError,
    QuorumNotMetError,
    StrategyError,
)
from artenic_ai_sdk.schemas import (
    AutoPruneResult,
    BasePrediction,
    EnsembleABTest,
    EnsembleResult,
    EnsembleSnapshot,
    EvalResult,
    EvolutionEvent,
    ModelHealthReport,
)
from artenic_ai_sdk.types import (
    EnsemblePhase,
    EnsembleStrategyType,
    EvolutionTrigger,
)

from .strategies import (
    DynamicWeighting,
    EnsembleStrategy,
    MajorityVoting,
    MetaLearner,
    Stacking,
    WeightedAverage,
)

if TYPE_CHECKING:
    from artenic_ai_sdk.base_model import BaseModel

    from .evolution import EvolutionPolicy


class EnsembleManager:
    """Orchestrates an ensemble of BaseModel instances.

    Handles model registration, parallel inference, strategy selection,
    and auto-evolution.

    Example::

        manager = EnsembleManager(
            strategy=WeightedAverage(weights={"lgbm": 0.6, "t_mamba": 0.4}),
        )
        await manager.register("lgbm", lgbm_model)
        await manager.register("t_mamba", tmamba_model)

        result = await manager.predict(features)
    """

    def __init__(
        self,
        strategy: EnsembleStrategy,
        quorum: float = 0.5,
        timeout_seconds: float = 30.0,
        evolution_policy: EvolutionPolicy | None = None,
    ) -> None:
        """
        Args:
            strategy: The combination strategy to use.
            quorum: Minimum fraction of models that must respond (0.0-1.0).
            timeout_seconds: Max time to wait for model predictions.
            evolution_policy: Optional auto-evolution policy (disabled by default).
        """
        self._models: dict[str, BaseModel] = {}
        self._strategy = strategy
        self._quorum = quorum
        self._timeout = timeout_seconds
        self._model_timeouts: dict[str, float] = {}
        self._fallback_prediction: BasePrediction | None = None
        self._lock = asyncio.Lock()
        # Evolution, versioning, A/B testing
        self._evolution_policy = evolution_policy
        self._evolution_log: list[EvolutionEvent] = []
        self._performance_observations: int = 0
        self._version_history: list[EnsembleSnapshot] = []
        self._ab_tests: dict[str, EnsembleABTest] = {}

    # =========================================================================
    # Model registration
    # =========================================================================

    async def register(self, model_id: str, model: BaseModel) -> None:
        """Add a model to the ensemble (thread-safe).

        Args:
            model_id: Unique identifier for this model.
            model: The model instance.
        """
        async with self._lock:
            self._models[model_id] = model

    async def unregister(self, model_id: str) -> None:
        """Remove a model from the ensemble (thread-safe).

        Args:
            model_id: The model to remove.
        """
        async with self._lock:
            self._models.pop(model_id, None)

    def set_model_timeout(self, model_id: str, timeout_seconds: float) -> None:
        """Set a per-model timeout override.

        Args:
            model_id: The model to configure.
            timeout_seconds: Max time for this model's predict().
        """
        self._model_timeouts[model_id] = timeout_seconds

    def set_fallback_prediction(self, prediction: BasePrediction) -> None:
        """Set a fallback prediction for graceful degradation.

        When all models fail and a fallback is configured, the ensemble
        returns this prediction instead of raising QuorumNotMetError.

        Args:
            prediction: The fallback prediction to use.
        """
        self._fallback_prediction = prediction

    # =========================================================================
    # Inference
    # =========================================================================

    async def predict(
        self,
        features: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EnsembleResult:
        """Run ensemble inference.

        1. All registered models run predict() in parallel.
        2. Strategy combines results.
        3. Returns EnsembleResult with full details.

        Args:
            features: Input features for all models.
            context: Optional context for strategy-specific logic.

        Returns:
            EnsembleResult with combined prediction and per-model details.
        """
        if not self._models:
            raise NoModelsRegisteredError("No models registered in ensemble")

        start = time.perf_counter()

        selected_ids = list(self._models.keys())

        # Parallel inference
        predictions, failed = await self._run_parallel(selected_ids, features)

        # Check quorum (with fallback support)
        required = max(1, int(len(selected_ids) * self._quorum))
        if len(predictions) < required:
            # Graceful degradation: if ALL failed and fallback exists
            if not predictions and self._fallback_prediction is not None:
                elapsed_ms = (time.perf_counter() - start) * 1000
                return EnsembleResult(
                    confidence=self._fallback_prediction.confidence,
                    metadata={
                        **self._fallback_prediction.metadata,
                        "degraded": True,
                        "fallback": True,
                    },
                    model_id="ensemble",
                    model_version="0.1.0",
                    inference_time_ms=elapsed_ms,
                    strategy_used=self._strategy.strategy_type,
                    models_responded=[],
                    models_failed=failed,
                    individual_predictions={},
                )
            raise QuorumNotMetError(
                f"Quorum not met: {len(predictions)}/{required} models responded",
                required=required,
                responded=len(predictions),
            )

        # Combine
        combined = await self._strategy.combine(predictions, context)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return EnsembleResult(
            confidence=combined.confidence,
            metadata=combined.metadata,
            model_id="ensemble",
            model_version="0.1.0",
            inference_time_ms=elapsed_ms,
            strategy_used=self._strategy.strategy_type,
            models_responded=list(predictions.keys()),
            models_failed=failed,
            individual_predictions=predictions,
        )

    async def _run_parallel(
        self,
        model_ids: list[str],
        features: dict[str, Any],
    ) -> tuple[dict[str, BasePrediction], list[str]]:
        """Run predict() on multiple models concurrently."""
        predictions: dict[str, BasePrediction] = {}
        failed: list[str] = []

        async def _call_model(model_id: str) -> tuple[str, BasePrediction | None]:
            try:
                model_timeout = self._model_timeouts.get(model_id, self._timeout)
                result = await asyncio.wait_for(
                    self._models[model_id].predict(features),
                    timeout=model_timeout,
                )
                return model_id, result
            except Exception:
                return model_id, None

        tasks = [_call_model(mid) for mid in model_ids if mid in self._models]
        results = await asyncio.gather(*tasks)

        for model_id, result in results:
            if result is not None:
                predictions[model_id] = result
            else:
                failed.append(model_id)

        return predictions, failed

    # =========================================================================
    # Strategy management
    # =========================================================================

    async def swap_strategy(
        self,
        new_strategy: EnsembleStrategy,
        *,
        transfer_weights: bool = True,
    ) -> None:
        """Hot-swap the combination strategy without downtime.

        Args:
            new_strategy: The new strategy to use.
            transfer_weights: If True, export state from old strategy and
                import it into the new one to preserve learned weights.
        """
        async with self._lock:
            if transfer_weights:
                state = self._strategy.export_state()
                new_strategy.import_state(state)
            self._strategy = new_strategy

    @property
    def strategy(self) -> EnsembleStrategy:
        """Current active strategy."""
        return self._strategy

    # =========================================================================
    # Introspection
    # =========================================================================

    @property
    def models(self) -> dict[str, BaseModel]:
        """Registered models (read-only view)."""
        return dict(self._models)

    @property
    def model_count(self) -> int:
        """Number of registered models."""
        return len(self._models)

    def get_weights(self) -> dict[str, float]:
        """Current strategy weights."""
        return self._strategy.get_weights()

    async def update_model_performance(
        self,
        model_id: str,
        performance: EvalResult,
    ) -> EvolutionEvent | None:
        """Report evaluation results to update strategy weights.

        Args:
            model_id: The model that was evaluated.
            performance: The evaluation results.

        Returns:
            An EvolutionEvent if an auto-evolution was triggered, else None.
        """
        await self._strategy.update_weights(model_id, performance)
        self._performance_observations += 1
        return await self.check_evolution()

    # =========================================================================
    # Auto-Evolution
    # =========================================================================

    async def check_evolution(self) -> EvolutionEvent | None:
        """Check if the strategy should evolve based on the evolution policy.

        Returns:
            An EvolutionEvent if evolution occurred, else None.
        """
        policy = self._evolution_policy
        if not policy or not policy.enabled:
            return None
        if self._performance_observations < policy.min_observations:
            return None

        current = self._strategy.strategy_type
        model_count = len(self._models)
        target: EnsembleStrategyType | None = None

        if (
            current == EnsembleStrategyType.WEIGHTED_AVERAGE
            and model_count >= policy.dynamic_weighting_threshold
        ):
            target = EnsembleStrategyType.DYNAMIC_WEIGHTING
        elif (
            current == EnsembleStrategyType.DYNAMIC_WEIGHTING
            and model_count >= policy.meta_learner_threshold
        ):
            target = EnsembleStrategyType.META_LEARNER

        if target is None:
            return None

        new_strategy = self._build_strategy(target)
        state = self._strategy.export_state()
        new_strategy.import_state(state)

        event = EvolutionEvent(
            from_strategy=current,
            to_strategy=target,
            trigger=EvolutionTrigger.MODEL_COUNT,
            model_count=model_count,
            performance_data=dict(self._strategy.get_weights()),
        )

        async with self._lock:
            self._strategy = new_strategy
        self._evolution_log.append(event)
        return event

    def _build_strategy(self, strategy_type: EnsembleStrategyType) -> EnsembleStrategy:
        """Factory: create a new strategy instance by type."""
        if strategy_type == EnsembleStrategyType.WEIGHTED_AVERAGE:
            return WeightedAverage()
        if strategy_type == EnsembleStrategyType.DYNAMIC_WEIGHTING:
            metric = "accuracy"
            if self._evolution_policy:
                metric = self._evolution_policy.performance_metric
            return DynamicWeighting(performance_metric=metric)
        if strategy_type == EnsembleStrategyType.META_LEARNER:
            metric = "accuracy"
            if self._evolution_policy:
                metric = self._evolution_policy.performance_metric
            return MetaLearner(performance_metric=metric)
        if strategy_type == EnsembleStrategyType.MAJORITY_VOTING:
            return MajorityVoting()
        if strategy_type == EnsembleStrategyType.STACKING:
            metric = "accuracy"
            if self._evolution_policy:
                metric = self._evolution_policy.performance_metric
            return Stacking(performance_metric=metric)
        raise StrategyError(f"Unknown strategy type: {strategy_type}")

    @property
    def evolution_log(self) -> list[EvolutionEvent]:
        """Read-only copy of evolution events."""
        return list(self._evolution_log)

    @property
    def evolution_policy(self) -> EvolutionPolicy | None:
        """Current evolution policy (read-only)."""
        return self._evolution_policy

    # =========================================================================
    # Ensemble Versioning
    # =========================================================================

    async def snapshot(
        self,
        metadata: dict[str, Any] | None = None,
    ) -> EnsembleSnapshot:
        """Create an immutable snapshot of the current ensemble configuration.

        Args:
            metadata: Optional metadata to attach to the snapshot.

        Returns:
            An EnsembleSnapshot with a unique version_id.
        """
        async with self._lock:
            parent = self._version_history[-1].version_id if self._version_history else None
            snap = EnsembleSnapshot(
                version_id=str(uuid.uuid4()),
                phase=EnsemblePhase.DRAFT,
                strategy_type=self._strategy.strategy_type,
                strategy_weights=self._strategy.get_weights(),
                model_ids=list(self._models.keys()),
                quorum=self._quorum,
                metadata=metadata or {},
                parent_version_id=parent,
            )
            self._version_history.append(snap)
        return snap

    async def restore_snapshot(self, version_id: str) -> None:
        """Restore ensemble state from a previously saved snapshot.

        Args:
            version_id: The snapshot to restore.

        Raises:
            ValueError: If the version_id is not found.
        """
        target: EnsembleSnapshot | None = None
        for snap in self._version_history:
            if snap.version_id == version_id:
                target = snap
                break
        if target is None:
            raise ValueError(f"Snapshot not found: {version_id}")

        new_strategy = self._build_strategy(target.strategy_type)
        new_strategy.import_state({"weights": dict(target.strategy_weights)})

        async with self._lock:
            self._strategy = new_strategy
            self._quorum = target.quorum

    def get_version_history(self) -> list[EnsembleSnapshot]:
        """Return all snapshots (read-only copies)."""
        return list(self._version_history)

    # =========================================================================
    # Auto-Pruning
    # =========================================================================

    async def auto_prune(
        self,
        weight_threshold: float = 0.05,
        min_models: int = 2,
    ) -> AutoPruneResult:
        """Remove underperforming models from the ensemble.

        Models with weight below ``weight_threshold`` are pruned,
        but at least ``min_models`` are always kept.

        Args:
            weight_threshold: Minimum weight to keep a model.
            min_models: Minimum number of models to retain.

        Returns:
            AutoPruneResult with lists of pruned and kept models.
        """
        weights = self._strategy.get_weights()
        if not weights:
            return AutoPruneResult(
                kept_model_ids=list(self._models.keys()),
                weight_threshold_used=weight_threshold,
            )

        # Sort by weight descending
        ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)

        kept: list[str] = []
        pruned: list[str] = []
        reasons: dict[str, str] = {}

        for model_id, weight in ranked:
            if model_id not in self._models:
                continue
            if len(kept) < min_models:
                kept.append(model_id)
            elif weight < weight_threshold:
                pruned.append(model_id)
                reasons[model_id] = f"weight {weight:.4f} below threshold {weight_threshold}"
            else:
                kept.append(model_id)

        # Unregister pruned models
        for model_id in pruned:
            async with self._lock:
                self._models.pop(model_id, None)

        return AutoPruneResult(
            pruned_model_ids=pruned,
            kept_model_ids=kept,
            reason=reasons,
            weight_threshold_used=weight_threshold,
        )

    # =========================================================================
    # Ensemble A/B Testing
    # =========================================================================

    async def create_ensemble_ab_test(
        self,
        test_id: str,
        variant_a: EnsembleSnapshot,
        variant_b: EnsembleSnapshot,
        traffic_split: float = 0.5,
    ) -> EnsembleABTest:
        """Create an A/B test between two ensemble configurations.

        Args:
            test_id: Unique test identifier.
            variant_a: First ensemble configuration.
            variant_b: Second ensemble configuration.
            traffic_split: Fraction of traffic routed to variant A (0.0-1.0).

        Returns:
            The created EnsembleABTest.
        """
        test = EnsembleABTest(
            test_id=test_id,
            variant_a=variant_a,
            variant_b=variant_b,
            traffic_split=traffic_split,
        )
        self._ab_tests[test_id] = test
        return test

    def select_ab_variant(
        self,
        test_id: str,
        random_value: float = 0.5,
    ) -> Literal["a", "b"]:
        """Select which A/B test variant to use for a request.

        Args:
            test_id: The A/B test identifier.
            random_value: A random float in [0, 1) for traffic routing.

        Returns:
            "a" or "b" indicating the selected variant.

        Raises:
            KeyError: If the test_id is not found.
        """
        test = self._ab_tests.get(test_id)
        if test is None:
            raise KeyError(f"A/B test not found: {test_id}")
        return "a" if random_value < test.traffic_split else "b"

    async def conclude_ensemble_ab_test(
        self,
        test_id: str,
        winner: Literal["a", "b"],
    ) -> EnsembleABTest:
        """Conclude an A/B test and apply the winning configuration.

        Args:
            test_id: The A/B test to conclude.
            winner: Which variant won ("a" or "b").

        Returns:
            The concluded EnsembleABTest with winner set.

        Raises:
            KeyError: If the test_id is not found.
        """
        test = self._ab_tests.get(test_id)
        if test is None:
            raise KeyError(f"A/B test not found: {test_id}")

        test.winner = winner
        winning_snap = test.variant_a if winner == "a" else test.variant_b

        # Apply winning config
        new_strategy = self._build_strategy(winning_snap.strategy_type)
        new_strategy.import_state({"weights": dict(winning_snap.strategy_weights)})

        async with self._lock:
            self._strategy = new_strategy
            self._quorum = winning_snap.quorum

        return test

    # =========================================================================
    # Model Health Reports
    # =========================================================================

    async def get_model_health_report(
        self,
        model_id: str,
    ) -> ModelHealthReport | None:
        """Generate a health report for a specific model.

        Combines the model's own health_check() with ensemble metadata
        (weight, performance trend) to produce a recommendation.

        Args:
            model_id: The model to report on.

        Returns:
            A ModelHealthReport, or None if the model is not registered.
        """
        model = self._models.get(model_id)
        if model is None:
            return None

        health = await model.health_check()
        weights = self._strategy.get_weights()
        weight = weights.get(model_id)

        # Determine recommendation
        recommendation: Literal["keep", "monitor", "prune"] = "keep"
        if health.status == "unhealthy":
            recommendation = "prune"
        elif health.status == "degraded" or (weight is not None and weight < 0.05):
            recommendation = "monitor"

        return ModelHealthReport(
            model_id=model_id,
            model_version=model.model_version,
            health_check=health,
            weight_in_ensemble=weight,
            recommendation=recommendation,
        )
