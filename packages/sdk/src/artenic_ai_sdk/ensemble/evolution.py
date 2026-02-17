"""Ensemble evolution policy — controls automatic strategy upgrades."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvolutionPolicy:
    """Controls when ensemble strategy auto-evolves.

    The ensemble starts with WeightedAverage and progresses to more
    sophisticated strategies as model count and data increase.

    Thresholds:
        - ``dynamic_weighting_threshold``: model count to switch WA -> DW.
        - ``meta_learner_threshold``: model count to switch DW -> ML.
        - ``min_observations``: minimum update_model_performance() calls
          before any evolution is considered.
    """

    dynamic_weighting_threshold: int = 3
    meta_learner_threshold: int = 5
    min_observations: int = 10
    performance_metric: str = "accuracy"
    enabled: bool = False
