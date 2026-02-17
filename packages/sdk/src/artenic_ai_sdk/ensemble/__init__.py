"""Ensemble management: strategies, evolution, and orchestration.

Re-exports the public API from sub-modules.
"""

from artenic_ai_sdk.ensemble.evolution import EvolutionPolicy
from artenic_ai_sdk.ensemble.manager import EnsembleManager
from artenic_ai_sdk.ensemble.strategies import (
    DynamicWeighting,
    EnsembleStrategy,
    MajorityVoting,
    MetaLearner,
    Stacking,
    WeightedAverage,
)

__all__ = [
    "DynamicWeighting",
    "EnsembleManager",
    "EnsembleStrategy",
    "EvolutionPolicy",
    "MajorityVoting",
    "MetaLearner",
    "Stacking",
    "WeightedAverage",
]
