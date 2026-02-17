"""Training Intelligence — SDK utilities for smart, efficient training.

All features are opt-in and configured via ``TrainingConfig``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from artenic_ai_sdk.training.callbacks import (
    CallbackRunner,
    TrainingCallback,
    TrainingContext,
)
from artenic_ai_sdk.training.config import (
    CheckpointConfig,
    DataSplitConfig,
    DataVersioningConfig,
    DistributedConfig,
    EarlyStoppingConfig,
    GradientCheckpointConfig,
    LRFinderConfig,
    MixedPrecisionConfig,
    TrainingConfig,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from pathlib import Path

__all__ = [
    "CallbackRunner",
    "CheckpointConfig",
    "DataSplitConfig",
    "DataVersioningConfig",
    "DatasetVersioner",
    "DistributedConfig",
    "EarlyStopping",
    "EarlyStoppingConfig",
    "FSDPManager",
    "GradientCheckpointConfig",
    "GradientCheckpointManager",
    "LRFinderConfig",
    "LearningRateFinder",
    "MixedPrecisionConfig",
    "MixedPrecisionManager",
    "SmartCheckpointer",
    # Features (lazy imports)
    "SmartDataSplitter",
    # Callbacks
    "TrainingCallback",
    # Config
    "TrainingConfig",
    "TrainingContext",
    # Factory
    "build_callbacks",
]


def build_callbacks(
    config: TrainingConfig,
    save_fn: Callable[[Path], Coroutine[Any, Any, None]] | None = None,
    output_dir: Path | None = None,
) -> CallbackRunner:
    """Build the callback chain from a TrainingConfig.

    Inspects config sub-sections and creates the appropriate
    callbacks for enabled features.
    """
    from pathlib import Path as _Path

    _output_dir = output_dir or _Path("/artifacts")

    callbacks: list[TrainingCallback] = []

    if config.early_stopping.enabled:
        from artenic_ai_sdk.training.early_stopping import EarlyStopping

        callbacks.append(EarlyStopping(config.early_stopping))

    if config.checkpoint.enabled:
        from artenic_ai_sdk.training.checkpointing import SmartCheckpointer

        callbacks.append(
            SmartCheckpointer(
                config=config.checkpoint,
                save_fn=save_fn,
                output_dir=_output_dir,
            )
        )

    return CallbackRunner(callbacks)


def __getattr__(name: str) -> Any:
    """Lazy imports for feature classes to avoid heavy dependencies at import time."""
    _lazy = {
        "SmartDataSplitter": "artenic_ai_sdk.training.data_splitting",
        "DataSplitResult": "artenic_ai_sdk.training.data_splitting",
        "DatasetVersioner": "artenic_ai_sdk.training.data_versioning",
        "DatasetVersion": "artenic_ai_sdk.training.data_versioning",
        "MixedPrecisionManager": "artenic_ai_sdk.training.mixed_precision",
        "PrecisionMode": "artenic_ai_sdk.training.mixed_precision",
        "GradientCheckpointManager": "artenic_ai_sdk.training.gradient_checkpoint",
        "SmartCheckpointer": "artenic_ai_sdk.training.checkpointing",
        "EarlyStopping": "artenic_ai_sdk.training.early_stopping",
        "LearningRateFinder": "artenic_ai_sdk.training.lr_finder",
        "LRFinderResult": "artenic_ai_sdk.training.lr_finder",
        "FSDPManager": "artenic_ai_sdk.training.distributed",
        "DistributedContext": "artenic_ai_sdk.training.distributed",
    }
    if name in _lazy:
        import importlib

        module = importlib.import_module(_lazy[name])
        return getattr(module, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
