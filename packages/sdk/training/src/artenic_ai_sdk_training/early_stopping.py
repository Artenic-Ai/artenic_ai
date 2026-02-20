"""Early Stopping — patience-based training termination."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from artenic_ai_sdk_training.callbacks import TrainingCallback, TrainingContext

if TYPE_CHECKING:
    from artenic_ai_sdk_training.config import EarlyStoppingConfig

logger = logging.getLogger(__name__)


class EarlyStopping(TrainingCallback):
    """Stop training when a monitored metric stops improving.

    Tracks the best observed value and counts consecutive epochs
    without improvement exceeding ``min_delta``.  When the counter
    reaches ``patience``, ``should_stop()`` returns True.
    """

    def __init__(self, config: EarlyStoppingConfig) -> None:
        self._config = config
        self._best_value: float | None = None
        self._best_epoch: int | None = None
        self._counter: int = 0
        self._stopped: bool = False

    @property
    def best_value(self) -> float | None:
        return self._best_value

    @property
    def best_epoch(self) -> int | None:
        return self._best_epoch

    @property
    def epochs_without_improvement(self) -> int:
        return self._counter

    def on_epoch_end(
        self,
        epoch: int,
        metrics: dict[str, float],
        context: TrainingContext,
    ) -> None:
        value = metrics.get(self._config.metric)
        if value is None:
            logger.warning(
                "EarlyStopping: metric %r not found in epoch %d metrics",
                self._config.metric,
                epoch,
            )
            return

        if self._is_improvement(value):
            self._best_value = value
            self._best_epoch = epoch
            self._counter = 0
            context.best_metric = value
            context.best_epoch = epoch
        else:
            self._counter += 1

        if self._counter >= self._config.patience:
            self._stopped = True
            context.stop_requested = True
            logger.info(
                "EarlyStopping: no improvement for %d epochs, stopping at epoch %d",
                self._config.patience,
                epoch,
            )

    def should_stop(self) -> bool:
        return self._stopped

    def _is_improvement(self, value: float) -> bool:
        if self._best_value is None:
            return True
        if self._config.mode == "min":
            return value < self._best_value - self._config.min_delta
        return value > self._best_value + self._config.min_delta
