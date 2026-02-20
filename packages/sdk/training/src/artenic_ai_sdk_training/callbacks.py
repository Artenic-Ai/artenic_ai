"""Training callback system — hooks into the training loop at specific points."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class TrainingContext:
    """Shared mutable state passed to all callbacks during training."""

    model_id: str
    output_dir: Path
    current_epoch: int = 0
    total_epochs: int = 0
    best_metric: float | None = None
    best_epoch: int | None = None
    device_info: dict[str, Any] = field(default_factory=dict)
    stop_requested: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class TrainingCallback(ABC):
    """Abstract base for training loop callbacks.

    Model authors call ``runner.on_epoch_end(...)`` etc. inside their
    ``train()`` implementation.  The SDK provides concrete callbacks
    for each Training Intelligence feature.
    """

    def on_train_start(self, context: TrainingContext) -> None:  # noqa: B027
        """Called once before the training loop begins."""

    def on_epoch_start(self, epoch: int, context: TrainingContext) -> None:  # noqa: B027
        """Called at the beginning of each epoch."""

    @abstractmethod
    def on_epoch_end(
        self,
        epoch: int,
        metrics: dict[str, float],
        context: TrainingContext,
    ) -> None:
        """Called at the end of each epoch with computed metrics."""

    def on_batch_end(  # noqa: B027
        self,
        batch: int,
        loss: float,
        context: TrainingContext,
    ) -> None:
        """Called at the end of each batch (optional, high-frequency)."""

    def on_train_end(self, context: TrainingContext) -> None:  # noqa: B027
        """Called once after the training loop finishes."""

    def should_stop(self) -> bool:
        """Return True if this callback wants to stop training early."""
        return False


class CallbackRunner:
    """Executes a list of callbacks in registration order."""

    def __init__(self, callbacks: list[TrainingCallback] | None = None) -> None:
        self._callbacks: list[TrainingCallback] = list(callbacks or [])

    @property
    def callbacks(self) -> list[TrainingCallback]:
        return list(self._callbacks)

    def add(self, callback: TrainingCallback) -> None:
        """Register an additional callback."""
        self._callbacks.append(callback)

    def on_train_start(self, context: TrainingContext) -> None:
        for cb in self._callbacks:
            cb.on_train_start(context)

    def on_epoch_start(self, epoch: int, context: TrainingContext) -> None:
        for cb in self._callbacks:
            cb.on_epoch_start(epoch, context)

    def on_epoch_end(
        self,
        epoch: int,
        metrics: dict[str, float],
        context: TrainingContext,
    ) -> None:
        context.current_epoch = epoch
        for cb in self._callbacks:
            cb.on_epoch_end(epoch, metrics, context)

    def on_batch_end(
        self,
        batch: int,
        loss: float,
        context: TrainingContext,
    ) -> None:
        for cb in self._callbacks:
            cb.on_batch_end(batch, loss, context)

    def on_train_end(self, context: TrainingContext) -> None:
        for cb in self._callbacks:
            cb.on_train_end(context)

    def should_stop(self) -> bool:
        """Return True if ANY callback wants to stop."""
        return any(cb.should_stop() for cb in self._callbacks)
