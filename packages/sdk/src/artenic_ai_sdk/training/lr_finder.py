"""Learning Rate Finder — Leslie Smith LR range test."""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from artenic_ai_sdk.training.config import LRFinderConfig

logger = logging.getLogger(__name__)


@dataclass
class LRFinderResult:
    """Result of an LR range test."""

    suggested_lr: float
    min_lr_tested: float
    max_lr_tested: float
    num_steps: int
    loss_history: list[float] = field(default_factory=list)
    lr_history: list[float] = field(default_factory=list)
    divergence_index: int | None = None


class LearningRateFinder:
    """Auto-find optimal learning rate using the Leslie Smith LR range test.

    Runs a short pre-training sweep with exponentially increasing
    learning rate.  Identifies the steepest descent point in the
    loss curve as the optimal LR.

    Reference: Smith, 2017 — "Cyclical Learning Rates for Training Neural Networks"
    """

    def __init__(self, config: LRFinderConfig) -> None:
        self._config = config

    def find(
        self,
        model: Any,
        train_loader: Any,
        criterion: Any,
        optimizer_class: Any | None = None,
    ) -> LRFinderResult:
        """Run the LR range test and return the suggested learning rate.

        The model is deep-copied so original weights are preserved.
        """
        import torch

        if optimizer_class is None:
            optimizer_class = torch.optim.Adam

        # Deep copy to preserve original weights
        model_copy = copy.deepcopy(model)
        device = next(model_copy.parameters()).device

        optimizer = optimizer_class(model_copy.parameters(), lr=self._config.min_lr)

        # Compute LR multiplier per step
        lr_mult = (self._config.max_lr / self._config.min_lr) ** (1.0 / self._config.num_steps)

        losses: list[float] = []
        lrs: list[float] = []
        smoothed_loss = 0.0
        best_loss = float("inf")
        divergence_idx: int | None = None

        data_iter = iter(train_loader)
        model_copy.train()

        for step in range(self._config.num_steps):
            # Get next batch (cycle if needed)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            # Handle (input, target) or dict batches
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0].to(device), batch[1].to(device)
            else:
                inputs, targets = batch.to(device), batch.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model_copy(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track loss
            current_loss = loss.item()
            if step == 0:
                smoothed_loss = current_loss
            else:
                smoothed_loss = (
                    self._config.smooth_factor * current_loss
                    + (1 - self._config.smooth_factor) * smoothed_loss
                )

            current_lr = optimizer.param_groups[0]["lr"]
            losses.append(smoothed_loss)
            lrs.append(current_lr)

            # Check for divergence
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
            if smoothed_loss > best_loss * self._config.divergence_threshold:
                divergence_idx = step
                logger.info(
                    "LR Finder: diverged at step %d (lr=%.2e)",
                    step,
                    current_lr,
                )
                break

            # Update LR
            for param_group in optimizer.param_groups:
                param_group["lr"] *= lr_mult

        # Find suggested LR: steepest descent in smoothed loss
        suggested = self._find_steepest_descent(lrs, losses)

        logger.info(
            "LR Finder: suggested lr=%.2e (tested %d steps)",
            suggested,
            len(losses),
        )

        return LRFinderResult(
            suggested_lr=suggested,
            min_lr_tested=self._config.min_lr,
            max_lr_tested=lrs[-1] if lrs else self._config.max_lr,
            num_steps=len(losses),
            loss_history=losses,
            lr_history=lrs,
            divergence_index=divergence_idx,
        )

    @staticmethod
    def _find_steepest_descent(
        lrs: list[float],
        losses: list[float],
    ) -> float:
        """Find the LR at the steepest negative gradient of the loss curve."""
        if len(lrs) < 3:
            return lrs[0] if lrs else 1e-3

        # Compute gradients in log-space
        log_lrs = [math.log10(lr) for lr in lrs]
        gradients: list[float] = []
        for i in range(1, len(losses) - 1):
            grad = (losses[i + 1] - losses[i - 1]) / (log_lrs[i + 1] - log_lrs[i - 1])
            gradients.append(grad)

        # Find the index of the steepest descent (most negative gradient)
        min_grad_idx = min(range(len(gradients)), key=lambda i: gradients[i])
        # Return the LR at that point (offset by 1 due to gradient indexing)
        return lrs[min_grad_idx + 1]
