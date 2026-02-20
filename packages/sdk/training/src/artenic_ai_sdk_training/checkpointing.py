"""Smart Checkpointing — periodic saves, best-model tracking, spot preemption."""

from __future__ import annotations

import json
import logging
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from artenic_ai_sdk_training.callbacks import TrainingCallback, TrainingContext

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from artenic_ai_sdk_training.config import CheckpointConfig

logger = logging.getLogger(__name__)


class SmartCheckpointer(TrainingCallback):
    """Periodic checkpoint saving with best-model tracking and preemption handling.

    Features:
    - Save at configurable intervals (``save_every_n_epochs``)
    - Track best checkpoint by monitored metric
    - Rotate old checkpoints (keep ``max_checkpoints``)
    - SIGTERM handler for spot instance preemption
    """

    def __init__(
        self,
        config: CheckpointConfig,
        save_fn: Callable[[Path], Coroutine[Any, Any, None]] | None = None,
        output_dir: Path = Path("/artifacts"),
    ) -> None:
        self._config = config
        self._save_fn = save_fn
        self._output_dir = output_dir
        self._checkpoints_dir = output_dir / "checkpoints"
        self._best_value: float | None = None
        self._best_epoch: int | None = None
        self._saved_checkpoints: list[Path] = []
        self._checkpoints_saved_count: int = 0
        self._preempted: bool = False
        self._previous_sigterm: Any = None
        self._context: TrainingContext | None = None
        self._pending_tasks: set[Any] = set()

    @property
    def best_value(self) -> float | None:
        return self._best_value

    @property
    def best_epoch(self) -> int | None:
        return self._best_epoch

    @property
    def checkpoints_saved(self) -> int:
        return self._checkpoints_saved_count

    @property
    def preempted(self) -> bool:
        return self._preempted

    def on_train_start(self, context: TrainingContext) -> None:
        self._context = context
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)
        if self._config.preemption_handler:
            self._install_sigterm_handler()

    def on_epoch_end(
        self,
        epoch: int,
        metrics: dict[str, float],
        context: TrainingContext,
    ) -> None:
        self._context = context
        value = metrics.get(self._config.metric)

        # Check if this is the best epoch
        is_best = False
        if value is not None and (self._best_value is None or self._is_improvement(value)):
            self._best_value = value
            self._best_epoch = epoch
            is_best = True

        # Save on interval or if best
        should_save = (epoch + 1) % self._config.save_every_n_epochs == 0
        if self._config.save_best_only:
            should_save = is_best
        elif is_best:
            should_save = True

        if should_save:
            ckpt_dir = self._checkpoints_dir / f"epoch_{epoch:04d}"
            self._save_checkpoint(ckpt_dir, epoch, metrics)

        if is_best:
            best_dir = self._checkpoints_dir / "best"
            self._save_checkpoint(best_dir, epoch, metrics)

    def on_train_end(self, context: TrainingContext) -> None:
        self._context = context
        self._restore_sigterm_handler()

    def _save_checkpoint(
        self,
        ckpt_dir: Path,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "epoch": epoch,
            "metrics": metrics,
            "is_best": ckpt_dir.name == "best",
        }
        metadata_path = ckpt_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        if self._save_fn is not None:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(self._save_fn(ckpt_dir))
                self._pending_tasks.add(task)
                task.add_done_callback(self._pending_tasks.discard)
            except RuntimeError:
                asyncio.run(self._save_fn(ckpt_dir))

        if ckpt_dir.name != "best":
            self._saved_checkpoints.append(ckpt_dir)
            self._checkpoints_saved_count += 1
            self._rotate_checkpoints()

        logger.info("Checkpoint saved: %s (epoch %d)", ckpt_dir, epoch)

    def _rotate_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints."""
        while len(self._saved_checkpoints) > self._config.max_checkpoints:
            old = self._saved_checkpoints.pop(0)
            if old.exists():
                import shutil

                shutil.rmtree(old)
                logger.info("Rotated old checkpoint: %s", old)

    def _is_improvement(self, value: float) -> bool:
        # Called only when _best_value is not None (caller short-circuits).
        assert self._best_value is not None
        if self._config.mode == "min":
            return value < self._best_value
        return value > self._best_value

    def _install_sigterm_handler(self) -> None:
        """Install SIGTERM handler for spot instance preemption."""
        if sys.platform == "win32":
            return  # SIGTERM not reliable on Windows

        def _handle_preemption(signum: int, frame: Any) -> None:
            logger.warning("SIGTERM received — spot instance preemption detected")
            self._preempted = True
            if self._context is not None:
                self._context.stop_requested = True

            # Write preemption marker
            marker = self._output_dir / ".preempted"
            ep_num = self._context.current_epoch if self._context else "?"
            marker.write_text(f"preempted at epoch {ep_num}", encoding="utf-8")

            # Emergency checkpoint
            emergency_dir = self._checkpoints_dir / "preempted"
            emergency_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "preempted": True,
                "epoch": self._context.current_epoch if self._context else -1,
            }
            (emergency_dir / "metadata.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

            # Chain to previous handler
            if callable(self._previous_sigterm):
                self._previous_sigterm(signum, frame)

        self._previous_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _handle_preemption)

    def _restore_sigterm_handler(self) -> None:
        """Restore the previous SIGTERM handler."""
        if sys.platform == "win32":
            return
        if self._previous_sigterm is not None:
            signal.signal(signal.SIGTERM, self._previous_sigterm)
