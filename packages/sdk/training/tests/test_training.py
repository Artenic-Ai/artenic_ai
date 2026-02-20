"""Tests for artenic_ai_sdk_training — config, callbacks, and all features."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pytest

from artenic_ai_sdk_training.callbacks import (
    CallbackRunner,
    TrainingCallback,
    TrainingContext,
)
from artenic_ai_sdk_training.config import (
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context(tmp_path: Path) -> TrainingContext:
    return TrainingContext(
        model_id="test-model",
        output_dir=tmp_path,
        total_epochs=10,
    )


class _DummyCallback(TrainingCallback):
    """Minimal concrete callback for testing the runner."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def on_train_start(self, context: TrainingContext) -> None:
        self.events.append("train_start")

    def on_epoch_start(self, epoch: int, context: TrainingContext) -> None:
        self.events.append(f"epoch_start:{epoch}")

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], context: TrainingContext) -> None:
        self.events.append(f"epoch_end:{epoch}")

    def on_batch_end(self, batch: int, loss: float, context: TrainingContext) -> None:
        self.events.append(f"batch_end:{batch}")

    def on_train_end(self, context: TrainingContext) -> None:
        self.events.append("train_end")


# ===================================================================
# TrainingConfig
# ===================================================================


class TestTrainingConfig:
    def test_defaults(self) -> None:
        cfg = TrainingConfig(version="1.0")
        assert cfg.epochs == 10
        assert cfg.batch_size == 32
        assert cfg.learning_rate == 0.001
        assert not cfg.data_split.enabled
        assert not cfg.early_stopping.enabled

    def test_active_features_none(self) -> None:
        cfg = TrainingConfig(version="1.0")
        summary = cfg.active_features()
        assert all(v is False for v in summary.values())

    def test_active_features_enabled(self) -> None:
        cfg = TrainingConfig(
            version="1.0",
            early_stopping=EarlyStoppingConfig(enabled=True),
            checkpoint=CheckpointConfig(enabled=True),
        )
        summary = cfg.active_features()
        assert summary["early_stopping"] is True
        assert summary["checkpoint"] is True
        assert summary["data_split"] is False

    def test_inherits_model_config(self) -> None:
        cfg = TrainingConfig(version="2.0.0")
        # Should have ModelConfig fields
        assert cfg.version == "2.0.0"
        # TrainingConfig extends ModelConfig
        from artenic_ai_sdk.schemas import ModelConfig

        assert isinstance(cfg, ModelConfig)

    def test_serialization_roundtrip(self) -> None:
        cfg = TrainingConfig(
            version="1.0",
            epochs=20,
            early_stopping=EarlyStoppingConfig(enabled=True, patience=10),
        )
        data = cfg.model_dump()
        restored = TrainingConfig.model_validate(data)
        assert restored.epochs == 20
        assert restored.early_stopping.patience == 10


# ===================================================================
# Sub-configs
# ===================================================================


class TestSubConfigs:
    def test_data_split_defaults(self) -> None:
        cfg = DataSplitConfig()
        assert cfg.train_ratio == 0.8
        assert cfg.val_ratio == 0.1
        assert cfg.test_ratio == 0.1
        assert cfg.n_folds == 5
        assert cfg.random_seed == 42

    def test_data_split_validation(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            DataSplitConfig(train_ratio=1.5)
        with pytest.raises(Exception):  # noqa: B017
            DataSplitConfig(n_folds=1)

    def test_checkpoint_config(self) -> None:
        cfg = CheckpointConfig(enabled=True, max_checkpoints=5)
        assert cfg.save_every_n_epochs == 1
        assert cfg.metric == "val_loss"
        assert cfg.mode == "min"

    def test_early_stopping_config(self) -> None:
        cfg = EarlyStoppingConfig(enabled=True, patience=3, min_delta=0.01)
        assert cfg.patience == 3
        assert cfg.min_delta == 0.01

    def test_lr_finder_config(self) -> None:
        cfg = LRFinderConfig(num_steps=200, max_lr=1.0)
        assert cfg.num_steps == 200
        assert cfg.min_lr == 1e-7

    def test_mixed_precision_config(self) -> None:
        cfg = MixedPrecisionConfig(mode="bf16")
        assert cfg.mode == "bf16"
        assert cfg.loss_scale == "dynamic"

    def test_gradient_checkpoint_config(self) -> None:
        cfg = GradientCheckpointConfig(mode="always")
        assert cfg.mode == "always"
        assert cfg.memory_threshold_pct == 0.85

    def test_distributed_config(self) -> None:
        cfg = DistributedConfig(strategy="ddp")
        assert cfg.strategy == "ddp"
        assert cfg.sharding_strategy == "full_shard"

    def test_data_versioning_config(self) -> None:
        cfg = DataVersioningConfig(hash_algorithm="xxhash")
        assert cfg.hash_algorithm == "xxhash"


# ===================================================================
# TrainingContext
# ===================================================================


class TestTrainingContext:
    def test_defaults(self, tmp_path: Path) -> None:
        ctx = _make_context(tmp_path)
        assert ctx.current_epoch == 0
        assert ctx.stop_requested is False
        assert ctx.best_metric is None

    def test_mutable(self, tmp_path: Path) -> None:
        ctx = _make_context(tmp_path)
        ctx.current_epoch = 5
        ctx.stop_requested = True
        ctx.best_metric = 0.5
        assert ctx.current_epoch == 5
        assert ctx.stop_requested is True


# ===================================================================
# CallbackRunner
# ===================================================================


class TestCallbackRunner:
    def test_empty_runner(self, tmp_path: Path) -> None:
        runner = CallbackRunner()
        ctx = _make_context(tmp_path)
        # Should not raise
        runner.on_train_start(ctx)
        runner.on_epoch_start(0, ctx)
        runner.on_epoch_end(0, {}, ctx)
        runner.on_train_end(ctx)
        assert runner.should_stop() is False

    def test_event_order(self, tmp_path: Path) -> None:
        cb = _DummyCallback()
        runner = CallbackRunner([cb])
        ctx = _make_context(tmp_path)

        runner.on_train_start(ctx)
        runner.on_epoch_start(0, ctx)
        runner.on_batch_end(0, 1.0, ctx)
        runner.on_epoch_end(0, {"loss": 1.0}, ctx)
        runner.on_train_end(ctx)

        assert cb.events == [
            "train_start",
            "epoch_start:0",
            "batch_end:0",
            "epoch_end:0",
            "train_end",
        ]

    def test_add_callback(self) -> None:
        runner = CallbackRunner()
        assert len(runner.callbacks) == 0
        cb = _DummyCallback()
        runner.add(cb)
        assert len(runner.callbacks) == 1

    def test_updates_current_epoch(self, tmp_path: Path) -> None:
        cb = _DummyCallback()
        runner = CallbackRunner([cb])
        ctx = _make_context(tmp_path)
        runner.on_epoch_end(7, {}, ctx)
        assert ctx.current_epoch == 7

    def test_should_stop_delegation(self, tmp_path: Path) -> None:
        class _StopCallback(TrainingCallback):
            def on_epoch_end(
                self, epoch: int, metrics: dict[str, float], context: TrainingContext
            ) -> None:
                pass

            def should_stop(self) -> bool:
                return True

        runner = CallbackRunner([_DummyCallback(), _StopCallback()])
        assert runner.should_stop() is True

    def test_multiple_callbacks(self, tmp_path: Path) -> None:
        cb1 = _DummyCallback()
        cb2 = _DummyCallback()
        runner = CallbackRunner([cb1, cb2])
        ctx = _make_context(tmp_path)
        runner.on_train_start(ctx)
        assert cb1.events == ["train_start"]
        assert cb2.events == ["train_start"]


# ===================================================================
# EarlyStopping
# ===================================================================


class TestEarlyStopping:
    def _make_es(self, **kwargs: Any) -> Any:
        from artenic_ai_sdk_training.early_stopping import EarlyStopping

        cfg = EarlyStoppingConfig(enabled=True, **kwargs)
        return EarlyStopping(cfg)

    def test_first_epoch_always_improves(self, tmp_path: Path) -> None:
        es = self._make_es(patience=3, metric="val_loss")
        ctx = _make_context(tmp_path)
        es.on_epoch_end(0, {"val_loss": 1.0}, ctx)
        assert es.best_value == 1.0
        assert es.best_epoch == 0
        assert es.should_stop() is False

    def test_improvement_resets_counter(self, tmp_path: Path) -> None:
        es = self._make_es(patience=2, metric="val_loss", min_delta=0.0)
        ctx = _make_context(tmp_path)
        es.on_epoch_end(0, {"val_loss": 1.0}, ctx)
        es.on_epoch_end(1, {"val_loss": 1.1}, ctx)  # no improvement
        assert es.epochs_without_improvement == 1
        es.on_epoch_end(2, {"val_loss": 0.5}, ctx)  # improvement!
        assert es.epochs_without_improvement == 0

    def test_stops_after_patience(self, tmp_path: Path) -> None:
        es = self._make_es(patience=2, metric="val_loss", min_delta=0.0)
        ctx = _make_context(tmp_path)
        es.on_epoch_end(0, {"val_loss": 1.0}, ctx)
        es.on_epoch_end(1, {"val_loss": 1.5}, ctx)
        assert es.should_stop() is False
        es.on_epoch_end(2, {"val_loss": 2.0}, ctx)
        assert es.should_stop() is True
        assert ctx.stop_requested is True

    def test_max_mode(self, tmp_path: Path) -> None:
        es = self._make_es(patience=2, metric="accuracy", mode="max", min_delta=0.0)
        ctx = _make_context(tmp_path)
        es.on_epoch_end(0, {"accuracy": 0.7}, ctx)
        es.on_epoch_end(1, {"accuracy": 0.8}, ctx)  # improvement
        assert es.epochs_without_improvement == 0
        es.on_epoch_end(2, {"accuracy": 0.75}, ctx)  # no improvement
        assert es.epochs_without_improvement == 1

    def test_min_delta(self, tmp_path: Path) -> None:
        es = self._make_es(patience=3, metric="val_loss", min_delta=0.1)
        ctx = _make_context(tmp_path)
        es.on_epoch_end(0, {"val_loss": 1.0}, ctx)
        # Improvement less than min_delta should not count
        es.on_epoch_end(1, {"val_loss": 0.95}, ctx)
        assert es.epochs_without_improvement == 1

    def test_missing_metric_warns(self, tmp_path: Path) -> None:
        es = self._make_es(patience=3, metric="val_loss")
        ctx = _make_context(tmp_path)
        # No val_loss in metrics — should log warning but not crash
        es.on_epoch_end(0, {"train_loss": 1.0}, ctx)
        assert es.best_value is None

    def test_updates_context(self, tmp_path: Path) -> None:
        es = self._make_es(patience=3, metric="val_loss")
        ctx = _make_context(tmp_path)
        es.on_epoch_end(0, {"val_loss": 0.5}, ctx)
        assert ctx.best_metric == 0.5
        assert ctx.best_epoch == 0


# ===================================================================
# SmartCheckpointer
# ===================================================================


class TestSmartCheckpointer:
    def _make_ckpt(self, tmp_path: Path, **kwargs: Any) -> Any:
        from artenic_ai_sdk_training.checkpointing import SmartCheckpointer

        cfg = CheckpointConfig(enabled=True, preemption_handler=False, **kwargs)
        return SmartCheckpointer(config=cfg, output_dir=tmp_path)

    def test_save_on_interval(self, tmp_path: Path) -> None:
        ckpt = self._make_ckpt(tmp_path, save_every_n_epochs=2, save_best_only=False)
        ctx = _make_context(tmp_path)
        ckpt.on_train_start(ctx)

        # Epoch 0 → (0+1)%2 == 1 → no save (but first value = best → best saved)
        ckpt.on_epoch_end(0, {"val_loss": 1.0}, ctx)
        # Epoch 1 → (1+1)%2 == 0 → save
        ckpt.on_epoch_end(1, {"val_loss": 0.9}, ctx)

        # Check files exist
        assert (tmp_path / "checkpoints" / "best" / "metadata.json").exists()
        assert (tmp_path / "checkpoints" / "epoch_0001" / "metadata.json").exists()

    def test_save_best_only(self, tmp_path: Path) -> None:
        ckpt = self._make_ckpt(tmp_path, save_best_only=True)
        ctx = _make_context(tmp_path)
        ckpt.on_train_start(ctx)

        ckpt.on_epoch_end(0, {"val_loss": 1.0}, ctx)
        ckpt.on_epoch_end(1, {"val_loss": 1.5}, ctx)  # worse
        ckpt.on_epoch_end(2, {"val_loss": 0.5}, ctx)  # better

        assert ckpt.best_value == 0.5
        assert ckpt.best_epoch == 2
        # epoch_0000 saved (first best), epoch_0001 NOT saved, epoch_0002 saved
        assert (tmp_path / "checkpoints" / "epoch_0000" / "metadata.json").exists()
        assert not (tmp_path / "checkpoints" / "epoch_0001").exists()
        assert (tmp_path / "checkpoints" / "epoch_0002" / "metadata.json").exists()

    def test_max_mode(self, tmp_path: Path) -> None:
        ckpt = self._make_ckpt(tmp_path, save_best_only=True, mode="max", metric="accuracy")
        ctx = _make_context(tmp_path)
        ckpt.on_train_start(ctx)

        ckpt.on_epoch_end(0, {"accuracy": 0.7}, ctx)
        ckpt.on_epoch_end(1, {"accuracy": 0.9}, ctx)
        assert ckpt.best_value == 0.9

    def test_rotation(self, tmp_path: Path) -> None:
        ckpt = self._make_ckpt(
            tmp_path,
            save_every_n_epochs=1,
            save_best_only=False,
            max_checkpoints=2,
        )
        ctx = _make_context(tmp_path)
        ckpt.on_train_start(ctx)

        # Save 4 checkpoints — should keep only last 2
        for i in range(4):
            ckpt.on_epoch_end(i, {"val_loss": 1.0 - i * 0.1}, ctx)

        assert not (tmp_path / "checkpoints" / "epoch_0000").exists()
        assert not (tmp_path / "checkpoints" / "epoch_0001").exists()
        assert (tmp_path / "checkpoints" / "epoch_0002" / "metadata.json").exists()
        assert (tmp_path / "checkpoints" / "epoch_0003" / "metadata.json").exists()

    def test_metadata_content(self, tmp_path: Path) -> None:
        ckpt = self._make_ckpt(tmp_path, save_best_only=True)
        ctx = _make_context(tmp_path)
        ckpt.on_train_start(ctx)
        ckpt.on_epoch_end(0, {"val_loss": 0.5, "accuracy": 0.9}, ctx)

        meta_path = tmp_path / "checkpoints" / "best" / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["epoch"] == 0
        assert meta["is_best"] is True
        assert meta["metrics"]["val_loss"] == 0.5

    def test_checkpoints_saved_count(self, tmp_path: Path) -> None:
        ckpt = self._make_ckpt(tmp_path, save_every_n_epochs=1, save_best_only=False)
        ctx = _make_context(tmp_path)
        ckpt.on_train_start(ctx)
        ckpt.on_epoch_end(0, {"val_loss": 1.0}, ctx)
        ckpt.on_epoch_end(1, {"val_loss": 0.8}, ctx)
        assert ckpt.checkpoints_saved >= 2

    def test_missing_metric(self, tmp_path: Path) -> None:
        ckpt = self._make_ckpt(tmp_path, save_best_only=True)
        ctx = _make_context(tmp_path)
        ckpt.on_train_start(ctx)
        # No val_loss in metrics — value is None, no best update
        ckpt.on_epoch_end(0, {"train_loss": 1.0}, ctx)
        assert ckpt.best_value is None

    def test_preempted_default(self, tmp_path: Path) -> None:
        ckpt = self._make_ckpt(tmp_path)
        assert ckpt.preempted is False

    def test_on_train_start_with_preemption_handler(self, tmp_path: Path) -> None:
        from artenic_ai_sdk_training.checkpointing import SmartCheckpointer

        cfg = CheckpointConfig(enabled=True, preemption_handler=True)
        ckpt = SmartCheckpointer(config=cfg, output_dir=tmp_path)
        ctx = _make_context(tmp_path)

        with patch.object(ckpt, "_install_sigterm_handler") as mock_install:
            ckpt.on_train_start(ctx)
        mock_install.assert_called_once()

    def test_save_fn_called_via_asyncio_run(self, tmp_path: Path) -> None:
        """Test that save_fn is called via asyncio.run when no running loop."""
        from artenic_ai_sdk_training.checkpointing import SmartCheckpointer

        save_calls: list[Any] = []

        async def mock_save_fn(path: Any) -> None:
            save_calls.append(path)

        cfg = CheckpointConfig(enabled=True, preemption_handler=False, save_best_only=False)
        ckpt = SmartCheckpointer(config=cfg, save_fn=mock_save_fn, output_dir=tmp_path)
        ctx = _make_context(tmp_path)
        ckpt.on_train_start(ctx)
        ckpt.on_epoch_end(0, {"val_loss": 0.5}, ctx)
        # save_fn should have been called (via asyncio.run fallback)
        assert len(save_calls) >= 1

    def test_rotation_removes_old_checkpoints(self, tmp_path: Path) -> None:
        """Test that shutil.rmtree is called during rotation."""
        ckpt = self._make_ckpt(
            tmp_path,
            save_every_n_epochs=1,
            save_best_only=False,
            max_checkpoints=1,
        )
        ctx = _make_context(tmp_path)
        ckpt.on_train_start(ctx)

        # Save 3 checkpoints, max is 1 → 2 should be removed
        for i in range(3):
            ckpt.on_epoch_end(i, {"val_loss": 1.0 - i * 0.1}, ctx)

        # Only 1 checkpoint should remain (plus best)
        assert ckpt.checkpoints_saved >= 3  # total saved count
        # The rotation should have deleted the old ones
        assert not (tmp_path / "checkpoints" / "epoch_0000").exists()
        assert not (tmp_path / "checkpoints" / "epoch_0001").exists()
        assert (tmp_path / "checkpoints" / "epoch_0002" / "metadata.json").exists()

    def test_on_train_end(self, tmp_path: Path) -> None:
        ckpt = self._make_ckpt(tmp_path)
        ctx = _make_context(tmp_path)
        ckpt.on_train_start(ctx)
        # Should not raise
        ckpt.on_train_end(ctx)

    def test_sigterm_handler_on_non_windows(self, tmp_path: Path) -> None:
        """Test SIGTERM handler installation on non-Windows platform."""
        import signal

        from artenic_ai_sdk_training.checkpointing import SmartCheckpointer

        cfg = CheckpointConfig(enabled=True, preemption_handler=True)
        ckpt = SmartCheckpointer(config=cfg, output_dir=tmp_path)

        with patch("artenic_ai_sdk_training.checkpointing.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("artenic_ai_sdk_training.checkpointing.signal") as mock_signal:
                mock_signal.SIGTERM = signal.SIGTERM
                mock_signal.getsignal.return_value = signal.SIG_DFL
                ckpt._install_sigterm_handler()
                mock_signal.signal.assert_called_once()

    def test_sigterm_handler_sets_preempted(self, tmp_path: Path) -> None:
        """Test the preemption handler behavior when SIGTERM is received."""
        import signal

        from artenic_ai_sdk_training.checkpointing import SmartCheckpointer

        cfg = CheckpointConfig(enabled=True, preemption_handler=True)
        ckpt = SmartCheckpointer(config=cfg, output_dir=tmp_path)
        ctx = _make_context(tmp_path)
        ckpt._context = ctx

        # Install handler on non-windows
        with patch("artenic_ai_sdk_training.checkpointing.sys") as mock_sys:
            mock_sys.platform = "linux"
            # Capture the signal handler
            handler_ref: list[Any] = []

            def capture_handler(sig: Any, handler: Any) -> None:
                handler_ref.append(handler)

            with patch("artenic_ai_sdk_training.checkpointing.signal") as mock_signal:
                mock_signal.SIGTERM = signal.SIGTERM
                mock_signal.getsignal.return_value = None
                mock_signal.signal.side_effect = capture_handler
                ckpt._install_sigterm_handler()

        assert len(handler_ref) == 1
        handler = handler_ref[0]

        # Call the handler to simulate SIGTERM
        handler(signal.SIGTERM, None)

        assert ckpt._preempted is True
        assert ctx.stop_requested is True
        assert (tmp_path / ".preempted").exists()
        assert (tmp_path / "checkpoints" / "preempted" / "metadata.json").exists()

    def test_restore_sigterm_handler_non_windows(self, tmp_path: Path) -> None:
        """Test restoring SIGTERM handler on non-Windows."""
        import signal

        from artenic_ai_sdk_training.checkpointing import SmartCheckpointer

        cfg = CheckpointConfig(enabled=True, preemption_handler=True)
        ckpt = SmartCheckpointer(config=cfg, output_dir=tmp_path)
        ckpt._previous_sigterm = signal.SIG_DFL

        with patch("artenic_ai_sdk_training.checkpointing.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("artenic_ai_sdk_training.checkpointing.signal") as mock_signal:
                mock_signal.SIGTERM = signal.SIGTERM
                ckpt._restore_sigterm_handler()
                mock_signal.signal.assert_called_once_with(signal.SIGTERM, signal.SIG_DFL)

    def test_restore_sigterm_handler_windows_noop(self, tmp_path: Path) -> None:
        """Test restoring SIGTERM handler is a no-op on Windows."""
        from artenic_ai_sdk_training.checkpointing import SmartCheckpointer

        cfg = CheckpointConfig(enabled=True, preemption_handler=True)
        ckpt = SmartCheckpointer(config=cfg, output_dir=tmp_path)

        with patch("artenic_ai_sdk_training.checkpointing.sys") as mock_sys:
            mock_sys.platform = "win32"
            with patch("artenic_ai_sdk_training.checkpointing.signal") as mock_signal:
                ckpt._restore_sigterm_handler()
                mock_signal.signal.assert_not_called()

    async def test_save_fn_called_via_running_loop(self, tmp_path: Path) -> None:
        """Test save_fn via create_task when a running event loop exists."""
        import asyncio

        from artenic_ai_sdk_training.checkpointing import SmartCheckpointer

        save_calls: list[Any] = []

        async def mock_save_fn(path: Any) -> None:
            save_calls.append(path)

        cfg = CheckpointConfig(enabled=True, preemption_handler=False, save_best_only=False)
        ckpt = SmartCheckpointer(config=cfg, save_fn=mock_save_fn, output_dir=tmp_path)
        ctx = _make_context(tmp_path)
        ckpt.on_train_start(ctx)

        # Call on_epoch_end from within a running loop
        ckpt.on_epoch_end(0, {"val_loss": 0.5}, ctx)
        # Let the event loop process the pending task
        await asyncio.sleep(0.01)
        assert len(save_calls) >= 1

    def test_sigterm_handler_chains_to_previous(self, tmp_path: Path) -> None:
        """Test that SIGTERM handler chains to a callable previous handler."""
        import signal

        from artenic_ai_sdk_training.checkpointing import SmartCheckpointer

        cfg = CheckpointConfig(enabled=True, preemption_handler=True)
        ckpt = SmartCheckpointer(config=cfg, output_dir=tmp_path)
        ctx = _make_context(tmp_path)
        ckpt._context = ctx

        # Install handler on non-windows with a callable previous handler
        prev_handler_calls: list[Any] = []

        def prev_handler(signum: int, frame: Any) -> None:
            prev_handler_calls.append(signum)

        with patch("artenic_ai_sdk_training.checkpointing.sys") as mock_sys:
            mock_sys.platform = "linux"
            handler_ref: list[Any] = []

            def capture_handler(sig: Any, handler: Any) -> None:
                handler_ref.append(handler)

            with patch("artenic_ai_sdk_training.checkpointing.signal") as mock_signal:
                mock_signal.SIGTERM = signal.SIGTERM
                mock_signal.getsignal.return_value = prev_handler
                mock_signal.signal.side_effect = capture_handler
                ckpt._install_sigterm_handler()

        handler = handler_ref[0]
        handler(signal.SIGTERM, None)

        # Previous handler should have been called
        assert len(prev_handler_calls) == 1
        assert prev_handler_calls[0] == signal.SIGTERM

    def test_install_sigterm_handler_windows_noop(self, tmp_path: Path) -> None:
        """Test that _install_sigterm_handler is a no-op on Windows."""
        from artenic_ai_sdk_training.checkpointing import SmartCheckpointer

        cfg = CheckpointConfig(enabled=True, preemption_handler=True)
        ckpt = SmartCheckpointer(config=cfg, output_dir=tmp_path)

        # On Windows (our actual platform), it should return early
        with patch("artenic_ai_sdk_training.checkpointing.sys") as mock_sys:
            mock_sys.platform = "win32"
            with patch("artenic_ai_sdk_training.checkpointing.signal") as mock_signal:
                ckpt._install_sigterm_handler()
                # signal.signal should NOT be called
                mock_signal.signal.assert_not_called()


# ===================================================================
# SmartDataSplitter
# ===================================================================


class TestSmartDataSplitter:
    def _make_splitter(self, **kwargs: Any) -> Any:
        from artenic_ai_sdk_training.data_splitting import SmartDataSplitter

        cfg = DataSplitConfig(enabled=True, **kwargs)
        return SmartDataSplitter(cfg)

    def test_random_holdout_list(self) -> None:
        splitter = self._make_splitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        data = list(range(100))
        result = splitter.split(data)
        assert len(result.train) == 60
        assert len(result.val) == 20
        assert result.test is not None
        assert len(result.test) == 20

    def test_random_holdout_numpy(self) -> None:
        splitter = self._make_splitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        data = np.arange(200)
        result = splitter.split(data)
        assert len(result.train) == 140
        assert result.split_info["strategy"] == "holdout"

    def test_time_series_split(self) -> None:
        splitter = self._make_splitter(
            strategy="time_series",
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )
        data = list(range(100))
        result = splitter.split(data)
        # Time-series should preserve order — first 60 items in train
        assert result.train == list(range(60))
        assert result.val == list(range(60, 80))
        assert result.test == list(range(80, 100))
        assert result.split_info["strategy"] == "time_series"

    def test_no_val_split(self) -> None:
        splitter = self._make_splitter(train_ratio=0.8, val_ratio=0.0, test_ratio=0.2)
        data = list(range(100))
        result = splitter.split(data)
        assert len(result.train) == 80
        assert result.val is None

    def test_kfold(self) -> None:
        splitter = self._make_splitter(strategy="kfold", n_folds=5)
        data = list(range(100))
        folds = list(splitter.kfold_splits(data))
        assert len(folds) == 5
        for i, fold in enumerate(folds):
            assert fold.fold_index == i
            assert len(fold.train) + len(fold.val) == 100

    def test_time_series_kfold(self) -> None:
        splitter = self._make_splitter(strategy="time_series", n_folds=4)
        data = list(range(100))
        folds = list(splitter.kfold_splits(data))
        assert len(folds) == 4
        # Each fold's train should be ordered and grow
        for i in range(1, len(folds)):
            assert len(folds[i].train) > len(folds[i - 1].train)

    def test_stratified_kfold_without_sklearn(self) -> None:
        """Stratified k-fold falls back to regular k-fold without sklearn."""
        splitter = self._make_splitter(strategy="stratified_kfold", n_folds=3)
        data = list(range(30))
        # Even without real target column, kfold_splits should fallback
        with patch.dict("sys.modules", {"sklearn": None, "sklearn.model_selection": None}):
            folds = list(splitter.kfold_splits(data, target_column="target"))
        assert len(folds) == 3

    def test_split_info(self) -> None:
        splitter = self._make_splitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        result = splitter.split(list(range(100)))
        assert "train_size" in result.split_info
        assert "val_size" in result.split_info

    def test_deterministic_with_seed(self) -> None:
        splitter = self._make_splitter(random_seed=42)
        data = list(range(100))
        r1 = splitter.split(data)
        r2 = splitter.split(data)
        assert r1.train == r2.train

    def test_stratified_holdout_with_sklearn(self) -> None:
        """Stratified holdout using mocked sklearn train_test_split."""
        splitter = self._make_splitter(
            strategy="holdout",
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )
        # Mock sklearn.model_selection.train_test_split so it works without sklearn
        mock_tts = MagicMock(
            side_effect=[
                (np.arange(60), np.arange(60, 100)),  # train vs remaining
                (np.arange(60, 80), np.arange(80, 100)),  # val vs test
            ]
        )
        mock_model_sel = MagicMock()
        mock_model_sel.train_test_split = mock_tts

        mock_df = MagicMock()
        mock_df.__len__ = MagicMock(return_value=100)
        mock_df.__getitem__ = MagicMock(return_value=np.array([0] * 50 + [1] * 50))
        mock_df.iloc.__getitem__ = MagicMock(return_value="sliced")

        with patch.dict(
            "sys.modules",
            {"sklearn": MagicMock(), "sklearn.model_selection": mock_model_sel},
        ):
            result = splitter.split(mock_df, target_column="target")
        assert result.split_info["strategy"] == "stratified_holdout"
        assert mock_tts.call_count == 2

    def test_stratified_holdout_no_sklearn(self) -> None:
        """Stratified holdout falls back to random when sklearn is missing."""
        splitter = self._make_splitter(
            strategy="holdout",
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )
        data = list(range(100))
        with patch.dict("sys.modules", {"sklearn": None, "sklearn.model_selection": None}):
            result = splitter.split(data, target_column="target")
        # Falls back to random holdout
        assert result.split_info["strategy"] == "holdout"

    def test_stratified_kfold_with_sklearn(self) -> None:
        """Stratified k-fold using mocked sklearn StratifiedKFold."""
        splitter = self._make_splitter(strategy="stratified_kfold", n_folds=3)

        # Mock StratifiedKFold.split() to return 3 folds
        mock_skf = MagicMock()
        mock_skf.split.return_value = [
            (np.arange(20, 30), np.arange(0, 10)),
            (np.concatenate([np.arange(0, 10), np.arange(20, 30)]), np.arange(10, 20)),
            (np.arange(0, 20), np.arange(20, 30)),
        ]
        mock_model_sel = MagicMock()
        mock_model_sel.StratifiedKFold = MagicMock(return_value=mock_skf)

        mock_df = MagicMock()
        mock_df.__len__ = MagicMock(return_value=30)
        mock_df.__getitem__ = MagicMock(return_value=np.array([0] * 15 + [1] * 15))
        mock_df.iloc.__getitem__ = MagicMock(return_value="sliced")

        with patch.dict(
            "sys.modules",
            {"sklearn": MagicMock(), "sklearn.model_selection": mock_model_sel},
        ):
            folds = list(splitter.kfold_splits(mock_df, target_column="target"))
        assert len(folds) == 3
        for i, fold in enumerate(folds):
            assert fold.fold_index == i
            assert fold.split_info["strategy"] == "stratified_kfold"

    def test_index_with_pandas_iloc(self) -> None:
        """_index uses iloc for pandas-like objects."""
        from artenic_ai_sdk_training.data_splitting import SmartDataSplitter

        mock_df = MagicMock()
        mock_df.iloc.__getitem__ = MagicMock(return_value="sliced_df")
        indices = np.array([0, 1, 2])
        result = SmartDataSplitter._index(mock_df, indices)
        assert result == "sliced_df"

    def test_get_targets_from_pandas(self) -> None:
        """_get_targets extracts column from pandas-like DataFrame."""
        from artenic_ai_sdk_training.data_splitting import SmartDataSplitter

        mock_df = MagicMock()
        mock_df.__getitem__ = MagicMock(return_value=[0, 1, 0, 1])
        result = SmartDataSplitter._get_targets(mock_df, "target")
        assert isinstance(result, np.ndarray)

    def test_get_targets_from_array(self) -> None:
        """_get_targets returns np.asarray for non-pandas data."""
        from artenic_ai_sdk_training.data_splitting import SmartDataSplitter

        data = [0, 1, 0, 1]
        # list has __getitem__ but no .iloc
        result = SmartDataSplitter._get_targets(data, "target")
        assert isinstance(result, np.ndarray)

    def test_stratified_holdout_no_test(self) -> None:
        """Stratified holdout with test_ratio=0 — val_idx=remaining, test_idx=empty."""
        splitter = self._make_splitter(
            strategy="holdout",
            train_ratio=0.8,
            val_ratio=0.2,
            test_ratio=0.0,
        )
        # Mock train_test_split: single call (no second split when test_ratio=0)
        mock_tts = MagicMock(return_value=(np.arange(80), np.arange(80, 100)))
        mock_model_sel = MagicMock()
        mock_model_sel.train_test_split = mock_tts

        mock_df = MagicMock()
        mock_df.__len__ = MagicMock(return_value=100)
        mock_df.__getitem__ = MagicMock(return_value=np.array([0] * 50 + [1] * 50))
        mock_df.iloc.__getitem__ = MagicMock(return_value="sliced")

        with patch.dict(
            "sys.modules",
            {"sklearn": MagicMock(), "sklearn.model_selection": mock_model_sel},
        ):
            result = splitter.split(mock_df, target_column="target")
        assert result.split_info["strategy"] == "stratified_holdout"
        assert mock_tts.call_count == 1  # only one split, no second for test


# ===================================================================
# DatasetVersioner
# ===================================================================


class TestDatasetVersioner:
    def test_hash_file(self, tmp_path: Path) -> None:
        from artenic_ai_sdk_training.data_versioning import DatasetVersioner

        f = tmp_path / "data.csv"
        f.write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")

        versioner = DatasetVersioner()
        version = versioner.hash_file(f)
        assert len(version.hash) == 64  # sha256 hex
        assert version.algorithm == "sha256"
        assert version.size_bytes > 0
        assert version.created_at != ""

    def test_hash_file_deterministic(self, tmp_path: Path) -> None:
        from artenic_ai_sdk_training.data_versioning import DatasetVersioner

        f = tmp_path / "data.csv"
        f.write_text("hello world\n", encoding="utf-8")

        versioner = DatasetVersioner()
        v1 = versioner.hash_file(f)
        v2 = versioner.hash_file(f)
        assert v1.hash == v2.hash

    def test_hash_directory(self, tmp_path: Path) -> None:
        from artenic_ai_sdk_training.data_versioning import DatasetVersioner

        (tmp_path / "a.txt").write_text("aaa", encoding="utf-8")
        (tmp_path / "b.txt").write_text("bbb", encoding="utf-8")

        versioner = DatasetVersioner()
        version = versioner.hash_directory(tmp_path)
        assert version.num_records == 2
        assert version.size_bytes > 0

    def test_verify_match(self, tmp_path: Path) -> None:
        from artenic_ai_sdk_training.data_versioning import DatasetVersioner

        f = tmp_path / "data.bin"
        f.write_bytes(b"hello world")

        versioner = DatasetVersioner()
        version = versioner.hash_file(f)
        assert versioner.verify(f, version) is True

    def test_verify_mismatch(self, tmp_path: Path) -> None:
        from artenic_ai_sdk_training.data_versioning import DatasetVersioner

        f = tmp_path / "data.bin"
        f.write_bytes(b"hello world")

        versioner = DatasetVersioner()
        version = versioner.hash_file(f)

        # Modify file
        f.write_bytes(b"changed!")
        assert versioner.verify(f, version) is False

    def test_to_dict(self, tmp_path: Path) -> None:
        from artenic_ai_sdk_training.data_versioning import DatasetVersioner

        f = tmp_path / "data.txt"
        f.write_text("test", encoding="utf-8")

        versioner = DatasetVersioner()
        version = versioner.hash_file(f)
        d = version.to_dict()
        assert "hash" in d
        assert "algorithm" in d
        assert "size_bytes" in d

    def test_xxhash_fallback(self, tmp_path: Path) -> None:
        from artenic_ai_sdk_training.data_versioning import DatasetVersioner

        f = tmp_path / "data.txt"
        f.write_text("test", encoding="utf-8")

        cfg = DataVersioningConfig(hash_algorithm="xxhash")
        versioner = DatasetVersioner(cfg)
        # Should fallback to sha256 if xxhash not installed
        version = versioner.hash_file(f)
        assert len(version.hash) > 0

    def test_xxhash_available(self, tmp_path: Path) -> None:
        """Test xxhash path when module is available."""
        from artenic_ai_sdk_training.data_versioning import DatasetVersioner

        f = tmp_path / "data.txt"
        f.write_text("test", encoding="utf-8")

        mock_xxhash = MagicMock()
        mock_hasher = MagicMock()
        mock_hasher.hexdigest.return_value = "abc123"
        mock_xxhash.xxh128.return_value = mock_hasher

        cfg = DataVersioningConfig(hash_algorithm="xxhash")
        versioner = DatasetVersioner(cfg)

        with patch.dict("sys.modules", {"xxhash": mock_xxhash}):
            version = versioner.hash_file(f)
        assert version.hash == "abc123"
        mock_xxhash.xxh128.assert_called_once()

    def test_hash_dataframe(self) -> None:
        """Test hash_dataframe with a mock pandas DataFrame."""
        from artenic_ai_sdk_training.data_versioning import DatasetVersioner

        mock_df = MagicMock()
        mock_df.__len__ = MagicMock(return_value=50)

        # Mock to_parquet writing some bytes
        def write_parquet(buf: Any, index: bool = False) -> None:
            buf.write(b"parquet_data_here")

        mock_df.to_parquet = MagicMock(side_effect=write_parquet)

        versioner = DatasetVersioner()
        version = versioner.hash_dataframe(mock_df)
        assert len(version.hash) == 64  # sha256
        assert version.num_records == 50
        assert version.size_bytes > 0

    def test_hash_dataframe_with_sampling(self) -> None:
        """Test hash_dataframe with sample_size smaller than dataset."""
        from artenic_ai_sdk_training.data_versioning import DatasetVersioner

        mock_df = MagicMock()
        mock_df.__len__ = MagicMock(return_value=1000)

        def write_parquet(buf: Any, index: bool = False) -> None:
            buf.write(b"parquet_data")

        mock_df.to_parquet = MagicMock(side_effect=write_parquet)

        mock_sample = MagicMock()
        mock_sample.to_parquet = MagicMock(side_effect=write_parquet)
        mock_df.head = MagicMock(return_value=mock_sample)

        cfg = DataVersioningConfig(enabled=True, sample_size=100)
        versioner = DatasetVersioner(cfg)
        version = versioner.hash_dataframe(mock_df)
        assert version.num_records == 1000
        mock_df.head.assert_called_once_with(100)


# ===================================================================
# MixedPrecisionManager
# ===================================================================


class TestMixedPrecisionManager:
    def test_detect_no_torch(self) -> None:
        from artenic_ai_sdk_training.mixed_precision import MixedPrecisionManager

        cfg = MixedPrecisionConfig(mode="auto")
        mgr = MixedPrecisionManager(cfg)

        with patch.dict("sys.modules", {"torch": None}):
            # Force re-detection
            mgr._mode = None
            mode = mgr.detect()
        assert mode.dtype == "float32"
        assert mode.device == "cpu"

    def test_detect_no_cuda(self) -> None:
        from artenic_ai_sdk_training.mixed_precision import MixedPrecisionManager

        cfg = MixedPrecisionConfig(mode="auto")
        mgr = MixedPrecisionManager(cfg)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            mgr._mode = None
            mode = mgr.detect()
        assert mode.dtype == "float32"
        assert mode.device == "cpu"

    def test_detect_cuda_off_mode(self) -> None:
        from artenic_ai_sdk_training.mixed_precision import MixedPrecisionManager

        cfg = MixedPrecisionConfig(mode="off")
        mgr = MixedPrecisionManager(cfg)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "Tesla V100"
        mock_torch.cuda.get_device_capability.return_value = (7, 0)
        with patch.dict("sys.modules", {"torch": mock_torch}):
            mgr._mode = None
            mode = mgr.detect()
        assert mode.dtype == "float32"
        assert mode.device == "cuda"

    def test_detect_bf16_forced(self) -> None:
        from artenic_ai_sdk_training.mixed_precision import MixedPrecisionManager

        cfg = MixedPrecisionConfig(mode="bf16")
        mgr = MixedPrecisionManager(cfg)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "A100"
        mock_torch.cuda.get_device_capability.return_value = (8, 0)
        with patch.dict("sys.modules", {"torch": mock_torch}):
            mgr._mode = None
            mode = mgr.detect()
        assert mode.dtype == "bfloat16"
        assert mode.scaler_needed is False

    def test_detect_fp16_forced(self) -> None:
        from artenic_ai_sdk_training.mixed_precision import MixedPrecisionManager

        cfg = MixedPrecisionConfig(mode="fp16")
        mgr = MixedPrecisionManager(cfg)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "V100"
        mock_torch.cuda.get_device_capability.return_value = (7, 0)
        with patch.dict("sys.modules", {"torch": mock_torch}):
            mgr._mode = None
            mode = mgr.detect()
        assert mode.dtype == "float16"
        assert mode.scaler_needed is True

    def test_caches_result(self) -> None:
        from artenic_ai_sdk_training.mixed_precision import MixedPrecisionManager

        cfg = MixedPrecisionConfig(mode="auto")
        mgr = MixedPrecisionManager(cfg)

        with patch.dict("sys.modules", {"torch": None}):
            mgr._mode = None
            m1 = mgr.detect()
            m2 = mgr.detect()
        assert m1 is m2

    def test_auto_detect_bf16_ampere(self) -> None:
        """Auto mode on Ampere+ GPU with BF16 support."""
        from artenic_ai_sdk_training.mixed_precision import MixedPrecisionManager

        cfg = MixedPrecisionConfig(mode="auto")
        mgr = MixedPrecisionManager(cfg)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA A100"
        mock_torch.cuda.get_device_capability.return_value = (8, 0)
        mock_torch.cuda.is_bf16_supported.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            mgr._mode = None
            mode = mgr.detect()
        assert mode.dtype == "bfloat16"
        assert mode.scaler_needed is False
        assert mode.compute_capability == (8, 0)

    def test_auto_detect_fp16_volta(self) -> None:
        """Auto mode on Volta GPU (CC 7.x) → FP16."""
        from artenic_ai_sdk_training.mixed_precision import MixedPrecisionManager

        cfg = MixedPrecisionConfig(mode="auto")
        mgr = MixedPrecisionManager(cfg)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "Tesla V100"
        mock_torch.cuda.get_device_capability.return_value = (7, 0)
        mock_torch.cuda.is_bf16_supported.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            mgr._mode = None
            mode = mgr.detect()
        assert mode.dtype == "float16"
        assert mode.scaler_needed is True

    def test_auto_detect_fp32_old_gpu(self) -> None:
        """Auto mode on old GPU (CC < 7.0) → FP32 fallback."""
        from artenic_ai_sdk_training.mixed_precision import MixedPrecisionManager

        cfg = MixedPrecisionConfig(mode="auto")
        mgr = MixedPrecisionManager(cfg)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "GTX 1080"
        mock_torch.cuda.get_device_capability.return_value = (6, 1)
        mock_torch.cuda.is_bf16_supported.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            mgr._mode = None
            mode = mgr.detect()
        assert mode.dtype == "float32"
        assert mode.scaler_needed is False

    def test_get_autocast_context(self) -> None:
        """get_autocast_context returns torch.autocast."""
        from artenic_ai_sdk_training.mixed_precision import MixedPrecisionManager

        cfg = MixedPrecisionConfig(mode="fp16")
        mgr = MixedPrecisionManager(cfg)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "V100"
        mock_torch.cuda.get_device_capability.return_value = (7, 0)
        mock_torch.float32 = "fp32"
        mock_torch.float16 = "fp16"
        mock_torch.bfloat16 = "bf16"

        with patch.dict("sys.modules", {"torch": mock_torch}):
            mgr._mode = None
            mgr.get_autocast_context()
        mock_torch.autocast.assert_called_once()

    def test_get_grad_scaler_dynamic(self) -> None:
        """get_grad_scaler with dynamic loss scale."""
        from artenic_ai_sdk_training.mixed_precision import MixedPrecisionManager

        cfg = MixedPrecisionConfig(mode="fp16", loss_scale="dynamic")
        mgr = MixedPrecisionManager(cfg)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "V100"
        mock_torch.cuda.get_device_capability.return_value = (7, 0)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            mgr._mode = None
            scaler = mgr.get_grad_scaler()
        mock_torch.amp.GradScaler.assert_called_once_with("cuda")
        assert scaler is not None

    def test_get_grad_scaler_static(self) -> None:
        """get_grad_scaler with static loss scale."""
        from artenic_ai_sdk_training.mixed_precision import MixedPrecisionManager

        cfg = MixedPrecisionConfig(mode="fp16", loss_scale="static", static_loss_scale=1024.0)
        mgr = MixedPrecisionManager(cfg)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "V100"
        mock_torch.cuda.get_device_capability.return_value = (7, 0)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            mgr._mode = None
            mgr.get_grad_scaler()
        mock_torch.amp.GradScaler.assert_called_once_with("cuda", init_scale=1024.0, enabled=True)

    def test_get_grad_scaler_not_needed(self) -> None:
        """get_grad_scaler returns None for BF16/FP32."""
        from artenic_ai_sdk_training.mixed_precision import MixedPrecisionManager

        cfg = MixedPrecisionConfig(mode="bf16")
        mgr = MixedPrecisionManager(cfg)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "A100"
        mock_torch.cuda.get_device_capability.return_value = (8, 0)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            mgr._mode = None
            scaler = mgr.get_grad_scaler()
        assert scaler is None

    def test_get_dtype(self) -> None:
        """get_dtype returns torch dtype."""
        from artenic_ai_sdk_training.mixed_precision import MixedPrecisionManager

        cfg = MixedPrecisionConfig(mode="fp16")
        mgr = MixedPrecisionManager(cfg)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "V100"
        mock_torch.cuda.get_device_capability.return_value = (7, 0)
        mock_torch.float32 = "fp32"
        mock_torch.float16 = "fp16"
        mock_torch.bfloat16 = "bf16"

        with patch.dict("sys.modules", {"torch": mock_torch}):
            mgr._mode = None
            dtype = mgr.get_dtype()
        assert dtype == "fp16"


# ===================================================================
# GradientCheckpointManager
# ===================================================================


class TestGradientCheckpointManager:
    def test_off_mode(self) -> None:
        from artenic_ai_sdk_training.gradient_checkpoint import (
            GradientCheckpointManager,
        )

        cfg = GradientCheckpointConfig(mode="off")
        mgr = GradientCheckpointManager(cfg)
        assert mgr.should_enable() is False

    def test_always_mode(self) -> None:
        from artenic_ai_sdk_training.gradient_checkpoint import (
            GradientCheckpointManager,
        )

        cfg = GradientCheckpointConfig(mode="always")
        mgr = GradientCheckpointManager(cfg)
        assert mgr.should_enable() is True

    def test_auto_no_torch(self) -> None:
        from artenic_ai_sdk_training.gradient_checkpoint import (
            GradientCheckpointManager,
        )

        cfg = GradientCheckpointConfig(mode="auto")
        mgr = GradientCheckpointManager(cfg)
        with patch.dict("sys.modules", {"torch": None}):
            assert mgr.should_enable() is False

    def test_apply_off_mode(self) -> None:
        from artenic_ai_sdk_training.gradient_checkpoint import (
            GradientCheckpointManager,
        )

        cfg = GradientCheckpointConfig(mode="off")
        mgr = GradientCheckpointManager(cfg)
        model = MagicMock()
        result = mgr.apply(model)
        assert result is model
        assert mgr.applied is False

    def test_apply_huggingface_model(self) -> None:
        from artenic_ai_sdk_training.gradient_checkpoint import (
            GradientCheckpointManager,
        )

        cfg = GradientCheckpointConfig(mode="always")
        mgr = GradientCheckpointManager(cfg)

        model = MagicMock()
        model.gradient_checkpointing_enable = MagicMock()

        result = mgr.apply(model)
        model.gradient_checkpointing_enable.assert_called_once()
        assert mgr.applied is True
        assert result is model

    def test_memory_stats_no_torch(self) -> None:
        from artenic_ai_sdk_training.gradient_checkpoint import (
            GradientCheckpointManager,
        )

        cfg = GradientCheckpointConfig(mode="auto")
        mgr = GradientCheckpointManager(cfg)
        with patch.dict("sys.modules", {"torch": None}):
            stats = mgr.get_memory_stats()
        assert stats["available"] is False

    def test_auto_mode_cuda_above_threshold(self) -> None:
        """Auto mode with GPU usage above threshold → should enable."""
        from artenic_ai_sdk_training.gradient_checkpoint import (
            GradientCheckpointManager,
        )

        cfg = GradientCheckpointConfig(mode="auto", memory_threshold_pct=0.5)
        mgr = GradientCheckpointManager(cfg)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_props = MagicMock()
        mock_props.total_mem = 8_000_000_000  # 8GB
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.memory_allocated.return_value = 6_000_000_000  # 6GB = 75%

        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert mgr.should_enable() is True

    def test_auto_mode_cuda_below_threshold(self) -> None:
        """Auto mode with GPU usage below threshold → should not enable."""
        from artenic_ai_sdk_training.gradient_checkpoint import (
            GradientCheckpointManager,
        )

        cfg = GradientCheckpointConfig(mode="auto", memory_threshold_pct=0.8)
        mgr = GradientCheckpointManager(cfg)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_props = MagicMock()
        mock_props.total_mem = 8_000_000_000
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.memory_allocated.return_value = 2_000_000_000  # 25%

        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert mgr.should_enable() is False

    def test_auto_mode_no_cuda(self) -> None:
        """Auto mode without CUDA → should not enable."""
        from artenic_ai_sdk_training.gradient_checkpoint import (
            GradientCheckpointManager,
        )

        cfg = GradientCheckpointConfig(mode="auto")
        mgr = GradientCheckpointManager(cfg)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert mgr.should_enable() is False

    def test_apply_auto_not_needed(self) -> None:
        """Apply when should_enable returns False (auto mode, no CUDA)."""
        from artenic_ai_sdk_training.gradient_checkpoint import (
            GradientCheckpointManager,
        )

        cfg = GradientCheckpointConfig(mode="auto")
        mgr = GradientCheckpointManager(cfg)
        model = MagicMock(spec=[])  # no gradient_checkpointing_enable

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = mgr.apply(model)
        assert result is model
        assert mgr.applied is False

    def test_apply_generic_pytorch_model(self) -> None:
        """Apply gradient checkpointing to a generic nn.Module."""
        from artenic_ai_sdk_training.gradient_checkpoint import (
            GradientCheckpointManager,
        )

        cfg = GradientCheckpointConfig(mode="always")
        mgr = GradientCheckpointManager(cfg)

        # Model without HF API — triggers generic PyTorch path
        mock_param = MagicMock()
        mock_child = MagicMock()
        mock_child.parameters.return_value = iter([mock_param])
        mock_child.forward = MagicMock()

        model = MagicMock(spec=["named_children", "forward"])
        model.named_children.return_value = [("layer1", mock_child)]

        mock_torch = MagicMock()
        mock_ckpt_utils = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "torch.utils": mock_torch.utils,
                "torch.utils.checkpoint": mock_ckpt_utils,
            },
        ):
            result = mgr.apply(model)

        assert result is model
        assert mgr.applied is True
        # forward was replaced with ckpt_forward wrapper
        assert callable(mock_child.forward)
        # Invoke the wrapped forward to cover the ckpt_forward inner function (line 87)
        mock_child.forward("dummy_input")

    def test_apply_generic_pytorch_no_torch(self) -> None:
        """Apply generic path when torch is not importable."""
        from artenic_ai_sdk_training.gradient_checkpoint import (
            GradientCheckpointManager,
        )

        cfg = GradientCheckpointConfig(mode="always")
        mgr = GradientCheckpointManager(cfg)

        model = MagicMock(spec=["named_children"])  # no HF API

        with patch.dict(
            "sys.modules",
            {"torch": MagicMock(), "torch.utils": None, "torch.utils.checkpoint": None},
        ):
            result = mgr.apply(model)
        assert result is model

    def test_get_memory_stats_with_cuda(self) -> None:
        """Memory stats when CUDA is available."""
        from artenic_ai_sdk_training.gradient_checkpoint import (
            GradientCheckpointManager,
        )

        cfg = GradientCheckpointConfig(mode="auto")
        mgr = GradientCheckpointManager(cfg)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_props = MagicMock()
        mock_props.total_mem = 8_000_000_000
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.memory_allocated.return_value = 2_000_000_000
        mock_torch.cuda.memory_reserved.return_value = 3_000_000_000

        with patch.dict("sys.modules", {"torch": mock_torch}):
            stats = mgr.get_memory_stats()
        assert stats["available"] is True
        assert "total_gb" in stats
        assert "allocated_gb" in stats
        assert "reserved_gb" in stats
        assert "free_gb" in stats
        assert "usage_pct" in stats

    def test_get_memory_stats_no_cuda(self) -> None:
        """Memory stats when CUDA is not available."""
        from artenic_ai_sdk_training.gradient_checkpoint import (
            GradientCheckpointManager,
        )

        cfg = GradientCheckpointConfig(mode="auto")
        mgr = GradientCheckpointManager(cfg)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            stats = mgr.get_memory_stats()
        assert stats["available"] is False

    def test_apply_generic_pytorch_child_no_params(self) -> None:
        """Generic path skips children with no parameters."""
        from artenic_ai_sdk_training.gradient_checkpoint import (
            GradientCheckpointManager,
        )

        cfg = GradientCheckpointConfig(mode="always")
        mgr = GradientCheckpointManager(cfg)

        # Child with no parameters
        mock_child = MagicMock()
        mock_child.parameters.return_value = iter([])

        model = MagicMock(spec=["named_children"])
        model.named_children.return_value = [("empty_layer", mock_child)]

        mock_torch = MagicMock()
        mock_ckpt_utils = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "torch.utils": mock_torch.utils,
                "torch.utils.checkpoint": mock_ckpt_utils,
            },
        ):
            result = mgr.apply(model)

        assert result is model
        assert mgr.applied is True


# ===================================================================
# LearningRateFinder
# ===================================================================


class TestLearningRateFinder:
    def test_find_steepest_descent_basic(self) -> None:
        from artenic_ai_sdk_training.lr_finder import LearningRateFinder

        # Create a simple descending-then-ascending loss curve
        lrs = [1e-5 * (10 ** (i / 10)) for i in range(20)]
        # Loss decreases then increases
        losses = [1.0 - 0.1 * i for i in range(10)] + [0.0 + 0.2 * i for i in range(10)]

        suggested = LearningRateFinder._find_steepest_descent(lrs, losses)
        assert suggested > 0
        assert isinstance(suggested, float)

    def test_find_steepest_descent_short(self) -> None:
        from artenic_ai_sdk_training.lr_finder import LearningRateFinder

        assert LearningRateFinder._find_steepest_descent([1e-3], [0.5]) == 1e-3
        assert LearningRateFinder._find_steepest_descent([], []) == 1e-3

    def test_find_steepest_descent_two_points(self) -> None:
        from artenic_ai_sdk_training.lr_finder import LearningRateFinder

        result = LearningRateFinder._find_steepest_descent([1e-4, 1e-3], [1.0, 0.5])
        assert result == 1e-4  # fallback to first

    def test_lr_finder_result_dataclass(self) -> None:
        from artenic_ai_sdk_training.lr_finder import LRFinderResult

        result = LRFinderResult(
            suggested_lr=1e-3,
            min_lr_tested=1e-7,
            max_lr_tested=10.0,
            num_steps=100,
            loss_history=[1.0, 0.5],
            lr_history=[1e-5, 1e-4],
            divergence_index=None,
        )
        assert result.suggested_lr == 1e-3
        assert result.divergence_index is None

    def test_init(self) -> None:
        from artenic_ai_sdk_training.lr_finder import LearningRateFinder

        cfg = LRFinderConfig(num_steps=50)
        finder = LearningRateFinder(cfg)
        assert finder._config.num_steps == 50

    def test_find_basic(self) -> None:
        """Test find() with a fully mocked torch environment."""
        from artenic_ai_sdk_training.lr_finder import LearningRateFinder

        cfg = LRFinderConfig(num_steps=10, min_lr=1e-5, max_lr=1.0)
        finder = LearningRateFinder(cfg)

        # Create mock model
        mock_model = MagicMock()
        mock_model_copy = MagicMock()
        mock_param = MagicMock()
        mock_param.device = "cpu"
        mock_model_copy.parameters.return_value = iter([mock_param])

        # Create mock optimizer
        mock_optimizer_cls = MagicMock()
        mock_optimizer = MagicMock()
        mock_optimizer.param_groups = [{"lr": 1e-5}]
        mock_optimizer_cls.return_value = mock_optimizer

        # Create mock data loader - list of (input, target) tuples
        mock_input = MagicMock()
        mock_input.to.return_value = mock_input
        mock_target = MagicMock()
        mock_target.to.return_value = mock_target
        train_loader = [(mock_input, mock_target)] * 10

        # Mock loss
        mock_loss = MagicMock()
        mock_loss.item.return_value = 0.5  # constant loss
        mock_loss.backward = MagicMock()
        mock_model_copy.return_value = MagicMock()  # model output

        mock_criterion = MagicMock(return_value=mock_loss)

        mock_torch = MagicMock()
        mock_torch.optim.Adam = mock_optimizer_cls
        with (
            patch.dict("sys.modules", {"torch": mock_torch}),
            patch("copy.deepcopy", return_value=mock_model_copy),
        ):
            result = finder.find(mock_model, train_loader, mock_criterion, mock_optimizer_cls)

        assert result.num_steps == 10
        assert result.suggested_lr > 0
        assert len(result.loss_history) == 10
        assert len(result.lr_history) == 10

    def test_find_with_divergence(self) -> None:
        """Test find() when loss diverges."""
        from artenic_ai_sdk_training.lr_finder import LearningRateFinder

        cfg = LRFinderConfig(num_steps=10, min_lr=1e-5, max_lr=10.0, divergence_threshold=2.0)
        finder = LearningRateFinder(cfg)

        mock_model = MagicMock()
        mock_model_copy = MagicMock()
        mock_param = MagicMock()
        mock_param.device = "cpu"
        mock_model_copy.parameters.return_value = iter([mock_param])

        mock_optimizer_cls = MagicMock()
        mock_optimizer = MagicMock()
        mock_optimizer.param_groups = [{"lr": 1e-5}]
        mock_optimizer_cls.return_value = mock_optimizer

        mock_input = MagicMock()
        mock_input.to.return_value = mock_input
        mock_target = MagicMock()
        mock_target.to.return_value = mock_target
        train_loader = [(mock_input, mock_target)] * 20

        # Losses that diverge: starts at 1.0, then jumps to 10.0
        call_count = [0]

        def loss_item() -> float:
            call_count[0] += 1
            if call_count[0] <= 2:
                return 1.0
            return 10.0  # divergence!

        mock_loss = MagicMock()
        mock_loss.item.side_effect = loss_item
        mock_criterion = MagicMock(return_value=mock_loss)

        mock_torch = MagicMock()
        with (
            patch.dict("sys.modules", {"torch": mock_torch}),
            patch("copy.deepcopy", return_value=mock_model_copy),
        ):
            result = finder.find(mock_model, train_loader, mock_criterion, mock_optimizer_cls)

        assert result.divergence_index is not None
        assert result.num_steps < 10

    def test_find_with_data_exhaustion(self) -> None:
        """Test find() when data loader is exhausted and needs cycling."""
        from artenic_ai_sdk_training.lr_finder import LearningRateFinder

        cfg = LRFinderConfig(num_steps=10, min_lr=1e-5, max_lr=1.0)
        finder = LearningRateFinder(cfg)

        mock_model = MagicMock()
        mock_model_copy = MagicMock()
        mock_param = MagicMock()
        mock_param.device = "cpu"
        mock_model_copy.parameters.return_value = iter([mock_param])

        mock_optimizer_cls = MagicMock()
        mock_optimizer = MagicMock()
        mock_optimizer.param_groups = [{"lr": 1e-5}]
        mock_optimizer_cls.return_value = mock_optimizer

        mock_input = MagicMock()
        mock_input.to.return_value = mock_input
        mock_target = MagicMock()
        mock_target.to.return_value = mock_target
        # Only 2 batches — needs cycling to reach 5 steps
        train_loader = [(mock_input, mock_target), (mock_input, mock_target)]

        mock_loss = MagicMock()
        mock_loss.item.return_value = 0.5
        mock_criterion = MagicMock(return_value=mock_loss)

        mock_torch = MagicMock()
        with (
            patch.dict("sys.modules", {"torch": mock_torch}),
            patch("copy.deepcopy", return_value=mock_model_copy),
        ):
            result = finder.find(mock_model, train_loader, mock_criterion, mock_optimizer_cls)

        assert result.num_steps == 10

    def test_find_with_dict_batch(self) -> None:
        """Test find() with non-tuple batch (dict-style)."""
        from artenic_ai_sdk_training.lr_finder import LearningRateFinder

        cfg = LRFinderConfig(num_steps=10, min_lr=1e-5, max_lr=1.0)
        finder = LearningRateFinder(cfg)

        mock_model = MagicMock()
        mock_model_copy = MagicMock()
        mock_param = MagicMock()
        mock_param.device = "cpu"
        mock_model_copy.parameters.return_value = iter([mock_param])

        mock_optimizer_cls = MagicMock()
        mock_optimizer = MagicMock()
        mock_optimizer.param_groups = [{"lr": 1e-5}]
        mock_optimizer_cls.return_value = mock_optimizer

        # Dict-style batch (not tuple) — hits the else branch
        mock_batch = MagicMock(spec=[])  # no list/tuple methods
        mock_batch.to = MagicMock(return_value=mock_batch)
        train_loader = [mock_batch] * 5

        mock_loss = MagicMock()
        mock_loss.item.return_value = 0.5
        mock_criterion = MagicMock(return_value=mock_loss)

        mock_torch = MagicMock()
        with (
            patch.dict("sys.modules", {"torch": mock_torch}),
            patch("copy.deepcopy", return_value=mock_model_copy),
        ):
            result = finder.find(mock_model, train_loader, mock_criterion, mock_optimizer_cls)

        assert result.num_steps == 10

    def test_find_default_optimizer(self) -> None:
        """Test find() without specifying optimizer_class (uses torch.optim.Adam)."""
        from artenic_ai_sdk_training.lr_finder import LearningRateFinder

        cfg = LRFinderConfig(num_steps=10, min_lr=1e-5, max_lr=1.0)
        finder = LearningRateFinder(cfg)

        mock_model = MagicMock()
        mock_model_copy = MagicMock()
        mock_param = MagicMock()
        mock_param.device = "cpu"
        mock_model_copy.parameters.return_value = iter([mock_param])

        mock_optimizer = MagicMock()
        mock_optimizer.param_groups = [{"lr": 1e-5}]

        mock_torch = MagicMock()
        mock_torch.optim.Adam.return_value = mock_optimizer

        mock_input = MagicMock()
        mock_input.to.return_value = mock_input
        mock_target = MagicMock()
        mock_target.to.return_value = mock_target
        train_loader = [(mock_input, mock_target)] * 5

        mock_loss = MagicMock()
        mock_loss.item.return_value = 0.5
        mock_criterion = MagicMock(return_value=mock_loss)

        with (
            patch.dict("sys.modules", {"torch": mock_torch}),
            patch("copy.deepcopy", return_value=mock_model_copy),
        ):
            result = finder.find(mock_model, train_loader, mock_criterion)

        assert result.num_steps == 10

    def test_find_steepest_descent_no_gradients(self) -> None:
        """Line 166 is unreachable dead code (gradients can't be empty when len >= 3)."""
        # Verified: range(1, len(losses) - 1) is empty only if len <= 2,
        # but that's caught by the len(lrs) < 3 early return.
        pass


# ===================================================================
# FSDPManager
# ===================================================================


class TestFSDPManager:
    def test_not_available_no_torch(self) -> None:
        from artenic_ai_sdk_training.distributed import FSDPManager

        cfg = DistributedConfig()
        mgr = FSDPManager(cfg)
        with patch.dict("sys.modules", {"torch": None}):
            assert mgr.is_available() is False

    def test_is_available_with_cuda(self) -> None:
        from artenic_ai_sdk_training.distributed import FSDPManager

        cfg = DistributedConfig()
        mgr = FSDPManager(cfg)
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 4
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert mgr.is_available() is True

    def test_is_available_single_gpu(self) -> None:
        from artenic_ai_sdk_training.distributed import FSDPManager

        cfg = DistributedConfig()
        mgr = FSDPManager(cfg)
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert mgr.is_available() is False

    def test_rank_defaults(self) -> None:
        from artenic_ai_sdk_training.distributed import FSDPManager

        cfg = DistributedConfig()
        mgr = FSDPManager(cfg)
        assert mgr.rank == 0
        assert mgr.world_size == 1
        assert mgr.is_main_process is True
        assert mgr.context is None

    def test_distributed_context_dataclass(self) -> None:
        from artenic_ai_sdk_training.distributed import DistributedContext

        ctx = DistributedContext(
            rank=1, world_size=4, local_rank=1, backend="nccl", device="cuda:1"
        )
        assert ctx.rank == 1
        assert ctx.world_size == 4
        assert ctx.device == "cuda:1"

    def test_cleanup_no_torch(self) -> None:
        from artenic_ai_sdk_training.distributed import FSDPManager

        cfg = DistributedConfig()
        mgr = FSDPManager(cfg)
        with patch.dict("sys.modules", {"torch": None, "torch.distributed": None}):
            mgr.cleanup()

    def test_cleanup_with_torch(self) -> None:
        from artenic_ai_sdk_training.distributed import FSDPManager

        cfg = DistributedConfig()
        mgr = FSDPManager(cfg)
        mgr._initialized = True

        mock_torch = MagicMock()
        mock_torch.distributed.is_initialized.return_value = True
        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "torch.distributed": mock_torch.distributed},
        ):
            mgr.cleanup()
        mock_torch.distributed.destroy_process_group.assert_called_once()
        assert mgr._initialized is False
        assert mgr._context is None

    def test_cleanup_not_initialized(self) -> None:
        from artenic_ai_sdk_training.distributed import FSDPManager

        cfg = DistributedConfig()
        mgr = FSDPManager(cfg)
        mock_torch = MagicMock()
        mock_torch.distributed.is_initialized.return_value = False
        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "torch.distributed": mock_torch.distributed},
        ):
            mgr.cleanup()
        mock_torch.distributed.destroy_process_group.assert_not_called()

    def test_wrap_model_calls_setup(self) -> None:
        from artenic_ai_sdk_training.distributed import FSDPManager

        cfg = DistributedConfig(strategy="ddp")
        mgr = FSDPManager(cfg)

        with (
            patch.object(mgr, "setup") as mock_setup,
            patch.object(mgr, "_wrap_ddp", return_value="wrapped"),
        ):
            mock_setup.return_value = MagicMock()
            result = mgr.wrap_model(MagicMock())
        mock_setup.assert_called_once()
        assert result == "wrapped"

    def test_wrap_model_fsdp_strategy(self) -> None:
        from artenic_ai_sdk_training.distributed import FSDPManager

        cfg = DistributedConfig(strategy="fsdp")
        mgr = FSDPManager(cfg)
        mgr._initialized = True

        with patch.object(mgr, "_wrap_fsdp", return_value="fsdp_wrapped"):
            result = mgr.wrap_model(MagicMock())
        assert result == "fsdp_wrapped"

    def test_setup_with_cuda(self) -> None:
        from artenic_ai_sdk_training.distributed import FSDPManager

        cfg = DistributedConfig()
        mgr = FSDPManager(cfg)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.distributed.is_initialized.return_value = False

        with (
            patch.dict(
                "sys.modules",
                {"torch": mock_torch, "torch.distributed": mock_torch.distributed},
            ),
            patch.dict("os.environ", {"RANK": "0", "WORLD_SIZE": "2", "LOCAL_RANK": "0"}),
        ):
            ctx = mgr.setup()

        assert ctx.rank == 0
        assert ctx.world_size == 2
        assert ctx.backend == "nccl"
        assert ctx.device == "cuda:0"
        assert mgr._initialized is True
        mock_torch.distributed.init_process_group.assert_called_once()

    def test_setup_cpu_only(self) -> None:
        from artenic_ai_sdk_training.distributed import FSDPManager

        cfg = DistributedConfig()
        mgr = FSDPManager(cfg)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.distributed.is_initialized.return_value = False

        with (
            patch.dict(
                "sys.modules",
                {"torch": mock_torch, "torch.distributed": mock_torch.distributed},
            ),
            patch.dict("os.environ", {"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"}),
        ):
            ctx = mgr.setup()

        assert ctx.backend == "gloo"
        assert ctx.device == "cpu"

    def test_setup_already_initialized(self) -> None:
        from artenic_ai_sdk_training.distributed import FSDPManager

        cfg = DistributedConfig()
        mgr = FSDPManager(cfg)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.distributed.is_initialized.return_value = True

        with (
            patch.dict(
                "sys.modules",
                {"torch": mock_torch, "torch.distributed": mock_torch.distributed},
            ),
            patch.dict("os.environ", {}),
        ):
            ctx = mgr.setup()

        mock_torch.distributed.init_process_group.assert_not_called()
        assert ctx.rank == 0

    def test_setup_invalid_env_var(self) -> None:
        from artenic_ai_sdk_training.distributed import FSDPManager

        cfg = DistributedConfig()
        mgr = FSDPManager(cfg)

        mock_torch = MagicMock()
        with (
            patch.dict(
                "sys.modules",
                {"torch": mock_torch, "torch.distributed": mock_torch.distributed},
            ),
            patch.dict("os.environ", {"RANK": "not_a_number", "WORLD_SIZE": "1"}),
            pytest.raises(RuntimeError, match="Invalid distributed environment variable"),
        ):
            mgr.setup()

    def test_setup_rank_out_of_range(self) -> None:
        from artenic_ai_sdk_training.distributed import FSDPManager

        cfg = DistributedConfig()
        mgr = FSDPManager(cfg)

        mock_torch = MagicMock()
        with (
            patch.dict(
                "sys.modules",
                {"torch": mock_torch, "torch.distributed": mock_torch.distributed},
            ),
            patch.dict("os.environ", {"RANK": "5", "WORLD_SIZE": "2", "LOCAL_RANK": "0"}),
            pytest.raises(RuntimeError, match="RANK=5 must be in"),
        ):
            mgr.setup()

    def test_wrap_ddp(self) -> None:
        from artenic_ai_sdk_training.distributed import DistributedContext, FSDPManager

        cfg = DistributedConfig(strategy="ddp")
        mgr = FSDPManager(cfg)
        mgr._context = DistributedContext(
            rank=0, world_size=2, local_rank=0, backend="nccl", device="cuda:0"
        )

        mock_ddp = MagicMock(return_value="ddp_wrapped")
        mock_torch = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "torch.nn": mock_torch.nn,
                "torch.nn.parallel": mock_torch.nn.parallel,
            },
        ):
            mock_torch.nn.parallel.DistributedDataParallel = mock_ddp
            result = mgr._wrap_ddp(MagicMock())
        assert result == "ddp_wrapped"

    def test_wrap_fsdp(self) -> None:
        from artenic_ai_sdk_training.distributed import DistributedContext, FSDPManager

        cfg = DistributedConfig(
            strategy="fsdp",
            sharding_strategy="full_shard",
            mixed_precision_in_fsdp=False,
            cpu_offload=False,
            auto_wrap_policy="none",
        )
        mgr = FSDPManager(cfg)
        mgr._context = DistributedContext(
            rank=0, world_size=2, local_rank=0, backend="nccl", device="cuda:0"
        )

        mock_fsdp_cls = MagicMock(return_value="fsdp_wrapped")
        mock_torch = MagicMock()
        mock_fsdp_module = MagicMock()
        mock_fsdp_module.FullyShardedDataParallel = mock_fsdp_cls
        mock_fsdp_module.ShardingStrategy.FULL_SHARD = "full_shard"
        mock_fsdp_module.ShardingStrategy.SHARD_GRAD_OP = "shard_grad_op"
        mock_fsdp_module.ShardingStrategy.NO_SHARD = "no_shard"

        with patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "torch.distributed": MagicMock(),
                "torch.distributed.fsdp": mock_fsdp_module,
            },
        ):
            result = mgr._wrap_fsdp(MagicMock())
        assert result == "fsdp_wrapped"

    def test_wrap_fsdp_with_mp_and_offload(self) -> None:
        from artenic_ai_sdk_training.distributed import DistributedContext, FSDPManager

        cfg = DistributedConfig(
            strategy="fsdp",
            sharding_strategy="shard_grad_op",
            mixed_precision_in_fsdp=True,
            cpu_offload=True,
            auto_wrap_policy="none",
        )
        mgr = FSDPManager(cfg)
        mgr._context = DistributedContext(
            rank=0, world_size=2, local_rank=0, backend="nccl", device="cuda:0"
        )

        mock_fsdp_cls = MagicMock(return_value="fsdp_mp_offload")
        mock_torch = MagicMock()
        mock_fsdp_module = MagicMock()
        mock_fsdp_module.FullyShardedDataParallel = mock_fsdp_cls
        mock_fsdp_module.ShardingStrategy.FULL_SHARD = "full_shard"
        mock_fsdp_module.ShardingStrategy.SHARD_GRAD_OP = "shard_grad_op"
        mock_fsdp_module.ShardingStrategy.NO_SHARD = "no_shard"

        with patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "torch.distributed": MagicMock(),
                "torch.distributed.fsdp": mock_fsdp_module,
            },
        ):
            result = mgr._wrap_fsdp(MagicMock())
        assert result == "fsdp_mp_offload"
        mock_fsdp_module.MixedPrecision.assert_called_once()
        mock_fsdp_module.CPUOffload.assert_called_once()

    def test_get_auto_wrap_policy_none(self) -> None:
        from artenic_ai_sdk_training.distributed import FSDPManager

        cfg = DistributedConfig(auto_wrap_policy="none")
        mgr = FSDPManager(cfg)
        assert mgr._get_auto_wrap_policy() is None

    def test_get_auto_wrap_policy_size_based(self) -> None:
        from artenic_ai_sdk_training.distributed import FSDPManager

        cfg = DistributedConfig(auto_wrap_policy="size_based", min_params_for_wrap=1000)
        mgr = FSDPManager(cfg)

        mock_wrap = MagicMock()
        mock_torch = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "torch.distributed": MagicMock(),
                "torch.distributed.fsdp": MagicMock(),
                "torch.distributed.fsdp.wrap": mock_wrap,
            },
        ):
            policy = mgr._get_auto_wrap_policy()
        assert policy is not None

    def test_get_auto_wrap_policy_transformer_based(self) -> None:
        from artenic_ai_sdk_training.distributed import FSDPManager

        cfg = DistributedConfig(auto_wrap_policy="transformer_based")
        mgr = FSDPManager(cfg)

        mock_wrap = MagicMock()
        mock_nn = MagicMock()
        mock_torch = MagicMock()
        mock_torch.nn = mock_nn
        with patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "torch.nn": mock_nn,
                "torch.distributed": MagicMock(),
                "torch.distributed.fsdp": MagicMock(),
                "torch.distributed.fsdp.wrap": mock_wrap,
            },
        ):
            policy = mgr._get_auto_wrap_policy()
        assert policy is not None

    def test_get_auto_wrap_policy_fallback(self) -> None:
        """Line 211: The final `return None` fallback in _get_auto_wrap_policy."""
        from artenic_ai_sdk_training.distributed import FSDPManager

        cfg = DistributedConfig(auto_wrap_policy="none")
        mgr = FSDPManager(cfg)
        # Force a non-standard value to hit the final return None
        mgr._config = MagicMock()
        mgr._config.auto_wrap_policy = "something_else"
        assert mgr._get_auto_wrap_policy() is None


# ===================================================================
# build_callbacks()
# ===================================================================


class TestBuildCallbacks:
    def test_empty_config(self) -> None:
        from artenic_ai_sdk_training import build_callbacks

        cfg = TrainingConfig(version="1.0")
        runner = build_callbacks(cfg)
        assert isinstance(runner, CallbackRunner)
        assert len(runner.callbacks) == 0

    def test_early_stopping_enabled(self) -> None:
        from artenic_ai_sdk_training import build_callbacks
        from artenic_ai_sdk_training.early_stopping import EarlyStopping

        cfg = TrainingConfig(
            version="1.0",
            early_stopping=EarlyStoppingConfig(enabled=True, patience=3),
        )
        runner = build_callbacks(cfg)
        assert len(runner.callbacks) == 1
        assert isinstance(runner.callbacks[0], EarlyStopping)

    def test_checkpoint_enabled(self, tmp_path: Path) -> None:
        from artenic_ai_sdk_training import build_callbacks
        from artenic_ai_sdk_training.checkpointing import SmartCheckpointer

        cfg = TrainingConfig(
            version="1.0",
            checkpoint=CheckpointConfig(enabled=True),
        )
        runner = build_callbacks(cfg, output_dir=tmp_path)
        assert len(runner.callbacks) == 1
        assert isinstance(runner.callbacks[0], SmartCheckpointer)

    def test_both_enabled(self, tmp_path: Path) -> None:
        from artenic_ai_sdk_training import build_callbacks

        cfg = TrainingConfig(
            version="1.0",
            early_stopping=EarlyStoppingConfig(enabled=True),
            checkpoint=CheckpointConfig(enabled=True),
        )
        runner = build_callbacks(cfg, output_dir=tmp_path)
        assert len(runner.callbacks) == 2


# ===================================================================
# Lazy imports (__getattr__)
# ===================================================================


class TestLazyImports:
    def test_import_smart_data_splitter(self) -> None:
        from artenic_ai_sdk_training import SmartDataSplitter
        from artenic_ai_sdk_training.data_splitting import (
            SmartDataSplitter as DirectImport,
        )

        assert SmartDataSplitter is DirectImport

    def test_import_dataset_versioner(self) -> None:
        from artenic_ai_sdk_training import DatasetVersioner
        from artenic_ai_sdk_training.data_versioning import (
            DatasetVersioner as DirectImport,
        )

        assert DatasetVersioner is DirectImport

    def test_import_mixed_precision_manager(self) -> None:
        from artenic_ai_sdk_training import MixedPrecisionManager
        from artenic_ai_sdk_training.mixed_precision import (
            MixedPrecisionManager as DirectImport,
        )

        assert MixedPrecisionManager is DirectImport

    def test_import_early_stopping(self) -> None:
        from artenic_ai_sdk_training import EarlyStopping
        from artenic_ai_sdk_training.early_stopping import (
            EarlyStopping as DirectImport,
        )

        assert EarlyStopping is DirectImport

    def test_import_lr_finder(self) -> None:
        from artenic_ai_sdk_training import LearningRateFinder
        from artenic_ai_sdk_training.lr_finder import (
            LearningRateFinder as DirectImport,
        )

        assert LearningRateFinder is DirectImport

    def test_import_fsdp_manager(self) -> None:
        from artenic_ai_sdk_training import FSDPManager
        from artenic_ai_sdk_training.distributed import FSDPManager as DirectImport

        assert FSDPManager is DirectImport

    def test_import_gradient_checkpoint_manager(self) -> None:
        from artenic_ai_sdk_training import GradientCheckpointManager
        from artenic_ai_sdk_training.gradient_checkpoint import (
            GradientCheckpointManager as DirectImport,
        )

        assert GradientCheckpointManager is DirectImport

    def test_getattr_nonexistent_raises(self) -> None:
        import artenic_ai_sdk_training as training_mod

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = training_mod.NonExistent  # type: ignore[attr-defined]


# ===================================================================
# Integration: full training loop simulation
# ===================================================================


class TestIntegration:
    def test_full_loop_with_early_stopping(self, tmp_path: Path) -> None:
        """Simulate a training loop with early stopping + checkpointing."""
        from artenic_ai_sdk_training import build_callbacks

        cfg = TrainingConfig(
            version="1.0",
            epochs=20,
            early_stopping=EarlyStoppingConfig(enabled=True, patience=3, min_delta=0.0),
            checkpoint=CheckpointConfig(
                enabled=True,
                save_best_only=True,
                preemption_handler=False,
            ),
        )
        runner = build_callbacks(cfg, output_dir=tmp_path)
        ctx = _make_context(tmp_path)
        runner.on_train_start(ctx)

        # Simulate: loss decreases then plateaus
        losses = [1.0, 0.8, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5]
        stopped_at = None
        for epoch, loss in enumerate(losses):
            runner.on_epoch_start(epoch, ctx)
            runner.on_epoch_end(epoch, {"val_loss": loss}, ctx)
            if runner.should_stop():
                stopped_at = epoch
                break

        runner.on_train_end(ctx)

        # Should stop at epoch 6 (3 epochs without improvement after epoch 3)
        assert stopped_at == 6
        assert ctx.stop_requested is True
        # Best checkpoint should exist
        assert (tmp_path / "checkpoints" / "best" / "metadata.json").exists()

    def test_data_split_then_version(self, tmp_path: Path) -> None:
        """Split data and version it."""
        from artenic_ai_sdk_training.data_splitting import SmartDataSplitter
        from artenic_ai_sdk_training.data_versioning import DatasetVersioner

        # Create a data file
        data_file = tmp_path / "train.csv"
        data_file.write_text(
            "x,y\n" + "\n".join(f"{i},{i * 2}" for i in range(100)),
            encoding="utf-8",
        )

        # Version it
        versioner = DatasetVersioner()
        version = versioner.hash_file(data_file)
        assert version.size_bytes > 0

        # Split in-memory data
        splitter = SmartDataSplitter(DataSplitConfig(train_ratio=0.8, val_ratio=0.2))
        result = splitter.split(list(range(100)))
        assert len(result.train) == 80
        assert len(result.val) == 20
