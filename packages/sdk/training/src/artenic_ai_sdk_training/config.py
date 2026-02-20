"""Training Intelligence configuration — extends ModelConfig with training options."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from artenic_ai_sdk.schemas import ModelConfig

# =============================================================================
# Sub-configs for each Training Intelligence feature
# =============================================================================


class DataSplitConfig(BaseModel):
    """Feature 1: Smart Data Splitting configuration."""

    enabled: bool = False
    strategy: Literal["holdout", "kfold", "stratified_kfold", "time_series"] = "holdout"
    train_ratio: float = Field(default=0.8, ge=0.0, le=1.0)
    val_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    test_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    n_folds: int = Field(default=5, ge=2)
    stratify_column: str | None = None
    time_column: str | None = None
    random_seed: int = 42


class DataVersioningConfig(BaseModel):
    """Feature 2: Data Versioning configuration."""

    enabled: bool = False
    hash_algorithm: Literal["sha256", "xxhash"] = "sha256"
    sample_size: int | None = Field(default=None, ge=1)


class MixedPrecisionConfig(BaseModel):
    """Feature 3: Mixed Precision Auto-Detect configuration."""

    enabled: bool = False
    mode: Literal["auto", "fp16", "bf16", "off"] = "auto"
    loss_scale: Literal["dynamic", "static"] = "dynamic"
    static_loss_scale: float = Field(default=1024.0, gt=0.0)


class GradientCheckpointConfig(BaseModel):
    """Feature 4: Gradient Checkpointing configuration."""

    enabled: bool = False
    mode: Literal["auto", "always", "off"] = "auto"
    memory_threshold_pct: float = Field(default=0.85, ge=0.0, le=1.0)


class CheckpointConfig(BaseModel):
    """Feature 5: Smart Checkpointing configuration."""

    enabled: bool = False
    save_every_n_epochs: int = Field(default=1, ge=1)
    save_best_only: bool = True
    metric: str = "val_loss"
    mode: Literal["min", "max"] = "min"
    max_checkpoints: int = Field(default=3, ge=1)
    preemption_handler: bool = True


class EarlyStoppingConfig(BaseModel):
    """Feature 6: Early Stopping configuration."""

    enabled: bool = False
    patience: int = Field(default=5, ge=1)
    min_delta: float = Field(default=0.001, ge=0.0)
    metric: str = "val_loss"
    mode: Literal["min", "max"] = "min"


class LRFinderConfig(BaseModel):
    """Feature 7: Learning Rate Finder configuration."""

    enabled: bool = False
    min_lr: float = Field(default=1e-7, gt=0.0)
    max_lr: float = Field(default=10.0, gt=0.0)
    num_steps: int = Field(default=100, ge=10)
    divergence_threshold: float = Field(default=4.0, gt=1.0)
    smooth_factor: float = Field(default=0.05, ge=0.0, le=1.0)


class DistributedConfig(BaseModel):
    """Feature 8: Distributed Training (FSDP/DDP) configuration."""

    enabled: bool = False
    strategy: Literal["fsdp", "ddp", "off"] = "fsdp"
    sharding_strategy: Literal["full_shard", "shard_grad_op", "no_shard"] = "full_shard"
    auto_wrap_policy: Literal["size_based", "transformer_based", "none"] = "size_based"
    min_params_for_wrap: int = Field(default=100_000, ge=1)
    mixed_precision_in_fsdp: bool = True
    cpu_offload: bool = False
    backward_prefetch: bool = True


# =============================================================================
# TrainingConfig — extends ModelConfig
# =============================================================================


class TrainingConfig(ModelConfig):
    """Extended ModelConfig with all Training Intelligence options.

    Inherits from ModelConfig so it passes through the existing
    dispatch pipeline unchanged. All training intelligence fields
    are optional with sane defaults (disabled by default).
    """

    # Training basics
    epochs: int = Field(default=10, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=0.001, gt=0.0)

    # Training Intelligence features
    data_split: DataSplitConfig = Field(default_factory=DataSplitConfig)
    data_versioning: DataVersioningConfig = Field(default_factory=DataVersioningConfig)
    mixed_precision: MixedPrecisionConfig = Field(default_factory=MixedPrecisionConfig)
    gradient_checkpoint: GradientCheckpointConfig = Field(
        default_factory=GradientCheckpointConfig,
    )
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    lr_finder: LRFinderConfig = Field(default_factory=LRFinderConfig)
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)

    def active_features(self) -> dict[str, bool]:
        """Return a summary of which training intelligence features are enabled."""
        return {
            "data_split": self.data_split.enabled,
            "data_versioning": self.data_versioning.enabled,
            "mixed_precision": self.mixed_precision.enabled,
            "gradient_checkpoint": self.gradient_checkpoint.enabled,
            "checkpoint": self.checkpoint.enabled,
            "early_stopping": self.early_stopping.enabled,
            "lr_finder": self.lr_finder.enabled,
            "distributed": self.distributed.enabled,
        }
