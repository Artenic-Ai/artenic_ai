"""Distributed Training — PyTorch FSDP/DDP manager."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from artenic_ai_sdk.training.config import DistributedConfig

logger = logging.getLogger(__name__)


@dataclass
class DistributedContext:
    """Information about the distributed training environment."""

    rank: int
    world_size: int
    local_rank: int
    backend: str
    device: str


class FSDPManager:
    """Configure and manage PyTorch FSDP (Fully Sharded Data Parallelism).

    Handles process group initialization, model wrapping with FSDP,
    auto-wrap policy selection, mixed precision within FSDP,
    and CPU offload for memory-constrained setups.
    """

    def __init__(self, config: DistributedConfig) -> None:
        self._config = config
        self._context: DistributedContext | None = None
        self._initialized = False

    @property
    def rank(self) -> int:
        return self._context.rank if self._context else 0

    @property
    def world_size(self) -> int:
        return self._context.world_size if self._context else 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def context(self) -> DistributedContext | None:
        return self._context

    def is_available(self) -> bool:
        """Check if multi-GPU distributed training is possible."""
        try:
            import torch

            return bool(torch.cuda.is_available() and torch.cuda.device_count() > 1)
        except ImportError:
            return False

    def setup(self) -> DistributedContext:
        """Initialize the distributed process group."""
        import torch
        import torch.distributed as dist

        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        backend = "nccl" if torch.cuda.is_available() else "gloo"

        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
            )

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

        self._context = DistributedContext(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            backend=backend,
            device=device,
        )
        self._initialized = True

        logger.info(
            "Distributed: rank %d/%d on %s (backend=%s)",
            rank,
            world_size,
            device,
            backend,
        )
        return self._context

    def wrap_model(self, model: Any) -> Any:
        """Wrap a model with FSDP or DDP based on configuration."""
        if not self._initialized:
            self.setup()

        if self._config.strategy == "ddp":
            return self._wrap_ddp(model)
        return self._wrap_fsdp(model)

    def cleanup(self) -> None:
        """Destroy the distributed process group."""
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
                self._initialized = False
                self._context = None
                logger.info("Distributed: process group destroyed")
        except ImportError:
            pass

    def _wrap_fsdp(self, model: Any) -> Any:
        """Wrap with FullyShardedDataParallel."""
        import torch
        from torch.distributed.fsdp import (
            CPUOffload,
            FullyShardedDataParallel,
            MixedPrecision,
            ShardingStrategy,
        )

        sharding_map = {
            "full_shard": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "no_shard": ShardingStrategy.NO_SHARD,
        }
        sharding = sharding_map[self._config.sharding_strategy]

        # Mixed precision policy for FSDP
        mp_policy = None
        if self._config.mixed_precision_in_fsdp:
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )

        # CPU offload
        cpu_offload = CPUOffload(offload_params=True) if self._config.cpu_offload else None

        # Auto-wrap policy
        auto_wrap_policy = self._get_auto_wrap_policy()

        wrapped = FullyShardedDataParallel(
            model,
            sharding_strategy=sharding,
            mixed_precision=mp_policy,
            cpu_offload=cpu_offload,
            auto_wrap_policy=auto_wrap_policy,
        )
        logger.info(
            "FSDP: model wrapped (sharding=%s)",
            self._config.sharding_strategy,
        )
        return wrapped

    def _wrap_ddp(self, model: Any) -> Any:
        """Wrap with DistributedDataParallel."""
        from torch.nn.parallel import DistributedDataParallel

        local_rank = self._context.local_rank if self._context else 0
        wrapped = DistributedDataParallel(model, device_ids=[local_rank])
        logger.info("DDP: model wrapped (local_rank=%d)", local_rank)
        return wrapped

    def _get_auto_wrap_policy(self) -> Any:
        """Get the FSDP auto-wrap policy."""
        if self._config.auto_wrap_policy == "none":
            return None

        if self._config.auto_wrap_policy == "size_based":
            from functools import partial

            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

            return partial(
                size_based_auto_wrap_policy,
                min_num_params=self._config.min_params_for_wrap,
            )

        if self._config.auto_wrap_policy == "transformer_based":
            from functools import partial

            import torch.nn as nn
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

            return partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={
                    nn.TransformerEncoderLayer,
                    nn.TransformerDecoderLayer,
                },
            )

        return None
