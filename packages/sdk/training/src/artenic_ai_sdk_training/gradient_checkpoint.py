"""Gradient Checkpointing — auto-enable when GPU memory is tight."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from artenic_ai_sdk_training.config import GradientCheckpointConfig

logger = logging.getLogger(__name__)


class GradientCheckpointManager:
    """Auto-enable gradient checkpointing when GPU memory is tight.

    In 'auto' mode, checks ``torch.cuda.memory_allocated()`` against
    the configured threshold and enables checkpointing if memory usage
    is high. In 'always' mode, unconditionally enables it.
    """

    def __init__(self, config: GradientCheckpointConfig) -> None:
        self._config = config
        self._applied = False

    @property
    def applied(self) -> bool:
        return self._applied

    def should_enable(self, model: Any = None) -> bool:
        """Determine whether gradient checkpointing should be enabled."""
        if self._config.mode == "off":
            return False
        if self._config.mode == "always":
            return True

        # Auto mode: check GPU memory usage
        try:
            import torch

            if not torch.cuda.is_available():
                return False

            total = torch.cuda.get_device_properties(0).total_mem
            allocated = torch.cuda.memory_allocated(0)
            usage = allocated / total if total > 0 else 0.0
            logger.info(
                "GradientCheckpoint: GPU memory %.1f%% used (threshold %.1f%%)",
                usage * 100,
                self._config.memory_threshold_pct * 100,
            )
            return usage >= self._config.memory_threshold_pct
        except ImportError:
            return False

    def apply(self, model: Any) -> Any:
        """Apply gradient checkpointing to the model.

        Supports HuggingFace models (``gradient_checkpointing_enable()``)
        and generic ``nn.Module`` via ``torch.utils.checkpoint``.
        Returns the model (in-place modification).
        """
        if self._config.mode == "off":
            return model

        if not self.should_enable(model):
            logger.info("GradientCheckpoint: not needed, skipping")
            return model

        # HuggingFace models
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            self._applied = True
            logger.info("GradientCheckpoint: enabled via HF API")
            return model

        # Generic PyTorch models
        try:
            import torch.utils.checkpoint as ckpt_utils

            for _name, module in model.named_children():
                if sum(1 for _ in module.parameters()) > 0:
                    original_forward = module.forward

                    def make_ckpt_forward(mod: Any, orig_fn: Any) -> Any:
                        def ckpt_forward(*args: Any, **kwargs: Any) -> Any:
                            return ckpt_utils.checkpoint(
                                orig_fn, *args, use_reentrant=False, **kwargs
                            )

                        return ckpt_forward

                    module.forward = make_ckpt_forward(module, original_forward)
            self._applied = True
            logger.info("GradientCheckpoint: enabled via torch.utils.checkpoint")
        except ImportError:
            logger.warning("GradientCheckpoint: torch not available")

        return model

    def get_memory_stats(self) -> dict[str, float | bool]:
        """Return current GPU memory statistics."""
        try:
            import torch

            if not torch.cuda.is_available():
                return {"available": False}
            props = torch.cuda.get_device_properties(0)
            return {
                "available": True,
                "total_gb": props.total_mem / (1024**3),
                "allocated_gb": torch.cuda.memory_allocated(0) / (1024**3),
                "reserved_gb": torch.cuda.memory_reserved(0) / (1024**3),
                "free_gb": (props.total_mem - torch.cuda.memory_allocated(0)) / (1024**3),
                "usage_pct": torch.cuda.memory_allocated(0) / props.total_mem,
            }
        except ImportError:
            return {"available": False}
