"""Mixed Precision Auto-Detect — FP16/BF16 based on GPU capability."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from artenic_ai_sdk_training.config import MixedPrecisionConfig

logger = logging.getLogger(__name__)


@dataclass
class PrecisionMode:
    """Detected precision mode for the current device."""

    dtype: str  # "float32", "float16", "bfloat16"
    device: str  # "cuda", "cpu"
    gpu_name: str
    compute_capability: tuple[int, int] | None
    scaler_needed: bool


class MixedPrecisionManager:
    """Auto-detect and configure mixed precision training.

    Detects GPU capabilities and selects the best precision mode:
    - BF16 for Ampere+ GPUs (A100, H100, RTX 3090+) — CC >= 8.0
    - FP16 for older GPUs (V100, T4) — CC >= 7.0
    - FP32 fallback for CPUs or unsupported GPUs
    """

    def __init__(self, config: MixedPrecisionConfig) -> None:
        self._config = config
        self._mode: PrecisionMode | None = None

    def detect(self) -> PrecisionMode:
        """Detect the best precision mode for the current hardware."""
        if self._mode is not None:
            return self._mode

        try:
            import torch
        except ImportError:
            self._mode = PrecisionMode(
                dtype="float32",
                device="cpu",
                gpu_name="N/A (torch not available)",
                compute_capability=None,
                scaler_needed=False,
            )
            return self._mode

        if not torch.cuda.is_available():
            self._mode = PrecisionMode(
                dtype="float32",
                device="cpu",
                gpu_name="CPU",
                compute_capability=None,
                scaler_needed=False,
            )
            return self._mode

        gpu_name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)

        if self._config.mode == "off":
            self._mode = PrecisionMode(
                dtype="float32",
                device="cuda",
                gpu_name=gpu_name,
                compute_capability=cc,
                scaler_needed=False,
            )
        elif self._config.mode == "bf16":
            self._mode = PrecisionMode(
                dtype="bfloat16",
                device="cuda",
                gpu_name=gpu_name,
                compute_capability=cc,
                scaler_needed=False,
            )
        elif self._config.mode == "fp16":
            self._mode = PrecisionMode(
                dtype="float16",
                device="cuda",
                gpu_name=gpu_name,
                compute_capability=cc,
                scaler_needed=True,
            )
        elif cc[0] >= 8 and torch.cuda.is_bf16_supported():
            # Auto-detect: BF16 for Ampere+
            self._mode = PrecisionMode(
                dtype="bfloat16",
                device="cuda",
                gpu_name=gpu_name,
                compute_capability=cc,
                scaler_needed=False,
            )
        elif cc[0] >= 7:
            # Auto-detect: FP16 for older GPUs
            self._mode = PrecisionMode(
                dtype="float16",
                device="cuda",
                gpu_name=gpu_name,
                compute_capability=cc,
                scaler_needed=True,
            )
        else:
            # Fallback: FP32
            self._mode = PrecisionMode(
                dtype="float32",
                device="cuda",
                gpu_name=gpu_name,
                compute_capability=cc,
                scaler_needed=False,
            )

        logger.info(
            "MixedPrecision: detected %s on %s (CC %s)",
            self._mode.dtype,
            gpu_name,
            cc,
        )
        return self._mode

    def get_autocast_context(self) -> Any:
        """Return a torch.autocast context manager for the detected precision."""
        import torch

        mode = self.detect()
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return torch.autocast(
            device_type=mode.device,
            dtype=dtype_map[mode.dtype],
            enabled=mode.dtype != "float32",
        )

    def get_grad_scaler(self) -> Any:
        """Return a GradScaler if FP16, else None."""
        import torch

        mode = self.detect()
        if mode.scaler_needed:
            if self._config.loss_scale == "dynamic":
                return torch.amp.GradScaler("cuda")
            return torch.amp.GradScaler(
                "cuda",
                init_scale=self._config.static_loss_scale,
                enabled=True,
            )
        return None

    def get_dtype(self) -> Any:
        """Return the torch dtype for the detected precision."""
        import torch

        mode = self.detect()
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[mode.dtype]
