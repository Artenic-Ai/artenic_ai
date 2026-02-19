"""Model serialization with safetensors (primary) and multiple format support.

Each saved model produces a directory with:
    model.safetensors (or model.onnx / model.pt / model.pts / model.pkl / model.joblib)
    metadata.json
    config.json (if config provided)

Security Warning:
    Pickle and torch.load can execute arbitrary code during deserialization.
    Only load models from trusted sources. Prefer safetensors format when possible.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from artenic_ai_sdk.exceptions import (
    ArtifactCorruptedError,
    FormatNotSupportedError,
    SerializationError,
)
from artenic_ai_sdk.schemas import ModelConfig, ModelMetadata
from artenic_ai_sdk.types import SerializationFormat


class ModelSerializer:
    """Save and load model weights + metadata."""

    @staticmethod
    async def save(
        state_dict: dict[str, Any],
        path: Path,
        metadata: ModelMetadata,
        format: SerializationFormat = SerializationFormat.SAFETENSORS,
        config: ModelConfig | None = None,
    ) -> Path:
        """Save model weights, metadata, and optionally config.

        Args:
            state_dict: Model weights (PyTorch state_dict or equivalent).
            path: Target directory. Created if it doesn't exist.
            metadata: Model metadata to persist alongside weights.
            format: Serialization format.
            config: Optional model config to save.

        Returns:
            Path to the saved directory.
        """
        path.mkdir(parents=True, exist_ok=True)

        # Save weights
        if format == SerializationFormat.SAFETENSORS:
            _save_safetensors(state_dict, path / "model.safetensors")
        elif format == SerializationFormat.ONNX:
            _save_onnx(state_dict, path / "model.onnx")
        elif format == SerializationFormat.TORCH:
            _save_torch(state_dict, path / "model.pt")
        elif format == SerializationFormat.TORCHSCRIPT:
            _save_torchscript(state_dict, path / "model.pts")
        elif format == SerializationFormat.PICKLE:
            _save_pickle(state_dict, path / "model.pkl")
        elif format == SerializationFormat.JOBLIB:
            _save_joblib(state_dict, path / "model.joblib")
        else:
            raise FormatNotSupportedError(f"Unsupported format: {format}")

        # Save metadata
        metadata_path = path / "metadata.json"
        metadata_path.write_text(
            metadata.model_dump_json(indent=2),
            encoding="utf-8",
        )

        # Save config if provided
        if config is not None:
            config_path = path / "config.json"
            config_path.write_text(
                config.model_dump_json(indent=2),
                encoding="utf-8",
            )

        return path

    @staticmethod
    async def load(
        path: Path,
        format: SerializationFormat = SerializationFormat.SAFETENSORS,
    ) -> tuple[dict[str, Any], ModelMetadata]:
        """Load model weights and metadata from disk.

        Args:
            path: Directory containing saved model files.
            format: Expected serialization format.

        Returns:
            Tuple of (state_dict, metadata).
        """
        if not path.exists():
            raise SerializationError(f"Model path does not exist: {path}")

        # Load metadata
        metadata_path = path / "metadata.json"
        if not metadata_path.exists():
            raise ArtifactCorruptedError(f"Missing metadata.json at {path}")

        metadata = ModelMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))

        # Load weights
        if format == SerializationFormat.SAFETENSORS:
            state_dict = _load_safetensors(path / "model.safetensors")
        elif format == SerializationFormat.ONNX:
            state_dict = _load_onnx(path / "model.onnx")
        elif format == SerializationFormat.TORCH:
            state_dict = _load_torch(path / "model.pt")
        elif format == SerializationFormat.TORCHSCRIPT:
            state_dict = _load_torchscript(path / "model.pts")
        elif format == SerializationFormat.PICKLE:
            state_dict = _load_pickle(path / "model.pkl")
        elif format == SerializationFormat.JOBLIB:
            state_dict = _load_joblib(path / "model.joblib")
        else:
            raise FormatNotSupportedError(f"Unsupported format: {format}")

        return state_dict, metadata

    @staticmethod
    def list_versions(base_path: Path) -> list[ModelMetadata]:
        """List all saved model versions under a directory.

        Expects structure: base_path/v0.1.0/metadata.json, base_path/v0.2.0/...
        """
        versions: list[ModelMetadata] = []
        if not base_path.exists():
            return versions

        for version_dir in sorted(base_path.iterdir()):
            metadata_path = version_dir / "metadata.json"
            if metadata_path.exists():
                metadata = ModelMetadata.model_validate_json(
                    metadata_path.read_text(encoding="utf-8")
                )
                versions.append(metadata)

        return versions

    @staticmethod
    async def load_config(path: Path) -> ModelConfig | None:
        """Load config.json from a model directory if it exists."""
        config_path = path / "config.json"
        if not config_path.exists():
            return None
        return ModelConfig.model_validate_json(config_path.read_text(encoding="utf-8"))


# =============================================================================
# Format-specific helpers (lazy imports to avoid mandatory heavy dependencies)
# =============================================================================


def _save_safetensors(state_dict: dict[str, Any], path: Path) -> None:
    try:
        from safetensors.torch import save_file

        save_file(state_dict, str(path))
    except ImportError as e:
        raise SerializationError(
            "safetensors not installed. Install with: pip install artenic-ai-sdk[torch]"
        ) from e


def _load_safetensors(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ArtifactCorruptedError(f"Missing model file: {path}")
    try:
        from safetensors.torch import load_file

        return load_file(str(path))  # type: ignore[no-any-return]
    except ImportError as e:
        raise SerializationError(
            "safetensors not installed. Install with: pip install artenic-ai-sdk[torch]"
        ) from e


def _save_onnx(state_dict: dict[str, Any], path: Path) -> None:
    """Export to ONNX format.

    If the input contains a torch.nn.Module, use torch.onnx.export.
    Otherwise, write a marker JSON.
    """
    try:
        import torch

        model = state_dict.get("_module")
        if model is not None and isinstance(model, torch.nn.Module):
            dummy_input = state_dict.get("_dummy_input")
            if dummy_input is None:
                dummy_input = torch.randn(1, 10)
            torch.onnx.export(model, dummy_input, str(path), opset_version=17)
            return
    except ImportError:
        pass

    # Fallback: write marker JSON for state_dict
    data = json.dumps({"format": "onnx", "keys": list(state_dict.keys())})
    path.write_text(data, encoding="utf-8")


def _load_onnx(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ArtifactCorruptedError(f"Missing model file: {path}")

    try:
        content = path.read_text(encoding="utf-8")
        marker = json.loads(content)
        if isinstance(marker, dict) and marker.get("format") == "onnx":
            return marker
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass

    try:
        import onnxruntime

        session = onnxruntime.InferenceSession(str(path))
        return {"format": "onnx", "path": str(path), "session": session}
    except ImportError as e:
        raise SerializationError(
            "onnxruntime not installed. Install with: pip install artenic-ai-sdk[onnx]"
        ) from e


def _save_torch(state_dict: dict[str, Any], path: Path) -> None:
    try:
        import torch

        torch.save(state_dict, path)
    except ImportError as e:
        raise SerializationError(
            "torch not installed. Install with: pip install artenic-ai-sdk[torch]"
        ) from e


def _load_torch(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ArtifactCorruptedError(f"Missing model file: {path}")
    try:
        import torch

        return torch.load(path, weights_only=True)  # type: ignore[no-any-return]
    except ImportError as e:
        raise SerializationError(
            "torch not installed. Install with: pip install artenic-ai-sdk[torch]"
        ) from e


# -- TorchScript --


def _save_torchscript(state_dict: dict[str, Any], path: Path) -> None:
    try:
        import torch
    except ImportError as e:
        raise SerializationError(
            "torch not installed. Install with: pip install artenic-ai-sdk[torch]"
        ) from e

    model = state_dict.get("_module")
    if model is not None and isinstance(model, torch.nn.Module):
        scripted = torch.jit.script(model)
        torch.jit.save(scripted, str(path))
    else:
        # Fallback: save state_dict via torch.save
        torch.save(state_dict, path)


def _load_torchscript(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ArtifactCorruptedError(f"Missing model file: {path}")
    try:
        import torch
    except ImportError as e:
        raise SerializationError(
            "torch not installed. Install with: pip install artenic-ai-sdk[torch]"
        ) from e

    try:
        model = torch.jit.load(str(path))
        return {"_module": model, "format": "torchscript"}
    except RuntimeError:
        return torch.load(path, weights_only=True)  # type: ignore[no-any-return]


# -- Pickle --


def _save_pickle(state_dict: dict[str, Any], path: Path) -> None:
    import pickle

    with path.open("wb") as f:
        pickle.dump(state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ArtifactCorruptedError(f"Missing model file: {path}")
    import pickle

    with path.open("rb") as f:
        return pickle.load(f)  # type: ignore[no-any-return]


# -- Joblib --


def _save_joblib(state_dict: dict[str, Any], path: Path) -> None:
    try:
        import joblib
    except ImportError as e:
        raise SerializationError(
            "joblib not installed. Install with: pip install artenic-ai-sdk[sklearn]"
        ) from e

    joblib.dump(state_dict, path)


def _load_joblib(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ArtifactCorruptedError(f"Missing model file: {path}")
    try:
        import joblib
    except ImportError as e:
        raise SerializationError(
            "joblib not installed. Install with: pip install artenic-ai-sdk[sklearn]"
        ) from e

    return joblib.load(path)  # type: ignore[no-any-return]
