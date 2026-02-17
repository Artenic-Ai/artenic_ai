"""Tests for artenic_ai_sdk.serialization — ModelSerializer (mock-based)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from artenic_ai_sdk.exceptions import (
    ArtifactCorruptedError,
    FormatNotSupportedError,
    SerializationError,
)
from artenic_ai_sdk.schemas import ModelConfig, ModelMetadata
from artenic_ai_sdk.serialization import ModelSerializer
from artenic_ai_sdk.types import ModelFramework, SerializationFormat


@pytest.fixture
def sample_metadata() -> ModelMetadata:
    return ModelMetadata(
        name="test_model",
        version="1.0.0",
        model_type="test",
        framework=ModelFramework.PYTORCH,
    )


class TestModelSerializerMetadata:
    """Test metadata and config save/load (no torch/safetensors needed)."""

    @pytest.mark.asyncio
    async def test_save_metadata(self, tmp_path: Path, sample_metadata: ModelMetadata) -> None:
        # We mock safetensors to avoid the dependency
        with patch("artenic_ai_sdk.serialization._save_safetensors"):
            path = await ModelSerializer.save(
                state_dict={"weight": "data"},
                path=tmp_path / "model_v1",
                metadata=sample_metadata,
            )

        assert (path / "metadata.json").exists()

    @pytest.mark.asyncio
    async def test_save_with_config(self, tmp_path: Path, sample_metadata: ModelMetadata) -> None:
        config = ModelConfig(version="2.0.0")
        with patch("artenic_ai_sdk.serialization._save_safetensors"):
            path = await ModelSerializer.save(
                state_dict={"weight": "data"},
                path=tmp_path / "model_v1",
                metadata=sample_metadata,
                config=config,
            )

        assert (path / "config.json").exists()

    @pytest.mark.asyncio
    async def test_load_config(self, tmp_path: Path, sample_metadata: ModelMetadata) -> None:
        config = ModelConfig(version="3.0.0")
        with patch("artenic_ai_sdk.serialization._save_safetensors"):
            path = await ModelSerializer.save(
                state_dict={},
                path=tmp_path / "model_v1",
                metadata=sample_metadata,
                config=config,
            )

        loaded_config = await ModelSerializer.load_config(path)
        assert loaded_config is not None
        assert loaded_config.version == "3.0.0"

    @pytest.mark.asyncio
    async def test_load_config_missing(self, tmp_path: Path) -> None:
        result = await ModelSerializer.load_config(tmp_path)
        assert result is None


class TestModelSerializerLoad:
    @pytest.mark.asyncio
    async def test_load_nonexistent_path(self) -> None:
        with pytest.raises(SerializationError, match="does not exist"):
            await ModelSerializer.load(Path("/nonexistent/path"))

    @pytest.mark.asyncio
    async def test_load_missing_metadata(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        with pytest.raises(ArtifactCorruptedError, match="Missing metadata"):
            await ModelSerializer.load(model_dir)


class TestModelSerializerFormat:
    def test_invalid_format_raises_value_error(self) -> None:
        """StrEnum rejects unknown values at construction time."""
        with pytest.raises(ValueError, match="is not a valid"):
            SerializationFormat("nonexistent")

    @pytest.mark.asyncio
    async def test_save_unsupported_format(
        self, tmp_path: Path, sample_metadata: ModelMetadata
    ) -> None:
        """Unsupported format on save raises FormatNotSupportedError."""
        with pytest.raises(FormatNotSupportedError, match="Unsupported format"):
            await ModelSerializer.save(
                state_dict={},
                path=tmp_path / "model",
                metadata=sample_metadata,
                format="unknown_format",  # type: ignore[arg-type]
            )

    @pytest.mark.asyncio
    async def test_load_unsupported_format(
        self, tmp_path: Path, sample_metadata: ModelMetadata
    ) -> None:
        """Unsupported format on load raises FormatNotSupportedError."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "metadata.json").write_text(
            sample_metadata.model_dump_json(indent=2), encoding="utf-8"
        )
        with pytest.raises(FormatNotSupportedError, match="Unsupported format"):
            await ModelSerializer.load(model_dir, format="unknown_format")  # type: ignore[arg-type]


class TestModelSerializerListVersions:
    def test_list_versions_empty(self, tmp_path: Path) -> None:
        versions = ModelSerializer.list_versions(tmp_path / "nonexistent")
        assert versions == []

    def test_list_versions(self, tmp_path: Path, sample_metadata: ModelMetadata) -> None:
        v1_dir = tmp_path / "v1.0.0"
        v1_dir.mkdir()
        (v1_dir / "metadata.json").write_text(
            sample_metadata.model_dump_json(indent=2), encoding="utf-8"
        )

        versions = ModelSerializer.list_versions(tmp_path)
        assert len(versions) == 1
        assert versions[0].name == "test_model"


class TestSafetensorsImportError:
    def test_save_import_error(self, tmp_path: Path) -> None:
        from artenic_ai_sdk.serialization import _save_safetensors

        with (
            patch.dict("sys.modules", {"safetensors": None, "safetensors.torch": None}),
            pytest.raises(SerializationError, match="safetensors not installed"),
        ):
            _save_safetensors({}, tmp_path / "model.safetensors")

    def test_load_import_error(self, tmp_path: Path) -> None:
        from artenic_ai_sdk.serialization import _load_safetensors

        model_file = tmp_path / "model.safetensors"
        model_file.write_text("dummy", encoding="utf-8")

        with (
            patch.dict("sys.modules", {"safetensors": None, "safetensors.torch": None}),
            pytest.raises(SerializationError, match="safetensors not installed"),
        ):
            _load_safetensors(model_file)

    def test_load_missing_file(self, tmp_path: Path) -> None:
        from artenic_ai_sdk.serialization import _load_safetensors

        with pytest.raises(ArtifactCorruptedError, match="Missing model file"):
            _load_safetensors(tmp_path / "nonexistent.safetensors")

    def test_save_success(self, tmp_path: Path) -> None:
        """Test _save_safetensors success path with mocked safetensors."""
        from artenic_ai_sdk.serialization import _save_safetensors

        mock_safetensors = MagicMock()
        mock_safetensors_torch = MagicMock()
        with patch.dict(
            "sys.modules",
            {"safetensors": mock_safetensors, "safetensors.torch": mock_safetensors_torch},
        ):
            target = tmp_path / "model.safetensors"
            _save_safetensors({"weight": "data"}, target)
        mock_safetensors_torch.save_file.assert_called_once()

    def test_load_success(self, tmp_path: Path) -> None:
        """Test _load_safetensors success path with mocked safetensors."""
        from artenic_ai_sdk.serialization import _load_safetensors

        model_file = tmp_path / "model.safetensors"
        model_file.write_text("dummy", encoding="utf-8")

        mock_safetensors = MagicMock()
        mock_safetensors_torch = MagicMock()
        mock_safetensors_torch.load_file.return_value = {"weight": "data"}
        with patch.dict(
            "sys.modules",
            {"safetensors": mock_safetensors, "safetensors.torch": mock_safetensors_torch},
        ):
            result = _load_safetensors(model_file)
        assert result == {"weight": "data"}


# =============================================================================
# ONNX format helpers
# =============================================================================


class TestOnnxHelpers:
    def test_save_onnx_marker_json_no_torch(self, tmp_path: Path) -> None:
        """When torch is not available, write marker JSON."""
        from artenic_ai_sdk.serialization import _save_onnx

        target = tmp_path / "model.onnx"
        with patch.dict("sys.modules", {"torch": None}):
            _save_onnx({"key1": "val1"}, target)

        data = json.loads(target.read_text(encoding="utf-8"))
        assert data["format"] == "onnx"
        assert "key1" in data["keys"]

    def test_save_onnx_with_torch_module(self, tmp_path: Path) -> None:
        """When torch is available and _module is an nn.Module, use torch.onnx.export."""
        from artenic_ai_sdk.serialization import _save_onnx

        mock_torch = MagicMock()
        mock_module = MagicMock()
        # isinstance(model, torch.nn.Module) => True
        mock_torch.nn.Module = type(mock_module)

        target = tmp_path / "model.onnx"
        with patch.dict("sys.modules", {"torch": mock_torch}):
            _save_onnx({"_module": mock_module, "_dummy_input": MagicMock()}, target)
        mock_torch.onnx.export.assert_called_once()

    def test_save_onnx_with_torch_no_module(self, tmp_path: Path) -> None:
        """When torch is available but no _module key, write marker JSON."""
        from artenic_ai_sdk.serialization import _save_onnx

        mock_torch = MagicMock()
        target = tmp_path / "model.onnx"
        with patch.dict("sys.modules", {"torch": mock_torch}):
            _save_onnx({"key1": "val1"}, target)

        data = json.loads(target.read_text(encoding="utf-8"))
        assert data["format"] == "onnx"

    def test_save_onnx_module_default_dummy_input(self, tmp_path: Path) -> None:
        """When _dummy_input is None, torch.randn is used."""
        from artenic_ai_sdk.serialization import _save_onnx

        mock_torch = MagicMock()
        mock_module = MagicMock()
        mock_torch.nn.Module = type(mock_module)

        target = tmp_path / "model.onnx"
        with patch.dict("sys.modules", {"torch": mock_torch}):
            _save_onnx({"_module": mock_module}, target)
        mock_torch.randn.assert_called_once_with(1, 10)

    def test_load_onnx_marker_json(self, tmp_path: Path) -> None:
        """Load ONNX marker JSON file."""
        from artenic_ai_sdk.serialization import _load_onnx

        target = tmp_path / "model.onnx"
        target.write_text(json.dumps({"format": "onnx", "keys": ["a"]}), encoding="utf-8")
        result = _load_onnx(target)
        assert result["format"] == "onnx"

    def test_load_onnx_missing_file(self, tmp_path: Path) -> None:
        from artenic_ai_sdk.serialization import _load_onnx

        with pytest.raises(ArtifactCorruptedError, match="Missing model file"):
            _load_onnx(tmp_path / "nonexistent.onnx")

    def test_load_onnx_binary_with_onnxruntime(self, tmp_path: Path) -> None:
        """Load ONNX binary file via onnxruntime."""
        from artenic_ai_sdk.serialization import _load_onnx

        target = tmp_path / "model.onnx"
        target.write_bytes(b"\x00\x01\x02")  # non-JSON binary

        mock_ort = MagicMock()
        mock_session = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session
        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            result = _load_onnx(target)
        assert result["format"] == "onnx"
        assert result["session"] is mock_session

    def test_load_onnx_binary_no_onnxruntime(self, tmp_path: Path) -> None:
        """Load ONNX binary without onnxruntime raises."""
        from artenic_ai_sdk.serialization import _load_onnx

        target = tmp_path / "model.onnx"
        target.write_bytes(b"\x00\x01\x02")

        with (
            patch.dict("sys.modules", {"onnxruntime": None}),
            pytest.raises(SerializationError, match="onnxruntime not installed"),
        ):
            _load_onnx(target)


# =============================================================================
# Torch format helpers
# =============================================================================


class TestTorchHelpers:
    def test_save_torch_success(self, tmp_path: Path) -> None:
        from artenic_ai_sdk.serialization import _save_torch

        mock_torch = MagicMock()
        target = tmp_path / "model.pt"
        with patch.dict("sys.modules", {"torch": mock_torch}):
            _save_torch({"weight": "data"}, target)
        mock_torch.save.assert_called_once()

    def test_save_torch_no_torch(self, tmp_path: Path) -> None:
        from artenic_ai_sdk.serialization import _save_torch

        with (
            patch.dict("sys.modules", {"torch": None}),
            pytest.raises(SerializationError, match="torch not installed"),
        ):
            _save_torch({}, tmp_path / "model.pt")

    def test_load_torch_success(self, tmp_path: Path) -> None:
        from artenic_ai_sdk.serialization import _load_torch

        target = tmp_path / "model.pt"
        target.write_text("dummy", encoding="utf-8")

        mock_torch = MagicMock()
        mock_torch.load.return_value = {"weight": "data"}
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = _load_torch(target)
        assert result == {"weight": "data"}

    def test_load_torch_no_torch(self, tmp_path: Path) -> None:
        from artenic_ai_sdk.serialization import _load_torch

        target = tmp_path / "model.pt"
        target.write_text("dummy", encoding="utf-8")

        with (
            patch.dict("sys.modules", {"torch": None}),
            pytest.raises(SerializationError, match="torch not installed"),
        ):
            _load_torch(target)

    def test_load_torch_missing_file(self, tmp_path: Path) -> None:
        from artenic_ai_sdk.serialization import _load_torch

        with pytest.raises(ArtifactCorruptedError, match="Missing model file"):
            _load_torch(tmp_path / "nonexistent.pt")


# =============================================================================
# ModelSerializer — format dispatch
# =============================================================================


class TestModelSerializerFormats:
    @pytest.mark.asyncio
    async def test_save_onnx_format(self, tmp_path: Path, sample_metadata: ModelMetadata) -> None:
        with patch("artenic_ai_sdk.serialization._save_onnx"):
            path = await ModelSerializer.save(
                state_dict={"weight": "data"},
                path=tmp_path / "model_onnx",
                metadata=sample_metadata,
                format=SerializationFormat.ONNX,
            )
        assert (path / "metadata.json").exists()

    @pytest.mark.asyncio
    async def test_save_torch_format(self, tmp_path: Path, sample_metadata: ModelMetadata) -> None:
        with patch("artenic_ai_sdk.serialization._save_torch"):
            path = await ModelSerializer.save(
                state_dict={"weight": "data"},
                path=tmp_path / "model_pt",
                metadata=sample_metadata,
                format=SerializationFormat.TORCH,
            )
        assert (path / "metadata.json").exists()

    @pytest.mark.asyncio
    async def test_load_safetensors_format(
        self, tmp_path: Path, sample_metadata: ModelMetadata
    ) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "metadata.json").write_text(
            sample_metadata.model_dump_json(indent=2), encoding="utf-8"
        )
        with patch(
            "artenic_ai_sdk.serialization._load_safetensors",
            return_value={"w": 1.0},
        ):
            state_dict, meta = await ModelSerializer.load(model_dir)
        assert state_dict == {"w": 1.0}
        assert meta.name == "test_model"

    @pytest.mark.asyncio
    async def test_load_onnx_format(self, tmp_path: Path, sample_metadata: ModelMetadata) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "metadata.json").write_text(
            sample_metadata.model_dump_json(indent=2), encoding="utf-8"
        )
        with patch(
            "artenic_ai_sdk.serialization._load_onnx",
            return_value={"format": "onnx"},
        ):
            state_dict, _meta = await ModelSerializer.load(
                model_dir, format=SerializationFormat.ONNX
            )
        assert state_dict["format"] == "onnx"

    @pytest.mark.asyncio
    async def test_load_torch_format(self, tmp_path: Path, sample_metadata: ModelMetadata) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "metadata.json").write_text(
            sample_metadata.model_dump_json(indent=2), encoding="utf-8"
        )
        with patch(
            "artenic_ai_sdk.serialization._load_torch",
            return_value={"w": 1.0},
        ):
            state_dict, _meta = await ModelSerializer.load(
                model_dir, format=SerializationFormat.TORCH
            )
        assert state_dict == {"w": 1.0}

    @pytest.mark.asyncio
    async def test_save_torchscript_format(
        self, tmp_path: Path, sample_metadata: ModelMetadata
    ) -> None:
        with patch("artenic_ai_sdk.serialization._save_torchscript"):
            path = await ModelSerializer.save(
                state_dict={"weight": "data"},
                path=tmp_path / "model_pts",
                metadata=sample_metadata,
                format=SerializationFormat.TORCHSCRIPT,
            )
        assert (path / "metadata.json").exists()

    @pytest.mark.asyncio
    async def test_load_torchscript_format(
        self, tmp_path: Path, sample_metadata: ModelMetadata
    ) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "metadata.json").write_text(
            sample_metadata.model_dump_json(indent=2), encoding="utf-8"
        )
        with patch(
            "artenic_ai_sdk.serialization._load_torchscript",
            return_value={"format": "torchscript"},
        ):
            state_dict, _meta = await ModelSerializer.load(
                model_dir, format=SerializationFormat.TORCHSCRIPT
            )
        assert state_dict["format"] == "torchscript"

    @pytest.mark.asyncio
    async def test_save_pickle_format(self, tmp_path: Path, sample_metadata: ModelMetadata) -> None:
        with patch("artenic_ai_sdk.serialization._save_pickle"):
            path = await ModelSerializer.save(
                state_dict={"weight": "data"},
                path=tmp_path / "model_pkl",
                metadata=sample_metadata,
                format=SerializationFormat.PICKLE,
            )
        assert (path / "metadata.json").exists()

    @pytest.mark.asyncio
    async def test_load_pickle_format(self, tmp_path: Path, sample_metadata: ModelMetadata) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "metadata.json").write_text(
            sample_metadata.model_dump_json(indent=2), encoding="utf-8"
        )
        with patch(
            "artenic_ai_sdk.serialization._load_pickle",
            return_value={"w": 1.0},
        ):
            state_dict, _meta = await ModelSerializer.load(
                model_dir, format=SerializationFormat.PICKLE
            )
        assert state_dict == {"w": 1.0}

    @pytest.mark.asyncio
    async def test_save_joblib_format(self, tmp_path: Path, sample_metadata: ModelMetadata) -> None:
        with patch("artenic_ai_sdk.serialization._save_joblib"):
            path = await ModelSerializer.save(
                state_dict={"weight": "data"},
                path=tmp_path / "model_jl",
                metadata=sample_metadata,
                format=SerializationFormat.JOBLIB,
            )
        assert (path / "metadata.json").exists()

    @pytest.mark.asyncio
    async def test_load_joblib_format(self, tmp_path: Path, sample_metadata: ModelMetadata) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "metadata.json").write_text(
            sample_metadata.model_dump_json(indent=2), encoding="utf-8"
        )
        with patch(
            "artenic_ai_sdk.serialization._load_joblib",
            return_value={"w": 1.0},
        ):
            state_dict, _meta = await ModelSerializer.load(
                model_dir, format=SerializationFormat.JOBLIB
            )
        assert state_dict == {"w": 1.0}


# =============================================================================
# TorchScript format helpers
# =============================================================================


class TestTorchScriptHelpers:
    def test_save_torchscript_with_module(self, tmp_path: Path) -> None:
        from artenic_ai_sdk.serialization import _save_torchscript

        mock_torch = MagicMock()
        mock_module = MagicMock()
        mock_torch.nn.Module = type(mock_module)
        mock_scripted = MagicMock()
        mock_torch.jit.script.return_value = mock_scripted

        target = tmp_path / "model.pts"
        with patch.dict("sys.modules", {"torch": mock_torch}):
            _save_torchscript({"_module": mock_module}, target)
        mock_torch.jit.script.assert_called_once_with(mock_module)
        mock_torch.jit.save.assert_called_once()

    def test_save_torchscript_state_dict_fallback(self, tmp_path: Path) -> None:
        from artenic_ai_sdk.serialization import _save_torchscript

        mock_torch = MagicMock()
        target = tmp_path / "model.pts"
        with patch.dict("sys.modules", {"torch": mock_torch}):
            _save_torchscript({"weight": "data"}, target)
        mock_torch.save.assert_called_once()

    def test_save_torchscript_no_torch(self, tmp_path: Path) -> None:
        from artenic_ai_sdk.serialization import _save_torchscript

        with (
            patch.dict("sys.modules", {"torch": None}),
            pytest.raises(SerializationError, match="torch not installed"),
        ):
            _save_torchscript({}, tmp_path / "model.pts")

    def test_load_torchscript_jit_success(self, tmp_path: Path) -> None:
        from artenic_ai_sdk.serialization import _load_torchscript

        target = tmp_path / "model.pts"
        target.write_text("dummy", encoding="utf-8")

        mock_torch = MagicMock()
        mock_model = MagicMock()
        mock_torch.jit.load.return_value = mock_model
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = _load_torchscript(target)
        assert result["format"] == "torchscript"
        assert result["_module"] is mock_model

    def test_load_torchscript_jit_fallback_to_torch_load(self, tmp_path: Path) -> None:
        from artenic_ai_sdk.serialization import _load_torchscript

        target = tmp_path / "model.pts"
        target.write_text("dummy", encoding="utf-8")

        mock_torch = MagicMock()
        mock_torch.jit.load.side_effect = RuntimeError("Not a TorchScript model")
        mock_torch.load.return_value = {"weight": "data"}
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = _load_torchscript(target)
        assert result == {"weight": "data"}

    def test_load_torchscript_no_torch(self, tmp_path: Path) -> None:
        from artenic_ai_sdk.serialization import _load_torchscript

        target = tmp_path / "model.pts"
        target.write_text("dummy", encoding="utf-8")

        with (
            patch.dict("sys.modules", {"torch": None}),
            pytest.raises(SerializationError, match="torch not installed"),
        ):
            _load_torchscript(target)

    def test_load_torchscript_missing_file(self, tmp_path: Path) -> None:
        from artenic_ai_sdk.serialization import _load_torchscript

        with pytest.raises(ArtifactCorruptedError, match="Missing model file"):
            _load_torchscript(tmp_path / "nonexistent.pts")


# =============================================================================
# Pickle format helpers
# =============================================================================


class TestPickleHelpers:
    def test_save_pickle(self, tmp_path: Path) -> None:
        from artenic_ai_sdk.serialization import _save_pickle

        target = tmp_path / "model.pkl"
        _save_pickle({"weight": 42, "bias": 3.14}, target)
        assert target.exists()

    def test_load_pickle(self, tmp_path: Path) -> None:
        from artenic_ai_sdk.serialization import _load_pickle, _save_pickle

        target = tmp_path / "model.pkl"
        data = {"weight": 42, "bias": 3.14}
        _save_pickle(data, target)
        result = _load_pickle(target)
        assert result == data

    def test_load_pickle_missing_file(self, tmp_path: Path) -> None:
        from artenic_ai_sdk.serialization import _load_pickle

        with pytest.raises(ArtifactCorruptedError, match="Missing model file"):
            _load_pickle(tmp_path / "nonexistent.pkl")


# =============================================================================
# Joblib format helpers
# =============================================================================


class TestJoblibHelpers:
    def test_save_joblib_success(self, tmp_path: Path) -> None:
        from artenic_ai_sdk.serialization import _save_joblib

        mock_joblib = MagicMock()
        target = tmp_path / "model.joblib"
        with patch.dict("sys.modules", {"joblib": mock_joblib}):
            _save_joblib({"weight": "data"}, target)
        mock_joblib.dump.assert_called_once()

    def test_save_joblib_no_joblib(self, tmp_path: Path) -> None:
        from artenic_ai_sdk.serialization import _save_joblib

        with (
            patch.dict("sys.modules", {"joblib": None}),
            pytest.raises(SerializationError, match="joblib not installed"),
        ):
            _save_joblib({}, tmp_path / "model.joblib")

    def test_load_joblib_success(self, tmp_path: Path) -> None:
        from artenic_ai_sdk.serialization import _load_joblib

        target = tmp_path / "model.joblib"
        target.write_text("dummy", encoding="utf-8")

        mock_joblib = MagicMock()
        mock_joblib.load.return_value = {"weight": "data"}
        with patch.dict("sys.modules", {"joblib": mock_joblib}):
            result = _load_joblib(target)
        assert result == {"weight": "data"}

    def test_load_joblib_no_joblib(self, tmp_path: Path) -> None:
        from artenic_ai_sdk.serialization import _load_joblib

        target = tmp_path / "model.joblib"
        target.write_text("dummy", encoding="utf-8")

        with (
            patch.dict("sys.modules", {"joblib": None}),
            pytest.raises(SerializationError, match="joblib not installed"),
        ):
            _load_joblib(target)

    def test_load_joblib_missing_file(self, tmp_path: Path) -> None:
        from artenic_ai_sdk.serialization import _load_joblib

        with pytest.raises(ArtifactCorruptedError, match="Missing model file"):
            _load_joblib(tmp_path / "nonexistent.joblib")
