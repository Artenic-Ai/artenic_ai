"""Tests for artenic_ai_sdk.config — ConfigManager + ConfigRegistry."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from artenic_ai_sdk.config import ConfigManager, ConfigRegistry
from artenic_ai_sdk.exceptions import ConfigNotFoundError, ConfigValidationError
from artenic_ai_sdk.schemas import ConfigDiff, EvalResult, ModelConfig
from artenic_ai_sdk.types import ConfigPhase

if TYPE_CHECKING:
    from pathlib import Path


# =============================================================================
# ConfigManager
# =============================================================================


class TestConfigManager:
    @pytest.mark.asyncio
    async def test_load_yaml(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text("version: '2.0.0'\n", encoding="utf-8")

        config = await ConfigManager.load(config_path)
        assert config.version == "2.0.0"

    @pytest.mark.asyncio
    async def test_load_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigNotFoundError):
            await ConfigManager.load(tmp_path / "nonexistent.yaml")

    @pytest.mark.asyncio
    async def test_load_empty(self, tmp_path: Path) -> None:
        config_path = tmp_path / "empty.yaml"
        config_path.write_text("", encoding="utf-8")
        with pytest.raises(ConfigValidationError, match="Empty config"):
            await ConfigManager.load(config_path)

    @pytest.mark.asyncio
    async def test_load_with_hyperparams(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text("version: '1.0'\nhyperparams:\n  some_key: 42\n", encoding="utf-8")
        config = await ConfigManager.load(config_path)
        assert config.version == "1.0"

    @pytest.mark.asyncio
    async def test_save_and_reload(self, tmp_path: Path) -> None:
        config = ModelConfig(version="3.0.0")
        path = await ConfigManager.save(config, tmp_path / "out.yaml")
        loaded = await ConfigManager.load(path)
        assert loaded.version == "3.0.0"

    @pytest.mark.asyncio
    async def test_propose_evolution_no_history(self) -> None:
        config = ModelConfig()
        result = await ConfigManager.propose_evolution(config, [])
        assert result is None

    @pytest.mark.asyncio
    async def test_propose_evolution_too_few_results(self) -> None:
        config = ModelConfig()
        history = [
            EvalResult(model_name="m", model_version="1.0", metrics={"acc": 0.9}),
            EvalResult(model_name="m", model_version="1.0", metrics={"acc": 0.8}),
        ]
        result = await ConfigManager.propose_evolution(config, history)
        assert result is None

    @pytest.mark.asyncio
    async def test_propose_evolution_degrading(self) -> None:
        config = ModelConfig()
        history = [
            EvalResult(model_name="m", model_version="1.0", metrics={"acc": 0.9}),
            EvalResult(model_name="m", model_version="1.0", metrics={"acc": 0.8}),
            EvalResult(model_name="m", model_version="1.0", metrics={"acc": 0.7}),
        ]
        result = await ConfigManager.propose_evolution(config, history)
        assert result is not None

    @pytest.mark.asyncio
    async def test_propose_evolution_stable(self) -> None:
        config = ModelConfig()
        history = [
            EvalResult(model_name="m", model_version="1.0", metrics={"acc": 0.8}),
            EvalResult(model_name="m", model_version="1.0", metrics={"acc": 0.9}),
            EvalResult(model_name="m", model_version="1.0", metrics={"acc": 0.85}),
        ]
        result = await ConfigManager.propose_evolution(config, history)
        assert result is None

    @pytest.mark.asyncio
    async def test_load_with_model_key(self, tmp_path: Path) -> None:
        """Load config with 'model' wrapper key."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("version: '1.0'\nmodel:\n  name: test\n", encoding="utf-8")
        config = await ConfigManager.load(config_path)
        assert config.version == "1.0"

    @pytest.mark.asyncio
    async def test_load_invalid_yaml(self, tmp_path: Path) -> None:
        """Load config with invalid YAML data raises ConfigValidationError."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("version: [invalid_list_value]\n", encoding="utf-8")
        with pytest.raises(ConfigValidationError, match="Invalid config"):
            await ConfigManager.load(config_path)

    @pytest.mark.asyncio
    async def test_propose_evolution_no_primary_metric(self) -> None:
        """propose_evolution returns None when metrics are empty."""
        config = ModelConfig()
        history = [
            EvalResult(model_name="m", model_version="1.0", metrics={}),
            EvalResult(model_name="m", model_version="1.0", metrics={}),
            EvalResult(model_name="m", model_version="1.0", metrics={}),
        ]
        result = await ConfigManager.propose_evolution(config, history)
        assert result is None


class TestConfigManagerLoadWithEnv:
    @pytest.mark.asyncio
    async def test_env_override_string(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text("version: '1.0'\n", encoding="utf-8")
        with patch.dict("os.environ", {"ARTENIC_VERSION": "9.9.9"}):
            config = await ConfigManager.load_with_env(config_path)
        assert config.version == "9.9.9"

    @pytest.mark.asyncio
    async def test_env_override_custom_prefix(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text("version: '1.0'\n", encoding="utf-8")
        with patch.dict("os.environ", {"MYAPP_VERSION": "5.0"}):
            config = await ConfigManager.load_with_env(config_path, prefix="MYAPP")
        assert config.version == "5.0"

    @pytest.mark.asyncio
    async def test_no_env_uses_yaml(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text("version: '2.0'\n", encoding="utf-8")
        config = await ConfigManager.load_with_env(config_path)
        assert config.version == "2.0"

    @pytest.mark.asyncio
    async def test_env_override_validation_error(self, tmp_path: Path) -> None:
        """Env override that passes coercion but fails validation raises ConfigValidationError."""
        from pydantic import Field

        class BoundedConfig(ModelConfig):
            value: int = Field(default=10, ge=0)

        config_path = tmp_path / "config.yaml"
        config_path.write_text("version: '1.0'\nvalue: 10\n", encoding="utf-8")
        with (
            patch.dict("os.environ", {"ARTENIC_VALUE": "-5"}),
            pytest.raises(ConfigValidationError, match="Invalid config after env override"),
        ):
            await ConfigManager.load_with_env(config_path, config_class=BoundedConfig)

    @pytest.mark.asyncio
    async def test_env_override_float(self, tmp_path: Path) -> None:
        """Test float coercion with a custom config class."""

        class LRConfig(ModelConfig):
            learning_rate: float = 0.01

        config_path = tmp_path / "config.yaml"
        config_path.write_text("version: '1.0'\nlearning_rate: 0.01\n", encoding="utf-8")
        with patch.dict("os.environ", {"ARTENIC_LEARNING_RATE": "0.001"}):
            config = await ConfigManager.load_with_env(config_path, config_class=LRConfig)
        assert config.learning_rate == 0.001  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_env_override_int(self, tmp_path: Path) -> None:
        """Test int coercion with a custom config class."""

        class EpochConfig(ModelConfig):
            epochs: int = 10

        config_path = tmp_path / "config.yaml"
        config_path.write_text("version: '1.0'\nepochs: 10\n", encoding="utf-8")
        with patch.dict("os.environ", {"ARTENIC_EPOCHS": "50"}):
            config = await ConfigManager.load_with_env(config_path, config_class=EpochConfig)
        assert config.epochs == 50  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_env_override_bool(self, tmp_path: Path) -> None:
        """Test bool coercion with a custom config class."""

        class FlagConfig(ModelConfig):
            debug: bool = False

        config_path = tmp_path / "config.yaml"
        config_path.write_text("version: '1.0'\ndebug: false\n", encoding="utf-8")
        with patch.dict("os.environ", {"ARTENIC_DEBUG": "true"}):
            config = await ConfigManager.load_with_env(config_path, config_class=FlagConfig)
        assert config.debug is True  # type: ignore[attr-defined]


class TestConfigManagerDiff:
    def test_identical_configs(self) -> None:
        config_a = ModelConfig(version="1.0")
        config_b = ModelConfig(version="1.0")
        diff = ConfigManager.diff(config_a, config_b)
        assert isinstance(diff, ConfigDiff)
        assert diff.added == {}
        assert diff.removed == {}
        assert diff.changed == {}

    def test_changed_field(self) -> None:
        config_a = ModelConfig(version="1.0")
        config_b = ModelConfig(version="2.0")
        diff = ConfigManager.diff(config_a, config_b)
        assert "version" in diff.changed
        assert diff.changed["version"] == ("1.0", "2.0")

    def test_added_and_removed_fields(self) -> None:
        """Custom config classes with different fields."""

        class ConfigA(ModelConfig):
            old_field: str = "old"

        class ConfigB(ModelConfig):
            new_field: str = "new"

        config_a = ConfigA(version="1.0")
        config_b = ConfigB(version="1.0")
        diff = ConfigManager.diff(config_a, config_b)
        assert "old_field" in diff.removed
        assert "new_field" in diff.added


# =============================================================================
# ConfigRegistry
# =============================================================================


class TestConfigRegistry:
    @pytest.mark.asyncio
    async def test_register_and_get_best(self) -> None:
        registry = ConfigRegistry()
        config_a = ModelConfig(version="1.0")
        config_b = ModelConfig(version="2.0")
        eval_a = EvalResult(model_name="m", model_version="1.0", metrics={"acc": 0.8})
        eval_b = EvalResult(model_name="m", model_version="2.0", metrics={"acc": 0.95})

        await registry.register("model_x", config_a, eval_a)
        await registry.register("model_x", config_b, eval_b)

        best = await registry.get_best("model_x", metric="acc")
        assert best is not None
        assert best.version == "2.0"

    @pytest.mark.asyncio
    async def test_get_best_no_entries(self) -> None:
        registry = ConfigRegistry()
        result = await registry.get_best("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_history(self) -> None:
        registry = ConfigRegistry()
        eval_result = EvalResult(model_name="m", model_version="1.0", metrics={"acc": 0.9})
        await registry.register("model_x", ModelConfig(), eval_result)
        history = await registry.get_history("model_x")
        assert len(history) == 1

    @pytest.mark.asyncio
    async def test_promote_and_retire(self) -> None:
        registry = ConfigRegistry()
        eval_result = EvalResult(model_name="m", model_version="1.0", metrics={"acc": 0.9})
        entry = await registry.register("model_x", ModelConfig(), eval_result)

        await registry.promote(entry)
        assert entry.phase == ConfigPhase.PROMOTED

        await registry.retire(entry)
        assert entry.phase == ConfigPhase.RETIRED

    @pytest.mark.asyncio
    async def test_rollback(self) -> None:
        registry = ConfigRegistry()
        eval_a = EvalResult(model_name="m", model_version="1.0", metrics={"acc": 0.9})
        eval_b = EvalResult(model_name="m", model_version="1.0", metrics={"acc": 0.8})

        entry_a = await registry.register("m", ModelConfig(version="1.0"), eval_a)
        await registry.promote(entry_a)

        entry_b = await registry.register("m", ModelConfig(version="2.0"), eval_b)
        assert entry_b.phase == ConfigPhase.ACTIVE

        # Rollback to v1.0
        rolled = await registry.rollback("m", to_version="1.0")
        assert rolled is not None
        assert rolled.config.version == "1.0"
        assert rolled.phase == ConfigPhase.ACTIVE

    @pytest.mark.asyncio
    async def test_rollback_no_target(self) -> None:
        registry = ConfigRegistry()
        result = await registry.rollback("nonexistent")
        assert result is None

    def test_ab_test(self) -> None:
        registry = ConfigRegistry()
        config_a = ModelConfig(version="1.0")
        config_b = ModelConfig(version="2.0")

        registry.create_ab_test("m", config_a, config_b, traffic_split=0.5)

        # Traffic below split → variant A
        result_a = registry.get_ab_config("m", 0.3)
        assert result_a is not None
        assert result_a.version == "1.0"

        # Traffic above split → variant B
        result_b = registry.get_ab_config("m", 0.7)
        assert result_b is not None
        assert result_b.version == "2.0"

    def test_ab_config_no_test(self) -> None:
        registry = ConfigRegistry()
        assert registry.get_ab_config("m", 0.5) is None

    @pytest.mark.asyncio
    async def test_conclude_ab_test(self) -> None:
        registry = ConfigRegistry()
        config_a = ModelConfig(version="1.0")
        config_b = ModelConfig(version="2.0")
        registry.create_ab_test("m", config_a, config_b)

        entry = await registry.conclude_ab_test("m", winner="b")
        assert entry is not None
        assert entry.config.version == "2.0"
        assert entry.phase == ConfigPhase.PROMOTED

    @pytest.mark.asyncio
    async def test_conclude_ab_test_no_test(self) -> None:
        registry = ConfigRegistry()
        result = await registry.conclude_ab_test("m", winner="a")
        assert result is None

    def test_export_import_json(self, tmp_path: Path) -> None:
        registry = ConfigRegistry()
        registry._entries.append(
            __import__("artenic_ai_sdk.schemas", fromlist=["ConfigEntry"]).ConfigEntry(
                config=ModelConfig(), model_id="test"
            )
        )

        path = tmp_path / "registry.json"
        registry.export_json(path)

        new_registry = ConfigRegistry()
        new_registry.import_json(path)
        assert len(new_registry._entries) == 1

    def test_import_json_nonexistent(self, tmp_path: Path) -> None:
        registry = ConfigRegistry()
        registry.import_json(tmp_path / "nope.json")
        assert len(registry._entries) == 0
