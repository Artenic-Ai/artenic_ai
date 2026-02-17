"""Auto-evolving configuration system.

Configs have a lifecycle: Default → Active → Candidate → Promoted → Retired.
The ConfigManager loads YAML, validates with Pydantic, and proposes
evolutions. The ConfigRegistry stores config-performance pairs to learn
which configs work best.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import yaml

from artenic_ai_sdk.exceptions import ConfigNotFoundError, ConfigValidationError
from artenic_ai_sdk.schemas import ConfigDiff, ConfigEntry, EvalResult, ModelConfig
from artenic_ai_sdk.types import ConfigPhase


class ConfigManager:
    """Load, save, and evolve model configurations."""

    @staticmethod
    async def load(path: Path, config_class: type[ModelConfig] = ModelConfig) -> ModelConfig:
        """Load a YAML config and validate with Pydantic.

        Args:
            path: Path to the YAML config file.
            config_class: The Pydantic model class to validate against.

        Returns:
            Validated ModelConfig instance.
        """
        if not path.exists():
            raise ConfigNotFoundError(f"Config not found: {path}")

        raw = path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)

        if data is None:
            raise ConfigValidationError(f"Empty config file: {path}")

        # Flatten nested 'hyperparams' key if present
        if "hyperparams" in data:
            hyperparams = data.pop("hyperparams")
            data.update(hyperparams)

        # Extract model metadata if wrapped
        if "model" in data and isinstance(data["model"], dict):
            data.pop("model")

        try:
            return config_class.model_validate(data)
        except Exception as e:
            raise ConfigValidationError(f"Invalid config at {path}: {e}") from e

    @staticmethod
    async def save(config: ModelConfig, path: Path) -> Path:
        """Save config as YAML with metadata."""
        path.parent.mkdir(parents=True, exist_ok=True)

        data = config.model_dump(mode="json")
        raw = yaml.dump(data, default_flow_style=False, sort_keys=False)
        path.write_text(raw, encoding="utf-8")
        return path

    @staticmethod
    async def load_with_env(
        path: Path,
        prefix: str = "ARTENIC",
        config_class: type[ModelConfig] = ModelConfig,
    ) -> ModelConfig:
        """Load a YAML config, then override fields with environment variables.

        Environment variable mapping: ``{PREFIX}_{FIELD_UPPER}`` overrides
        the corresponding snake_case field. For example,
        ``ARTENIC_LEARNING_RATE=0.001`` overrides ``learning_rate``.

        Args:
            path: Path to the YAML config file.
            prefix: Environment variable prefix.
            config_class: The Pydantic model class to validate against.

        Returns:
            Validated ModelConfig with env overrides applied.
        """
        config = await ConfigManager.load(path, config_class=config_class)
        data = config.model_dump(mode="json")

        for field_name in config_class.model_fields:
            env_key = f"{prefix}_{field_name.upper()}"
            env_val = os.environ.get(env_key)
            if env_val is not None:
                # Coerce to the same type as the current value
                # bool must be checked before int (bool is subclass of int)
                current = data.get(field_name)
                if isinstance(current, bool):
                    data[field_name] = env_val.lower() in ("true", "1", "yes")
                elif isinstance(current, float):
                    data[field_name] = float(env_val)
                elif isinstance(current, int):
                    data[field_name] = int(env_val)
                else:
                    data[field_name] = env_val

        try:
            return config_class.model_validate(data)
        except Exception as e:
            raise ConfigValidationError(f"Invalid config after env override: {e}") from e

    @staticmethod
    def diff(config_a: ModelConfig, config_b: ModelConfig) -> ConfigDiff:
        """Compute the diff between two ModelConfig instances.

        Args:
            config_a: The baseline config.
            config_b: The config to compare against.

        Returns:
            ConfigDiff with added, removed, and changed fields.
        """
        data_a = config_a.model_dump(mode="json")
        data_b = config_b.model_dump(mode="json")

        keys_a = set(data_a.keys())
        keys_b = set(data_b.keys())

        added = {k: data_b[k] for k in keys_b - keys_a}
        removed = {k: data_a[k] for k in keys_a - keys_b}
        changed: dict[str, tuple[Any, Any]] = {}
        for k in keys_a & keys_b:
            if data_a[k] != data_b[k]:
                changed[k] = (data_a[k], data_b[k])

        return ConfigDiff(added=added, removed=removed, changed=changed)

    @staticmethod
    async def propose_evolution(
        current: ModelConfig,
        performance_history: list[EvalResult],
    ) -> ModelConfig | None:
        """Propose config adjustments based on performance history.

        Analyzes recent performance and suggests changes if performance
        is degrading.

        Returns:
            A new ModelConfig candidate, or None if current config is optimal.
        """
        if not performance_history:
            return None

        # Analyze performance trend
        recent = performance_history[-5:]
        if len(recent) < 3:
            return None

        primary_metric = next(iter(recent[0].metrics), None)
        if primary_metric is None:
            return None

        values = [r.metrics.get(primary_metric, 0.0) for r in recent]

        # Detect degrading performance (3 consecutive drops)
        is_degrading = all(values[i] > values[i + 1] for i in range(len(values) - 1))
        if not is_degrading:
            return None

        # Performance is degrading — return a candidate for evolution
        # This is the hook for more sophisticated evolution logic
        return current.model_copy()


class ConfigRegistry:
    """In-memory registry of config-performance pairs.

    Stores the relationship between configs and their evaluation results.
    Thread-safe: mutations are guarded by an asyncio.Lock.
    """

    def __init__(self) -> None:
        self._entries: list[ConfigEntry] = []
        self._ab_tests: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        model_id: str,
        config: ModelConfig,
        eval_result: EvalResult,
    ) -> ConfigEntry:
        """Record a config-performance pair."""
        entry = ConfigEntry(
            config=config,
            model_id=model_id,
            phase=ConfigPhase.ACTIVE,
            eval_result=eval_result,
        )
        async with self._lock:
            self._entries.append(entry)
        return entry

    async def get_best(
        self,
        model_id: str,
        metric: str = "accuracy",
    ) -> ModelConfig | None:
        """Find the best-performing config for a model.

        Args:
            model_id: Filter entries for this model.
            metric: The metric to maximize.

        Returns:
            The config that produced the best result, or None.
        """
        async with self._lock:
            candidates = [
                e for e in self._entries if e.model_id == model_id and e.eval_result is not None
            ]

        if not candidates:
            return None

        best = max(
            candidates,
            key=lambda e: e.eval_result.metrics.get(metric, 0.0),  # type: ignore[union-attr]
        )
        return best.config

    async def get_history(self, model_id: str) -> list[ConfigEntry]:
        """Get full config history for a model."""
        async with self._lock:
            return [e for e in self._entries if e.model_id == model_id]

    async def promote(self, entry: ConfigEntry) -> None:
        """Mark a config as promoted (validated by performance)."""
        async with self._lock:
            entry.phase = ConfigPhase.PROMOTED

    async def retire(self, entry: ConfigEntry) -> None:
        """Mark a config as retired (no longer used)."""
        async with self._lock:
            entry.phase = ConfigPhase.RETIRED

    def export_json(self, path: Path) -> None:
        """Export registry to JSON for persistence."""
        data = [e.model_dump(mode="json") for e in self._entries]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def import_json(self, path: Path) -> None:
        """Import registry from JSON."""
        if not path.exists():
            return
        raw = json.loads(path.read_text(encoding="utf-8"))
        self._entries = [ConfigEntry.model_validate(e) for e in raw]

    async def rollback(
        self,
        model_id: str,
        to_version: str | None = None,
    ) -> ConfigEntry | None:
        """Rollback a model's config to a previous version.

        Retires the current ACTIVE entry and re-activates the target.

        Args:
            model_id: The model to rollback.
            to_version: Specific config version to rollback to. If None,
                uses the most recent PROMOTED entry.

        Returns:
            The re-activated ConfigEntry, or None if no rollback target found.
        """
        async with self._lock:
            active = [
                e for e in self._entries if e.model_id == model_id and e.phase == ConfigPhase.ACTIVE
            ]

            if to_version:
                candidates = [
                    e
                    for e in self._entries
                    if e.model_id == model_id
                    and e.config.version == to_version
                    and e.phase in (ConfigPhase.PROMOTED, ConfigPhase.RETIRED)
                ]
            else:
                candidates = [
                    e
                    for e in self._entries
                    if e.model_id == model_id and e.phase == ConfigPhase.PROMOTED
                ]

            if not candidates:
                return None

            target = candidates[-1]

            for entry in active:
                entry.phase = ConfigPhase.RETIRED

            target.phase = ConfigPhase.ACTIVE
            return target

    def create_ab_test(
        self,
        model_id: str,
        variant_a: ModelConfig,
        variant_b: ModelConfig,
        traffic_split: float = 0.5,
    ) -> dict[str, Any]:
        """Create an A/B test between two configs.

        Args:
            model_id: The model to test.
            variant_a: Control config.
            variant_b: Test config.
            traffic_split: Fraction of traffic going to variant_a (0.0-1.0).

        Returns:
            A/B test descriptor dict.
        """
        test: dict[str, Any] = {
            "model_id": model_id,
            "variant_a": variant_a,
            "variant_b": variant_b,
            "traffic_split": traffic_split,
        }
        self._ab_tests[model_id] = test
        return test

    def get_ab_config(
        self,
        model_id: str,
        random_value: float,
    ) -> ModelConfig | None:
        """Get the config for this request based on A/B traffic split.

        Args:
            model_id: The model to query.
            random_value: A random float in [0, 1) used for traffic routing.

        Returns:
            The selected config, or None if no A/B test is active.
        """
        test = self._ab_tests.get(model_id)
        if test is None:
            return None

        if random_value < test["traffic_split"]:
            return test["variant_a"]  # type: ignore[no-any-return]
        return test["variant_b"]  # type: ignore[no-any-return]

    async def conclude_ab_test(
        self,
        model_id: str,
        winner: str,
    ) -> ConfigEntry | None:
        """Conclude an A/B test and promote the winner.

        Args:
            model_id: The model being tested.
            winner: Either "a" or "b".

        Returns:
            The promoted ConfigEntry, or None if no A/B test found.
        """
        test = self._ab_tests.pop(model_id, None)
        if test is None:
            return None

        winning_config = test["variant_a"] if winner == "a" else test["variant_b"]
        entry = ConfigEntry(
            config=winning_config,
            model_id=model_id,
            phase=ConfigPhase.PROMOTED,
        )
        async with self._lock:
            self._entries.append(entry)
        return entry
