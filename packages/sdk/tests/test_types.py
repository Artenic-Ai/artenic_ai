"""Tests for artenic_ai_sdk.types — enums and type aliases."""

from __future__ import annotations

import pytest

from artenic_ai_sdk.types import (
    CircuitBreakerState,
    ConfigPhase,
    DriftType,
    EnsemblePhase,
    EnsembleStrategyType,
    EvolutionTrigger,
    ModelFramework,
    ModelPhase,
    SerializationFormat,
)


class TestModelPhase:
    def test_all_values(self) -> None:
        expected = {"created", "warming_up", "ready", "inference", "training", "error", "shutdown"}
        assert {v.value for v in ModelPhase} == expected

    def test_str_enum(self) -> None:
        assert str(ModelPhase.READY) == "ready"
        assert ModelPhase("ready") == ModelPhase.READY

    def test_member_count(self) -> None:
        assert len(ModelPhase) == 7


class TestModelFramework:
    def test_all_values(self) -> None:
        expected = {
            "pytorch",
            "tensorflow",
            "jax",
            "transformers",
            "lightgbm",
            "xgboost",
            "catboost",
            "sklearn",
            "onnx",
            "custom",
        }
        assert {v.value for v in ModelFramework} == expected

    def test_new_frameworks(self) -> None:
        """New frameworks vs v1."""
        assert ModelFramework.TRANSFORMERS == "transformers"
        assert ModelFramework.JAX == "jax"
        assert ModelFramework.TENSORFLOW == "tensorflow"
        assert ModelFramework.XGBOOST == "xgboost"
        assert ModelFramework.CATBOOST == "catboost"
        assert ModelFramework.CUSTOM == "custom"

    def test_member_count(self) -> None:
        assert len(ModelFramework) == 10


class TestSerializationFormat:
    def test_all_values(self) -> None:
        expected = {"safetensors", "onnx", "torch", "torchscript", "pickle", "joblib"}
        assert {v.value for v in SerializationFormat} == expected

    def test_new_formats(self) -> None:
        assert SerializationFormat.TORCHSCRIPT == "torchscript"
        assert SerializationFormat.PICKLE == "pickle"
        assert SerializationFormat.JOBLIB == "joblib"

    def test_member_count(self) -> None:
        assert len(SerializationFormat) == 6


class TestEnsembleStrategyType:
    def test_all_values(self) -> None:
        expected = {
            "weighted_average",
            "dynamic_weighting",
            "meta_learner",
            "majority_voting",
            "stacking",
        }
        assert {v.value for v in EnsembleStrategyType} == expected

    def test_new_strategies(self) -> None:
        assert EnsembleStrategyType.MAJORITY_VOTING == "majority_voting"
        assert EnsembleStrategyType.STACKING == "stacking"

    def test_member_count(self) -> None:
        assert len(EnsembleStrategyType) == 5


class TestEnsemblePhase:
    def test_all_values(self) -> None:
        expected = {"draft", "staging", "canary", "production", "archived"}
        assert {v.value for v in EnsemblePhase} == expected

    def test_member_count(self) -> None:
        assert len(EnsemblePhase) == 5


class TestEvolutionTrigger:
    def test_all_values(self) -> None:
        expected = {"model_count", "performance_threshold", "manual"}
        assert {v.value for v in EvolutionTrigger} == expected


class TestDriftType:
    def test_all_values(self) -> None:
        expected = {"data_drift", "concept_drift", "prediction_drift", "performance_drift"}
        assert {v.value for v in DriftType} == expected


class TestConfigPhase:
    def test_all_values(self) -> None:
        expected = {"default", "active", "candidate", "promoted", "retired"}
        assert {v.value for v in ConfigPhase} == expected


class TestCircuitBreakerState:
    def test_all_values(self) -> None:
        expected = {"closed", "open", "half_open"}
        assert {v.value for v in CircuitBreakerState} == expected


class TestStrEnumBehavior:
    """Cross-cutting: all enums should behave as proper StrEnums."""

    @pytest.mark.parametrize(
        "enum_cls",
        [
            ModelPhase,
            ModelFramework,
            SerializationFormat,
            EnsembleStrategyType,
            EnsemblePhase,
            EvolutionTrigger,
            DriftType,
            ConfigPhase,
            CircuitBreakerState,
        ],
    )
    def test_all_members_are_strings(self, enum_cls: type) -> None:
        for member in enum_cls:
            assert isinstance(member, str)
            assert isinstance(member.value, str)

    @pytest.mark.parametrize(
        "enum_cls",
        [
            ModelPhase,
            ModelFramework,
            SerializationFormat,
            EnsembleStrategyType,
            EnsemblePhase,
            EvolutionTrigger,
            DriftType,
            ConfigPhase,
            CircuitBreakerState,
        ],
    )
    def test_invalid_value_raises(self, enum_cls: type) -> None:
        with pytest.raises(ValueError):
            enum_cls("__NONEXISTENT__")
