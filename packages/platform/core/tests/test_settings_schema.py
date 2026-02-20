"""Tests for artenic_ai_platform.config.settings_schema — 100% coverage."""

from __future__ import annotations

from artenic_ai_platform.config.settings_schema import (
    BUDGET_SECTION,
    CORE_SECTION,
    GLOBAL_SECTIONS,
    OTEL_SECTION,
    SCHEMA_REGISTRY,
    WEBHOOK_SECTION,
    FieldMeta,
    SectionMeta,
    get_schema_all,
    get_schema_for_scope,
    get_section_schema,
)

# ======================================================================
# FieldMeta
# ======================================================================


class TestFieldMeta:
    def test_defaults(self) -> None:
        f = FieldMeta(key="k", label="K", type="str")
        assert f.key == "k"
        assert f.default is None
        assert f.secret is False
        assert f.restart_required is False
        assert f.choices == []
        assert f.description == ""

    def test_all_fields(self) -> None:
        f = FieldMeta(
            key="k",
            label="K",
            type="str",
            default="v",
            secret=True,
            restart_required=True,
            choices=["a", "b"],
            description="desc",
        )
        assert f.default == "v"
        assert f.secret is True
        assert f.choices == ["a", "b"]

    def test_frozen(self) -> None:
        f = FieldMeta(key="k", label="K", type="str")
        import dataclasses

        assert dataclasses.is_dataclass(f)
        # Frozen dataclass should raise on attribute modification
        try:
            f.key = "other"  # type: ignore[misc]
            raise AssertionError("Should have raised")  # pragma: no cover
        except dataclasses.FrozenInstanceError:
            pass


# ======================================================================
# SectionMeta
# ======================================================================


class TestSectionMeta:
    def test_defaults(self) -> None:
        s = SectionMeta(name="s", label="S")
        assert s.fields == []
        assert s.description == ""

    def test_with_fields(self) -> None:
        f = FieldMeta(key="k", label="K", type="str")
        s = SectionMeta(name="s", label="S", fields=[f], description="test")
        assert len(s.fields) == 1
        assert s.description == "test"


# ======================================================================
# Schema constants
# ======================================================================


class TestSchemaConstants:
    def test_all_sections_in_global_list(self) -> None:
        expected_names = {
            "core",
            "mlflow",
            "otel",
            "rate_limit",
            "cors",
            "budget",
            "webhook",
            "spot",
            "ensemble",
            "ab_test",
            "health",
            "canary",
        }
        actual_names = {s.name for s in GLOBAL_SECTIONS}
        assert actual_names == expected_names
        assert len(GLOBAL_SECTIONS) == 12

    def test_schema_registry_has_global(self) -> None:
        assert "global" in SCHEMA_REGISTRY
        assert SCHEMA_REGISTRY["global"] is GLOBAL_SECTIONS

    def test_core_has_secret_fields(self) -> None:
        secret_keys = {f.key for f in CORE_SECTION.fields if f.secret}
        assert "api_key" in secret_keys
        assert "secret_key" in secret_keys
        assert "database_url" in secret_keys

    def test_otel_exporter_has_choices(self) -> None:
        exporter = next(f for f in OTEL_SECTION.fields if f.key == "otel_exporter")
        assert exporter.choices == ["prometheus", "otlp"]

    def test_budget_enforcement_has_choices(self) -> None:
        mode = next(f for f in BUDGET_SECTION.fields if f.key == "enforcement_mode")
        assert mode.choices == ["block", "warn"]

    def test_webhook_secret_is_secret(self) -> None:
        secret = next(f for f in WEBHOOK_SECTION.fields if f.key == "secret")
        assert secret.secret is True

    def test_every_section_has_at_least_one_field(self) -> None:
        for section in GLOBAL_SECTIONS:
            assert len(section.fields) >= 1, f"{section.name} has no fields"


# ======================================================================
# get_schema_all
# ======================================================================


class TestGetSchemaAll:
    def test_returns_all_scopes(self) -> None:
        result = get_schema_all()
        assert "global" in result
        assert len(result["global"]) == 12

    def test_sections_are_dicts(self) -> None:
        result = get_schema_all()
        for section in result["global"]:
            assert "name" in section
            assert "label" in section
            assert "fields" in section


# ======================================================================
# get_schema_for_scope
# ======================================================================


class TestGetSchemaForScope:
    def test_global_scope(self) -> None:
        result = get_schema_for_scope("global")
        assert len(result) == 12
        names = {s["name"] for s in result}
        assert "core" in names
        assert "mlflow" in names

    def test_unknown_scope(self) -> None:
        result = get_schema_for_scope("nonexistent")
        assert result == []


# ======================================================================
# get_section_schema
# ======================================================================


class TestGetSectionSchema:
    def test_existing_section(self) -> None:
        result = get_section_schema("global", "core")
        assert result is not None
        assert result["name"] == "core"
        assert len(result["fields"]) == len(CORE_SECTION.fields)

    def test_unknown_section(self) -> None:
        result = get_section_schema("global", "nonexistent")
        assert result is None

    def test_unknown_scope(self) -> None:
        result = get_section_schema("nonexistent", "core")
        assert result is None

    def test_field_serialization_with_choices(self) -> None:
        result = get_section_schema("global", "otel")
        assert result is not None
        exporter = next(f for f in result["fields"] if f["key"] == "otel_exporter")
        assert "choices" in exporter
        assert exporter["choices"] == ["prometheus", "otlp"]

    def test_field_serialization_without_choices(self) -> None:
        result = get_section_schema("global", "core")
        assert result is not None
        host = next(f for f in result["fields"] if f["key"] == "host")
        assert "choices" not in host
