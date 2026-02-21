"""Tests for artenic_ai_platform.plugins.loader — 100% coverage."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from artenic_ai_platform.plugins.loader import (
    PLUGIN_GROUPS,
    PluginInfo,
    PluginRegistry,
    _get_entry_points,
    _group_to_attr,
    discover_plugins,
)

# ======================================================================
# PluginInfo
# ======================================================================


class TestPluginInfo:
    def test_defaults(self) -> None:
        info = PluginInfo(name="p", group="g", module="m")
        assert info.enabled is True
        assert info.error is None
        assert info.obj is None

    def test_disabled_with_error(self) -> None:
        info = PluginInfo(name="p", group="g", module="m", enabled=False, error="import fail")
        assert info.enabled is False
        assert info.error == "import fail"

    def test_frozen(self) -> None:
        info = PluginInfo(name="p", group="g", module="m")
        try:
            info.name = "other"  # type: ignore[misc]
            raise AssertionError("Should have raised")  # pragma: no cover
        except AttributeError:
            pass


# ======================================================================
# PluginRegistry
# ======================================================================


class TestPluginRegistry:
    def test_empty_registry(self) -> None:
        registry = PluginRegistry()
        assert registry.count == 0
        assert registry.all_plugins == []

    def test_with_plugins(self) -> None:
        registry = PluginRegistry()
        registry.providers["gcp"] = PluginInfo(
            name="gcp", group="artenic_ai.providers", module="my.gcp"
        )
        registry.strategies["ltr"] = PluginInfo(
            name="ltr", group="artenic_ai.strategies", module="my.ltr"
        )
        registry.services["train"] = PluginInfo(
            name="train", group="artenic_ai.services", module="my.train"
        )

        assert registry.count == 3
        assert len(registry.all_plugins) == 3


# ======================================================================
# _group_to_attr
# ======================================================================


class TestGroupToAttr:
    def test_providers(self) -> None:
        assert _group_to_attr("artenic_ai.providers") == "providers"

    def test_strategies(self) -> None:
        assert _group_to_attr("artenic_ai.strategies") == "strategies"

    def test_services(self) -> None:
        assert _group_to_attr("artenic_ai.services") == "services"


# ======================================================================
# _get_entry_points
# ======================================================================


class TestGetEntryPoints:
    def test_returns_list(self) -> None:
        # In test environment, no plugins are installed
        result = _get_entry_points("artenic_ai.providers")
        assert isinstance(result, list)


# ======================================================================
# PLUGIN_GROUPS constant
# ======================================================================


class TestPluginGroups:
    def test_groups_defined(self) -> None:
        assert "artenic_ai.providers" in PLUGIN_GROUPS
        assert "artenic_ai.strategies" in PLUGIN_GROUPS
        assert "artenic_ai.services" in PLUGIN_GROUPS


# ======================================================================
# discover_plugins
# ======================================================================


class TestDiscoverPlugins:
    def test_empty_discovery(self) -> None:
        with patch(
            "artenic_ai_platform.plugins.loader._get_entry_points",
            return_value=[],
        ):
            registry = discover_plugins()
        assert registry.count == 0

    def test_successful_plugin_load(self) -> None:
        mock_ep = SimpleNamespace(
            name="mock_provider",
            value="my.module:Provider",
            load=MagicMock(return_value=object),
        )

        with patch(
            "artenic_ai_platform.plugins.loader._get_entry_points",
            side_effect=lambda group: [mock_ep] if group == "artenic_ai.providers" else [],
        ):
            registry = discover_plugins()

        assert "mock_provider" in registry.providers
        info = registry.providers["mock_provider"]
        assert info.enabled is True
        assert info.error is None
        assert info.module == "my.module:Provider"
        assert info.obj is object

    def test_failed_plugin_load(self) -> None:
        mock_ep = SimpleNamespace(
            name="broken",
            value="bad.module:Thing",
            load=MagicMock(side_effect=ImportError("no module")),
        )

        with patch(
            "artenic_ai_platform.plugins.loader._get_entry_points",
            side_effect=lambda group: [mock_ep] if group == "artenic_ai.providers" else [],
        ):
            registry = discover_plugins()

        assert "broken" in registry.providers
        info = registry.providers["broken"]
        assert info.enabled is False
        assert "no module" in info.error  # type: ignore[operator]
        assert info.obj is None

    def test_mixed_success_and_failure(self) -> None:
        good_ep = SimpleNamespace(
            name="good",
            value="good.module:Good",
            load=MagicMock(return_value=object),
        )
        bad_ep = SimpleNamespace(
            name="bad",
            value="bad.module:Bad",
            load=MagicMock(side_effect=RuntimeError("boom")),
        )

        with patch(
            "artenic_ai_platform.plugins.loader._get_entry_points",
            side_effect=lambda group: [good_ep, bad_ep] if group == "artenic_ai.strategies" else [],
        ):
            registry = discover_plugins()

        assert registry.strategies["good"].enabled is True
        assert registry.strategies["bad"].enabled is False
        assert registry.count == 2

    def test_all_groups_scanned(self) -> None:
        calls: list[str] = []

        def fake_get(group: str) -> list:
            calls.append(group)
            return []

        with patch(
            "artenic_ai_platform.plugins.loader._get_entry_points",
            side_effect=fake_get,
        ):
            discover_plugins()

        assert set(calls) == set(PLUGIN_GROUPS)
