"""Tests for artenic_ai_platform.inference.model_loader — 100% coverage."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from artenic_ai_platform.inference.model_loader import ModelLoader
from artenic_ai_platform.plugins.loader import PluginInfo, PluginRegistry
from artenic_ai_sdk.exceptions import ModelNotFoundError
from artenic_ai_sdk.testing import MockModel


def _make_registry(**services: PluginInfo) -> PluginRegistry:
    """Build a PluginRegistry with only services populated."""
    return PluginRegistry(services=dict(services))


# ======================================================================
# ModelLoader — load_from_registry
# ======================================================================


class TestLoadFromRegistry:
    async def test_empty_registry(self) -> None:
        loader = ModelLoader()
        await loader.load_from_registry(PluginRegistry())
        assert loader.count == 0
        assert loader.list_models() == []

    async def test_load_single_model(self) -> None:
        registry = _make_registry(
            my_model=PluginInfo(
                name="my_model",
                group="artenic_ai.services",
                module="m",
                obj=MockModel,
            ),
        )
        loader = ModelLoader()
        await loader.load_from_registry(registry)

        assert loader.count == 1
        assert "mock_model" in loader.list_models()

    async def test_load_multiple_models(self) -> None:
        class ModelA(MockModel):
            @property
            def model_id(self) -> str:
                return "model_a"

        class ModelB(MockModel):
            @property
            def model_id(self) -> str:
                return "model_b"

        registry = _make_registry(
            a=PluginInfo(
                name="a", group="artenic_ai.services", module="m", obj=ModelA
            ),
            b=PluginInfo(
                name="b", group="artenic_ai.services", module="m", obj=ModelB
            ),
        )
        loader = ModelLoader()
        await loader.load_from_registry(registry)

        assert loader.count == 2
        assert loader.list_models() == ["model_a", "model_b"]

    async def test_skip_disabled_plugin(self) -> None:
        registry = _make_registry(
            broken=PluginInfo(
                name="broken",
                group="artenic_ai.services",
                module="m",
                enabled=False,
                error="import fail",
            ),
        )
        loader = ModelLoader()
        await loader.load_from_registry(registry)
        assert loader.count == 0

    async def test_skip_plugin_with_no_obj(self) -> None:
        registry = _make_registry(
            empty=PluginInfo(
                name="empty",
                group="artenic_ai.services",
                module="m",
                obj=None,
            ),
        )
        loader = ModelLoader()
        await loader.load_from_registry(registry)
        assert loader.count == 0

    async def test_skip_failed_instantiation(self) -> None:
        def bad_class() -> None:
            msg = "cannot init"
            raise RuntimeError(msg)

        registry = _make_registry(
            bad=PluginInfo(
                name="bad",
                group="artenic_ai.services",
                module="m",
                obj=bad_class,
            ),
        )
        loader = ModelLoader()
        await loader.load_from_registry(registry)
        assert loader.count == 0

    async def test_skip_failed_warmup(self) -> None:
        class FailWarmup(MockModel):
            async def _do_warmup(self) -> None:
                msg = "warmup failed"
                raise RuntimeError(msg)

        registry = _make_registry(
            fail=PluginInfo(
                name="fail",
                group="artenic_ai.services",
                module="m",
                obj=FailWarmup,
            ),
        )
        loader = ModelLoader()
        await loader.load_from_registry(registry)
        assert loader.count == 0

    async def test_duplicate_model_id_skipped(self) -> None:
        """Second plugin with same model_id is skipped and torn down."""
        registry = _make_registry(
            first=PluginInfo(
                name="first",
                group="artenic_ai.services",
                module="m",
                obj=MockModel,
            ),
            second=PluginInfo(
                name="second",
                group="artenic_ai.services",
                module="m",
                obj=MockModel,
            ),
        )
        loader = ModelLoader()
        await loader.load_from_registry(registry)

        # Only first should be registered
        assert loader.count == 1


# ======================================================================
# ModelLoader — get_model
# ======================================================================


class TestGetModel:
    async def test_found(self) -> None:
        registry = _make_registry(
            m=PluginInfo(
                name="m",
                group="artenic_ai.services",
                module="m",
                obj=MockModel,
            ),
        )
        loader = ModelLoader()
        await loader.load_from_registry(registry)

        model = loader.get_model("mock_model")
        assert model.model_id == "mock_model"
        assert model.is_ready

    async def test_not_found(self) -> None:
        loader = ModelLoader()
        with pytest.raises(ModelNotFoundError, match="not_here"):
            loader.get_model("not_here")

    async def test_not_found_includes_available(self) -> None:
        registry = _make_registry(
            m=PluginInfo(
                name="m",
                group="artenic_ai.services",
                module="m",
                obj=MockModel,
            ),
        )
        loader = ModelLoader()
        await loader.load_from_registry(registry)

        with pytest.raises(ModelNotFoundError) as exc_info:
            loader.get_model("missing")
        assert exc_info.value.details is not None
        assert "mock_model" in exc_info.value.details["available"]


# ======================================================================
# ModelLoader — list_models / count
# ======================================================================


class TestListModels:
    def test_empty(self) -> None:
        loader = ModelLoader()
        assert loader.list_models() == []
        assert loader.count == 0

    async def test_sorted(self) -> None:
        class ZModel(MockModel):
            @property
            def model_id(self) -> str:
                return "z_model"

        class AModel(MockModel):
            @property
            def model_id(self) -> str:
                return "a_model"

        registry = _make_registry(
            z=PluginInfo(
                name="z", group="artenic_ai.services", module="m", obj=ZModel
            ),
            a=PluginInfo(
                name="a", group="artenic_ai.services", module="m", obj=AModel
            ),
        )
        loader = ModelLoader()
        await loader.load_from_registry(registry)
        assert loader.list_models() == ["a_model", "z_model"]


# ======================================================================
# ModelLoader — teardown_all
# ======================================================================


class TestTeardownAll:
    async def test_teardown_clears_registry(self) -> None:
        registry = _make_registry(
            m=PluginInfo(
                name="m",
                group="artenic_ai.services",
                module="m",
                obj=MockModel,
            ),
        )
        loader = ModelLoader()
        await loader.load_from_registry(registry)
        assert loader.count == 1

        await loader.teardown_all()
        assert loader.count == 0
        assert loader.list_models() == []

    async def test_teardown_error_logged_not_raised(self) -> None:
        """Teardown errors are caught and logged, not raised."""
        registry = _make_registry(
            m=PluginInfo(
                name="m",
                group="artenic_ai.services",
                module="m",
                obj=MockModel,
            ),
        )
        loader = ModelLoader()
        await loader.load_from_registry(registry)

        model = loader.get_model("mock_model")
        with patch.object(
            model, "teardown", new_callable=AsyncMock, side_effect=RuntimeError("boom")
        ):
            await loader.teardown_all()  # should not raise

        assert loader.count == 0
