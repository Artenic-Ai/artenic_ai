"""Dynamic model loading via plugin discovery."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from artenic_ai_sdk.exceptions import ModelNotFoundError

if TYPE_CHECKING:
    from artenic_ai_platform.plugins.loader import PluginRegistry
    from artenic_ai_sdk.base_model import BaseModel

logger = logging.getLogger(__name__)


class ModelLoader:
    """Discover, instantiate, warm up and manage BaseModel plugins.

    Uses :func:`~artenic_ai_platform.plugins.loader.discover_plugins` to
    find model classes declared as ``artenic_ai.services`` entry points,
    instantiates each class, calls :meth:`warmup`, and registers them
    by their :attr:`model_id` property.

    Example::

        from artenic_ai_platform.plugins.loader import discover_plugins

        registry = discover_plugins()
        loader = ModelLoader()
        await loader.load_from_registry(registry)

        model = loader.get_model("forex:lgbm:intraday:v1")
        prediction = await model.predict(features)
    """

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}

    async def load_from_registry(self, registry: PluginRegistry) -> None:
        """Instantiate and warm up all enabled services from *registry*.

        Models that fail to instantiate or warm up are logged and skipped.
        Duplicate ``model_id`` values are logged and the duplicate is
        torn down.
        """
        for name, info in registry.services.items():
            if not info.enabled:
                logger.warning(
                    "Skipping disabled plugin %s: %s", name, info.error
                )
                continue

            if info.obj is None:
                logger.warning("Plugin %s has no loaded class, skipping", name)
                continue

            try:
                instance = info.obj()
            except Exception:
                logger.warning(
                    "Failed to instantiate plugin %s", name, exc_info=True
                )
                continue

            try:
                await instance.warmup()
            except Exception:
                logger.warning(
                    "Failed to warm up plugin %s", name, exc_info=True
                )
                continue

            model_id: str = instance.model_id
            if model_id in self._models:
                logger.warning(
                    "Duplicate model_id %s from plugin %s, skipping",
                    model_id,
                    name,
                )
                await instance.teardown()
                continue

            self._models[model_id] = instance
            logger.info("Loaded model %s (plugin=%s)", model_id, name)

    def get_model(self, model_id: str) -> BaseModel:
        """Return a ready model or raise :exc:`ModelNotFoundError`."""
        try:
            return self._models[model_id]  # type: ignore[no-any-return]
        except KeyError:
            raise ModelNotFoundError(
                f"Model '{model_id}' not found",
                details={"available": self.list_models()},
            ) from None

    def list_models(self) -> list[str]:
        """Return all registered model IDs."""
        return sorted(self._models)

    @property
    def count(self) -> int:
        """Number of loaded models."""
        return len(self._models)

    async def teardown_all(self) -> None:
        """Teardown every loaded model. Errors are logged, not raised."""
        for model_id, model in self._models.items():
            try:
                await model.teardown()
            except Exception:
                logger.warning(
                    "Error tearing down %s", model_id, exc_info=True
                )
        self._models.clear()
