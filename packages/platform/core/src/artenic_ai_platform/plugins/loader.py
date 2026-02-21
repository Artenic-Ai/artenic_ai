"""Plugin discovery via Python entry points."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any

logger = logging.getLogger(__name__)

#: Well-known entry-point groups scanned for plugins.
PLUGIN_GROUPS = (
    "artenic_ai.providers",
    "artenic_ai.strategies",
    "artenic_ai.services",
)


@dataclass(frozen=True)
class PluginInfo:
    """Metadata about a discovered plugin."""

    name: str
    group: str
    module: str
    enabled: bool = True
    error: str | None = None
    obj: Any = None


@dataclass
class PluginRegistry:
    """Container for all discovered plugins."""

    providers: dict[str, PluginInfo] = field(default_factory=dict)
    strategies: dict[str, PluginInfo] = field(default_factory=dict)
    services: dict[str, PluginInfo] = field(default_factory=dict)

    @property
    def all_plugins(self) -> list[PluginInfo]:
        """All plugins across every group."""
        return (
            list(self.providers.values())
            + list(self.strategies.values())
            + list(self.services.values())
        )

    @property
    def count(self) -> int:
        """Total number of discovered plugins."""
        return len(self.providers) + len(self.strategies) + len(self.services)


def _get_entry_points(group: str) -> list[Any]:
    """Load entry points for a group."""
    return list(entry_points(group=group))


def _group_to_attr(group: str) -> str:
    """Map entry-point group name to PluginRegistry attribute."""
    suffix = group.rsplit(".", 1)[-1]
    return suffix


def discover_plugins() -> PluginRegistry:
    """Scan entry points and build a PluginRegistry.

    Failed imports are logged as warnings and recorded with
    ``enabled=False`` + the error message.
    """
    registry = PluginRegistry()

    for group in PLUGIN_GROUPS:
        attr = _group_to_attr(group)
        target: dict[str, PluginInfo] = getattr(registry, attr)

        for ep in _get_entry_points(group):
            try:
                loaded = ep.load()
                info = PluginInfo(
                    name=ep.name,
                    group=group,
                    module=str(ep.value),
                    obj=loaded,
                )
                target[ep.name] = info
                logger.debug("Loaded plugin %s from %s", ep.name, group)
            except Exception as exc:
                info = PluginInfo(
                    name=ep.name,
                    group=group,
                    module=str(ep.value),
                    enabled=False,
                    error=str(exc),
                )
                target[ep.name] = info
                logger.warning(
                    "Failed to load plugin %s from %s: %s",
                    ep.name,
                    group,
                    exc,
                )

    logger.info("Discovered %d plugins", registry.count)
    return registry
