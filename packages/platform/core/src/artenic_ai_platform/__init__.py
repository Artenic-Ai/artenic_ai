"""Artenic AI Platform — Central gateway, registry, and orchestrator."""

from __future__ import annotations

__version__ = "0.7.0"

from artenic_ai_platform.app import create_app
from artenic_ai_platform.settings import PlatformSettings

__all__ = [
    "PlatformSettings",
    "__version__",
    "create_app",
]
