"""CLI context object passed through Click's ``ctx.obj``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console

    from artenic_ai_cli._client import ApiClient
    from artenic_ai_cli._config import CliConfig


@dataclass
class CliContext:
    """Holds shared state for all CLI commands."""

    api: ApiClient
    console: Console
    err_console: Console
    json_mode: bool
    config: CliConfig
