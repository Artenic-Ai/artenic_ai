"""``artenic config`` — local CLI configuration management."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from artenic_ai_cli._config import CliConfig, save_config_value
from artenic_ai_cli._output import print_result, print_success
from artenic_ai_cli.main import handle_errors

if TYPE_CHECKING:
    from artenic_ai_cli._context import CliContext

_SENSITIVE_KEYS = ("key", "secret", "token", "password")


@click.group("config")
def config_group() -> None:
    """Manage local CLI configuration (~/.artenic/config.toml)."""


@config_group.command("show")
@click.pass_obj
@handle_errors
def config_show(ctx: CliContext) -> None:
    """Show current effective configuration."""
    cfg = ctx.config
    data = {
        "url": cfg.url,
        "api_key": _mask(cfg.api_key),
        "timeout": cfg.timeout,
        "profile": cfg.profile,
    }
    print_result(ctx, data, title="Current Config")


@config_group.command("set")
@click.argument("key")
@click.argument("value")
@click.option("--profile", default="default", help="Target profile.")
@click.pass_obj
@handle_errors
def config_set(ctx: CliContext, key: str, value: str, profile: str) -> None:
    """Set a config value in ~/.artenic/config.toml."""
    allowed = CliConfig.__dataclass_fields__
    if key not in allowed:
        valid = ", ".join(sorted(allowed))
        raise click.BadParameter(f"Unknown key '{key}'. Valid keys: {valid}", param_hint="KEY")
    save_config_value(key, value, profile=profile)
    display_value = _mask(value) if any(k in key for k in _SENSITIVE_KEYS) else value
    print_success(ctx, f"[{profile}] {key} = {display_value}")


@config_group.command("use-profile")
@click.argument("name")
@click.pass_obj
@handle_errors
def config_use_profile(ctx: CliContext, name: str) -> None:
    """Switch active profile (writes ARTENIC_PROFILE hint to config)."""
    save_config_value("_active_profile", name)
    print_success(ctx, f"Active profile set to '{name}'.")


def _mask(value: str) -> str:
    if not value:
        return "(not set)"
    if len(value) <= 8:
        return "***"
    return "***..." + value[-4:]
