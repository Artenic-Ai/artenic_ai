"""``artenic settings`` — runtime settings management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import click

from artenic_ai_cli import _async
from artenic_ai_cli._output import print_result, print_success
from artenic_ai_cli.main import handle_errors

if TYPE_CHECKING:
    from artenic_ai_cli._context import CliContext


@click.group("settings")
def settings_group() -> None:
    """View and update platform runtime settings."""


@settings_group.command("schema")
@click.argument("scope", required=False, default=None)
@click.argument("section", required=False, default=None)
@click.pass_obj
@handle_errors
def settings_schema(ctx: CliContext, scope: str | None, section: str | None) -> None:
    """Show configuration schema (optionally filtered by SCOPE and SECTION)."""
    if scope and section:
        path = f"/api/v1/settings/schema/{scope}/{section}"
    elif scope:
        path = f"/api/v1/settings/schema/{scope}"
    else:
        path = "/api/v1/settings/schema"

    async def _run() -> Any:
        async with ctx.api as api:
            return await api.get(path)

    print_result(ctx, _async.run_async(_run()), title="Settings Schema")


@settings_group.command("get")
@click.argument("scope")
@click.argument("section", required=False, default=None)
@click.pass_obj
@handle_errors
def settings_get(ctx: CliContext, scope: str, section: str | None) -> None:
    """Get current settings for SCOPE (and optional SECTION)."""
    path = f"/api/v1/settings/{scope}"
    if section:
        path += f"/{section}"

    async def _run() -> Any:
        async with ctx.api as api:
            return await api.get(path)

    title = f"Settings: {scope}" + (f"/{section}" if section else "")
    print_result(ctx, _async.run_async(_run()), title=title)


@settings_group.command("update")
@click.argument("scope")
@click.argument("section")
@click.option("--set", "pairs", multiple=True, required=True, help="KEY=VALUE pair (repeatable).")
@click.pass_obj
@handle_errors
def settings_update(ctx: CliContext, scope: str, section: str, pairs: tuple[str, ...]) -> None:
    """Update settings for SCOPE/SECTION with --set KEY=VALUE pairs."""
    body: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise click.BadParameter(f"Expected KEY=VALUE, got: '{pair}'", param_hint="--set")
        key, _, value = pair.partition("=")
        body[key] = value

    async def _run() -> Any:
        async with ctx.api as api:
            return await api.put(f"/api/v1/settings/{scope}/{section}", json=body)

    result = _async.run_async(_run())
    if ctx.json_mode:
        print_result(ctx, result)
    else:
        print_success(ctx, f"Settings {scope}/{section} updated.")


@settings_group.command("audit")
@click.option("--limit", type=int, default=50, help="Max entries.")
@click.option("--offset", type=int, default=0, help="Result offset.")
@click.pass_obj
@handle_errors
def settings_audit(ctx: CliContext, limit: int, offset: int) -> None:
    """Show settings audit log."""

    async def _run() -> Any:
        async with ctx.api as api:
            return await api.get(
                "/api/v1/settings/audit/log", params={"limit": limit, "offset": offset}
            )

    print_result(ctx, _async.run_async(_run()), title="Audit Log")
