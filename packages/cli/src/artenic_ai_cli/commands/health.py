"""``artenic health`` — platform health checks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import click

from artenic_ai_cli import _async
from artenic_ai_cli._output import print_result
from artenic_ai_cli.main import handle_errors

if TYPE_CHECKING:
    from artenic_ai_cli._context import CliContext


@click.group("health")
def health_group() -> None:
    """Check platform health status."""


@health_group.command("check")
@click.pass_obj
@handle_errors
def health_check(ctx: CliContext) -> None:
    """Basic liveness probe."""

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.get("/health")  # type: ignore[no-any-return]

    print_result(ctx, _async.run_async(_run()), title="Health")


@health_group.command("ready")
@click.pass_obj
@handle_errors
def health_ready(ctx: CliContext) -> None:
    """Readiness probe (checks DB connectivity)."""

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.get("/health/ready")  # type: ignore[no-any-return]

    print_result(ctx, _async.run_async(_run()), title="Readiness")


@health_group.command("detailed")
@click.pass_obj
@handle_errors
def health_detailed(ctx: CliContext) -> None:
    """Detailed component health."""

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.get("/health/detailed")  # type: ignore[no-any-return]

    print_result(ctx, _async.run_async(_run()), title="Detailed Health")
