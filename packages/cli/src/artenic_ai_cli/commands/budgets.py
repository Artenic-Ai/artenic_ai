"""``artenic budget`` — budget management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import click

from artenic_ai_cli import _async
from artenic_ai_cli._output import print_result, print_success
from artenic_ai_cli.main import handle_errors

if TYPE_CHECKING:
    from artenic_ai_cli._context import CliContext

_BUDGET_COLUMNS: list[tuple[str, str]] = [
    ("ID", "id"),
    ("Scope", "scope"),
    ("Value", "scope_value"),
    ("Period", "period"),
    ("Limit (EUR)", "limit_eur"),
    ("Enabled", "enabled"),
]


@click.group("budget")
def budget_group() -> None:
    """Manage budget rules and spending."""


@budget_group.command("list")
@click.option("--scope", default=None, help="Filter by scope (global/service/provider).")
@click.option("--all", "show_all", is_flag=True, help="Include disabled budgets.")
@click.pass_obj
@handle_errors
def list_budgets(ctx: CliContext, scope: str | None, show_all: bool) -> None:
    """List budget rules."""
    params: dict[str, Any] = {"enabled_only": not show_all}
    if scope:
        params["scope"] = scope

    async def _run() -> list[dict[str, Any]]:
        async with ctx.api as api:
            return await api.get("/api/v1/budgets", params=params)  # type: ignore[no-any-return]

    print_result(ctx, _async.run_async(_run()), columns=_BUDGET_COLUMNS, title="Budgets")


@budget_group.command("create")
@click.option("--scope", required=True, help="Budget scope (global/service/provider).")
@click.option("--scope-value", required=True, help="Scope value (e.g. service name).")
@click.option("--period", required=True, help="Budget period (daily/weekly/monthly).")
@click.option("--limit-eur", type=float, required=True, help="Budget limit in EUR.")
@click.option("--alert-threshold", type=float, default=None, help="Alert threshold (0-1).")
@click.pass_obj
@handle_errors
def create_budget(
    ctx: CliContext,
    scope: str,
    scope_value: str,
    period: str,
    limit_eur: float,
    alert_threshold: float | None,
) -> None:
    """Create a budget rule."""
    body: dict[str, Any] = {
        "scope": scope,
        "scope_value": scope_value,
        "period": period,
        "limit_eur": limit_eur,
    }
    if alert_threshold is not None:
        body["alert_threshold_pct"] = alert_threshold

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.post("/api/v1/budgets", json=body)  # type: ignore[no-any-return]

    result = _async.run_async(_run())
    if ctx.json_mode:
        print_result(ctx, result)
    else:
        print_success(ctx, f"Budget created: {scope}/{scope_value} — {limit_eur} EUR/{period}")


@budget_group.command("update")
@click.argument("budget_id")
@click.option("--limit-eur", type=float, default=None, help="New limit in EUR.")
@click.option("--alert-threshold", type=float, default=None, help="New alert threshold (0-1).")
@click.option("--enabled/--disabled", default=None, help="Enable or disable the budget.")
@click.option("--period", default=None, help="New period (daily/weekly/monthly).")
@click.pass_obj
@handle_errors
def update_budget(
    ctx: CliContext,
    budget_id: str,
    limit_eur: float | None,
    alert_threshold: float | None,
    enabled: bool | None,
    period: str | None,
) -> None:
    """Update a budget rule."""
    body: dict[str, Any] = {}
    if limit_eur is not None:
        body["limit_eur"] = limit_eur
    if alert_threshold is not None:
        body["alert_threshold_pct"] = alert_threshold
    if enabled is not None:
        body["enabled"] = enabled
    if period:
        body["period"] = period

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.put(f"/api/v1/budgets/{budget_id}", json=body)  # type: ignore[no-any-return]

    result = _async.run_async(_run())
    if ctx.json_mode:
        print_result(ctx, result)
    else:
        print_success(ctx, f"Budget {budget_id} updated.")


@budget_group.command("spending")
@click.option("--scope", required=True, help="Scope (global/service/provider).")
@click.option("--scope-value", required=True, help="Scope value.")
@click.pass_obj
@handle_errors
def spending(ctx: CliContext, scope: str, scope_value: str) -> None:
    """Get current spending for a scope."""
    params: dict[str, Any] = {"scope": scope, "scope_value": scope_value}

    async def _run() -> Any:
        async with ctx.api as api:
            return await api.get("/api/v1/budgets/spending", params=params)

    print_result(ctx, _async.run_async(_run()), title=f"Spending: {scope}/{scope_value}")
