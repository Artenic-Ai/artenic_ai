"""Output helpers — Rich tables or JSON depending on ``--json`` flag."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from artenic_ai_cli._context import CliContext


def print_result(
    ctx: CliContext,
    data: Any,
    *,
    columns: list[tuple[str, str]] | None = None,
    title: str = "",
) -> None:
    """Print *data* as a Rich table or raw JSON.

    Args:
        ctx: CLI context (determines output mode).
        data: Payload — a dict, list of dicts, or any JSON-serialisable value.
        columns: ``(header, key)`` pairs for tabular display.  When *None*
            and *data* is a list of dicts, keys are inferred from the first item.
        title: Optional title rendered above the table.
    """
    if ctx.json_mode:
        ctx.console.print_json(json.dumps(data, default=str))
        return

    if isinstance(data, list):
        if not data:
            ctx.err_console.print("[dim]No results found.[/dim]")
            return
        if isinstance(data[0], dict):
            _print_table(ctx, data, columns=columns, title=title)
            return
    if isinstance(data, dict):
        _print_dict(ctx, data, title=title)
    else:
        ctx.console.print(data)


def print_success(ctx: CliContext, message: str) -> None:
    """Print a success message to stderr."""
    ctx.err_console.print(f"[green]{message}[/green]")


def print_error(ctx: CliContext, message: str) -> None:
    """Print an error message to stderr."""
    ctx.err_console.print(Panel(message, title="Error", border_style="red"))


def _print_table(
    ctx: CliContext,
    rows: list[dict[str, Any]],
    *,
    columns: list[tuple[str, str]] | None = None,
    title: str = "",
) -> None:
    if not columns and rows:
        columns = [(k, k) for k in rows[0]]

    table = Table(title=title or None)
    for header, _ in columns or []:
        table.add_column(header)

    for row in rows:
        table.add_row(*(str(row.get(key, "")) for _, key in columns or []))

    ctx.console.print(table)


def _print_dict(ctx: CliContext, data: dict[str, Any], *, title: str = "") -> None:
    table = Table(title=title or None, show_header=False)
    table.add_column("Key", style="bold")
    table.add_column("Value")
    for k, v in data.items():
        table.add_row(str(k), str(v))
    ctx.console.print(table)
