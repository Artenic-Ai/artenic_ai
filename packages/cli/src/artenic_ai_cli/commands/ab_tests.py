"""``artenic ab-test`` — A/B testing management."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import click

from artenic_ai_cli import _async
from artenic_ai_cli._output import print_result, print_success
from artenic_ai_cli.main import handle_errors

if TYPE_CHECKING:
    from artenic_ai_cli._context import CliContext

_TEST_COLUMNS: list[tuple[str, str]] = [
    ("ID", "id"),
    ("Name", "name"),
    ("Service", "service"),
    ("Status", "status"),
    ("Primary Metric", "primary_metric"),
]


@click.group("ab-test")
def ab_test_group() -> None:
    """Manage A/B tests."""


@ab_test_group.command("create")
@click.option("--name", required=True, help="Test name.")
@click.option("--service", required=True, help="Target service.")
@click.option("--variants", required=True, help="Variants as JSON array.")
@click.option("--primary-metric", required=True, help="Primary metric to optimise.")
@click.option("--min-samples", type=int, default=None, help="Min samples per variant.")
@click.pass_obj
@handle_errors
def create_test(
    ctx: CliContext,
    name: str,
    service: str,
    variants: str,
    primary_metric: str,
    min_samples: int | None,
) -> None:
    """Create a new A/B test."""
    try:
        parsed_variants = json.loads(variants)
    except json.JSONDecodeError as exc:
        raise click.BadParameter(f"Invalid JSON: {exc}", param_hint="--variants") from None
    body: dict[str, Any] = {
        "name": name,
        "service": service,
        "variants": parsed_variants,
        "primary_metric": primary_metric,
    }
    if min_samples is not None:
        body["min_samples"] = min_samples

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.post("/api/v1/ab-tests", json=body)  # type: ignore[no-any-return]

    result = _async.run_async(_run())
    if ctx.json_mode:
        print_result(ctx, result)
    else:
        print_success(ctx, f"A/B test created: {result.get('test_id', result)}")


@ab_test_group.command("list")
@click.option("--service", default=None, help="Filter by service.")
@click.option("--status", default=None, help="Filter by status.")
@click.option("--limit", type=int, default=50, help="Max results.")
@click.option("--offset", type=int, default=0, help="Result offset.")
@click.pass_obj
@handle_errors
def list_tests(
    ctx: CliContext,
    service: str | None,
    status: str | None,
    limit: int,
    offset: int,
) -> None:
    """List A/B tests."""
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if service:
        params["service"] = service
    if status:
        params["status"] = status

    async def _run() -> list[dict[str, Any]]:
        async with ctx.api as api:
            return await api.get("/api/v1/ab-tests", params=params)  # type: ignore[no-any-return]

    print_result(ctx, _async.run_async(_run()), columns=_TEST_COLUMNS, title="A/B Tests")


@ab_test_group.command("get")
@click.argument("test_id")
@click.pass_obj
@handle_errors
def get_test(ctx: CliContext, test_id: str) -> None:
    """Get A/B test details."""

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.get(f"/api/v1/ab-tests/{test_id}")  # type: ignore[no-any-return]

    print_result(ctx, _async.run_async(_run()), title=f"A/B Test {test_id}")


@ab_test_group.command("results")
@click.argument("test_id")
@click.pass_obj
@handle_errors
def test_results(ctx: CliContext, test_id: str) -> None:
    """Get A/B test results and metrics."""

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.get(f"/api/v1/ab-tests/{test_id}/results")  # type: ignore[no-any-return]

    print_result(ctx, _async.run_async(_run()), title=f"A/B Test {test_id} — Results")


@ab_test_group.command("conclude")
@click.argument("test_id")
@click.option("--winner", default=None, help="Winning variant name.")
@click.option("--reason", default=None, help="Conclusion reason.")
@click.pass_obj
@handle_errors
def conclude_test(ctx: CliContext, test_id: str, winner: str | None, reason: str | None) -> None:
    """Conclude an A/B test."""
    body: dict[str, Any] = {}
    if winner:
        body["winner"] = winner
    if reason:
        body["reason"] = reason

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.post(  # type: ignore[no-any-return]
                f"/api/v1/ab-tests/{test_id}/conclude", json=body
            )

    result = _async.run_async(_run())
    if ctx.json_mode:
        print_result(ctx, result)
    else:
        print_success(ctx, f"A/B test {test_id} concluded.")


@ab_test_group.command("pause")
@click.argument("test_id")
@click.pass_obj
@handle_errors
def pause_test(ctx: CliContext, test_id: str) -> None:
    """Pause a running A/B test."""

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.post(f"/api/v1/ab-tests/{test_id}/pause")  # type: ignore[no-any-return]

    _async.run_async(_run())
    if ctx.json_mode:
        print_result(ctx, {"status": "paused", "test_id": test_id})
    else:
        print_success(ctx, f"A/B test {test_id} paused.")


@ab_test_group.command("resume")
@click.argument("test_id")
@click.pass_obj
@handle_errors
def resume_test(ctx: CliContext, test_id: str) -> None:
    """Resume a paused A/B test."""

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.post(f"/api/v1/ab-tests/{test_id}/resume")  # type: ignore[no-any-return]

    _async.run_async(_run())
    if ctx.json_mode:
        print_result(ctx, {"status": "running", "test_id": test_id})
    else:
        print_success(ctx, f"A/B test {test_id} resumed.")
