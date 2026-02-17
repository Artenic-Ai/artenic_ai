"""``artenic ensemble`` — ensemble management."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import click

from artenic_ai_cli import _async
from artenic_ai_cli._output import print_result, print_success
from artenic_ai_cli.main import handle_errors

if TYPE_CHECKING:
    from artenic_ai_cli._context import CliContext

_ENSEMBLE_COLUMNS: list[tuple[str, str]] = [
    ("ID", "id"),
    ("Name", "name"),
    ("Service", "service"),
    ("Strategy", "strategy"),
    ("Stage", "stage"),
    ("Version", "version"),
]


@click.group("ensemble")
def ensemble_group() -> None:
    """Manage model ensembles."""


@ensemble_group.command("create")
@click.option("--name", required=True, help="Ensemble name.")
@click.option("--service", required=True, help="Target service.")
@click.option("--strategy", required=True, help="Ensemble strategy.")
@click.option("--model-ids", required=True, help="Comma-separated model IDs.")
@click.option("--description", default=None, help="Description.")
@click.option("--strategy-config", default=None, help="Strategy config as JSON.")
@click.pass_obj
@handle_errors
def create_ensemble(
    ctx: CliContext,
    name: str,
    service: str,
    strategy: str,
    model_ids: str,
    description: str | None,
    strategy_config: str | None,
) -> None:
    """Create a new ensemble."""
    body: dict[str, Any] = {
        "name": name,
        "service": service,
        "strategy": strategy,
        "model_ids": [m.strip() for m in model_ids.split(",") if m.strip()],
    }
    if description:
        body["description"] = description
    if strategy_config:
        try:
            body["strategy_config"] = json.loads(strategy_config)
        except json.JSONDecodeError as exc:
            raise click.BadParameter(
                f"Invalid JSON: {exc}", param_hint="--strategy-config"
            ) from None

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.post("/api/v1/ensembles", json=body)  # type: ignore[no-any-return]

    result = _async.run_async(_run())
    if ctx.json_mode:
        print_result(ctx, result)
    else:
        print_success(ctx, f"Ensemble created: {result.get('ensemble_id', result)}")


@ensemble_group.command("list")
@click.option("--service", default=None, help="Filter by service.")
@click.option("--stage", default=None, help="Filter by stage.")
@click.option("--limit", type=int, default=50, help="Max results.")
@click.option("--offset", type=int, default=0, help="Result offset.")
@click.pass_obj
@handle_errors
def list_ensembles(
    ctx: CliContext,
    service: str | None,
    stage: str | None,
    limit: int,
    offset: int,
) -> None:
    """List ensembles."""
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if service:
        params["service"] = service
    if stage:
        params["stage"] = stage

    async def _run() -> list[dict[str, Any]]:
        async with ctx.api as api:
            return await api.get("/api/v1/ensembles", params=params)  # type: ignore[no-any-return]

    print_result(ctx, _async.run_async(_run()), columns=_ENSEMBLE_COLUMNS, title="Ensembles")


@ensemble_group.command("get")
@click.argument("ensemble_id")
@click.pass_obj
@handle_errors
def get_ensemble(ctx: CliContext, ensemble_id: str) -> None:
    """Get ensemble details."""

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.get(f"/api/v1/ensembles/{ensemble_id}")  # type: ignore[no-any-return]

    print_result(ctx, _async.run_async(_run()), title=f"Ensemble {ensemble_id}")


@ensemble_group.command("update")
@click.argument("ensemble_id")
@click.option("--name", default=None, help="New name.")
@click.option("--model-ids", default=None, help="Comma-separated model IDs.")
@click.option("--strategy", default=None, help="New strategy.")
@click.option("--strategy-config", default=None, help="Strategy config as JSON.")
@click.option("--description", default=None, help="New description.")
@click.option("--reason", default=None, help="Change reason.")
@click.pass_obj
@handle_errors
def update_ensemble(
    ctx: CliContext,
    ensemble_id: str,
    name: str | None,
    model_ids: str | None,
    strategy: str | None,
    strategy_config: str | None,
    description: str | None,
    reason: str | None,
) -> None:
    """Update an ensemble (bumps version)."""
    body: dict[str, Any] = {}
    if name:
        body["name"] = name
    if model_ids:
        body["model_ids"] = [m.strip() for m in model_ids.split(",") if m.strip()]
    if strategy:
        body["strategy"] = strategy
    if strategy_config:
        try:
            body["strategy_config"] = json.loads(strategy_config)
        except json.JSONDecodeError as exc:
            raise click.BadParameter(
                f"Invalid JSON: {exc}", param_hint="--strategy-config"
            ) from None
    if description:
        body["description"] = description
    if reason:
        body["change_reason"] = reason

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.put(f"/api/v1/ensembles/{ensemble_id}", json=body)  # type: ignore[no-any-return]

    result = _async.run_async(_run())
    if ctx.json_mode:
        print_result(ctx, result)
    else:
        print_success(ctx, f"Ensemble {ensemble_id} updated.")


@ensemble_group.command("train")
@click.argument("ensemble_id")
@click.option("--provider", required=True, help="Training provider.")
@click.option("--config-json", default=None, help="Training config as JSON.")
@click.pass_obj
@handle_errors
def train_ensemble(
    ctx: CliContext,
    ensemble_id: str,
    provider: str,
    config_json: str | None,
) -> None:
    """Dispatch ensemble training."""
    body: dict[str, Any] = {"provider": provider}
    if config_json:
        try:
            body["config"] = json.loads(config_json)
        except json.JSONDecodeError as exc:
            raise click.BadParameter(f"Invalid JSON: {exc}", param_hint="--config-json") from None

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.post(  # type: ignore[no-any-return]
                f"/api/v1/ensembles/{ensemble_id}/train", json=body
            )

    result = _async.run_async(_run())
    if ctx.json_mode:
        print_result(ctx, result)
    else:
        print_success(ctx, f"Ensemble training dispatched: {result.get('job_id', result)}")


@ensemble_group.command("job-status")
@click.argument("ensemble_id")
@click.argument("job_id")
@click.pass_obj
@handle_errors
def ensemble_job_status(ctx: CliContext, ensemble_id: str, job_id: str) -> None:
    """Get ensemble training job status."""

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.get(  # type: ignore[no-any-return]
                f"/api/v1/ensembles/{ensemble_id}/jobs/{job_id}"
            )

    print_result(ctx, _async.run_async(_run()), title=f"Ensemble Job {job_id}")


@ensemble_group.command("versions")
@click.argument("ensemble_id")
@click.option("--limit", type=int, default=20, help="Max versions to show.")
@click.pass_obj
@handle_errors
def ensemble_versions(ctx: CliContext, ensemble_id: str, limit: int) -> None:
    """Show ensemble version history."""

    async def _run() -> list[dict[str, Any]]:
        async with ctx.api as api:
            return await api.get(  # type: ignore[no-any-return]
                f"/api/v1/ensembles/{ensemble_id}/versions", params={"limit": limit}
            )

    print_result(ctx, _async.run_async(_run()), title=f"Ensemble {ensemble_id} — Versions")
