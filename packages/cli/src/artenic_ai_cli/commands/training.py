"""``artenic training`` — training job management."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import click

from artenic_ai_cli import _async
from artenic_ai_cli._output import print_result, print_success
from artenic_ai_cli.main import handle_errors

if TYPE_CHECKING:
    from artenic_ai_cli._context import CliContext

_JOB_COLUMNS: list[tuple[str, str]] = [
    ("Job ID", "job_id"),
    ("Service", "service"),
    ("Provider", "provider"),
    ("Status", "status"),
]


@click.group("training")
def training_group() -> None:
    """Dispatch and manage training jobs."""


@training_group.command("dispatch")
@click.option("--service", required=True, help="Target service.")
@click.option("--model", required=True, help="Model name.")
@click.option("--provider", required=True, help="Training provider (e.g. gcp, aws, local).")
@click.option("--config-json", default=None, help="Training config as JSON string.")
@click.option("--instance-type", default=None, help="Instance type override.")
@click.option("--region", default=None, help="Region override.")
@click.option("--spot", is_flag=True, default=False, help="Use spot/preemptible instances.")
@click.option("--max-hours", type=float, default=None, help="Max runtime in hours.")
@click.pass_obj
@handle_errors
def dispatch_training(
    ctx: CliContext,
    service: str,
    model: str,
    provider: str,
    config_json: str | None,
    instance_type: str | None,
    region: str | None,
    spot: bool,
    max_hours: float | None,
) -> None:
    """Dispatch a new training job."""
    body: dict[str, Any] = {
        "service": service,
        "model": model,
        "provider": provider,
    }
    if config_json:
        try:
            body["config"] = json.loads(config_json)
        except json.JSONDecodeError as exc:
            raise click.BadParameter(f"Invalid JSON: {exc}", param_hint="--config-json") from None
    if instance_type:
        body["instance_type"] = instance_type
    if region:
        body["region"] = region
    if spot:
        body["is_spot"] = True
    if max_hours is not None:
        body["max_runtime_hours"] = max_hours

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.post("/api/v1/training/dispatch", json=body)  # type: ignore[no-any-return]

    result = _async.run_async(_run())
    if ctx.json_mode:
        print_result(ctx, result)
    else:
        print_success(ctx, f"Training dispatched: {result.get('job_id', result)}")


@training_group.command("list")
@click.option("--service", default=None, help="Filter by service.")
@click.option("--provider", default=None, help="Filter by provider.")
@click.option("--status", default=None, help="Filter by status.")
@click.option("--limit", type=int, default=50, help="Max results.")
@click.option("--offset", type=int, default=0, help="Result offset.")
@click.pass_obj
@handle_errors
def list_jobs(
    ctx: CliContext,
    service: str | None,
    provider: str | None,
    status: str | None,
    limit: int,
    offset: int,
) -> None:
    """List training jobs."""
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if service:
        params["service"] = service
    if provider:
        params["provider"] = provider
    if status:
        params["status"] = status

    async def _run() -> list[dict[str, Any]]:
        async with ctx.api as api:
            return await api.get("/api/v1/training/jobs", params=params)  # type: ignore[no-any-return]

    print_result(ctx, _async.run_async(_run()), columns=_JOB_COLUMNS, title="Training Jobs")


@training_group.command("status")
@click.argument("job_id")
@click.pass_obj
@handle_errors
def job_status(ctx: CliContext, job_id: str) -> None:
    """Get training job status."""

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.get(f"/api/v1/training/{job_id}")  # type: ignore[no-any-return]

    print_result(ctx, _async.run_async(_run()), title=f"Job {job_id}")


@training_group.command("cancel")
@click.argument("job_id")
@click.pass_obj
@handle_errors
def cancel_job(ctx: CliContext, job_id: str) -> None:
    """Cancel a running or pending job."""

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.post(f"/api/v1/training/{job_id}/cancel")  # type: ignore[no-any-return]

    result = _async.run_async(_run())
    if ctx.json_mode:
        print_result(ctx, result)
    else:
        print_success(ctx, f"Job {job_id} cancelled.")
