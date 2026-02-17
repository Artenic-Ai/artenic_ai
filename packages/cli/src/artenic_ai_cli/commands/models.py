"""``artenic model`` — model registry management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import click

from artenic_ai_cli import _async
from artenic_ai_cli._output import print_result, print_success
from artenic_ai_cli.main import handle_errors

if TYPE_CHECKING:
    from artenic_ai_cli._context import CliContext

_MODEL_COLUMNS: list[tuple[str, str]] = [
    ("ID", "id"),
    ("Name", "name"),
    ("Version", "version"),
    ("Type", "model_type"),
    ("Stage", "stage"),
]


@click.group("model")
def model_group() -> None:
    """Manage model registry."""


@model_group.command("list")
@click.pass_obj
@handle_errors
def list_models(ctx: CliContext) -> None:
    """List all registered models."""

    async def _run() -> list[dict[str, Any]]:
        async with ctx.api as api:
            return await api.get("/api/v1/models")  # type: ignore[no-any-return]

    print_result(ctx, _async.run_async(_run()), columns=_MODEL_COLUMNS, title="Models")


@model_group.command("get")
@click.argument("model_id")
@click.pass_obj
@handle_errors
def get_model(ctx: CliContext, model_id: str) -> None:
    """Get model details by ID."""

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.get(f"/api/v1/models/{model_id}")  # type: ignore[no-any-return]

    print_result(ctx, _async.run_async(_run()), title=f"Model {model_id}")


@model_group.command("register")
@click.option("--name", required=True, help="Model name.")
@click.option("--version", "model_version", required=True, help="Model version.")
@click.option("--type", "model_type", required=True, help="Model type.")
@click.option("--framework", default=None, help="Framework (e.g. pytorch, sklearn).")
@click.option("--description", default=None, help="Short description.")
@click.option("--tag", multiple=True, help="Tag as key=value (repeatable).")
@click.pass_obj
@handle_errors
def register_model(
    ctx: CliContext,
    name: str,
    model_version: str,
    model_type: str,
    framework: str | None,
    description: str | None,
    tag: tuple[str, ...],
) -> None:
    """Register a new model."""
    body: dict[str, Any] = {
        "name": name,
        "version": model_version,
        "model_type": model_type,
    }
    if framework:
        body["framework"] = framework
    if description:
        body["description"] = description
    if tag:
        tags: dict[str, str] = {}
        for t in tag:
            if "=" not in t:
                raise click.BadParameter(f"Tag must be key=value, got: '{t}'", param_hint="--tag")
            k, v = t.split("=", 1)
            tags[k] = v
        body["tags"] = tags

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.post("/api/v1/models", json=body)  # type: ignore[no-any-return]

    result = _async.run_async(_run())
    if ctx.json_mode:
        print_result(ctx, result)
    else:
        print_success(ctx, f"Model registered: {result.get('model_id', result)}")


@model_group.command("promote")
@click.argument("model_id")
@click.option("--version", "model_version", required=True, help="Version to promote.")
@click.pass_obj
@handle_errors
def promote_model(ctx: CliContext, model_id: str, model_version: str) -> None:
    """Promote a model version to production."""

    async def _run() -> None:
        async with ctx.api as api:
            await api.post(
                f"/api/v1/models/{model_id}/promote",
                json={"version": model_version},
            )

    _async.run_async(_run())
    if ctx.json_mode:
        print_result(ctx, {"status": "promoted", "model_id": model_id})
    else:
        print_success(ctx, f"Model {model_id} promoted to production (v{model_version}).")


@model_group.command("retire")
@click.argument("model_id")
@click.pass_obj
@handle_errors
def retire_model(ctx: CliContext, model_id: str) -> None:
    """Retire (archive) a model."""

    async def _run() -> None:
        async with ctx.api as api:
            await api.post(f"/api/v1/models/{model_id}/retire")

    _async.run_async(_run())
    if ctx.json_mode:
        print_result(ctx, {"status": "retired", "model_id": model_id})
    else:
        print_success(ctx, f"Model {model_id} retired.")
