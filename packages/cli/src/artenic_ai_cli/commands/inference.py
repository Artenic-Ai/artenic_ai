"""``artenic predict`` / ``artenic predict-batch`` — inference commands."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import click

from artenic_ai_cli import _async
from artenic_ai_cli._output import print_result
from artenic_ai_cli.main import handle_errors

if TYPE_CHECKING:
    from artenic_ai_cli._context import CliContext


@click.command("predict")
@click.argument("service")
@click.option("--data", required=True, help="Input data as JSON string.")
@click.option("--model-id", default=None, help="Target model ID (optional).")
@click.pass_obj
@handle_errors
def predict_cmd(ctx: CliContext, service: str, data: str, model_id: str | None) -> None:
    """Run a single prediction on SERVICE."""
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as exc:
        raise click.BadParameter(f"Invalid JSON: {exc}", param_hint="--data") from None
    body: dict[str, Any] = {"data": parsed}
    if model_id:
        body["model_id"] = model_id

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.post(f"/api/v1/services/{service}/predict", json=body)  # type: ignore[no-any-return]

    print_result(ctx, _async.run_async(_run()), title="Prediction")


@click.command("predict-batch")
@click.argument("service")
@click.option("--data", required=True, help="Batch input data as JSON array string.")
@click.option("--model-id", default=None, help="Target model ID (optional).")
@click.pass_obj
@handle_errors
def predict_batch_cmd(
    ctx: CliContext,
    service: str,
    data: str,
    model_id: str | None,
) -> None:
    """Run batch predictions on SERVICE."""
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as exc:
        raise click.BadParameter(f"Invalid JSON: {exc}", param_hint="--data") from None
    body: dict[str, Any] = {"batch": parsed}
    if model_id:
        body["model_id"] = model_id

    async def _run() -> list[dict[str, Any]]:
        async with ctx.api as api:
            return await api.post(  # type: ignore[no-any-return]
                f"/api/v1/services/{service}/predict_batch",
                json=body,
            )

    print_result(ctx, _async.run_async(_run()), title="Batch Predictions")
