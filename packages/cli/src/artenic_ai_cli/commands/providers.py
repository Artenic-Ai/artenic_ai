"""``artenic provider`` — cloud provider management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import click

from artenic_ai_cli import _async
from artenic_ai_cli._output import print_result, print_success
from artenic_ai_cli.main import handle_errors

if TYPE_CHECKING:
    from artenic_ai_cli._context import CliContext

_PROVIDER_COLUMNS: list[tuple[str, str]] = [
    ("ID", "id"),
    ("Name", "display_name"),
    ("Status", "status"),
    ("Enabled", "enabled"),
]

_STORAGE_COLUMNS: list[tuple[str, str]] = [
    ("Provider", "provider_id"),
    ("Name", "name"),
    ("Type", "type"),
    ("Region", "region"),
]

_COMPUTE_COLUMNS: list[tuple[str, str]] = [
    ("Provider", "provider_id"),
    ("Name", "name"),
    ("vCPUs", "vcpus"),
    ("Memory (GB)", "memory_gb"),
    ("GPU", "gpu_type"),
    ("GPUs", "gpu_count"),
]

_REGION_COLUMNS: list[tuple[str, str]] = [
    ("ID", "id"),
    ("Name", "name"),
    ("Provider", "provider_id"),
]

BASE = "/api/v1/providers"


@click.group("provider")
def provider_group() -> None:
    """Manage cloud providers."""


# ---- List / Get ----


@provider_group.command("list")
@click.pass_obj
@handle_errors
def list_providers(ctx: CliContext) -> None:
    """List all providers."""

    async def _run() -> list[dict[str, Any]]:
        async with ctx.api as api:
            return await api.get(BASE)  # type: ignore[no-any-return]

    print_result(ctx, _async.run_async(_run()), columns=_PROVIDER_COLUMNS, title="Providers")


@provider_group.command("get")
@click.argument("provider_id")
@click.pass_obj
@handle_errors
def get_provider(ctx: CliContext, provider_id: str) -> None:
    """Get provider details."""

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.get(f"{BASE}/{provider_id}")  # type: ignore[no-any-return]

    print_result(ctx, _async.run_async(_run()), title=f"Provider {provider_id}")


# ---- Configure ----


@provider_group.command("configure")
@click.argument("provider_id")
@click.option("--credentials", type=str, required=True, help="JSON string of credentials.")
@click.option("--config", "config_json", type=str, default="{}", help="JSON string of config.")
@click.pass_obj
@handle_errors
def configure_provider(
    ctx: CliContext,
    provider_id: str,
    credentials: str,
    config_json: str,
) -> None:
    """Configure a provider with credentials and settings."""
    import json as json_mod

    try:
        creds = json_mod.loads(credentials)
    except json_mod.JSONDecodeError as exc:
        raise click.BadParameter(f"Invalid JSON: {exc}", param_hint="--credentials") from None
    try:
        cfg = json_mod.loads(config_json)
    except json_mod.JSONDecodeError as exc:
        raise click.BadParameter(f"Invalid JSON: {exc}", param_hint="--config") from None

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.put(  # type: ignore[no-any-return]
                f"{BASE}/{provider_id}/configure",
                json={"credentials": creds, "config": cfg},
            )

    result = _async.run_async(_run())
    if ctx.json_mode:
        print_result(ctx, result)
    else:
        print_success(ctx, f"Provider {provider_id} configured.")


# ---- Enable / Disable ----


@provider_group.command("enable")
@click.argument("provider_id")
@click.pass_obj
@handle_errors
def enable_provider(ctx: CliContext, provider_id: str) -> None:
    """Enable a provider (tests connection first)."""

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.post(f"{BASE}/{provider_id}/enable")  # type: ignore[no-any-return]

    result = _async.run_async(_run())
    if ctx.json_mode:
        print_result(ctx, result)
    else:
        print_success(ctx, f"Provider {provider_id} enabled.")


@provider_group.command("disable")
@click.argument("provider_id")
@click.pass_obj
@handle_errors
def disable_provider(ctx: CliContext, provider_id: str) -> None:
    """Disable a provider."""

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.post(f"{BASE}/{provider_id}/disable")  # type: ignore[no-any-return]

    result = _async.run_async(_run())
    if ctx.json_mode:
        print_result(ctx, result)
    else:
        print_success(ctx, f"Provider {provider_id} disabled.")


# ---- Test connection ----


@provider_group.command("test")
@click.argument("provider_id")
@click.pass_obj
@handle_errors
def test_provider(ctx: CliContext, provider_id: str) -> None:
    """Test connection to a provider."""

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.post(f"{BASE}/{provider_id}/test")  # type: ignore[no-any-return]

    result = _async.run_async(_run())
    if ctx.json_mode:
        print_result(ctx, result)
    else:
        success = result.get("success", False)
        msg = result.get("message", "")
        latency = result.get("latency_ms")
        if success:
            detail = f"{msg} ({latency}ms)" if latency else msg
            print_success(ctx, f"Connection OK: {detail}")
        else:
            raise click.ClickException(f"Connection failed: {msg}")


# ---- Delete config ----


@provider_group.command("delete")
@click.argument("provider_id")
@click.pass_obj
@handle_errors
def delete_provider(ctx: CliContext, provider_id: str) -> None:
    """Remove provider configuration."""

    async def _run() -> None:
        async with ctx.api as api:
            await api.delete(f"{BASE}/{provider_id}")

    _async.run_async(_run())
    if ctx.json_mode:
        print_result(ctx, {"status": "deleted", "provider_id": provider_id})
    else:
        print_success(ctx, f"Provider {provider_id} configuration removed.")


# ---- Capabilities: storage / compute / regions ----


@provider_group.command("storage")
@click.argument("provider_id")
@click.pass_obj
@handle_errors
def provider_storage(ctx: CliContext, provider_id: str) -> None:
    """List storage options for a provider."""

    async def _run() -> list[dict[str, Any]]:
        async with ctx.api as api:
            return await api.get(  # type: ignore[no-any-return]
                f"{BASE}/{provider_id}/storage"
            )

    print_result(
        ctx,
        _async.run_async(_run()),
        columns=_STORAGE_COLUMNS,
        title=f"Provider {provider_id} — Storage",
    )


@provider_group.command("compute")
@click.argument("provider_id")
@click.option("--gpu-only", is_flag=True, default=False, help="Show only GPU instances.")
@click.pass_obj
@handle_errors
def provider_compute(ctx: CliContext, provider_id: str, gpu_only: bool) -> None:
    """List compute instances for a provider."""
    params: dict[str, Any] = {}
    if gpu_only:
        params["gpu_only"] = True

    async def _run() -> list[dict[str, Any]]:
        async with ctx.api as api:
            return await api.get(  # type: ignore[no-any-return]
                f"{BASE}/{provider_id}/compute", params=params or None
            )

    print_result(
        ctx,
        _async.run_async(_run()),
        columns=_COMPUTE_COLUMNS,
        title=f"Provider {provider_id} — Compute",
    )


@provider_group.command("regions")
@click.argument("provider_id")
@click.pass_obj
@handle_errors
def provider_regions(ctx: CliContext, provider_id: str) -> None:
    """List regions for a provider."""

    async def _run() -> list[dict[str, Any]]:
        async with ctx.api as api:
            return await api.get(  # type: ignore[no-any-return]
                f"{BASE}/{provider_id}/regions"
            )

    print_result(
        ctx,
        _async.run_async(_run()),
        columns=_REGION_COLUMNS,
        title=f"Provider {provider_id} — Regions",
    )
