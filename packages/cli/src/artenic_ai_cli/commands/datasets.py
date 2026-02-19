"""``artenic dataset`` — dataset management."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from artenic_ai_cli import _async
from artenic_ai_cli._output import print_result, print_success
from artenic_ai_cli.main import handle_errors

if TYPE_CHECKING:
    from artenic_ai_cli._context import CliContext

_DATASET_COLUMNS: list[tuple[str, str]] = [
    ("ID", "id"),
    ("Name", "name"),
    ("Format", "format"),
    ("Created", "created_at"),
    ("Size", "size"),
]

_FILE_COLUMNS: list[tuple[str, str]] = [
    ("Filename", "filename"),
    ("Size", "size"),
    ("Uploaded", "uploaded_at"),
]

_VERSION_COLUMNS: list[tuple[str, str]] = [
    ("Version", "version"),
    ("Message", "message"),
    ("Created", "created_at"),
]


@click.group("dataset")
def dataset_group() -> None:
    """Manage datasets."""


# ---- Core CRUD ----


@dataset_group.command("list")
@click.pass_obj
@handle_errors
def list_datasets(ctx: CliContext) -> None:
    """List all datasets."""

    async def _run() -> list[dict[str, Any]]:
        async with ctx.api as api:
            return await api.get("/api/v1/datasets")  # type: ignore[no-any-return]

    print_result(ctx, _async.run_async(_run()), columns=_DATASET_COLUMNS, title="Datasets")


@dataset_group.command("create")
@click.option("--name", required=True, help="Dataset name.")
@click.option("--format", "fmt", required=True, help="Data format (e.g. csv, parquet, json).")
@click.option("--description", default=None, help="Short description.")
@click.pass_obj
@handle_errors
def create_dataset(
    ctx: CliContext,
    name: str,
    fmt: str,
    description: str | None,
) -> None:
    """Create a new dataset."""
    body: dict[str, Any] = {"name": name, "format": fmt}
    if description:
        body["description"] = description

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.post("/api/v1/datasets", json=body)  # type: ignore[no-any-return]

    result = _async.run_async(_run())
    if ctx.json_mode:
        print_result(ctx, result)
    else:
        print_success(ctx, f"Dataset created: {result.get('dataset_id', result)}")


@dataset_group.command("get")
@click.argument("dataset_id")
@click.pass_obj
@handle_errors
def get_dataset(ctx: CliContext, dataset_id: str) -> None:
    """Get dataset details by ID."""

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.get(f"/api/v1/datasets/{dataset_id}")  # type: ignore[no-any-return]

    print_result(ctx, _async.run_async(_run()), title=f"Dataset {dataset_id}")


@dataset_group.command("update")
@click.argument("dataset_id")
@click.option("--name", default=None, help="New name.")
@click.option("--description", default=None, help="New description.")
@click.pass_obj
@handle_errors
def update_dataset(
    ctx: CliContext,
    dataset_id: str,
    name: str | None,
    description: str | None,
) -> None:
    """Update dataset metadata."""
    body: dict[str, Any] = {}
    if name:
        body["name"] = name
    if description:
        body["description"] = description

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.patch(  # type: ignore[no-any-return]
                f"/api/v1/datasets/{dataset_id}", json=body
            )

    result = _async.run_async(_run())
    if ctx.json_mode:
        print_result(ctx, result)
    else:
        print_success(ctx, f"Dataset {dataset_id} updated.")


@dataset_group.command("delete")
@click.argument("dataset_id")
@click.pass_obj
@handle_errors
def delete_dataset(ctx: CliContext, dataset_id: str) -> None:
    """Delete a dataset."""

    async def _run() -> None:
        async with ctx.api as api:
            await api.delete(f"/api/v1/datasets/{dataset_id}")

    _async.run_async(_run())
    if ctx.json_mode:
        print_result(ctx, {"status": "deleted", "dataset_id": dataset_id})
    else:
        print_success(ctx, f"Dataset {dataset_id} deleted.")


# ---- File operations ----


@dataset_group.command("upload")
@click.argument("dataset_id")
@click.argument("filepath", type=click.Path(exists=True))
@click.pass_obj
@handle_errors
def upload_file(ctx: CliContext, dataset_id: str, filepath: str) -> None:
    """Upload a file to a dataset."""
    path = Path(filepath)
    data = path.read_bytes()
    name = path.name
    mime = mimetypes.guess_type(name)[0] or "application/octet-stream"

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.upload_file(  # type: ignore[no-any-return]
                f"/api/v1/datasets/{dataset_id}/files", name, data, mime
            )

    result = _async.run_async(_run())
    if ctx.json_mode:
        print_result(ctx, result)
    else:
        print_success(ctx, f"Uploaded {name} to dataset {dataset_id}.")


@dataset_group.command("download")
@click.argument("dataset_id")
@click.argument("filename")
@click.option("-o", "--output", default=None, help="Output file path (default: ./<filename>).")
@click.pass_obj
@handle_errors
def download_file(
    ctx: CliContext,
    dataset_id: str,
    filename: str,
    output: str | None,
) -> None:
    """Download a file from a dataset."""

    async def _run() -> bytes:
        async with ctx.api as api:
            return await api.download_bytes(
                f"/api/v1/datasets/{dataset_id}/files/{filename}"
            )

    content = _async.run_async(_run())
    dest = Path(output) if output else Path(filename)
    dest.write_bytes(content)
    if ctx.json_mode:
        print_result(ctx, {"status": "downloaded", "path": str(dest)})
    else:
        print_success(ctx, f"Downloaded {filename} to {dest}.")


@dataset_group.command("files")
@click.argument("dataset_id")
@click.pass_obj
@handle_errors
def list_files(ctx: CliContext, dataset_id: str) -> None:
    """List files in a dataset."""

    async def _run() -> list[dict[str, Any]]:
        async with ctx.api as api:
            return await api.get(  # type: ignore[no-any-return]
                f"/api/v1/datasets/{dataset_id}/files"
            )

    print_result(
        ctx,
        _async.run_async(_run()),
        columns=_FILE_COLUMNS,
        title=f"Dataset {dataset_id} — Files",
    )


# ---- Analytics ----


@dataset_group.command("stats")
@click.argument("dataset_id")
@click.pass_obj
@handle_errors
def dataset_stats(ctx: CliContext, dataset_id: str) -> None:
    """Show dataset statistics."""

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.get(  # type: ignore[no-any-return]
                f"/api/v1/datasets/{dataset_id}/stats"
            )

    print_result(ctx, _async.run_async(_run()), title=f"Dataset {dataset_id} — Stats")


@dataset_group.command("preview")
@click.argument("dataset_id")
@click.option("--limit", type=int, default=20, help="Number of rows to preview.")
@click.pass_obj
@handle_errors
def preview_dataset(ctx: CliContext, dataset_id: str, limit: int) -> None:
    """Preview tabular data from a dataset."""

    async def _run() -> list[dict[str, Any]]:
        async with ctx.api as api:
            return await api.get(  # type: ignore[no-any-return]
                f"/api/v1/datasets/{dataset_id}/preview",
                params={"limit": limit},
            )

    print_result(
        ctx,
        _async.run_async(_run()),
        title=f"Dataset {dataset_id} — Preview (limit={limit})",
    )


# ---- Versioning ----


@dataset_group.group("version")
def version_group() -> None:
    """Manage dataset versions."""


@version_group.command("list")
@click.argument("dataset_id")
@click.pass_obj
@handle_errors
def list_versions(ctx: CliContext, dataset_id: str) -> None:
    """List dataset versions."""

    async def _run() -> list[dict[str, Any]]:
        async with ctx.api as api:
            return await api.get(  # type: ignore[no-any-return]
                f"/api/v1/datasets/{dataset_id}/versions"
            )

    print_result(
        ctx,
        _async.run_async(_run()),
        columns=_VERSION_COLUMNS,
        title=f"Dataset {dataset_id} — Versions",
    )


@version_group.command("create")
@click.argument("dataset_id")
@click.option("-m", "--message", default=None, help="Version message.")
@click.pass_obj
@handle_errors
def create_version(ctx: CliContext, dataset_id: str, message: str | None) -> None:
    """Create a new dataset version."""
    body: dict[str, Any] = {}
    if message:
        body["change_summary"] = message

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.post(  # type: ignore[no-any-return]
                f"/api/v1/datasets/{dataset_id}/versions", json=body
            )

    result = _async.run_async(_run())
    if ctx.json_mode:
        print_result(ctx, result)
    else:
        print_success(ctx, f"Version created: {result.get('version', result)}")


# ---- Lineage ----


@dataset_group.command("lineage")
@click.argument("dataset_id")
@click.pass_obj
@handle_errors
def dataset_lineage(ctx: CliContext, dataset_id: str) -> None:
    """Show dataset lineage."""

    async def _run() -> dict[str, Any]:
        async with ctx.api as api:
            return await api.get(  # type: ignore[no-any-return]
                f"/api/v1/datasets/{dataset_id}/lineage"
            )

    print_result(ctx, _async.run_async(_run()), title=f"Dataset {dataset_id} — Lineage")
