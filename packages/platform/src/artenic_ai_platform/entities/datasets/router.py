"""REST API for datasets — /api/v1/datasets/*."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: TC002

from artenic_ai_platform.entities.datasets.schemas import (  # noqa: TC001
    ChangeStatusRequest,
    CreateDatasetRequest,
    CreateVersionRequest,
    UpdateDatasetRequest,
)
from artenic_ai_platform.entities.datasets.service import DatasetService
from artenic_ai_platform.entities.datasets.storage import StorageBackend  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from artenic_ai_platform.db.models import MLDataset

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])


# ------------------------------------------------------------------
# Dependency helpers
# ------------------------------------------------------------------


async def _get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Yield a DB session from app state."""
    factory = request.app.state.session_factory
    async with factory() as session:
        yield session


def _get_storage(request: Request) -> StorageBackend:
    """Get dataset storage backend from app state."""
    return request.app.state.dataset_storage  # type: ignore[no-any-return]


def _get_service(
    session: Annotated[AsyncSession, Depends(_get_db)],
    storage: Annotated[StorageBackend, Depends(_get_storage)],
) -> DatasetService:
    return DatasetService(session, storage)


Svc = Annotated[DatasetService, Depends(_get_service)]


# ------------------------------------------------------------------
# Storage options (must be before /{dataset_id} to avoid path capture)
# ------------------------------------------------------------------


@router.get("/storage-options")
async def storage_options(
    session: Annotated[AsyncSession, Depends(_get_db)],
) -> list[dict[str, Any]]:
    """List available storage backends based on configured providers."""
    from sqlalchemy import select

    from artenic_ai_platform.db.models import ProviderRecord
    from artenic_ai_platform.providers_hub.catalog import PROVIDER_CATALOG

    options: list[dict[str, Any]] = [
        {"id": "filesystem", "label": "Self-hosted (local)", "available": True},
    ]

    result = await session.execute(select(ProviderRecord))
    records = {r.id: r for r in result.scalars().all()}

    for defn in PROVIDER_CATALOG.values():
        has_storage = any(c.type == "storage" for c in defn.capabilities)
        if not has_storage:
            continue
        rec = records.get(defn.id)
        available = rec is not None and rec.enabled
        options.append(
            {
                "id": defn.id,
                "label": f"{defn.display_name} Object Storage",
                "available": available,
            }
        )

    return options


# ------------------------------------------------------------------
# CRUD
# ------------------------------------------------------------------


@router.post("", status_code=201)
async def create_dataset(body: CreateDatasetRequest, svc: Svc) -> dict[str, Any]:
    """Create a new dataset with a client-provided ID."""
    data = body.model_dump(exclude={"id", "version"})
    if body.version is not None:
        data["version"] = body.version
    else:
        data["version"] = await svc.next_version(body.name)

    # Map 'metadata' to 'metadata_' for the ORM
    if "metadata" in data:
        data["metadata_"] = data.pop("metadata")

    record = await svc.create(body.id, data)
    return _dataset_to_dict(record)


@router.get("")
async def list_datasets(
    svc: Svc,
    offset: int = 0,
    limit: int = 50,
    status: str | None = None,
    name: str | None = None,
) -> list[dict[str, Any]]:
    """List datasets with optional filters."""
    filters: dict[str, Any] = {}
    if status is not None:
        filters["status"] = status
    if name is not None:
        filters["name"] = name
    records = await svc.list_all(offset=offset, limit=limit, filters=filters)
    return [_dataset_to_dict(r) for r in records]


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str, svc: Svc) -> dict[str, Any]:
    """Get dataset details."""
    record = await svc.get(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return _dataset_to_dict(record)


@router.patch("/{dataset_id}")
async def update_dataset(
    dataset_id: str, body: UpdateDatasetRequest, svc: Svc
) -> dict[str, Any]:
    """Update dataset metadata/description."""
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if "metadata" in updates:
        updates["metadata_"] = updates.pop("metadata")
    try:
        record = await svc.update(dataset_id, updates)
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found") from None
    return _dataset_to_dict(record)


@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(dataset_id: str, svc: Svc) -> None:
    """Delete a dataset and all its files."""
    try:
        await svc.delete(dataset_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found") from None


# ------------------------------------------------------------------
# Status lifecycle
# ------------------------------------------------------------------


@router.patch("/{dataset_id}/status")
async def change_status(
    dataset_id: str, body: ChangeStatusRequest, svc: Svc
) -> dict[str, Any]:
    """Transition dataset status (created -> active -> archived)."""
    try:
        record = await svc.change_status(dataset_id, body.status)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return _dataset_to_dict(record)


# ------------------------------------------------------------------
# Versioning
# ------------------------------------------------------------------


@router.post("/{dataset_id}/versions", status_code=201)
async def create_version(
    dataset_id: str, body: CreateVersionRequest, svc: Svc
) -> dict[str, Any]:
    """Create a new version of this dataset (new row, incremented version)."""
    try:
        source = await svc.get_or_raise(dataset_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found") from None

    next_ver = await svc.next_version(source.name)
    new_id = f"{source.name}_v{next_ver}"
    data = {
        "name": source.name,
        "version": next_ver,
        "format": source.format,
        "description": body.change_summary or source.description,
        "metadata_": source.metadata_,
        "status": "created",
    }
    record = await svc.create(new_id, data)
    return _dataset_to_dict(record)


# ------------------------------------------------------------------
# Files
# ------------------------------------------------------------------


@router.post("/{dataset_id}/files", status_code=201)
async def upload_file(
    dataset_id: str, file: UploadFile, request: Request, svc: Svc
) -> dict[str, Any]:
    """Upload a file to a dataset."""
    settings = request.app.state.settings
    max_bytes = settings.dataset.max_upload_size_mb * 1024 * 1024

    data = await file.read()
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum upload size ({settings.dataset.max_upload_size_mb} MB)",
        )

    filename = file.filename or "unnamed"

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    allowed = settings.dataset.allowed_extensions
    if allowed and ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"File extension '.{ext}' not allowed. Accepted: {', '.join(allowed)}",
        )

    try:
        record = await svc.upload_file(dataset_id, filename, data)
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found") from None
    return {
        "id": record.id,
        "dataset_id": record.dataset_id,
        "path": record.path,
        "sha256": record.sha256,
        "size_bytes": record.size_bytes,
        "num_records": record.num_records,
        "storage_path": record.storage_path,
    }


@router.get("/{dataset_id}/files")
async def list_files(dataset_id: str, svc: Svc) -> list[dict[str, Any]]:
    """List files in a dataset."""
    if await svc.get(dataset_id) is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    files = await svc.list_files(dataset_id)
    return [
        {
            "id": f.id,
            "dataset_id": f.dataset_id,
            "path": f.path,
            "sha256": f.sha256,
            "size_bytes": f.size_bytes,
            "num_records": f.num_records,
            "storage_path": f.storage_path,
        }
        for f in files
    ]


@router.get("/{dataset_id}/files/{path:path}")
async def download_file(dataset_id: str, path: str, svc: Svc) -> Response:
    """Download a file from a dataset."""
    try:
        data = await svc.download_file(dataset_id, path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found") from None
    return Response(content=data, media_type="application/octet-stream")


@router.delete("/{dataset_id}/files/{path:path}", status_code=204)
async def delete_file_endpoint(dataset_id: str, path: str, svc: Svc) -> None:
    """Delete a file from a dataset."""
    try:
        await svc.delete_file(dataset_id, path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found") from None


# ------------------------------------------------------------------
# Stats & Preview
# ------------------------------------------------------------------


@router.get("/{dataset_id}/stats")
async def get_stats(dataset_id: str, svc: Svc) -> dict[str, Any]:
    """Compute and return dataset statistics."""
    try:
        return await svc.compute_stats(dataset_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found") from None


@router.get("/{dataset_id}/preview")
async def get_preview(dataset_id: str, svc: Svc, limit: int = 50) -> dict[str, Any]:
    """Return first N rows for tabular datasets."""
    try:
        return await svc.preview(dataset_id, limit=limit)
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found") from None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _dataset_to_dict(r: MLDataset) -> dict[str, Any]:
    """Convert an MLDataset to a JSON-safe dict."""
    return {
        "id": r.id,
        "name": r.name,
        "version": r.version,
        "format": r.format,
        "description": r.description,
        "metadata": r.metadata_,
        "status": r.status,
        "total_size_bytes": r.total_size_bytes,
        "created_at": r.created_at.isoformat() if r.created_at else "",
    }
