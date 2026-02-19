"""REST API for dataset management — /api/v1/datasets/*."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: TC002

from artenic_ai_platform.datasets.service import DatasetService
from artenic_ai_platform.datasets.storage import StorageBackend  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from artenic_ai_platform.db.models import DatasetRecord

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])


# ------------------------------------------------------------------
# Request / response schemas
# ------------------------------------------------------------------


class CreateDatasetRequest(BaseModel):
    """Body for ``POST /api/v1/datasets``."""

    name: str = Field(min_length=1, max_length=255)
    format: str = Field(min_length=1, max_length=50)
    storage_backend: str = Field(default="filesystem", max_length=50)
    description: str = Field(default="", max_length=2000)
    source: str = Field(default="", max_length=500)
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, v: str) -> str:
        if not v.strip():
            msg = "name must not be blank"
            raise ValueError(msg)
        return v.strip()


class CreateDatasetResponse(BaseModel):
    """Response for ``POST /api/v1/datasets``."""

    id: str


class UpdateDatasetRequest(BaseModel):
    """Body for ``PATCH /api/v1/datasets/{dataset_id}``."""

    name: str | None = None
    description: str | None = None
    tags: dict[str, str] | None = None
    source: str | None = None


class CreateVersionRequest(BaseModel):
    """Body for ``POST /api/v1/datasets/{dataset_id}/versions``."""

    change_summary: str = ""


class AddLineageRequest(BaseModel):
    """Body for ``POST /api/v1/datasets/{dataset_id}/lineage``."""

    dataset_version: int
    entity_type: str
    entity_id: str
    role: str = "input"


class StorageOptionResponse(BaseModel):
    """A single storage option."""

    id: str
    label: str
    available: bool


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
# Storage options
# ------------------------------------------------------------------


@router.get("/storage-options", response_model=list[StorageOptionResponse])
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

    # Fetch all provider records from DB
    result = await session.execute(select(ProviderRecord))
    records = {r.id: r for r in result.scalars().all()}

    # For each catalog provider with storage capability, check DB status
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


@router.post("", response_model=CreateDatasetResponse, status_code=201)
async def create_dataset(body: CreateDatasetRequest, svc: Svc) -> CreateDatasetResponse:
    """Create a new dataset."""
    dataset_id = await svc.create(body.model_dump())
    return CreateDatasetResponse(id=dataset_id)


@router.get("")
async def list_datasets(svc: Svc, offset: int = 0, limit: int = 100) -> list[dict[str, Any]]:
    """List datasets with pagination."""
    records = await svc.list_all(offset=offset, limit=limit)
    return [_dataset_to_dict(r) for r in records]


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str, svc: Svc) -> dict[str, Any]:
    """Get dataset details."""
    record = await svc.get(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return _dataset_to_dict(record)


@router.patch("/{dataset_id}")
async def update_dataset(dataset_id: str, body: UpdateDatasetRequest, svc: Svc) -> dict[str, Any]:
    """Update dataset metadata."""
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
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

    # Validate extension against allowlist
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
        "filename": record.filename,
        "mime_type": record.mime_type,
        "size_bytes": record.size_bytes,
        "hash": record.hash,
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
            "version": f.version,
            "filename": f.filename,
            "mime_type": f.mime_type,
            "size_bytes": f.size_bytes,
            "hash": f.hash,
            "created_at": f.created_at.isoformat() if f.created_at else "",
        }
        for f in files
    ]


@router.get("/{dataset_id}/files/{filename:path}")
async def download_file(dataset_id: str, filename: str, svc: Svc) -> Response:
    """Download a file from a dataset."""
    try:
        data = await svc.download_file(dataset_id, filename)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found") from None
    return Response(content=data, media_type="application/octet-stream")


@router.delete("/{dataset_id}/files/{filename:path}", status_code=204)
async def delete_file_endpoint(dataset_id: str, filename: str, svc: Svc) -> None:
    """Delete a file from a dataset."""
    try:
        await svc.delete_file(dataset_id, filename)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found") from None


# ------------------------------------------------------------------
# Versioning
# ------------------------------------------------------------------


@router.post("/{dataset_id}/versions", status_code=201)
async def create_version(dataset_id: str, body: CreateVersionRequest, svc: Svc) -> dict[str, Any]:
    """Create a new version snapshot."""
    try:
        record = await svc.create_version(dataset_id, body.change_summary)
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found") from None
    return {
        "id": record.id,
        "dataset_id": record.dataset_id,
        "version": record.version,
        "hash": record.hash,
        "size_bytes": record.size_bytes,
        "num_files": record.num_files,
        "change_summary": record.change_summary,
        "created_at": record.created_at.isoformat() if record.created_at else "",
    }


@router.get("/{dataset_id}/versions")
async def list_versions(dataset_id: str, svc: Svc) -> list[dict[str, Any]]:
    """List all versions of a dataset."""
    if await svc.get(dataset_id) is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    versions = await svc.list_versions(dataset_id)
    return [
        {
            "id": v.id,
            "dataset_id": v.dataset_id,
            "version": v.version,
            "hash": v.hash,
            "size_bytes": v.size_bytes,
            "num_files": v.num_files,
            "num_records": v.num_records,
            "change_summary": v.change_summary,
            "created_at": v.created_at.isoformat() if v.created_at else "",
        }
        for v in versions
    ]


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
# Lineage
# ------------------------------------------------------------------


@router.get("/{dataset_id}/lineage")
async def get_lineage(dataset_id: str, svc: Svc) -> list[dict[str, Any]]:
    """Get lineage records for a dataset."""
    if await svc.get(dataset_id) is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    records = await svc.get_lineage(dataset_id)
    return [
        {
            "id": r.id,
            "dataset_id": r.dataset_id,
            "dataset_version": r.dataset_version,
            "entity_type": r.entity_type,
            "entity_id": r.entity_id,
            "role": r.role,
            "created_at": r.created_at.isoformat() if r.created_at else "",
        }
        for r in records
    ]


@router.post("/{dataset_id}/lineage", status_code=201)
async def add_lineage(dataset_id: str, body: AddLineageRequest, svc: Svc) -> dict[str, Any]:
    """Add a lineage record linking dataset to model or training job."""
    try:
        record = await svc.add_lineage(
            dataset_id,
            body.dataset_version,
            body.entity_type,
            body.entity_id,
            body.role,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found") from None
    return {
        "id": record.id,
        "dataset_id": record.dataset_id,
        "dataset_version": record.dataset_version,
        "entity_type": record.entity_type,
        "entity_id": record.entity_id,
        "role": record.role,
    }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _dataset_to_dict(r: DatasetRecord) -> dict[str, Any]:
    """Convert a DatasetRecord to a JSON-safe dict."""
    return {
        "id": r.id,
        "name": r.name,
        "description": r.description,
        "format": r.format,
        "storage_backend": r.storage_backend,
        "source": r.source,
        "tags": r.tags or {},
        "current_version": r.current_version,
        "total_size_bytes": r.total_size_bytes,
        "total_files": r.total_files,
        "schema_info": r.schema_info,
        "created_at": r.created_at.isoformat() if r.created_at else "",
        "updated_at": r.updated_at.isoformat() if r.updated_at else None,
    }
