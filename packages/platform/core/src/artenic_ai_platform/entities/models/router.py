"""REST API for models — /api/v1/models/*."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: TC002

from artenic_ai_platform.entities.datasets.storage import StorageBackend  # noqa: TC001
from artenic_ai_platform.entities.models.schemas import (  # noqa: TC001
    ChangeStageRequest,
    CreateModelRequest,
    UpdateModelRequest,
)
from artenic_ai_platform.entities.models.service import ModelService

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from artenic_ai_platform.db.models import MLModel

router = APIRouter(prefix="/api/v1/models", tags=["models"])


# ------------------------------------------------------------------
# Dependency helpers
# ------------------------------------------------------------------


async def _get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Yield a DB session from app state."""
    factory = request.app.state.session_factory
    async with factory() as session:
        yield session


def _get_storage(request: Request) -> StorageBackend:
    """Get storage backend from app state (shared with datasets for now)."""
    return request.app.state.dataset_storage  # type: ignore[no-any-return]


def _get_service(
    session: Annotated[AsyncSession, Depends(_get_db)],
    storage: Annotated[StorageBackend, Depends(_get_storage)],
) -> ModelService:
    return ModelService(session, storage)


Svc = Annotated[ModelService, Depends(_get_service)]


# ------------------------------------------------------------------
# CRUD
# ------------------------------------------------------------------


@router.post("", status_code=201)
async def create_model(body: CreateModelRequest, svc: Svc) -> dict[str, Any]:
    """Create a new model with a client-provided ID."""
    data = body.model_dump(exclude={"id", "version"})
    if body.version is not None:
        data["version"] = body.version
    else:
        data["version"] = await svc.next_version(body.name)

    if "metadata" in data:
        data["metadata_"] = data.pop("metadata")

    record = await svc.create(body.id, data)
    return _model_to_dict(record)


@router.get("")
async def list_models(
    svc: Svc,
    offset: int = 0,
    limit: int = 50,
    stage: str | None = None,
    framework: str | None = None,
    name: str | None = None,
) -> list[dict[str, Any]]:
    """List models with optional filters."""
    filters: dict[str, Any] = {}
    if stage is not None:
        filters["stage"] = stage
    if framework is not None:
        filters["framework"] = framework
    if name is not None:
        filters["name"] = name
    records = await svc.list_all(offset=offset, limit=limit, filters=filters)
    return [_model_to_dict(r) for r in records]


@router.get("/{model_id}")
async def get_model(model_id: str, svc: Svc) -> dict[str, Any]:
    """Get model details."""
    record = await svc.get(model_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return _model_to_dict(record)


@router.patch("/{model_id}")
async def update_model(model_id: str, body: UpdateModelRequest, svc: Svc) -> dict[str, Any]:
    """Update model metadata/metrics."""
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if "metadata" in updates:
        updates["metadata_"] = updates.pop("metadata")
    try:
        record = await svc.update(model_id, updates)
    except ValueError:
        raise HTTPException(status_code=404, detail="Model not found") from None
    return _model_to_dict(record)


@router.delete("/{model_id}", status_code=204)
async def delete_model(model_id: str, svc: Svc) -> None:
    """Delete a model and its artifact."""
    try:
        await svc.delete(model_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Model not found") from None


# ------------------------------------------------------------------
# Stage lifecycle
# ------------------------------------------------------------------


@router.patch("/{model_id}/stage")
async def change_stage(model_id: str, body: ChangeStageRequest, svc: Svc) -> dict[str, Any]:
    """Transition model stage (draft -> staging -> production -> retired)."""
    try:
        record = await svc.change_stage(model_id, body.stage)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return _model_to_dict(record)


# ------------------------------------------------------------------
# Artifact
# ------------------------------------------------------------------


@router.put("/{model_id}/artifact", status_code=200)
async def upload_artifact(model_id: str, file: UploadFile, svc: Svc) -> dict[str, Any]:
    """Upload a model artifact."""
    data = await file.read()
    filename = file.filename or "model.bin"
    try:
        record = await svc.upload_artifact(model_id, filename, data)
    except ValueError:
        raise HTTPException(status_code=404, detail="Model not found") from None
    return _model_to_dict(record)


@router.get("/{model_id}/artifact")
async def download_artifact(model_id: str, svc: Svc) -> Response:
    """Download the model artifact."""
    try:
        data = await svc.download_artifact(model_id)
    except (ValueError, FileNotFoundError) as exc:
        status = 404
        detail = str(exc) if isinstance(exc, FileNotFoundError) else "Model not found"
        raise HTTPException(status_code=status, detail=detail) from None
    return Response(content=data, media_type="application/octet-stream")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _model_to_dict(r: MLModel) -> dict[str, Any]:
    """Convert an MLModel to a JSON-safe dict."""
    return {
        "id": r.id,
        "name": r.name,
        "version": r.version,
        "framework": r.framework,
        "description": r.description,
        "metadata": r.metadata_,
        "metrics": r.metrics,
        "stage": r.stage,
        "artifact_path": r.artifact_path,
        "artifact_format": r.artifact_format,
        "artifact_size_bytes": r.artifact_size_bytes,
        "artifact_sha256": r.artifact_sha256,
        "created_at": r.created_at.isoformat() if r.created_at else "",
    }
