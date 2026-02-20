"""REST API for features — /api/v1/features/*."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: TC002

from artenic_ai_platform.entities.features.schemas import (  # noqa: TC001
    CreateFeatureRequest,
    UpdateFeatureRequest,
)
from artenic_ai_platform.entities.features.service import FeatureService

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from artenic_ai_platform.db.models import MLFeature

router = APIRouter(prefix="/api/v1/features", tags=["features"])


# ------------------------------------------------------------------
# Dependency helpers
# ------------------------------------------------------------------


async def _get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    factory = request.app.state.session_factory
    async with factory() as session:
        yield session


def _get_service(session: Annotated[AsyncSession, Depends(_get_db)]) -> FeatureService:
    return FeatureService(session)


Svc = Annotated[FeatureService, Depends(_get_service)]


# ------------------------------------------------------------------
# CRUD
# ------------------------------------------------------------------


@router.post("", status_code=201)
async def create_feature(body: CreateFeatureRequest, svc: Svc) -> dict[str, Any]:
    """Create a new feature schema."""
    data = body.model_dump(exclude={"id", "version"})
    if body.version is not None:
        data["version"] = body.version
    else:
        data["version"] = await svc.next_version(body.name)
    if "metadata" in data:
        data["metadata_"] = data.pop("metadata")

    record = await svc.create(body.id, data)
    return _feature_to_dict(record)


@router.get("")
async def list_features(
    svc: Svc,
    offset: int = 0,
    limit: int = 50,
    name: str | None = None,
) -> list[dict[str, Any]]:
    """List features with optional name filter."""
    filters: dict[str, Any] = {}
    if name is not None:
        filters["name"] = name
    records = await svc.list_all(offset=offset, limit=limit, filters=filters)
    return [_feature_to_dict(r) for r in records]


@router.get("/{feature_id}")
async def get_feature(feature_id: str, svc: Svc) -> dict[str, Any]:
    """Get feature details."""
    record = await svc.get(feature_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Feature not found")
    return _feature_to_dict(record)


@router.patch("/{feature_id}")
async def update_feature(
    feature_id: str, body: UpdateFeatureRequest, svc: Svc
) -> dict[str, Any]:
    """Update feature metadata."""
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if "metadata" in updates:
        updates["metadata_"] = updates.pop("metadata")
    try:
        record = await svc.update(feature_id, updates)
    except ValueError:
        raise HTTPException(status_code=404, detail="Feature not found") from None
    return _feature_to_dict(record)


@router.delete("/{feature_id}", status_code=204)
async def delete_feature(feature_id: str, svc: Svc) -> None:
    """Delete a feature schema."""
    try:
        await svc.delete(feature_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Feature not found") from None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _feature_to_dict(r: MLFeature) -> dict[str, Any]:
    return {
        "id": r.id,
        "name": r.name,
        "version": r.version,
        "metadata": r.metadata_,
        "created_at": r.created_at.isoformat() if r.created_at else "",
    }
