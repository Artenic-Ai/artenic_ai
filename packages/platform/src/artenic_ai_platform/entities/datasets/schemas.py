"""Pydantic request/response schemas for datasets."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CreateDatasetRequest(BaseModel):
    """Body for POST /api/v1/datasets."""

    id: str = Field(min_length=1, max_length=500)
    name: str = Field(min_length=1, max_length=255)
    format: str = Field(min_length=1, max_length=50)
    description: str = Field(default="", max_length=2000)
    metadata: dict[str, Any] = Field(default_factory=dict)
    version: int | None = Field(
        default=None,
        description="If omitted, auto-increments per name.",
    )


class UpdateDatasetRequest(BaseModel):
    """Body for PATCH /api/v1/datasets/{id}."""

    description: str | None = None
    metadata: dict[str, Any] | None = None


class ChangeStatusRequest(BaseModel):
    """Body for PATCH /api/v1/datasets/{id}/status."""

    status: str = Field(min_length=1)


class CreateVersionRequest(BaseModel):
    """Body for POST /api/v1/datasets/{id}/versions."""

    change_summary: str = Field(default="")
