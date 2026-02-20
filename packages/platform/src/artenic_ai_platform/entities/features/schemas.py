"""Pydantic schemas for the features entity API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CreateFeatureRequest(BaseModel):
    """POST /api/v1/features — create a new feature schema."""

    id: str = Field(min_length=1, max_length=500)
    name: str = Field(min_length=1, max_length=255)
    metadata: dict[str, Any] = Field(default_factory=dict)
    version: int | None = Field(default=None)


class UpdateFeatureRequest(BaseModel):
    """PATCH /api/v1/features/{id} — update feature metadata."""

    metadata: dict[str, Any] | None = None
