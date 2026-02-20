"""Pydantic request/response schemas for models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CreateModelRequest(BaseModel):
    """Body for POST /api/v1/models."""

    id: str = Field(min_length=1, max_length=500)
    name: str = Field(min_length=1, max_length=255)
    framework: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=2000)
    metadata: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    version: int | None = Field(
        default=None,
        description="If omitted, auto-increments per name.",
    )


class UpdateModelRequest(BaseModel):
    """Body for PATCH /api/v1/models/{id}."""

    description: str | None = None
    metadata: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None


class ChangeStageRequest(BaseModel):
    """Body for PATCH /api/v1/models/{id}/stage."""

    stage: str = Field(min_length=1)
