"""Pydantic schemas for the runs entity API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CreateRunRequest(BaseModel):
    """POST /api/v1/runs — create a new run record."""

    id: str = Field(min_length=1, max_length=500)
    config: dict[str, Any] = Field(default_factory=dict)
    triggered_by: str = Field(default="", max_length=255)
    metrics: dict[str, Any] = Field(default_factory=dict)


class UpdateRunStatusRequest(BaseModel):
    """PATCH /api/v1/runs/{id}/status — update run status + optional fields."""

    status: str = Field(min_length=1)
    metrics: dict[str, Any] | None = None
    duration_seconds: float | None = None


class AddRunIORequest(BaseModel):
    """POST /api/v1/runs/{id}/io — add an input/output reference."""

    entity_id: str = Field(min_length=1, max_length=500)
    direction: str = Field(pattern="^(input|output)$")
