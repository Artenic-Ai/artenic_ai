"""Pydantic schemas for the ensembles entity API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CreateEnsembleRequest(BaseModel):
    """POST /api/v1/ensembles — create a new ensemble."""

    id: str = Field(min_length=1, max_length=500)
    name: str = Field(min_length=1, max_length=255)
    strategy_type: str = Field(min_length=1, max_length=100)
    metadata: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    model_ids: list[str] = Field(default_factory=list)
    version: int | None = Field(default=None)


class UpdateEnsembleRequest(BaseModel):
    """PATCH /api/v1/ensembles/{id} — update ensemble fields."""

    metadata: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None


class ChangeStageRequest(BaseModel):
    """PATCH /api/v1/ensembles/{id}/stage — change lifecycle stage."""

    stage: str = Field(min_length=1)


class AddModelRequest(BaseModel):
    """POST /api/v1/ensembles/{id}/models — add a model reference."""

    model_id: str = Field(min_length=1, max_length=500)
