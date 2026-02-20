"""Pydantic schemas for the lineage entity API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AddLineageRequest(BaseModel):
    """POST /api/v1/lineage — add a lineage link."""

    source_id: str = Field(min_length=1, max_length=500)
    target_id: str = Field(min_length=1, max_length=500)
    relation_type: str = Field(min_length=1, max_length=100)
