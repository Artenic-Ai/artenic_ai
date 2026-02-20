"""Shared Pydantic schemas for the entities module."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DatasetStatus(StrEnum):
    """Lifecycle states for datasets."""

    CREATED = "created"
    ACTIVE = "active"
    ARCHIVED = "archived"


class ModelStage(StrEnum):
    """Lifecycle stages for models."""

    DRAFT = "draft"
    STAGING = "staging"
    PRODUCTION = "production"
    RETIRED = "retired"


class RunStatus(StrEnum):
    """Status of a training/experiment run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EnsembleStage(StrEnum):
    """Lifecycle stages for ensembles."""

    STAGING = "staging"
    PRODUCTION = "production"
    RETIRED = "retired"


# ---------------------------------------------------------------------------
# Lifecycle transition rules
# ---------------------------------------------------------------------------

DATASET_TRANSITIONS: dict[str, set[str]] = {
    DatasetStatus.CREATED: {DatasetStatus.ACTIVE, DatasetStatus.ARCHIVED},
    DatasetStatus.ACTIVE: {DatasetStatus.ARCHIVED},
    DatasetStatus.ARCHIVED: set(),
}

MODEL_TRANSITIONS: dict[str, set[str]] = {
    ModelStage.DRAFT: {ModelStage.STAGING},
    ModelStage.STAGING: {ModelStage.PRODUCTION, ModelStage.RETIRED},
    ModelStage.PRODUCTION: {ModelStage.RETIRED},
    ModelStage.RETIRED: set(),
}

ENSEMBLE_TRANSITIONS: dict[str, set[str]] = {
    EnsembleStage.STAGING: {EnsembleStage.PRODUCTION, EnsembleStage.RETIRED},
    EnsembleStage.PRODUCTION: {EnsembleStage.RETIRED},
    EnsembleStage.RETIRED: set(),
}

RUN_TRANSITIONS: dict[str, set[str]] = {
    RunStatus.PENDING: {RunStatus.RUNNING, RunStatus.FAILED},
    RunStatus.RUNNING: {RunStatus.COMPLETED, RunStatus.FAILED},
    RunStatus.COMPLETED: set(),
    RunStatus.FAILED: set(),
}


# ---------------------------------------------------------------------------
# Shared request/response models
# ---------------------------------------------------------------------------


class PaginationParams(BaseModel):
    """Reusable pagination query parameters."""

    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=500)
