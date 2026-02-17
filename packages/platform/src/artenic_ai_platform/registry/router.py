"""REST API for the model registry — /api/v1/models/*."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from artenic_ai_platform.registry.service import ModelRegistry

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

router = APIRouter(prefix="/api/v1/models", tags=["registry"])


# ------------------------------------------------------------------
# Request / response schemas
# ------------------------------------------------------------------


class RegisterModelRequest(BaseModel):
    """Body for ``POST /api/v1/models``."""

    name: str
    version: str
    model_type: str
    framework: str = "custom"
    description: str = ""
    tags: dict[str, str] = Field(default_factory=dict)
    input_features: list[dict[str, Any]] = Field(default_factory=list)
    output_schema: dict[str, Any] = Field(default_factory=dict)


class RegisterModelResponse(BaseModel):
    """Response for ``POST /api/v1/models``."""

    model_id: str


class PromoteRequest(BaseModel):
    """Body for ``POST /api/v1/models/{model_id}/promote``."""

    version: str


# ------------------------------------------------------------------
# Dependency helpers
# ------------------------------------------------------------------


async def _get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Yield a DB session from app state session_factory."""
    factory = request.app.state.session_factory
    async with factory() as session:
        yield session


def _get_mlflow(request: Request) -> Any:
    """Get MLflow tracker from app state (may be None)."""
    return getattr(request.app.state, "mlflow", None)


DbSession = Annotated[AsyncSession, Depends(_get_db)]


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@router.post("", response_model=RegisterModelResponse, status_code=201)
async def register_model(
    body: RegisterModelRequest,
    request: Request,
    session: DbSession,
) -> RegisterModelResponse:
    """Register a new model."""
    registry = ModelRegistry(session, _get_mlflow(request))
    model_id = await registry.register(body.model_dump())
    return RegisterModelResponse(model_id=model_id)


@router.get("")
async def list_models(
    request: Request,
    session: DbSession,
) -> list[dict[str, Any]]:
    """List all registered models."""
    registry = ModelRegistry(session, _get_mlflow(request))
    return await registry.list_all()


@router.get("/{model_id}")
async def get_model(
    model_id: str,
    request: Request,
    session: DbSession,
) -> dict[str, Any]:
    """Get a model by ID."""
    registry = ModelRegistry(session, _get_mlflow(request))
    return await registry.get(model_id)


@router.post("/{model_id}/promote", status_code=204)
async def promote_model(
    model_id: str,
    body: PromoteRequest,
    request: Request,
    session: DbSession,
) -> None:
    """Promote a model version to production."""
    registry = ModelRegistry(session, _get_mlflow(request))
    await registry.promote(model_id, body.version)


@router.post("/{model_id}/retire", status_code=204)
async def retire_model(
    model_id: str,
    request: Request,
    session: DbSession,
) -> None:
    """Archive a model."""
    registry = ModelRegistry(session, _get_mlflow(request))
    await registry.retire(model_id)
