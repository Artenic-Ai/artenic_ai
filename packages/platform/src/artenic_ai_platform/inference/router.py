"""Inference REST API — /api/v1/services/*."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from artenic_ai_platform.inference.service import InferenceService

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/services", tags=["inference"])


# ------------------------------------------------------------------
# Request / Response schemas
# ------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Body for POST /{service}/predict."""

    data: dict[str, Any]
    model_id: str | None = None


class PredictBatchRequest(BaseModel):
    """Body for POST /{service}/predict_batch."""

    batch: list[dict[str, Any]]
    model_id: str | None = None


# ------------------------------------------------------------------
# Dependency helpers
# ------------------------------------------------------------------


async def _get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Yield a DB session from app state session_factory."""
    factory = request.app.state.session_factory
    async with factory() as session:
        yield session


DbSession = Annotated[AsyncSession, Depends(_get_db)]


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@router.post("/{service}/predict")
async def predict(
    service: str,
    body: PredictRequest,
    request: Request,
    session: DbSession,
) -> dict[str, Any]:
    """Run a single prediction on *service*."""
    svc = InferenceService(
        session,
        health_monitor=getattr(request.app.state, "health_monitor", None),
        ab_test_manager=getattr(request.app.state, "ab_test_manager", None),
        event_bus=getattr(request.app.state, "event_bus", None),
    )
    return await svc.predict(service, body.data, model_id=body.model_id)


@router.post("/{service}/predict_batch")
async def predict_batch(
    service: str,
    body: PredictBatchRequest,
    request: Request,
    session: DbSession,
) -> list[dict[str, Any]]:
    """Run batch predictions on *service*."""
    svc = InferenceService(
        session,
        health_monitor=getattr(request.app.state, "health_monitor", None),
        ab_test_manager=getattr(request.app.state, "ab_test_manager", None),
        event_bus=getattr(request.app.state, "event_bus", None),
    )
    return await svc.predict_batch(service, body.batch, model_id=body.model_id)
