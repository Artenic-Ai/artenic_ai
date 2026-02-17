"""Health-check router.

Exposes liveness, readiness, and detailed health endpoints consumed by
container orchestrators and monitoring systems.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

import artenic_ai_platform as _platform

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health")
async def liveness() -> dict[str, str]:
    """Liveness probe — always returns healthy if the process is up."""
    return {"status": "healthy"}


@router.get("/health/ready")
async def readiness(request: Request) -> JSONResponse:
    """Readiness probe — verifies database connectivity."""
    try:
        engine = request.app.state.engine
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return JSONResponse(
            status_code=200,
            content={
                "status": "ready",
                "database": "connected",
            },
        )
    except Exception:
        logger.warning(
            "Readiness check failed — database unreachable",
            exc_info=True,
        )
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "database": "disconnected",
            },
        )


@router.get("/health/detailed")
async def detailed(request: Request) -> JSONResponse:
    """Detailed health check with per-component status."""
    components: dict[str, Any] = {
        "version": _platform.__version__,
    }

    # Database ---------------------------------------------------------
    try:
        engine = request.app.state.engine
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        components["database"] = "connected"
    except Exception:
        logger.warning(
            "Detailed health: database unreachable",
            exc_info=True,
        )
        components["database"] = "disconnected"

    all_healthy = components.get("database") == "connected"
    status_code = 200 if all_healthy else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if all_healthy else "degraded",
            "components": components,
        },
    )
