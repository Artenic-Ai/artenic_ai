"""Provider Hub REST API — 12 endpoints for cloud provider management."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from artenic_ai_platform.providers_hub.schemas import (
    ComputeInstance,
    ConfigureProviderRequest,
    ConnectionTestResult,
    ProviderDetail,
    ProviderRegion,
    ProviderSummary,
    StorageOption,
)
from artenic_ai_platform.providers_hub.service import ProviderService

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/providers", tags=["providers"])


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


async def _get_service(request: Request) -> AsyncGenerator[ProviderService, None]:
    factory = request.app.state.session_factory
    secret_manager = request.app.state.secret_manager
    async with factory() as session:
        yield ProviderService(session, secret_manager)


Svc = Annotated[ProviderService, Depends(_get_service)]


# ---------------------------------------------------------------------------
# Provider CRUD
# ---------------------------------------------------------------------------


@router.get("/", response_model=list[ProviderSummary])
async def list_providers(svc: Svc) -> list[ProviderSummary]:
    """List all known providers with their current activation state."""
    return await svc.list_providers()


@router.get(
    "/capabilities/storage",
    response_model=list[StorageOption],
)
async def list_all_storage(svc: Svc) -> list[StorageOption]:
    """Aggregate storage options from all active providers."""
    return await svc.list_all_storage_options()


@router.get(
    "/capabilities/compute",
    response_model=list[ComputeInstance],
)
async def list_all_compute(
    svc: Svc,
    region: str | None = Query(default=None),
    gpu_only: bool = Query(default=False),
) -> list[ComputeInstance]:
    """Aggregate compute instances from all active providers."""
    return await svc.list_all_compute_instances(region=region, gpu_only=gpu_only)


# IMPORTANT: Routes with static paths (/capabilities/*) MUST be defined
# above this dynamic /{provider_id} route to avoid shadowing.
@router.get("/{provider_id}", response_model=ProviderDetail)
async def get_provider(provider_id: str, svc: Svc) -> ProviderDetail:
    """Get full detail for a single provider."""
    try:
        return await svc.get_provider(provider_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put(
    "/{provider_id}/configure",
    response_model=ProviderDetail,
)
async def configure_provider(
    provider_id: str,
    body: ConfigureProviderRequest,
    svc: Svc,
) -> ProviderDetail:
    """Store credentials and non-sensitive config for a provider."""
    try:
        return await svc.configure_provider(
            provider_id,
            credentials=body.credentials,
            config=body.config,
        )
    except ValueError as exc:
        msg = str(exc)
        status = 400 if "Missing required" in msg else 404
        raise HTTPException(status_code=status, detail=msg) from exc


@router.post("/{provider_id}/enable", response_model=ProviderDetail)
async def enable_provider(provider_id: str, svc: Svc) -> ProviderDetail:
    """Enable a provider (runs connection test first)."""
    try:
        return await svc.enable_provider(provider_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{provider_id}/disable", response_model=ProviderDetail)
async def disable_provider(provider_id: str, svc: Svc) -> ProviderDetail:
    """Disable a provider."""
    try:
        return await svc.disable_provider(provider_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/{provider_id}/test",
    response_model=ConnectionTestResult,
)
async def test_provider(provider_id: str, svc: Svc) -> ConnectionTestResult:
    """Run a connection test for a configured provider."""
    try:
        return await svc.test_provider(provider_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/{provider_id}", status_code=204)
async def delete_provider(provider_id: str, svc: Svc) -> None:
    """Remove all stored credentials and config for a provider."""
    await svc.delete_provider_config(provider_id)


# ---------------------------------------------------------------------------
# Per-provider capability endpoints (live data from provider API)
# ---------------------------------------------------------------------------


@router.get(
    "/{provider_id}/storage",
    response_model=list[StorageOption],
)
async def list_provider_storage(
    provider_id: str,
    svc: Svc,
) -> list[StorageOption]:
    """List storage options from a single active provider."""
    try:
        return await svc.list_storage_for_provider(provider_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get(
    "/{provider_id}/compute",
    response_model=list[ComputeInstance],
)
async def list_provider_compute(
    provider_id: str,
    svc: Svc,
    region: str | None = Query(default=None),
    gpu_only: bool = Query(default=False),
) -> list[ComputeInstance]:
    """List compute instances from a single active provider."""
    try:
        return await svc.list_compute_for_provider(
            provider_id,
            region=region,
            gpu_only=gpu_only,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get(
    "/{provider_id}/regions",
    response_model=list[ProviderRegion],
)
async def list_provider_regions(
    provider_id: str,
    svc: Svc,
) -> list[ProviderRegion]:
    """List regions available on a single active provider."""
    try:
        return await svc.list_regions_for_provider(provider_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
