"""REST API for runtime settings — /api/v1/settings/*."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from artenic_ai_platform.config.settings_manager import SettingsManager

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])


# ------------------------------------------------------------------
# Dependency helpers
# ------------------------------------------------------------------


async def _get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Yield a DB session from app state session_factory."""
    factory = request.app.state.session_factory
    async with factory() as session:
        yield session


def _get_secret_manager(request: Request) -> Any:
    """Get SecretManager from app state (may be None)."""
    return getattr(request.app.state, "secret_manager", None)


DbSession = Annotated[AsyncSession, Depends(_get_db)]


# ------------------------------------------------------------------
# Endpoints — static paths MUST come before parameterized paths
# ------------------------------------------------------------------


@router.get("/schema")
async def get_schema() -> dict[str, list[dict[str, Any]]]:
    """Return the full configuration schema."""
    return SettingsManager.get_schema_all()


@router.get("/schema/{scope}")
async def get_scope_schema(scope: str) -> list[dict[str, Any]]:
    """Return sections for a scope."""
    return SettingsManager.get_schema_for_scope(scope)


@router.get("/schema/{scope}/{section}")
async def get_section_schema(scope: str, section: str) -> dict[str, Any] | None:
    """Return a single section schema."""
    return SettingsManager.get_section_schema(scope, section)


@router.get("/audit/log")
async def get_audit_log(
    request: Request,
    session: DbSession,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[dict[str, Any]]:
    """Return paginated audit log."""
    mgr = SettingsManager(session, _get_secret_manager(request))
    return await mgr.get_audit_log(limit=limit, offset=offset)


@router.get("/{scope}")
async def get_all_settings(
    scope: str,
    request: Request,
    session: DbSession,
) -> dict[str, dict[str, str]]:
    """Get all settings for a scope."""
    mgr = SettingsManager(session, _get_secret_manager(request))
    return await mgr.get_all(scope)


@router.get("/{scope}/{section}")
async def get_section_settings(
    scope: str,
    section: str,
    request: Request,
    session: DbSession,
) -> dict[str, str]:
    """Get all settings for a section within a scope."""
    mgr = SettingsManager(session, _get_secret_manager(request))
    return await mgr.get_section(scope, section)


@router.put("/{scope}/{section}")
async def update_section_settings(
    scope: str,
    section: str,
    updates: dict[str, str],
    request: Request,
    session: DbSession,
) -> dict[str, str]:
    """Update settings in a section (hot-reload)."""
    mgr = SettingsManager(session, _get_secret_manager(request))
    return await mgr.update_section(scope, section, updates)
