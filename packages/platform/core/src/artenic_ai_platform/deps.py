"""Dependency injection for FastAPI route handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncSession


async def get_db(app_state: Any = None) -> AsyncGenerator[AsyncSession, None]:
    """Yield a database session from the app-level session factory.

    Usage in routes::

        @router.get("/example")
        async def example(db: AsyncSession = Depends(get_db)):
            ...

    The actual dependency is wired in ``app.py`` via a closure that
    captures ``app.state.session_factory``.
    """
    # This function is replaced at runtime by create_app() with a closure
    # that captures the real session factory.  Keeping a top-level signature
    # so mypy/IDE can resolve the type.
    raise RuntimeError(  # pragma: no cover
        "get_db() called before app lifespan initialised the session factory."
    )
    yield  # pragma: no cover


def build_get_db(session_factory: Any) -> Any:
    """Create a concrete ``get_db`` dependency bound to *session_factory*."""

    async def _get_db() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            yield session

    return _get_db
