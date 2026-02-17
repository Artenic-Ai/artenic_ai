"""Async database engine, session factory, and table bootstrapping."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy.ext.asyncio import (
    create_async_engine as _sa_create_async_engine,
)

from artenic_ai_platform.db.models import Base


def create_async_engine(database_url: str) -> AsyncEngine:
    """Create an :class:`AsyncEngine` for the given *database_url*.

    Supported schemes:

    * ``postgresql+asyncpg://...``
    * ``sqlite+aiosqlite://...``

    For SQLite the ``check_same_thread`` connect-arg is disabled
    automatically so the engine can be used from any thread.
    """
    kwargs: dict[str, object] = {}

    if database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}

    return _sa_create_async_engine(database_url, **kwargs)


def create_session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    """Return a session factory bound to *engine*.

    Sessions produced by the factory have
    ``expire_on_commit=False`` so that attributes remain
    accessible after a commit without an additional query.
    """
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


async def create_tables(engine: AsyncEngine) -> None:
    """Create all tables declared on :class:`Base`.

    Uses ``run_sync`` to execute the blocking
    ``metadata.create_all`` inside the async context.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
