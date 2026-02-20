"""Generic entity service — shared CRUD + version auto-increment."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from sqlalchemy import func, select

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="DeclarativeBase")


class GenericEntityService(Generic[T]):  # noqa: UP046
    """Base service providing CRUD for any ml_* entity.

    Subclasses must set ``_model_class`` to the ORM model type.
    """

    _model_class: type[T]

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def create(self, entity_id: str, data: dict[str, Any]) -> T:
        """Create a new entity.  *entity_id* is provided by the client."""
        record = self._model_class(id=entity_id, **data)
        self._session.add(record)
        await self._session.commit()
        await self._session.refresh(record)
        logger.info("Created %s %s", self._model_class.__tablename__, entity_id)
        return record

    async def get(self, entity_id: str) -> T | None:
        """Get an entity by its primary key."""
        result = await self._session.execute(
            select(self._model_class).where(self._model_class.id == entity_id)  # type: ignore[attr-defined]
        )
        return result.scalar_one_or_none()

    async def get_or_raise(self, entity_id: str) -> T:
        """Get an entity or raise ``ValueError``."""
        record = await self.get(entity_id)
        if record is None:
            table = self._model_class.__tablename__
            msg = f"Not found: {table} {entity_id}"
            raise ValueError(msg)
        return record

    async def list_all(
        self,
        *,
        offset: int = 0,
        limit: int = 50,
        filters: dict[str, Any] | None = None,
    ) -> list[T]:
        """List entities with pagination and optional column filters."""
        limit = min(max(limit, 1), 500)
        offset = max(offset, 0)
        stmt = select(self._model_class).order_by(
            self._model_class.created_at.desc()  # type: ignore[attr-defined]
        )
        for col, val in (filters or {}).items():
            if val is not None and hasattr(self._model_class, col):
                stmt = stmt.where(
                    getattr(self._model_class, col) == val
                )
        stmt = stmt.offset(offset).limit(limit)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def update(self, entity_id: str, updates: dict[str, Any]) -> T:
        """Update fields on an existing entity."""
        record = await self.get_or_raise(entity_id)
        for key, value in updates.items():
            if hasattr(record, key) and key != "id":
                setattr(record, key, value)
        await self._session.commit()
        await self._session.refresh(record)
        return record

    async def delete(self, entity_id: str) -> None:
        """Delete an entity by ID."""
        record = await self.get_or_raise(entity_id)
        await self._session.delete(record)
        await self._session.commit()
        logger.info("Deleted %s %s", self._model_class.__tablename__, entity_id)

    # ------------------------------------------------------------------
    # Version helper
    # ------------------------------------------------------------------

    async def next_version(self, name: str) -> int:
        """Compute the next version number for *name*."""
        result = await self._session.execute(
            select(func.coalesce(func.max(self._model_class.version), 0))  # type: ignore[attr-defined]
            .where(self._model_class.name == name)  # type: ignore[attr-defined]
        )
        return int(result.scalar_one()) + 1
