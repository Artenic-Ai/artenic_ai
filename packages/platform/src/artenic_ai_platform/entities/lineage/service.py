"""Lineage service — add, query, and graph lineage links."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy import or_, select

from artenic_ai_platform.db.models import MLLineage

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class LineageService:
    """Manages lineage links between ML entities."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_link(
        self, source_id: str, target_id: str, relation_type: str
    ) -> MLLineage:
        """Add a lineage link. Idempotent (unique constraint)."""
        link = MLLineage(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
        )
        self._session.add(link)
        await self._session.commit()
        await self._session.refresh(link)
        logger.info("Lineage: %s -[%s]-> %s", source_id, relation_type, target_id)
        return link

    async def get_links(self, entity_id: str) -> dict[str, Any]:
        """Get all lineage for an entity (upstream + downstream)."""
        result = await self._session.execute(
            select(MLLineage).where(
                or_(
                    MLLineage.source_id == entity_id,
                    MLLineage.target_id == entity_id,
                )
            )
        )
        links = list(result.scalars().all())

        upstream = [
            _link_to_dict(lk) for lk in links if lk.target_id == entity_id
        ]
        downstream = [
            _link_to_dict(lk) for lk in links if lk.source_id == entity_id
        ]
        return {
            "entity_id": entity_id,
            "upstream": upstream,
            "downstream": downstream,
        }

    async def get_graph(self, entity_id: str) -> dict[str, Any]:
        """Get full dependency graph starting from *entity_id* (BFS)."""
        visited: set[str] = set()
        edges: list[dict[str, Any]] = []
        queue = [entity_id]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            result = await self._session.execute(
                select(MLLineage).where(
                    or_(
                        MLLineage.source_id == current,
                        MLLineage.target_id == current,
                    )
                )
            )
            for link in result.scalars().all():
                edge = _link_to_dict(link)
                if edge not in edges:
                    edges.append(edge)
                neighbor = (
                    link.target_id if link.source_id == current else link.source_id
                )
                if neighbor not in visited:
                    queue.append(neighbor)

        return {
            "root": entity_id,
            "nodes": sorted(visited),
            "edges": edges,
        }

    async def remove_link(
        self, source_id: str, target_id: str, relation_type: str
    ) -> None:
        """Remove a lineage link."""
        result = await self._session.execute(
            select(MLLineage).where(
                MLLineage.source_id == source_id,
                MLLineage.target_id == target_id,
                MLLineage.relation_type == relation_type,
            )
        )
        record = result.scalar_one_or_none()
        if record is None:
            msg = f"Lineage link not found: {source_id} -[{relation_type}]-> {target_id}"
            raise ValueError(msg)
        await self._session.delete(record)
        await self._session.commit()


def _link_to_dict(link: MLLineage) -> dict[str, Any]:
    """Convert an MLLineage to a JSON-safe dict."""
    return {
        "id": link.id,
        "source_id": link.source_id,
        "target_id": link.target_id,
        "relation_type": link.relation_type,
        "created_at": link.created_at.isoformat() if link.created_at else "",
    }
