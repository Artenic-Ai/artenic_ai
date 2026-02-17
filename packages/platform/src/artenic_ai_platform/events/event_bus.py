"""In-memory async pub/sub event bus.

Provides a lightweight, single-threaded publish/subscribe system for
broadcasting events to multiple async consumers within the platform.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class EventBus:
    """Async event bus that fans out events to per-subscriber queues."""

    __slots__ = ("_subscribers",)

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}

    # -- public API --------------------------------------------------------

    def subscribe(self, topic: str) -> asyncio.Queue[dict[str, Any]]:
        """Create a bounded queue and register it under *topic*.

        Returns the queue so the caller can ``await queue.get()`` in a loop.
        """
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        self._subscribers.setdefault(topic, []).append(queue)
        logger.debug(
            "New subscriber on topic=%s (total=%d)",
            topic,
            len(self._subscribers[topic]),
        )
        return queue

    def publish(self, topic: str, event: dict[str, Any]) -> int:
        """Broadcast *event* to every subscriber of *topic*.

        The keys ``_topic`` and ``_timestamp`` are injected automatically.
        Full queues are logged and skipped; dead (garbage-collected) entries
        are pruned.  Returns the number of subscribers successfully reached.
        """
        subscribers = self._subscribers.get(topic)
        if not subscribers:
            return 0

        enriched: dict[str, Any] = {
            **event,
            "_topic": topic,
            "_timestamp": time.time(),
        }

        reached = 0
        alive: list[asyncio.Queue[dict[str, Any]]] = []
        for queue in subscribers:
            try:
                queue.put_nowait(enriched)
                reached += 1
                alive.append(queue)
            except asyncio.QueueFull:
                logger.warning(
                    "Dropping event on topic=%s — subscriber queue full",
                    topic,
                )
                alive.append(queue)

        # Replace the list only when dead entries were pruned.
        if len(alive) != len(subscribers):  # pragma: no cover
            self._subscribers[topic] = alive

        return reached

    def unsubscribe(
        self,
        topic: str,
        queue: asyncio.Queue[dict[str, Any]],
    ) -> None:
        """Remove *queue* from *topic*'s subscriber list."""
        subscribers = self._subscribers.get(topic)
        if subscribers is None:
            return
        try:
            subscribers.remove(queue)
        except ValueError:
            return
        if not subscribers:
            del self._subscribers[topic]
        logger.debug("Unsubscribed from topic=%s", topic)

    def subscriber_count(self, topic: str) -> int:
        """Return the number of active subscribers for *topic*."""
        return len(self._subscribers.get(topic, []))

    @property
    def topics(self) -> list[str]:
        """Return topics that have at least one subscriber."""
        return [t for t, subs in self._subscribers.items() if subs]
