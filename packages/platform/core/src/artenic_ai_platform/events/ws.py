"""WebSocket endpoint for real-time event streaming.

Clients connect to ``/ws?topics=training,health`` (or omit the query param
to receive all default topics) and receive JSON-encoded events as they are
published through the :class:`~artenic_ai_platform.events.event_bus.EventBus`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

if TYPE_CHECKING:
    from artenic_ai_platform.events.event_bus import EventBus

logger = logging.getLogger(__name__)

_DEFAULT_TOPICS: str = "training,ensemble,health,lifecycle,config"
_HEARTBEAT_INTERVAL: float = 30.0

router = APIRouter(tags=["websocket"])


async def websocket_endpoint(
    websocket: WebSocket,
    topics: str = Query(default=_DEFAULT_TOPICS),
) -> None:
    """Stream events over a WebSocket connection.

    Query parameters:
        topics: Comma-separated list of event topics to subscribe to.
               Defaults to all standard topics.
    """
    await websocket.accept()

    event_bus: EventBus = websocket.app.state.event_bus
    topic_list = [t.strip() for t in topics.split(",") if t.strip()]

    logger.info(
        "WebSocket connected — topics=%s",
        ", ".join(topic_list),
    )

    queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}
    for topic in topic_list:
        queues[topic] = event_bus.subscribe(topic)

    try:
        await _relay_events(websocket, queues)
    finally:
        for topic, queue in queues.items():
            event_bus.unsubscribe(topic, queue)
        logger.info("WebSocket disconnected")


async def _relay_events(
    websocket: WebSocket,
    queues: dict[str, asyncio.Queue[dict[str, Any]]],
) -> None:
    """Read from multiple queues and forward events to the WebSocket."""
    pending: dict[asyncio.Task[dict[str, Any]], str] = {}

    def _ensure_tasks() -> None:
        """Make sure there is exactly one pending get-task per queue."""
        active_topics = set(pending.values())
        for topic, queue in queues.items():
            if topic not in active_topics:
                task: asyncio.Task[dict[str, Any]] = asyncio.create_task(
                    queue.get(),
                )
                pending[task] = topic

    _ensure_tasks()

    try:
        while True:
            heartbeat_task: asyncio.Task[None] = asyncio.create_task(
                asyncio.sleep(_HEARTBEAT_INTERVAL),
            )
            waitables: set[asyncio.Task[object]] = {
                heartbeat_task,
                *pending,
            }

            done, _ = await asyncio.wait(
                waitables,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                if task is heartbeat_task:
                    await websocket.send_text(
                        json.dumps({"type": "heartbeat"}),
                    )
                    continue

                # Must be a queue-get task.
                event_task: asyncio.Task[dict[str, Any]] = task  # type: ignore[assignment]
                topic = pending.pop(event_task)
                event: dict[str, Any] = event_task.result()
                try:
                    await websocket.send_text(json.dumps(event, default=str))
                except WebSocketDisconnect:  # pragma: no cover
                    raise  # pragma: no cover

                # Re-arm a new get for this topic.
                new_task: asyncio.Task[dict[str, Any]] = asyncio.create_task(
                    queues[topic].get(),
                )
                pending[new_task] = topic

            # Cancel the heartbeat timer if it did not fire this round.
            if not heartbeat_task.done():
                heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat_task
    except WebSocketDisconnect:  # pragma: no cover
        pass
    finally:
        # Cancel all outstanding queue-get tasks.
        for task in pending:
            task.cancel()


router.add_api_websocket_route("/ws", websocket_endpoint)
