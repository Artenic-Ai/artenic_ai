"""Tests for artenic_ai_platform.events.ws — WebSocket endpoint coverage."""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import patch

from starlette.testclient import TestClient

from artenic_ai_platform.events.event_bus import EventBus


def _make_app() -> Any:
    """Build a minimal FastAPI app with an EventBus and the WS router."""
    from fastapi import FastAPI

    from artenic_ai_platform.events.ws import router

    app = FastAPI()
    app.state.event_bus = EventBus()
    app.include_router(router)
    return app


# ======================================================================
# connect
# ======================================================================


class TestWsConnect:
    """Client can connect to /ws successfully."""

    def test_ws_connect(self) -> None:
        app = _make_app()
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            # Connection accepted — simply close cleanly.
            ws.close()


# ======================================================================
# receives event
# ======================================================================


class TestWsReceivesEvent:
    """Publishing an event on the bus is delivered over the WebSocket."""

    def test_ws_receives_event(self) -> None:
        app = _make_app()
        bus: EventBus = app.state.event_bus
        client = TestClient(app)

        with client.websocket_connect("/ws?topics=health") as ws:
            bus.publish("health", {"status": "ok"})
            raw: str = ws.receive_text()
            event: dict[str, Any] = json.loads(raw)

            assert event["status"] == "ok"
            assert event["_topic"] == "health"
            assert "_timestamp" in event


# ======================================================================
# topic filtering
# ======================================================================


class TestWsTopicFiltering:
    """Only events matching subscribed topics are delivered."""

    def test_ws_topic_filtering(self) -> None:
        app = _make_app()
        bus: EventBus = app.state.event_bus
        client = TestClient(app)

        with client.websocket_connect("/ws?topics=health") as ws:
            # Publish to a topic the client did NOT subscribe to.
            bus.publish("training", {"epoch": 1})
            # Publish to the subscribed topic.
            bus.publish("health", {"status": "ok"})

            raw: str = ws.receive_text()
            event: dict[str, Any] = json.loads(raw)

            # The first (and only) message should be the health event,
            # not the training event.
            assert event["_topic"] == "health"
            assert event["status"] == "ok"


# ======================================================================
# heartbeat
# ======================================================================


class TestWsHeartbeat:
    """A heartbeat message with type=heartbeat is sent periodically."""

    def test_ws_heartbeat(self) -> None:
        app = _make_app()
        client = TestClient(app)

        with (
            patch(
                "artenic_ai_platform.events.ws._HEARTBEAT_INTERVAL",
                0.1,
            ),
            client.websocket_connect("/ws?topics=health") as ws,
        ):
            raw: str = ws.receive_text()
            msg: dict[str, Any] = json.loads(raw)
            assert msg == {"type": "heartbeat"}


# ======================================================================
# disconnect unsubscribes
# ======================================================================


class TestWsDisconnectUnsubscribes:
    """After the WebSocket disconnects, subscriber_count returns to 0."""

    def test_ws_disconnect_unsubscribes(self) -> None:
        app = _make_app()
        bus: EventBus = app.state.event_bus
        client = TestClient(app)

        with client.websocket_connect("/ws?topics=health") as ws:
            # While connected, there should be at least one subscriber.
            assert bus.subscriber_count("health") >= 1
            ws.close()

        # Give the server a moment to run cleanup.
        time.sleep(0.05)
        assert bus.subscriber_count("health") == 0
