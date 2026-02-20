"""Tests for artenic_ai_platform.events.event_bus — EventBus coverage."""

from __future__ import annotations

import asyncio
from typing import Any

from artenic_ai_platform.events.event_bus import EventBus

# ======================================================================
# subscribe
# ======================================================================


class TestSubscribeReturnsQueue:
    """subscribe() returns an asyncio.Queue."""

    async def test_subscribe_returns_queue(self) -> None:
        bus = EventBus()
        queue = bus.subscribe("training")
        assert isinstance(queue, asyncio.Queue)


# ======================================================================
# publish — single subscriber
# ======================================================================


class TestPublishReachesSubscriber:
    """publish() delivers an event that contains _topic and _timestamp."""

    async def test_publish_reaches_subscriber(self) -> None:
        bus = EventBus()
        queue = bus.subscribe("health")
        reached = bus.publish("health", {"status": "ok"})

        assert reached == 1
        event: dict[str, Any] = queue.get_nowait()
        assert event["status"] == "ok"
        assert event["_topic"] == "health"
        assert "_timestamp" in event


# ======================================================================
# publish — multiple subscribers
# ======================================================================


class TestPublishMultipleSubscribers:
    """All subscribers receive the same event."""

    async def test_publish_multiple_subscribers(self) -> None:
        bus = EventBus()
        q1 = bus.subscribe("training")
        q2 = bus.subscribe("training")
        q3 = bus.subscribe("training")

        reached = bus.publish("training", {"epoch": 1})

        assert reached == 3
        for q in (q1, q2, q3):
            event = q.get_nowait()
            assert event["epoch"] == 1
            assert event["_topic"] == "training"


# ======================================================================
# publish — no subscribers
# ======================================================================


class TestPublishNoSubscribers:
    """publish() returns 0 when nobody is listening."""

    async def test_publish_no_subscribers(self) -> None:
        bus = EventBus()
        reached = bus.publish("unknown_topic", {"data": 42})
        assert reached == 0


# ======================================================================
# unsubscribe — removes queue
# ======================================================================


class TestUnsubscribeRemovesQueue:
    """After unsubscribe, subscriber_count drops."""

    async def test_unsubscribe_removes_queue(self) -> None:
        bus = EventBus()
        queue = bus.subscribe("config")
        assert bus.subscriber_count("config") == 1

        bus.unsubscribe("config", queue)
        assert bus.subscriber_count("config") == 0


# ======================================================================
# unsubscribe — nonexistent topic
# ======================================================================


class TestUnsubscribeNonexistentTopic:
    """unsubscribe with an unknown topic does not raise."""

    async def test_unsubscribe_nonexistent_topic(self) -> None:
        bus = EventBus()
        dummy_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        # Must not raise
        bus.unsubscribe("no_such_topic", dummy_queue)


# ======================================================================
# unsubscribe — nonexistent queue
# ======================================================================


class TestUnsubscribeNonexistentQueue:
    """unsubscribe with an unknown queue does not raise."""

    async def test_unsubscribe_nonexistent_queue(self) -> None:
        bus = EventBus()
        real_queue = bus.subscribe("health")
        other_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        # Unsubscribing a queue that was never registered must not raise.
        bus.unsubscribe("health", other_queue)
        # The real subscriber should still be present.
        assert bus.subscriber_count("health") == 1

        # Clean up
        bus.unsubscribe("health", real_queue)


# ======================================================================
# subscriber_count
# ======================================================================


class TestSubscriberCount:
    """subscriber_count reflects the accurate number of subscribers."""

    async def test_subscriber_count(self) -> None:
        bus = EventBus()
        assert bus.subscriber_count("training") == 0

        q1 = bus.subscribe("training")
        assert bus.subscriber_count("training") == 1

        q2 = bus.subscribe("training")
        assert bus.subscriber_count("training") == 2

        bus.unsubscribe("training", q1)
        assert bus.subscriber_count("training") == 1

        bus.unsubscribe("training", q2)
        assert bus.subscriber_count("training") == 0


# ======================================================================
# topics property
# ======================================================================


class TestTopicsProperty:
    """topics lists only topics that have at least one subscriber."""

    async def test_topics_property(self) -> None:
        bus = EventBus()
        assert bus.topics == []

        q_train = bus.subscribe("training")
        q_health = bus.subscribe("health")
        assert sorted(bus.topics) == ["health", "training"]

        bus.unsubscribe("training", q_train)
        assert bus.topics == ["health"]

        bus.unsubscribe("health", q_health)
        assert bus.topics == []


# ======================================================================
# full queue drops event
# ======================================================================


class TestFullQueueDropsEvent:
    """When a queue is full (maxsize=256), publish logs a warning but
    does not crash, and the subscriber stays alive."""

    async def test_full_queue_drops_event(self) -> None:
        bus = EventBus()
        queue = bus.subscribe("training")

        # Fill the queue to capacity (maxsize=256).
        for i in range(256):
            bus.publish("training", {"i": i})

        assert queue.full()

        # Publishing one more should not crash (the warning is logged
        # internally but we only care about the return value here).
        reached = bus.publish("training", {"i": 256})

        # The event was dropped, so reached == 0, but the subscriber
        # should still be registered (alive).
        assert reached == 0
        assert bus.subscriber_count("training") == 1


# ======================================================================
# publish injects metadata
# ======================================================================


class TestPublishInjectsMetadata:
    """Published events contain _topic and _timestamp keys."""

    async def test_publish_injects_metadata(self) -> None:
        bus = EventBus()
        queue = bus.subscribe("lifecycle")

        bus.publish("lifecycle", {"action": "start"})
        event = queue.get_nowait()

        assert "_topic" in event
        assert "_timestamp" in event
        assert event["_topic"] == "lifecycle"
        assert isinstance(event["_timestamp"], float)
