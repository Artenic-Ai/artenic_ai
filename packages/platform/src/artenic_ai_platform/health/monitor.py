"""Background health-monitoring service.

Continuously records inference observations, computes per-model health
snapshots (error rate, latency percentiles, confidence drift), persists
results to the database, and publishes alerts on the event bus when a
model becomes degraded or unhealthy.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from artenic_ai_platform.db.models import ModelHealthRecord

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from artenic_ai_platform.events.event_bus import EventBus

logger = logging.getLogger(__name__)

_MAX_BUFFER_SIZE = 1000


class HealthMonitor:
    """Background service that buffers inference observations and
    periodically computes health snapshots for every active model.

    Parameters
    ----------
    session_factory:
        An async session-maker bound to the platform database engine.
    event_bus:
        Optional ``EventBus`` instance used to publish degraded/unhealthy
        alerts on the ``"health"`` topic.
    check_interval_seconds:
        How often (in seconds) the background loop runs.
    drift_threshold:
        Maximum acceptable absolute confidence drift before a model is
        considered degraded.
    """

    __slots__ = (
        "_check_interval",
        "_drift_threshold",
        "_event_bus",
        "_observations",
        "_running",
        "_session_factory",
        "_task",
    )

    def __init__(
        self,
        session_factory: async_sessionmaker[Any],
        event_bus: EventBus | None = None,
        *,
        check_interval_seconds: float = 60.0,
        drift_threshold: float = 0.1,
    ) -> None:
        self._session_factory = session_factory
        self._event_bus = event_bus
        self._check_interval = check_interval_seconds
        self._drift_threshold = drift_threshold

        self._observations: dict[str, list[dict[str, float | bool]]] = {}
        self._running: bool = False
        self._task: asyncio.Task[None] | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start the background monitoring loop."""
        if self._running:
            logger.warning("HealthMonitor is already running")
            return
        self._running = True
        self._task = asyncio.get_event_loop().create_task(self._monitor_loop())
        logger.info(
            "HealthMonitor started (interval=%.1fs, drift_threshold=%.2f)",
            self._check_interval,
            self._drift_threshold,
        )

    def stop(self) -> None:
        """Cancel the background task and stop the monitor."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            self._task = None
        logger.info("HealthMonitor stopped")

    # -- recording ---------------------------------------------------------

    def record_inference(
        self,
        model_id: str,
        latency_ms: float,
        confidence: float = 1.0,
        error: bool = False,
    ) -> None:
        """Buffer a single inference observation for *model_id*.

        The internal buffer is capped at :data:`_MAX_BUFFER_SIZE` entries
        per model; the oldest observations are discarded when the limit
        is reached.
        """
        observation: dict[str, float | bool] = {
            "latency_ms": latency_ms,
            "confidence": confidence,
            "error": error,
            "timestamp": time.time(),
        }
        buffer = self._observations.setdefault(model_id, [])
        buffer.append(observation)
        if len(buffer) > _MAX_BUFFER_SIZE:
            # Discard the oldest entries to stay within the cap.
            self._observations[model_id] = buffer[-_MAX_BUFFER_SIZE:]

    # -- snapshot computation ----------------------------------------------

    def compute_health_snapshot(self, model_id: str) -> dict[str, Any]:
        """Analyse buffered observations and return a health snapshot.

        Returns a dict with keys: ``error_rate``, ``avg_latency_ms``,
        ``p99_latency_ms``, ``avg_confidence``, ``confidence_drift``,
        ``sample_count``, and ``status``.
        """
        buffer = self._observations.get(model_id, [])
        sample_count = len(buffer)

        if sample_count == 0:
            return {
                "model_id": model_id,
                "error_rate": 0.0,
                "avg_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "avg_confidence": 1.0,
                "confidence_drift": 0.0,
                "sample_count": 0,
                "status": "healthy",
            }

        error_count = sum(1 for o in buffer if o["error"])
        error_rate = error_count / sample_count

        latencies = [float(o["latency_ms"]) for o in buffer]
        avg_latency_ms = sum(latencies) / sample_count

        sorted_latencies = sorted(latencies)
        p99_index = max(0, int(len(sorted_latencies) * 0.99) - 1)
        p99_latency_ms = sorted_latencies[p99_index]

        confidences = [float(o["confidence"]) for o in buffer]
        avg_confidence = sum(confidences) / sample_count
        confidence_drift = abs(avg_confidence - 1.0)

        # Determine overall status.
        if error_rate < 0.1 and confidence_drift < self._drift_threshold:
            status = "healthy"
        elif error_rate < 0.5:
            status = "degraded"
        else:
            status = "unhealthy"

        return {
            "model_id": model_id,
            "error_rate": error_rate,
            "avg_latency_ms": avg_latency_ms,
            "p99_latency_ms": p99_latency_ms,
            "avg_confidence": avg_confidence,
            "confidence_drift": confidence_drift,
            "sample_count": sample_count,
            "status": status,
        }

    # -- batch check -------------------------------------------------------

    async def check_all_models(self) -> list[dict[str, Any]]:
        """Compute snapshots for every model with buffered observations.

        Each snapshot is persisted to the database as a
        :class:`ModelHealthRecord`.  If the model is degraded or unhealthy
        the snapshot is also published to the ``"health"`` topic on the
        event bus.
        """
        snapshots: list[dict[str, Any]] = []

        for model_id in list(self._observations):
            snapshot = self.compute_health_snapshot(model_id)
            snapshots.append(snapshot)

            # Determine window boundaries from the buffered observations.
            buffer = self._observations.get(model_id, [])
            window_start: datetime | None = None
            window_end: datetime | None = None
            if buffer:
                window_start = datetime.fromtimestamp(
                    float(buffer[0]["timestamp"]),
                    tz=UTC,
                )
                window_end = datetime.fromtimestamp(
                    float(buffer[-1]["timestamp"]),
                    tz=UTC,
                )

            alert_triggered = snapshot["status"] != "healthy"

            # Persist to the database.
            try:
                async with self._session_factory() as session:
                    record = ModelHealthRecord(
                        model_id=model_id,
                        metric_name="health_snapshot",
                        metric_value=snapshot["error_rate"],
                        drift_score=snapshot["confidence_drift"],
                        alert_triggered=alert_triggered,
                        sample_count=snapshot["sample_count"],
                        window_start=window_start,
                        window_end=window_end,
                    )
                    session.add(record)
                    await session.commit()
            except Exception:
                logger.exception(
                    "Failed to persist health record for model=%s",
                    model_id,
                )

            # Publish an event when the model is not healthy.
            if alert_triggered and self._event_bus is not None:
                self._event_bus.publish("health", snapshot)

        return snapshots

    # -- history -----------------------------------------------------------

    async def get_health_history(
        self,
        model_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Query persisted :class:`ModelHealthRecord` rows for *model_id*.

        Returns up to *limit* records ordered by ``created_at`` descending.
        """
        async with self._session_factory() as session:
            stmt = (
                select(ModelHealthRecord)
                .where(ModelHealthRecord.model_id == model_id)
                .order_by(ModelHealthRecord.created_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

        return [
            {
                "id": row.id,
                "model_id": row.model_id,
                "metric_name": row.metric_name,
                "metric_value": row.metric_value,
                "drift_score": row.drift_score,
                "alert_triggered": row.alert_triggered,
                "sample_count": row.sample_count,
                "window_start": (row.window_start.isoformat() if row.window_start else None),
                "window_end": (row.window_end.isoformat() if row.window_end else None),
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ]

    # -- private background loop -------------------------------------------

    async def _monitor_loop(self) -> None:
        """Run :meth:`check_all_models` on a fixed interval until stopped."""
        logger.info("Health monitor loop started")
        while self._running:
            try:
                await self.check_all_models()
            except Exception:
                logger.exception("Error in health monitor loop")
            await asyncio.sleep(self._check_interval)
        logger.info("Health monitor loop exited")
