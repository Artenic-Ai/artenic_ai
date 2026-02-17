"""Inference service — prediction dispatch with A/B testing and health tracking."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from artenic_ai_platform.ab_testing.service import ABTestManager
    from artenic_ai_platform.events.event_bus import EventBus
    from artenic_ai_platform.health.monitor import HealthMonitor

logger = logging.getLogger(__name__)


class InferenceService:
    """Stateless inference gateway with optional monitoring hooks.

    Dispatches prediction requests, records latency and confidence to
    the :class:`HealthMonitor`, routes traffic through active A/B tests,
    and publishes events for downstream consumers.
    """

    def __init__(
        self,
        session: AsyncSession,
        health_monitor: HealthMonitor | None = None,
        ab_test_manager: ABTestManager | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._session = session
        self._health_monitor = health_monitor
        self._ab_test_manager = ab_test_manager
        self._event_bus = event_bus

    # ------------------------------------------------------------------
    # Single prediction
    # ------------------------------------------------------------------

    async def predict(
        self,
        service: str,
        data: dict[str, Any],
        *,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        """Run a single prediction, with optional A/B routing and monitoring.

        Parameters
        ----------
        service:
            Logical service name (e.g. ``"sentiment"``, ``"ner"``).
        data:
            Arbitrary payload forwarded to the underlying model.
        model_id:
            Explicit model override.  When *None* and no A/B variant is
            selected, defaults to ``"{service}_default"``.

        Returns
        -------
        dict
            ``{"prediction": ..., "model_id": ..., "service": ..., "timestamp": ...}``
        """
        variant: dict[str, Any] | None = None
        active_model_id: str = model_id or f"{service}_default"

        # --- A/B variant selection ------------------------------------
        if self._ab_test_manager is not None:
            variant = await self._ab_test_manager.select_variant(service)
            if variant is not None:
                active_model_id = variant["model_id"]

        # --- Execute prediction (stub) --------------------------------
        t0 = time.monotonic()
        try:
            result: dict[str, Any] = {
                "prediction": data,
                "model_id": active_model_id,
                "service": service,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            t1 = time.monotonic()
            latency_ms = (t1 - t0) * 1_000
            confidence = 1.0  # stub confidence

            # --- Health tracking (sync method) ------------------------
            if self._health_monitor is not None:
                self._health_monitor.record_inference(
                    active_model_id,
                    latency_ms,
                    confidence,
                    error=False,
                )

            # --- A/B metric recording ---------------------------------
            if self._ab_test_manager is not None and variant is not None:
                await self._ab_test_manager.record_metric(
                    test_id=variant["test_id"],
                    variant_name=variant["variant_name"],
                    metric_name="latency_ms",
                    metric_value=latency_ms,
                    latency_ms=latency_ms,
                )

            # --- Event publishing -------------------------------------
            if self._event_bus is not None:
                self._event_bus.publish(
                    "inference",
                    {
                        "service": service,
                        "model_id": active_model_id,
                        "latency_ms": latency_ms,
                        "confidence": confidence,
                    },
                )

        except Exception:
            t1 = time.monotonic()
            latency_ms = (t1 - t0) * 1_000
            if self._health_monitor is not None:
                self._health_monitor.record_inference(
                    active_model_id,
                    latency_ms,
                    0.0,
                    error=True,
                )
            raise

        return result

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    async def predict_batch(
        self,
        service: str,
        batch: list[dict[str, Any]],
        *,
        model_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run :meth:`predict` for every item in *batch*.

        Parameters
        ----------
        service:
            Logical service name.
        batch:
            List of data payloads.
        model_id:
            Optional model override applied to every prediction.

        Returns
        -------
        list[dict]
            One result dict per input item, in the same order.
        """
        return [await self.predict(service, item, model_id=model_id) for item in batch]
