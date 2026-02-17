"""A/B test management — create, route traffic, record metrics, conclude."""

from __future__ import annotations

import logging
import random
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import select, update

from artenic_ai_platform.db.models import ABTestMetricRecord, ABTestRecord

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from artenic_ai_platform.events.event_bus import EventBus

logger = logging.getLogger(__name__)


class ABTestManager:
    """Service layer for A/B test lifecycle and metric aggregation."""

    def __init__(
        self,
        session: AsyncSession,
        event_bus: EventBus | None = None,
    ) -> None:
        self._session = session
        self._event_bus = event_bus

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    async def create_test(
        self,
        name: str,
        service: str,
        variants: dict[str, Any],
        primary_metric: str,
        *,
        min_samples: int = 100,
    ) -> str:
        """Create a new A/B test.  Returns the test ID.

        *variants* must contain at least two entries, each with
        ``model_id`` and ``traffic_pct``.  The percentages must sum to 100.
        """
        # -- validation ------------------------------------------------
        if len(variants) < 2:
            msg = "At least two variants are required"
            raise ValueError(msg)

        total_pct = 0
        for variant_name, variant_cfg in variants.items():
            if "model_id" not in variant_cfg:
                msg = f"Variant '{variant_name}' is missing 'model_id'"
                raise ValueError(msg)
            if "traffic_pct" not in variant_cfg:
                msg = f"Variant '{variant_name}' is missing 'traffic_pct'"
                raise ValueError(msg)
            total_pct += variant_cfg["traffic_pct"]

        if total_pct != 100:
            msg = f"Traffic percentages must sum to 100, got {total_pct}"
            raise ValueError(msg)

        # -- persist ---------------------------------------------------
        test_id = str(uuid.uuid4())
        record = ABTestRecord(
            id=test_id,
            name=name,
            service=service,
            status="running",
            variants=variants,
            primary_metric=primary_metric,
            min_samples=min_samples,
        )
        self._session.add(record)
        await self._session.commit()

        logger.info("Created A/B test %s (%s) for service %s", test_id, name, service)

        if self._event_bus is not None:
            self._event_bus.publish(
                "ab_test",
                {
                    "action": "created",
                    "test_id": test_id,
                    "name": name,
                    "service": service,
                },
            )

        return test_id

    # ------------------------------------------------------------------
    # Traffic routing
    # ------------------------------------------------------------------

    async def select_variant(self, service: str) -> dict[str, Any] | None:
        """Pick a variant for *service* using weighted random selection.

        Returns ``{"test_id": ..., "variant_name": ..., "model_id": ...}``
        or ``None`` when no running test exists for the service.
        """
        stmt = (
            select(ABTestRecord)
            .where(
                ABTestRecord.service == service,
                ABTestRecord.status == "running",
            )
            .limit(1)
        )
        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()
        if record is None:
            return None

        variants = record.variants
        names = list(variants.keys())
        weights = [variants[n]["traffic_pct"] for n in names]

        chosen = random.choices(names, weights=weights, k=1)[0]
        return {
            "test_id": record.id,
            "variant_name": chosen,
            "model_id": variants[chosen]["model_id"],
        }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    async def record_metric(
        self,
        test_id: str,
        variant_name: str,
        metric_name: str,
        metric_value: float,
        *,
        latency_ms: float | None = None,
        error: bool = False,
    ) -> None:
        """Record a single metric observation for an A/B test variant."""
        metric = ABTestMetricRecord(
            ab_test_id=test_id,
            variant_name=variant_name,
            metric_name=metric_name,
            metric_value=metric_value,
            latency_ms=latency_ms,
            error=error,
        )
        self._session.add(metric)
        await self._session.commit()

    async def get_results(self, test_id: str) -> dict[str, Any]:
        """Aggregate metrics per variant for a test.

        Returns::

            {
                "test_id": "...",
                "status": "running",
                "variants": {
                    "control": {
                        "accuracy": {"mean": ..., "std": ..., "min": ..., "max": ..., "count": ...},
                        "error_rate": 0.02,
                        "avg_latency_ms": 42.1,
                        "sample_count": 150,
                    },
                    ...
                }
            }
        """
        # Fetch test record
        record = await self._session.get(ABTestRecord, test_id)
        if record is None:
            msg = f"A/B test '{test_id}' not found"
            raise KeyError(msg)

        # Fetch all metrics for this test
        stmt = select(ABTestMetricRecord).where(
            ABTestMetricRecord.ab_test_id == test_id,
        )
        result = await self._session.execute(stmt)
        metrics = result.scalars().all()

        # Group by variant
        variant_data: dict[str, list[ABTestMetricRecord]] = {}
        for m in metrics:
            variant_data.setdefault(m.variant_name, []).append(m)

        variants_out: dict[str, Any] = {}
        for variant_name, observations in variant_data.items():
            # Group by metric name
            by_metric: dict[str, list[float]] = {}
            total_count = len(observations)
            error_count = sum(1 for o in observations if o.error)
            latencies = [o.latency_ms for o in observations if o.latency_ms is not None]

            for o in observations:
                by_metric.setdefault(o.metric_name, []).append(o.metric_value)

            variant_summary: dict[str, Any] = {}
            for metric_name, values in by_metric.items():
                count = len(values)
                mean = sum(values) / count
                std = (sum((v - mean) ** 2 for v in values) / count) ** 0.5
                variant_summary[metric_name] = {
                    "mean": mean,
                    "std": std,
                    "min": min(values),
                    "max": max(values),
                    "count": count,
                }

            variant_summary["error_rate"] = error_count / total_count if total_count > 0 else 0.0
            variant_summary["avg_latency_ms"] = (
                sum(latencies) / len(latencies) if latencies else None
            )
            variant_summary["sample_count"] = total_count
            variants_out[variant_name] = variant_summary

        return {
            "test_id": test_id,
            "status": record.status,
            "variants": variants_out,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def conclude_test(
        self,
        test_id: str,
        *,
        winner: str | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        """Conclude an A/B test, optionally declaring a winner."""
        record = await self._session.get(ABTestRecord, test_id)
        if record is None:
            msg = f"A/B test '{test_id}' not found"
            raise KeyError(msg)

        now = datetime.now(UTC)
        await self._session.execute(
            update(ABTestRecord)
            .where(ABTestRecord.id == test_id)
            .values(
                status="concluded",
                winner=winner,
                conclusion_reason=reason,
                concluded_at=now,
            )
        )
        await self._session.commit()

        # Refresh to get updated values
        await self._session.refresh(record)

        logger.info("Concluded A/B test %s — winner=%s", test_id, winner)

        if self._event_bus is not None:
            self._event_bus.publish(
                "ab_test",
                {
                    "action": "concluded",
                    "test_id": test_id,
                    "winner": winner,
                    "reason": reason,
                },
            )

        return self._test_to_dict(record)

    async def pause_test(self, test_id: str) -> dict[str, Any]:
        """Pause a running A/B test."""
        record = await self._session.get(ABTestRecord, test_id)
        if record is None:
            msg = f"A/B test '{test_id}' not found"
            raise KeyError(msg)

        await self._session.execute(
            update(ABTestRecord).where(ABTestRecord.id == test_id).values(status="paused")
        )
        await self._session.commit()
        await self._session.refresh(record)

        logger.info("Paused A/B test %s", test_id)
        return self._test_to_dict(record)

    async def resume_test(self, test_id: str) -> dict[str, Any]:
        """Resume a paused A/B test.  Only works when status is ``paused``."""
        record = await self._session.get(ABTestRecord, test_id)
        if record is None:
            msg = f"A/B test '{test_id}' not found"
            raise KeyError(msg)

        if record.status != "paused":
            msg = f"Cannot resume test '{test_id}' — current status is '{record.status}'"
            raise ValueError(msg)

        await self._session.execute(
            update(ABTestRecord).where(ABTestRecord.id == test_id).values(status="running")
        )
        await self._session.commit()
        await self._session.refresh(record)

        logger.info("Resumed A/B test %s", test_id)
        return self._test_to_dict(record)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def get_test(self, test_id: str) -> dict[str, Any]:
        """Get a single A/B test by ID.  Raises ``KeyError`` if not found."""
        record = await self._session.get(ABTestRecord, test_id)
        if record is None:
            msg = f"A/B test '{test_id}' not found"
            raise KeyError(msg)
        return self._test_to_dict(record)

    async def list_tests(
        self,
        *,
        service: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List A/B tests with optional filters."""
        stmt = select(ABTestRecord)
        if service is not None:
            stmt = stmt.where(ABTestRecord.service == service)
        if status is not None:
            stmt = stmt.where(ABTestRecord.status == status)
        stmt = stmt.order_by(ABTestRecord.created_at.desc()).limit(limit).offset(offset)

        result = await self._session.execute(stmt)
        return [self._test_to_dict(r) for r in result.scalars().all()]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _test_to_dict(record: ABTestRecord) -> dict[str, Any]:
        """Convert an ``ABTestRecord`` ORM instance to a plain dict."""
        return {
            "id": record.id,
            "name": record.name,
            "service": record.service,
            "status": record.status,
            "variants": record.variants,
            "primary_metric": record.primary_metric,
            "min_samples": record.min_samples,
            "winner": record.winner,
            "conclusion_reason": record.conclusion_reason,
            "created_at": (record.created_at.isoformat() if record.created_at else None),
            "concluded_at": (record.concluded_at.isoformat() if record.concluded_at else None),
        }
