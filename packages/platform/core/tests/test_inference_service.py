"""Tests for artenic_ai_platform.inference.service — InferenceService."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from artenic_ai_platform.db.models import Base
from artenic_ai_platform.inference.model_loader import ModelLoader
from artenic_ai_platform.inference.service import InferenceService
from artenic_ai_platform.plugins.loader import PluginInfo, PluginRegistry
from artenic_ai_sdk.exceptions import ModelNotFoundError
from artenic_ai_sdk.testing import MockModel

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
async def session() -> AsyncGenerator[AsyncSession, None]:
    """In-memory aiosqlite session for service tests."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as sess:
        yield sess

    await engine.dispose()


# ======================================================================
# predict — basic behaviour
# ======================================================================


class TestPredictBasic:
    async def test_predict_basic(self, session: AsyncSession) -> None:
        """predict returns a dict with the expected keys."""
        svc = InferenceService(session)
        result = await svc.predict("sentiment", {"text": "hello"})

        assert "prediction" in result
        assert "model_id" in result
        assert "service" in result
        assert "timestamp" in result
        assert result["service"] == "sentiment"

    async def test_predict_default_model_id(self, session: AsyncSession) -> None:
        """Without model_id, uses '{service}_default'."""
        svc = InferenceService(session)
        result = await svc.predict("sentiment", {"text": "hi"})

        assert result["model_id"] == "sentiment_default"

    async def test_predict_explicit_model_id(self, session: AsyncSession) -> None:
        """With an explicit model_id, the provided one is used."""
        svc = InferenceService(session)
        result = await svc.predict("sentiment", {"text": "hi"}, model_id="custom-model-v3")

        assert result["model_id"] == "custom-model-v3"


# ======================================================================
# predict — A/B testing integration
# ======================================================================


class TestPredictABTesting:
    async def test_predict_ab_routing(self, session: AsyncSession) -> None:
        """When ab_test_manager returns a variant, use its model_id."""
        ab = AsyncMock()
        ab.select_variant = AsyncMock(
            return_value={
                "test_id": "t-1",
                "variant_name": "variant-b",
                "model_id": "ab-model-v2",
            }
        )
        ab.record_metric = AsyncMock()

        svc = InferenceService(session, ab_test_manager=ab)
        result = await svc.predict("sentiment", {"text": "test"})

        assert result["model_id"] == "ab-model-v2"

    async def test_predict_ab_records_metric(self, session: AsyncSession) -> None:
        """record_metric is called when a variant is selected."""
        ab = AsyncMock()
        ab.select_variant = AsyncMock(
            return_value={
                "test_id": "t-1",
                "variant_name": "variant-b",
                "model_id": "ab-model-v2",
            }
        )
        ab.record_metric = AsyncMock()

        svc = InferenceService(session, ab_test_manager=ab)
        await svc.predict("sentiment", {"text": "test"})

        ab.record_metric.assert_awaited_once()
        kw = ab.record_metric.call_args
        assert kw.kwargs["test_id"] == "t-1"
        assert kw.kwargs["variant_name"] == "variant-b"
        assert kw.kwargs["metric_name"] == "latency_ms"


# ======================================================================
# predict — health tracking
# ======================================================================


class TestPredictHealthTracking:
    async def test_predict_health_tracking(self, session: AsyncSession) -> None:
        """record_inference is called (sync) with correct args on success."""
        health = MagicMock()
        health.record_inference = MagicMock()

        svc = InferenceService(session, health_monitor=health)
        await svc.predict("sentiment", {"text": "ok"})

        health.record_inference.assert_called_once()
        args = health.record_inference.call_args
        # positional: model_id, latency_ms, confidence, error=False
        assert args[0][0] == "sentiment_default"  # model_id
        assert isinstance(args[0][1], float)  # latency_ms
        assert args[0][2] == 1.0  # confidence
        assert args[1]["error"] is False

    async def test_predict_error_records_health(self, session: AsyncSession) -> None:
        """When prediction raises, record_inference gets error=True."""
        health = MagicMock()
        health.record_inference = MagicMock()

        # Inject an error via a faulty event_bus.publish that raises
        bus = MagicMock()
        bus.publish = MagicMock(side_effect=RuntimeError("boom"))

        svc = InferenceService(session, health_monitor=health, event_bus=bus)

        with pytest.raises(RuntimeError, match="boom"):
            await svc.predict("sentiment", {"text": "fail"})

        # The success-path call fires first, then event_bus.publish raises
        # which triggers the error-path call.  The *last* call must have
        # error=True.
        assert health.record_inference.call_count == 2
        error_call = health.record_inference.call_args_list[-1]
        assert error_call[0][0] == "sentiment_default"
        assert error_call[0][2] == 0.0  # confidence on error
        assert error_call[1]["error"] is True


# ======================================================================
# predict — event publishing
# ======================================================================


class TestPredictEventPublishing:
    async def test_predict_event_publishing(self, session: AsyncSession) -> None:
        """event_bus.publish is called with 'inference' topic."""
        bus = MagicMock()
        bus.publish = MagicMock()

        svc = InferenceService(session, event_bus=bus)
        await svc.predict("sentiment", {"text": "data"})

        bus.publish.assert_called_once()
        topic = bus.publish.call_args[0][0]
        assert topic == "inference"
        payload = bus.publish.call_args[0][1]
        assert payload["service"] == "sentiment"
        assert payload["model_id"] == "sentiment_default"
        assert "latency_ms" in payload
        assert "confidence" in payload


# ======================================================================
# predict_batch
# ======================================================================


class TestPredictBatch:
    async def test_predict_batch(self, session: AsyncSession) -> None:
        """predict_batch returns one result per input item."""
        svc = InferenceService(session)
        batch = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        results = await svc.predict_batch("ner", batch)

        assert isinstance(results, list)
        assert len(results) == 3
        for idx, r in enumerate(results):
            assert r["service"] == "ner"
            assert r["model_id"] == "ner_default"
            assert r["prediction"] == batch[idx]


# ======================================================================
# predict — no optional dependencies
# ======================================================================


class TestPredictNoOptionalDeps:
    async def test_predict_no_optional_deps(self, session: AsyncSession) -> None:
        """predict works fine with all optional deps set to None."""
        svc = InferenceService(
            session,
            health_monitor=None,
            ab_test_manager=None,
            event_bus=None,
        )
        result = await svc.predict("translation", {"text": "bonjour"})

        assert result["service"] == "translation"
        assert result["model_id"] == "translation_default"
        assert result["prediction"] == {"text": "bonjour"}
        assert "timestamp" in result


# ======================================================================
# Helpers for model_loader tests
# ======================================================================


async def _make_loader() -> ModelLoader:
    """Create a ModelLoader with a single MockModel registered."""
    registry = PluginRegistry(
        services={
            "mock": PluginInfo(
                name="mock",
                group="artenic_ai.services",
                module="m",
                obj=MockModel,
            ),
        },
    )
    loader = ModelLoader()
    await loader.load_from_registry(registry)
    return loader


# ======================================================================
# predict — with model_loader
# ======================================================================


class TestPredictWithModelLoader:
    async def test_predict_with_model(self, session: AsyncSession) -> None:
        """When model_loader has the model, predict calls model.predict()."""
        loader = await _make_loader()
        svc = InferenceService(session, model_loader=loader)
        result = await svc.predict("forex", {"price": 1.23}, model_id="mock_model")

        assert result["model_id"] == "mock_model"
        assert result["service"] == "forex"
        # prediction is a dict from BasePrediction.model_dump()
        pred = result["prediction"]
        assert "confidence" in pred
        assert "model_id" in pred
        assert pred["model_id"] == "mock_model"

    async def test_predict_model_not_found(self, session: AsyncSession) -> None:
        """When model_id doesn't exist, raises ModelNotFoundError."""
        loader = await _make_loader()
        svc = InferenceService(session, model_loader=loader)

        with pytest.raises(ModelNotFoundError, match="no_such_model"):
            await svc.predict("forex", {"x": 1}, model_id="no_such_model")

    async def test_predict_no_loader_uses_stub(self, session: AsyncSession) -> None:
        """Without model_loader, the stub fallback is used."""
        svc = InferenceService(session, model_loader=None)
        result = await svc.predict("sentiment", {"text": "hi"})

        # Stub echoes data back
        assert result["prediction"] == {"text": "hi"}

    async def test_predict_with_loader_health_confidence(
        self, session: AsyncSession
    ) -> None:
        """Health monitor receives real confidence from model prediction."""
        loader = await _make_loader()
        health = MagicMock()
        health.record_inference = MagicMock()

        svc = InferenceService(session, health_monitor=health, model_loader=loader)
        await svc.predict("forex", {"x": 1.0}, model_id="mock_model")

        health.record_inference.assert_called_once()
        args = health.record_inference.call_args
        # MockModel returns confidence=0.8 by default
        assert args[0][2] == 0.8

    async def test_predict_batch_with_loader(self, session: AsyncSession) -> None:
        """predict_batch works with model_loader."""
        loader = await _make_loader()
        svc = InferenceService(session, model_loader=loader)
        batch = [{"x": 1}, {"x": 2}]
        results = await svc.predict_batch("forex", batch, model_id="mock_model")

        assert len(results) == 2
        for r in results:
            assert r["model_id"] == "mock_model"
            assert "confidence" in r["prediction"]

    async def test_predict_with_loader_and_ab(self, session: AsyncSession) -> None:
        """A/B routing overrides model_id when model_loader is present."""
        loader = await _make_loader()
        ab = AsyncMock()
        ab.select_variant = AsyncMock(
            return_value={
                "test_id": "t-1",
                "variant_name": "v-b",
                "model_id": "mock_model",
            }
        )
        ab.record_metric = AsyncMock()

        svc = InferenceService(session, ab_test_manager=ab, model_loader=loader)
        result = await svc.predict("forex", {"x": 1})

        assert result["model_id"] == "mock_model"
        ab.record_metric.assert_awaited_once()
