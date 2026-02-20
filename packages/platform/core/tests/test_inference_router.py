"""Tests for artenic_ai_platform.inference.router — /api/v1/services/*."""

from __future__ import annotations

from typing import Any

from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from artenic_ai_platform.db.models import Base

# ======================================================================
# Helper — minimal FastAPI app with inference router
# ======================================================================


async def _make_client() -> tuple[AsyncClient, Any]:
    """Return (AsyncClient, engine) wired to a fresh in-memory DB."""
    from fastapi import FastAPI

    from artenic_ai_platform.inference.router import router

    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    app = FastAPI()
    app.state.session_factory = factory
    app.include_router(router)

    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://test")
    return client, engine


# ======================================================================
# POST /{service}/predict
# ======================================================================


class TestPredictEndpoint:
    async def test_predict_endpoint(self) -> None:
        """POST /api/v1/services/sentiment/predict returns 200."""
        client, engine = await _make_client()
        try:
            resp = await client.post(
                "/api/v1/services/sentiment/predict",
                json={"data": {"text": "hello"}},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["service"] == "sentiment"
            assert body["model_id"] == "sentiment_default"
            assert "prediction" in body
            assert "timestamp" in body
        finally:
            await client.aclose()
            await engine.dispose()

    async def test_predict_with_model_id(self) -> None:
        """POST with model_id in body uses the provided model_id."""
        client, engine = await _make_client()
        try:
            resp = await client.post(
                "/api/v1/services/ner/predict",
                json={"data": {"text": "hello"}, "model_id": "ner-custom-v5"},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["model_id"] == "ner-custom-v5"
        finally:
            await client.aclose()
            await engine.dispose()


# ======================================================================
# POST /{service}/predict_batch
# ======================================================================


class TestPredictBatchEndpoint:
    async def test_predict_batch_endpoint(self) -> None:
        """POST /api/v1/services/sentiment/predict_batch returns 200 list."""
        client, engine = await _make_client()
        try:
            resp = await client.post(
                "/api/v1/services/sentiment/predict_batch",
                json={"batch": [{"text": "a"}, {"text": "b"}]},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert isinstance(body, list)
            assert len(body) == 2
            assert body[0]["service"] == "sentiment"
            assert body[1]["service"] == "sentiment"
        finally:
            await client.aclose()
            await engine.dispose()
