"""Tests for artenic_ai_platform.registry.service — 100% coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from artenic_ai_platform.db.models import Base, PromotionRecord, RegisteredModel
from artenic_ai_platform.registry.service import ModelRegistry

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


def _model_metadata(
    *,
    name: str = "sentiment",
    version: str = "1.0",
    model_type: str = "classifier",
    framework: str = "pytorch",
) -> dict:
    return {
        "name": name,
        "version": version,
        "model_type": model_type,
        "framework": framework,
        "description": "A test model",
        "tags": {"env": "test"},
        "input_features": [{"name": "text", "type": "string"}],
        "output_schema": {"label": "string"},
    }


# ======================================================================
# register
# ======================================================================


class TestRegister:
    async def test_register_creates_model(self, session: AsyncSession) -> None:
        registry = ModelRegistry(session)
        model_id = await registry.register(_model_metadata())

        assert model_id == "sentiment_v1.0"
        row = await session.get(RegisteredModel, "sentiment_v1.0")
        assert row is not None
        assert row.name == "sentiment"
        assert row.version == "1.0"
        assert row.model_type == "classifier"
        assert row.framework == "pytorch"
        assert row.stage == "registered"

    async def test_register_with_mlflow_run_id(self, session: AsyncSession) -> None:
        registry = ModelRegistry(session)
        model_id = await registry.register(_model_metadata(), mlflow_run_id="run-42")

        row = await session.get(RegisteredModel, model_id)
        assert row is not None
        assert row.mlflow_run_id == "run-42"

    async def test_register_minimal_metadata(self, session: AsyncSession) -> None:
        registry = ModelRegistry(session)
        model_id = await registry.register({"name": "m", "version": "1"})

        assert model_id == "m_v1"
        row = await session.get(RegisteredModel, "m_v1")
        assert row is not None
        assert row.model_type == ""
        assert row.framework == "custom"
        assert row.tags == {}
        assert row.input_features == []
        assert row.output_schema == {}


# ======================================================================
# get
# ======================================================================


class TestGet:
    async def test_get_existing_model(self, session: AsyncSession) -> None:
        registry = ModelRegistry(session)
        await registry.register(_model_metadata())

        result = await registry.get("sentiment_v1.0")
        assert result["name"] == "sentiment"
        assert result["model_id"] == "sentiment_v1.0"
        assert result["stage"] == "registered"
        assert result["tags"] == {"env": "test"}

    async def test_get_not_found_raises(self, session: AsyncSession) -> None:
        from artenic_ai_sdk.exceptions import ModelNotFoundError

        registry = ModelRegistry(session)

        with pytest.raises(ModelNotFoundError, match="not-there"):
            await registry.get("not-there")


# ======================================================================
# list_all
# ======================================================================


class TestListAll:
    async def test_list_empty(self, session: AsyncSession) -> None:
        registry = ModelRegistry(session)
        result = await registry.list_all()
        assert result == []

    async def test_list_multiple_models(self, session: AsyncSession) -> None:
        registry = ModelRegistry(session)
        await registry.register(_model_metadata(name="a", version="1"))
        await registry.register(_model_metadata(name="b", version="2"))

        result = await registry.list_all()
        assert len(result) == 2
        names = {r["name"] for r in result}
        assert names == {"a", "b"}


# ======================================================================
# promote
# ======================================================================


class TestPromote:
    async def test_promote_updates_stage(self, session: AsyncSession) -> None:
        registry = ModelRegistry(session)
        await registry.register(_model_metadata())

        await registry.promote("sentiment_v1.0", "1.0")

        row = await session.get(RegisteredModel, "sentiment_v1.0")
        assert row is not None
        assert row.stage == "production"

    async def test_promote_creates_promotion_record(self, session: AsyncSession) -> None:
        registry = ModelRegistry(session)
        await registry.register(_model_metadata())

        await registry.promote("sentiment_v1.0", "1.0")

        from sqlalchemy import select

        stmt = select(PromotionRecord).where(PromotionRecord.model_id == "sentiment_v1.0")
        result = await session.execute(stmt)
        promo = result.scalar_one()
        assert promo.from_stage == "registered"
        assert promo.to_stage == "production"
        assert promo.version == "1.0"

    async def test_promote_syncs_mlflow(self, session: AsyncSession) -> None:
        mock_mlflow = MagicMock()
        mock_mlflow.available = True
        mock_mlflow.transition_stage = AsyncMock(return_value=True)

        registry = ModelRegistry(session, mlflow=mock_mlflow)
        await registry.register(_model_metadata())
        await registry.promote("sentiment_v1.0", "1.0")

        mock_mlflow.transition_stage.assert_awaited_once_with("sentiment", "1.0", "Production")

    async def test_promote_not_found_raises(self, session: AsyncSession) -> None:
        from artenic_ai_sdk.exceptions import ModelNotFoundError

        registry = ModelRegistry(session)

        with pytest.raises(ModelNotFoundError, match="not-there"):
            await registry.promote("not-there", "1.0")

    async def test_promote_no_mlflow_sync_when_unavailable(self, session: AsyncSession) -> None:
        mock_mlflow = MagicMock()
        mock_mlflow.available = False

        registry = ModelRegistry(session, mlflow=mock_mlflow)
        await registry.register(_model_metadata())
        await registry.promote("sentiment_v1.0", "1.0")

        mock_mlflow.transition_stage.assert_not_called()


# ======================================================================
# retire
# ======================================================================


class TestRetire:
    async def test_retire_updates_stage(self, session: AsyncSession) -> None:
        registry = ModelRegistry(session)
        await registry.register(_model_metadata())

        await registry.retire("sentiment_v1.0")

        row = await session.get(RegisteredModel, "sentiment_v1.0")
        assert row is not None
        assert row.stage == "archived"

    async def test_retire_creates_promotion_record(self, session: AsyncSession) -> None:
        registry = ModelRegistry(session)
        await registry.register(_model_metadata())

        await registry.retire("sentiment_v1.0")

        from sqlalchemy import select

        stmt = select(PromotionRecord).where(PromotionRecord.model_id == "sentiment_v1.0")
        result = await session.execute(stmt)
        promo = result.scalar_one()
        assert promo.from_stage == "registered"
        assert promo.to_stage == "archived"

    async def test_retire_syncs_mlflow(self, session: AsyncSession) -> None:
        mock_mlflow = MagicMock()
        mock_mlflow.available = True
        mock_mlflow.transition_stage = AsyncMock(return_value=True)

        registry = ModelRegistry(session, mlflow=mock_mlflow)
        await registry.register(_model_metadata())
        await registry.retire("sentiment_v1.0")

        mock_mlflow.transition_stage.assert_awaited_once_with("sentiment", "1.0", "Archived")

    async def test_retire_not_found_raises(self, session: AsyncSession) -> None:
        from artenic_ai_sdk.exceptions import ModelNotFoundError

        registry = ModelRegistry(session)

        with pytest.raises(ModelNotFoundError, match="gone"):
            await registry.retire("gone")

    async def test_retire_no_mlflow_sync_when_unavailable(self, session: AsyncSession) -> None:
        mock_mlflow = MagicMock()
        mock_mlflow.available = False

        registry = ModelRegistry(session, mlflow=mock_mlflow)
        await registry.register(_model_metadata())
        await registry.retire("sentiment_v1.0")

        mock_mlflow.transition_stage.assert_not_called()


# ======================================================================
# get_best_model
# ======================================================================


class TestGetBestModel:
    async def test_returns_none_without_mlflow(self, session: AsyncSession) -> None:
        registry = ModelRegistry(session)
        result = await registry.get_best_model("exp-1", "accuracy")
        assert result is None

    async def test_returns_none_when_mlflow_unavailable(self, session: AsyncSession) -> None:
        mock_mlflow = MagicMock()
        mock_mlflow.available = False

        registry = ModelRegistry(session, mlflow=mock_mlflow)
        result = await registry.get_best_model("exp-1", "accuracy")
        assert result is None

    async def test_returns_best_run(self, session: AsyncSession) -> None:
        mock_mlflow = MagicMock()
        mock_mlflow.available = True
        mock_mlflow.get_best_run = AsyncMock(
            return_value={"run_id": "best", "metrics": {"accuracy": 0.99}}
        )

        registry = ModelRegistry(session, mlflow=mock_mlflow)
        result = await registry.get_best_model("exp-1", "accuracy")
        assert result is not None
        assert result["run_id"] == "best"

    async def test_returns_none_when_no_runs(self, session: AsyncSession) -> None:
        mock_mlflow = MagicMock()
        mock_mlflow.available = True
        mock_mlflow.get_best_run = AsyncMock(return_value=None)

        registry = ModelRegistry(session, mlflow=mock_mlflow)
        result = await registry.get_best_model("exp-1", "accuracy")
        assert result is None


# ======================================================================
# _to_dict
# ======================================================================


class TestToDict:
    async def test_all_fields_present(self, session: AsyncSession) -> None:
        registry = ModelRegistry(session)
        await registry.register(_model_metadata())

        result = await registry.get("sentiment_v1.0")
        expected_keys = {
            "name",
            "version",
            "model_type",
            "framework",
            "description",
            "tags",
            "input_features",
            "output_schema",
            "created_at",
            "updated_at",
            "model_id",
            "stage",
            "mlflow_run_id",
        }
        assert set(result.keys()) == expected_keys

    async def test_timestamps_are_iso_or_none(self, session: AsyncSession) -> None:
        registry = ModelRegistry(session)
        await registry.register(_model_metadata())

        result = await registry.get("sentiment_v1.0")
        # created_at should be an ISO string (or None for aiosqlite)
        if result["created_at"] is not None:
            assert isinstance(result["created_at"], str)
        # updated_at is None for a new model
        assert result["updated_at"] is None
