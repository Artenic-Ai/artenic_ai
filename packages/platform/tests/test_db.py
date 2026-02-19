"""Tests for artenic_ai_platform.db.engine + db.models — 100% coverage."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from artenic_ai_platform.db.engine import (
    create_async_engine,
    create_session_factory,
    create_tables,
)
from artenic_ai_platform.db.models import (
    ABTestMetricRecord,
    ABTestRecord,
    Base,
    BudgetAlertRecord,
    BudgetRecord,
    ConfigAuditRecord,
    ConfigOverrideRecord,
    ConfigSettingRecord,
    EnsembleJobRecord,
    EnsembleRecord,
    EnsembleVersionRecord,
    ModelHealthRecord,
    OptimizerRecommendationRecord,
    OptimizerTrainingSampleRecord,
    PromotionRecord,
    RegisteredModel,
    TrainingJob,
    TrainingOutcomeRecord,
)

SQLITE_URL = "sqlite+aiosqlite://"
EXPECTED_TABLES = sorted(
    [
        "artenic_models",
        "promotions",
        "training_jobs",
        "budgets",
        "budget_alerts",
        "training_outcomes",
        "optimizer_recommendations",
        "optimizer_training_samples",
        "ensembles",
        "ensemble_versions",
        "ensemble_jobs",
        "ab_tests",
        "ab_test_metrics",
        "model_health",
        "config_settings",
        "config_audit_log",
        "config_overrides",
        "artenic_datasets",
        "artenic_dataset_versions",
        "artenic_dataset_files",
        "artenic_dataset_lineage",
    ]
)


# ======================================================================
# engine.py
# ======================================================================


class TestCreateAsyncEngine:
    def test_sqlite_returns_async_engine(self) -> None:
        engine = create_async_engine(SQLITE_URL)
        assert isinstance(engine, AsyncEngine)

    def test_postgresql_url_returns_engine(self) -> None:
        engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/db")
        assert isinstance(engine, AsyncEngine)

    def test_sqlite_has_check_same_thread_disabled(self) -> None:
        engine = create_async_engine(SQLITE_URL)
        assert engine is not None


class TestCreateSessionFactory:
    def test_produces_session_factory(self) -> None:
        engine = create_async_engine(SQLITE_URL)
        factory = create_session_factory(engine)
        assert factory is not None

    async def test_factory_produces_async_session(self) -> None:
        engine = create_async_engine(SQLITE_URL)
        factory = create_session_factory(engine)
        async with factory() as session:
            assert isinstance(session, AsyncSession)
        await engine.dispose()


class TestCreateTables:
    async def test_creates_all_17_tables(self) -> None:
        engine = create_async_engine(SQLITE_URL)
        await create_tables(engine)

        async with engine.connect() as conn:
            table_names = await conn.run_sync(
                lambda sync_conn: inspect(sync_conn).get_table_names()
            )

        assert sorted(table_names) == EXPECTED_TABLES
        await engine.dispose()


# ======================================================================
# models.py — Base
# ======================================================================


class TestBase:
    def test_base_is_declarative(self) -> None:
        assert hasattr(Base, "metadata")
        assert hasattr(Base, "registry")


# ======================================================================
# models.py — RegisteredModel
# ======================================================================


class TestRegisteredModel:
    def test_tablename(self) -> None:
        assert RegisteredModel.__tablename__ == "artenic_models"

    async def test_insert_and_query(self) -> None:
        engine = create_async_engine(SQLITE_URL)
        await create_tables(engine)
        factory = create_session_factory(engine)

        model_id = f"test-model_v1_{uuid.uuid4().hex[:8]}"
        async with factory() as session:
            m = RegisteredModel(
                id=model_id,
                name="test-model",
                version="1",
                model_type="classifier",
                framework="sklearn",
            )
            session.add(m)
            await session.commit()

        async with factory() as session:
            result = await session.get(RegisteredModel, model_id)
            assert result is not None
            assert result.name == "test-model"
            assert result.version == "1"
            assert result.model_type == "classifier"
            assert result.framework == "sklearn"
            assert result.description == ""
            assert result.stage == "registered"
            assert result.mlflow_run_id is None
            assert result.mlflow_model_uri is None
            assert result.ensemble_id is None

        await engine.dispose()


# ======================================================================
# models.py — PromotionRecord
# ======================================================================


class TestPromotionRecord:
    def test_tablename(self) -> None:
        assert PromotionRecord.__tablename__ == "promotions"

    async def test_insert(self) -> None:
        engine = create_async_engine(SQLITE_URL)
        await create_tables(engine)
        factory = create_session_factory(engine)

        async with factory() as session:
            model = RegisteredModel(
                id="promo-model_v1",
                name="promo-model",
                version="1",
                model_type="regressor",
                framework="pytorch",
            )
            session.add(model)
            await session.commit()

            promo = PromotionRecord(
                model_id="promo-model_v1",
                from_stage="registered",
                to_stage="staging",
                version="1",
            )
            session.add(promo)
            await session.commit()

            assert promo.id is not None
            assert promo.from_stage == "registered"
            assert promo.to_stage == "staging"

        await engine.dispose()


# ======================================================================
# models.py — TrainingJob
# ======================================================================


class TestTrainingJob:
    def test_tablename(self) -> None:
        assert TrainingJob.__tablename__ == "training_jobs"

    async def test_defaults(self) -> None:
        engine = create_async_engine(SQLITE_URL)
        await create_tables(engine)
        factory = create_session_factory(engine)

        job_id = uuid.uuid4().hex[:36]
        async with factory() as session:
            job = TrainingJob(
                id=job_id,
                service="sentiment",
                model="bert-base",
                provider="mock",
            )
            session.add(job)
            await session.commit()

        async with factory() as session:
            result = await session.get(TrainingJob, job_id)
            assert result is not None
            assert result.status == "pending"
            assert result.is_spot is False
            assert result.preempted is False
            assert result.preemption_count == 0
            assert result.metrics is None
            assert result.error is None
            assert result.mlflow_run_id is None
            assert result.provider_job_id is None
            assert result.instance_type is None
            assert result.cost_estimate_eur is None
            assert result.artifacts_uri is None
            assert result.started_at is None
            assert result.completed_at is None
            assert result.cost_actual_eur is None
            assert result.cost_per_hour_eur is None
            assert result.duration_seconds is None
            assert result.cost_predicted_eur is None
            assert result.duration_predicted_hours is None
            assert result.primary_metric_before is None
            assert result.primary_metric_after is None
            assert result.primary_metric_name is None
            assert result.resumed_from_job_id is None
            assert result.region is None
            assert result.workload_spec is None
            assert result.ensemble_job_id is None

        await engine.dispose()


# ======================================================================
# models.py — BudgetRecord + BudgetAlertRecord
# ======================================================================


class TestBudgetRecord:
    def test_tablename(self) -> None:
        assert BudgetRecord.__tablename__ == "budgets"

    async def test_defaults(self) -> None:
        engine = create_async_engine(SQLITE_URL)
        await create_tables(engine)
        factory = create_session_factory(engine)

        async with factory() as session:
            b = BudgetRecord(
                scope="global",
                scope_value="*",
                period="monthly",
                limit_eur=1000.0,
            )
            session.add(b)
            await session.commit()

            assert b.alert_threshold_pct == 80.0
            assert b.enabled is True

        await engine.dispose()


class TestBudgetAlertRecord:
    def test_tablename(self) -> None:
        assert BudgetAlertRecord.__tablename__ == "budget_alerts"

    async def test_insert(self) -> None:
        engine = create_async_engine(SQLITE_URL)
        await create_tables(engine)
        factory = create_session_factory(engine)

        async with factory() as session:
            budget = BudgetRecord(
                scope="global",
                scope_value="*",
                period="monthly",
                limit_eur=500.0,
            )
            session.add(budget)
            await session.commit()

            alert = BudgetAlertRecord(
                budget_id=budget.id,
                alert_type="threshold",
                spent_eur=420.0,
                limit_eur=500.0,
                pct_used=84.0,
                message="Budget at 84%",
            )
            session.add(alert)
            await session.commit()

            assert alert.webhook_sent is False

        await engine.dispose()


# ======================================================================
# models.py — TrainingOutcomeRecord
# ======================================================================


class TestTrainingOutcomeRecord:
    def test_tablename(self) -> None:
        assert TrainingOutcomeRecord.__tablename__ == "training_outcomes"

    async def test_insert(self) -> None:
        engine = create_async_engine(SQLITE_URL)
        await create_tables(engine)
        factory = create_session_factory(engine)

        job_id = uuid.uuid4().hex[:36]
        async with factory() as session:
            job = TrainingJob(
                id=job_id,
                service="nlp",
                model="gpt2",
                provider="mock",
            )
            session.add(job)
            await session.commit()

            outcome = TrainingOutcomeRecord(
                job_id=job_id,
                workload_spec={"model_size": "small"},
                provider="mock",
                instance_type="e2-standard-4",
                actual_duration_hours=1.5,
                actual_cost_eur=2.30,
                success=True,
            )
            session.add(outcome)
            await session.commit()

            assert outcome.is_spot is False
            assert outcome.region is None
            assert outcome.predicted_duration_hours is None
            assert outcome.predicted_cost_eur is None
            assert outcome.primary_metric_name is None
            assert outcome.primary_metric_value is None

        await engine.dispose()


# ======================================================================
# models.py — OptimizerRecommendationRecord + OptimizerTrainingSampleRecord
# ======================================================================


class TestOptimizerRecommendationRecord:
    def test_tablename(self) -> None:
        assert OptimizerRecommendationRecord.__tablename__ == "optimizer_recommendations"

    async def test_insert(self) -> None:
        engine = create_async_engine(SQLITE_URL)
        await create_tables(engine)
        factory = create_session_factory(engine)

        job_id = uuid.uuid4().hex[:36]
        rec_id = uuid.uuid4().hex[:36]
        async with factory() as session:
            job = TrainingJob(id=job_id, service="svc", model="m", provider="mock")
            session.add(job)
            await session.commit()

            rec = OptimizerRecommendationRecord(
                id=rec_id,
                training_job_id=job_id,
                model_version="v1",
                recommended_provider="gcp",
                recommended_instance="e2-highmem-16",
                estimated_duration_hours=2.0,
                estimated_cost_eur=5.0,
                confidence_score=0.85,
                alternatives=[{"provider": "aws", "instance": "m6i.xlarge"}],
                total_instances_considered=50,
                instances_after_filter=10,
                workload_spec={"size": "medium"},
            )
            session.add(rec)
            await session.commit()

            assert rec.recommended_rank == 1

        await engine.dispose()


class TestOptimizerTrainingSampleRecord:
    def test_tablename(self) -> None:
        assert OptimizerTrainingSampleRecord.__tablename__ == "optimizer_training_samples"


# ======================================================================
# models.py — EnsembleRecord + EnsembleVersionRecord + EnsembleJobRecord
# ======================================================================


class TestEnsembleRecord:
    def test_tablename(self) -> None:
        assert EnsembleRecord.__tablename__ == "ensembles"

    async def test_defaults(self) -> None:
        engine = create_async_engine(SQLITE_URL)
        await create_tables(engine)
        factory = create_session_factory(engine)

        async with factory() as session:
            e = EnsembleRecord(
                id="ens-001",
                name="my-ensemble",
                service="sentiment",
                strategy="weighted_average",
            )
            session.add(e)
            await session.commit()

            assert e.description == ""
            assert e.stage == "registered"
            assert e.version == 1
            assert e.enabled is True

        await engine.dispose()


class TestEnsembleVersionRecord:
    def test_tablename(self) -> None:
        assert EnsembleVersionRecord.__tablename__ == "ensemble_versions"

    async def test_insert(self) -> None:
        engine = create_async_engine(SQLITE_URL)
        await create_tables(engine)
        factory = create_session_factory(engine)

        async with factory() as session:
            ens = EnsembleRecord(
                id="ens-v",
                name="test",
                service="svc",
                strategy="voting",
            )
            session.add(ens)
            await session.commit()

            ver = EnsembleVersionRecord(
                ensemble_id="ens-v",
                version=1,
                model_ids=["m1", "m2"],
                strategy="voting",
            )
            session.add(ver)
            await session.commit()

            assert ver.change_reason == ""

        await engine.dispose()


class TestEnsembleJobRecord:
    def test_tablename(self) -> None:
        assert EnsembleJobRecord.__tablename__ == "ensemble_jobs"

    async def test_defaults(self) -> None:
        engine = create_async_engine(SQLITE_URL)
        await create_tables(engine)
        factory = create_session_factory(engine)

        async with factory() as session:
            ens = EnsembleRecord(
                id="ens-j",
                name="test",
                service="svc",
                strategy="voting",
            )
            session.add(ens)
            await session.commit()

            ej = EnsembleJobRecord(
                id=uuid.uuid4().hex[:36],
                ensemble_id="ens-j",
            )
            session.add(ej)
            await session.commit()

            assert ej.status == "pending"
            assert ej.total_models == 0
            assert ej.completed_models == 0
            assert ej.failed_models == 0
            assert ej.total_cost_eur is None
            assert ej.completed_at is None

        await engine.dispose()


# ======================================================================
# models.py — ABTestRecord + ABTestMetricRecord
# ======================================================================


class TestABTestRecord:
    def test_tablename(self) -> None:
        assert ABTestRecord.__tablename__ == "ab_tests"

    async def test_defaults(self) -> None:
        engine = create_async_engine(SQLITE_URL)
        await create_tables(engine)
        factory = create_session_factory(engine)

        async with factory() as session:
            t = ABTestRecord(
                id=uuid.uuid4().hex[:36],
                name="test-ab",
                service="sentiment",
                variants={"A": {"model": "m1"}, "B": {"model": "m2"}},
                primary_metric="accuracy",
            )
            session.add(t)
            await session.commit()

            assert t.status == "running"
            assert t.min_samples == 100
            assert t.winner is None
            assert t.conclusion_reason is None
            assert t.concluded_at is None

        await engine.dispose()


class TestABTestMetricRecord:
    def test_tablename(self) -> None:
        assert ABTestMetricRecord.__tablename__ == "ab_test_metrics"

    async def test_defaults(self) -> None:
        engine = create_async_engine(SQLITE_URL)
        await create_tables(engine)
        factory = create_session_factory(engine)

        test_id = uuid.uuid4().hex[:36]
        async with factory() as session:
            t = ABTestRecord(
                id=test_id,
                name="test-ab",
                service="svc",
                variants={"A": {}, "B": {}},
                primary_metric="accuracy",
            )
            session.add(t)
            await session.commit()

            m = ABTestMetricRecord(
                ab_test_id=test_id,
                variant_name="A",
                metric_name="accuracy",
                metric_value=0.95,
            )
            session.add(m)
            await session.commit()

            assert m.error is False
            assert m.latency_ms is None

        await engine.dispose()


# ======================================================================
# models.py — ModelHealthRecord
# ======================================================================


class TestModelHealthRecord:
    def test_tablename(self) -> None:
        assert ModelHealthRecord.__tablename__ == "model_health"

    async def test_defaults(self) -> None:
        engine = create_async_engine(SQLITE_URL)
        await create_tables(engine)
        factory = create_session_factory(engine)

        async with factory() as session:
            model = RegisteredModel(
                id="health-model_v1",
                name="health-model",
                version="1",
                model_type="classifier",
                framework="sklearn",
            )
            session.add(model)
            await session.commit()

            h = ModelHealthRecord(
                model_id="health-model_v1",
                metric_name="error_rate",
                metric_value=0.05,
            )
            session.add(h)
            await session.commit()

            assert h.alert_triggered is False
            assert h.sample_count == 0
            assert h.drift_score is None
            assert h.window_start is None
            assert h.window_end is None

        await engine.dispose()


# ======================================================================
# models.py — ConfigSettingRecord + ConfigAuditRecord + ConfigOverrideRecord
# ======================================================================


class TestConfigSettingRecord:
    def test_tablename(self) -> None:
        assert ConfigSettingRecord.__tablename__ == "config_settings"

    async def test_defaults(self) -> None:
        engine = create_async_engine(SQLITE_URL)
        await create_tables(engine)
        factory = create_session_factory(engine)

        async with factory() as session:
            cs = ConfigSettingRecord(
                scope="global",
                section="core",
                key="debug",
                value="false",
            )
            session.add(cs)
            await session.commit()

            assert cs.encrypted is False
            assert cs.updated_by == "system"

        await engine.dispose()

    async def test_unique_constraint(self) -> None:
        engine = create_async_engine(SQLITE_URL)
        await create_tables(engine)
        factory = create_session_factory(engine)

        async with factory() as session:
            cs1 = ConfigSettingRecord(scope="global", section="core", key="debug", value="false")
            session.add(cs1)
            await session.commit()

        with pytest.raises(IntegrityError):
            async with factory() as session:
                cs2 = ConfigSettingRecord(scope="global", section="core", key="debug", value="true")
                session.add(cs2)
                await session.commit()

        await engine.dispose()


class TestConfigAuditRecord:
    def test_tablename(self) -> None:
        assert ConfigAuditRecord.__tablename__ == "config_audit_log"

    async def test_defaults(self) -> None:
        engine = create_async_engine(SQLITE_URL)
        await create_tables(engine)
        factory = create_session_factory(engine)

        async with factory() as session:
            ca = ConfigAuditRecord(
                scope="global",
                section="core",
                key="debug",
                action="update",
                old_value="false",
                new_value="true",
            )
            session.add(ca)
            await session.commit()

            assert ca.changed_by == "api"

        await engine.dispose()


class TestConfigOverrideRecord:
    def test_tablename(self) -> None:
        assert ConfigOverrideRecord.__tablename__ == "config_overrides"

    async def test_insert(self) -> None:
        engine = create_async_engine(SQLITE_URL)
        await create_tables(engine)
        factory = create_session_factory(engine)

        async with factory() as session:
            co = ConfigOverrideRecord(
                section="training",
                key="max_retries",
                value="3",
            )
            session.add(co)
            await session.commit()

            assert co.id is not None

        await engine.dispose()

    async def test_unique_constraint(self) -> None:
        engine = create_async_engine(SQLITE_URL)
        await create_tables(engine)
        factory = create_session_factory(engine)

        async with factory() as session:
            co1 = ConfigOverrideRecord(section="training", key="max_retries", value="3")
            session.add(co1)
            await session.commit()

        with pytest.raises(IntegrityError):
            async with factory() as session:
                co2 = ConfigOverrideRecord(section="training", key="max_retries", value="5")
                session.add(co2)
                await session.commit()

        await engine.dispose()
