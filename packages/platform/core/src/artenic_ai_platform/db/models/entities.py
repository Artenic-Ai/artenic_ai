"""ML entity ORM models (ml_* tables)."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 - SQLAlchemy needs runtime access

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from artenic_ai_platform.db.models.base import Base


class MLDataset(Base):
    """A managed dataset with opaque metadata."""

    __tablename__ = "ml_datasets"
    __table_args__ = (UniqueConstraint("name", "version"),)

    id: Mapped[str] = mapped_column(String(500), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    version: Mapped[int] = mapped_column(Integer, default=1)
    format: Mapped[str] = mapped_column(String(50))
    description: Mapped[str] = mapped_column(Text, default="")
    metadata_: Mapped[dict] = mapped_column(  # type: ignore[type-arg]
        "metadata",
        JSON,
        default=dict,
    )
    status: Mapped[str] = mapped_column(String(50), default="created")
    total_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class MLDatasetFile(Base):
    """A single file within a dataset."""

    __tablename__ = "ml_dataset_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(
        String(500),
        ForeignKey("ml_datasets.id", ondelete="CASCADE"),
    )
    path: Mapped[str] = mapped_column(String(1000))
    sha256: Mapped[str] = mapped_column(String(64), default="")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    num_records: Mapped[int | None] = mapped_column(Integer, nullable=True)
    storage_path: Mapped[str] = mapped_column(String(1000))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class MLModel(Base):
    """A registered ML model with opaque metadata."""

    __tablename__ = "ml_models"
    __table_args__ = (UniqueConstraint("name", "version"),)

    id: Mapped[str] = mapped_column(String(500), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    version: Mapped[int] = mapped_column(Integer, default=1)
    framework: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(Text, default="")
    metadata_: Mapped[dict] = mapped_column(  # type: ignore[type-arg]
        "metadata",
        JSON,
        default=dict,
    )
    metrics: Mapped[dict] = mapped_column(  # type: ignore[type-arg]
        JSON,
        default=dict,
    )
    stage: Mapped[str] = mapped_column(String(50), default="draft")
    artifact_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    artifact_format: Mapped[str | None] = mapped_column(String(50), nullable=True)
    artifact_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    artifact_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class MLRun(Base):
    """A training/experiment run (metadata registry, not execution)."""

    __tablename__ = "ml_runs"

    id: Mapped[str] = mapped_column(String(500), primary_key=True)
    config: Mapped[dict] = mapped_column(  # type: ignore[type-arg]
        JSON,
        default=dict,
    )
    status: Mapped[str] = mapped_column(String(50), default="pending")
    metrics: Mapped[dict] = mapped_column(  # type: ignore[type-arg]
        JSON,
        default=dict,
    )
    triggered_by: Mapped[str] = mapped_column(String(255), default="")
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class MLRunIO(Base):
    """Input/output entity reference for a run."""

    __tablename__ = "ml_run_io"
    __table_args__ = (UniqueConstraint("run_id", "entity_id", "direction"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(500),
        ForeignKey("ml_runs.id", ondelete="CASCADE"),
    )
    entity_id: Mapped[str] = mapped_column(String(500))
    direction: Mapped[str] = mapped_column(String(10))


class MLEnsemble(Base):
    """An ensemble of models with a strategy."""

    __tablename__ = "ml_ensembles"
    __table_args__ = (UniqueConstraint("name", "version"),)

    id: Mapped[str] = mapped_column(String(500), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    version: Mapped[int] = mapped_column(Integer, default=1)
    strategy_type: Mapped[str] = mapped_column(String(100))
    metadata_: Mapped[dict] = mapped_column(  # type: ignore[type-arg]
        "metadata",
        JSON,
        default=dict,
    )
    metrics: Mapped[dict] = mapped_column(  # type: ignore[type-arg]
        JSON,
        default=dict,
    )
    stage: Mapped[str] = mapped_column(String(50), default="staging")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class MLEnsembleModel(Base):
    """Junction table: ensemble -> model references."""

    __tablename__ = "ml_ensemble_models"
    __table_args__ = (UniqueConstraint("ensemble_id", "model_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ensemble_id: Mapped[str] = mapped_column(
        String(500),
        ForeignKey("ml_ensembles.id", ondelete="CASCADE"),
    )
    model_id: Mapped[str] = mapped_column(String(500))


class MLFeature(Base):
    """A feature schema definition."""

    __tablename__ = "ml_features"
    __table_args__ = (UniqueConstraint("name", "version"),)

    id: Mapped[str] = mapped_column(String(500), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    version: Mapped[int] = mapped_column(Integer, default=1)
    metadata_: Mapped[dict] = mapped_column(  # type: ignore[type-arg]
        "metadata",
        JSON,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class MLLineage(Base):
    """Unified lineage: source -> target with relation type."""

    __tablename__ = "ml_lineage"
    __table_args__ = (UniqueConstraint("source_id", "target_id", "relation_type"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[str] = mapped_column(String(500))
    target_id: Mapped[str] = mapped_column(String(500))
    relation_type: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
