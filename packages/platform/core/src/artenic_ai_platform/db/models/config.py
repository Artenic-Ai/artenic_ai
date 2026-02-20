"""Configuration ORM models."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 - SQLAlchemy needs runtime access

from sqlalchemy import (
    Boolean,
    DateTime,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from artenic_ai_platform.db.models.base import Base


class ConfigSettingRecord(Base):
    """Persisted configuration key-value pair."""

    __tablename__ = "config_settings"
    __table_args__ = (UniqueConstraint("scope", "section", "key"),)

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    scope: Mapped[str] = mapped_column(String(50))
    section: Mapped[str] = mapped_column(String(100))
    key: Mapped[str] = mapped_column(String(100))
    value: Mapped[str] = mapped_column(Text)
    encrypted: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_by: Mapped[str] = mapped_column(
        String(100),
        default="system",
    )


class ConfigAuditRecord(Base):
    """Audit log entry for configuration changes."""

    __tablename__ = "config_audit_log"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    scope: Mapped[str] = mapped_column(String(50))
    section: Mapped[str] = mapped_column(String(100))
    key: Mapped[str] = mapped_column(String(100))
    old_value: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    new_value: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    action: Mapped[str] = mapped_column(String(20))
    changed_by: Mapped[str] = mapped_column(
        String(100),
        default="api",
    )
    changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class ConfigOverrideRecord(Base):
    """Configuration override (section + key)."""

    __tablename__ = "config_overrides"
    __table_args__ = (UniqueConstraint("section", "key"),)

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    section: Mapped[str] = mapped_column(String(100))
    key: Mapped[str] = mapped_column(String(100))
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
