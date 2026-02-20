"""Provider ORM models."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 - SQLAlchemy needs runtime access

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from artenic_ai_platform.db.models.base import Base


class ProviderRecord(Base):
    """A configured cloud provider (OVH, AWS, GCP, etc.).

    Credentials are stored as JSON encrypted via SecretManager.
    Config holds non-sensitive settings (region, domain, etc.).
    """

    __tablename__ = "providers"

    id: Mapped[str] = mapped_column(
        String(50),
        primary_key=True,
    )
    display_name: Mapped[str] = mapped_column(String(255))
    enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )
    credentials: Mapped[str] = mapped_column(
        Text,
        default="",
    )
    config: Mapped[dict] = mapped_column(  # type: ignore[type-arg]
        JSON,
        default=dict,
    )
    status: Mapped[str] = mapped_column(
        String(50),
        default="unconfigured",
    )
    status_message: Mapped[str] = mapped_column(
        Text,
        default="",
    )
    last_checked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        onupdate=func.now(),
    )
