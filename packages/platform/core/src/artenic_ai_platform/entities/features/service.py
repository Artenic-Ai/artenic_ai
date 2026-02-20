"""Feature service — pure CRUD, no special behavior."""

from __future__ import annotations

from artenic_ai_platform.db.models import MLFeature
from artenic_ai_platform.entities.base_service import GenericEntityService


class FeatureService(GenericEntityService[MLFeature]):
    """Service for feature schema definitions — pure CRUD."""

    _model_class = MLFeature
