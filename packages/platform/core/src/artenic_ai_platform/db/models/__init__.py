"""SQLAlchemy 2.0 ORM models for the Artenic AI platform.

All model classes are re-exported here for backward compatibility.
Import from this package as before:

    from artenic_ai_platform.db.models import Base, MLModel, TrainingJob, ...
"""

from __future__ import annotations

from artenic_ai_platform.db.models.base import Base
from artenic_ai_platform.db.models.config import (
    ConfigAuditRecord,
    ConfigOverrideRecord,
    ConfigSettingRecord,
)
from artenic_ai_platform.db.models.ensemble_legacy import (
    EnsembleJobRecord,
    EnsembleRecord,
    EnsembleVersionRecord,
    OptimizerRecommendationRecord,
    OptimizerTrainingSampleRecord,
)
from artenic_ai_platform.db.models.entities import (
    MLDataset,
    MLDatasetFile,
    MLEnsemble,
    MLEnsembleModel,
    MLFeature,
    MLLineage,
    MLModel,
    MLRun,
    MLRunIO,
)
from artenic_ai_platform.db.models.monitoring import (
    ABTestMetricRecord,
    ABTestRecord,
    BudgetAlertRecord,
    BudgetRecord,
    ModelHealthRecord,
)
from artenic_ai_platform.db.models.providers import ProviderRecord
from artenic_ai_platform.db.models.training import (
    TrainingJob,
    TrainingOutcomeRecord,
)

__all__ = [
    "ABTestMetricRecord",
    "ABTestRecord",
    "Base",
    "BudgetAlertRecord",
    "BudgetRecord",
    "ConfigAuditRecord",
    "ConfigOverrideRecord",
    "ConfigSettingRecord",
    "EnsembleJobRecord",
    "EnsembleRecord",
    "EnsembleVersionRecord",
    "MLDataset",
    "MLDatasetFile",
    "MLEnsemble",
    "MLEnsembleModel",
    "MLFeature",
    "MLLineage",
    "MLModel",
    "MLRun",
    "MLRunIO",
    "ModelHealthRecord",
    "OptimizerRecommendationRecord",
    "OptimizerTrainingSampleRecord",
    "ProviderRecord",
    "TrainingJob",
    "TrainingOutcomeRecord",
]
