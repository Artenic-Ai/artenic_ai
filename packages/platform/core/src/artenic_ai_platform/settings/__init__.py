"""Pydantic-settings configuration for the Artenic AI Platform.

All configuration classes are re-exported here for backward compatibility.
Import from this package as before:

    from artenic_ai_platform.settings import PlatformSettings, GCPConfig, ...
"""

from __future__ import annotations

from artenic_ai_platform.settings.core import PlatformSettings
from artenic_ai_platform.settings.features import (
    ABTestConfig,
    BudgetConfig,
    CanaryConfig,
    DatasetConfig,
    DatasetStorageConfig,
    EnsembleConfig,
    HealthMonitoringConfig,
    SpotConfig,
    WebhookConfig,
)
from artenic_ai_platform.settings.providers import (
    AWSConfig,
    AzureConfig,
    CoreWeaveConfig,
    GCPConfig,
    HetznerConfig,
    KubernetesConfig,
    LambdaLabsConfig,
    LocalConfig,
    OCIConfig,
    OpenStackConfig,
    RunPodConfig,
    S3StorageConfig,
    ScalewayConfig,
    VastAIConfig,
)

__all__ = [
    "ABTestConfig",
    "AWSConfig",
    "AzureConfig",
    "BudgetConfig",
    "CanaryConfig",
    "CoreWeaveConfig",
    "DatasetConfig",
    "DatasetStorageConfig",
    "EnsembleConfig",
    "GCPConfig",
    "HealthMonitoringConfig",
    "HetznerConfig",
    "KubernetesConfig",
    "LambdaLabsConfig",
    "LocalConfig",
    "OCIConfig",
    "OpenStackConfig",
    "PlatformSettings",
    "RunPodConfig",
    "S3StorageConfig",
    "ScalewayConfig",
    "SpotConfig",
    "VastAIConfig",
    "WebhookConfig",
]
