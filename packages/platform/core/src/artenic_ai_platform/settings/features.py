"""Feature sub-configuration models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BudgetConfig(BaseModel):
    """Budget guardrails for training runs."""

    enabled: bool = False
    enforcement_mode: Literal["block", "warn"] = "block"
    alert_threshold_pct: float = Field(
        default=80.0,
        description="Percentage of budget at which an alert is raised.",
    )


class WebhookConfig(BaseModel):
    """Outgoing webhook notifications."""

    enabled: bool = False
    url: str = ""
    secret: str = ""
    timeout_seconds: float = 10.0
    retry_count: int = 2


class SpotConfig(BaseModel):
    """Spot / preemptible instance settings."""

    enabled: bool = False
    prefer_spot: bool = False
    max_preemption_retries: int = 3
    failover_regions: list[str] = Field(default_factory=list)
    checkpoint_before_resume: bool = True


class EnsembleConfig(BaseModel):
    """Model-ensemble management."""

    enabled: bool = False
    max_models_per_ensemble: int = 10
    auto_prune_enabled: bool = False
    prune_weight_threshold: float = 0.05
    prune_min_models: int = 2


class ABTestConfig(BaseModel):
    """A/B testing defaults."""

    enabled: bool = False
    default_min_samples: int = 100
    max_concurrent_tests: int = 5


class HealthMonitoringConfig(BaseModel):
    """Model-health monitoring and drift detection."""

    enabled: bool = False
    check_interval_seconds: float = 300.0
    drift_threshold: float = 0.3
    degradation_pct: float = 10.0
    retention_days: int = 90


class CanaryConfig(BaseModel):
    """Canary-deployment strategy."""

    enabled: bool = False
    initial_traffic_pct: float = 10.0
    evaluation_window_seconds: float = 3600.0
    auto_promote_threshold: float = 0.95
    auto_rollback_threshold: float = 0.80


class DatasetStorageConfig(BaseModel):
    """Dataset storage backend configuration."""

    backend: Literal["filesystem", "s3", "gcs", "azure", "ovh"] = "filesystem"
    # Filesystem
    local_dir: str = "data/datasets"
    # Cloud common
    bucket: str = ""
    container: str = ""
    prefix: str = "datasets/"
    # S3 / OVH
    endpoint_url: str = ""
    access_key: str = ""
    secret_key: str = ""
    region: str = ""
    # GCS
    credentials_path: str = ""
    project_id: str = ""
    # Azure
    connection_string: str = ""


class DatasetConfig(BaseModel):
    """Dataset management configuration."""

    enabled: bool = True
    storage: DatasetStorageConfig = Field(
        default_factory=DatasetStorageConfig,
    )
    max_upload_size_mb: int = 500
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [
            "csv",
            "parquet",
            "json",
            "jsonl",
            "png",
            "jpg",
            "jpeg",
            "wav",
            "mp3",
            "txt",
            "tsv",
            "feather",
            "arrow",
        ],
    )
