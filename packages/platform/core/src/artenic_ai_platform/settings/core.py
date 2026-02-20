"""Root PlatformSettings and core configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from artenic_ai_platform.settings.features import (
    ABTestConfig,
    BudgetConfig,
    CanaryConfig,
    DatasetConfig,
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
    ScalewayConfig,
    VastAIConfig,
)


class PlatformSettings(BaseSettings):
    """Central configuration for the Artenic AI Platform.

    All values can be overridden via environment variables prefixed
    with ``ARTENIC_``.  Nested models use ``__`` as a delimiter,
    e.g. ``ARTENIC_GCP__PROJECT_ID``.
    """

    model_config = SettingsConfigDict(
        env_prefix="ARTENIC_",
        env_nested_delimiter="__",
    )

    # -- Core -----------------------------------------------------------------

    host: str = "0.0.0.0"
    port: int = 9000
    debug: bool = False
    api_key: str = Field(
        default="",
        description="Empty string disables auth (dev only).",
    )
    secret_key: str = Field(
        default="",
        description="Fernet master key for encryption.",
    )
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/artenic_ai"

    # -- MLflow ---------------------------------------------------------------

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_artifact_root: str = "./mlflow-artifacts"
    mlflow_experiment_prefix: str = "artenic"

    # -- OpenTelemetry --------------------------------------------------------

    otel_enabled: bool = True
    otel_service_name: str = "artenic-ai-platform"
    otel_exporter: Literal["prometheus", "otlp"] = "prometheus"

    # -- CORS -----------------------------------------------------------------

    cors_origins: list[str] = Field(
        default=["http://localhost:3000"],
    )

    # -- Rate limiting --------------------------------------------------------

    rate_limit_per_minute: int = 60
    rate_limit_burst: int = 10

    # -- Feature sub-configs --------------------------------------------------

    budget: BudgetConfig = Field(
        default_factory=BudgetConfig,
    )
    webhook: WebhookConfig = Field(
        default_factory=WebhookConfig,
    )
    spot: SpotConfig = Field(default_factory=SpotConfig)
    ensemble: EnsembleConfig = Field(
        default_factory=EnsembleConfig,
    )
    ab_test: ABTestConfig = Field(
        default_factory=ABTestConfig,
    )
    health: HealthMonitoringConfig = Field(
        default_factory=HealthMonitoringConfig,
    )
    canary: CanaryConfig = Field(
        default_factory=CanaryConfig,
    )
    dataset: DatasetConfig = Field(
        default_factory=DatasetConfig,
    )

    # -- Cloud-provider configs -----------------------------------------------

    gcp: GCPConfig = Field(default_factory=GCPConfig)
    aws: AWSConfig = Field(default_factory=AWSConfig)
    azure: AzureConfig = Field(default_factory=AzureConfig)
    oci: OCIConfig = Field(default_factory=OCIConfig)

    openstack: OpenStackConfig = Field(
        default_factory=OpenStackConfig,
    )
    ovh: OpenStackConfig = Field(
        default_factory=lambda: OpenStackConfig(
            provider_label="ovh",
            auth_url="https://auth.cloud.ovh.net/v3",
            region="GRA11",
            default_flavor="b2-30",
            network_name="Ext-Net",
            image_name="Ubuntu 22.04",
        ),
    )
    infomaniak: OpenStackConfig = Field(
        default_factory=lambda: OpenStackConfig(
            provider_label="infomaniak",
            auth_url=("https://api.pub1.infomaniak.cloud/identity/v3"),
            region="dc3-a",
            default_flavor="a4-ram8-disk0",
            network_name="ext-net1",
            image_name="Ubuntu 22.04 LTS Jammy Jellyfish",
        ),
    )

    hetzner: HetznerConfig = Field(
        default_factory=HetznerConfig,
    )
    scaleway: ScalewayConfig = Field(
        default_factory=ScalewayConfig,
    )
    lambda_labs: LambdaLabsConfig = Field(
        default_factory=LambdaLabsConfig,
    )
    runpod: RunPodConfig = Field(
        default_factory=RunPodConfig,
    )
    coreweave: CoreWeaveConfig = Field(
        default_factory=CoreWeaveConfig,
    )
    vastai: VastAIConfig = Field(
        default_factory=VastAIConfig,
    )
    kubernetes: KubernetesConfig = Field(
        default_factory=KubernetesConfig,
    )
    local: LocalConfig = Field(default_factory=LocalConfig)
