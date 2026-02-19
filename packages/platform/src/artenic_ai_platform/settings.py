"""Pydantic-settings configuration for the Artenic AI Platform."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Reusable sub-configs
# ---------------------------------------------------------------------------


class S3StorageConfig(BaseModel):
    """Reusable S3-compatible storage configuration."""

    endpoint_url: str = ""
    access_key: str = ""
    secret_key: str = ""
    code_bucket: str = "artenic-code"
    models_bucket: str = "artenic-models"
    region: str = ""


# ---------------------------------------------------------------------------
# Feature sub-configs
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Cloud-provider configs
# ---------------------------------------------------------------------------


class GCPConfig(BaseModel):
    """Google Cloud Platform provider."""

    enabled: bool = False
    project_id: str = ""
    credentials_path: str = ""
    region: str = "europe-west4"
    zone: str = "europe-west4-a"
    code_bucket: str = "artenic-code"
    data_bucket: str = "artenic-training-data"
    models_bucket: str = "artenic-models"
    default_machine_type: str = "e2-highmem-16"
    ssh_key_path: str = ""


class AWSConfig(BaseModel):
    """Amazon Web Services provider."""

    enabled: bool = False
    access_key_id: str = ""
    secret_access_key: str = ""
    session_token: str = ""
    region: str = "eu-west-1"
    code_bucket: str = "artenic-code"
    models_bucket: str = "artenic-models"
    default_instance_type: str = "m6i.xlarge"
    ami_id: str = ""
    subnet_id: str = ""
    security_group_id: str = ""
    key_name: str = ""
    iam_instance_profile: str = ""


class AzureConfig(BaseModel):
    """Microsoft Azure provider."""

    enabled: bool = False
    subscription_id: str = ""
    tenant_id: str = ""
    client_id: str = ""
    client_secret: str = ""
    region: str = "switzerlandnorth"
    resource_group: str = "artenic-ai"
    storage_account: str = "artenicai"
    code_container: str = "artenic-code"
    models_container: str = "artenic-models"
    storage_connection_string: str = ""
    default_vm_size: str = "Standard_D4s_v5"
    image_publisher: str = "Canonical"
    image_offer: str = "0001-com-ubuntu-server-jammy"
    image_sku: str = "22_04-lts-gen2"
    vnet_name: str = ""
    subnet_name: str = ""
    ssh_key_path: str = ""


class OCIConfig(BaseModel):
    """Oracle Cloud Infrastructure provider."""

    enabled: bool = False
    config_file: str = "~/.oci/config"
    config_profile: str = "DEFAULT"
    tenancy_ocid: str = ""
    user_ocid: str = ""
    fingerprint: str = ""
    key_file: str = ""
    region: str = "eu-frankfurt-1"
    compartment_id: str = ""
    availability_domain: str = ""
    default_shape: str = "VM.Standard.E4.Flex"
    subnet_id: str = ""
    ssh_public_key: str = ""
    image_id: str = ""
    namespace: str = ""
    code_bucket: str = "artenic-code"
    models_bucket: str = "artenic-models"


class OpenStackConfig(BaseModel):
    """OpenStack-based provider (OVH, Infomaniak, etc.)."""

    enabled: bool = False
    provider_label: str = ""
    auth_url: str = ""
    username: str = ""
    password: str = ""
    application_credential_id: str = ""
    application_credential_secret: str = ""
    project_id: str = ""
    project_name: str = ""
    user_domain_name: str = "Default"
    project_domain_name: str = "Default"
    region: str = ""
    code_container: str = "artenic-code"
    models_container: str = "artenic-models"
    default_flavor: str = ""
    network_name: str = ""
    image_name: str = ""
    ssh_key_name: str = ""
    security_group: str = "default"


class HetznerConfig(BaseModel):
    """Hetzner Cloud provider."""

    enabled: bool = False
    api_key: str = ""
    api_url: str = "https://api.hetzner.cloud/v1"
    location: str = "fsn1"
    ssh_key_name: str = ""
    default_server_type: str = "ccx23"
    image: str = "ubuntu-22.04"
    storage: S3StorageConfig = Field(
        default_factory=S3StorageConfig,
    )


class ScalewayConfig(BaseModel):
    """Scaleway provider."""

    enabled: bool = False
    secret_key: str = ""
    access_key: str = ""
    project_id: str = ""
    zone: str = "fr-par-2"
    default_instance_type: str = ""
    image_id: str = ""
    storage_endpoint: str = ""
    storage_region: str = ""
    code_bucket: str = "artenic-code"
    models_bucket: str = "artenic-models"


class LambdaLabsConfig(BaseModel):
    """Lambda Labs provider."""

    enabled: bool = False
    api_key: str = ""
    api_url: str = "https://cloud.lambdalabs.com/api/v1"
    ssh_key_name: str = ""
    default_instance_type: str = ""
    region: str = "us-east-1"
    storage: S3StorageConfig = Field(
        default_factory=S3StorageConfig,
    )


class RunPodConfig(BaseModel):
    """RunPod provider."""

    enabled: bool = False
    api_key: str = ""
    api_url: str = "https://api.runpod.io/v2"
    default_gpu_type: str = ""
    cloud_type: str = "ALL"
    docker_image: str = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
    storage: S3StorageConfig = Field(
        default_factory=S3StorageConfig,
    )


class CoreWeaveConfig(BaseModel):
    """CoreWeave Kubernetes provider."""

    enabled: bool = False
    kubeconfig_path: str = ""
    namespace: str = "artenic-training"
    training_image: str = "nvcr.io/nvidia/pytorch:24.01-py3"
    storage_class: str = "shared-hdd"
    node_selector: dict[str, str] = Field(default_factory=dict)
    tolerations: list[dict[str, str]] = Field(
        default_factory=list,
    )
    image_pull_secrets: list[str] = Field(default_factory=list)
    service_account: str = ""
    code_pvc_size: str = "10Gi"
    artifacts_pvc_size: str = "50Gi"


class VastAIConfig(BaseModel):
    """Vast.ai provider."""

    enabled: bool = False
    api_key: str = ""
    api_url: str = "https://console.vast.ai/api/v0"
    docker_image: str = "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
    max_price_per_hour: float = 5.0
    min_reliability: float = 0.95
    storage: S3StorageConfig = Field(
        default_factory=S3StorageConfig,
    )


class KubernetesConfig(BaseModel):
    """Generic Kubernetes provider."""

    enabled: bool = False
    provider_label: str = ""
    kubeconfig_path: str = ""
    namespace: str = "artenic-training"
    training_image: str = "python:3.12-slim"
    storage_class: str = ""
    node_selector: dict[str, str] = Field(default_factory=dict)
    tolerations: list[dict[str, str]] = Field(
        default_factory=list,
    )
    image_pull_secrets: list[str] = Field(default_factory=list)
    service_account: str = ""
    code_pvc_size: str = "10Gi"
    artifacts_pvc_size: str = "50Gi"


class LocalConfig(BaseModel):
    """Local subprocess training provider."""

    enabled: bool = False
    work_dir: str = ""
    max_concurrent_jobs: int = 4
    default_timeout_hours: float = 24.0
    gpu_enabled: bool = False
    python_executable: str = ""


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------


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
