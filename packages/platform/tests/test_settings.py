"""Tests for artenic_ai_platform.settings — 100% coverage."""

from __future__ import annotations

import pytest

from artenic_ai_platform.settings import (
    ABTestConfig,
    AWSConfig,
    AzureConfig,
    BudgetConfig,
    CanaryConfig,
    CoreWeaveConfig,
    EnsembleConfig,
    GCPConfig,
    HealthMonitoringConfig,
    HetznerConfig,
    KubernetesConfig,
    LambdaLabsConfig,
    LocalConfig,
    OCIConfig,
    OpenStackConfig,
    PlatformSettings,
    RunPodConfig,
    S3StorageConfig,
    ScalewayConfig,
    SpotConfig,
    VastAIConfig,
    WebhookConfig,
)

# ======================================================================
# S3StorageConfig
# ======================================================================


class TestS3StorageConfig:
    def test_defaults(self) -> None:
        cfg = S3StorageConfig()
        assert cfg.endpoint_url == ""
        assert cfg.access_key == ""
        assert cfg.secret_key == ""
        assert cfg.code_bucket == "artenic-code"
        assert cfg.models_bucket == "artenic-models"
        assert cfg.region == ""

    def test_custom_values(self) -> None:
        cfg = S3StorageConfig(
            endpoint_url="https://s3.example.com",
            access_key="ak",
            secret_key="sk",
            code_bucket="my-code",
            models_bucket="my-models",
            region="eu-west-1",
        )
        assert cfg.endpoint_url == "https://s3.example.com"
        assert cfg.access_key == "ak"
        assert cfg.secret_key == "sk"
        assert cfg.code_bucket == "my-code"
        assert cfg.models_bucket == "my-models"
        assert cfg.region == "eu-west-1"


# ======================================================================
# Feature sub-configs
# ======================================================================


class TestBudgetConfig:
    def test_defaults(self) -> None:
        cfg = BudgetConfig()
        assert cfg.enabled is False
        assert cfg.enforcement_mode == "block"
        assert cfg.alert_threshold_pct == 80.0

    def test_warn_mode(self) -> None:
        cfg = BudgetConfig(enforcement_mode="warn")
        assert cfg.enforcement_mode == "warn"

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValueError):
            BudgetConfig(enforcement_mode="invalid")  # type: ignore[arg-type]


class TestWebhookConfig:
    def test_defaults(self) -> None:
        cfg = WebhookConfig()
        assert cfg.enabled is False
        assert cfg.url == ""
        assert cfg.secret == ""
        assert cfg.timeout_seconds == 10.0
        assert cfg.retry_count == 2

    def test_custom_values(self) -> None:
        cfg = WebhookConfig(
            enabled=True,
            url="https://hooks.example.com",
            secret="s3cr3t",
            timeout_seconds=5.0,
            retry_count=3,
        )
        assert cfg.enabled is True
        assert cfg.url == "https://hooks.example.com"
        assert cfg.secret == "s3cr3t"
        assert cfg.timeout_seconds == 5.0
        assert cfg.retry_count == 3


class TestSpotConfig:
    def test_defaults(self) -> None:
        cfg = SpotConfig()
        assert cfg.enabled is False
        assert cfg.prefer_spot is False
        assert cfg.max_preemption_retries == 3
        assert cfg.failover_regions == []
        assert cfg.checkpoint_before_resume is True

    def test_custom_values(self) -> None:
        cfg = SpotConfig(
            enabled=True,
            prefer_spot=True,
            max_preemption_retries=5,
            failover_regions=["us-east-1", "eu-west-1"],
            checkpoint_before_resume=False,
        )
        assert cfg.enabled is True
        assert cfg.prefer_spot is True
        assert cfg.max_preemption_retries == 5
        assert cfg.failover_regions == ["us-east-1", "eu-west-1"]
        assert cfg.checkpoint_before_resume is False


class TestEnsembleConfig:
    def test_defaults(self) -> None:
        cfg = EnsembleConfig()
        assert cfg.enabled is False
        assert cfg.max_models_per_ensemble == 10
        assert cfg.auto_prune_enabled is False
        assert cfg.prune_weight_threshold == 0.05
        assert cfg.prune_min_models == 2


class TestABTestConfig:
    def test_defaults(self) -> None:
        cfg = ABTestConfig()
        assert cfg.enabled is False
        assert cfg.default_min_samples == 100
        assert cfg.max_concurrent_tests == 5


class TestHealthMonitoringConfig:
    def test_defaults(self) -> None:
        cfg = HealthMonitoringConfig()
        assert cfg.enabled is False
        assert cfg.check_interval_seconds == 300.0
        assert cfg.drift_threshold == 0.3
        assert cfg.degradation_pct == 10.0
        assert cfg.retention_days == 90


class TestCanaryConfig:
    def test_defaults(self) -> None:
        cfg = CanaryConfig()
        assert cfg.enabled is False
        assert cfg.initial_traffic_pct == 10.0
        assert cfg.evaluation_window_seconds == 3600.0
        assert cfg.auto_promote_threshold == 0.95
        assert cfg.auto_rollback_threshold == 0.80


# ======================================================================
# Cloud-provider configs
# ======================================================================


class TestGCPConfig:
    def test_defaults(self) -> None:
        cfg = GCPConfig()
        assert cfg.enabled is False
        assert cfg.project_id == ""
        assert cfg.credentials_path == ""
        assert cfg.region == "europe-west4"
        assert cfg.zone == "europe-west4-a"
        assert cfg.code_bucket == "artenic-code"
        assert cfg.data_bucket == "artenic-training-data"
        assert cfg.models_bucket == "artenic-models"
        assert cfg.default_machine_type == "e2-highmem-16"
        assert cfg.ssh_key_path == ""


class TestAWSConfig:
    def test_defaults(self) -> None:
        cfg = AWSConfig()
        assert cfg.enabled is False
        assert cfg.access_key_id == ""
        assert cfg.secret_access_key == ""
        assert cfg.session_token == ""
        assert cfg.region == "eu-west-1"
        assert cfg.code_bucket == "artenic-code"
        assert cfg.models_bucket == "artenic-models"
        assert cfg.default_instance_type == "m6i.xlarge"
        assert cfg.ami_id == ""
        assert cfg.subnet_id == ""
        assert cfg.security_group_id == ""
        assert cfg.key_name == ""
        assert cfg.iam_instance_profile == ""


class TestAzureConfig:
    def test_defaults(self) -> None:
        cfg = AzureConfig()
        assert cfg.enabled is False
        assert cfg.subscription_id == ""
        assert cfg.tenant_id == ""
        assert cfg.client_id == ""
        assert cfg.client_secret == ""
        assert cfg.region == "switzerlandnorth"
        assert cfg.resource_group == "artenic-ai"
        assert cfg.storage_account == "artenicai"
        assert cfg.code_container == "artenic-code"
        assert cfg.models_container == "artenic-models"
        assert cfg.storage_connection_string == ""
        assert cfg.default_vm_size == "Standard_D4s_v5"
        assert cfg.image_publisher == "Canonical"
        assert cfg.image_offer == "0001-com-ubuntu-server-jammy"
        assert cfg.image_sku == "22_04-lts-gen2"
        assert cfg.vnet_name == ""
        assert cfg.subnet_name == ""
        assert cfg.ssh_key_path == ""


class TestOCIConfig:
    def test_defaults(self) -> None:
        cfg = OCIConfig()
        assert cfg.enabled is False
        assert cfg.config_file == "~/.oci/config"
        assert cfg.config_profile == "DEFAULT"
        assert cfg.tenancy_ocid == ""
        assert cfg.user_ocid == ""
        assert cfg.fingerprint == ""
        assert cfg.key_file == ""
        assert cfg.region == "eu-frankfurt-1"
        assert cfg.compartment_id == ""
        assert cfg.availability_domain == ""
        assert cfg.default_shape == "VM.Standard.E4.Flex"
        assert cfg.subnet_id == ""
        assert cfg.ssh_public_key == ""
        assert cfg.image_id == ""
        assert cfg.namespace == ""
        assert cfg.code_bucket == "artenic-code"
        assert cfg.models_bucket == "artenic-models"


class TestOpenStackConfig:
    def test_defaults(self) -> None:
        cfg = OpenStackConfig()
        assert cfg.enabled is False
        assert cfg.provider_label == ""
        assert cfg.auth_url == ""
        assert cfg.username == ""
        assert cfg.password == ""
        assert cfg.application_credential_id == ""
        assert cfg.application_credential_secret == ""
        assert cfg.project_id == ""
        assert cfg.project_name == ""
        assert cfg.user_domain_name == "Default"
        assert cfg.project_domain_name == "Default"
        assert cfg.region == ""
        assert cfg.code_container == "artenic-code"
        assert cfg.models_container == "artenic-models"
        assert cfg.default_flavor == ""
        assert cfg.network_name == ""
        assert cfg.image_name == ""
        assert cfg.ssh_key_name == ""
        assert cfg.security_group == "default"


class TestHetznerConfig:
    def test_defaults(self) -> None:
        cfg = HetznerConfig()
        assert cfg.enabled is False
        assert cfg.api_key == ""
        assert cfg.api_url == "https://api.hetzner.cloud/v1"
        assert cfg.location == "fsn1"
        assert cfg.ssh_key_name == ""
        assert cfg.default_server_type == "ccx23"
        assert cfg.image == "ubuntu-22.04"
        assert isinstance(cfg.storage, S3StorageConfig)
        assert cfg.storage.code_bucket == "artenic-code"


class TestScalewayConfig:
    def test_defaults(self) -> None:
        cfg = ScalewayConfig()
        assert cfg.enabled is False
        assert cfg.secret_key == ""
        assert cfg.access_key == ""
        assert cfg.project_id == ""
        assert cfg.zone == "fr-par-2"
        assert cfg.default_instance_type == ""
        assert cfg.image_id == ""
        assert cfg.storage_endpoint == ""
        assert cfg.storage_region == ""
        assert cfg.code_bucket == "artenic-code"
        assert cfg.models_bucket == "artenic-models"


class TestLambdaLabsConfig:
    def test_defaults(self) -> None:
        cfg = LambdaLabsConfig()
        assert cfg.enabled is False
        assert cfg.api_key == ""
        assert cfg.api_url == "https://cloud.lambdalabs.com/api/v1"
        assert cfg.ssh_key_name == ""
        assert cfg.default_instance_type == ""
        assert cfg.region == "us-east-1"
        assert isinstance(cfg.storage, S3StorageConfig)


class TestRunPodConfig:
    def test_defaults(self) -> None:
        cfg = RunPodConfig()
        assert cfg.enabled is False
        assert cfg.api_key == ""
        assert cfg.api_url == "https://api.runpod.io/v2"
        assert cfg.default_gpu_type == ""
        assert cfg.cloud_type == "ALL"
        assert "pytorch" in cfg.docker_image
        assert isinstance(cfg.storage, S3StorageConfig)


class TestCoreWeaveConfig:
    def test_defaults(self) -> None:
        cfg = CoreWeaveConfig()
        assert cfg.enabled is False
        assert cfg.kubeconfig_path == ""
        assert cfg.namespace == "artenic-training"
        assert "pytorch" in cfg.training_image
        assert cfg.storage_class == "shared-hdd"
        assert cfg.node_selector == {}
        assert cfg.tolerations == []
        assert cfg.image_pull_secrets == []
        assert cfg.service_account == ""
        assert cfg.code_pvc_size == "10Gi"
        assert cfg.artifacts_pvc_size == "50Gi"


class TestVastAIConfig:
    def test_defaults(self) -> None:
        cfg = VastAIConfig()
        assert cfg.enabled is False
        assert cfg.api_key == ""
        assert cfg.api_url == "https://console.vast.ai/api/v0"
        assert "pytorch" in cfg.docker_image
        assert cfg.max_price_per_hour == 5.0
        assert cfg.min_reliability == 0.95
        assert isinstance(cfg.storage, S3StorageConfig)


class TestKubernetesConfig:
    def test_defaults(self) -> None:
        cfg = KubernetesConfig()
        assert cfg.enabled is False
        assert cfg.provider_label == ""
        assert cfg.kubeconfig_path == ""
        assert cfg.namespace == "artenic-training"
        assert cfg.training_image == "python:3.12-slim"
        assert cfg.storage_class == ""
        assert cfg.node_selector == {}
        assert cfg.tolerations == []
        assert cfg.image_pull_secrets == []
        assert cfg.service_account == ""
        assert cfg.code_pvc_size == "10Gi"
        assert cfg.artifacts_pvc_size == "50Gi"


# ======================================================================
# PlatformSettings — root
# ======================================================================


class TestPlatformSettings:
    def test_core_defaults(self) -> None:
        s = PlatformSettings()
        assert s.host == "0.0.0.0"
        assert s.port == 9000
        assert s.debug is False
        assert s.api_key == ""
        assert s.secret_key == ""
        assert s.database_url.startswith("postgresql+asyncpg://")

    def test_mlflow_defaults(self) -> None:
        s = PlatformSettings()
        assert s.mlflow_tracking_uri == "http://localhost:5000"
        assert s.mlflow_artifact_root == "./mlflow-artifacts"
        assert s.mlflow_experiment_prefix == "artenic"

    def test_otel_defaults(self) -> None:
        s = PlatformSettings()
        assert s.otel_enabled is True
        assert s.otel_service_name == "artenic-ai-platform"
        assert s.otel_exporter == "prometheus"

    def test_cors_defaults(self) -> None:
        s = PlatformSettings()
        assert s.cors_origins == ["http://localhost:3000"]

    def test_rate_limit_defaults(self) -> None:
        s = PlatformSettings()
        assert s.rate_limit_per_minute == 60
        assert s.rate_limit_burst == 10

    def test_feature_sub_configs_present(self) -> None:
        s = PlatformSettings()
        assert isinstance(s.budget, BudgetConfig)
        assert isinstance(s.webhook, WebhookConfig)
        assert isinstance(s.spot, SpotConfig)
        assert isinstance(s.ensemble, EnsembleConfig)
        assert isinstance(s.ab_test, ABTestConfig)
        assert isinstance(s.health, HealthMonitoringConfig)
        assert isinstance(s.canary, CanaryConfig)

    def test_all_features_disabled_by_default(self) -> None:
        s = PlatformSettings()
        assert s.budget.enabled is False
        assert s.webhook.enabled is False
        assert s.spot.enabled is False
        assert s.ensemble.enabled is False
        assert s.ab_test.enabled is False
        assert s.health.enabled is False
        assert s.canary.enabled is False

    def test_all_providers_disabled_by_default(self) -> None:
        s = PlatformSettings()
        for name in (
            "gcp",
            "aws",
            "azure",
            "oci",
            "openstack",
            "ovh",
            "infomaniak",
            "hetzner",
            "scaleway",
            "lambda_labs",
            "runpod",
            "coreweave",
            "vastai",
            "kubernetes",
            "local",
        ):
            provider = getattr(s, name)
            assert provider.enabled is False, f"{name} should be disabled"

    def test_provider_configs_correct_type(self) -> None:
        s = PlatformSettings()
        assert isinstance(s.gcp, GCPConfig)
        assert isinstance(s.aws, AWSConfig)
        assert isinstance(s.azure, AzureConfig)
        assert isinstance(s.oci, OCIConfig)
        assert isinstance(s.openstack, OpenStackConfig)
        assert isinstance(s.ovh, OpenStackConfig)
        assert isinstance(s.infomaniak, OpenStackConfig)
        assert isinstance(s.hetzner, HetznerConfig)
        assert isinstance(s.scaleway, ScalewayConfig)
        assert isinstance(s.lambda_labs, LambdaLabsConfig)
        assert isinstance(s.runpod, RunPodConfig)
        assert isinstance(s.coreweave, CoreWeaveConfig)
        assert isinstance(s.vastai, VastAIConfig)
        assert isinstance(s.kubernetes, KubernetesConfig)
        assert isinstance(s.local, LocalConfig)

    def test_ovh_preset_values(self) -> None:
        s = PlatformSettings()
        assert s.ovh.provider_label == "ovh"
        assert s.ovh.auth_url == "https://auth.cloud.ovh.net/v3"
        assert s.ovh.region == "GRA"
        assert s.ovh.default_flavor == "b2-30"
        assert s.ovh.network_name == "Ext-Net"
        assert s.ovh.image_name == "Ubuntu 22.04"

    def test_infomaniak_preset_values(self) -> None:
        s = PlatformSettings()
        assert s.infomaniak.provider_label == "infomaniak"
        assert "infomaniak" in s.infomaniak.auth_url
        assert s.infomaniak.region == "dc3-a"
        assert s.infomaniak.default_flavor == "a4-ram8-disk0"
        assert s.infomaniak.network_name == "ext-net1"
        assert "Ubuntu 22.04" in s.infomaniak.image_name

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARTENIC_HOST", "127.0.0.1")
        monkeypatch.setenv("ARTENIC_PORT", "8080")
        monkeypatch.setenv("ARTENIC_DEBUG", "true")
        s = PlatformSettings()
        assert s.host == "127.0.0.1"
        assert s.port == 8080
        assert s.debug is True

    def test_nested_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARTENIC_GCP__PROJECT_ID", "my-project")
        monkeypatch.setenv("ARTENIC_GCP__ENABLED", "true")
        s = PlatformSettings()
        assert s.gcp.project_id == "my-project"
        assert s.gcp.enabled is True

    def test_nested_feature_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARTENIC_BUDGET__ENABLED", "true")
        monkeypatch.setenv("ARTENIC_BUDGET__ENFORCEMENT_MODE", "warn")
        monkeypatch.setenv("ARTENIC_BUDGET__ALERT_THRESHOLD_PCT", "90.0")
        s = PlatformSettings()
        assert s.budget.enabled is True
        assert s.budget.enforcement_mode == "warn"
        assert s.budget.alert_threshold_pct == 90.0

    def test_otel_exporter_otlp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARTENIC_OTEL_EXPORTER", "otlp")
        s = PlatformSettings()
        assert s.otel_exporter == "otlp"

    def test_cors_origins_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARTENIC_CORS_ORIGINS", '["https://app.example.com"]')
        s = PlatformSettings()
        assert s.cors_origins == ["https://app.example.com"]

    def test_settings_config_prefix(self) -> None:
        assert PlatformSettings.model_config["env_prefix"] == "ARTENIC_"

    def test_settings_config_nested_delimiter(self) -> None:
        assert PlatformSettings.model_config["env_nested_delimiter"] == "__"

    def test_explicit_construction(self) -> None:
        s = PlatformSettings(
            host="0.0.0.0",
            port=3000,
            debug=True,
            api_key="my-key",
            secret_key="my-secret",
            database_url="sqlite+aiosqlite://",
        )
        assert s.port == 3000
        assert s.debug is True
        assert s.api_key == "my-key"
        assert s.secret_key == "my-secret"
        assert s.database_url == "sqlite+aiosqlite://"
