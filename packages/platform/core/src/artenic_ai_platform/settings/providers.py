"""Cloud-provider configuration models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class S3StorageConfig(BaseModel):
    """Reusable S3-compatible storage configuration."""

    endpoint_url: str = ""
    access_key: str = ""
    secret_key: str = ""
    code_bucket: str = "artenic-code"
    models_bucket: str = "artenic-models"
    region: str = ""


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
