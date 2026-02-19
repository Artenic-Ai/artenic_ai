"""Static catalog of supported cloud providers.

This module contains only **public metadata** — provider names,
capability types, and credential field descriptions.  No secrets,
no proprietary data.  All live data (instances, storage containers,
pricing) is fetched at runtime from the provider's API using the
user's own credentials.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Catalog data-classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderCapability:
    """A capability offered by a provider (storage, compute, …)."""

    type: str  # "storage" | "compute"
    name: str  # human-readable, e.g. "Object Storage"
    description: str


@dataclass(frozen=True)
class CredentialField:
    """Describes one credential field the user must provide."""

    key: str  # machine key, e.g. "username"
    label: str  # human label, e.g. "OpenStack Username"
    required: bool = True
    secret: bool = False  # masked in UI (passwords, API keys)
    placeholder: str = ""


@dataclass(frozen=True)
class ConfigField:
    """Describes one non-sensitive configuration field."""

    key: str
    label: str
    default: str = ""
    description: str = ""


@dataclass(frozen=True)
class ProviderDefinition:
    """Complete definition of a supported provider.

    All information here is publicly available and does not contain
    any proprietary or user-specific data.
    """

    id: str  # slug, e.g. "ovh"
    display_name: str  # "OVH Public Cloud"
    description: str
    website: str  # link to public documentation
    connector_type: str  # "openstack" | "aws" | "gcp" | …
    capabilities: tuple[ProviderCapability, ...]
    credential_fields: tuple[CredentialField, ...]
    config_fields: tuple[ConfigField, ...]


# ---------------------------------------------------------------------------
# Provider catalog — public metadata only
# ---------------------------------------------------------------------------

PROVIDER_CATALOG: dict[str, ProviderDefinition] = {
    # ------------------------------------------------------------------
    # OVH Public Cloud (OpenStack)
    # ------------------------------------------------------------------
    "ovh": ProviderDefinition(
        id="ovh",
        display_name="OVH Public Cloud",
        description="European cloud provider based on OpenStack",
        website="https://www.ovhcloud.com/en/public-cloud/",
        connector_type="openstack",
        capabilities=(
            ProviderCapability(
                "storage",
                "Object Storage",
                "S3-compatible object storage via OpenStack Swift",
            ),
            ProviderCapability(
                "compute",
                "Public Cloud Instances",
                "CPU and GPU compute instances via OpenStack Nova",
            ),
        ),
        credential_fields=(
            CredentialField(
                "auth_url",
                "Keystone Auth URL",
                placeholder="https://auth.cloud.ovh.net/v3",
            ),
            CredentialField("username", "OpenStack Username"),
            CredentialField("password", "OpenStack Password", secret=True),
            CredentialField("project_id", "Project ID"),
        ),
        config_fields=(
            ConfigField("region", "Default Region", default="GRA11"),
            ConfigField(
                "user_domain_name",
                "User Domain Name",
                default="Default",
                description="OpenStack user domain (usually 'Default')",
            ),
            ConfigField(
                "project_domain_name",
                "Project Domain Name",
                default="Default",
                description="OpenStack project domain (usually 'Default')",
            ),
        ),
    ),
    # ------------------------------------------------------------------
    # Infomaniak Public Cloud (OpenStack — reuses openstack connector)
    # ------------------------------------------------------------------
    "infomaniak": ProviderDefinition(
        id="infomaniak",
        display_name="Infomaniak Public Cloud",
        description="Swiss cloud provider based on OpenStack",
        website="https://www.infomaniak.com/en/hosting/public-cloud",
        connector_type="openstack",
        capabilities=(
            ProviderCapability(
                "storage",
                "Object Storage",
                "S3-compatible object storage via OpenStack Swift",
            ),
            ProviderCapability(
                "compute",
                "Public Cloud Instances",
                "CPU and GPU compute instances via OpenStack Nova",
            ),
        ),
        credential_fields=(
            CredentialField(
                "auth_url",
                "Keystone Auth URL",
                placeholder="https://api.pub1.infomaniak.cloud/identity/v3",
            ),
            CredentialField("username", "OpenStack Username"),
            CredentialField("password", "OpenStack Password", secret=True),
            CredentialField("project_id", "Project ID"),
        ),
        config_fields=(
            ConfigField("region", "Default Region", default="dc3-a"),
            ConfigField(
                "user_domain_name",
                "User Domain Name",
                default="Default",
                description="OpenStack user domain (usually 'Default')",
            ),
            ConfigField(
                "project_domain_name",
                "Project Domain Name",
                default="Default",
                description="OpenStack project domain (usually 'Default')",
            ),
        ),
    ),
    # ------------------------------------------------------------------
    # Scaleway
    # ------------------------------------------------------------------
    "scaleway": ProviderDefinition(
        id="scaleway",
        display_name="Scaleway",
        description="European cloud provider with GPU instances and S3 storage",
        website="https://www.scaleway.com/en/",
        connector_type="scaleway",
        capabilities=(
            ProviderCapability(
                "storage",
                "Object Storage",
                "S3-compatible object storage (Scaleway Object Storage)",
            ),
            ProviderCapability(
                "compute",
                "Instances",
                "CPU and GPU compute instances (Scaleway Instances)",
            ),
        ),
        credential_fields=(
            CredentialField("access_key", "Access Key", placeholder="SCW..."),
            CredentialField("secret_key", "Secret Key", secret=True),
            CredentialField("project_id", "Project ID"),
        ),
        config_fields=(
            ConfigField(
                "zone",
                "Default Zone",
                default="fr-par-1",
                description="Availability zone (e.g. fr-par-1, nl-ams-1)",
            ),
        ),
    ),
    # ------------------------------------------------------------------
    # Vast.ai (compute-only, GPU marketplace)
    # ------------------------------------------------------------------
    "vastai": ProviderDefinition(
        id="vastai",
        display_name="Vast.ai",
        description="GPU compute marketplace for deep learning",
        website="https://vast.ai/",
        connector_type="vastai",
        capabilities=(
            ProviderCapability(
                "compute",
                "GPU Instances",
                "On-demand and interruptible GPU instances",
            ),
        ),
        credential_fields=(CredentialField("api_key", "API Key", secret=True),),
        config_fields=(),
    ),
    # ------------------------------------------------------------------
    # Amazon Web Services
    # ------------------------------------------------------------------
    "aws": ProviderDefinition(
        id="aws",
        display_name="Amazon Web Services",
        description="Global cloud platform — S3, EC2, and more",
        website="https://aws.amazon.com/",
        connector_type="aws",
        capabilities=(
            ProviderCapability(
                "storage",
                "Amazon S3",
                "Scalable object storage (Simple Storage Service)",
            ),
            ProviderCapability(
                "compute",
                "Amazon EC2",
                "Elastic Compute Cloud — CPU and GPU instances",
            ),
        ),
        credential_fields=(
            CredentialField("access_key_id", "Access Key ID", placeholder="AKIA..."),
            CredentialField("secret_access_key", "Secret Access Key", secret=True),
        ),
        config_fields=(
            ConfigField(
                "region",
                "Default Region",
                default="eu-west-1",
                description="AWS region (e.g. us-east-1, eu-west-1)",
            ),
        ),
    ),
    # ------------------------------------------------------------------
    # Google Cloud Platform
    # ------------------------------------------------------------------
    "gcp": ProviderDefinition(
        id="gcp",
        display_name="Google Cloud Platform",
        description="Global cloud platform — GCS, Compute Engine, and more",
        website="https://cloud.google.com/",
        connector_type="gcp",
        capabilities=(
            ProviderCapability(
                "storage",
                "Cloud Storage",
                "Google Cloud Storage buckets",
            ),
            ProviderCapability(
                "compute",
                "Compute Engine",
                "CPU and GPU virtual machines",
            ),
        ),
        credential_fields=(
            CredentialField("project_id", "Project ID", placeholder="my-project-123"),
            CredentialField(
                "credentials_json",
                "Service Account JSON",
                secret=True,
                placeholder="Paste the full JSON key file content",
            ),
        ),
        config_fields=(
            ConfigField(
                "zone",
                "Default Zone",
                default="europe-west1-b",
                description="Compute Engine zone (e.g. us-central1-a, europe-west1-b)",
            ),
        ),
    ),
    # ------------------------------------------------------------------
    # Microsoft Azure
    # ------------------------------------------------------------------
    "azure": ProviderDefinition(
        id="azure",
        display_name="Microsoft Azure",
        description="Global cloud platform — Blob Storage, Virtual Machines, and more",
        website="https://azure.microsoft.com/",
        connector_type="azure",
        capabilities=(
            ProviderCapability(
                "storage",
                "Blob Storage",
                "Azure Blob Storage containers",
            ),
            ProviderCapability(
                "compute",
                "Virtual Machines",
                "CPU and GPU virtual machines",
            ),
        ),
        credential_fields=(
            CredentialField("subscription_id", "Subscription ID"),
            CredentialField("tenant_id", "Tenant ID"),
            CredentialField("client_id", "Client ID (App ID)"),
            CredentialField("client_secret", "Client Secret", secret=True),
        ),
        config_fields=(
            ConfigField(
                "region",
                "Default Region",
                default="westeurope",
                description="Azure region (e.g. westeurope, eastus, westus2)",
            ),
            ConfigField(
                "resource_group",
                "Resource Group",
                default="",
                description="Azure resource group name (optional, for scoped queries)",
            ),
        ),
    ),
}


def get_provider_definition(provider_id: str) -> ProviderDefinition | None:
    """Look up a provider definition by ID."""
    return PROVIDER_CATALOG.get(provider_id)


def list_provider_definitions() -> list[ProviderDefinition]:
    """Return all known provider definitions."""
    return list(PROVIDER_CATALOG.values())
