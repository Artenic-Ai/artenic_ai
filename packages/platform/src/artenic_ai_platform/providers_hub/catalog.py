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
}


def get_provider_definition(provider_id: str) -> ProviderDefinition | None:
    """Look up a provider definition by ID."""
    return PROVIDER_CATALOG.get(provider_id)


def list_provider_definitions() -> list[ProviderDefinition]:
    """Return all known provider definitions."""
    return list(PROVIDER_CATALOG.values())
