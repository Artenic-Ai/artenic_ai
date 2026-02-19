"""Pydantic request / response models for the Provider Hub API."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 — Pydantic needs this at runtime

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------


class CapabilityOut(BaseModel):
    """One capability offered by a provider."""

    type: str
    name: str
    description: str


class CredentialFieldOut(BaseModel):
    """Describes a credential field (for the configuration form)."""

    key: str
    label: str
    required: bool
    secret: bool
    placeholder: str = ""


class ConfigFieldOut(BaseModel):
    """Describes a non-sensitive configuration field."""

    key: str
    label: str
    default: str = ""
    description: str = ""


# ---------------------------------------------------------------------------
# Provider list / detail
# ---------------------------------------------------------------------------


class ProviderSummary(BaseModel):
    """Lightweight provider info for list views."""

    id: str
    display_name: str
    description: str
    website: str
    connector_type: str
    capabilities: list[CapabilityOut]
    enabled: bool = False
    status: str = "unconfigured"
    status_message: str = ""


class ProviderDetail(ProviderSummary):
    """Full provider detail including form fields and timestamps."""

    credential_fields: list[CredentialFieldOut]
    config_fields: list[ConfigFieldOut]
    config: dict[str, str] = Field(default_factory=dict)
    has_credentials: bool = False
    last_checked_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class ConfigureProviderRequest(BaseModel):
    """Configure credentials and non-sensitive settings for a provider."""

    credentials: dict[str, str] = Field(
        ...,
        description="Credential key-value pairs (will be encrypted at rest).",
    )
    config: dict[str, str] = Field(
        default_factory=dict,
        description="Non-sensitive config (region, domain, etc.).",
    )


# ---------------------------------------------------------------------------
# Connection test
# ---------------------------------------------------------------------------


class ConnectionTestResult(BaseModel):
    """Result of a provider connection test."""

    success: bool
    message: str
    latency_ms: float | None = None


# ---------------------------------------------------------------------------
# Capability responses (live data from provider API)
# ---------------------------------------------------------------------------


class StorageOption(BaseModel):
    """A storage container / bucket discovered from the provider."""

    provider_id: str
    name: str
    type: str = "object_storage"
    region: str = ""
    bytes_used: int | None = None
    object_count: int | None = None


class ComputeInstance(BaseModel):
    """A compute flavor / instance type from the provider."""

    provider_id: str
    name: str
    vcpus: int = 0
    memory_gb: float = 0.0
    disk_gb: float = 0.0
    gpu_type: str | None = None
    gpu_count: int = 0
    price_per_hour_eur: float | None = None
    region: str = ""
    available: bool = True


class ProviderRegion(BaseModel):
    """A region available on the provider."""

    provider_id: str
    id: str
    name: str = ""
    description: str = ""


# ---------------------------------------------------------------------------
# Public catalog responses (no authentication required)
# ---------------------------------------------------------------------------


class CatalogComputeFlavor(BaseModel):
    """A compute flavor from a provider's public pricing catalog."""

    provider_id: str
    name: str
    vcpus: int = 0
    memory_gb: float = 0.0
    disk_gb: float = 0.0
    gpu_type: str | None = None
    gpu_count: int = 0
    price_per_hour: float | None = None
    currency: str = "EUR"
    region: str = ""
    category: str = ""


class CatalogStorageTier(BaseModel):
    """A storage tier from a provider's public pricing catalog."""

    provider_id: str
    name: str
    type: str = "object_storage"
    price_per_gb_month: float | None = None
    currency: str = "EUR"
    region: str = ""
    description: str = ""


class CatalogResponse(BaseModel):
    """Wrapper for public catalog data with metadata."""

    provider_id: str
    provider_name: str
    compute: list[CatalogComputeFlavor] = Field(default_factory=list)
    storage: list[CatalogStorageTier] = Field(default_factory=list)
    is_live: bool = True
    cached: bool = False
    last_fetched_at: datetime | None = None
