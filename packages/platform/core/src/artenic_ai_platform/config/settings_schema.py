"""Declarative schema registry for platform configuration fields."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FieldMeta:
    """Metadata for a single configurable field."""

    key: str
    label: str
    type: str  # bool, int, float, str, list[str], json
    default: Any = None
    secret: bool = False
    restart_required: bool = False
    choices: list[str] = field(default_factory=list)
    description: str = ""


@dataclass(frozen=True)
class SectionMeta:
    """A group of related configuration fields."""

    name: str
    label: str
    fields: list[FieldMeta] = field(default_factory=list)
    description: str = ""


# ======================================================================
# Schema definitions — one SectionMeta per logical config group
# ======================================================================

CORE_SECTION = SectionMeta(
    name="core",
    label="Core",
    description="Core platform settings",
    fields=[
        FieldMeta(
            key="host",
            label="Host",
            type="str",
            default="0.0.0.0",
            description="Server bind address",
        ),
        FieldMeta(
            key="port",
            label="Port",
            type="int",
            default=8000,
            description="Server port",
        ),
        FieldMeta(
            key="debug",
            label="Debug",
            type="bool",
            default=False,
            description="Enable debug mode",
        ),
        FieldMeta(
            key="api_key",
            label="API Key",
            type="str",
            default="",
            secret=True,
            description="API authentication key",
        ),
        FieldMeta(
            key="secret_key",
            label="Secret Key",
            type="str",
            default="",
            secret=True,
            restart_required=True,
            description="Encryption passphrase",
        ),
        FieldMeta(
            key="database_url",
            label="Database URL",
            type="str",
            default="sqlite+aiosqlite:///artenic.db",
            secret=True,
            restart_required=True,
            description="Database connection URL",
        ),
    ],
)

MLFLOW_SECTION = SectionMeta(
    name="mlflow",
    label="MLflow",
    description="MLflow tracking integration",
    fields=[
        FieldMeta(
            key="mlflow_tracking_uri",
            label="Tracking URI",
            type="str",
            default="",
            description="MLflow tracking server URI",
        ),
        FieldMeta(
            key="mlflow_artifact_root",
            label="Artifact Root",
            type="str",
            default="",
            description="Default artifact storage root",
        ),
        FieldMeta(
            key="mlflow_experiment_prefix",
            label="Experiment Prefix",
            type="str",
            default="artenic",
            description="Prefix for experiment names",
        ),
    ],
)

OTEL_SECTION = SectionMeta(
    name="otel",
    label="OpenTelemetry",
    description="Observability and metrics",
    fields=[
        FieldMeta(
            key="otel_enabled",
            label="Enabled",
            type="bool",
            default=False,
            description="Enable OpenTelemetry",
        ),
        FieldMeta(
            key="otel_service_name",
            label="Service Name",
            type="str",
            default="artenic-ai-platform",
            description="OTel service name",
        ),
        FieldMeta(
            key="otel_exporter",
            label="Exporter",
            type="str",
            default="prometheus",
            choices=["prometheus", "otlp"],
            description="Metrics exporter type",
        ),
    ],
)

RATE_LIMIT_SECTION = SectionMeta(
    name="rate_limit",
    label="Rate Limiting",
    description="API rate limiting",
    fields=[
        FieldMeta(
            key="rate_limit_per_minute",
            label="Requests/min",
            type="int",
            default=60,
            description="Max requests per minute",
        ),
        FieldMeta(
            key="rate_limit_burst",
            label="Burst",
            type="int",
            default=10,
            description="Burst capacity",
        ),
    ],
)

CORS_SECTION = SectionMeta(
    name="cors",
    label="CORS",
    description="Cross-Origin Resource Sharing",
    fields=[
        FieldMeta(
            key="cors_origins",
            label="Origins",
            type="list[str]",
            default=["*"],
            description="Allowed origins",
        ),
    ],
)

BUDGET_SECTION = SectionMeta(
    name="budget",
    label="Budget",
    description="Budget governance",
    fields=[
        FieldMeta(
            key="enabled",
            label="Enabled",
            type="bool",
            default=False,
            description="Enable budget enforcement",
        ),
        FieldMeta(
            key="enforcement_mode",
            label="Enforcement Mode",
            type="str",
            default="warn",
            choices=["block", "warn"],
            description="Block or warn on budget exceeded",
        ),
        FieldMeta(
            key="alert_threshold_pct",
            label="Alert Threshold %",
            type="float",
            default=80.0,
            description="Alert when budget exceeds %",
        ),
    ],
)

WEBHOOK_SECTION = SectionMeta(
    name="webhook",
    label="Webhook",
    description="Webhook notifications",
    fields=[
        FieldMeta(
            key="enabled",
            label="Enabled",
            type="bool",
            default=False,
            description="Enable webhooks",
        ),
        FieldMeta(
            key="url",
            label="URL",
            type="str",
            default="",
            description="Webhook endpoint URL",
        ),
        FieldMeta(
            key="secret",
            label="Secret",
            type="str",
            default="",
            secret=True,
            description="HMAC signing secret",
        ),
        FieldMeta(
            key="timeout_seconds",
            label="Timeout (s)",
            type="int",
            default=30,
            description="Request timeout in seconds",
        ),
        FieldMeta(
            key="retry_count",
            label="Retry Count",
            type="int",
            default=3,
            description="Number of retry attempts",
        ),
    ],
)

SPOT_SECTION = SectionMeta(
    name="spot",
    label="Spot Instances",
    description="Spot/preemptible instance settings",
    fields=[
        FieldMeta(
            key="enabled",
            label="Enabled",
            type="bool",
            default=False,
            description="Enable spot instances",
        ),
        FieldMeta(
            key="prefer_spot",
            label="Prefer Spot",
            type="bool",
            default=True,
            description="Prefer spot over on-demand",
        ),
        FieldMeta(
            key="max_preemption_retries",
            label="Max Retries",
            type="int",
            default=3,
            description="Max preemption retry count",
        ),
        FieldMeta(
            key="failover_regions",
            label="Failover Regions",
            type="list[str]",
            default=[],
            description="Fallback regions on preemption",
        ),
    ],
)

ENSEMBLE_SECTION = SectionMeta(
    name="ensemble",
    label="Ensemble",
    description="Ensemble model management",
    fields=[
        FieldMeta(
            key="enabled",
            label="Enabled",
            type="bool",
            default=False,
            description="Enable ensembles",
        ),
        FieldMeta(
            key="max_models_per_ensemble",
            label="Max Models",
            type="int",
            default=10,
            description="Max models per ensemble",
        ),
        FieldMeta(
            key="auto_prune_enabled",
            label="Auto Prune",
            type="bool",
            default=False,
            description="Automatically prune underperforming models",
        ),
    ],
)

AB_TEST_SECTION = SectionMeta(
    name="ab_test",
    label="A/B Testing",
    description="A/B test configuration",
    fields=[
        FieldMeta(
            key="enabled",
            label="Enabled",
            type="bool",
            default=False,
            description="Enable A/B testing",
        ),
        FieldMeta(
            key="default_min_samples",
            label="Min Samples",
            type="int",
            default=100,
            description="Default minimum samples per variant",
        ),
        FieldMeta(
            key="max_concurrent_tests",
            label="Max Concurrent",
            type="int",
            default=5,
            description="Max concurrent A/B tests",
        ),
    ],
)

HEALTH_SECTION = SectionMeta(
    name="health",
    label="Health Monitoring",
    description="Model health and drift detection",
    fields=[
        FieldMeta(
            key="enabled",
            label="Enabled",
            type="bool",
            default=False,
            description="Enable health monitoring",
        ),
        FieldMeta(
            key="check_interval_seconds",
            label="Check Interval (s)",
            type="int",
            default=300,
            description="Health check interval in seconds",
        ),
        FieldMeta(
            key="drift_threshold",
            label="Drift Threshold",
            type="float",
            default=0.1,
            description="Drift detection threshold",
        ),
    ],
)

CANARY_SECTION = SectionMeta(
    name="canary",
    label="Canary Deployment",
    description="Canary deployment settings",
    fields=[
        FieldMeta(
            key="enabled",
            label="Enabled",
            type="bool",
            default=False,
            description="Enable canary deployments",
        ),
        FieldMeta(
            key="initial_traffic_pct",
            label="Initial Traffic %",
            type="float",
            default=10.0,
            description="Initial canary traffic percentage",
        ),
        FieldMeta(
            key="evaluation_window_seconds",
            label="Eval Window (s)",
            type="int",
            default=3600,
            description="Evaluation window in seconds",
        ),
    ],
)

# ======================================================================
# Schema registry — all sections grouped by scope
# ======================================================================

#: All sections available in the "global" scope.
GLOBAL_SECTIONS: list[SectionMeta] = [
    CORE_SECTION,
    MLFLOW_SECTION,
    OTEL_SECTION,
    RATE_LIMIT_SECTION,
    CORS_SECTION,
    BUDGET_SECTION,
    WEBHOOK_SECTION,
    SPOT_SECTION,
    ENSEMBLE_SECTION,
    AB_TEST_SECTION,
    HEALTH_SECTION,
    CANARY_SECTION,
]

#: Maps scope -> list of sections.
SCHEMA_REGISTRY: dict[str, list[SectionMeta]] = {
    "global": GLOBAL_SECTIONS,
}


def get_schema_all() -> dict[str, list[dict[str, Any]]]:
    """Return the full schema registry as a serialisable dict."""
    result: dict[str, list[dict[str, Any]]] = {}
    for scope, sections in SCHEMA_REGISTRY.items():
        result[scope] = [_section_to_dict(s) for s in sections]
    return result


def get_schema_for_scope(scope: str) -> list[dict[str, Any]]:
    """Return sections for a given scope."""
    sections = SCHEMA_REGISTRY.get(scope, [])
    return [_section_to_dict(s) for s in sections]


def get_section_schema(scope: str, section_name: str) -> dict[str, Any] | None:
    """Return a single section schema, or None."""
    for section in SCHEMA_REGISTRY.get(scope, []):
        if section.name == section_name:
            return _section_to_dict(section)
    return None


def _field_to_dict(f: FieldMeta) -> dict[str, Any]:
    """Serialise a FieldMeta."""
    result: dict[str, Any] = {
        "key": f.key,
        "label": f.label,
        "type": f.type,
        "default": f.default,
        "secret": f.secret,
        "restart_required": f.restart_required,
        "description": f.description,
    }
    if f.choices:
        result["choices"] = f.choices
    return result


def _section_to_dict(s: SectionMeta) -> dict[str, Any]:
    """Serialise a SectionMeta."""
    return {
        "name": s.name,
        "label": s.label,
        "description": s.description,
        "fields": [_field_to_dict(f) for f in s.fields],
    }
