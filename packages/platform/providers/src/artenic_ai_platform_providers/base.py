"""Training provider protocol and core data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class JobStatus(StrEnum):
    """Status of a training job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PREEMPTED = "preempted"


@dataclass
class InstanceType:
    """Description of a compute instance available from a provider."""

    name: str
    vcpus: int
    memory_gb: float
    gpu_type: str | None = None
    gpu_count: int = 0
    price_per_hour_eur: float = 0.0
    spot_price_per_hour_eur: float | None = None
    region: str | None = None
    available: bool = True


@dataclass
class TrainingSpec:
    """Specification for a training job."""

    service: str
    model: str
    provider: str
    config: dict[str, Any] = field(default_factory=dict)
    instance_type: str | None = None
    region: str | None = None
    is_spot: bool = False
    max_runtime_hours: float = 24.0
    workload_spec: dict[str, Any] | None = None


@dataclass
class CloudJobStatus:
    """Status of a job reported by a cloud provider."""

    provider_job_id: str
    status: JobStatus
    metrics: dict[str, Any] | None = None
    error: str | None = None
    artifacts_uri: str | None = None
    cost_eur: float | None = None
    duration_seconds: float | None = None


@runtime_checkable
class TrainingProvider(Protocol):
    """Protocol that all training providers must implement."""

    @property
    def provider_name(self) -> str:
        """Unique provider identifier."""
        ...  # pragma: no cover

    async def submit_job(self, spec: TrainingSpec) -> str:
        """Submit a training job.  Returns provider-specific job ID."""
        ...  # pragma: no cover

    async def poll_status(self, job_id: str) -> CloudJobStatus:
        """Poll the current status of a job."""
        ...  # pragma: no cover

    async def cancel_job(self, job_id: str) -> None:
        """Cancel a running job."""
        ...  # pragma: no cover

    async def list_instance_types(
        self,
        region: str | None = None,
        gpu_only: bool = False,
    ) -> list[InstanceType]:
        """Query available instance types and live pricing from the provider API."""
        ...  # pragma: no cover
