"""Mock training provider for development and testing."""

from __future__ import annotations

import uuid

from artenic_ai_platform.providers.base import (
    CloudJobStatus,
    InstanceType,
    JobStatus,
    TrainingSpec,
)


class MockProvider:
    """Instant-completion provider for dev/test environments."""

    def __init__(self, latency_seconds: float = 0.0) -> None:
        self._latency = latency_seconds
        self._jobs: dict[str, dict] = {}  # type: ignore[type-arg]

    @property
    def provider_name(self) -> str:
        return "mock"

    async def submit_job(self, spec: TrainingSpec) -> str:
        job_id = f"mock-{uuid.uuid4().hex[:8]}"
        self._jobs[job_id] = {
            "spec": spec,
            "status": JobStatus.COMPLETED,
            "metrics": {"accuracy": 0.95, "loss": 0.05},
        }
        return job_id

    async def poll_status(self, job_id: str) -> CloudJobStatus:
        job = self._jobs.get(job_id)
        if job is None:
            return CloudJobStatus(
                provider_job_id=job_id,
                status=JobStatus.FAILED,
                error=f"Job {job_id} not found",
            )
        return CloudJobStatus(
            provider_job_id=job_id,
            status=job["status"],
            metrics=job.get("metrics"),
            cost_eur=0.0,
            duration_seconds=self._latency,
        )

    async def cancel_job(self, job_id: str) -> None:
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = JobStatus.CANCELLED

    async def list_instance_types(
        self,
        region: str | None = None,
        gpu_only: bool = False,
    ) -> list[InstanceType]:
        instances = [
            InstanceType(
                name="mock-cpu-small",
                vcpus=2,
                memory_gb=4.0,
                price_per_hour_eur=0.01,
            ),
            InstanceType(
                name="mock-cpu-large",
                vcpus=8,
                memory_gb=32.0,
                price_per_hour_eur=0.05,
            ),
            InstanceType(
                name="mock-gpu-a100",
                vcpus=12,
                memory_gb=80.0,
                gpu_type="A100",
                gpu_count=1,
                price_per_hour_eur=2.50,
                spot_price_per_hour_eur=0.75,
            ),
        ]
        if gpu_only:
            instances = [i for i in instances if i.gpu_count > 0]
        return instances
