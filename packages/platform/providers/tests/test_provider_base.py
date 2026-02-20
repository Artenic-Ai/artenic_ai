"""Tests for artenic_ai_platform_providers.base — data structures and protocol."""

from __future__ import annotations

from artenic_ai_platform_providers.base import (
    CloudJobStatus,
    InstanceType,
    JobStatus,
    TrainingProvider,
    TrainingSpec,
)
from artenic_ai_platform_providers.mock import MockProvider


class TestJobStatusEnum:
    def test_job_status_enum_values(self) -> None:
        assert JobStatus.PENDING == "pending"
        assert JobStatus.RUNNING == "running"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.FAILED == "failed"
        assert JobStatus.CANCELLED == "cancelled"
        assert JobStatus.PREEMPTED == "preempted"
        # Ensure exactly 6 members
        assert len(JobStatus) == 6


class TestTrainingSpec:
    def test_training_spec_defaults(self) -> None:
        spec = TrainingSpec(service="nlp", model="bert", provider="mock")
        assert spec.service == "nlp"
        assert spec.model == "bert"
        assert spec.provider == "mock"
        assert spec.config == {}
        assert spec.instance_type is None
        assert spec.region is None
        assert spec.is_spot is False
        assert spec.max_runtime_hours == 24.0
        assert spec.workload_spec is None


class TestCloudJobStatus:
    def test_cloud_job_status_defaults(self) -> None:
        status = CloudJobStatus(
            provider_job_id="job-123",
            status=JobStatus.RUNNING,
        )
        assert status.provider_job_id == "job-123"
        assert status.status == JobStatus.RUNNING
        assert status.metrics is None
        assert status.error is None
        assert status.artifacts_uri is None
        assert status.cost_eur is None
        assert status.duration_seconds is None


class TestInstanceType:
    def test_instance_type_defaults(self) -> None:
        inst = InstanceType(name="e2-standard-4", vcpus=4, memory_gb=16.0)
        assert inst.name == "e2-standard-4"
        assert inst.vcpus == 4
        assert inst.memory_gb == 16.0
        assert inst.gpu_type is None
        assert inst.gpu_count == 0
        assert inst.price_per_hour_eur == 0.0
        assert inst.spot_price_per_hour_eur is None
        assert inst.region is None
        assert inst.available is True


class TestTrainingProviderProtocol:
    def test_mock_provider_implements_protocol(self) -> None:
        provider = MockProvider()
        assert isinstance(provider, TrainingProvider)
