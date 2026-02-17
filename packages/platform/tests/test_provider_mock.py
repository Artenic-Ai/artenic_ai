"""Tests for artenic_ai_platform.providers.mock — MockProvider behaviour."""

from __future__ import annotations

from artenic_ai_platform.providers.base import JobStatus, TrainingSpec
from artenic_ai_platform.providers.mock import MockProvider


class TestMockProviderSubmit:
    async def test_submit_job_returns_id(self) -> None:
        provider = MockProvider()
        spec = TrainingSpec(service="nlp", model="bert", provider="mock")
        job_id = await provider.submit_job(spec)
        assert job_id.startswith("mock-")
        assert len(job_id) == 13  # "mock-" + 8 hex chars


class TestMockProviderPoll:
    async def test_poll_completed(self) -> None:
        provider = MockProvider()
        spec = TrainingSpec(service="nlp", model="bert", provider="mock")
        job_id = await provider.submit_job(spec)

        status = await provider.poll_status(job_id)
        assert status.status == JobStatus.COMPLETED
        assert status.provider_job_id == job_id
        assert status.metrics == {"accuracy": 0.95, "loss": 0.05}
        assert status.cost_eur == 0.0
        assert status.duration_seconds == 0.0
        assert status.error is None

    async def test_poll_unknown_job(self) -> None:
        provider = MockProvider()
        status = await provider.poll_status("nonexistent-id")
        assert status.status == JobStatus.FAILED
        assert status.error is not None
        assert "nonexistent-id" in status.error


class TestMockProviderCancel:
    async def test_cancel_job(self) -> None:
        provider = MockProvider()
        spec = TrainingSpec(service="nlp", model="bert", provider="mock")
        job_id = await provider.submit_job(spec)

        await provider.cancel_job(job_id)

        status = await provider.poll_status(job_id)
        assert status.status == JobStatus.CANCELLED


class TestMockProviderName:
    def test_provider_name(self) -> None:
        provider = MockProvider()
        assert provider.provider_name == "mock"


class TestMockProviderInstances:
    async def test_list_instance_types(self) -> None:
        provider = MockProvider()
        instances = await provider.list_instance_types()
        assert len(instances) == 3
        names = [i.name for i in instances]
        assert "mock-cpu-small" in names
        assert "mock-cpu-large" in names
        assert "mock-gpu-a100" in names

    async def test_list_instance_types_gpu_only(self) -> None:
        provider = MockProvider()
        instances = await provider.list_instance_types(gpu_only=True)
        assert len(instances) == 1
        assert instances[0].name == "mock-gpu-a100"
        assert instances[0].gpu_type == "A100"
        assert instances[0].gpu_count == 1
