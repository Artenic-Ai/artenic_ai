"""Tests for artenic_ai_platform.providers.cloud_base — CloudProvider lifecycle."""

from __future__ import annotations

from artenic_ai_platform.providers.base import (
    CloudJobStatus,
    InstanceType,
    JobStatus,
    TrainingSpec,
)
from artenic_ai_platform.providers.cloud_base import CloudProvider


class _FakeCloud(CloudProvider):
    """Concrete in-memory CloudProvider for testing the abstract base."""

    def __init__(self) -> None:
        super().__init__()
        self.connect_count = 0
        self.disconnect_count = 0
        self.uploaded: list[TrainingSpec] = []
        self.provisioned: list[TrainingSpec] = []
        self.cancelled: list[str] = []
        self.cleaned: list[str] = []
        self.collected: list[str] = []

        # Configurable poll response
        self._poll_response: CloudJobStatus | None = None

    @property
    def provider_name(self) -> str:
        return "fake"

    async def _connect(self) -> None:
        self.connect_count += 1

    async def _disconnect(self) -> None:
        self.disconnect_count += 1

    async def _upload_code(self, spec: TrainingSpec) -> str:
        self.uploaded.append(spec)
        return "gs://bucket/code.tar.gz"

    async def _provision_and_start(self, spec: TrainingSpec) -> str:
        self.provisioned.append(spec)
        return "fake-job-001"

    async def _poll_provider(self, provider_job_id: str) -> CloudJobStatus:
        if self._poll_response is not None:
            return self._poll_response
        return CloudJobStatus(
            provider_job_id=provider_job_id,
            status=JobStatus.RUNNING,
        )

    async def _collect_artifacts(
        self,
        provider_job_id: str,
        status: CloudJobStatus,
    ) -> str | None:
        self.collected.append(provider_job_id)
        return "/tmp/artifacts"

    async def _cleanup_compute(self, provider_job_id: str) -> None:
        self.cleaned.append(provider_job_id)

    async def _cancel_provider_job(self, provider_job_id: str) -> None:
        self.cancelled.append(provider_job_id)

    async def _list_instances(
        self,
        region: str | None = None,
        gpu_only: bool = False,
    ) -> list[InstanceType]:
        return [
            InstanceType(name="fake-cpu", vcpus=4, memory_gb=16.0),
            InstanceType(
                name="fake-gpu",
                vcpus=8,
                memory_gb=32.0,
                gpu_type="T4",
                gpu_count=1,
            ),
        ]


class TestEnsureConnected:
    async def test_ensure_connected(self) -> None:
        cloud = _FakeCloud()
        assert cloud._connected is False

        await cloud.ensure_connected()
        assert cloud._connected is True
        assert cloud.connect_count == 1

        # Second call should not connect again
        await cloud.ensure_connected()
        assert cloud.connect_count == 1


class TestSubmitJobLifecycle:
    async def test_submit_job_lifecycle(self) -> None:
        cloud = _FakeCloud()
        spec = TrainingSpec(service="nlp", model="bert", provider="fake")

        job_id = await cloud.submit_job(spec)

        # Verify connection was established
        assert cloud._connected is True
        # Verify upload step
        assert len(cloud.uploaded) == 1
        assert cloud.uploaded[0] is spec
        # Verify provisioning step
        assert len(cloud.provisioned) == 1
        assert cloud.provisioned[0] is spec
        # Verify returned job ID
        assert job_id == "fake-job-001"


class TestPollStatus:
    async def test_poll_status_completed(self) -> None:
        cloud = _FakeCloud()
        cloud._poll_response = CloudJobStatus(
            provider_job_id="job-x",
            status=JobStatus.COMPLETED,
            metrics={"accuracy": 0.99},
        )

        status = await cloud.poll_status("job-x")

        assert status.status == JobStatus.COMPLETED
        # Artifacts should have been collected
        assert "job-x" in cloud.collected
        assert status.artifacts_uri == "/tmp/artifacts"
        # Compute should have been cleaned up
        assert "job-x" in cloud.cleaned

    async def test_poll_status_failed(self) -> None:
        cloud = _FakeCloud()
        cloud._poll_response = CloudJobStatus(
            provider_job_id="job-y",
            status=JobStatus.FAILED,
            error="OOM killed",
        )

        status = await cloud.poll_status("job-y")

        assert status.status == JobStatus.FAILED
        assert status.error == "OOM killed"
        # Even on failure, collect + cleanup should run
        assert "job-y" in cloud.collected
        assert "job-y" in cloud.cleaned


class TestCancelJob:
    async def test_cancel_job(self) -> None:
        cloud = _FakeCloud()

        await cloud.cancel_job("job-z")

        assert "job-z" in cloud.cancelled
        assert "job-z" in cloud.cleaned


class TestListInstanceTypes:
    async def test_list_instance_types(self) -> None:
        cloud = _FakeCloud()

        instances = await cloud.list_instance_types()

        assert len(instances) == 2
        names = [i.name for i in instances]
        assert "fake-cpu" in names
        assert "fake-gpu" in names


class TestDisconnect:
    async def test_disconnect(self) -> None:
        cloud = _FakeCloud()
        await cloud.ensure_connected()
        assert cloud._connected is True

        await cloud.disconnect()
        assert cloud._connected is False
        assert cloud.disconnect_count == 1

    async def test_disconnect_when_not_connected(self) -> None:
        cloud = _FakeCloud()
        # Should be a no-op
        await cloud.disconnect()
        assert cloud.disconnect_count == 0


class _FailingCollectCloud(_FakeCloud):
    """CloudProvider that fails on _collect_artifacts (lines 119-120)."""

    async def _collect_artifacts(
        self,
        provider_job_id: str,
        status: CloudJobStatus,
    ) -> str | None:
        raise RuntimeError("Failed to download artifacts")


class TestPollStatusCollectArtifactsFails:
    """When _collect_artifacts raises, poll_status still completes (lines 119-120)."""

    async def test_poll_status_collect_artifacts_fails(self) -> None:
        cloud = _FailingCollectCloud()
        cloud._poll_response = CloudJobStatus(
            provider_job_id="job-a",
            status=JobStatus.COMPLETED,
        )

        status = await cloud.poll_status("job-a")

        assert status.status == JobStatus.COMPLETED
        # artifacts_uri should remain None (collection failed)
        assert status.artifacts_uri is None
        # Cleanup should still run
        assert "job-a" in cloud.cleaned


class _FailingCleanupCloud(_FakeCloud):
    """CloudProvider that fails on _cleanup_compute (lines 128-129)."""

    async def _cleanup_compute(self, provider_job_id: str) -> None:
        raise RuntimeError("Cleanup failed")


class TestPollStatusCleanupFails:
    """When _cleanup_compute raises during poll, poll_status still returns (lines 128-129)."""

    async def test_poll_status_cleanup_fails(self) -> None:
        cloud = _FailingCleanupCloud()
        cloud._poll_response = CloudJobStatus(
            provider_job_id="job-b",
            status=JobStatus.COMPLETED,
        )

        status = await cloud.poll_status("job-b")

        assert status.status == JobStatus.COMPLETED
        # Artifacts should still have been collected
        assert "job-b" in cloud.collected


class _FailingCancelCloud(_FakeCloud):
    """CloudProvider that fails on _cancel_provider_job (lines 138-139)."""

    async def _cancel_provider_job(self, provider_job_id: str) -> None:
        raise RuntimeError("Cancel failed")


class TestCancelJobCancelFails:
    """When _cancel_provider_job raises, cancel_job still cleans up (lines 138-139)."""

    async def test_cancel_job_cancel_fails(self) -> None:
        cloud = _FailingCancelCloud()

        await cloud.cancel_job("job-c")

        # Cleanup should still run despite cancel failure
        assert "job-c" in cloud.cleaned


class _FailingBothCloud(_FakeCloud):
    """CloudProvider that fails on both cancel and cleanup (lines 142-143)."""

    async def _cancel_provider_job(self, provider_job_id: str) -> None:
        raise RuntimeError("Cancel failed")

    async def _cleanup_compute(self, provider_job_id: str) -> None:
        raise RuntimeError("Cleanup failed")


class TestCancelJobBothFail:
    """When both cancel and cleanup fail, cancel_job still completes (lines 138-143)."""

    async def test_cancel_job_both_fail(self) -> None:
        cloud = _FailingBothCloud()

        # Should not raise
        await cloud.cancel_job("job-d")
