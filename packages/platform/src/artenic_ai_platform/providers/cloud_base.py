"""Abstract base for cloud training providers.

Implements the full job lifecycle:
PACKAGE → UPLOAD → PROVISION → EXECUTE → MONITOR → COLLECT → CLEANUP

Concrete providers override the abstract hooks.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from artenic_ai_platform.providers.base import (
    CloudJobStatus,
    InstanceType,
    JobStatus,
    TrainingSpec,
)

logger = logging.getLogger(__name__)


class CloudProvider(ABC):
    """Base class for real cloud training providers.

    Subclasses implement the ``_*`` hooks for their specific cloud SDK.
    """

    def __init__(self) -> None:
        self._connected = False

    # ------------------------------------------------------------------
    # Abstract hooks — each provider implements these
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique identifier, e.g. 'gcp', 'aws', 'hetzner'."""

    @abstractmethod
    async def _connect(self) -> None:
        """Authenticate and initialise the provider SDK client."""

    @abstractmethod
    async def _disconnect(self) -> None:
        """Tear down SDK connections."""

    @abstractmethod
    async def _upload_code(self, spec: TrainingSpec) -> str:
        """Package and upload training code.  Returns remote URI."""

    @abstractmethod
    async def _provision_and_start(self, spec: TrainingSpec) -> str:
        """Provision compute and start training.  Returns provider job ID."""

    @abstractmethod
    async def _poll_provider(self, provider_job_id: str) -> CloudJobStatus:
        """Query provider for current job status."""

    @abstractmethod
    async def _collect_artifacts(
        self,
        provider_job_id: str,
        status: CloudJobStatus,
    ) -> str | None:
        """Download artifacts from cloud storage.  Returns local/remote URI."""

    @abstractmethod
    async def _cleanup_compute(self, provider_job_id: str) -> None:
        """Delete VMs / pods / instances after job completion."""

    @abstractmethod
    async def _cancel_provider_job(self, provider_job_id: str) -> None:
        """Provider-specific cancellation logic."""

    @abstractmethod
    async def _list_instances(
        self,
        region: str | None = None,
        gpu_only: bool = False,
    ) -> list[InstanceType]:
        """Query provider API for available instance types and pricing."""

    # ------------------------------------------------------------------
    # Concrete lifecycle — shared across all providers
    # ------------------------------------------------------------------

    async def ensure_connected(self) -> None:
        """Connect if not already connected."""
        if not self._connected:
            await self._connect()
            self._connected = True

    async def submit_job(self, spec: TrainingSpec) -> str:
        """Full lifecycle: connect → upload → provision → start."""
        await self.ensure_connected()

        logger.info("[%s] Uploading code for %s/%s", self.provider_name, spec.service, spec.model)
        await self._upload_code(spec)

        logger.info("[%s] Provisioning compute", self.provider_name)
        provider_job_id = await self._provision_and_start(spec)

        logger.info("[%s] Job started: %s", self.provider_name, provider_job_id)
        return provider_job_id

    async def poll_status(self, job_id: str) -> CloudJobStatus:
        """Poll provider and handle post-completion cleanup."""
        await self.ensure_connected()
        status = await self._poll_provider(job_id)

        if status.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            try:
                artifacts_uri = await self._collect_artifacts(job_id, status)
                if artifacts_uri:
                    status.artifacts_uri = artifacts_uri
            except Exception:
                logger.warning(
                    "[%s] Failed to collect artifacts for %s",
                    self.provider_name,
                    job_id,
                )

            try:
                await self._cleanup_compute(job_id)
            except Exception:
                logger.warning("[%s] Failed to cleanup compute for %s", self.provider_name, job_id)

        return status

    async def cancel_job(self, job_id: str) -> None:
        """Cancel a running job and clean up."""
        await self.ensure_connected()
        try:
            await self._cancel_provider_job(job_id)
        except Exception:
            logger.warning("[%s] Failed to cancel job %s", self.provider_name, job_id)
        try:
            await self._cleanup_compute(job_id)
        except Exception:
            logger.warning("[%s] Failed to cleanup after cancel %s", self.provider_name, job_id)

    async def list_instance_types(
        self,
        region: str | None = None,
        gpu_only: bool = False,
    ) -> list[InstanceType]:
        """Fetch live instance types and pricing from provider API."""
        await self.ensure_connected()
        return await self._list_instances(region=region, gpu_only=gpu_only)

    async def disconnect(self) -> None:
        """Gracefully disconnect from the provider."""
        if self._connected:
            await self._disconnect()
            self._connected = False
