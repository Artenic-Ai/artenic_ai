"""Local subprocess training provider for development and on-premise use."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from artenic_ai_platform_providers.base import (
    CloudJobStatus,
    InstanceType,
    JobStatus,
    TrainingSpec,
)

logger = logging.getLogger(__name__)

_GRACE_PERIOD_SECONDS = 5.0


@dataclass
class _LocalJob:
    """Internal tracking state for a local subprocess job."""

    job_id: str
    process: asyncio.subprocess.Process | None
    work_dir: Path
    start_time: float
    timeout_seconds: float
    status: JobStatus = JobStatus.PENDING
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    _stdout_file: Any = None
    _stderr_file: Any = None
    _background_task: asyncio.Task[None] | None = None


def _get_system_memory_gb() -> float:
    """Best-effort system memory detection in GB (stdlib only)."""
    try:
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and page_size > 0:
                return float(round((pages * page_size) / (1024**3), 1))
    except (ValueError, OSError):
        pass
    return 0.0


class LocalProvider:
    """Runs training as local subprocesses.

    Implements the ``TrainingProvider`` protocol directly (no cloud
    upload/provisioning lifecycle).
    """

    def __init__(
        self,
        *,
        work_dir: str = "",
        max_concurrent_jobs: int = 4,
        default_timeout_hours: float = 24.0,
        gpu_enabled: bool = False,
        python_executable: str = "",
    ) -> None:
        self._work_dir_root = (
            Path(work_dir) if work_dir else Path(tempfile.gettempdir()) / "artenic-local"
        )
        self._default_timeout_hours = default_timeout_hours
        self._gpu_enabled = gpu_enabled
        self._python = python_executable or sys.executable
        self._semaphore = asyncio.Semaphore(max_concurrent_jobs)
        self._jobs: dict[str, _LocalJob] = {}

    @property
    def provider_name(self) -> str:
        return "local"

    # ------------------------------------------------------------------
    # TrainingProvider protocol
    # ------------------------------------------------------------------

    async def submit_job(self, spec: TrainingSpec) -> str:
        """Start a training subprocess and return the job ID."""
        job_id = f"local-{uuid.uuid4().hex[:8]}"

        job_dir = self._work_dir_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Resolve entry point
        entry_point = spec.config.get("entry_point") or spec.config.get("code_path", "train.py")

        # Build command with hyperparameters
        cmd: list[str] = [self._python, entry_point]
        for key, value in spec.config.get("hyperparameters", {}).items():
            cmd.extend([f"--{key}", str(value)])

        # Build environment
        env = {**os.environ, **spec.config.get("env", {})}
        env["ARTENIC_JOB_ID"] = job_id
        env["ARTENIC_JOB_DIR"] = str(job_dir)
        env["ARTENIC_SERVICE"] = spec.service
        env["ARTENIC_MODEL"] = spec.model
        if not self._gpu_enabled:
            env["CUDA_VISIBLE_DEVICES"] = ""

        timeout_hours = spec.max_runtime_hours or self._default_timeout_hours
        timeout_seconds = timeout_hours * 3600

        local_job = _LocalJob(
            job_id=job_id,
            process=None,
            work_dir=job_dir,
            start_time=time.monotonic(),
            timeout_seconds=timeout_seconds,
        )
        self._jobs[job_id] = local_job

        await self._semaphore.acquire()
        try:
            stdout_file = (job_dir / "stdout.log").open("w")
            stderr_file = (job_dir / "stderr.log").open("w")
            local_job._stdout_file = stdout_file
            local_job._stderr_file = stderr_file

            cwd = spec.config.get("cwd", str(job_dir))
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=stdout_file,
                stderr=stderr_file,
                cwd=cwd,
                env=env,
            )
            local_job.process = process
            local_job.status = JobStatus.RUNNING
        except Exception as exc:
            self._semaphore.release()
            local_job.status = JobStatus.FAILED
            local_job.error = str(exc)
            self._close_files(local_job)
            raise

        task = asyncio.create_task(self._wait_for_completion(job_id))
        local_job._background_task = task  # prevent GC
        return job_id

    async def poll_status(self, job_id: str) -> CloudJobStatus:
        """Query the current status of a local job."""
        job = self._jobs.get(job_id)
        if job is None:
            return CloudJobStatus(
                provider_job_id=job_id,
                status=JobStatus.FAILED,
                error=f"Job {job_id} not found",
            )

        elapsed = time.monotonic() - job.start_time

        # Try reading live metrics
        metrics = self._read_metrics(job.work_dir) or job.metrics

        return CloudJobStatus(
            provider_job_id=job_id,
            status=job.status,
            metrics=metrics if metrics else None,
            error=job.error,
            artifacts_uri=str(job.work_dir) if job.status == JobStatus.COMPLETED else None,
            cost_eur=0.0,
            duration_seconds=elapsed,
        )

    async def cancel_job(self, job_id: str) -> None:
        """Terminate a running local job."""
        job = self._jobs.get(job_id)
        if job is None:
            return

        if job.process is not None and job.process.returncode is None:
            await self._terminate_process(job.process)

        job.status = JobStatus.CANCELLED

    async def list_instance_types(
        self,
        region: str | None = None,
        gpu_only: bool = False,
    ) -> list[InstanceType]:
        """Return a single entry describing the local machine."""
        if gpu_only:
            return []

        return [
            InstanceType(
                name="local-cpu",
                vcpus=os.cpu_count() or 1,
                memory_gb=_get_system_memory_gb(),
                price_per_hour_eur=0.0,
                region="local",
                available=True,
            ),
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _wait_for_completion(self, job_id: str) -> None:
        """Background task: wait for process exit, handle timeout, release semaphore."""
        job = self._jobs.get(job_id)
        if job is None or job.process is None:  # pragma: no cover
            return

        try:
            await asyncio.wait_for(job.process.wait(), timeout=job.timeout_seconds)
        except TimeoutError:
            await self._terminate_process(job.process)
            job.status = JobStatus.FAILED
            job.error = f"Job timed out after {job.timeout_seconds:.0f}s"
        else:
            if job.status == JobStatus.RUNNING:
                returncode = job.process.returncode
                if returncode == 0:
                    job.status = JobStatus.COMPLETED
                    job.metrics = self._read_metrics(job.work_dir) or {}
                else:
                    job.status = JobStatus.FAILED
                    job.error = f"Process exited with code {returncode}"
        finally:
            self._close_files(job)
            self._semaphore.release()

    @staticmethod
    async def _terminate_process(process: asyncio.subprocess.Process) -> None:
        """SIGTERM then SIGKILL after grace period."""
        try:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=_GRACE_PERIOD_SECONDS)
            except TimeoutError:
                process.kill()
                await process.wait()
        except ProcessLookupError:
            pass

    @staticmethod
    def _read_metrics(work_dir: Path) -> dict[str, Any] | None:
        """Read ``metrics.json`` from a job's working directory."""
        metrics_path = work_dir / "metrics.json"
        if not metrics_path.exists():
            return None
        try:
            with metrics_path.open() as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
        return None

    @staticmethod
    def _close_files(job: _LocalJob) -> None:
        """Close stdout/stderr file handles if open."""
        for fh in (job._stdout_file, job._stderr_file):
            if fh is not None:
                with contextlib.suppress(OSError):
                    fh.close()
        job._stdout_file = None
        job._stderr_file = None
