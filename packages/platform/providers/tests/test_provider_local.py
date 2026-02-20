"""Tests for artenic_ai_platform_providers.local — LocalProvider behaviour."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import artenic_ai_platform_providers.local as local_module
from artenic_ai_platform_providers.base import (
    InstanceType,
    JobStatus,
    TrainingSpec,
)
from artenic_ai_platform_providers.local import (
    LocalProvider,
    _get_system_memory_gb,
    _LocalJob,
)

if TYPE_CHECKING:
    from pathlib import Path


# ===================================================================
# Helpers
# ===================================================================


def _make_spec(**overrides: Any) -> TrainingSpec:
    defaults: dict[str, Any] = {
        "service": "nlp",
        "model": "bert",
        "provider": "local",
        "config": {},
    }
    defaults.update(overrides)
    return TrainingSpec(**defaults)


class _FakeProcess:
    """Fake asyncio.subprocess.Process for testing."""

    def __init__(
        self,
        returncode: int = 0,
        *,
        hang: bool = False,
    ) -> None:
        self._final_code = returncode
        self.returncode: int | None = None if hang else returncode
        self.terminated = False
        self.killed = False
        self.pid = 12345
        self._wait_event = asyncio.Event()
        if not hang:
            self._wait_event.set()

    async def wait(self) -> int:
        await self._wait_event.wait()
        self.returncode = self._final_code
        return self._final_code

    def terminate(self) -> None:
        self.terminated = True
        self._final_code = -15
        self._wait_event.set()

    def kill(self) -> None:
        self.killed = True
        self._final_code = -9
        self._wait_event.set()


def _build_provider(tmp_path: Path, **kwargs: Any) -> LocalProvider:
    defaults: dict[str, Any] = {
        "work_dir": str(tmp_path),
        "max_concurrent_jobs": 4,
        "default_timeout_hours": 24.0,
        "gpu_enabled": False,
        "python_executable": "/usr/bin/python3",
    }
    defaults.update(kwargs)
    return LocalProvider(**defaults)


# ===================================================================
# TestLocalProviderName
# ===================================================================


class TestLocalProviderName:
    def test_provider_name(self, tmp_path: Path) -> None:
        provider = _build_provider(tmp_path)
        assert provider.provider_name == "local"


# ===================================================================
# TestLocalProviderProtocol
# ===================================================================


class TestLocalProviderProtocol:
    def test_implements_training_provider(self, tmp_path: Path) -> None:
        from artenic_ai_platform_providers.base import TrainingProvider

        provider = _build_provider(tmp_path)
        assert isinstance(provider, TrainingProvider)


# ===================================================================
# TestLocalProviderSubmit
# ===================================================================


class TestLocalProviderSubmit:
    async def test_submit_returns_local_id(self, tmp_path: Path, monkeypatch: Any) -> None:
        provider = _build_provider(tmp_path)
        fake = _FakeProcess(returncode=0)
        monkeypatch.setattr(
            asyncio,
            "create_subprocess_exec",
            AsyncMock(return_value=fake),
        )
        spec = _make_spec(config={"entry_point": "train.py"})
        job_id = await provider.submit_job(spec)
        assert job_id.startswith("local-")
        assert len(job_id) == 14

    async def test_submit_creates_work_dir(self, tmp_path: Path, monkeypatch: Any) -> None:
        provider = _build_provider(tmp_path)
        fake = _FakeProcess(returncode=0)
        monkeypatch.setattr(
            asyncio,
            "create_subprocess_exec",
            AsyncMock(return_value=fake),
        )
        spec = _make_spec()
        job_id = await provider.submit_job(spec)
        assert (tmp_path / job_id).is_dir()

    async def test_submit_passes_hyperparameters(self, tmp_path: Path, monkeypatch: Any) -> None:
        provider = _build_provider(tmp_path)
        fake = _FakeProcess(returncode=0)
        mock_exec = AsyncMock(return_value=fake)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_exec)
        spec = _make_spec(
            config={"entry_point": "train.py", "hyperparameters": {"lr": "0.01", "epochs": "10"}}
        )
        await provider.submit_job(spec)
        call_args = mock_exec.call_args
        cmd = call_args.args
        assert "--lr" in cmd
        assert "0.01" in cmd
        assert "--epochs" in cmd
        assert "10" in cmd

    async def test_submit_uses_code_path_fallback(self, tmp_path: Path, monkeypatch: Any) -> None:
        provider = _build_provider(tmp_path)
        fake = _FakeProcess(returncode=0)
        mock_exec = AsyncMock(return_value=fake)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_exec)
        spec = _make_spec(config={"code_path": "my_train.py"})
        await provider.submit_job(spec)
        cmd = mock_exec.call_args.args
        assert "my_train.py" in cmd

    async def test_submit_failure_sets_failed(self, tmp_path: Path, monkeypatch: Any) -> None:
        provider = _build_provider(tmp_path)
        monkeypatch.setattr(
            asyncio,
            "create_subprocess_exec",
            AsyncMock(side_effect=OSError("cannot start")),
        )
        spec = _make_spec(config={"entry_point": "train.py"})
        with contextlib.suppress(OSError):
            await provider.submit_job(spec)
        # The job should be tracked and marked FAILED
        job = next(iter(provider._jobs.values()))
        assert job.status == JobStatus.FAILED
        assert "cannot start" in (job.error or "")


# ===================================================================
# TestLocalProviderPoll
# ===================================================================


class TestLocalProviderPoll:
    async def test_poll_unknown_job(self, tmp_path: Path) -> None:
        provider = _build_provider(tmp_path)
        result = await provider.poll_status("local-nonexist")
        assert result.status == JobStatus.FAILED
        assert "not found" in (result.error or "")

    async def test_poll_running_job(self, tmp_path: Path, monkeypatch: Any) -> None:
        provider = _build_provider(tmp_path)
        fake = _FakeProcess(returncode=0, hang=True)
        monkeypatch.setattr(
            asyncio,
            "create_subprocess_exec",
            AsyncMock(return_value=fake),
        )
        spec = _make_spec(config={"entry_point": "train.py"})
        job_id = await provider.submit_job(spec)
        result = await provider.poll_status(job_id)
        assert result.status == JobStatus.RUNNING
        assert result.cost_eur == 0.0
        assert result.duration_seconds >= 0
        # Clean up
        fake.terminate()

    async def test_poll_completed_job(self, tmp_path: Path, monkeypatch: Any) -> None:
        provider = _build_provider(tmp_path)
        fake = _FakeProcess(returncode=0)
        monkeypatch.setattr(
            asyncio,
            "create_subprocess_exec",
            AsyncMock(return_value=fake),
        )
        spec = _make_spec(config={"entry_point": "train.py"})
        job_id = await provider.submit_job(spec)
        # Wait for background task to complete
        await asyncio.sleep(0.05)
        result = await provider.poll_status(job_id)
        assert result.status == JobStatus.COMPLETED
        assert result.artifacts_uri is not None

    async def test_poll_completed_with_metrics(self, tmp_path: Path, monkeypatch: Any) -> None:
        provider = _build_provider(tmp_path)
        fake = _FakeProcess(returncode=0)
        monkeypatch.setattr(
            asyncio,
            "create_subprocess_exec",
            AsyncMock(return_value=fake),
        )
        spec = _make_spec(config={"entry_point": "train.py"})
        job_id = await provider.submit_job(spec)
        # Write metrics file
        job_dir = tmp_path / job_id
        (job_dir / "metrics.json").write_text(
            json.dumps({"accuracy": 0.95, "loss": 0.05}), encoding="utf-8"
        )
        await asyncio.sleep(0.05)
        result = await provider.poll_status(job_id)
        assert result.status == JobStatus.COMPLETED
        assert result.metrics is not None
        assert result.metrics["accuracy"] == 0.95

    async def test_poll_failed_job(self, tmp_path: Path, monkeypatch: Any) -> None:
        provider = _build_provider(tmp_path)
        fake = _FakeProcess(returncode=1)
        monkeypatch.setattr(
            asyncio,
            "create_subprocess_exec",
            AsyncMock(return_value=fake),
        )
        spec = _make_spec(config={"entry_point": "train.py"})
        job_id = await provider.submit_job(spec)
        await asyncio.sleep(0.05)
        result = await provider.poll_status(job_id)
        assert result.status == JobStatus.FAILED
        assert "exit" in (result.error or "").lower()


# ===================================================================
# TestLocalProviderCancel
# ===================================================================


class TestLocalProviderCancel:
    async def test_cancel_running_job(self, tmp_path: Path, monkeypatch: Any) -> None:
        provider = _build_provider(tmp_path)
        fake = _FakeProcess(returncode=0, hang=True)
        monkeypatch.setattr(
            asyncio,
            "create_subprocess_exec",
            AsyncMock(return_value=fake),
        )
        spec = _make_spec(config={"entry_point": "train.py"})
        job_id = await provider.submit_job(spec)
        await provider.cancel_job(job_id)
        assert fake.terminated
        result = await provider.poll_status(job_id)
        assert result.status == JobStatus.CANCELLED

    async def test_cancel_unknown_job(self, tmp_path: Path) -> None:
        provider = _build_provider(tmp_path)
        # Should not raise
        await provider.cancel_job("local-nonexist")

    async def test_cancel_already_finished(self, tmp_path: Path, monkeypatch: Any) -> None:
        provider = _build_provider(tmp_path)
        fake = _FakeProcess(returncode=0)
        monkeypatch.setattr(
            asyncio,
            "create_subprocess_exec",
            AsyncMock(return_value=fake),
        )
        spec = _make_spec(config={"entry_point": "train.py"})
        job_id = await provider.submit_job(spec)
        await asyncio.sleep(0.05)
        await provider.cancel_job(job_id)
        result = await provider.poll_status(job_id)
        assert result.status == JobStatus.CANCELLED


# ===================================================================
# TestLocalProviderInstances
# ===================================================================


class TestLocalProviderInstances:
    async def test_list_instance_types(self, tmp_path: Path) -> None:
        provider = _build_provider(tmp_path)
        instances = await provider.list_instance_types()
        assert len(instances) == 1
        inst = instances[0]
        assert isinstance(inst, InstanceType)
        assert inst.name == "local-cpu"
        assert inst.vcpus >= 1
        assert inst.price_per_hour_eur == 0.0
        assert inst.region == "local"

    async def test_list_instance_types_gpu_only(self, tmp_path: Path) -> None:
        provider = _build_provider(tmp_path)
        instances = await provider.list_instance_types(gpu_only=True)
        assert instances == []


# ===================================================================
# TestLocalProviderTimeout
# ===================================================================


class TestLocalProviderTimeout:
    async def test_timeout_terminates_process(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Process responds to SIGTERM: wait_for times out, terminate kills it."""
        monkeypatch.setattr(local_module, "_GRACE_PERIOD_SECONDS", 0.5)
        provider = _build_provider(tmp_path)

        fake = _FakeProcess(returncode=0, hang=True)

        # Set up job directly (avoid background task)
        job_dir = tmp_path / "local-timeout1"
        job_dir.mkdir()
        job = _LocalJob(
            job_id="local-timeout1",
            process=fake,  # type: ignore[arg-type]
            work_dir=job_dir,
            start_time=time.monotonic(),
            timeout_seconds=0.1,
        )
        job.status = JobStatus.RUNNING
        provider._jobs["local-timeout1"] = job

        # Acquire semaphore so _wait_for_completion can release it
        await provider._semaphore.acquire()

        # Call directly — no background task
        await provider._wait_for_completion("local-timeout1")

        assert job.status == JobStatus.FAILED
        assert "timed out" in (job.error or "")
        assert fake.terminated

    async def test_timeout_kills_stubborn_process(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Process ignores SIGTERM: wait_for times out, terminate, grace timeout, then kill."""
        monkeypatch.setattr(local_module, "_GRACE_PERIOD_SECONDS", 0.1)
        provider = _build_provider(tmp_path)

        class _StubbornProcess(_FakeProcess):
            """Process that ignores terminate, only responds to kill."""

            def __init__(self) -> None:
                super().__init__(returncode=0, hang=True)

            def terminate(self) -> None:
                self.terminated = True
                # Don't set the event — process ignores SIGTERM

            def kill(self) -> None:
                self.killed = True
                self._final_code = -9
                self._wait_event.set()

        stubborn = _StubbornProcess()

        # Set up job directly
        job_dir = tmp_path / "local-timeout2"
        job_dir.mkdir()
        job = _LocalJob(
            job_id="local-timeout2",
            process=stubborn,  # type: ignore[arg-type]
            work_dir=job_dir,
            start_time=time.monotonic(),
            timeout_seconds=0.1,
        )
        job.status = JobStatus.RUNNING
        provider._jobs["local-timeout2"] = job

        await provider._semaphore.acquire()
        await provider._wait_for_completion("local-timeout2")

        assert stubborn.terminated
        assert stubborn.killed
        assert job.status == JobStatus.FAILED


# ===================================================================
# TestLocalProviderEnvironment
# ===================================================================


class TestLocalProviderEnvironment:
    async def test_cuda_disabled_when_gpu_not_enabled(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        provider = _build_provider(tmp_path, gpu_enabled=False)
        fake = _FakeProcess(returncode=0)
        mock_exec = AsyncMock(return_value=fake)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_exec)
        spec = _make_spec(config={"entry_point": "train.py"})
        await provider.submit_job(spec)
        env = mock_exec.call_args.kwargs["env"]
        assert env["CUDA_VISIBLE_DEVICES"] == ""

    async def test_cuda_inherited_when_gpu_enabled(self, tmp_path: Path, monkeypatch: Any) -> None:
        provider = _build_provider(tmp_path, gpu_enabled=True)
        fake = _FakeProcess(returncode=0)
        mock_exec = AsyncMock(return_value=fake)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_exec)
        spec = _make_spec(config={"entry_point": "train.py"})
        await provider.submit_job(spec)
        env = mock_exec.call_args.kwargs["env"]
        # CUDA_VISIBLE_DEVICES should NOT be forced to empty
        assert env.get("CUDA_VISIBLE_DEVICES") != ""

    async def test_custom_env_vars_passed(self, tmp_path: Path, monkeypatch: Any) -> None:
        provider = _build_provider(tmp_path)
        fake = _FakeProcess(returncode=0)
        mock_exec = AsyncMock(return_value=fake)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_exec)
        spec = _make_spec(config={"entry_point": "train.py", "env": {"MY_VAR": "hello"}})
        await provider.submit_job(spec)
        env = mock_exec.call_args.kwargs["env"]
        assert env["MY_VAR"] == "hello"
        assert env["ARTENIC_JOB_ID"].startswith("local-")
        assert env["ARTENIC_SERVICE"] == "nlp"
        assert env["ARTENIC_MODEL"] == "bert"


# ===================================================================
# TestLocalProviderMetrics
# ===================================================================


class TestLocalProviderMetrics:
    def test_read_metrics_valid(self, tmp_path: Path) -> None:
        (tmp_path / "metrics.json").write_text(json.dumps({"accuracy": 0.9}), encoding="utf-8")
        result = LocalProvider._read_metrics(tmp_path)
        assert result == {"accuracy": 0.9}

    def test_read_metrics_no_file(self, tmp_path: Path) -> None:
        result = LocalProvider._read_metrics(tmp_path)
        assert result is None

    def test_read_metrics_invalid_json(self, tmp_path: Path) -> None:
        (tmp_path / "metrics.json").write_text("not json{", encoding="utf-8")
        result = LocalProvider._read_metrics(tmp_path)
        assert result is None

    def test_read_metrics_not_dict(self, tmp_path: Path) -> None:
        (tmp_path / "metrics.json").write_text("[1, 2, 3]", encoding="utf-8")
        result = LocalProvider._read_metrics(tmp_path)
        assert result is None


# ===================================================================
# TestLocalProviderCost
# ===================================================================


class TestLocalProviderCost:
    async def test_cost_always_zero(self, tmp_path: Path, monkeypatch: Any) -> None:
        provider = _build_provider(tmp_path)
        fake = _FakeProcess(returncode=0)
        monkeypatch.setattr(
            asyncio,
            "create_subprocess_exec",
            AsyncMock(return_value=fake),
        )
        spec = _make_spec(config={"entry_point": "train.py"})
        job_id = await provider.submit_job(spec)
        await asyncio.sleep(0.05)
        result = await provider.poll_status(job_id)
        assert result.cost_eur == 0.0


# ===================================================================
# TestGetSystemMemoryGb
# ===================================================================


class TestGetSystemMemoryGb:
    def test_returns_float(self) -> None:
        result = _get_system_memory_gb()
        assert isinstance(result, float)
        assert result >= 0.0

    def test_fallback_returns_zero(self, monkeypatch: Any) -> None:
        if hasattr(os, "sysconf"):
            monkeypatch.setattr(os, "sysconf", lambda _: -1)
        result = _get_system_memory_gb()
        assert result == 0.0

    def test_oserror_returns_zero(self, monkeypatch: Any) -> None:
        if hasattr(os, "sysconf"):
            monkeypatch.setattr(os, "sysconf", lambda _: (_ for _ in ()).throw(OSError("nope")))
        result = _get_system_memory_gb()
        assert result == 0.0

    def test_sysconf_positive_values(self, monkeypatch: Any) -> None:
        """Cover the os.sysconf path with positive values (Linux-like)."""

        # Simulate 8 GB: 2_097_152 pages * 4096 bytes/page = 8 GiB
        def _fake_sysconf(name: str) -> int:
            if name == "SC_PHYS_PAGES":
                return 2_097_152
            if name == "SC_PAGE_SIZE":
                return 4096
            return 0

        monkeypatch.setattr(os, "sysconf", _fake_sysconf, raising=False)
        result = _get_system_memory_gb()
        assert result == 8.0

    def test_sysconf_raises_value_error(self, monkeypatch: Any) -> None:
        """Cover the except (ValueError, OSError) branch."""

        def _raise_value_error(_name: str) -> int:
            raise ValueError("bad sysconf")

        monkeypatch.setattr(os, "sysconf", _raise_value_error, raising=False)
        result = _get_system_memory_gb()
        assert result == 0.0


# ===================================================================
# TestLocalJob
# ===================================================================


class TestLocalJob:
    def test_local_job_defaults(self, tmp_path: Path) -> None:
        job = _LocalJob(
            job_id="local-test1234",
            process=None,
            work_dir=tmp_path,
            start_time=0.0,
            timeout_seconds=3600.0,
        )
        assert job.status == JobStatus.PENDING
        assert job.error is None
        assert job.metrics == {}


# ===================================================================
# TestTerminateProcess
# ===================================================================


class TestTerminateProcess:
    async def test_process_lookup_error(self) -> None:
        """If the process is already gone, ProcessLookupError is caught."""
        fake = _FakeProcess(returncode=0, hang=True)

        def _raise_plookup() -> None:
            raise ProcessLookupError

        fake.terminate = _raise_plookup  # type: ignore[assignment]
        # Should not raise
        await LocalProvider._terminate_process(fake)  # type: ignore[arg-type]


# ===================================================================
# TestCloseFiles
# ===================================================================


class TestCloseFiles:
    def test_close_files_noop_when_none(self, tmp_path: Path) -> None:
        job = _LocalJob(
            job_id="local-test",
            process=None,
            work_dir=tmp_path,
            start_time=0.0,
            timeout_seconds=3600.0,
        )
        # Should not raise
        LocalProvider._close_files(job)
        assert job._stdout_file is None
        assert job._stderr_file is None

    def test_close_files_closes_handles(self, tmp_path: Path) -> None:
        job = _LocalJob(
            job_id="local-test",
            process=None,
            work_dir=tmp_path,
            start_time=0.0,
            timeout_seconds=3600.0,
        )
        f = (tmp_path / "test.log").open("w")
        job._stdout_file = f
        LocalProvider._close_files(job)
        assert job._stdout_file is None
        assert f.closed


# ===================================================================
# TestLocalProviderSpecTimeout
# ===================================================================


class TestLocalProviderSpecTimeout:
    async def test_spec_max_runtime_overrides_default(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        provider = _build_provider(tmp_path, default_timeout_hours=24.0)
        fake = _FakeProcess(returncode=0)
        monkeypatch.setattr(
            asyncio,
            "create_subprocess_exec",
            AsyncMock(return_value=fake),
        )
        spec = _make_spec(
            config={"entry_point": "train.py"},
            max_runtime_hours=2.0,
        )
        job_id = await provider.submit_job(spec)
        job = provider._jobs[job_id]
        assert job.timeout_seconds == 2.0 * 3600

    async def test_default_timeout_used_when_spec_is_zero(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        provider = _build_provider(tmp_path, default_timeout_hours=12.0)
        fake = _FakeProcess(returncode=0)
        monkeypatch.setattr(
            asyncio,
            "create_subprocess_exec",
            AsyncMock(return_value=fake),
        )
        spec = _make_spec(
            config={"entry_point": "train.py"},
            max_runtime_hours=0,
        )
        job_id = await provider.submit_job(spec)
        job = provider._jobs[job_id]
        assert job.timeout_seconds == 12.0 * 3600


# ===================================================================
# TestLocalProviderDefaultWorkDir
# ===================================================================


class TestLocalProviderDefaultWorkDir:
    def test_empty_work_dir_uses_tempdir(self) -> None:
        import tempfile

        provider = LocalProvider(work_dir="")
        assert "artenic-local" in str(provider._work_dir_root)
        assert tempfile.gettempdir() in str(provider._work_dir_root)

    def test_custom_work_dir(self, tmp_path: Path) -> None:
        provider = LocalProvider(work_dir=str(tmp_path / "custom"))
        assert str(provider._work_dir_root).endswith("custom")


# ===================================================================
# TestLocalProviderDefaultPython
# ===================================================================


class TestLocalProviderDefaultPython:
    def test_empty_python_uses_sys_executable(self) -> None:
        import sys

        provider = LocalProvider(python_executable="")
        assert provider._python == sys.executable

    def test_custom_python(self) -> None:
        provider = LocalProvider(python_executable="/opt/python/bin/python3")
        assert provider._python == "/opt/python/bin/python3"
