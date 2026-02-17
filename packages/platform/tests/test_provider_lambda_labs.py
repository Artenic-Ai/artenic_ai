"""Tests for artenic_ai_platform.providers.lambda_labs — LambdaLabsProvider."""

from __future__ import annotations

import asyncio
import contextlib
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from artenic_ai_platform.providers.base import (
    CloudJobStatus,
    JobStatus,
    TrainingSpec,
)

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _spec(**overrides) -> TrainingSpec:
    """Build a minimal TrainingSpec for tests."""
    defaults = {
        "service": "training",
        "model": "llama-7b",
        "provider": "lambda_labs",
        "config": {},
    }
    defaults.update(overrides)
    return TrainingSpec(**defaults)


def _mock_response(
    *,
    status_code: int = 200,
    json_data: dict | None = None,
    text: str = "ok",
):
    """Return a MagicMock that behaves like an httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


# ---------------------------------------------------------------
# Module-level patches applied to every test
# ---------------------------------------------------------------

_MODULE = "artenic_ai_platform.providers.lambda_labs"


@pytest.fixture(autouse=True)
def _patch_httpx():
    """Ensure httpx is available and mock AsyncClient."""
    with patch(f"{_MODULE}._HTTPX_AVAILABLE", True), patch(f"{_MODULE}.httpx") as mock_httpx:
        mock_httpx.AsyncClient = MagicMock
        mock_httpx.Timeout = MagicMock
        yield mock_httpx


@pytest.fixture()
def provider():
    """Create a LambdaLabsProvider with mocked httpx."""
    from artenic_ai_platform.providers.lambda_labs import (
        LambdaLabsProvider,
    )

    return LambdaLabsProvider(
        api_key="test-key",
        ssh_key_name="my-ssh-key",
        instance_type="gpu_1x_a10",
        region="us-tx-3",
    )


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------


class TestLambdaLabsInit:
    """Constructor and provider_name tests."""

    def test_provider_name(self, provider):
        assert provider.provider_name == "lambda_labs"

    def test_init_stores_fields(self, provider):
        assert provider._api_key == "test-key"
        assert provider._ssh_key_name == "my-ssh-key"
        assert provider._default_instance_type == "gpu_1x_a10"
        assert provider._default_region == "us-tx-3"
        assert provider._connected is False

    def test_init_raises_without_httpx(self):
        with patch(f"{_MODULE}._HTTPX_AVAILABLE", False), pytest.raises(ImportError, match="httpx"):
            from artenic_ai_platform.providers.lambda_labs import (
                LambdaLabsProvider,
            )

            LambdaLabsProvider(
                api_key="k",
                ssh_key_name="s",
            )


class TestLambdaLabsConnect:
    """_connect / _disconnect tests."""

    @pytest.mark.asyncio
    async def test_connect_creates_client(self, provider):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_mock_response(json_data={"data": {}}))
        with patch(f"{_MODULE}.httpx") as httpx_mod:
            httpx_mod.AsyncClient.return_value = mock_client
            httpx_mod.Timeout = MagicMock()
            await provider._connect()
        assert provider._client is mock_client
        mock_client.get.assert_awaited_once_with("/instance-types")

    @pytest.mark.asyncio
    async def test_disconnect_closes_client(self, provider):
        mock_client = AsyncMock()
        provider._client = mock_client
        await provider._disconnect()
        mock_client.aclose.assert_awaited_once()
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_disconnect_noop_when_no_client(self, provider):
        provider._client = None
        await provider._disconnect()  # should not raise


class TestLambdaLabsListInstances:
    """_list_instances tests."""

    @pytest.mark.asyncio
    async def test_list_instances_parses_response(self, provider):
        api_data = {
            "data": {
                "gpu_1x_a10": {
                    "instance_type": {
                        "description": "1x NVIDIA A10 (24 GB)",
                        "price_cents_per_hour": 75,
                        "specs": {
                            "vcpus": 30,
                            "memory_gib": 200,
                            "gpus": 1,
                        },
                    },
                    "regions_with_capacity_available": [
                        {"name": "us-tx-3"},
                    ],
                },
            },
        }
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_mock_response(json_data=api_data))
        provider._client = mock_client

        result = await provider._list_instances()
        assert len(result) == 1
        inst = result[0]
        assert inst.name == "gpu_1x_a10"
        assert inst.vcpus == 30
        assert inst.gpu_type == "A10"
        assert inst.gpu_count == 1
        assert inst.price_per_hour_eur == 0.75
        assert inst.available is True

    @pytest.mark.asyncio
    async def test_list_instances_gpu_only_filter(self, provider):
        api_data = {
            "data": {
                "cpu_only": {
                    "instance_type": {
                        "description": "CPU instance",
                        "price_cents_per_hour": 10,
                        "specs": {"vcpus": 4, "memory_gib": 16, "gpus": 0},
                    },
                    "regions_with_capacity_available": [],
                },
            },
        }
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_mock_response(json_data=api_data))
        provider._client = mock_client

        result = await provider._list_instances(gpu_only=True)
        assert len(result) == 0


class TestLambdaLabsUploadCode:
    """_upload_code tests."""

    @pytest.mark.asyncio
    async def test_upload_code_with_path(self, provider):
        spec = _spec(config={"code_path": "/tmp/train"})
        uri = await provider._upload_code(spec)
        assert uri == "deferred:///tmp/train"

    @pytest.mark.asyncio
    async def test_upload_code_without_path(self, provider):
        spec = _spec(config={})
        uri = await provider._upload_code(spec)
        assert uri == "deferred://"


class TestLambdaLabsProvision:
    """_provision_and_start tests."""

    @pytest.mark.asyncio
    async def test_provision_happy_path(self, provider):
        launch_resp = _mock_response(
            status_code=200,
            json_data={
                "data": {"instance_ids": ["i-abc123"]},
            },
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=launch_resp)
        provider._client = mock_client

        spec = _spec()
        job_id = await provider._provision_and_start(spec)
        assert job_id.startswith("lambda-")
        assert job_id in provider._jobs
        state = provider._jobs[job_id]
        assert state.instance_id == "i-abc123"

    @pytest.mark.asyncio
    async def test_provision_failure_status(self, provider):
        fail_resp = _mock_response(status_code=500, text="server error")
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=fail_resp)
        provider._client = mock_client

        with pytest.raises(RuntimeError, match="launch failed"):
            await provider._provision_and_start(_spec())

    @pytest.mark.asyncio
    async def test_provision_no_instance_ids(self, provider):
        resp = _mock_response(
            status_code=200,
            json_data={"data": {"instance_ids": []}},
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=resp)
        provider._client = mock_client

        with pytest.raises(RuntimeError, match="no instance IDs"):
            await provider._provision_and_start(_spec())


class TestLambdaLabsPoll:
    """_poll_provider tests."""

    @pytest.mark.asyncio
    async def test_poll_unknown_job(self, provider):
        provider._client = AsyncMock()
        status = await provider._poll_provider("no-such-job")
        assert status.status == JobStatus.FAILED
        assert "Unknown job" in (status.error or "")

    @pytest.mark.asyncio
    async def test_poll_active_instance(self, provider):
        from artenic_ai_platform.providers.lambda_labs import (
            _LambdaJobState,
        )

        provider._jobs["j1"] = _LambdaJobState(
            instance_id="i-1",
            region="us-tx-3",
            created_at=time.time() - 60,
            spec=_spec(),
            instance_type="gpu_1x_a10",
        )
        resp = _mock_response(
            json_data={
                "data": {"status": "active", "ip": "1.2.3.4"},
            }
        )
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=resp)
        provider._client = mock_client

        result = await provider._poll_provider("j1")
        assert result.status == JobStatus.RUNNING
        assert provider._jobs["j1"].ip_address == "1.2.3.4"

    @pytest.mark.asyncio
    async def test_poll_api_error(self, provider):
        from artenic_ai_platform.providers.lambda_labs import (
            _LambdaJobState,
        )

        provider._jobs["j2"] = _LambdaJobState(
            instance_id="i-2",
            region="us-tx-3",
            created_at=time.time(),
            spec=_spec(),
            instance_type="gpu_1x_a10",
        )
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("timeout"))
        provider._client = mock_client

        result = await provider._poll_provider("j2")
        assert result.status == JobStatus.FAILED
        assert "Cannot reach" in (result.error or "")


class TestLambdaLabsCollectArtifacts:
    """_collect_artifacts tests."""

    @pytest.mark.asyncio
    async def test_collect_unknown_job(self, provider):
        result = await provider._collect_artifacts(
            "unknown",
            CloudJobStatus(
                provider_job_id="unknown",
                status=JobStatus.COMPLETED,
            ),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_collect_no_ip(self, provider):
        from artenic_ai_platform.providers.lambda_labs import (
            _LambdaJobState,
        )

        provider._jobs["j1"] = _LambdaJobState(
            instance_id="i-1",
            region="us-tx-3",
            created_at=time.time(),
            spec=_spec(),
            instance_type="gpu_1x_a10",
        )
        result = await provider._collect_artifacts(
            "j1",
            CloudJobStatus(
                provider_job_id="j1",
                status=JobStatus.COMPLETED,
            ),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_collect_scp_success(self, provider):
        from artenic_ai_platform.providers.lambda_labs import (
            _LambdaJobState,
        )

        state = _LambdaJobState(
            instance_id="i-1",
            region="us-tx-3",
            created_at=time.time(),
            spec=_spec(),
            instance_type="gpu_1x_a10",
        )
        state.ip_address = "1.2.3.4"
        provider._jobs["j1"] = state

        with (
            patch(
                "asyncio.to_thread",
                new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
            ),
            patch(f"{_MODULE}.subprocess.run") as mock_run,
        ):
            result = await provider._collect_artifacts(
                "j1",
                CloudJobStatus(
                    provider_job_id="j1",
                    status=JobStatus.COMPLETED,
                ),
            )
        assert result is not None
        assert "j1" in result
        mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_collect_scp_failure(self, provider):
        from artenic_ai_platform.providers.lambda_labs import (
            _LambdaJobState,
        )

        state = _LambdaJobState(
            instance_id="i-1",
            region="us-tx-3",
            created_at=time.time(),
            spec=_spec(),
            instance_type="gpu_1x_a10",
        )
        state.ip_address = "1.2.3.4"
        provider._jobs["j1"] = state

        with patch(
            "asyncio.to_thread",
            new=AsyncMock(side_effect=OSError("scp failed")),
        ):
            result = await provider._collect_artifacts(
                "j1",
                CloudJobStatus(
                    provider_job_id="j1",
                    status=JobStatus.COMPLETED,
                ),
            )
        assert result is None


class TestLambdaLabsCleanup:
    """_cleanup_compute tests."""

    @pytest.mark.asyncio
    async def test_cleanup_unknown_job(self, provider):
        provider._client = AsyncMock()
        await provider._cleanup_compute("nope")
        # Should return without error

    @pytest.mark.asyncio
    async def test_cleanup_terminates_instance(self, provider):
        from artenic_ai_platform.providers.lambda_labs import (
            _LambdaJobState,
        )

        provider._jobs["j1"] = _LambdaJobState(
            instance_id="i-1",
            region="us-tx-3",
            created_at=time.time(),
            spec=_spec(),
            instance_type="gpu_1x_a10",
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=_mock_response(status_code=200))
        provider._client = mock_client

        await provider._cleanup_compute("j1")
        assert "j1" not in provider._jobs
        mock_client.post.assert_awaited_once()


class TestLambdaLabsCancel:
    """_cancel_provider_job tests."""

    @pytest.mark.asyncio
    async def test_cancel_unknown_job(self, provider):
        provider._client = AsyncMock()
        await provider._cancel_provider_job("nope")

    @pytest.mark.asyncio
    async def test_cancel_terminates_instance(self, provider):
        from artenic_ai_platform.providers.lambda_labs import (
            _LambdaJobState,
        )

        provider._jobs["j1"] = _LambdaJobState(
            instance_id="i-1",
            region="us-tx-3",
            created_at=time.time(),
            spec=_spec(),
            instance_type="gpu_1x_a10",
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=_mock_response(status_code=200))
        provider._client = mock_client

        await provider._cancel_provider_job("j1")
        mock_client.post.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cancel_raises_on_http_error(self, provider):
        from artenic_ai_platform.providers.lambda_labs import (
            _LambdaJobState,
        )

        provider._jobs["j1"] = _LambdaJobState(
            instance_id="i-1",
            region="us-tx-3",
            created_at=time.time(),
            spec=_spec(),
            instance_type="gpu_1x_a10",
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("network error"))
        provider._client = mock_client

        with pytest.raises(Exception, match="network error"):
            await provider._cancel_provider_job("j1")


# ---------------------------------------------------------------
# Additional tests for full coverage
# ---------------------------------------------------------------


class TestLambdaLabsProvisionFileSystem:
    """Cover the file_system_names branch (line 248) in _provision_and_start."""

    @pytest.mark.asyncio
    async def test_provision_with_file_system_names(self, provider):
        """When spec.config has 'file_system_names', they are included in the launch payload."""
        launch_resp = _mock_response(
            status_code=200,
            json_data={"data": {"instance_ids": ["i-fs1"]}},
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=launch_resp)
        provider._client = mock_client

        spec = _spec(config={"file_system_names": ["fs-data", "fs-models"]})
        job_id = await provider._provision_and_start(spec)

        assert job_id.startswith("lambda-")
        # Verify file_system_names was passed in the launch payload
        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["file_system_names"] == ["fs-data", "fs-models"]


class TestLambdaLabsProvisionDeferredUpload:
    """Cover the deferred code upload branch (lines 294-298)."""

    @pytest.mark.asyncio
    async def test_provision_with_code_path_schedules_upload(self, provider):
        """When spec.config has 'code_path', a background task is created."""
        launch_resp = _mock_response(
            status_code=200,
            json_data={"data": {"instance_ids": ["i-upload1"]}},
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=launch_resp)
        # Stub get so _deferred_code_upload's polling does not hang
        mock_client.get = AsyncMock(
            return_value=_mock_response(
                json_data={"data": {"status": "terminated", "ip": None}},
            )
        )
        provider._client = mock_client

        spec = _spec(config={"code_path": "/tmp/my_code"})

        with patch(
            f"{_MODULE}.asyncio.ensure_future",
            wraps=asyncio.ensure_future,
        ) as mock_ensure:
            job_id = await provider._provision_and_start(spec)
            # ensure_future should have been called to schedule the upload
            mock_ensure.assert_called_once()

        assert job_id.startswith("lambda-")
        assert hasattr(provider, "_background_tasks")
        assert len(provider._background_tasks) == 1
        # Cancel the background task to avoid test side-effects
        provider._background_tasks[0].cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await provider._background_tasks[0]


class TestLambdaLabsCleanupEdgeCases:
    """Cover cleanup non-success status (lines 458-462) and exception (lines 463-469)."""

    @pytest.mark.asyncio
    async def test_cleanup_non_success_status(self, provider):
        """When terminate returns a non-success status code, a warning is logged."""
        from artenic_ai_platform.providers.lambda_labs import _LambdaJobState

        provider._jobs["j-warn"] = _LambdaJobState(
            instance_id="i-warn",
            region="us-tx-3",
            created_at=time.time(),
            spec=_spec(),
            instance_type="gpu_1x_a10",
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            return_value=_mock_response(status_code=500, text="internal error"),
        )
        provider._client = mock_client

        # Should not raise but should log a warning
        await provider._cleanup_compute("j-warn")
        # Job should still be removed from tracking
        assert "j-warn" not in provider._jobs

    @pytest.mark.asyncio
    async def test_cleanup_exception_reraises(self, provider):
        """When terminate raises an exception, it is re-raised."""
        from artenic_ai_platform.providers.lambda_labs import _LambdaJobState

        provider._jobs["j-exc"] = _LambdaJobState(
            instance_id="i-exc",
            region="us-tx-3",
            created_at=time.time(),
            spec=_spec(),
            instance_type="gpu_1x_a10",
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=RuntimeError("connection reset"))
        provider._client = mock_client

        with pytest.raises(RuntimeError, match="connection reset"):
            await provider._cleanup_compute("j-exc")


class TestLambdaLabsCancelNonSuccessStatus:
    """Cover cancel non-success status warning (line 506)."""

    @pytest.mark.asyncio
    async def test_cancel_non_success_status(self, provider):
        """When terminate returns non-success, a warning is logged but no error raised."""
        from artenic_ai_platform.providers.lambda_labs import _LambdaJobState

        provider._jobs["j-cs"] = _LambdaJobState(
            instance_id="i-cs",
            region="us-tx-3",
            created_at=time.time(),
            spec=_spec(),
            instance_type="gpu_1x_a10",
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            return_value=_mock_response(status_code=409, text="conflict"),
        )
        provider._client = mock_client

        # Should not raise
        await provider._cancel_provider_job("j-cs")


class TestLambdaLabsDeferredCodeUpload:
    """Cover _deferred_code_upload (lines 533-628)."""

    def _make_state(self, jid="j-1", iid="i-1"):
        from artenic_ai_platform.providers.lambda_labs import _LambdaJobState

        return _LambdaJobState(
            instance_id=iid,
            region="us-tx-3",
            created_at=time.time(),
            spec=_spec(),
            instance_type="gpu_1x_a10",
        )

    @pytest.mark.asyncio
    async def test_deferred_upload_active_instance(self, provider):
        """Upload code when instance becomes active with an IP."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=[
                _mock_response(json_data={"data": {"status": "booting", "ip": None}}),
                _mock_response(json_data={"data": {"status": "active", "ip": "10.0.0.1"}}),
            ]
        )
        provider._client = mock_client
        provider._jobs["j1"] = self._make_state("j1", "i1")

        mock_run = MagicMock()
        with (
            patch(f"{_MODULE}.asyncio.sleep", new=AsyncMock()),
            patch.object(provider, "_wait_for_ssh", new=AsyncMock()),
            patch(
                f"{_MODULE}.asyncio.to_thread",
                new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
            ),
            patch(f"{_MODULE}.subprocess.run", mock_run),
        ):
            await provider._deferred_code_upload("j1", "i1", "/tmp/code")

        assert provider._jobs["j1"].ip_address == "10.0.0.1"
        assert mock_run.call_count == 2

    @pytest.mark.asyncio
    async def test_deferred_upload_terminated_instance(self, provider):
        """Terminated instance aborts upload."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            return_value=_mock_response(json_data={"data": {"status": "terminated", "ip": None}})
        )
        provider._client = mock_client
        provider._jobs["j2"] = self._make_state("j2", "i2")

        with patch(f"{_MODULE}.asyncio.sleep", new=AsyncMock()):
            await provider._deferred_code_upload("j2", "i2", "/tmp/code")

        assert provider._jobs["j2"].ip_address is None

    @pytest.mark.asyncio
    async def test_deferred_upload_timeout(self, provider):
        """Timeout aborts upload."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            return_value=_mock_response(json_data={"data": {"status": "booting", "ip": None}})
        )
        provider._client = mock_client
        provider._jobs["j3"] = self._make_state("j3", "i3")

        call_count = 0

        def fake_time():
            nonlocal call_count
            call_count += 1
            return 0.0 if call_count <= 2 else 301.0

        with (
            patch(f"{_MODULE}.asyncio.sleep", new=AsyncMock()),
            patch(f"{_MODULE}.time.time", side_effect=fake_time),
        ):
            await provider._deferred_code_upload("j3", "i3", "/tmp/code")

        assert provider._jobs["j3"].ip_address is None

    @pytest.mark.asyncio
    async def test_deferred_upload_polling_exception(self, provider):
        """Polling exception is caught; retry succeeds."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=[
                ConnectionError("net"),
                _mock_response(json_data={"data": {"status": "active", "ip": "10.0.0.2"}}),
            ]
        )
        provider._client = mock_client
        provider._jobs["j4"] = self._make_state("j4", "i4")

        with (
            patch(f"{_MODULE}.asyncio.sleep", new=AsyncMock()),
            patch.object(provider, "_wait_for_ssh", new=AsyncMock()),
            patch(
                f"{_MODULE}.asyncio.to_thread",
                new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
            ),
            patch(f"{_MODULE}.subprocess.run"),
        ):
            await provider._deferred_code_upload("j4", "i4", "/tmp/code")

        assert provider._jobs["j4"].ip_address == "10.0.0.2"

    @pytest.mark.asyncio
    async def test_deferred_upload_scp_failure(self, provider):
        """SCP failure is logged, no exception raised."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            return_value=_mock_response(json_data={"data": {"status": "active", "ip": "10.0.0.3"}})
        )
        provider._client = mock_client
        provider._jobs["j5"] = self._make_state("j5", "i5")

        with (
            patch(f"{_MODULE}.asyncio.sleep", new=AsyncMock()),
            patch.object(provider, "_wait_for_ssh", new=AsyncMock()),
            patch(
                f"{_MODULE}.asyncio.to_thread",
                new=AsyncMock(side_effect=OSError("scp failed")),
            ),
        ):
            await provider._deferred_code_upload("j5", "i5", "/tmp/code")

    @pytest.mark.asyncio
    async def test_deferred_upload_unhealthy_instance(self, provider):
        """Unhealthy instance aborts upload."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            return_value=_mock_response(json_data={"data": {"status": "unhealthy", "ip": None}})
        )
        provider._client = mock_client
        provider._jobs["j6"] = self._make_state("j6", "i6")

        with patch(f"{_MODULE}.asyncio.sleep", new=AsyncMock()):
            await provider._deferred_code_upload("j6", "i6", "/tmp/code")

        assert provider._jobs["j6"].ip_address is None


class TestLambdaLabsWaitForSSH:
    """Cover _wait_for_ssh (lines 637-667)."""

    @pytest.mark.asyncio
    async def test_wait_for_ssh_success(self, provider):
        """SSH becomes ready on first attempt."""
        mock_result = MagicMock()
        mock_result.returncode = 0

        with (
            patch(
                f"{_MODULE}.asyncio.to_thread",
                new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
            ),
            patch(f"{_MODULE}.subprocess.run", return_value=mock_result),
        ):
            await provider._wait_for_ssh("10.0.0.1")

    @pytest.mark.asyncio
    async def test_wait_for_ssh_eventually_ready(self, provider):
        """SSH fails initially, then succeeds."""
        fail_result = MagicMock()
        fail_result.returncode = 1
        ok_result = MagicMock()
        ok_result.returncode = 0

        with (
            patch(
                f"{_MODULE}.asyncio.to_thread",
                new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
            ),
            patch(
                f"{_MODULE}.subprocess.run",
                side_effect=[fail_result, ok_result],
            ),
            patch(f"{_MODULE}.asyncio.sleep", new=AsyncMock()),
        ):
            await provider._wait_for_ssh("10.0.0.1")

    @pytest.mark.asyncio
    async def test_wait_for_ssh_timeout(self, provider):
        """SSH never becomes ready -- TimeoutError is raised."""
        fail_result = MagicMock()
        fail_result.returncode = 1

        with (
            patch(
                f"{_MODULE}.asyncio.to_thread",
                new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
            ),
            patch(f"{_MODULE}.subprocess.run", return_value=fail_result),
            patch(f"{_MODULE}.asyncio.sleep", new=AsyncMock()),
            patch(f"{_MODULE}.time.time") as mock_time,
            pytest.raises(TimeoutError, match="SSH not available"),
        ):
            # First call sets deadline, second is in loop, third exceeds deadline
            mock_time.side_effect = [0.0, 0.0, 200.0]
            await provider._wait_for_ssh("10.0.0.1", timeout=180.0)

    @pytest.mark.asyncio
    async def test_wait_for_ssh_exception_in_subprocess(self, provider):
        """When subprocess.run raises, the exception is caught and SSH wait continues."""
        ok_result = MagicMock()
        ok_result.returncode = 0

        with (
            patch(
                f"{_MODULE}.asyncio.to_thread",
                new=AsyncMock(
                    side_effect=[
                        OSError("connection refused"),
                        ok_result,
                    ]
                ),
            ),
            patch(f"{_MODULE}.asyncio.sleep", new=AsyncMock()),
        ):
            # Wrapping to_thread needs to match the actual usage; let's do it
            # by patching subprocess.run instead
            pass

        # Better approach: patch subprocess.run to raise, then succeed
        with (
            patch(
                f"{_MODULE}.asyncio.to_thread",
                new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
            ),
            patch(
                f"{_MODULE}.subprocess.run",
                side_effect=[OSError("refused"), ok_result],
            ),
            patch(f"{_MODULE}.asyncio.sleep", new=AsyncMock()),
        ):
            await provider._wait_for_ssh("10.0.0.1")


class TestLambdaLabsEstimateCost:
    """Cover _estimate_cost with hourly_price set (lines 680-681)."""

    def test_estimate_cost_with_hourly_price(self, provider):
        from artenic_ai_platform.providers.lambda_labs import _LambdaJobState

        state = _LambdaJobState(
            instance_id="i-cost",
            region="us-tx-3",
            created_at=time.time(),
            spec=_spec(),
            instance_type="gpu_1x_a10",
            hourly_price=1.50,
        )
        cost = provider._estimate_cost(state, 3600.0)
        assert cost == 1.50

    def test_estimate_cost_with_hourly_price_partial_hour(self, provider):
        from artenic_ai_platform.providers.lambda_labs import _LambdaJobState

        state = _LambdaJobState(
            instance_id="i-cost2",
            region="us-tx-3",
            created_at=time.time(),
            spec=_spec(),
            instance_type="gpu_1x_a10",
            hourly_price=2.00,
        )
        cost = provider._estimate_cost(state, 1800.0)  # 0.5 hours
        assert cost == 1.0

    def test_estimate_cost_no_hourly_price(self, provider):
        from artenic_ai_platform.providers.lambda_labs import _LambdaJobState

        state = _LambdaJobState(
            instance_id="i-cost3",
            region="us-tx-3",
            created_at=time.time(),
            spec=_spec(),
            instance_type="gpu_1x_a10",
        )
        cost = provider._estimate_cost(state, 3600.0)
        assert cost is None


class TestLambdaLabsGpuParsing:
    """Cover _parse_gpu_type_from_description fallback (line 747)."""

    def test_parse_gpu_from_description(self):
        from artenic_ai_platform.providers.lambda_labs import (
            _parse_gpu_type_from_description,
        )

        assert _parse_gpu_type_from_description("1x NVIDIA A10 (24 GB)", "gpu_1x_a10") == "A10"
        assert _parse_gpu_type_from_description("8x NVIDIA H100 SXM", "gpu_8x_h100_sxm5") == "H100"

    def test_parse_gpu_from_type_name_fallback(self):
        """When description has no known GPU, fall back to type name."""
        from artenic_ai_platform.providers.lambda_labs import (
            _parse_gpu_type_from_description,
        )

        # Description doesn't contain known GPU, but type name does
        result = _parse_gpu_type_from_description("Some generic GPU instance", "gpu_1x_rtx4090")
        assert result == "RTX 4090"

    def test_parse_gpu_fallback_to_generic(self):
        """When neither description nor type name match, return 'GPU'."""
        from artenic_ai_platform.providers.lambda_labs import (
            _parse_gpu_type_from_description,
        )

        result = _parse_gpu_type_from_description("Unknown accelerator", "custom_instance_xyz")
        assert result == "GPU"

    def test_parse_gpu_l40s_from_description(self):
        """L40S should be found before L40."""
        from artenic_ai_platform.providers.lambda_labs import (
            _parse_gpu_type_from_description,
        )

        result = _parse_gpu_type_from_description("1x NVIDIA L40S", "gpu_1x_l40s")
        assert result == "L40S"

    def test_parse_gpu_gh200_from_name(self):
        """GH200 from type name when not in description."""
        from artenic_ai_platform.providers.lambda_labs import (
            _parse_gpu_type_from_description,
        )

        result = _parse_gpu_type_from_description("Some instance", "gpu-1x-gh200")
        assert result == "GH200"


class TestLambdaLabsListInstancesRegion:
    """Cover region filtering and pricing edge cases in _list_instances."""

    @pytest.mark.asyncio
    async def test_list_instances_with_region_filter(self, provider):
        """When region is specified and instance is not available there."""
        api_data = {
            "data": {
                "gpu_1x_a10": {
                    "instance_type": {
                        "description": "1x NVIDIA A10 (24 GB)",
                        "price_cents_per_hour": 75,
                        "specs": {"vcpus": 30, "memory_gib": 200, "gpus": 1},
                    },
                    "regions_with_capacity_available": [
                        {"name": "us-west-1"},
                    ],
                },
            },
        }
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_mock_response(json_data=api_data))
        provider._client = mock_client

        result = await provider._list_instances(region="us-east-1")
        assert len(result) == 1
        inst = result[0]
        # Instance is not available in us-east-1
        assert inst.available is False
        assert inst.region is None

    @pytest.mark.asyncio
    async def test_list_instances_no_price(self, provider):
        """When price_cents_per_hour is None, price is 0.0."""
        api_data = {
            "data": {
                "gpu_1x_a10": {
                    "instance_type": {
                        "description": "1x NVIDIA A10",
                        "price_cents_per_hour": None,
                        "specs": {"vcpus": 4, "memory_gib": 16, "gpus": 1},
                    },
                    "regions_with_capacity_available": [],
                },
            },
        }
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_mock_response(json_data=api_data))
        provider._client = mock_client

        result = await provider._list_instances()
        assert len(result) == 1
        assert result[0].price_per_hour_eur == 0.0
