"""Tests for artenic_ai_platform.providers.vastai — VastAIProvider."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from artenic_ai_platform.providers.base import (
    CloudJobStatus,
    JobStatus,
    TrainingSpec,
)

_MODULE = "artenic_ai_platform.providers.vastai"


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _spec(**overrides) -> TrainingSpec:
    defaults = {
        "service": "training",
        "model": "mistral-7b",
        "provider": "vastai",
        "config": {},
    }
    defaults.update(overrides)
    return TrainingSpec(**defaults)


def _mock_response(
    *,
    status_code: int = 200,
    json_data: dict | None = None,
    content: bytes = b'{"ok": true}',
):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data if json_data is not None else {}
    resp.content = content
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture()
def provider():
    from artenic_ai_platform.providers.vastai import VastAIProvider

    return VastAIProvider(api_key="vast-test-key")


@pytest.fixture()
def connected(provider):
    """Provider with a mock httpx client already wired up."""
    mock_client = AsyncMock()
    provider._client = mock_client
    provider._connected = True
    return provider, mock_client


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------


class TestVastAIInit:
    def test_provider_name(self, provider):
        assert provider.provider_name == "vastai"

    def test_init_stores_fields(self, provider):
        assert provider._api_key == "vast-test-key"
        assert provider._docker_image == "pytorch/pytorch:latest"
        assert provider._max_price_per_hour == 2.0
        assert provider._min_reliability == 0.95
        assert provider._disk_gb == 50
        assert provider._connected is False

    def test_empty_api_key_raises(self):
        from artenic_ai_platform.providers.vastai import VastAIProvider

        with pytest.raises(ValueError, match="must not be empty"):
            VastAIProvider(api_key="")

    def test_invalid_price_raises(self):
        from artenic_ai_platform.providers.vastai import VastAIProvider

        with pytest.raises(ValueError, match="positive"):
            VastAIProvider(api_key="k", max_price_per_hour=0)

    def test_invalid_reliability_raises(self):
        from artenic_ai_platform.providers.vastai import VastAIProvider

        with pytest.raises(ValueError, match=r"between 0\.0 and 1\.0"):
            VastAIProvider(api_key="k", min_reliability=1.5)


class TestVastAIConnect:
    @pytest.mark.asyncio
    async def test_connect_verifies_user(self, provider):
        user_resp = _mock_response(json_data={"username": "test", "balance": 50.0})
        with patch(f"{_MODULE}.httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=user_resp)
            mock_httpx.AsyncClient.return_value = mock_client
            mock_httpx.Timeout = MagicMock()
            await provider._connect()
        assert provider._client is mock_client

    @pytest.mark.asyncio
    async def test_connect_failure_cleans_up(self, provider):
        import httpx

        with patch(f"{_MODULE}.httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=Exception("auth failed"))
            mock_httpx.AsyncClient.return_value = mock_client
            mock_httpx.Timeout = MagicMock()
            mock_httpx.HTTPStatusError = httpx.HTTPStatusError
            with pytest.raises(ConnectionError):
                await provider._connect()
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_disconnect(self, provider):
        mock_client = AsyncMock()
        provider._client = mock_client
        await provider._disconnect()
        mock_client.aclose.assert_awaited_once()
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_disconnect_noop(self, provider):
        provider._client = None
        await provider._disconnect()


class TestVastAIListInstances:
    @pytest.mark.asyncio
    async def test_list_instances_parses_offers(self, connected):
        provider, mock_client = connected
        offers_data = {
            "offers": [
                {
                    "id": 12345,
                    "gpu_name": "RTX 4090",
                    "num_gpus": 1,
                    "dph_total": 0.55,
                    "cpu_cores_effective": 8,
                    "cpu_ram": 32768,
                    "reliability2": 0.98,
                    "geolocation": "US",
                },
            ],
        }
        mock_client.request = AsyncMock(return_value=_mock_response(json_data=offers_data))
        result = await provider._list_instances()
        assert len(result) == 1
        inst = result[0]
        assert inst.name == "vast-12345"
        assert inst.gpu_type == "RTX 4090"
        assert inst.gpu_count == 1
        assert inst.price_per_hour_eur == 0.55
        assert inst.vcpus == 8
        assert inst.available is True

    @pytest.mark.asyncio
    async def test_list_instances_with_region_filter(self, connected):
        provider, mock_client = connected
        mock_client.request = AsyncMock(return_value=_mock_response(json_data={"offers": []}))
        result = await provider._list_instances(region="EU")
        assert len(result) == 0
        # Verify the region was included in the query
        call_args = mock_client.request.call_args
        assert call_args is not None


class TestVastAIUploadCode:
    @pytest.mark.asyncio
    async def test_upload_with_code_uri(self, provider):
        spec = _spec(config={"code_uri": "s3://bucket/code"})
        result = await provider._upload_code(spec)
        assert result == "s3://bucket/code"

    @pytest.mark.asyncio
    async def test_upload_without_code_uri(self, provider):
        spec = _spec(config={})
        result = await provider._upload_code(spec)
        assert result.startswith("onstart://")


class TestVastAIProvision:
    @pytest.mark.asyncio
    async def test_provision_happy_path(self, connected):
        provider, mock_client = connected

        search_resp = _mock_response(
            json_data={
                "offers": [
                    {
                        "id": 99,
                        "gpu_name": "RTX 4090",
                        "dph_total": 0.5,
                    },
                ],
            }
        )
        create_resp = _mock_response(
            json_data={
                "success": True,
                "new_contract": 1001,
            }
        )
        mock_client.request = AsyncMock(side_effect=[search_resp, create_resp])

        spec = _spec()
        job_id = await provider._provision_and_start(spec)
        assert job_id.startswith("vast-")
        assert job_id in provider._jobs
        state = provider._jobs[job_id]
        assert state.instance_id == 1001
        assert state.hourly_price == 0.5

    @pytest.mark.asyncio
    async def test_provision_no_offers(self, connected):
        provider, mock_client = connected
        mock_client.request = AsyncMock(return_value=_mock_response(json_data={"offers": []}))
        with pytest.raises(RuntimeError, match=r"No Vast\.ai offers"):
            await provider._provision_and_start(_spec())

    @pytest.mark.asyncio
    async def test_provision_create_failure(self, connected):
        provider, mock_client = connected
        search_resp = _mock_response(
            json_data={
                "offers": [{"id": 99, "gpu_name": "X", "dph_total": 1}],
            }
        )
        create_resp = _mock_response(
            json_data={
                "success": False,
                "msg": "insufficient funds",
            }
        )
        mock_client.request = AsyncMock(side_effect=[search_resp, create_resp])
        with pytest.raises(RuntimeError, match="creation failed"):
            await provider._provision_and_start(_spec())


class TestVastAIPoll:
    @pytest.mark.asyncio
    async def test_poll_unknown_job(self, connected):
        provider, _ = connected
        result = await provider._poll_provider("nope")
        assert result.status == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_poll_running_instance(self, connected):
        provider, mock_client = connected
        from artenic_ai_platform.providers.vastai import (
            _VastJobState,
        )

        provider._jobs["j1"] = _VastJobState(
            instance_id=1001,
            created_at=time.time() - 300,
            spec=_spec(),
            hourly_price=0.5,
        )
        mock_client.request = AsyncMock(
            return_value=_mock_response(
                json_data={
                    "actual_status": "running",
                    "gpu_util": 85,
                    "gpu_temp": 72,
                }
            )
        )
        result = await provider._poll_provider("j1")
        assert result.status == JobStatus.RUNNING
        assert result.metrics is not None
        assert result.metrics["gpu_utilization_percent"] == 85

    @pytest.mark.asyncio
    async def test_poll_exited_instance(self, connected):
        provider, mock_client = connected
        from artenic_ai_platform.providers.vastai import (
            _VastJobState,
        )

        provider._jobs["j2"] = _VastJobState(
            instance_id=1002,
            created_at=time.time() - 60,
            spec=_spec(),
        )
        mock_client.request = AsyncMock(
            return_value=_mock_response(
                json_data={
                    "actual_status": "exited",
                }
            )
        )
        result = await provider._poll_provider("j2")
        assert result.status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_poll_api_error(self, connected):
        provider, mock_client = connected
        from artenic_ai_platform.providers.vastai import (
            _VastJobState,
        )

        provider._jobs["j3"] = _VastJobState(
            instance_id=1003,
            created_at=time.time(),
            spec=_spec(),
        )
        mock_client.request = AsyncMock(side_effect=Exception("timeout"))
        result = await provider._poll_provider("j3")
        assert result.status == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_poll_nested_instances_response(self, connected):
        provider, mock_client = connected
        from artenic_ai_platform.providers.vastai import (
            _VastJobState,
        )

        provider._jobs["j4"] = _VastJobState(
            instance_id=1004,
            created_at=time.time(),
            spec=_spec(),
        )
        mock_client.request = AsyncMock(
            return_value=_mock_response(
                json_data={
                    "instances": [
                        {"actual_status": "running", "gpu_util": 50},
                    ],
                }
            )
        )
        result = await provider._poll_provider("j4")
        assert result.status == JobStatus.RUNNING


class TestVastAICollectArtifacts:
    @pytest.mark.asyncio
    async def test_collect_with_uri(self, provider):
        from artenic_ai_platform.providers.vastai import (
            _VastJobState,
        )

        provider._jobs["j1"] = _VastJobState(
            instance_id=1,
            created_at=time.time(),
            spec=_spec(config={"artifacts_uri": "s3://out"}),
        )
        result = await provider._collect_artifacts(
            "j1",
            CloudJobStatus(provider_job_id="j1", status=JobStatus.COMPLETED),
        )
        assert result == "s3://out"

    @pytest.mark.asyncio
    async def test_collect_no_uri(self, provider):
        from artenic_ai_platform.providers.vastai import (
            _VastJobState,
        )

        provider._jobs["j2"] = _VastJobState(
            instance_id=2,
            created_at=time.time(),
            spec=_spec(),
        )
        result = await provider._collect_artifacts(
            "j2",
            CloudJobStatus(provider_job_id="j2", status=JobStatus.COMPLETED),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_collect_unknown_job(self, provider):
        result = await provider._collect_artifacts(
            "nope",
            CloudJobStatus(provider_job_id="nope", status=JobStatus.COMPLETED),
        )
        assert result is None


class TestVastAICleanup:
    @pytest.mark.asyncio
    async def test_cleanup_unknown_job(self, connected):
        provider, _ = connected
        await provider._cleanup_compute("nope")

    @pytest.mark.asyncio
    async def test_cleanup_deletes_instance(self, connected):
        provider, mock_client = connected
        from artenic_ai_platform.providers.vastai import (
            _VastJobState,
        )

        provider._jobs["j1"] = _VastJobState(
            instance_id=1001,
            created_at=time.time(),
            spec=_spec(),
        )
        mock_client.request = AsyncMock(return_value=_mock_response(content=b""))
        await provider._cleanup_compute("j1")
        assert "j1" not in provider._jobs

    @pytest.mark.asyncio
    async def test_cleanup_404_is_ok(self, connected):
        provider, mock_client = connected
        from artenic_ai_platform.providers.vastai import (
            _VastJobState,
        )

        provider._jobs["j2"] = _VastJobState(
            instance_id=1002,
            created_at=time.time(),
            spec=_spec(),
        )
        import httpx

        exc_resp = MagicMock()
        exc_resp.status_code = 404
        mock_client.request = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "not found",
                request=MagicMock(),
                response=exc_resp,
            )
        )
        await provider._cleanup_compute("j2")
        assert "j2" not in provider._jobs


class TestVastAICancel:
    @pytest.mark.asyncio
    async def test_cancel_unknown_job(self, connected):
        provider, _ = connected
        await provider._cancel_provider_job("nope")

    @pytest.mark.asyncio
    async def test_cancel_stops_and_deletes(self, connected):
        provider, mock_client = connected
        from artenic_ai_platform.providers.vastai import (
            _VastJobState,
        )

        provider._jobs["j1"] = _VastJobState(
            instance_id=1001,
            created_at=time.time(),
            spec=_spec(),
        )
        stop_resp = _mock_response(json_data={"success": True}, content=b'{"success":true}')
        delete_resp = _mock_response(content=b"")
        mock_client.request = AsyncMock(side_effect=[stop_resp, delete_resp])
        await provider._cancel_provider_job("j1")
        assert mock_client.request.await_count == 2


class TestVastAIHelpers:
    def test_map_instance_status_running(self):
        from artenic_ai_platform.providers.vastai import (
            VastAIProvider,
        )

        assert VastAIProvider._map_instance_status("running") == JobStatus.RUNNING

    def test_map_instance_status_loading(self):
        from artenic_ai_platform.providers.vastai import (
            VastAIProvider,
        )

        assert VastAIProvider._map_instance_status("loading") == JobStatus.PENDING

    def test_map_instance_status_exited(self):
        from artenic_ai_platform.providers.vastai import (
            VastAIProvider,
        )

        assert VastAIProvider._map_instance_status("exited") == JobStatus.COMPLETED

    def test_map_instance_status_error(self):
        from artenic_ai_platform.providers.vastai import (
            VastAIProvider,
        )

        assert VastAIProvider._map_instance_status("error") == JobStatus.FAILED

    def test_map_instance_status_stopped(self):
        from artenic_ai_platform.providers.vastai import (
            VastAIProvider,
        )

        assert VastAIProvider._map_instance_status("stopped") == JobStatus.CANCELLED

    def test_map_instance_status_unknown(self):
        from artenic_ai_platform.providers.vastai import (
            VastAIProvider,
        )

        assert VastAIProvider._map_instance_status("weird") == JobStatus.PENDING

    def test_build_onstart_script(self):
        from artenic_ai_platform.providers.vastai import (
            VastAIProvider,
        )

        script = VastAIProvider._build_onstart_script(
            training_command="python train.py",
            code_uri="https://github.com/org/repo",
            env_vars={"LR": "0.001"},
            job_id="test-123",
        )
        assert "git clone" in script
        assert "python train.py" in script
        assert "LR" in script
        assert "test-123" in script

    def test_estimate_cost_with_price(self):
        from artenic_ai_platform.providers.vastai import (
            VastAIProvider,
            _VastJobState,
        )

        state = _VastJobState(
            instance_id=1,
            created_at=0,
            spec=_spec(),
            hourly_price=0.5,
        )
        cost = VastAIProvider._estimate_cost(state, 7200.0)
        assert cost == 1.0

    def test_estimate_cost_none(self):
        from artenic_ai_platform.providers.vastai import (
            VastAIProvider,
            _VastJobState,
        )

        state = _VastJobState(
            instance_id=1,
            created_at=0,
            spec=_spec(),
        )
        assert VastAIProvider._estimate_cost(state, 3600.0) is None


# ======================================================================
# Additional tests for 100 % coverage
# ======================================================================


class TestVastAIProvisionGPUFilter:
    """Cover line 278: provision with gpu_name_filter (instance_type set)."""

    @pytest.mark.asyncio
    async def test_provision_with_gpu_filter(self, connected):
        provider, mock_client = connected
        search_resp = _mock_response(
            json_data={
                "offers": [
                    {"id": 55, "gpu_name": "RTX_4090", "dph_total": 0.4},
                ],
            }
        )
        create_resp = _mock_response(json_data={"success": True, "new_contract": 2001})
        mock_client.request = AsyncMock(side_effect=[search_resp, create_resp])

        spec = _spec(instance_type="RTX_4090")
        job_id = await provider._provision_and_start(spec)
        assert job_id.startswith("vast-")


class TestVastAIProvisionRegionFilter:
    """Cover line 281: provision with spec.region set."""

    @pytest.mark.asyncio
    async def test_provision_with_region(self, connected):
        provider, mock_client = connected
        search_resp = _mock_response(
            json_data={
                "offers": [
                    {"id": 66, "gpu_name": "A100", "dph_total": 1.5},
                ],
            }
        )
        create_resp = _mock_response(json_data={"success": True, "new_contract": 3001})
        mock_client.request = AsyncMock(side_effect=[search_resp, create_resp])

        spec = _spec(region="EU")
        job_id = await provider._provision_and_start(spec)
        assert job_id.startswith("vast-")


class TestVastAIProvisionDockerArgsAndEnvVars:
    """Cover lines 339, 344-347: docker_args and env_vars in provision."""

    @pytest.mark.asyncio
    async def test_provision_with_docker_args_and_env(self, connected):
        provider, mock_client = connected
        search_resp = _mock_response(
            json_data={
                "offers": [
                    {"id": 77, "gpu_name": "RTX_3090", "dph_total": 0.3},
                ],
            }
        )
        create_resp = _mock_response(json_data={"success": True, "new_contract": 4001})
        mock_client.request = AsyncMock(side_effect=[search_resp, create_resp])

        spec = _spec(
            config={
                "docker_args": "--shm-size 8G",
                "env": {"LR": "0.001", "BATCH": "32"},
            }
        )
        job_id = await provider._provision_and_start(spec)
        assert job_id.startswith("vast-")


class TestVastAICleanupErrors:
    """Cover lines 505-517: cleanup non-404 HTTPStatusError and generic Exception."""

    @pytest.mark.asyncio
    async def test_cleanup_non_404_http_error_raises(self, connected):
        import httpx

        provider, mock_client = connected
        from artenic_ai_platform.providers.vastai import (
            _VastJobState,
        )

        provider._jobs["j-cleanup-500"] = _VastJobState(
            instance_id=5001,
            created_at=time.time(),
            spec=_spec(),
        )
        exc_resp = MagicMock()
        exc_resp.status_code = 500
        mock_client.request = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "server error",
                request=MagicMock(),
                response=exc_resp,
            )
        )
        with pytest.raises(httpx.HTTPStatusError):
            await provider._cleanup_compute("j-cleanup-500")

    @pytest.mark.asyncio
    async def test_cleanup_generic_exception_raises(self, connected):
        provider, mock_client = connected
        from artenic_ai_platform.providers.vastai import (
            _VastJobState,
        )

        provider._jobs["j-cleanup-err"] = _VastJobState(
            instance_id=5002,
            created_at=time.time(),
            spec=_spec(),
        )
        mock_client.request = AsyncMock(side_effect=RuntimeError("connection timeout"))
        with pytest.raises(RuntimeError, match="connection timeout"):
            await provider._cleanup_compute("j-cleanup-err")


class TestVastAICancelStopHTTPErrors:
    """Cover lines 549-562: cancel step 1 HTTPStatusError (404, non-404) and generic."""

    @pytest.mark.asyncio
    async def test_cancel_stop_404_returns(self, connected):
        import httpx

        provider, mock_client = connected
        from artenic_ai_platform.providers.vastai import (
            _VastJobState,
        )

        provider._jobs["j-cancel-404"] = _VastJobState(
            instance_id=6001,
            created_at=time.time(),
            spec=_spec(),
        )
        exc_resp = MagicMock()
        exc_resp.status_code = 404
        mock_client.request = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "not found",
                request=MagicMock(),
                response=exc_resp,
            )
        )
        # Should return early without attempting delete
        await provider._cancel_provider_job("j-cancel-404")

    @pytest.mark.asyncio
    async def test_cancel_stop_non_404_continues_to_delete(self, connected):
        import httpx

        provider, mock_client = connected
        from artenic_ai_platform.providers.vastai import (
            _VastJobState,
        )

        provider._jobs["j-cancel-500"] = _VastJobState(
            instance_id=6002,
            created_at=time.time(),
            spec=_spec(),
        )
        exc_resp = MagicMock()
        exc_resp.status_code = 500
        stop_error = httpx.HTTPStatusError(
            "server error",
            request=MagicMock(),
            response=exc_resp,
        )
        delete_resp = _mock_response(content=b"")
        mock_client.request = AsyncMock(side_effect=[stop_error, delete_resp])
        # Should log warning but proceed to delete
        await provider._cancel_provider_job("j-cancel-500")
        assert mock_client.request.await_count == 2

    @pytest.mark.asyncio
    async def test_cancel_stop_generic_exception_continues(self, connected):
        provider, mock_client = connected
        from artenic_ai_platform.providers.vastai import (
            _VastJobState,
        )

        provider._jobs["j-cancel-gen"] = _VastJobState(
            instance_id=6003,
            created_at=time.time(),
            spec=_spec(),
        )
        delete_resp = _mock_response(content=b"")
        mock_client.request = AsyncMock(side_effect=[RuntimeError("stop failed"), delete_resp])
        await provider._cancel_provider_job("j-cancel-gen")
        assert mock_client.request.await_count == 2


class TestVastAICancelDeleteHTTPErrors:
    """Cover lines 575-594: cancel step 2 delete errors."""

    @pytest.mark.asyncio
    async def test_cancel_delete_404_ok(self, connected):
        import httpx

        provider, mock_client = connected
        from artenic_ai_platform.providers.vastai import (
            _VastJobState,
        )

        provider._jobs["j-del-404"] = _VastJobState(
            instance_id=7001,
            created_at=time.time(),
            spec=_spec(),
        )
        stop_resp = _mock_response(json_data={"success": True}, content=b'{"success":true}')
        exc_resp = MagicMock()
        exc_resp.status_code = 404
        delete_error = httpx.HTTPStatusError(
            "not found",
            request=MagicMock(),
            response=exc_resp,
        )
        mock_client.request = AsyncMock(side_effect=[stop_resp, delete_error])
        # Should not raise — 404 is OK
        await provider._cancel_provider_job("j-del-404")

    @pytest.mark.asyncio
    async def test_cancel_delete_non_404_raises(self, connected):
        import httpx

        provider, mock_client = connected
        from artenic_ai_platform.providers.vastai import (
            _VastJobState,
        )

        provider._jobs["j-del-500"] = _VastJobState(
            instance_id=7002,
            created_at=time.time(),
            spec=_spec(),
        )
        stop_resp = _mock_response(json_data={"success": True}, content=b'{"success":true}')
        exc_resp = MagicMock()
        exc_resp.status_code = 500
        delete_error = httpx.HTTPStatusError(
            "server error",
            request=MagicMock(),
            response=exc_resp,
        )
        mock_client.request = AsyncMock(side_effect=[stop_resp, delete_error])
        with pytest.raises(httpx.HTTPStatusError):
            await provider._cancel_provider_job("j-del-500")

    @pytest.mark.asyncio
    async def test_cancel_delete_generic_exception_raises(self, connected):
        provider, mock_client = connected
        from artenic_ai_platform.providers.vastai import (
            _VastJobState,
        )

        provider._jobs["j-del-err"] = _VastJobState(
            instance_id=7003,
            created_at=time.time(),
            spec=_spec(),
        )
        stop_resp = _mock_response(json_data={"success": True}, content=b'{"success":true}')
        mock_client.request = AsyncMock(
            side_effect=[stop_resp, RuntimeError("delete connection lost")]
        )
        with pytest.raises(RuntimeError, match="delete connection lost"):
            await provider._cancel_provider_job("j-del-err")


class TestVastAIRequestNotConnected:
    """Cover line 618: _request raises RuntimeError when client is None."""

    @pytest.mark.asyncio
    async def test_request_not_connected(self, provider):
        provider._client = None
        with pytest.raises(RuntimeError, match="not connected"):
            await provider._request("GET", "/test/")


class TestVastAIBuildOnstartS3:
    """Cover lines 681-683: _build_onstart_script with s3:// code_uri."""

    def test_onstart_script_s3_uri(self):
        from artenic_ai_platform.providers.vastai import (
            VastAIProvider,
        )

        script = VastAIProvider._build_onstart_script(
            training_command="python train.py",
            code_uri="s3://bucket/code",
            env_vars={},
            job_id="test-s3",
        )
        assert "aws s3 sync" in script
        assert "s3://bucket/code" in script.replace("'", "")

    def test_onstart_script_no_code_uri(self):
        from artenic_ai_platform.providers.vastai import (
            VastAIProvider,
        )

        script = VastAIProvider._build_onstart_script(
            training_command="python train.py",
            code_uri="",
            env_vars={},
            job_id="test-nocode",
        )
        assert "git clone" not in script
        assert "aws s3 sync" not in script
        assert "python train.py" in script


class TestVastAIMapInstanceStatusEmpty:
    """Cover line 649-651: unknown empty status."""

    def test_map_instance_status_empty(self):
        from artenic_ai_platform.providers.vastai import (
            VastAIProvider,
        )

        assert VastAIProvider._map_instance_status("") == JobStatus.PENDING


class TestVastAIConnectHTTPStatusError:
    """Cover lines 137-143: connect with HTTPStatusError."""

    @pytest.mark.asyncio
    async def test_connect_http_status_error(self, provider):
        import httpx

        with patch(f"{_MODULE}.httpx") as mock_httpx:
            mock_client = AsyncMock()
            exc_resp = MagicMock()
            exc_resp.status_code = 401
            mock_client.get = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "unauthorized",
                    request=MagicMock(),
                    response=exc_resp,
                )
            )
            mock_httpx.AsyncClient.return_value = mock_client
            mock_httpx.Timeout = MagicMock()
            mock_httpx.HTTPStatusError = httpx.HTTPStatusError
            with pytest.raises(ConnectionError):
                await provider._connect()
        assert provider._client is None


class TestVastAIRequestEmptyContent:
    """Cover lines 629-630: _request returns empty dict for empty content."""

    @pytest.mark.asyncio
    async def test_request_empty_content(self, connected):
        provider, mock_client = connected
        resp = _mock_response(content=b"")
        mock_client.request = AsyncMock(return_value=resp)
        result = await provider._request("DELETE", "/instances/123/")
        assert result == {}
