"""Tests for artenic_ai_platform.providers.runpod — RunPodProvider."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from artenic_ai_platform.providers.base import (
    CloudJobStatus,
    JobStatus,
    TrainingSpec,
)

_MODULE = "artenic_ai_platform.providers.runpod"


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _spec(**overrides) -> TrainingSpec:
    defaults = {
        "service": "training",
        "model": "llama-7b",
        "provider": "runpod",
        "config": {},
    }
    defaults.update(overrides)
    return TrainingSpec(**defaults)


def _gql_response(data: dict | None = None, errors: list | None = None):
    """Build a dict that looks like a GraphQL JSON response."""
    resp: dict = {}
    if data is not None:
        resp["data"] = data
    if errors is not None:
        resp["errors"] = errors
    return resp


def _mock_http_response(*, status_code: int = 200, json_data: dict | None = None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture()
def provider():
    from artenic_ai_platform.providers.runpod import RunPodProvider

    return RunPodProvider(api_key="rp-test-key")


@pytest.fixture()
def connected_provider(provider):
    """Provider with a mock httpx client already wired up."""
    mock_client = AsyncMock()
    provider._client = mock_client
    provider._connected = True
    return provider, mock_client


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------


class TestRunPodInit:
    def test_provider_name(self, provider):
        assert provider.provider_name == "runpod"

    def test_init_stores_fields(self, provider):
        assert provider._api_key == "rp-test-key"
        assert provider._gpu_type == "NVIDIA RTX A6000"
        assert provider._docker_image == "runpod/pytorch:latest"
        assert provider._cloud_type == "ALL"
        assert provider._connected is False

    def test_empty_api_key_raises(self):
        from artenic_ai_platform.providers.runpod import RunPodProvider

        with pytest.raises(ValueError, match="must not be empty"):
            RunPodProvider(api_key="")

    def test_invalid_cloud_type_raises(self):
        from artenic_ai_platform.providers.runpod import RunPodProvider

        with pytest.raises(ValueError, match="cloud_type"):
            RunPodProvider(api_key="key", cloud_type="INVALID")


class TestRunPodConnect:
    @pytest.mark.asyncio
    async def test_connect_creates_client(self, provider):
        gpu_resp = _gql_response(data={"gpuTypes": [{"id": "a6000"}]})
        with patch(f"{_MODULE}.httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=_mock_http_response(json_data=gpu_resp))
            mock_httpx.AsyncClient.return_value = mock_client
            mock_httpx.Timeout = MagicMock()
            await provider._connect()

        assert provider._client is mock_client

    @pytest.mark.asyncio
    async def test_connect_failure_cleans_up(self, provider):
        with patch(f"{_MODULE}.httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=Exception("boom"))
            mock_httpx.AsyncClient.return_value = mock_client
            mock_httpx.Timeout = MagicMock()
            with pytest.raises(ConnectionError):
                await provider._connect()
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_disconnect_closes_client(self, provider):
        mock_client = AsyncMock()
        provider._client = mock_client
        await provider._disconnect()
        mock_client.aclose.assert_awaited_once()
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_disconnect_noop_when_none(self, provider):
        provider._client = None
        await provider._disconnect()


class TestRunPodListInstances:
    @pytest.mark.asyncio
    async def test_list_instances_parses_gpu_types(self, connected_provider):
        provider, mock_client = connected_provider
        gpu_data = _gql_response(
            data={
                "gpuTypes": [
                    {
                        "id": "NVIDIA RTX A6000",
                        "displayName": "RTX A6000",
                        "memoryInGb": 48,
                        "secureCloud": True,
                        "communityCloud": True,
                        "lowestPrice": {
                            "uninterruptablePrice": 0.79,
                            "minimumBidPrice": 0.30,
                        },
                    },
                ],
            }
        )
        mock_client.post = AsyncMock(return_value=_mock_http_response(json_data=gpu_data))
        result = await provider._list_instances()
        assert len(result) == 1
        assert result[0].gpu_type == "RTX A6000"
        assert result[0].memory_gb == 48.0
        assert result[0].price_per_hour_eur == 0.79
        assert result[0].spot_price_per_hour_eur == 0.30

    @pytest.mark.asyncio
    async def test_list_instances_filters_by_cloud_type(self, connected_provider):
        provider, mock_client = connected_provider
        provider._cloud_type = "SECURE"
        gpu_data = _gql_response(
            data={
                "gpuTypes": [
                    {
                        "id": "gpu-1",
                        "displayName": "GPU-1",
                        "memoryInGb": 24,
                        "secureCloud": False,
                        "communityCloud": True,
                        "lowestPrice": {
                            "uninterruptablePrice": 0.5,
                            "minimumBidPrice": 0,
                        },
                    },
                ],
            }
        )
        mock_client.post = AsyncMock(return_value=_mock_http_response(json_data=gpu_data))
        result = await provider._list_instances()
        assert len(result) == 0


class TestRunPodUploadCode:
    @pytest.mark.asyncio
    async def test_upload_with_code_uri(self, provider):
        spec = _spec(config={"code_uri": "s3://bucket/code"})
        result = await provider._upload_code(spec)
        assert result == "s3://bucket/code"

    @pytest.mark.asyncio
    async def test_upload_without_code_uri(self, provider):
        spec = _spec(config={})
        result = await provider._upload_code(spec)
        assert result.startswith("docker://")


class TestRunPodProvision:
    @pytest.mark.asyncio
    async def test_provision_happy_path(self, connected_provider):
        provider, mock_client = connected_provider
        deploy_resp = _gql_response(
            data={
                "podFindAndDeployOnDemand": {
                    "id": "pod-abc",
                    "name": "artenic-test",
                    "desiredStatus": "RUNNING",
                },
            }
        )
        mock_client.post = AsyncMock(return_value=_mock_http_response(json_data=deploy_resp))
        spec = _spec()
        job_id = await provider._provision_and_start(spec)
        assert job_id.startswith("runpod-")
        assert job_id in provider._jobs
        assert provider._jobs[job_id].pod_id == "pod-abc"

    @pytest.mark.asyncio
    async def test_provision_failure(self, connected_provider):
        provider, mock_client = connected_provider
        fail_resp = _gql_response(
            data={"podFindAndDeployOnDemand": None},
            errors=[{"message": "No GPU available"}],
        )
        mock_client.post = AsyncMock(return_value=_mock_http_response(json_data=fail_resp))
        with pytest.raises(RuntimeError, match="deployment failed"):
            await provider._provision_and_start(_spec())


class TestRunPodPoll:
    @pytest.mark.asyncio
    async def test_poll_unknown_job(self, connected_provider):
        provider, _ = connected_provider
        result = await provider._poll_provider("nope")
        assert result.status == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_poll_running_pod(self, connected_provider):
        provider, mock_client = connected_provider
        from artenic_ai_platform.providers.runpod import (
            _RunPodJobState,
        )

        provider._jobs["j1"] = _RunPodJobState(
            pod_id="pod-1",
            created_at=time.time() - 120,
            spec=_spec(),
        )
        pod_resp = _gql_response(
            data={
                "pod": {
                    "id": "pod-1",
                    "desiredStatus": "RUNNING",
                    "runtime": {
                        "uptimeInSeconds": 120,
                        "gpus": [{"id": "0", "gpuUtilPercent": 80}],
                    },
                },
            }
        )
        mock_client.post = AsyncMock(return_value=_mock_http_response(json_data=pod_resp))
        result = await provider._poll_provider("j1")
        assert result.status == JobStatus.RUNNING
        assert result.metrics is not None
        assert result.metrics["gpus"][0]["gpuUtilPercent"] == 80

    @pytest.mark.asyncio
    async def test_poll_terminated_pod(self, connected_provider):
        provider, mock_client = connected_provider
        from artenic_ai_platform.providers.runpod import (
            _RunPodJobState,
        )

        provider._jobs["j2"] = _RunPodJobState(
            pod_id="pod-2",
            created_at=time.time() - 60,
            spec=_spec(),
        )
        pod_resp = _gql_response(
            data={
                "pod": {
                    "id": "pod-2",
                    "desiredStatus": "TERMINATED",
                    "runtime": None,
                },
            }
        )
        mock_client.post = AsyncMock(return_value=_mock_http_response(json_data=pod_resp))
        result = await provider._poll_provider("j2")
        assert result.status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_poll_api_error(self, connected_provider):
        provider, mock_client = connected_provider
        from artenic_ai_platform.providers.runpod import (
            _RunPodJobState,
        )

        provider._jobs["j3"] = _RunPodJobState(
            pod_id="pod-3",
            created_at=time.time(),
            spec=_spec(),
        )
        mock_client.post = AsyncMock(side_effect=Exception("network"))
        result = await provider._poll_provider("j3")
        assert result.status == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_poll_pod_gone(self, connected_provider):
        provider, mock_client = connected_provider
        from artenic_ai_platform.providers.runpod import (
            _RunPodJobState,
        )

        provider._jobs["j4"] = _RunPodJobState(
            pod_id="pod-4",
            created_at=time.time(),
            spec=_spec(),
        )
        gone_resp = _gql_response(data={"pod": None})
        mock_client.post = AsyncMock(return_value=_mock_http_response(json_data=gone_resp))
        result = await provider._poll_provider("j4")
        assert result.status == JobStatus.FAILED
        assert "no longer exists" in (result.error or "")


class TestRunPodCollectArtifacts:
    @pytest.mark.asyncio
    async def test_collect_with_uri(self, provider):
        from artenic_ai_platform.providers.runpod import (
            _RunPodJobState,
        )

        provider._jobs["j1"] = _RunPodJobState(
            pod_id="pod-1",
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
        from artenic_ai_platform.providers.runpod import (
            _RunPodJobState,
        )

        provider._jobs["j2"] = _RunPodJobState(
            pod_id="pod-2",
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


class TestRunPodCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_unknown_job(self, connected_provider):
        provider, _ = connected_provider
        await provider._cleanup_compute("nope")

    @pytest.mark.asyncio
    async def test_cleanup_terminates_pod(self, connected_provider):
        provider, mock_client = connected_provider
        from artenic_ai_platform.providers.runpod import (
            _RunPodJobState,
        )

        provider._jobs["j1"] = _RunPodJobState(
            pod_id="pod-1",
            created_at=time.time(),
            spec=_spec(),
        )
        term_resp = _gql_response(data={"podTerminate": None})
        mock_client.post = AsyncMock(return_value=_mock_http_response(json_data=term_resp))
        await provider._cleanup_compute("j1")
        assert "j1" not in provider._jobs

    @pytest.mark.asyncio
    async def test_cleanup_graphql_error_raises(self, connected_provider):
        provider, mock_client = connected_provider
        from artenic_ai_platform.providers.runpod import (
            _RunPodJobState,
        )

        provider._jobs["j2"] = _RunPodJobState(
            pod_id="pod-2",
            created_at=time.time(),
            spec=_spec(),
        )
        err_resp = _gql_response(
            data={"podTerminate": None},
            errors=[{"message": "not found"}],
        )
        mock_client.post = AsyncMock(return_value=_mock_http_response(json_data=err_resp))
        with pytest.raises(RuntimeError, match="Failed to terminate"):
            await provider._cleanup_compute("j2")


class TestRunPodCancel:
    @pytest.mark.asyncio
    async def test_cancel_unknown_job(self, connected_provider):
        provider, _ = connected_provider
        await provider._cancel_provider_job("nope")

    @pytest.mark.asyncio
    async def test_cancel_terminates_pod(self, connected_provider):
        provider, mock_client = connected_provider
        from artenic_ai_platform.providers.runpod import (
            _RunPodJobState,
        )

        provider._jobs["j1"] = _RunPodJobState(
            pod_id="pod-1",
            created_at=time.time(),
            spec=_spec(),
        )
        term_resp = _gql_response(data={"podTerminate": None})
        mock_client.post = AsyncMock(return_value=_mock_http_response(json_data=term_resp))
        await provider._cancel_provider_job("j1")
        mock_client.post.assert_awaited_once()


class TestRunPodHelpers:
    def test_map_pod_status_running_with_runtime(self):
        from artenic_ai_platform.providers.runpod import (
            RunPodProvider,
        )

        assert (
            RunPodProvider._map_pod_status("RUNNING", {"uptimeInSeconds": 10}) == JobStatus.RUNNING
        )

    def test_map_pod_status_running_no_runtime(self):
        from artenic_ai_platform.providers.runpod import (
            RunPodProvider,
        )

        assert RunPodProvider._map_pod_status("RUNNING", None) == JobStatus.PENDING

    def test_map_pod_status_exited(self):
        from artenic_ai_platform.providers.runpod import (
            RunPodProvider,
        )

        assert RunPodProvider._map_pod_status("EXITED", None) == JobStatus.COMPLETED

    def test_map_pod_status_error(self):
        from artenic_ai_platform.providers.runpod import (
            RunPodProvider,
        )

        assert RunPodProvider._map_pod_status("ERROR", None) == JobStatus.FAILED

    def test_build_docker_args_with_code_uri(self):
        from artenic_ai_platform.providers.runpod import (
            RunPodProvider,
        )

        args = RunPodProvider._build_docker_args(
            training_command="python train.py",
            code_uri="https://github.com/org/repo",
            env_vars={"LR": "0.001"},
            job_id="test-123",
        )
        assert "git clone" in args
        assert "python train.py" in args
        assert "LR" in args

    def test_build_env_list(self):
        from artenic_ai_platform.providers.runpod import (
            RunPodProvider,
        )

        result = RunPodProvider._build_env_list({"A": "1", "B": "2"})
        assert len(result) == 2
        assert {"key": "A", "value": "1"} in result

    def test_estimate_cost_with_price(self):
        from artenic_ai_platform.providers.runpod import (
            RunPodProvider,
            _RunPodJobState,
        )

        state = _RunPodJobState(
            pod_id="p",
            created_at=0,
            spec=_spec(),
            hourly_price=1.0,
        )
        cost = RunPodProvider._estimate_cost(state, 3600.0)
        assert cost == 1.0

    def test_estimate_cost_none_without_price(self):
        from artenic_ai_platform.providers.runpod import (
            RunPodProvider,
            _RunPodJobState,
        )

        state = _RunPodJobState(
            pod_id="p",
            created_at=0,
            spec=_spec(),
        )
        assert RunPodProvider._estimate_cost(state, 3600.0) is None


# ======================================================================
# Additional tests for 100 % coverage
# ======================================================================


class TestRunPodListInstancesCommunityFilter:
    """Cover line 229: COMMUNITY cloud_type filter."""

    @pytest.mark.asyncio
    async def test_list_instances_community_filter(self, connected_provider):
        provider, mock_client = connected_provider
        provider._cloud_type = "COMMUNITY"
        gpu_data = _gql_response(
            data={
                "gpuTypes": [
                    {
                        "id": "gpu-secure-only",
                        "displayName": "GPU Secure Only",
                        "memoryInGb": 24,
                        "secureCloud": True,
                        "communityCloud": False,
                        "lowestPrice": {
                            "uninterruptablePrice": 0.5,
                            "minimumBidPrice": 0,
                        },
                    },
                    {
                        "id": "gpu-community",
                        "displayName": "GPU Community",
                        "memoryInGb": 16,
                        "secureCloud": False,
                        "communityCloud": True,
                        "lowestPrice": {
                            "uninterruptablePrice": 0.3,
                            "minimumBidPrice": 0,
                        },
                    },
                ],
            }
        )
        mock_client.post = AsyncMock(return_value=_mock_http_response(json_data=gpu_data))
        result = await provider._list_instances()
        # Only the community GPU should pass
        assert len(result) == 1
        assert result[0].name == "gpu-community"


class TestRunPodProvisionWithAllowedIds:
    """Cover line 328: allowed_gpu_ids in spec config."""

    @pytest.mark.asyncio
    async def test_provision_with_allowed_gpu_ids(self, connected_provider):
        provider, mock_client = connected_provider
        deploy_resp = _gql_response(
            data={
                "podFindAndDeployOnDemand": {
                    "id": "pod-allowed",
                    "name": "artenic-test",
                    "desiredStatus": "RUNNING",
                },
            }
        )
        mock_client.post = AsyncMock(return_value=_mock_http_response(json_data=deploy_resp))
        spec = _spec(config={"allowed_gpu_ids": ["gpu-1", "gpu-2"]})
        job_id = await provider._provision_and_start(spec)
        assert job_id.startswith("runpod-")


class TestRunPodProvisionWithEnvVars:
    """Cover line 333: env vars passed to deploy."""

    @pytest.mark.asyncio
    async def test_provision_with_env_vars(self, connected_provider):
        provider, mock_client = connected_provider
        deploy_resp = _gql_response(
            data={
                "podFindAndDeployOnDemand": {
                    "id": "pod-env",
                    "name": "artenic-env",
                    "desiredStatus": "RUNNING",
                },
            }
        )
        mock_client.post = AsyncMock(return_value=_mock_http_response(json_data=deploy_resp))
        spec = _spec(config={"env": {"LR": "0.001", "BATCH": "32"}})
        job_id = await provider._provision_and_start(spec)
        assert job_id.startswith("runpod-")


class TestRunPodCleanupHTTPError:
    """Cover lines 502-507: httpx.HTTPError during pod termination."""

    @pytest.mark.asyncio
    async def test_cleanup_http_error_raises(self, connected_provider):
        import httpx

        provider, mock_client = connected_provider
        from artenic_ai_platform.providers.runpod import (
            _RunPodJobState,
        )

        provider._jobs["j-http-err"] = _RunPodJobState(
            pod_id="pod-http-err",
            created_at=time.time(),
            spec=_spec(),
        )
        mock_client.post = AsyncMock(side_effect=httpx.HTTPError("connection failed"))
        with pytest.raises(httpx.HTTPError):
            await provider._cleanup_compute("j-http-err")


class TestRunPodCancelException:
    """Cover lines 536-542: exception during cancel terminate."""

    @pytest.mark.asyncio
    async def test_cancel_exception_raises(self, connected_provider):
        provider, mock_client = connected_provider
        from artenic_ai_platform.providers.runpod import (
            _RunPodJobState,
        )

        provider._jobs["j-cancel-err"] = _RunPodJobState(
            pod_id="pod-cancel-err",
            created_at=time.time(),
            spec=_spec(),
        )
        mock_client.post = AsyncMock(side_effect=RuntimeError("API down"))
        with pytest.raises(RuntimeError, match="API down"):
            await provider._cancel_provider_job("j-cancel-err")


class TestRunPodGqlNotConnected:
    """Cover line 559: _gql raises RuntimeError when client is None."""

    @pytest.mark.asyncio
    async def test_gql_not_connected(self, provider):
        provider._client = None
        with pytest.raises(RuntimeError, match="not connected"):
            await provider._gql({"query": "test"})


class TestRunPodMapPodStatusEdgeCases:
    """Cover lines 591-596: STOPPED status and unknown with runtime."""

    def test_map_pod_status_stopped(self):
        from artenic_ai_platform.providers.runpod import (
            RunPodProvider,
        )

        assert RunPodProvider._map_pod_status("STOPPED", None) == JobStatus.CANCELLED

    def test_map_pod_status_unknown_with_runtime(self):
        from artenic_ai_platform.providers.runpod import (
            RunPodProvider,
        )

        assert (
            RunPodProvider._map_pod_status("UNKNOWN", {"uptimeInSeconds": 10}) == JobStatus.RUNNING
        )

    def test_map_pod_status_unknown_no_runtime(self):
        from artenic_ai_platform.providers.runpod import (
            RunPodProvider,
        )

        assert RunPodProvider._map_pod_status("UNKNOWN", None) == JobStatus.PENDING


class TestRunPodBuildDockerArgsS3:
    """Cover lines 625-627: _build_docker_args with s3:// code_uri."""

    def test_build_docker_args_s3_uri(self):
        from artenic_ai_platform.providers.runpod import (
            RunPodProvider,
        )

        args = RunPodProvider._build_docker_args(
            training_command="python train.py",
            code_uri="s3://bucket/code",
            env_vars={},
            job_id="test-s3",
        )
        assert "aws s3 sync" in args
        assert "s3://bucket/code" in args.replace("'", "")

    def test_build_docker_args_no_code_uri(self):
        from artenic_ai_platform.providers.runpod import (
            RunPodProvider,
        )

        args = RunPodProvider._build_docker_args(
            training_command="python train.py",
            code_uri="",
            env_vars={},
            job_id="test-nocodeuri",
        )
        assert "git clone" not in args
        assert "aws s3 sync" not in args
        assert "python train.py" in args
