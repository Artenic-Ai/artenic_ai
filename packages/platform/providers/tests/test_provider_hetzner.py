"""Tests for artenic_ai_platform_providers.hetzner -- HetznerProvider."""

from __future__ import annotations

import asyncio
import subprocess
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from artenic_ai_platform_providers.base import (
    CloudJobStatus,
    JobStatus,
    TrainingSpec,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(**overrides) -> TrainingSpec:
    defaults = {
        "service": "nlp",
        "model": "bert",
        "provider": "hetzner",
        "config": {},
    }
    defaults.update(overrides)
    return TrainingSpec(**defaults)


def _mock_server_type(
    name="cx22",
    cores=2,
    memory=4.0,
    description="CX22",
    prices=None,
    architecture="x86",
):
    st = MagicMock()
    st.name = name
    st.cores = cores
    st.memory = memory
    st.description = description
    st.architecture = architecture
    if prices is None:
        prices = [
            {
                "location": "fsn1",
                "price_hourly": {"net": "0.006"},
            }
        ]
    st.prices = prices
    return st


def _mock_server(
    server_id=42,
    name="artenic-hetzner-abc",
    status="running",
    ipv4_ip="1.2.3.4",
):
    srv = MagicMock()
    srv.id = server_id
    srv.name = name
    srv.status = status
    srv.public_net = MagicMock()
    srv.public_net.ipv4 = MagicMock()
    srv.public_net.ipv4.ip = ipv4_ip
    return srv


def _mock_create_response(server=None):
    resp = MagicMock()
    resp.server = server or _mock_server()
    return resp


@pytest.fixture(autouse=True)
def _mock_to_thread(monkeypatch):
    """Replace ``asyncio.to_thread`` with a synchronous call wrapper.

    This lets us test async provider methods without actually spawning
    threads.  The mock is an ``AsyncMock`` whose ``side_effect`` simply
    invokes the target function directly.
    """
    mock = AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw))
    monkeypatch.setattr(asyncio, "to_thread", mock)
    return mock


@pytest.fixture(autouse=True)
def _patch_hcloud(monkeypatch):
    """Patch the hcloud SDK references on the hetzner module."""
    mock_hcloud = MagicMock()
    mock_client = MagicMock()
    mock_hcloud.Client.return_value = mock_client
    mock_client.datacenters.get_all.return_value = [MagicMock()]

    import artenic_ai_platform_providers.hetzner as hetzner_mod

    monkeypatch.setattr(hetzner_mod, "_HCLOUD_AVAILABLE", True)
    monkeypatch.setattr(hetzner_mod, "hcloud", mock_hcloud)


def _build_provider(**kwargs):
    from artenic_ai_platform_providers.hetzner import HetznerProvider

    defaults = {
        "api_token": "tok-123",
        "location": "fsn1",
    }
    defaults.update(kwargs)
    return HetznerProvider(**defaults)


# ===================================================================
# Tests
# ===================================================================


class TestHetznerInit:
    def test_init_stores_params(self):
        p = _build_provider()
        assert p._api_token == "tok-123"
        assert p._location == "fsn1"
        assert p._client is None

    def test_init_raises_without_hcloud(self, monkeypatch):
        import artenic_ai_platform_providers.hetzner as mod

        monkeypatch.setattr(mod, "_HCLOUD_AVAILABLE", False)
        with pytest.raises(ImportError, match="hcloud"):
            _build_provider()

    def test_s3_configured_false_by_default(self):
        p = _build_provider()
        assert p._s3_configured is False

    def test_s3_configured_true_when_all_set(self):
        p = _build_provider(
            s3_endpoint="https://s3.example.com",
            s3_access_key="ak",
            s3_secret_key="sk",
            s3_bucket="bkt",
        )
        assert p._s3_configured is True


class TestHetznerProviderName:
    def test_provider_name(self):
        p = _build_provider()
        assert p.provider_name == "hetzner"


class TestHetznerConnect:
    async def test_connect_creates_client(self):
        p = _build_provider()
        await p._connect()
        assert p._client is not None


class TestHetznerDisconnect:
    async def test_disconnect_clears_client(self):
        p = _build_provider()
        await p._connect()
        await p._disconnect()
        assert p._client is None


class TestHetznerListInstances:
    async def test_list_returns_instances(self):
        p = _build_provider()
        await p._connect()

        p._client.server_types.get_all.return_value = [_mock_server_type()]
        instances = await p._list_instances()
        assert len(instances) == 1
        assert instances[0].name == "cx22"
        assert instances[0].vcpus == 2
        assert instances[0].memory_gb == 4.0
        assert instances[0].price_per_hour_eur == 0.006

    async def test_list_gpu_only(self):
        p = _build_provider()
        await p._connect()

        gpu_st = _mock_server_type(
            name="gpu-a100-x2",
            cores=16,
            memory=128.0,
            description="GPU A100 x2",
        )
        p._client.server_types.get_all.return_value = [
            _mock_server_type(),
            gpu_st,
        ]

        instances = await p._list_instances(gpu_only=True)
        assert len(instances) == 1
        assert instances[0].gpu_count >= 1
        assert instances[0].gpu_type == "A100"

    async def test_list_fallback_pricing(self):
        p = _build_provider()
        await p._connect()

        st = _mock_server_type(
            prices=[
                {
                    "location": "nbg1",
                    "price_hourly": {"net": "0.01"},
                }
            ]
        )
        p._client.server_types.get_all.return_value = [st]

        instances = await p._list_instances()
        assert instances[0].price_per_hour_eur == 0.01


class TestHetznerUploadCode:
    async def test_upload_no_s3_returns_local(self):
        p = _build_provider()
        await p._connect()
        spec = _make_spec(config={"code_path": "/tmp/code"})

        uri = await p._upload_code(spec)
        assert uri.startswith("local://")

    async def test_upload_with_s3(self):
        p = _build_provider(
            s3_endpoint="https://s3.example.com",
            s3_access_key="ak",
            s3_secret_key="sk",
            s3_bucket="bkt",
        )
        await p._connect()
        spec = _make_spec(config={"code_path": "/tmp/code"})

        mock_s3_upload = AsyncMock(return_value="s3://bkt/artenic-training/nlp/bert/abc")
        p._s3_upload_directory = mock_s3_upload

        uri = await p._upload_code(spec)
        assert uri.startswith("s3://")


class TestHetznerProvisionAndStart:
    async def test_provision_returns_job_id(self):
        p = _build_provider()
        await p._connect()

        p._client.server_types.get_by_name.return_value = _mock_server_type()
        p._client.images.get_by_name_and_architecture.return_value = MagicMock()
        p._client.locations.get_by_name.return_value = MagicMock(name="fsn1")
        p._client.servers.create.return_value = _mock_create_response()

        spec = _make_spec()
        job_id = await p._provision_and_start(spec)
        assert job_id.startswith("hetzner-")
        assert job_id in p._jobs

    async def test_provision_unknown_server_type(self):
        p = _build_provider()
        await p._connect()

        p._client.server_types.get_by_name.return_value = None
        spec = _make_spec(instance_type="nonexistent")

        with pytest.raises(ValueError, match="Unknown Hetzner"):
            await p._provision_and_start(spec)

    async def test_provision_unknown_image(self):
        p = _build_provider()
        await p._connect()

        p._client.server_types.get_by_name.return_value = _mock_server_type()
        p._client.images.get_by_name_and_architecture.return_value = None
        spec = _make_spec()

        with pytest.raises(ValueError, match="Image"):
            await p._provision_and_start(spec)


class TestHetznerPollProvider:
    async def test_poll_unknown_job(self):
        p = _build_provider()
        await p._connect()
        r = await p._poll_provider("no-job")
        assert r.status == JobStatus.FAILED
        assert "Unknown job" in r.error

    async def test_poll_running(self):
        p = _build_provider()
        await p._connect()

        # Provision a job first
        p._client.server_types.get_by_name.return_value = _mock_server_type()
        p._client.images.get_by_name_and_architecture.return_value = MagicMock()
        p._client.locations.get_by_name.return_value = MagicMock(name="fsn1")
        p._client.servers.create.return_value = _mock_create_response()
        jid = await p._provision_and_start(_make_spec())

        # Mock the poll
        running_server = _mock_server(status="running")
        p._client.servers.get_by_id.return_value = running_server

        # Mock _parse_training_log to avoid SSH
        p._parse_training_log = AsyncMock(return_value=(None, False, None))
        r = await p._poll_provider(jid)
        assert r.status == JobStatus.RUNNING

    async def test_poll_initializing(self):
        p = _build_provider()
        await p._connect()

        p._client.server_types.get_by_name.return_value = _mock_server_type()
        p._client.images.get_by_name_and_architecture.return_value = MagicMock()
        p._client.locations.get_by_name.return_value = MagicMock(name="fsn1")
        p._client.servers.create.return_value = _mock_create_response()
        jid = await p._provision_and_start(_make_spec())

        p._client.servers.get_by_id.return_value = _mock_server(status="initializing")
        r = await p._poll_provider(jid)
        assert r.status == JobStatus.PENDING

    async def test_poll_off(self):
        p = _build_provider()
        await p._connect()

        p._client.server_types.get_by_name.return_value = _mock_server_type()
        p._client.images.get_by_name_and_architecture.return_value = MagicMock()
        p._client.locations.get_by_name.return_value = MagicMock(name="fsn1")
        p._client.servers.create.return_value = _mock_create_response()
        jid = await p._provision_and_start(_make_spec())

        p._client.servers.get_by_id.return_value = _mock_server(status="off")
        r = await p._poll_provider(jid)
        assert r.status == JobStatus.COMPLETED

    async def test_poll_server_none(self):
        p = _build_provider()
        await p._connect()

        p._client.server_types.get_by_name.return_value = _mock_server_type()
        p._client.images.get_by_name_and_architecture.return_value = MagicMock()
        p._client.locations.get_by_name.return_value = MagicMock(name="fsn1")
        p._client.servers.create.return_value = _mock_create_response()
        jid = await p._provision_and_start(_make_spec())

        p._client.servers.get_by_id.return_value = None
        r = await p._poll_provider(jid)
        assert r.status == JobStatus.FAILED
        assert "no longer exists" in r.error

    async def test_poll_api_error(self):
        p = _build_provider()
        await p._connect()

        p._client.server_types.get_by_name.return_value = _mock_server_type()
        p._client.images.get_by_name_and_architecture.return_value = MagicMock()
        p._client.locations.get_by_name.return_value = MagicMock(name="fsn1")
        p._client.servers.create.return_value = _mock_create_response()
        jid = await p._provision_and_start(_make_spec())

        p._client.servers.get_by_id.side_effect = RuntimeError("boom")
        r = await p._poll_provider(jid)
        assert r.status == JobStatus.FAILED
        assert "Cannot reach" in r.error

    async def test_poll_training_completed(self):
        p = _build_provider()
        await p._connect()

        p._client.server_types.get_by_name.return_value = _mock_server_type()
        p._client.images.get_by_name_and_architecture.return_value = MagicMock()
        p._client.locations.get_by_name.return_value = MagicMock(name="fsn1")
        p._client.servers.create.return_value = _mock_create_response()
        jid = await p._provision_and_start(_make_spec())

        p._client.servers.get_by_id.return_value = _mock_server(status="running")
        p._parse_training_log = AsyncMock(return_value=({"loss": 0.1}, True, None))
        r = await p._poll_provider(jid)
        assert r.status == JobStatus.COMPLETED
        assert r.metrics == {"loss": 0.1}


class TestHetznerCollectArtifacts:
    async def test_collect_unknown_job(self):
        p = _build_provider()
        await p._connect()
        st = CloudJobStatus(provider_job_id="x", status=JobStatus.COMPLETED)
        assert await p._collect_artifacts("x", st) is None

    async def test_collect_with_s3(self):
        p = _build_provider(
            s3_endpoint="https://s3.example.com",
            s3_access_key="ak",
            s3_secret_key="sk",
            s3_bucket="bkt",
        )
        await p._connect()

        p._client.server_types.get_by_name.return_value = _mock_server_type()
        p._client.images.get_by_name_and_architecture.return_value = MagicMock()
        p._client.locations.get_by_name.return_value = MagicMock(name="fsn1")
        p._client.servers.create.return_value = _mock_create_response()
        jid = await p._provision_and_start(_make_spec())

        st = CloudJobStatus(provider_job_id=jid, status=JobStatus.COMPLETED)
        uri = await p._collect_artifacts(jid, st)
        assert uri is not None
        assert uri.startswith("s3://bkt/")


class TestHetznerCleanupCompute:
    async def test_cleanup_deletes_server(self):
        p = _build_provider()
        await p._connect()

        p._client.server_types.get_by_name.return_value = _mock_server_type()
        p._client.images.get_by_name_and_architecture.return_value = MagicMock()
        p._client.locations.get_by_name.return_value = MagicMock(name="fsn1")
        p._client.servers.create.return_value = _mock_create_response()
        jid = await p._provision_and_start(_make_spec())

        p._client.servers.get_by_id.return_value = _mock_server()
        await p._cleanup_compute(jid)
        p._client.servers.delete.assert_called_once()
        assert jid not in p._jobs

    async def test_cleanup_unknown_job(self):
        p = _build_provider()
        await p._connect()
        await p._cleanup_compute("nope")

    async def test_cleanup_raises_on_delete_failure(self):
        p = _build_provider()
        await p._connect()

        p._client.server_types.get_by_name.return_value = _mock_server_type()
        p._client.images.get_by_name_and_architecture.return_value = MagicMock()
        p._client.locations.get_by_name.return_value = MagicMock(name="fsn1")
        p._client.servers.create.return_value = _mock_create_response()
        jid = await p._provision_and_start(_make_spec())

        p._client.servers.get_by_id.return_value = _mock_server()
        p._client.servers.delete.side_effect = RuntimeError("fail")
        with pytest.raises(RuntimeError):
            await p._cleanup_compute(jid)


class TestHetznerCancelJob:
    async def test_cancel_shuts_down_server(self):
        p = _build_provider()
        await p._connect()

        p._client.server_types.get_by_name.return_value = _mock_server_type()
        p._client.images.get_by_name_and_architecture.return_value = MagicMock()
        p._client.locations.get_by_name.return_value = MagicMock(name="fsn1")
        p._client.servers.create.return_value = _mock_create_response()
        jid = await p._provision_and_start(_make_spec())

        p._client.servers.get_by_id.return_value = _mock_server()
        await p._cancel_provider_job(jid)
        p._client.servers.shutdown.assert_called_once()

    async def test_cancel_unknown_job(self):
        p = _build_provider()
        await p._connect()
        await p._cancel_provider_job("nope")

    async def test_cancel_falls_back_to_power_off(self):
        p = _build_provider()
        await p._connect()

        p._client.server_types.get_by_name.return_value = _mock_server_type()
        p._client.images.get_by_name_and_architecture.return_value = MagicMock()
        p._client.locations.get_by_name.return_value = MagicMock(name="fsn1")
        p._client.servers.create.return_value = _mock_create_response()
        jid = await p._provision_and_start(_make_spec())

        srv = _mock_server()
        p._client.servers.get_by_id.return_value = srv
        p._client.servers.shutdown.side_effect = RuntimeError("nope")

        await p._cancel_provider_job(jid)
        p._client.servers.power_off.assert_called_once_with(srv)


class TestHetznerEstimateCost:
    def test_estimate_with_hourly_price(self):
        from artenic_ai_platform_providers.hetzner import (
            _HetznerJobState,
        )

        p = _build_provider()
        state = _HetznerJobState(
            server_id=1,
            server_name="test",
            created_at=time.time(),
            spec=_make_spec(instance_type="cx22"),
            hourly_price=0.5,
        )
        cost = p._estimate_cost(state, 7200.0)
        assert cost == 1.0

    def test_estimate_without_price(self):
        from artenic_ai_platform_providers.hetzner import (
            _HetznerJobState,
        )

        p = _build_provider()
        state = _HetznerJobState(
            server_id=1,
            server_name="test",
            created_at=time.time(),
            spec=_make_spec(),
        )
        assert p._estimate_cost(state, 3600.0) is None

    def test_estimate_no_instance_type(self):
        from artenic_ai_platform_providers.hetzner import (
            _HetznerJobState,
        )

        p = _build_provider()
        state = _HetznerJobState(
            server_id=1,
            server_name="test",
            created_at=time.time(),
            spec=_make_spec(instance_type=None),
        )
        assert p._estimate_cost(state, 3600.0) is None


# ===================================================================
# Coverage: SSH key resolution in _provision_and_start (lines 293-297)
# ===================================================================


class TestHetznerProvisionSSHKey:
    async def test_provision_with_ssh_key_found(self):
        p = _build_provider(ssh_key_name="my-key")
        await p._connect()

        ssh_key_mock = MagicMock()
        p._client.server_types.get_by_name.return_value = _mock_server_type()
        p._client.images.get_by_name_and_architecture.return_value = MagicMock()
        p._client.locations.get_by_name.return_value = MagicMock(name="fsn1")
        p._client.ssh_keys.get_by_name.return_value = ssh_key_mock
        p._client.servers.create.return_value = _mock_create_response()

        spec = _make_spec()
        job_id = await p._provision_and_start(spec)
        assert job_id.startswith("hetzner-")
        p._client.ssh_keys.get_by_name.assert_called_once_with("my-key")

    async def test_provision_with_ssh_key_not_found(self):
        p = _build_provider(ssh_key_name="missing-key")
        await p._connect()

        p._client.server_types.get_by_name.return_value = _mock_server_type()
        p._client.images.get_by_name_and_architecture.return_value = MagicMock()
        p._client.locations.get_by_name.return_value = MagicMock(name="fsn1")
        p._client.ssh_keys.get_by_name.return_value = None
        p._client.servers.create.return_value = _mock_create_response()

        spec = _make_spec()
        job_id = await p._provision_and_start(spec)
        assert job_id.startswith("hetzner-")


# ===================================================================
# Coverage: SCP code upload path (lines 352-353)
# ===================================================================


class TestHetznerProvisionSCPUpload:
    async def test_provision_triggers_scp_upload(self):
        p = _build_provider(ssh_key_name="my-key")
        await p._connect()

        ssh_key_mock = MagicMock()
        p._client.server_types.get_by_name.return_value = _mock_server_type()
        p._client.images.get_by_name_and_architecture.return_value = MagicMock()
        p._client.locations.get_by_name.return_value = MagicMock(name="fsn1")
        p._client.ssh_keys.get_by_name.return_value = ssh_key_mock
        p._client.servers.create.return_value = _mock_create_response()

        # Mock wait and scp
        p._wait_for_ssh = AsyncMock()
        p._scp_upload = AsyncMock()

        spec = _make_spec(config={"code_path": "/tmp/mycode"})
        job_id = await p._provision_and_start(spec)
        assert job_id.startswith("hetzner-")
        p._wait_for_ssh.assert_called_once()
        p._scp_upload.assert_called_once()


# ===================================================================
# Coverage: Poll unknown status -> FAILED (line 430)
# ===================================================================


class TestHetznerPollUnknownStatus:
    async def test_poll_unknown_hetzner_status(self):
        p = _build_provider()
        await p._connect()

        p._client.server_types.get_by_name.return_value = _mock_server_type()
        p._client.images.get_by_name_and_architecture.return_value = MagicMock()
        p._client.locations.get_by_name.return_value = MagicMock(name="fsn1")
        p._client.servers.create.return_value = _mock_create_response()
        jid = await p._provision_and_start(_make_spec())

        # Use an unexpected server status
        p._client.servers.get_by_id.return_value = _mock_server(status="rebuilding")
        r = await p._poll_provider(jid)
        assert r.status == JobStatus.FAILED


# ===================================================================
# Coverage: _collect_artifacts without S3 / SCP paths (lines 471-516)
# ===================================================================


class TestHetznerCollectArtifactsSCP:
    async def test_collect_scp_success(self, monkeypatch):
        p = _build_provider()
        await p._connect()

        p._client.server_types.get_by_name.return_value = _mock_server_type()
        p._client.images.get_by_name_and_architecture.return_value = MagicMock()
        p._client.locations.get_by_name.return_value = MagicMock(name="fsn1")
        p._client.servers.create.return_value = _mock_create_response()
        jid = await p._provision_and_start(_make_spec())

        server = _mock_server(server_id=42)
        p._client.servers.get_by_id.return_value = server

        # Mock subprocess.run to succeed
        monkeypatch.setattr(
            subprocess,
            "run",
            MagicMock(return_value=MagicMock(returncode=0)),
        )

        st = CloudJobStatus(provider_job_id=jid, status=JobStatus.COMPLETED)
        result = await p._collect_artifacts(jid, st)
        assert result is not None

    async def test_collect_scp_server_fetch_fails(self):
        p = _build_provider()
        await p._connect()

        p._client.server_types.get_by_name.return_value = _mock_server_type()
        p._client.images.get_by_name_and_architecture.return_value = MagicMock()
        p._client.locations.get_by_name.return_value = MagicMock(name="fsn1")
        p._client.servers.create.return_value = _mock_create_response()
        jid = await p._provision_and_start(_make_spec())

        p._client.servers.get_by_id.side_effect = RuntimeError("gone")

        st = CloudJobStatus(provider_job_id=jid, status=JobStatus.COMPLETED)
        result = await p._collect_artifacts(jid, st)
        assert result is None

    async def test_collect_scp_server_none(self):
        p = _build_provider()
        await p._connect()

        p._client.server_types.get_by_name.return_value = _mock_server_type()
        p._client.images.get_by_name_and_architecture.return_value = MagicMock()
        p._client.locations.get_by_name.return_value = MagicMock(name="fsn1")
        p._client.servers.create.return_value = _mock_create_response()
        jid = await p._provision_and_start(_make_spec())

        p._client.servers.get_by_id.return_value = None

        st = CloudJobStatus(provider_job_id=jid, status=JobStatus.COMPLETED)
        result = await p._collect_artifacts(jid, st)
        assert result is None

    async def test_collect_scp_no_ipv4(self):
        p = _build_provider()
        await p._connect()

        p._client.server_types.get_by_name.return_value = _mock_server_type()
        p._client.images.get_by_name_and_architecture.return_value = MagicMock()
        p._client.locations.get_by_name.return_value = MagicMock(name="fsn1")
        p._client.servers.create.return_value = _mock_create_response()
        jid = await p._provision_and_start(_make_spec())

        server = _mock_server()
        server.public_net.ipv4 = None
        p._client.servers.get_by_id.return_value = server

        st = CloudJobStatus(provider_job_id=jid, status=JobStatus.COMPLETED)
        result = await p._collect_artifacts(jid, st)
        assert result is None

    async def test_collect_scp_download_fails(self, monkeypatch):
        p = _build_provider()
        await p._connect()

        p._client.server_types.get_by_name.return_value = _mock_server_type()
        p._client.images.get_by_name_and_architecture.return_value = MagicMock()
        p._client.locations.get_by_name.return_value = MagicMock(name="fsn1")
        p._client.servers.create.return_value = _mock_create_response()
        jid = await p._provision_and_start(_make_spec())

        server = _mock_server(server_id=42)
        p._client.servers.get_by_id.return_value = server

        monkeypatch.setattr(
            subprocess,
            "run",
            MagicMock(side_effect=subprocess.CalledProcessError(1, "scp")),
        )

        st = CloudJobStatus(provider_job_id=jid, status=JobStatus.COMPLETED)
        result = await p._collect_artifacts(jid, st)
        assert result is None


# ===================================================================
# Coverage: _cancel_provider_job outer exception (lines 582-588)
# ===================================================================


class TestHetznerCancelOuterException:
    async def test_cancel_outer_exception_raises(self):
        p = _build_provider()
        await p._connect()

        p._client.server_types.get_by_name.return_value = _mock_server_type()
        p._client.images.get_by_name_and_architecture.return_value = MagicMock()
        p._client.locations.get_by_name.return_value = MagicMock(name="fsn1")
        p._client.servers.create.return_value = _mock_create_response()
        jid = await p._provision_and_start(_make_spec())

        # Make get_by_id itself raise (outer except block)
        p._client.servers.get_by_id.side_effect = RuntimeError("api down")

        with pytest.raises(RuntimeError, match="api down"):
            await p._cancel_provider_job(jid)


# ===================================================================
# Coverage: _build_user_data env vars (lines 649-651)
# ===================================================================


class TestHetznerBuildUserDataEnv:
    def test_build_user_data_with_env_vars(self):
        p = _build_provider()
        spec = _make_spec(config={"env": {"MY_VAR": "hello", "NUM": 42}})
        user_data = p._build_user_data(spec, "job-1")
        assert "MY_VAR" in user_data
        assert "hello" in user_data
        assert "42" in user_data


# ===================================================================
# Coverage: _wait_for_ssh (lines 673-707)
# ===================================================================


class TestHetznerWaitForSSH:
    async def test_wait_for_ssh_success(self, monkeypatch):
        p = _build_provider()

        server = _mock_server(ipv4_ip="10.0.0.1")
        monkeypatch.setattr(
            subprocess,
            "run",
            MagicMock(return_value=MagicMock(returncode=0)),
        )

        await p._wait_for_ssh(server, timeout=5.0, interval=0.01)

    async def test_wait_for_ssh_no_ipv4(self):
        p = _build_provider()
        server = _mock_server()
        server.public_net.ipv4 = None

        with pytest.raises(RuntimeError, match="no public IPv4"):
            await p._wait_for_ssh(server)

    async def test_wait_for_ssh_timeout(self, monkeypatch):
        p = _build_provider()
        server = _mock_server(ipv4_ip="10.0.0.1")

        # Always fail SSH
        monkeypatch.setattr(
            subprocess,
            "run",
            MagicMock(return_value=MagicMock(returncode=1)),
        )
        # Patch asyncio.sleep to avoid real sleeping
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

        with pytest.raises(TimeoutError, match="SSH not available"):
            await p._wait_for_ssh(server, timeout=0.01, interval=0.001)

    async def test_wait_for_ssh_exception_in_subprocess(self, monkeypatch):
        p = _build_provider()
        server = _mock_server(ipv4_ip="10.0.0.1")

        call_count = 0

        def _failing_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise OSError("connection refused")
            return MagicMock(returncode=0)

        monkeypatch.setattr(subprocess, "run", _failing_run)
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

        await p._wait_for_ssh(server, timeout=10.0, interval=0.001)


# ===================================================================
# Coverage: _scp_upload (lines 711-751)
# ===================================================================


class TestHetznerSCPUpload:
    async def test_scp_upload_success(self, monkeypatch):
        p = _build_provider()
        server = _mock_server(ipv4_ip="10.0.0.1")

        monkeypatch.setattr(
            subprocess,
            "run",
            MagicMock(return_value=MagicMock(returncode=0)),
        )

        await p._scp_upload(server, "/tmp/code")

    async def test_scp_upload_no_ipv4(self):
        p = _build_provider()
        server = _mock_server()
        server.public_net.ipv4 = None

        with pytest.raises(RuntimeError, match="no public IPv4"):
            await p._scp_upload(server, "/tmp/code")


# ===================================================================
# Coverage: _parse_training_log (lines 768-827)
# ===================================================================


class TestHetznerParseTrainingLog:
    async def test_parse_log_no_ipv4(self):
        p = _build_provider()
        server = _mock_server()
        server.public_net.ipv4 = None

        metrics, done, err = await p._parse_training_log(server)
        assert metrics is None
        assert done is False
        assert err is None

    async def test_parse_log_ssh_exception(self, monkeypatch):
        p = _build_provider()
        server = _mock_server(ipv4_ip="10.0.0.1")

        monkeypatch.setattr(
            subprocess,
            "run",
            MagicMock(side_effect=OSError("connection refused")),
        )

        metrics, done, _err = await p._parse_training_log(server)
        assert metrics is None
        assert done is False

    async def test_parse_log_nonzero_returncode(self, monkeypatch):
        p = _build_provider()
        server = _mock_server(ipv4_ip="10.0.0.1")

        monkeypatch.setattr(
            subprocess,
            "run",
            MagicMock(return_value=MagicMock(returncode=1, stdout="")),
        )

        metrics, done, _err = await p._parse_training_log(server)
        assert metrics is None
        assert done is False

    async def test_parse_log_empty_output(self, monkeypatch):
        p = _build_provider()
        server = _mock_server(ipv4_ip="10.0.0.1")

        monkeypatch.setattr(
            subprocess,
            "run",
            MagicMock(return_value=MagicMock(returncode=0, stdout="   ")),
        )

        metrics, done, _err = await p._parse_training_log(server)
        assert metrics is None
        assert done is False

    async def test_parse_log_training_done_success(self, monkeypatch):
        p = _build_provider()
        server = _mock_server(ipv4_ip="10.0.0.1")

        log_output = '{"event":"training_done","exit_code":0}\n'
        monkeypatch.setattr(
            subprocess,
            "run",
            MagicMock(return_value=MagicMock(returncode=0, stdout=log_output)),
        )

        _metrics, done, err = await p._parse_training_log(server)
        assert done is True
        assert err is None

    async def test_parse_log_training_done_failure(self, monkeypatch):
        p = _build_provider()
        server = _mock_server(ipv4_ip="10.0.0.1")

        log_output = '{"event":"training_done","exit_code":1}\n'
        monkeypatch.setattr(
            subprocess,
            "run",
            MagicMock(return_value=MagicMock(returncode=0, stdout=log_output)),
        )

        _metrics, done, err = await p._parse_training_log(server)
        assert done is True
        assert err is not None
        assert "exit" in err.lower()

    async def test_parse_log_metrics_event(self, monkeypatch):
        p = _build_provider()
        server = _mock_server(ipv4_ip="10.0.0.1")

        log_output = '{"event":"metrics","data":{"loss":0.05,"acc":0.95}}\n'
        monkeypatch.setattr(
            subprocess,
            "run",
            MagicMock(return_value=MagicMock(returncode=0, stdout=log_output)),
        )

        metrics, done, _err = await p._parse_training_log(server)
        assert metrics is not None
        assert metrics["loss"] == 0.05
        assert done is False

    async def test_parse_log_artifacts_uploaded(self, monkeypatch):
        p = _build_provider()
        server = _mock_server(ipv4_ip="10.0.0.1")

        log_output = '{"event":"artifacts_uploaded"}\n'
        monkeypatch.setattr(
            subprocess,
            "run",
            MagicMock(return_value=MagicMock(returncode=0, stdout=log_output)),
        )

        metrics, _done, _err = await p._parse_training_log(server)
        assert metrics is not None
        assert metrics["artifacts_uploaded"] is True

    async def test_parse_log_invalid_json(self, monkeypatch):
        p = _build_provider()
        server = _mock_server(ipv4_ip="10.0.0.1")

        log_output = "not-json\n\n{broken\n"
        monkeypatch.setattr(
            subprocess,
            "run",
            MagicMock(return_value=MagicMock(returncode=0, stdout=log_output)),
        )

        metrics, done, _err = await p._parse_training_log(server)
        assert metrics is None
        assert done is False

    async def test_parse_log_training_done_with_error(self, monkeypatch):
        """Training done with error + metrics in one log."""
        p = _build_provider()
        server = _mock_server(ipv4_ip="10.0.0.1")

        log_output = (
            '{"event":"metrics","data":{"loss":0.5}}\n{"event":"training_done","exit_code":2}\n'
        )
        monkeypatch.setattr(
            subprocess,
            "run",
            MagicMock(return_value=MagicMock(returncode=0, stdout=log_output)),
        )

        metrics, done, err = await p._parse_training_log(server)
        assert done is True
        assert err is not None
        assert metrics is not None
        assert metrics["loss"] == 0.5


# ===================================================================
# Coverage: _s3_upload_directory (lines 835-864)
# ===================================================================


class TestHetznerS3UploadDirectory:
    async def test_s3_upload_single_file(self, tmp_path):
        p = _build_provider(
            s3_endpoint="https://s3.example.com",
            s3_access_key="ak",
            s3_secret_key="sk",
            s3_bucket="bkt",
        )

        code_file = tmp_path / "train.py"
        code_file.write_text("print('hello')")

        mock_boto3 = MagicMock()
        mock_s3_client = MagicMock()
        mock_boto3.client.return_value = mock_s3_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            uri = await p._s3_upload_directory(str(code_file), "remote/key")

        assert uri == "s3://bkt/remote/key"
        mock_s3_client.upload_file.assert_called_once()

    async def test_s3_upload_directory(self, tmp_path):
        p = _build_provider(
            s3_endpoint="https://s3.example.com",
            s3_access_key="ak",
            s3_secret_key="sk",
            s3_bucket="bkt",
        )

        (tmp_path / "train.py").write_text("print('hi')")
        (tmp_path / "utils.py").write_text("pass")

        mock_boto3 = MagicMock()
        mock_s3_client = MagicMock()
        mock_boto3.client.return_value = mock_s3_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            uri = await p._s3_upload_directory(str(tmp_path), "remote/key")

        assert uri == "s3://bkt/remote/key"
        assert mock_s3_client.upload_file.call_count == 2

    async def test_s3_upload_no_boto3(self, tmp_path):
        p = _build_provider(
            s3_endpoint="https://s3.example.com",
            s3_access_key="ak",
            s3_secret_key="sk",
            s3_bucket="bkt",
        )

        code_file = tmp_path / "train.py"
        code_file.write_text("print('hi')")

        with patch.dict("sys.modules", {"boto3": None}), pytest.raises(ImportError, match="boto3"):
            await p._s3_upload_directory(str(code_file), "remote/key")


# ===================================================================
# Coverage: _estimate_cost with hourly_price=None but instance_type set
# (line 884)
# ===================================================================


class TestHetznerEstimateCostNoPriceWithInstanceType:
    def test_estimate_cost_no_hourly_price_but_instance_type_set(self):
        from artenic_ai_platform_providers.hetzner import _HetznerJobState

        p = _build_provider()
        state = _HetznerJobState(
            server_id=1,
            server_name="test",
            created_at=time.time(),
            spec=_make_spec(instance_type="cx22"),
            hourly_price=None,
        )
        result = p._estimate_cost(state, 3600.0)
        assert result is None


# ===================================================================
# Coverage: GPU parsing helpers (lines 939-941, 955)
# ===================================================================


class TestGPUParsingHelpers:
    def test_parse_gpu_count_from_name(self):
        from artenic_ai_platform_providers.hetzner import _parse_gpu_count

        st = MagicMock()
        st.name = "gpu-a100-x4"
        st.description = "GPU server"
        assert _parse_gpu_count(st) == 4

    def test_parse_gpu_count_from_description(self):
        from artenic_ai_platform_providers.hetzner import _parse_gpu_count

        st = MagicMock()
        st.name = "custom-server"
        st.description = "GPU server -x2"
        assert _parse_gpu_count(st) == 2

    def test_parse_gpu_count_default_one(self):
        from artenic_ai_platform_providers.hetzner import _parse_gpu_count

        st = MagicMock()
        st.name = "gpu-server"
        st.description = "A GPU server"
        assert _parse_gpu_count(st) == 1

    def test_parse_gpu_count_zero(self):
        from artenic_ai_platform_providers.hetzner import _parse_gpu_count

        st = MagicMock()
        st.name = "cx22"
        st.description = "Standard server"
        assert _parse_gpu_count(st) == 0

    def test_parse_gpu_type_known(self):
        from artenic_ai_platform_providers.hetzner import _parse_gpu_type

        st = MagicMock()
        st.name = "gpu-a100-x2"
        st.description = "A100 GPU server"
        assert _parse_gpu_type(st) == "A100"

    def test_parse_gpu_type_h100(self):
        from artenic_ai_platform_providers.hetzner import _parse_gpu_type

        st = MagicMock()
        st.name = "gpu-h100-80gb"
        st.description = "H100 server"
        assert _parse_gpu_type(st) == "H100"

    def test_parse_gpu_type_fallback(self):
        from artenic_ai_platform_providers.hetzner import _parse_gpu_type

        st = MagicMock()
        st.name = "gpu-unknown-model"
        st.description = "Some GPU"
        assert _parse_gpu_type(st) == "GPU"

    def test_parse_gpu_type_none_description(self):
        from artenic_ai_platform_providers.hetzner import _parse_gpu_type

        st = MagicMock()
        st.name = "gpu-mystery"
        st.description = None
        assert _parse_gpu_type(st) == "GPU"


# ===================================================================
# Coverage: Poll training done with error (lines 413-426)
# ===================================================================


class TestHetznerPollTrainingFailed:
    async def test_poll_training_done_with_error(self):
        p = _build_provider()
        await p._connect()

        p._client.server_types.get_by_name.return_value = _mock_server_type()
        p._client.images.get_by_name_and_architecture.return_value = MagicMock()
        p._client.locations.get_by_name.return_value = MagicMock(name="fsn1")
        p._client.servers.create.return_value = _mock_create_response()
        jid = await p._provision_and_start(_make_spec())

        p._client.servers.get_by_id.return_value = _mock_server(status="running")
        p._parse_training_log = AsyncMock(
            return_value=(None, True, "Training exited with code 1"),
        )
        r = await p._poll_provider(jid)
        assert r.status == JobStatus.FAILED
        assert r.error is not None
