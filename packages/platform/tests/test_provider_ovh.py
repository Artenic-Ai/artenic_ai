"""Tests for artenic_ai_platform.providers.ovh — OVHProvider."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from artenic_ai_platform.providers.base import (
    CloudJobStatus,
    JobStatus,
    TrainingSpec,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(**overrides) -> TrainingSpec:
    """Create a TrainingSpec with sensible defaults."""
    defaults = {
        "service": "nlp",
        "model": "bert",
        "provider": "ovh",
        "config": {"image_id": "img-123", "code_path": "."},
    }
    defaults.update(overrides)
    return TrainingSpec(**defaults)


def _mock_flavor(name="b2-7", vcpus=2, ram=2048):
    """Return a lightweight flavor mock."""
    flv = MagicMock()
    flv.name = name
    flv.vcpus = vcpus
    flv.ram = ram
    return flv


def _mock_server(
    server_id="srv-001",
    status="ACTIVE",
):
    srv = MagicMock()
    srv.id = server_id
    srv.status = status
    return srv


# ---------------------------------------------------------------------------
# Module-level patches applied to every test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_to_thread(monkeypatch):
    """Make asyncio.to_thread synchronous for testing."""
    monkeypatch.setattr(
        "asyncio.to_thread",
        AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
    )


@pytest.fixture(autouse=True)
def _patch_openstack(monkeypatch):
    """Patch the openstack SDK references on the ovh module."""
    mock_openstack = MagicMock()
    mock_conn = MagicMock()
    mock_openstack.connect.return_value = mock_conn
    mock_conn.compute.flavors.return_value = [_mock_flavor()]
    mock_conn.close = MagicMock()

    import artenic_ai_platform.providers.ovh as ovh_mod

    monkeypatch.setattr(ovh_mod, "openstack", mock_openstack)


def _build_provider():
    """Instantiate an OVHProvider with mocked SDK."""
    from artenic_ai_platform.providers.ovh import OVHProvider

    return OVHProvider(
        auth_url="https://auth.cloud.ovh.net/v3",
        username="user",
        password="pass",
        project_id="proj-123",
        region="GRA",
        image_id="img-123",
        network_id="net-456",
    )


# ===================================================================
# Tests
# ===================================================================


class TestOVHProviderInit:
    def test_init_stores_params(self):
        p = _build_provider()
        assert p._auth_url == "https://auth.cloud.ovh.net/v3"
        assert p._username == "user"
        assert p._region == "GRA"
        assert p._container_name == "artenic-training"


class TestOVHProviderName:
    def test_provider_name(self):
        p = _build_provider()
        assert p.provider_name == "ovh"


class TestOVHConnect:
    async def test_connect_creates_connection(self):
        p = _build_provider()
        await p._connect()
        assert p._conn is not None


class TestOVHDisconnect:
    async def test_disconnect_clears_connection(self):
        p = _build_provider()
        await p._connect()
        assert p._conn is not None
        await p._disconnect()
        assert p._conn is None

    async def test_disconnect_noop_when_not_connected(self):
        p = _build_provider()
        await p._disconnect()  # should not raise
        assert p._conn is None


class TestOVHListInstances:
    async def test_list_instances_returns_types(self):
        p = _build_provider()
        await p._connect()
        # Override pricing fetch
        p._fetch_ovh_pricing = AsyncMock(return_value={})
        instances = await p._list_instances()
        assert len(instances) >= 1
        assert instances[0].name == "b2-7"

    async def test_list_instances_gpu_only(self):
        p = _build_provider()
        await p._connect()
        p._fetch_ovh_pricing = AsyncMock(return_value={})

        # Add a GPU flavor to the mock
        gpu_flv = _mock_flavor(name="gpu-a100-80g", vcpus=12, ram=81920)
        p._conn.compute.flavors.return_value = [
            _mock_flavor(),
            gpu_flv,
        ]

        instances = await p._list_instances(gpu_only=True)
        assert len(instances) == 1
        assert instances[0].gpu_count >= 1
        assert instances[0].gpu_type == "A100"

    async def test_list_instances_with_pricing(self):
        p = _build_provider()
        await p._connect()
        p._fetch_ovh_pricing = AsyncMock(return_value={"b2-7": 0.05})
        instances = await p._list_instances()
        assert instances[0].price_per_hour_eur == 0.05


class TestOVHUploadCode:
    async def test_upload_code_returns_uri(self):
        p = _build_provider()
        await p._connect()
        spec = _make_spec()

        # Mock tarball creation
        with (
            patch.object(type(p), "_create_tarball", return_value="/tmp/fake.tar.gz"),
            patch("builtins.open", MagicMock()),
            patch("os.path.exists", return_value=True),
            patch("os.remove"),
        ):
            uri = await p._upload_code(spec)

        assert uri.startswith("swift://artenic-training/code/")

    async def test_upload_code_cleans_up_tarball(self):
        p = _build_provider()
        await p._connect()
        spec = _make_spec()

        with (
            patch.object(type(p), "_create_tarball", return_value="/tmp/fake.tar.gz"),
            patch("builtins.open", MagicMock()),
            patch("os.path.exists", return_value=True) as mock_exists,
            patch("os.remove") as mock_remove,
        ):
            await p._upload_code(spec)

        mock_exists.assert_called_with("/tmp/fake.tar.gz")
        mock_remove.assert_called_once_with("/tmp/fake.tar.gz")


class TestOVHProvisionAndStart:
    async def test_provision_returns_job_id(self):
        p = _build_provider()
        await p._connect()
        p._conn.compute.create_server.return_value = _mock_server()
        spec = _make_spec()

        job_id = await p._provision_and_start(spec)
        assert job_id.startswith("ovh-")
        assert job_id in p._jobs

    async def test_provision_raises_without_image(self):
        p = _build_provider()
        p._image_id = None
        await p._connect()
        spec = _make_spec(config={})

        with pytest.raises(ValueError, match="image_id"):
            await p._provision_and_start(spec)

    async def test_provision_with_network(self):
        p = _build_provider()
        await p._connect()
        p._conn.compute.create_server.return_value = _mock_server()
        spec = _make_spec(config={"image_id": "img-1", "network_id": "n-1"})

        await p._provision_and_start(spec)
        call_kwargs = p._conn.compute.create_server.call_args
        assert "networks" in call_kwargs.kwargs


class TestOVHPollProvider:
    async def test_poll_unknown_job(self):
        p = _build_provider()
        await p._connect()
        result = await p._poll_provider("nonexistent")
        assert result.status == JobStatus.FAILED
        assert "Unknown job" in result.error

    async def test_poll_active_server(self):
        p = _build_provider()
        await p._connect()
        p._conn.compute.create_server.return_value = _mock_server()
        spec = _make_spec()
        job_id = await p._provision_and_start(spec)

        p._conn.compute.get_server.return_value = _mock_server(status="ACTIVE")
        result = await p._poll_provider(job_id)
        assert result.status == JobStatus.RUNNING

    async def test_poll_build_server(self):
        p = _build_provider()
        await p._connect()
        p._conn.compute.create_server.return_value = _mock_server()
        spec = _make_spec()
        job_id = await p._provision_and_start(spec)

        p._conn.compute.get_server.return_value = _mock_server(status="BUILD")
        result = await p._poll_provider(job_id)
        assert result.status == JobStatus.PENDING

    async def test_poll_shutoff_server(self):
        p = _build_provider()
        await p._connect()
        p._conn.compute.create_server.return_value = _mock_server()
        spec = _make_spec()
        job_id = await p._provision_and_start(spec)

        p._conn.compute.get_server.return_value = _mock_server(status="SHUTOFF")
        result = await p._poll_provider(job_id)
        assert result.status == JobStatus.COMPLETED

    async def test_poll_error_server(self):
        p = _build_provider()
        await p._connect()
        p._conn.compute.create_server.return_value = _mock_server()
        spec = _make_spec()
        job_id = await p._provision_and_start(spec)

        p._conn.compute.get_server.return_value = _mock_server(status="ERROR")
        result = await p._poll_provider(job_id)
        assert result.status == JobStatus.FAILED

    async def test_poll_api_error(self):
        p = _build_provider()
        await p._connect()
        p._conn.compute.create_server.return_value = _mock_server()
        spec = _make_spec()
        job_id = await p._provision_and_start(spec)

        p._conn.compute.get_server.side_effect = RuntimeError("API down")
        result = await p._poll_provider(job_id)
        assert result.status == JobStatus.FAILED
        assert "Cannot reach server" in result.error


class TestOVHCollectArtifacts:
    async def test_collect_unknown_job(self):
        p = _build_provider()
        await p._connect()
        status = CloudJobStatus(provider_job_id="x", status=JobStatus.COMPLETED)
        result = await p._collect_artifacts("x", status)
        assert result is None

    async def test_collect_download_failure(self):
        p = _build_provider()
        await p._connect()
        p._conn.compute.create_server.return_value = _mock_server()
        spec = _make_spec()
        job_id = await p._provision_and_start(spec)

        p._conn.object_store.download_object.side_effect = RuntimeError("not found")
        status = CloudJobStatus(provider_job_id=job_id, status=JobStatus.COMPLETED)
        result = await p._collect_artifacts(job_id, status)
        assert result is None


class TestOVHCleanupCompute:
    async def test_cleanup_deletes_server(self):
        p = _build_provider()
        await p._connect()
        p._conn.compute.create_server.return_value = _mock_server()
        spec = _make_spec()
        job_id = await p._provision_and_start(spec)

        await p._cleanup_compute(job_id)
        p._conn.compute.delete_server.assert_called_once()
        assert job_id not in p._jobs

    async def test_cleanup_unknown_job(self):
        p = _build_provider()
        await p._connect()
        await p._cleanup_compute("nonexistent")  # should not raise

    async def test_cleanup_delete_failure(self):
        p = _build_provider()
        await p._connect()
        p._conn.compute.create_server.return_value = _mock_server()
        spec = _make_spec()
        job_id = await p._provision_and_start(spec)

        p._conn.compute.delete_server.side_effect = RuntimeError("delete failed")
        with pytest.raises(RuntimeError, match="delete failed"):
            await p._cleanup_compute(job_id)


class TestOVHCancelProviderJob:
    async def test_cancel_stops_server(self):
        p = _build_provider()
        await p._connect()
        p._conn.compute.create_server.return_value = _mock_server()
        spec = _make_spec()
        job_id = await p._provision_and_start(spec)

        await p._cancel_provider_job(job_id)
        p._conn.compute.stop_server.assert_called_once()

    async def test_cancel_unknown_job(self):
        p = _build_provider()
        await p._connect()
        await p._cancel_provider_job("nonexistent")  # no raise

    async def test_cancel_stop_failure_is_logged(self):
        p = _build_provider()
        await p._connect()
        p._conn.compute.create_server.return_value = _mock_server()
        spec = _make_spec()
        job_id = await p._provision_and_start(spec)

        p._conn.compute.stop_server.side_effect = RuntimeError("stop failed")
        # Should NOT raise — failure is logged and swallowed
        await p._cancel_provider_job(job_id)


class TestOVHEstimateCost:
    def test_estimate_cost_with_price(self):
        from artenic_ai_platform.providers.ovh import _OVHJobState

        p = _build_provider()
        state = _OVHJobState(
            server_id="s-1",
            created_at=time.time(),
            spec=_make_spec(),
            hourly_price=1.0,
        )
        cost = p._estimate_cost(state, 3600.0)
        assert cost == 1.0

    def test_estimate_cost_without_price(self):
        from artenic_ai_platform.providers.ovh import _OVHJobState

        p = _build_provider()
        state = _OVHJobState(
            server_id="s-1",
            created_at=time.time(),
            spec=_make_spec(),
            hourly_price=None,
        )
        cost = p._estimate_cost(state, 3600.0)
        assert cost is None


# ===================================================================
# Additional tests for 100% coverage
# ===================================================================


class TestOVHPollUnknownStatus:
    """Cover line 421 — the else branch for unknown OpenStack server status."""

    async def test_poll_unknown_status_maps_to_running(self):
        p = _build_provider()
        await p._connect()
        p._conn.compute.create_server.return_value = _mock_server()
        spec = _make_spec()
        job_id = await p._provision_and_start(spec)

        p._conn.compute.get_server.return_value = _mock_server(status="REBOOT")
        result = await p._poll_provider(job_id)
        assert result.status == JobStatus.RUNNING


class TestOVHCollectArtifactsSuccess:
    """Cover lines 470-476 — successful artifact download, write and extract."""

    async def test_collect_artifacts_success(self):
        p = _build_provider()
        await p._connect()
        p._conn.compute.create_server.return_value = _mock_server()
        spec = _make_spec()
        job_id = await p._provision_and_start(spec)

        # download_object returns bytes
        p._conn.object_store.download_object.return_value = b"fake-tar-data"

        with (
            patch("builtins.open", MagicMock()),
            patch("os.makedirs"),
            patch.object(type(p), "_extract_tarball"),
        ):
            status = CloudJobStatus(provider_job_id=job_id, status=JobStatus.COMPLETED)
            result = await p._collect_artifacts(job_id, status)

        assert result is not None
        assert "artenic-artifacts" in result


class TestOVHBuildUserDataEnvVars:
    """Cover lines 576-579 — env vars block in _build_user_data."""

    def test_build_user_data_with_env_vars(self):
        p = _build_provider()
        spec = _make_spec(
            config={
                "image_id": "img-123",
                "code_path": ".",
                "env": {"MY_VAR": "hello", "NUM": 42},
            }
        )
        user_data = p._build_user_data(spec, "ovh-test123")
        assert "export" in user_data
        assert "MY_VAR" in user_data
        assert "hello" in user_data
        assert "42" in user_data


class TestOVHFetchPricing:
    """Cover lines 594-617 — _fetch_ovh_pricing success and failure paths."""

    async def test_fetch_pricing_success(self):
        p = _build_provider()

        mock_response_data = [
            {
                "name": "b2-7",
                "pricingsHourly": {"price": 0.05},
            },
            {
                "name": "b2-15",
                "pricingsHourly": {"price": 0.10},
            },
        ]
        import json

        response_bytes = json.dumps(mock_response_data).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = response_bytes
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = await p._fetch_ovh_pricing()

        assert result == {"b2-7": 0.05, "b2-15": 0.10}

    async def test_fetch_pricing_failure_returns_empty(self):
        p = _build_provider()

        with patch(
            "urllib.request.urlopen",
            side_effect=RuntimeError("network error"),
        ):
            result = await p._fetch_ovh_pricing()

        assert result == {}

    async def test_fetch_pricing_non_dict_hourly(self):
        """Cover the isinstance(hourly, dict) check returning False."""
        p = _build_provider()

        mock_response_data = [
            {
                "name": "b2-7",
                "pricingsHourly": "not-a-dict",
            },
        ]
        import json

        response_bytes = json.dumps(mock_response_data).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_bytes
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = await p._fetch_ovh_pricing()

        assert result == {"b2-7": 0.0}


class TestOVHCreateTarball:
    """Cover lines 625-635 — _create_tarball for files and directories."""

    def test_create_tarball_from_file(self, tmp_path):
        from artenic_ai_platform.providers.ovh import OVHProvider

        test_file = tmp_path / "train.py"
        test_file.write_text("print('hello')")
        result = OVHProvider._create_tarball(str(test_file))
        assert result.endswith(".tar.gz")
        import os

        os.remove(result)

    def test_create_tarball_from_directory(self, tmp_path):
        from artenic_ai_platform.providers.ovh import OVHProvider

        (tmp_path / "file1.py").write_text("a = 1")
        (tmp_path / "file2.py").write_text("b = 2")
        result = OVHProvider._create_tarball(str(tmp_path))
        assert result.endswith(".tar.gz")
        import os

        os.remove(result)


class TestOVHExtractTarball:
    """Cover lines 640-641 — _extract_tarball."""

    def test_extract_tarball(self, tmp_path):
        import tarfile

        from artenic_ai_platform.providers.ovh import OVHProvider

        # Create a tarball
        src_file = tmp_path / "data.txt"
        src_file.write_text("test data")
        tar_path = str(tmp_path / "archive.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(str(src_file), arcname="data.txt")

        dest_dir = str(tmp_path / "extracted")
        import os

        os.makedirs(dest_dir)
        OVHProvider._extract_tarball(tar_path, dest_dir)
        assert os.path.exists(os.path.join(dest_dir, "data.txt"))


class TestOVHGPUParsingEdgeCases:
    """Cover lines 697, 702, 711 — GPU parsing edge cases."""

    def test_parse_gpu_count_multi_gpu(self):
        from artenic_ai_platform.providers.ovh import _parse_gpu_count_from_name

        # Line 697: -xN suffix returns N
        assert _parse_gpu_count_from_name("gpu-a100-80g-x4") == 4
        assert _parse_gpu_count_from_name("gpu-v100-x2") == 2

    def test_parse_gpu_count_non_gpu_returns_zero(self):
        from artenic_ai_platform.providers.ovh import _parse_gpu_count_from_name

        # Line 702: non-GPU name returns 0
        assert _parse_gpu_count_from_name("b2-7") == 0
        assert _parse_gpu_count_from_name("s1-2") == 0

    def test_parse_gpu_type_generic_fallback(self):
        from artenic_ai_platform.providers.ovh import _parse_gpu_type_from_name

        # Line 711: unknown GPU model returns "GPU"
        assert _parse_gpu_type_from_name("gpu-unknown-model") == "GPU"
        assert _parse_gpu_type_from_name("gpu-b2-120") == "GPU"
