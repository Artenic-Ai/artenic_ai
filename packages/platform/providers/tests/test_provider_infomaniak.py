"""Tests for artenic_ai_platform_providers.infomaniak — InfomaniakProvider."""

from __future__ import annotations

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
        "provider": "infomaniak",
        "config": {"image_id": "img-123", "code_path": "."},
    }
    defaults.update(overrides)
    return TrainingSpec(**defaults)


def _mock_flavor(name="a2-ram4-disk20", vcpus=2, ram=4096):
    flv = MagicMock()
    flv.name = name
    flv.vcpus = vcpus
    flv.ram = ram
    return flv


def _mock_server(server_id="srv-001", status="ACTIVE"):
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
    """Patch the openstack SDK references on the infomaniak module."""
    mock_openstack = MagicMock()
    mock_conn = MagicMock()
    mock_openstack.connect.return_value = mock_conn
    mock_conn.compute.flavors.return_value = [_mock_flavor()]
    mock_conn.close = MagicMock()

    import artenic_ai_platform_providers.infomaniak as info_mod

    monkeypatch.setattr(info_mod, "openstack", mock_openstack)


def _build_provider():
    from artenic_ai_platform_providers.infomaniak import InfomaniakProvider

    return InfomaniakProvider(
        auth_url="https://api.pub1.infomaniak.cloud/identity/v3",
        username="user",
        password="pass",
        project_id="proj-123",
        region="dc3-a",
        image_id="img-123",
        network_id="net-456",
    )


# ===================================================================
# Tests
# ===================================================================


class TestInfomaniakInit:
    def test_init_stores_params(self):
        p = _build_provider()
        assert p._auth_url == ("https://api.pub1.infomaniak.cloud/identity/v3")
        assert p._username == "user"
        assert p._region == "dc3-a"
        assert p._container_name == "artenic-training"


class TestInfomaniakProviderName:
    def test_provider_name(self):
        p = _build_provider()
        assert p.provider_name == "infomaniak"


class TestInfomaniakConnect:
    async def test_connect_creates_connection(self):
        p = _build_provider()
        await p._connect()
        assert p._conn is not None


class TestInfomaniakDisconnect:
    async def test_disconnect_clears_connection(self):
        p = _build_provider()
        await p._connect()
        assert p._conn is not None
        await p._disconnect()
        assert p._conn is None

    async def test_disconnect_noop_when_not_connected(self):
        p = _build_provider()
        await p._disconnect()
        assert p._conn is None


class TestInfomaniakListInstances:
    async def test_list_returns_instances(self):
        p = _build_provider()
        await p._connect()
        instances = await p._list_instances()
        assert len(instances) >= 1
        assert instances[0].name == "a2-ram4-disk20"
        assert instances[0].region == "dc3-a"

    async def test_list_gpu_only(self):
        p = _build_provider()
        await p._connect()

        gpu_flv = _mock_flavor(name="gpu-a100-x2", vcpus=16, ram=131072)
        p._conn.compute.flavors.return_value = [
            _mock_flavor(),
            gpu_flv,
        ]

        instances = await p._list_instances(gpu_only=True)
        assert len(instances) == 1
        assert instances[0].gpu_count == 2
        assert instances[0].gpu_type == "A100"

    async def test_list_custom_region(self):
        p = _build_provider()
        await p._connect()
        instances = await p._list_instances(region="dc5-b")
        assert instances[0].region == "dc5-b"


class TestInfomaniakUploadCode:
    async def test_upload_returns_swift_uri(self):
        p = _build_provider()
        await p._connect()
        spec = _make_spec()

        with (
            patch.object(type(p), "_create_tarball", return_value="/tmp/f.tar.gz"),
            patch("builtins.open", MagicMock()),
            patch("os.path.exists", return_value=True),
            patch("os.remove"),
        ):
            uri = await p._upload_code(spec)

        assert uri.startswith("swift://artenic-training/code/")

    async def test_upload_cleans_tarball(self):
        p = _build_provider()
        await p._connect()
        spec = _make_spec()

        with (
            patch.object(type(p), "_create_tarball", return_value="/tmp/f.tar.gz"),
            patch("builtins.open", MagicMock()),
            patch("os.path.exists", return_value=True),
            patch("os.remove") as rm,
        ):
            await p._upload_code(spec)

        rm.assert_called_once_with("/tmp/f.tar.gz")


class TestInfomaniakProvisionAndStart:
    async def test_provision_returns_job_id(self):
        p = _build_provider()
        await p._connect()
        p._conn.compute.create_server.return_value = _mock_server()
        spec = _make_spec()

        job_id = await p._provision_and_start(spec)
        assert job_id.startswith("infomaniak-")
        assert job_id in p._jobs

    async def test_provision_raises_without_image(self):
        p = _build_provider()
        p._image_id = None
        await p._connect()
        spec = _make_spec(config={})

        with pytest.raises(ValueError, match="image_id"):
            await p._provision_and_start(spec)

    async def test_provision_includes_network(self):
        p = _build_provider()
        await p._connect()
        p._conn.compute.create_server.return_value = _mock_server()
        spec = _make_spec(config={"image_id": "i-1", "network_id": "n-1"})

        await p._provision_and_start(spec)
        kw = p._conn.compute.create_server.call_args
        assert "networks" in kw.kwargs


class TestInfomaniakPollProvider:
    async def test_poll_unknown_job(self):
        p = _build_provider()
        await p._connect()
        r = await p._poll_provider("no-such-job")
        assert r.status == JobStatus.FAILED
        assert "Unknown job" in r.error

    async def test_poll_active(self):
        p = _build_provider()
        await p._connect()
        p._conn.compute.create_server.return_value = _mock_server()
        jid = await p._provision_and_start(_make_spec())

        p._conn.compute.get_server.return_value = _mock_server(status="ACTIVE")
        r = await p._poll_provider(jid)
        assert r.status == JobStatus.RUNNING

    async def test_poll_build(self):
        p = _build_provider()
        await p._connect()
        p._conn.compute.create_server.return_value = _mock_server()
        jid = await p._provision_and_start(_make_spec())

        p._conn.compute.get_server.return_value = _mock_server(status="BUILD")
        r = await p._poll_provider(jid)
        assert r.status == JobStatus.PENDING

    async def test_poll_stopped(self):
        p = _build_provider()
        await p._connect()
        p._conn.compute.create_server.return_value = _mock_server()
        jid = await p._provision_and_start(_make_spec())

        p._conn.compute.get_server.return_value = _mock_server(status="STOPPED")
        r = await p._poll_provider(jid)
        assert r.status == JobStatus.COMPLETED

    async def test_poll_error(self):
        p = _build_provider()
        await p._connect()
        p._conn.compute.create_server.return_value = _mock_server()
        jid = await p._provision_and_start(_make_spec())

        p._conn.compute.get_server.return_value = _mock_server(status="ERROR")
        r = await p._poll_provider(jid)
        assert r.status == JobStatus.FAILED

    async def test_poll_api_failure(self):
        p = _build_provider()
        await p._connect()
        p._conn.compute.create_server.return_value = _mock_server()
        jid = await p._provision_and_start(_make_spec())

        p._conn.compute.get_server.side_effect = RuntimeError("boom")
        r = await p._poll_provider(jid)
        assert r.status == JobStatus.FAILED
        assert "Cannot reach" in r.error


class TestInfomaniakCollectArtifacts:
    async def test_collect_unknown_job(self):
        p = _build_provider()
        await p._connect()
        st = CloudJobStatus(provider_job_id="x", status=JobStatus.COMPLETED)
        assert await p._collect_artifacts("x", st) is None

    async def test_collect_download_error(self):
        p = _build_provider()
        await p._connect()
        p._conn.compute.create_server.return_value = _mock_server()
        jid = await p._provision_and_start(_make_spec())

        p._conn.object_store.download_object.side_effect = RuntimeError("404")
        st = CloudJobStatus(provider_job_id=jid, status=JobStatus.COMPLETED)
        assert await p._collect_artifacts(jid, st) is None


class TestInfomaniakCleanupCompute:
    async def test_cleanup_deletes_server(self):
        p = _build_provider()
        await p._connect()
        p._conn.compute.create_server.return_value = _mock_server()
        jid = await p._provision_and_start(_make_spec())

        await p._cleanup_compute(jid)
        p._conn.compute.delete_server.assert_called_once()
        assert jid not in p._jobs

    async def test_cleanup_unknown_job(self):
        p = _build_provider()
        await p._connect()
        await p._cleanup_compute("nope")  # no raise

    async def test_cleanup_delete_failure(self):
        p = _build_provider()
        await p._connect()
        p._conn.compute.create_server.return_value = _mock_server()
        jid = await p._provision_and_start(_make_spec())

        p._conn.compute.delete_server.side_effect = RuntimeError("fail")
        with pytest.raises(RuntimeError):
            await p._cleanup_compute(jid)


class TestInfomaniakCancelJob:
    async def test_cancel_stops_server(self):
        p = _build_provider()
        await p._connect()
        p._conn.compute.create_server.return_value = _mock_server()
        jid = await p._provision_and_start(_make_spec())

        await p._cancel_provider_job(jid)
        p._conn.compute.stop_server.assert_called_once()

    async def test_cancel_unknown_job(self):
        p = _build_provider()
        await p._connect()
        await p._cancel_provider_job("nope")

    async def test_cancel_stop_failure_swallowed(self):
        p = _build_provider()
        await p._connect()
        p._conn.compute.create_server.return_value = _mock_server()
        jid = await p._provision_and_start(_make_spec())

        p._conn.compute.stop_server.side_effect = RuntimeError("err")
        await p._cancel_provider_job(jid)  # must not raise


class TestInfomaniakEstimateCost:
    def test_with_hourly_price(self):
        from artenic_ai_platform_providers.infomaniak import (
            _InfomaniakJobState,
        )

        p = _build_provider()
        state = _InfomaniakJobState(
            server_id="s-1",
            created_at=time.time(),
            spec=_make_spec(),
            hourly_price=2.0,
        )
        cost = p._estimate_cost(state, 1800.0)
        assert cost == 1.0

    def test_without_hourly_price(self):
        from artenic_ai_platform_providers.infomaniak import (
            _InfomaniakJobState,
        )

        p = _build_provider()
        state = _InfomaniakJobState(
            server_id="s-1",
            created_at=time.time(),
            spec=_make_spec(),
        )
        assert p._estimate_cost(state, 3600.0) is None


# ===================================================================
# Additional tests for 100% coverage
# ===================================================================


class TestInfomaniakPollUnknownStatus:
    """Cover line 415 — the else branch for unknown server status."""

    async def test_poll_unknown_status_maps_to_running(self):
        p = _build_provider()
        await p._connect()
        p._conn.compute.create_server.return_value = _mock_server()
        jid = await p._provision_and_start(_make_spec())

        p._conn.compute.get_server.return_value = _mock_server(status="REBOOT")
        r = await p._poll_provider(jid)
        assert r.status == JobStatus.RUNNING


class TestInfomaniakCollectArtifactsSuccess:
    """Cover lines 464-470 — successful artifact download, write and extract."""

    async def test_collect_artifacts_success(self):
        p = _build_provider()
        await p._connect()
        p._conn.compute.create_server.return_value = _mock_server()
        jid = await p._provision_and_start(_make_spec())

        # download_object returns bytes
        p._conn.object_store.download_object.return_value = b"fake-tar-data"

        with (
            patch("builtins.open", MagicMock()),
            patch("os.makedirs"),
            patch.object(type(p), "_extract_tarball"),
        ):
            st = CloudJobStatus(provider_job_id=jid, status=JobStatus.COMPLETED)
            result = await p._collect_artifacts(jid, st)

        assert result is not None
        assert "artenic-artifacts" in result


class TestInfomaniakBuildUserDataEnvVars:
    """Cover lines 577-580 — env vars block in _build_user_data."""

    def test_build_user_data_with_env_vars(self):
        p = _build_provider()
        spec = _make_spec(
            config={
                "image_id": "img-123",
                "code_path": ".",
                "env": {"BATCH_SIZE": "32", "LR": "0.001"},
            }
        )
        user_data = p._build_user_data(spec, "infomaniak-test123")
        assert "export" in user_data
        assert "BATCH_SIZE" in user_data
        assert "32" in user_data
        assert "LR" in user_data


class TestInfomaniakCreateTarball:
    """Cover lines 594-604 — _create_tarball for files and directories."""

    def test_create_tarball_from_file(self, tmp_path):
        from artenic_ai_platform_providers.infomaniak import (
            InfomaniakProvider,
        )

        test_file = tmp_path / "train.py"
        test_file.write_text("print('hello')")
        result = InfomaniakProvider._create_tarball(str(test_file))
        assert result.endswith(".tar.gz")
        import os

        os.remove(result)

    def test_create_tarball_from_directory(self, tmp_path):
        from artenic_ai_platform_providers.infomaniak import (
            InfomaniakProvider,
        )

        (tmp_path / "file1.py").write_text("a = 1")
        (tmp_path / "file2.py").write_text("b = 2")
        result = InfomaniakProvider._create_tarball(str(tmp_path))
        assert result.endswith(".tar.gz")
        import os

        os.remove(result)


class TestInfomaniakExtractTarball:
    """Cover lines 609-610 — _extract_tarball."""

    def test_extract_tarball(self, tmp_path):
        import tarfile

        from artenic_ai_platform_providers.infomaniak import (
            InfomaniakProvider,
        )

        src_file = tmp_path / "data.txt"
        src_file.write_text("test data")
        tar_path = str(tmp_path / "archive.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(str(src_file), arcname="data.txt")

        dest_dir = str(tmp_path / "extracted")
        import os

        os.makedirs(dest_dir)
        InfomaniakProvider._extract_tarball(tar_path, dest_dir)
        assert os.path.exists(os.path.join(dest_dir, "data.txt"))


class TestInfomaniakGPUParsingEdgeCases:
    """Cover lines 668-670, 679 — GPU parsing edge cases."""

    def test_parse_gpu_count_multi_gpu(self):
        from artenic_ai_platform_providers.infomaniak import (
            _parse_gpu_count_from_name,
        )

        assert _parse_gpu_count_from_name("gpu-a100-x4") == 4
        assert _parse_gpu_count_from_name("gpu-v100-x2") == 2

    def test_parse_gpu_count_gpu_no_suffix_returns_one(self):
        from artenic_ai_platform_providers.infomaniak import (
            _parse_gpu_count_from_name,
        )

        # Line 669: GPU name without -xN suffix returns 1
        assert _parse_gpu_count_from_name("gpu-generic") == 1
        assert _parse_gpu_count_from_name("gpu-custom-10g") == 1

    def test_parse_gpu_count_non_gpu_returns_zero(self):
        from artenic_ai_platform_providers.infomaniak import (
            _parse_gpu_count_from_name,
        )

        # Line 670: non-GPU name returns 0
        assert _parse_gpu_count_from_name("a2-ram4-disk20") == 0
        assert _parse_gpu_count_from_name("standard-2") == 0

    def test_parse_gpu_type_generic_fallback(self):
        from artenic_ai_platform_providers.infomaniak import (
            _parse_gpu_type_from_name,
        )

        # Line 679: unknown GPU model returns "GPU"
        assert _parse_gpu_type_from_name("gpu-unknown-xyz") == "GPU"
        assert _parse_gpu_type_from_name("gpu-custom-10g") == "GPU"
