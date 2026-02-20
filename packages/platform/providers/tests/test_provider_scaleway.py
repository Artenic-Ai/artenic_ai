"""Tests for artenic_ai_platform_providers.scaleway — ScalewayProvider."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

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
        "provider": "scaleway",
        "config": {"image_id": "img-abc"},
    }
    defaults.update(overrides)
    return TrainingSpec(**defaults)


def _mock_response(
    status_code=200,
    json_data=None,
    text="",
):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = RuntimeError(f"HTTP {status_code}")
    return resp


@pytest.fixture(autouse=True)
def _patch_httpx(monkeypatch):
    """Patch the httpx SDK references on the scaleway module."""
    mock_httpx = MagicMock()
    mock_httpx.Timeout.return_value = MagicMock()

    # Build a mock AsyncClient that supports async context manager
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=_mock_response(200, {"servers": []}))
    mock_client.post = AsyncMock(return_value=_mock_response(201, {}))
    mock_client.delete = AsyncMock(return_value=_mock_response(204))
    mock_client.aclose = AsyncMock()

    mock_httpx.AsyncClient.return_value = mock_client

    import artenic_ai_platform_providers.scaleway as scw_mod

    monkeypatch.setattr(scw_mod, "_HTTPX_AVAILABLE", True)
    monkeypatch.setattr(scw_mod, "httpx", mock_httpx)


def _build_provider(**kwargs):
    from artenic_ai_platform_providers.scaleway import ScalewayProvider

    defaults = {
        "secret_key": "scw-secret",
        "access_key": "scw-access",
        "project_id": "proj-12345678",
        "zone": "fr-par-1",
        "image_id": "img-abc",
    }
    defaults.update(kwargs)
    return ScalewayProvider(**defaults)


# ===================================================================
# Tests
# ===================================================================


class TestScalewayInit:
    def test_init_stores_params(self):
        p = _build_provider()
        assert p._secret_key == "scw-secret"
        assert p._access_key == "scw-access"
        assert p._project_id == "proj-12345678"
        assert p._zone == "fr-par-1"
        assert p._default_instance_type == "DEV1-S"

    def test_init_raises_without_httpx(self, monkeypatch):
        import artenic_ai_platform_providers.scaleway as mod

        monkeypatch.setattr(mod, "_HTTPX_AVAILABLE", False)
        with pytest.raises(ImportError, match="httpx"):
            _build_provider()


class TestScalewayProviderName:
    def test_provider_name(self):
        p = _build_provider()
        assert p.provider_name == "scaleway"


class TestScalewayConnect:
    async def test_connect_creates_client(self):
        p = _build_provider()
        await p._connect()
        assert p._client is not None

    async def test_connect_verifies_connectivity(self):
        p = _build_provider()
        await p._connect()
        p._client.get.assert_called_once()


class TestScalewayDisconnect:
    async def test_disconnect_closes_client(self):
        p = _build_provider()
        await p._connect()
        client = p._client
        await p._disconnect()
        assert p._client is None
        client.aclose.assert_called_once()

    async def test_disconnect_noop_when_not_connected(self):
        p = _build_provider()
        await p._disconnect()
        assert p._client is None


class TestScalewayListInstances:
    async def test_list_returns_instances(self):
        p = _build_provider()
        await p._connect()

        p._client.get = AsyncMock(
            return_value=_mock_response(
                200,
                {
                    "servers": {
                        "DEV1-S": {
                            "ncpus": 2,
                            "ram": 2147483648,
                            "hourly_price": 0.01,
                        },
                    }
                },
            )
        )

        instances = await p._list_instances()
        assert len(instances) == 1
        assert instances[0].name == "DEV1-S"
        assert instances[0].vcpus == 2
        assert instances[0].price_per_hour_eur == 0.01

    async def test_list_gpu_only(self):
        p = _build_provider()
        await p._connect()

        p._client.get = AsyncMock(
            return_value=_mock_response(
                200,
                {
                    "servers": {
                        "DEV1-S": {
                            "ncpus": 2,
                            "ram": 2147483648,
                            "hourly_price": 0.01,
                        },
                        "H100-1-80G": {
                            "ncpus": 16,
                            "ram": 137438953472,
                            "hourly_price": 3.5,
                        },
                    }
                },
            )
        )

        instances = await p._list_instances(gpu_only=True)
        assert len(instances) == 1
        assert instances[0].name == "H100-1-80G"
        assert instances[0].gpu_type == "H100"
        assert instances[0].gpu_count >= 1

    async def test_list_custom_zone(self):
        p = _build_provider()
        await p._connect()

        p._client.get = AsyncMock(return_value=_mock_response(200, {"servers": {}}))

        instances = await p._list_instances(region="nl-ams-1")
        assert instances == []
        call_args = p._client.get.call_args
        assert "nl-ams-1" in call_args.args[0]


class TestScalewayUploadCode:
    async def test_upload_no_code_path(self):
        p = _build_provider()
        await p._connect()
        spec = _make_spec(config={"image_id": "img-abc"})

        uri = await p._upload_code(spec)
        assert uri.startswith("image://")

    async def test_upload_with_code_path(self):
        p = _build_provider()
        await p._connect()

        import artenic_ai_platform_providers.scaleway as scw_mod

        mock_httpx = scw_mod.httpx

        # Mock the inner AsyncClient used for S3 upload
        inner_client = AsyncMock()
        inner_client.get = AsyncMock(return_value=_mock_response(200, {"servers": []}))
        inner_client.put = AsyncMock(return_value=_mock_response(200))
        inner_client.aclose = AsyncMock()
        inner_client.__aenter__ = AsyncMock(return_value=inner_client)
        inner_client.__aexit__ = AsyncMock(return_value=False)
        mock_httpx.AsyncClient.return_value = inner_client
        mock_httpx.Timeout.return_value = MagicMock()

        spec = _make_spec(config={"image_id": "img-abc", "code_path": __file__})

        # Reconnect with fresh mock
        await p._connect()
        uri = await p._upload_code(spec)
        assert uri.startswith("s3://")

    async def test_upload_failure_raises(self):
        p = _build_provider()
        await p._connect()

        import artenic_ai_platform_providers.scaleway as scw_mod

        mock_httpx = scw_mod.httpx

        inner_client = AsyncMock()
        inner_client.get = AsyncMock(return_value=_mock_response(200, {"servers": []}))
        inner_client.put = AsyncMock(side_effect=RuntimeError("upload failed"))
        inner_client.__aenter__ = AsyncMock(return_value=inner_client)
        inner_client.__aexit__ = AsyncMock(return_value=False)
        mock_httpx.AsyncClient.return_value = inner_client
        mock_httpx.Timeout.return_value = MagicMock()

        spec = _make_spec(config={"image_id": "img-abc", "code_path": __file__})
        await p._connect()

        with pytest.raises(RuntimeError, match="Failed to upload"):
            await p._upload_code(spec)


class TestScalewayProvisionAndStart:
    async def test_provision_returns_job_id(self):
        p = _build_provider()
        await p._connect()

        p._client.post = AsyncMock(
            side_effect=[
                # create_server response
                _mock_response(
                    201,
                    {
                        "server": {
                            "id": "srv-001",
                            "name": "artenic-scw-abc",
                            "state": "stopped",
                        }
                    },
                ),
                # power-on response
                _mock_response(202, {}),
            ]
        )

        spec = _make_spec()
        job_id = await p._provision_and_start(spec)
        assert job_id.startswith("scw-")
        assert job_id in p._jobs

    async def test_provision_raises_without_image(self):
        p = _build_provider(image_id="")
        await p._connect()
        spec = _make_spec(config={})

        with pytest.raises(ValueError, match="image_id"):
            await p._provision_and_start(spec)

    async def test_provision_create_failure(self):
        p = _build_provider()
        await p._connect()

        p._client.post = AsyncMock(return_value=_mock_response(500, text="Internal error"))

        spec = _make_spec()
        with pytest.raises(RuntimeError, match="creation failed"):
            await p._provision_and_start(spec)

    async def test_provision_poweron_failure(self):
        p = _build_provider()
        await p._connect()

        p._client.post = AsyncMock(
            side_effect=[
                # create succeeds
                _mock_response(
                    201,
                    {
                        "server": {
                            "id": "srv-001",
                            "name": "test",
                            "state": "stopped",
                        }
                    },
                ),
                # power-on fails
                _mock_response(500, text="power-on failed"),
                # delete (from cleanup inside the method)
            ]
        )
        # Mock _delete_server to avoid issues
        p._delete_server = AsyncMock()

        spec = _make_spec()
        with pytest.raises(RuntimeError, match="power-on failed"):
            await p._provision_and_start(spec)

        p._delete_server.assert_called_once()


class TestScalewayPollProvider:
    async def test_poll_unknown_job(self):
        p = _build_provider()
        await p._connect()
        r = await p._poll_provider("no-job")
        assert r.status == JobStatus.FAILED
        assert "Unknown job" in r.error

    async def test_poll_running(self):
        p = _build_provider()
        await p._connect()

        # Provision a job
        p._client.post = AsyncMock(
            side_effect=[
                _mock_response(
                    201,
                    {
                        "server": {
                            "id": "srv-001",
                            "name": "test",
                            "state": "stopped",
                        }
                    },
                ),
                _mock_response(202, {}),
            ]
        )
        jid = await p._provision_and_start(_make_spec())

        # Poll
        p._client.get = AsyncMock(
            return_value=_mock_response(
                200,
                {"server": {"state": "running"}},
            )
        )
        r = await p._poll_provider(jid)
        assert r.status == JobStatus.RUNNING

    async def test_poll_stopped(self):
        p = _build_provider()
        await p._connect()

        p._client.post = AsyncMock(
            side_effect=[
                _mock_response(
                    201,
                    {
                        "server": {
                            "id": "srv-001",
                            "name": "test",
                            "state": "stopped",
                        }
                    },
                ),
                _mock_response(202, {}),
            ]
        )
        jid = await p._provision_and_start(_make_spec())

        p._client.get = AsyncMock(
            return_value=_mock_response(
                200,
                {"server": {"state": "stopped"}},
            )
        )
        r = await p._poll_provider(jid)
        assert r.status == JobStatus.COMPLETED

    async def test_poll_starting(self):
        p = _build_provider()
        await p._connect()

        p._client.post = AsyncMock(
            side_effect=[
                _mock_response(
                    201,
                    {
                        "server": {
                            "id": "srv-001",
                            "name": "test",
                            "state": "stopped",
                        }
                    },
                ),
                _mock_response(202, {}),
            ]
        )
        jid = await p._provision_and_start(_make_spec())

        p._client.get = AsyncMock(
            return_value=_mock_response(
                200,
                {"server": {"state": "starting"}},
            )
        )
        r = await p._poll_provider(jid)
        assert r.status == JobStatus.PENDING

    async def test_poll_locked(self):
        p = _build_provider()
        await p._connect()

        p._client.post = AsyncMock(
            side_effect=[
                _mock_response(
                    201,
                    {
                        "server": {
                            "id": "srv-001",
                            "name": "test",
                            "state": "stopped",
                        }
                    },
                ),
                _mock_response(202, {}),
            ]
        )
        jid = await p._provision_and_start(_make_spec())

        p._client.get = AsyncMock(
            return_value=_mock_response(
                200,
                {"server": {"state": "locked"}},
            )
        )
        r = await p._poll_provider(jid)
        assert r.status == JobStatus.FAILED

    async def test_poll_api_error(self):
        p = _build_provider()
        await p._connect()

        p._client.post = AsyncMock(
            side_effect=[
                _mock_response(
                    201,
                    {
                        "server": {
                            "id": "srv-001",
                            "name": "test",
                            "state": "stopped",
                        }
                    },
                ),
                _mock_response(202, {}),
            ]
        )
        jid = await p._provision_and_start(_make_spec())

        p._client.get = AsyncMock(side_effect=RuntimeError("boom"))
        r = await p._poll_provider(jid)
        assert r.status == JobStatus.FAILED
        assert "Cannot reach" in r.error


class TestScalewayCollectArtifacts:
    async def test_collect_unknown_job(self):
        p = _build_provider()
        await p._connect()
        st = CloudJobStatus(provider_job_id="x", status=JobStatus.COMPLETED)
        assert await p._collect_artifacts("x", st) is None

    async def test_collect_returns_s3_uri(self):
        p = _build_provider()
        await p._connect()

        p._client.post = AsyncMock(
            side_effect=[
                _mock_response(
                    201,
                    {
                        "server": {
                            "id": "srv-001",
                            "name": "test",
                            "state": "stopped",
                        }
                    },
                ),
                _mock_response(202, {}),
            ]
        )
        jid = await p._provision_and_start(_make_spec())

        st = CloudJobStatus(provider_job_id=jid, status=JobStatus.COMPLETED)
        uri = await p._collect_artifacts(jid, st)
        assert uri is not None
        assert uri.startswith("s3://")
        assert "artifacts" in uri


class TestScalewayCleanupCompute:
    async def test_cleanup_terminates_and_deletes(self):
        p = _build_provider()
        await p._connect()

        p._client.post = AsyncMock(
            side_effect=[
                _mock_response(
                    201,
                    {
                        "server": {
                            "id": "srv-001",
                            "name": "test",
                            "state": "stopped",
                        }
                    },
                ),
                _mock_response(202, {}),
            ]
        )
        jid = await p._provision_and_start(_make_spec())

        # Reset post mock for terminate
        p._client.post = AsyncMock(return_value=_mock_response(202, {}))
        p._client.delete = AsyncMock(return_value=_mock_response(204))

        await p._cleanup_compute(jid)
        assert jid not in p._jobs
        p._client.post.assert_called_once()
        p._client.delete.assert_called_once()

    async def test_cleanup_unknown_job(self):
        p = _build_provider()
        await p._connect()
        await p._cleanup_compute("nope")

    async def test_cleanup_terminate_failure_continues(self):
        p = _build_provider()
        await p._connect()

        p._client.post = AsyncMock(
            side_effect=[
                _mock_response(
                    201,
                    {
                        "server": {
                            "id": "srv-001",
                            "name": "test",
                            "state": "stopped",
                        }
                    },
                ),
                _mock_response(202, {}),
            ]
        )
        jid = await p._provision_and_start(_make_spec())

        # Terminate fails but delete succeeds
        p._client.post = AsyncMock(side_effect=RuntimeError("terminate failed"))
        p._client.delete = AsyncMock(return_value=_mock_response(204))

        await p._cleanup_compute(jid)
        assert jid not in p._jobs


class TestScalewayCancelJob:
    async def test_cancel_powers_off(self):
        p = _build_provider()
        await p._connect()

        p._client.post = AsyncMock(
            side_effect=[
                _mock_response(
                    201,
                    {
                        "server": {
                            "id": "srv-001",
                            "name": "test",
                            "state": "stopped",
                        }
                    },
                ),
                _mock_response(202, {}),
            ]
        )
        jid = await p._provision_and_start(_make_spec())

        p._client.post = AsyncMock(return_value=_mock_response(202, {}))

        await p._cancel_provider_job(jid)
        p._client.post.assert_called_once()
        call_args = p._client.post.call_args
        assert "poweroff" in str(call_args)

    async def test_cancel_unknown_job(self):
        p = _build_provider()
        await p._connect()
        await p._cancel_provider_job("nope")

    async def test_cancel_failure_raises(self):
        p = _build_provider()
        await p._connect()

        p._client.post = AsyncMock(
            side_effect=[
                _mock_response(
                    201,
                    {
                        "server": {
                            "id": "srv-001",
                            "name": "test",
                            "state": "stopped",
                        }
                    },
                ),
                _mock_response(202, {}),
            ]
        )
        jid = await p._provision_and_start(_make_spec())

        p._client.post = AsyncMock(side_effect=RuntimeError("power-off boom"))
        with pytest.raises(RuntimeError, match="power-off boom"):
            await p._cancel_provider_job(jid)


class TestScalewayEstimateCost:
    def test_estimate_with_price(self):
        from artenic_ai_platform_providers.scaleway import (
            _ScalewayJobState,
        )

        p = _build_provider()
        state = _ScalewayJobState(
            server_id="s-1",
            zone="fr-par-1",
            created_at=time.time(),
            spec=_make_spec(),
            commercial_type="DEV1-S",
            hourly_price=0.5,
        )
        cost = p._estimate_cost(state, 7200.0)
        assert cost == 1.0

    def test_estimate_without_price(self):
        from artenic_ai_platform_providers.scaleway import (
            _ScalewayJobState,
        )

        p = _build_provider()
        state = _ScalewayJobState(
            server_id="s-1",
            zone="fr-par-1",
            created_at=time.time(),
            spec=_make_spec(),
            commercial_type="DEV1-S",
        )
        assert p._estimate_cost(state, 3600.0) is None


class TestScalewayDeleteServer:
    async def test_delete_success(self):
        p = _build_provider()
        await p._connect()
        p._client.delete = AsyncMock(return_value=_mock_response(204))
        await p._delete_server("fr-par-1", "srv-001")
        p._client.delete.assert_called_once()

    async def test_delete_failure_raises(self):
        p = _build_provider()
        await p._connect()
        p._client.delete = AsyncMock(side_effect=RuntimeError("delete err"))
        with pytest.raises(RuntimeError, match="delete err"):
            await p._delete_server("fr-par-1", "srv-001")


# ===================================================================
# Additional tests for 100% coverage
# ===================================================================


class TestScalewayUploadCodeDirectory:
    """Cover lines 243-249 — uploading a directory (not a file) creates a tarball."""

    async def test_upload_directory_creates_tarball(self, tmp_path):
        p = _build_provider()
        await p._connect()

        import artenic_ai_platform_providers.scaleway as scw_mod

        mock_httpx = scw_mod.httpx

        # Create a directory with files
        code_dir = tmp_path / "training_code"
        code_dir.mkdir()
        (code_dir / "train.py").write_text("print('train')")
        (code_dir / "config.yaml").write_text("lr: 0.01")

        inner_client = AsyncMock()
        inner_client.get = AsyncMock(return_value=_mock_response(200, {"servers": []}))
        inner_client.put = AsyncMock(return_value=_mock_response(200))
        inner_client.aclose = AsyncMock()
        inner_client.__aenter__ = AsyncMock(return_value=inner_client)
        inner_client.__aexit__ = AsyncMock(return_value=False)
        mock_httpx.AsyncClient.return_value = inner_client
        mock_httpx.Timeout.return_value = MagicMock()

        spec = _make_spec(config={"image_id": "img-abc", "code_path": str(code_dir)})
        await p._connect()
        uri = await p._upload_code(spec)
        assert uri.startswith("s3://")

        # Verify a tarball was uploaded (the put call should have been made)
        inner_client.put.assert_called_once()


class TestScalewayCleanupTerminateWarning:
    """Cover line 473 — terminate action returns unexpected status code."""

    async def test_cleanup_terminate_unexpected_status(self):
        p = _build_provider()
        await p._connect()

        p._client.post = AsyncMock(
            side_effect=[
                _mock_response(
                    201,
                    {
                        "server": {
                            "id": "srv-001",
                            "name": "test",
                            "state": "stopped",
                        }
                    },
                ),
                _mock_response(202, {}),
            ]
        )
        jid = await p._provision_and_start(_make_spec())

        # Terminate returns 500 (unexpected) — should log warning but continue
        p._client.post = AsyncMock(return_value=_mock_response(500))
        p._client.delete = AsyncMock(return_value=_mock_response(204))

        await p._cleanup_compute(jid)
        assert jid not in p._jobs
        # Both terminate and delete should have been called
        p._client.post.assert_called_once()
        p._client.delete.assert_called_once()


class TestScalewayCancelPoweroffWarning:
    """Cover line 511 — poweroff returns unexpected status code."""

    async def test_cancel_poweroff_unexpected_status(self):
        p = _build_provider()
        await p._connect()

        p._client.post = AsyncMock(
            side_effect=[
                _mock_response(
                    201,
                    {
                        "server": {
                            "id": "srv-001",
                            "name": "test",
                            "state": "stopped",
                        }
                    },
                ),
                _mock_response(202, {}),
            ]
        )
        jid = await p._provision_and_start(_make_spec())

        # poweroff returns 500 — should log warning but not raise
        p._client.post = AsyncMock(return_value=_mock_response(500))

        # This should NOT raise, just log a warning
        await p._cancel_provider_job(jid)
        p._client.post.assert_called_once()


class TestScalewayDeleteServerWarning:
    """Cover line 539 — delete returns non-200/204 status code."""

    async def test_delete_server_unexpected_status(self):
        p = _build_provider()
        await p._connect()
        p._client.delete = AsyncMock(return_value=_mock_response(409))
        # Should not raise, just log a warning
        await p._delete_server("fr-par-1", "srv-001")
        p._client.delete.assert_called_once()


class TestScalewayDetectGPUEdgeCases:
    """Cover lines 614, 622, 645 — GPU detection edge cases."""

    def test_detect_gpu_from_api_response(self):
        from artenic_ai_platform_providers.scaleway import _detect_gpu

        # Line 614: GPU info in API response
        info = {"gpu": {"count": 2, "type": "A100"}}
        gpu_type, gpu_count = _detect_gpu("CUSTOM-INSTANCE", info)
        assert gpu_type == "A100"
        assert gpu_count == 2

    def test_detect_gpu_heuristic_keyword(self):
        from artenic_ai_platform_providers.scaleway import _detect_gpu

        # Line 622: heuristic keyword match
        gpu_type, gpu_count = _detect_gpu("my-gpu-instance", {})
        assert gpu_type == "GPU"
        assert gpu_count == 1

    def test_detect_gpu_render_keyword(self):
        from artenic_ai_platform_providers.scaleway import _detect_gpu

        # "render" keyword triggers heuristic
        gpu_type, gpu_count = _detect_gpu("render-custom", {})
        assert gpu_type == "GPU"
        assert gpu_count == 1

    def test_parse_gpu_count_default(self):
        from artenic_ai_platform_providers.scaleway import (
            _parse_gpu_count_from_name,
        )

        # Line 645: no digit parts found, returns 1
        assert _parse_gpu_count_from_name("GPU-S") == 1
        assert _parse_gpu_count_from_name("RENDER") == 1

    def test_parse_gpu_count_out_of_range_ignored(self):
        from artenic_ai_platform_providers.scaleway import (
            _parse_gpu_count_from_name,
        )

        # Digit >16 is ignored, falls through to default
        assert _parse_gpu_count_from_name("GPU-100-S") == 1

    def test_detect_gpu_api_response_zero_count(self):
        from artenic_ai_platform_providers.scaleway import _detect_gpu

        # GPU info present but count is 0 — should not match
        info = {"gpu": {"count": 0, "type": "A100"}}
        gpu_type, gpu_count = _detect_gpu("standard-instance", info)
        assert gpu_type is None
        assert gpu_count == 0
