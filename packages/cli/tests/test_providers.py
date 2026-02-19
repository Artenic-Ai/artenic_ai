"""Tests for provider commands."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from artenic_ai_cli.main import cli

if TYPE_CHECKING:
    from click.testing import CliRunner


class TestProviderList:
    def test_table(self, runner: CliRunner, patch_run_async: Any) -> None:
        data = [
            {
                "id": "ovh",
                "display_name": "OVH Public Cloud",
                "status": "unconfigured",
                "enabled": False,
            },
        ]
        with patch_run_async(return_value=data):
            result = runner.invoke(cli, ["provider", "list"])
            assert result.exit_code == 0
            assert "OVH" in result.output

    def test_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        data = [{"id": "ovh", "display_name": "OVH Public Cloud"}]
        with patch_run_async(return_value=data):
            result = runner.invoke(cli, ["--json", "provider", "list"])
            assert result.exit_code == 0
            assert '"ovh"' in result.output

    def test_empty_list(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value=[]):
            result = runner.invoke(cli, ["provider", "list"])
            assert result.exit_code == 0


class TestProviderGet:
    def test_get(self, runner: CliRunner, patch_run_async: Any) -> None:
        data = {
            "id": "ovh",
            "display_name": "OVH Public Cloud",
            "connector_type": "openstack",
            "status": "unconfigured",
            "has_credentials": False,
        }
        with patch_run_async(return_value=data):
            result = runner.invoke(cli, ["provider", "get", "ovh"])
            assert result.exit_code == 0
            assert "OVH" in result.output

    def test_get_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        data = {"id": "ovh", "display_name": "OVH Public Cloud"}
        with patch_run_async(return_value=data):
            result = runner.invoke(cli, ["--json", "provider", "get", "ovh"])
            assert result.exit_code == 0
            assert '"ovh"' in result.output


class TestProviderConfigure:
    def test_configure(self, runner: CliRunner, patch_run_async: Any) -> None:
        creds = json.dumps({"auth_url": "https://auth.cloud.ovh.net/v3", "username": "u"})
        with patch_run_async(return_value={"id": "ovh", "status": "configured"}):
            result = runner.invoke(cli, ["provider", "configure", "ovh", "--credentials", creds])
            assert result.exit_code == 0
            assert "configured" in result.stderr

    def test_configure_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        creds = json.dumps({"auth_url": "https://auth.cloud.ovh.net/v3"})
        with patch_run_async(return_value={"id": "ovh", "status": "configured"}):
            result = runner.invoke(
                cli, ["--json", "provider", "configure", "ovh", "--credentials", creds]
            )
            assert result.exit_code == 0
            assert '"configured"' in result.output

    def test_configure_with_config(self, runner: CliRunner, patch_run_async: Any) -> None:
        creds = json.dumps({"auth_url": "https://auth.cloud.ovh.net/v3"})
        cfg = json.dumps({"region": "GRA11"})
        with patch_run_async(return_value={"id": "ovh", "status": "configured"}):
            result = runner.invoke(
                cli,
                ["provider", "configure", "ovh", "--credentials", creds, "--config", cfg],
            )
            assert result.exit_code == 0


class TestProviderConfigureInvalidJson:
    def test_invalid_credentials(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["provider", "configure", "ovh", "--credentials", "not-json"])
        assert result.exit_code != 0
        assert "Invalid JSON" in result.output

    def test_invalid_config(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli,
            [
                "provider",
                "configure",
                "ovh",
                "--credentials",
                '{"key": "val"}',
                "--config",
                "{bad",
            ],
        )
        assert result.exit_code != 0
        assert "Invalid JSON" in result.output


class TestProviderEnable:
    def test_enable(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"id": "ovh", "enabled": True}):
            result = runner.invoke(cli, ["provider", "enable", "ovh"])
            assert result.exit_code == 0
            assert "enabled" in result.stderr

    def test_enable_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"id": "ovh", "enabled": True}):
            result = runner.invoke(cli, ["--json", "provider", "enable", "ovh"])
            assert result.exit_code == 0
            assert '"enabled"' in result.output


class TestProviderDisable:
    def test_disable(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"id": "ovh", "enabled": False}):
            result = runner.invoke(cli, ["provider", "disable", "ovh"])
            assert result.exit_code == 0
            assert "disabled" in result.stderr

    def test_disable_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"id": "ovh", "enabled": False}):
            result = runner.invoke(cli, ["--json", "provider", "disable", "ovh"])
            assert result.exit_code == 0
            assert '"enabled"' in result.output


class TestProviderTest:
    def test_success(self, runner: CliRunner, patch_run_async: Any) -> None:
        data = {"success": True, "message": "Connected — 10 flavors", "latency_ms": 42.0}
        with patch_run_async(return_value=data):
            result = runner.invoke(cli, ["provider", "test", "ovh"])
            assert result.exit_code == 0
            assert "Connection OK" in result.stderr

    def test_failure_exit_code_1(self, runner: CliRunner, patch_run_async: Any) -> None:
        data = {"success": False, "message": "Auth failed", "latency_ms": 100.0}
        with patch_run_async(return_value=data):
            result = runner.invoke(cli, ["provider", "test", "ovh"])
            assert result.exit_code == 1
            assert "failed" in result.output

    def test_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        data = {"success": True, "message": "OK", "latency_ms": 42.0}
        with patch_run_async(return_value=data):
            result = runner.invoke(cli, ["--json", "provider", "test", "ovh"])
            assert result.exit_code == 0
            assert '"success"' in result.output


class TestProviderDelete:
    def test_delete(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value=None):
            result = runner.invoke(cli, ["provider", "delete", "ovh"])
            assert result.exit_code == 0
            assert "removed" in result.stderr

    def test_delete_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value=None):
            result = runner.invoke(cli, ["--json", "provider", "delete", "ovh"])
            assert result.exit_code == 0
            assert '"deleted"' in result.output


class TestProviderStorage:
    def test_storage(self, runner: CliRunner, patch_run_async: Any) -> None:
        data = [
            {
                "provider_id": "ovh",
                "name": "my-container",
                "type": "object_storage",
                "region": "GRA11",
            },
        ]
        with patch_run_async(return_value=data):
            result = runner.invoke(cli, ["provider", "storage", "ovh"])
            assert result.exit_code == 0
            assert "my-container" in result.output

    def test_storage_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        data = [{"provider_id": "ovh", "name": "c1"}]
        with patch_run_async(return_value=data):
            result = runner.invoke(cli, ["--json", "provider", "storage", "ovh"])
            assert result.exit_code == 0
            assert '"c1"' in result.output

    def test_storage_empty(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value=[]):
            result = runner.invoke(cli, ["provider", "storage", "ovh"])
            assert result.exit_code == 0


class TestProviderCompute:
    def test_compute(self, runner: CliRunner, patch_run_async: Any) -> None:
        data = [
            {
                "provider_id": "ovh",
                "name": "b2-30",
                "vcpus": 8,
                "memory_gb": 30.0,
                "gpu_type": None,
                "gpu_count": 0,
            },
        ]
        with patch_run_async(return_value=data):
            result = runner.invoke(cli, ["provider", "compute", "ovh"])
            assert result.exit_code == 0
            assert "b2-30" in result.output

    def test_compute_gpu_only(self, runner: CliRunner, patch_run_async: Any) -> None:
        data = [
            {
                "provider_id": "ovh",
                "name": "gpu-a100",
                "vcpus": 12,
                "memory_gb": 120.0,
                "gpu_type": "A100",
                "gpu_count": 1,
            },
        ]
        with patch_run_async(return_value=data):
            result = runner.invoke(cli, ["provider", "compute", "ovh", "--gpu-only"])
            assert result.exit_code == 0
            assert "A100" in result.output

    def test_compute_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        data = [{"provider_id": "ovh", "name": "b2-30"}]
        with patch_run_async(return_value=data):
            result = runner.invoke(cli, ["--json", "provider", "compute", "ovh"])
            assert result.exit_code == 0
            assert '"b2-30"' in result.output


class TestProviderRegions:
    def test_regions(self, runner: CliRunner, patch_run_async: Any) -> None:
        data = [
            {"id": "GRA11", "name": "Gravelines", "provider_id": "ovh"},
            {"id": "SBG5", "name": "Strasbourg", "provider_id": "ovh"},
        ]
        with patch_run_async(return_value=data):
            result = runner.invoke(cli, ["provider", "regions", "ovh"])
            assert result.exit_code == 0
            assert "Gravelines" in result.output

    def test_regions_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        data = [{"id": "GRA11", "name": "Gravelines", "provider_id": "ovh"}]
        with patch_run_async(return_value=data):
            result = runner.invoke(cli, ["--json", "provider", "regions", "ovh"])
            assert result.exit_code == 0
            assert '"GRA11"' in result.output

    def test_regions_empty(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value=[]):
            result = runner.invoke(cli, ["provider", "regions", "ovh"])
            assert result.exit_code == 0
