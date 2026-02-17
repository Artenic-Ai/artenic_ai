"""Tests for training commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from artenic_ai_cli.main import cli

if TYPE_CHECKING:
    from click.testing import CliRunner


class TestTrainingDispatch:
    def test_dispatch(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"job_id": "j1"}):
            result = runner.invoke(
                cli,
                ["training", "dispatch", "--service", "svc", "--model", "m", "--provider", "local"],
            )
            assert result.exit_code == 0
            assert "j1" in result.stderr

    def test_dispatch_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"job_id": "j1"}):
            result = runner.invoke(
                cli,
                [
                    "--json",
                    "training",
                    "dispatch",
                    "--service",
                    "s",
                    "--model",
                    "m",
                    "--provider",
                    "p",
                ],
            )
            assert result.exit_code == 0
            assert '"job_id"' in result.output

    def test_dispatch_full_options(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"job_id": "j2"}):
            result = runner.invoke(
                cli,
                [
                    "training",
                    "dispatch",
                    "--service",
                    "s",
                    "--model",
                    "m",
                    "--provider",
                    "gcp",
                    "--config-json",
                    '{"lr": 0.01}',
                    "--instance-type",
                    "a100",
                    "--region",
                    "eu-west1",
                    "--spot",
                    "--max-hours",
                    "4",
                ],
            )
            assert result.exit_code == 0


class TestTrainingList:
    def test_list(self, runner: CliRunner, patch_run_async: Any) -> None:
        data = [{"job_id": "j1", "service": "s", "provider": "local", "status": "running"}]
        with patch_run_async(return_value=data):
            result = runner.invoke(cli, ["training", "list"])
            assert result.exit_code == 0
            assert "j1" in result.output

    def test_list_with_filters(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value=[]):
            result = runner.invoke(
                cli,
                [
                    "training",
                    "list",
                    "--service",
                    "s",
                    "--provider",
                    "gcp",
                    "--status",
                    "completed",
                    "--limit",
                    "10",
                    "--offset",
                    "5",
                ],
            )
            assert result.exit_code == 0


class TestTrainingStatus:
    def test_status(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"job_id": "j1", "status": "completed"}):
            result = runner.invoke(cli, ["training", "status", "j1"])
            assert result.exit_code == 0
            assert "completed" in result.output


class TestTrainingCancel:
    def test_cancel(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"status": "cancelled"}):
            result = runner.invoke(cli, ["training", "cancel", "j1"])
            assert result.exit_code == 0
            assert "cancelled" in result.stderr

    def test_cancel_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"status": "cancelled"}):
            result = runner.invoke(cli, ["--json", "training", "cancel", "j1"])
            assert result.exit_code == 0
            assert '"cancelled"' in result.output


class TestInvalidJson:
    def test_dispatch_invalid_config_json(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli,
            [
                "training",
                "dispatch",
                "--service",
                "s",
                "--model",
                "m",
                "--provider",
                "p",
                "--config-json",
                "not-json",
            ],
        )
        assert result.exit_code != 0
        assert "Invalid JSON" in result.output
