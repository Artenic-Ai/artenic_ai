"""Tests for ensemble commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from artenic_ai_cli.main import cli

if TYPE_CHECKING:
    from click.testing import CliRunner


class TestEnsembleCreate:
    def test_create(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"ensemble_id": "e1"}):
            result = runner.invoke(
                cli,
                [
                    "ensemble",
                    "create",
                    "--name",
                    "ens1",
                    "--service",
                    "svc",
                    "--strategy",
                    "weighted_average",
                    "--model-ids",
                    "m1,m2",
                ],
            )
            assert result.exit_code == 0
            assert "e1" in result.stderr

    def test_create_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"ensemble_id": "e1"}):
            result = runner.invoke(
                cli,
                [
                    "--json",
                    "ensemble",
                    "create",
                    "--name",
                    "e",
                    "--service",
                    "s",
                    "--strategy",
                    "stacking",
                    "--model-ids",
                    "m1",
                ],
            )
            assert result.exit_code == 0
            assert '"ensemble_id"' in result.output

    def test_create_full_options(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"ensemble_id": "e2"}):
            result = runner.invoke(
                cli,
                [
                    "ensemble",
                    "create",
                    "--name",
                    "e",
                    "--service",
                    "s",
                    "--strategy",
                    "stacking",
                    "--model-ids",
                    "m1,m2",
                    "--description",
                    "test",
                    "--strategy-config",
                    '{"k": 1}',
                ],
            )
            assert result.exit_code == 0


class TestEnsembleList:
    def test_list(self, runner: CliRunner, patch_run_async: Any) -> None:
        data = [
            {
                "id": "e1",
                "name": "ens",
                "service": "s",
                "strategy": "avg",
                "stage": "dev",
                "version": 1,
            },
        ]
        with patch_run_async(return_value=data):
            result = runner.invoke(cli, ["ensemble", "list"])
            assert result.exit_code == 0
            assert "ens" in result.output

    def test_list_filters(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value=[]):
            result = runner.invoke(
                cli,
                [
                    "ensemble",
                    "list",
                    "--service",
                    "s",
                    "--stage",
                    "prod",
                    "--limit",
                    "10",
                    "--offset",
                    "5",
                ],
            )
            assert result.exit_code == 0


class TestEnsembleGet:
    def test_get(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"id": "e1", "name": "ens"}):
            result = runner.invoke(cli, ["ensemble", "get", "e1"])
            assert result.exit_code == 0
            assert "ens" in result.output


class TestEnsembleUpdate:
    def test_update(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"id": "e1", "version": 2}):
            result = runner.invoke(cli, ["ensemble", "update", "e1", "--name", "new_name"])
            assert result.exit_code == 0
            assert "updated" in result.stderr

    def test_update_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"id": "e1"}):
            result = runner.invoke(
                cli, ["--json", "ensemble", "update", "e1", "--strategy", "voting"]
            )
            assert result.exit_code == 0

    def test_update_full_options(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"id": "e1"}):
            result = runner.invoke(
                cli,
                [
                    "ensemble",
                    "update",
                    "e1",
                    "--model-ids",
                    "m3,m4",
                    "--strategy-config",
                    '{"w": [0.5]}',
                    "--description",
                    "updated",
                    "--reason",
                    "rebalance",
                ],
            )
            assert result.exit_code == 0


class TestEnsembleTrain:
    def test_train(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"job_id": "j1"}):
            result = runner.invoke(cli, ["ensemble", "train", "e1", "--provider", "local"])
            assert result.exit_code == 0
            assert "j1" in result.stderr

    def test_train_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"job_id": "j1"}):
            result = runner.invoke(cli, ["--json", "ensemble", "train", "e1", "--provider", "gcp"])
            assert result.exit_code == 0

    def test_train_with_config(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"job_id": "j2"}):
            result = runner.invoke(
                cli,
                ["ensemble", "train", "e1", "--provider", "local", "--config-json", '{"lr": 0.1}'],
            )
            assert result.exit_code == 0


class TestEnsembleJobStatus:
    def test_job_status(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"status": "running"}):
            result = runner.invoke(cli, ["ensemble", "job-status", "e1", "j1"])
            assert result.exit_code == 0
            assert "running" in result.output


class TestEnsembleVersions:
    def test_versions(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value=[{"version": 1}, {"version": 2}]):
            result = runner.invoke(cli, ["ensemble", "versions", "e1"])
            assert result.exit_code == 0

    def test_versions_limit(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value=[]):
            result = runner.invoke(cli, ["ensemble", "versions", "e1", "--limit", "5"])
            assert result.exit_code == 0


class TestInvalidJson:
    def test_create_invalid_strategy_config(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli,
            [
                "ensemble",
                "create",
                "--name",
                "e",
                "--service",
                "s",
                "--strategy",
                "avg",
                "--model-ids",
                "m1",
                "--strategy-config",
                "not-json",
            ],
        )
        assert result.exit_code != 0
        assert "Invalid JSON" in result.output

    def test_update_invalid_strategy_config(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["ensemble", "update", "e1", "--strategy-config", "{bad"])
        assert result.exit_code != 0
        assert "Invalid JSON" in result.output

    def test_train_invalid_config_json(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli, ["ensemble", "train", "e1", "--provider", "p", "--config-json", "xyz"]
        )
        assert result.exit_code != 0
        assert "Invalid JSON" in result.output
