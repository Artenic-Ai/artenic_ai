"""Tests for A/B test commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from artenic_ai_cli.main import cli

if TYPE_CHECKING:
    from click.testing import CliRunner


class TestAbTestCreate:
    def test_create(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"test_id": "t1"}):
            result = runner.invoke(
                cli,
                [
                    "ab-test",
                    "create",
                    "--name",
                    "test1",
                    "--service",
                    "svc",
                    "--variants",
                    '[{"name": "A"}, {"name": "B"}]',
                    "--primary-metric",
                    "accuracy",
                ],
            )
            assert result.exit_code == 0
            assert "t1" in result.stderr

    def test_create_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"test_id": "t1"}):
            result = runner.invoke(
                cli,
                [
                    "--json",
                    "ab-test",
                    "create",
                    "--name",
                    "t",
                    "--service",
                    "s",
                    "--variants",
                    '[{"name": "A"}]',
                    "--primary-metric",
                    "acc",
                ],
            )
            assert result.exit_code == 0
            assert '"test_id"' in result.output

    def test_create_with_min_samples(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"test_id": "t2"}):
            result = runner.invoke(
                cli,
                [
                    "ab-test",
                    "create",
                    "--name",
                    "t",
                    "--service",
                    "s",
                    "--variants",
                    "[]",
                    "--primary-metric",
                    "p",
                    "--min-samples",
                    "500",
                ],
            )
            assert result.exit_code == 0


class TestAbTestList:
    def test_list(self, runner: CliRunner, patch_run_async: Any) -> None:
        data = [
            {
                "id": "t1",
                "name": "test1",
                "service": "s",
                "status": "running",
                "primary_metric": "acc",
            },
        ]
        with patch_run_async(return_value=data):
            result = runner.invoke(cli, ["ab-test", "list"])
            assert result.exit_code == 0
            assert "test1" in result.output

    def test_list_filters(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value=[]):
            result = runner.invoke(
                cli,
                [
                    "ab-test",
                    "list",
                    "--service",
                    "s",
                    "--status",
                    "completed",
                    "--limit",
                    "10",
                    "--offset",
                    "5",
                ],
            )
            assert result.exit_code == 0


class TestAbTestGet:
    def test_get(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"id": "t1", "name": "test1"}):
            result = runner.invoke(cli, ["ab-test", "get", "t1"])
            assert result.exit_code == 0
            assert "test1" in result.output


class TestAbTestResults:
    def test_results(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"winner": "A", "p_value": 0.01}):
            result = runner.invoke(cli, ["ab-test", "results", "t1"])
            assert result.exit_code == 0


class TestAbTestConclude:
    def test_conclude(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"status": "concluded"}):
            result = runner.invoke(
                cli,
                ["ab-test", "conclude", "t1", "--winner", "A", "--reason", "better"],
            )
            assert result.exit_code == 0
            assert "concluded" in result.stderr

    def test_conclude_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"status": "concluded"}):
            result = runner.invoke(cli, ["--json", "ab-test", "conclude", "t1"])
            assert result.exit_code == 0

    def test_conclude_no_options(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"status": "concluded"}):
            result = runner.invoke(cli, ["ab-test", "conclude", "t1"])
            assert result.exit_code == 0


class TestAbTestPause:
    def test_pause(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={}):
            result = runner.invoke(cli, ["ab-test", "pause", "t1"])
            assert result.exit_code == 0
            assert "paused" in result.stderr

    def test_pause_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={}):
            result = runner.invoke(cli, ["--json", "ab-test", "pause", "t1"])
            assert result.exit_code == 0
            assert '"paused"' in result.output


class TestAbTestResume:
    def test_resume(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={}):
            result = runner.invoke(cli, ["ab-test", "resume", "t1"])
            assert result.exit_code == 0
            assert "resumed" in result.stderr

    def test_resume_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={}):
            result = runner.invoke(cli, ["--json", "ab-test", "resume", "t1"])
            assert result.exit_code == 0
            assert '"running"' in result.output


class TestInvalidJson:
    def test_create_invalid_variants(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli,
            [
                "ab-test",
                "create",
                "--name",
                "t",
                "--service",
                "s",
                "--variants",
                "not-json",
                "--primary-metric",
                "acc",
            ],
        )
        assert result.exit_code != 0
        assert "Invalid JSON" in result.output
