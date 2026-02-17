"""Tests for health commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from artenic_ai_cli.main import cli

if TYPE_CHECKING:
    from click.testing import CliRunner


class TestHealthCheck:
    def test_table_output(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"status": "healthy"}):
            result = runner.invoke(cli, ["health", "check"])
            assert result.exit_code == 0
            assert "healthy" in result.output

    def test_json_output(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"status": "healthy"}):
            result = runner.invoke(cli, ["--json", "health", "check"])
            assert result.exit_code == 0
            assert '"status"' in result.output


class TestHealthReady:
    def test_ready(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"status": "ready", "database": "connected"}):
            result = runner.invoke(cli, ["health", "ready"])
            assert result.exit_code == 0
            assert "ready" in result.output


class TestHealthDetailed:
    def test_detailed(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"status": "healthy", "components": {"db": "ok"}}):
            result = runner.invoke(cli, ["health", "detailed"])
            assert result.exit_code == 0
            assert "healthy" in result.output
