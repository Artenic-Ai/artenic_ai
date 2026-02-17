"""Tests for main CLI entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from artenic_ai_cli.main import cli

if TYPE_CHECKING:
    from click.testing import CliRunner


class TestCli:
    def test_version(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "artenic" in result.output

    def test_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Artenic AI Platform CLI" in result.output

    def test_global_json_flag(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"status": "healthy"}):
            result = runner.invoke(cli, ["--json", "health", "check"])
            assert result.exit_code == 0
            assert '"status"' in result.output

    def test_global_url_option(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"status": "ok"}):
            result = runner.invoke(cli, ["--url", "http://custom:1234", "health", "check"])
            assert result.exit_code == 0

    def test_subcommand_groups_registered(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        groups = [
            "health",
            "model",
            "training",
            "predict",
            "ensemble",
            "ab-test",
            "budget",
            "settings",
            "config",
        ]
        for group in groups:
            assert group in result.output


class TestHandleErrors:
    def test_auth_error(self, runner: CliRunner, patch_run_async: Any) -> None:
        from artenic_ai_sdk.exceptions import AuthenticationError

        with patch_run_async(side_effect=AuthenticationError("bad key")):
            result = runner.invoke(cli, ["health", "check"])
            assert result.exit_code == 1
            assert "Authentication failed" in result.stderr

    def test_rate_limit_error(self, runner: CliRunner, patch_run_async: Any) -> None:
        from artenic_ai_sdk.exceptions import RateLimitError

        with patch_run_async(side_effect=RateLimitError("limited", retry_after=10.0)):
            result = runner.invoke(cli, ["health", "check"])
            assert result.exit_code == 1
            assert "Rate limited" in result.stderr

    def test_rate_limit_no_retry_after(self, runner: CliRunner, patch_run_async: Any) -> None:
        from artenic_ai_sdk.exceptions import RateLimitError

        with patch_run_async(side_effect=RateLimitError("limited")):
            result = runner.invoke(cli, ["health", "check"])
            assert result.exit_code == 1
            assert "Rate limited" in result.stderr

    def test_service_unavailable(self, runner: CliRunner, patch_run_async: Any) -> None:
        from artenic_ai_sdk.exceptions import ServiceUnavailableError

        with patch_run_async(side_effect=ServiceUnavailableError("down")):
            result = runner.invoke(cli, ["health", "check"])
            assert result.exit_code == 1
            assert "down" in result.stderr

    def test_platform_error(self, runner: CliRunner, patch_run_async: Any) -> None:
        from artenic_ai_sdk.exceptions import PlatformError

        with patch_run_async(side_effect=PlatformError("500 internal")):
            result = runner.invoke(cli, ["health", "check"])
            assert result.exit_code == 1
            assert "500" in result.stderr

    def test_unexpected_error(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(side_effect=RuntimeError("unexpected")):
            result = runner.invoke(cli, ["health", "check"])
            assert result.exit_code == 1
            assert "Unexpected" in result.stderr

    def test_unexpected_error_sanitizes_credentials(
        self, runner: CliRunner, patch_run_async: Any
    ) -> None:
        with patch_run_async(side_effect=RuntimeError("Bearer sk-secret-key-123 in header")):
            result = runner.invoke(cli, ["health", "check"])
            assert result.exit_code == 1
            assert "sk-secret" not in result.stderr
            assert "An unexpected error occurred" in result.stderr

    def test_click_exit_propagates(self, runner: CliRunner, patch_run_async: Any) -> None:
        import click

        with patch_run_async(side_effect=click.exceptions.Exit(0)):
            result = runner.invoke(cli, ["health", "check"])
            assert result.exit_code == 0
