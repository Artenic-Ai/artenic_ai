"""Tests for settings commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from artenic_ai_cli.main import cli

if TYPE_CHECKING:
    from click.testing import CliRunner


class TestSettingsSchema:
    def test_schema_all(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"platform": [{"key": "debug"}]}):
            result = runner.invoke(cli, ["settings", "schema"])
            assert result.exit_code == 0

    def test_schema_scope(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value=[{"key": "debug"}]):
            result = runner.invoke(cli, ["settings", "schema", "platform"])
            assert result.exit_code == 0

    def test_schema_scope_section(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"key": "debug", "type": "bool"}):
            result = runner.invoke(cli, ["settings", "schema", "platform", "general"])
            assert result.exit_code == 0


class TestSettingsGet:
    def test_get_scope(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"general": {"debug": "false"}}):
            result = runner.invoke(cli, ["settings", "get", "platform"])
            assert result.exit_code == 0

    def test_get_scope_section(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"debug": "false"}):
            result = runner.invoke(cli, ["settings", "get", "platform", "general"])
            assert result.exit_code == 0


class TestSettingsUpdate:
    def test_update(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"debug": "true"}):
            result = runner.invoke(
                cli, ["settings", "update", "platform", "general", "--set", "debug=true"]
            )
            assert result.exit_code == 0
            assert "updated" in result.stderr

    def test_update_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"debug": "true"}):
            result = runner.invoke(
                cli, ["--json", "settings", "update", "platform", "general", "--set", "debug=true"]
            )
            assert result.exit_code == 0

    def test_update_multiple(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"a": "1", "b": "2"}):
            result = runner.invoke(
                cli, ["settings", "update", "platform", "general", "--set", "a=1", "--set", "b=2"]
            )
            assert result.exit_code == 0


class TestSettingsAudit:
    def test_audit(self, runner: CliRunner, patch_run_async: Any) -> None:
        data = [{"timestamp": "2026-01-01", "action": "update", "scope": "platform"}]
        with patch_run_async(return_value=data):
            result = runner.invoke(cli, ["settings", "audit"])
            assert result.exit_code == 0

    def test_audit_with_options(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value=[]):
            result = runner.invoke(cli, ["settings", "audit", "--limit", "10", "--offset", "5"])
            assert result.exit_code == 0


class TestInvalidSetFormat:
    def test_update_missing_equals(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli, ["settings", "update", "platform", "general", "--set", "no-equals"]
        )
        assert result.exit_code != 0
        assert "KEY=VALUE" in result.output
