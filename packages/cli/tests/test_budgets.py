"""Tests for budget commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from artenic_ai_cli.main import cli

if TYPE_CHECKING:
    from click.testing import CliRunner


class TestBudgetList:
    def test_list(self, runner: CliRunner, patch_run_async: Any) -> None:
        data = [
            {
                "id": "b1",
                "scope": "global",
                "scope_value": "*",
                "period": "monthly",
                "limit_eur": 500.0,
                "enabled": True,
            },
        ]
        with patch_run_async(return_value=data):
            result = runner.invoke(cli, ["budget", "list"])
            assert result.exit_code == 0
            assert "global" in result.output

    def test_list_with_scope(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value=[]):
            result = runner.invoke(cli, ["budget", "list", "--scope", "service"])
            assert result.exit_code == 0

    def test_list_all(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value=[]):
            result = runner.invoke(cli, ["budget", "list", "--all"])
            assert result.exit_code == 0


class TestBudgetCreate:
    def test_create(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"id": "b1"}):
            result = runner.invoke(
                cli,
                [
                    "budget",
                    "create",
                    "--scope",
                    "service",
                    "--scope-value",
                    "my-service",
                    "--period",
                    "monthly",
                    "--limit-eur",
                    "1000",
                ],
            )
            assert result.exit_code == 0
            assert "Budget created" in result.stderr

    def test_create_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"id": "b1"}):
            result = runner.invoke(
                cli,
                [
                    "--json",
                    "budget",
                    "create",
                    "--scope",
                    "global",
                    "--scope-value",
                    "*",
                    "--period",
                    "daily",
                    "--limit-eur",
                    "100",
                ],
            )
            assert result.exit_code == 0

    def test_create_with_threshold(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"id": "b2"}):
            result = runner.invoke(
                cli,
                [
                    "budget",
                    "create",
                    "--scope",
                    "provider",
                    "--scope-value",
                    "gcp",
                    "--period",
                    "weekly",
                    "--limit-eur",
                    "500",
                    "--alert-threshold",
                    "0.9",
                ],
            )
            assert result.exit_code == 0


class TestBudgetUpdate:
    def test_update(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"id": "b1"}):
            result = runner.invoke(cli, ["budget", "update", "b1", "--limit-eur", "2000"])
            assert result.exit_code == 0
            assert "updated" in result.stderr

    def test_update_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"id": "b1"}):
            result = runner.invoke(cli, ["--json", "budget", "update", "b1", "--enabled"])
            assert result.exit_code == 0

    def test_update_multiple_fields(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"id": "b1"}):
            result = runner.invoke(
                cli,
                [
                    "budget",
                    "update",
                    "b1",
                    "--limit-eur",
                    "3000",
                    "--alert-threshold",
                    "0.8",
                    "--disabled",
                    "--period",
                    "weekly",
                ],
            )
            assert result.exit_code == 0


class TestBudgetSpending:
    def test_spending(self, runner: CliRunner, patch_run_async: Any) -> None:
        data = [{"spent": 200, "limit": 500, "pct_used": 40}]
        with patch_run_async(return_value=data):
            result = runner.invoke(
                cli, ["budget", "spending", "--scope", "service", "--scope-value", "my-service"]
            )
            assert result.exit_code == 0
