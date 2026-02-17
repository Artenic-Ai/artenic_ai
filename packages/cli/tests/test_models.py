"""Tests for model commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from artenic_ai_cli.main import cli

if TYPE_CHECKING:
    from click.testing import CliRunner


class TestModelList:
    def test_table(self, runner: CliRunner, patch_run_async: Any) -> None:
        data = [
            {
                "id": "m1",
                "name": "lgbm",
                "version": "1.0",
                "model_type": "classifier",
                "stage": "production",
            },
        ]
        with patch_run_async(return_value=data):
            result = runner.invoke(cli, ["model", "list"])
            assert result.exit_code == 0
            assert "lgbm" in result.output

    def test_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        data = [{"id": "m1", "name": "lgbm"}]
        with patch_run_async(return_value=data):
            result = runner.invoke(cli, ["--json", "model", "list"])
            assert result.exit_code == 0
            assert '"lgbm"' in result.output


class TestModelGet:
    def test_get(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"id": "m1", "name": "lgbm"}):
            result = runner.invoke(cli, ["model", "get", "m1"])
            assert result.exit_code == 0
            assert "lgbm" in result.output


class TestModelRegister:
    def test_register(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"model_id": "m1"}):
            result = runner.invoke(
                cli,
                ["model", "register", "--name", "lgbm", "--version", "1.0", "--type", "classifier"],
            )
            assert result.exit_code == 0
            assert "m1" in result.stderr

    def test_register_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"model_id": "m1"}):
            result = runner.invoke(
                cli,
                [
                    "--json",
                    "model",
                    "register",
                    "--name",
                    "lgbm",
                    "--version",
                    "1.0",
                    "--type",
                    "cls",
                ],
            )
            assert result.exit_code == 0
            assert '"model_id"' in result.output

    def test_register_with_tags(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"model_id": "m2"}):
            result = runner.invoke(
                cli,
                [
                    "model",
                    "register",
                    "--name",
                    "x",
                    "--version",
                    "1",
                    "--type",
                    "t",
                    "--tag",
                    "env=prod",
                    "--tag",
                    "team=ml",
                    "--framework",
                    "pytorch",
                    "--description",
                    "test model",
                ],
            )
            assert result.exit_code == 0


class TestModelPromote:
    def test_promote(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value=None):
            result = runner.invoke(cli, ["model", "promote", "m1", "--version", "2.0"])
            assert result.exit_code == 0
            assert "promoted" in result.stderr

    def test_promote_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value=None):
            result = runner.invoke(cli, ["--json", "model", "promote", "m1", "--version", "2.0"])
            assert result.exit_code == 0
            assert '"promoted"' in result.output


class TestModelRetire:
    def test_retire(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value=None):
            result = runner.invoke(cli, ["model", "retire", "m1"])
            assert result.exit_code == 0
            assert "retired" in result.stderr

    def test_retire_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value=None):
            result = runner.invoke(cli, ["--json", "model", "retire", "m1"])
            assert result.exit_code == 0
            assert '"retired"' in result.output


class TestInvalidTag:
    def test_register_invalid_tag_format(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli,
            [
                "model",
                "register",
                "--name",
                "m",
                "--version",
                "1",
                "--type",
                "t",
                "--tag",
                "no-equals-sign",
            ],
        )
        assert result.exit_code != 0
        assert "key=value" in result.output
