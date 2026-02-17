"""Tests for inference commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from artenic_ai_cli.main import cli

if TYPE_CHECKING:
    from click.testing import CliRunner


class TestPredict:
    def test_predict(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"prediction": 0.95}):
            result = runner.invoke(cli, ["predict", "trading", "--data", '{"features": [1,2,3]}'])
            assert result.exit_code == 0
            assert "0.95" in result.output

    def test_predict_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"prediction": 0.95}):
            result = runner.invoke(
                cli, ["--json", "predict", "trading", "--data", '{"features": [1]}']
            )
            assert result.exit_code == 0
            assert "0.95" in result.output

    def test_predict_with_model_id(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"prediction": 0.8}):
            result = runner.invoke(cli, ["predict", "svc", "--data", "{}", "--model-id", "m1"])
            assert result.exit_code == 0


class TestPredictBatch:
    def test_batch(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value=[{"p": 0.1}, {"p": 0.2}]):
            result = runner.invoke(cli, ["predict-batch", "svc", "--data", '[{"x": 1}, {"x": 2}]'])
            assert result.exit_code == 0

    def test_batch_with_model_id(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value=[{"p": 0.5}]):
            result = runner.invoke(
                cli, ["predict-batch", "svc", "--data", '[{"x": 1}]', "--model-id", "m2"]
            )
            assert result.exit_code == 0


class TestInvalidJson:
    def test_predict_invalid_json(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["predict", "svc", "--data", "not-json"])
        assert result.exit_code != 0
        assert "Invalid JSON" in result.output

    def test_predict_batch_invalid_json(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["predict-batch", "svc", "--data", "{bad"])
        assert result.exit_code != 0
        assert "Invalid JSON" in result.output
