"""Tests for artenic_ai_platform.tracking.mlflow_client — 100% coverage."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from artenic_ai_platform.tracking.mlflow_client import MLflowTracker

# ======================================================================
# MLflowTracker.__init__
# ======================================================================


class TestMLflowTrackerInit:
    def test_defaults(self) -> None:
        tracker = MLflowTracker()
        assert tracker._tracking_uri == ""
        assert tracker._artifact_root == ""
        assert tracker.available is False
        assert tracker._mlflow is None

    def test_custom_values(self) -> None:
        tracker = MLflowTracker(
            tracking_uri="http://mlflow:5000",
            artifact_root="s3://bucket",
        )
        assert tracker._tracking_uri == "http://mlflow:5000"
        assert tracker._artifact_root == "s3://bucket"


# ======================================================================
# setup
# ======================================================================


class TestSetup:
    async def test_setup_when_mlflow_not_installed(self) -> None:
        tracker = MLflowTracker()
        # Make `import mlflow` raise ImportError
        with patch.dict("sys.modules", {"mlflow": None}):
            result = await tracker.setup()
        assert result is False
        assert tracker.available is False

    async def test_setup_available_no_uri(self) -> None:
        mock_mlflow = MagicMock()
        tracker = MLflowTracker()

        # Inject a fake mlflow module so `import mlflow` succeeds
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            result = await tracker.setup()
        assert result is True
        assert tracker.available is True
        assert tracker._mlflow is mock_mlflow

    async def test_setup_with_tracking_uri(self) -> None:
        mock_mlflow = MagicMock()
        tracker = MLflowTracker(tracking_uri="http://localhost:5000")

        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            result = await tracker.setup()
        assert result is True
        assert tracker.available is True
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")


# ======================================================================
# Unavailable guard — every method returns None/False/nothing
# ======================================================================


class TestUnavailableGuard:
    async def test_create_experiment_returns_none(self) -> None:
        tracker = MLflowTracker()
        result = await tracker.create_experiment("domain", "name")
        assert result is None

    async def test_start_run_returns_none(self) -> None:
        tracker = MLflowTracker()
        result = await tracker.start_run("exp-1", "run-1")
        assert result is None

    async def test_end_run_returns_none(self) -> None:
        tracker = MLflowTracker()
        await tracker.end_run()  # Should not raise

    async def test_log_params_returns_none(self) -> None:
        tracker = MLflowTracker()
        await tracker.log_params("run-1", {"key": "val"})

    async def test_log_metrics_returns_none(self) -> None:
        tracker = MLflowTracker()
        await tracker.log_metrics("run-1", {"acc": 0.95})

    async def test_log_model_returns_none(self) -> None:
        tracker = MLflowTracker()
        result = await tracker.log_model("run-1", "/path", "model")
        assert result is None

    async def test_register_model_returns_none(self) -> None:
        tracker = MLflowTracker()
        result = await tracker.register_model("uri", "name")
        assert result is None

    async def test_transition_stage_returns_false(self) -> None:
        tracker = MLflowTracker()
        result = await tracker.transition_stage("name", "1", "Production")
        assert result is False

    async def test_get_best_run_returns_none(self) -> None:
        tracker = MLflowTracker()
        result = await tracker.get_best_run("exp-1", "accuracy")
        assert result is None


# ======================================================================
# Available — success paths (all with mocked mlflow)
# ======================================================================


def _make_available_tracker() -> MLflowTracker:
    """Create a tracker with mocked MLflow and available=True."""
    tracker = MLflowTracker()
    tracker._available = True
    tracker._mlflow = MagicMock()
    return tracker


class TestCreateExperiment:
    async def test_success(self) -> None:
        tracker = _make_available_tracker()
        tracker._mlflow.create_experiment = MagicMock(return_value="exp-42")

        result = await tracker.create_experiment("ml", "sentiment")
        assert result == "exp-42"

    async def test_already_exists_fallback(self) -> None:
        tracker = _make_available_tracker()
        tracker._mlflow.create_experiment = MagicMock(side_effect=Exception("already exists"))
        mock_exp = SimpleNamespace(experiment_id="exp-99")
        tracker._mlflow.get_experiment_by_name = MagicMock(return_value=mock_exp)

        result = await tracker.create_experiment("ml", "sentiment")
        assert result == "exp-99"

    async def test_already_exists_returns_none_if_not_found(self) -> None:
        tracker = _make_available_tracker()
        tracker._mlflow.create_experiment = MagicMock(side_effect=Exception("fail"))
        tracker._mlflow.get_experiment_by_name = MagicMock(return_value=None)

        result = await tracker.create_experiment("ml", "test")
        assert result is None

    async def test_both_fail(self) -> None:
        tracker = _make_available_tracker()
        tracker._mlflow.create_experiment = MagicMock(side_effect=Exception("create fail"))
        tracker._mlflow.get_experiment_by_name = MagicMock(side_effect=Exception("get fail"))

        result = await tracker.create_experiment("ml", "test")
        assert result is None

    async def test_with_artifact_root(self) -> None:
        tracker = _make_available_tracker()
        tracker._artifact_root = "s3://bucket"
        tracker._mlflow.create_experiment = MagicMock(return_value="42")

        result = await tracker.create_experiment("ml", "test")
        assert result == "42"


class TestStartRun:
    async def test_success(self) -> None:
        tracker = _make_available_tracker()
        mock_run = SimpleNamespace(info=SimpleNamespace(run_id="run-abc"))
        tracker._mlflow.start_run = MagicMock(return_value=mock_run)

        result = await tracker.start_run("exp-1", "my-run")
        assert result == "run-abc"

    async def test_with_tags(self) -> None:
        tracker = _make_available_tracker()
        mock_run = SimpleNamespace(info=SimpleNamespace(run_id="run-123"))
        tracker._mlflow.start_run = MagicMock(return_value=mock_run)

        result = await tracker.start_run("exp-1", "my-run", tags={"env": "test"})
        assert result == "run-123"

    async def test_failure(self) -> None:
        tracker = _make_available_tracker()
        tracker._mlflow.start_run = MagicMock(side_effect=Exception("fail"))

        result = await tracker.start_run("exp-1", "run")
        assert result is None


class TestEndRun:
    async def test_success(self) -> None:
        tracker = _make_available_tracker()
        tracker._mlflow.end_run = MagicMock()
        await tracker.end_run()

    async def test_failure(self) -> None:
        tracker = _make_available_tracker()
        tracker._mlflow.end_run = MagicMock(side_effect=Exception("fail"))
        await tracker.end_run()  # Should not raise

    async def test_custom_status(self) -> None:
        tracker = _make_available_tracker()
        tracker._mlflow.end_run = MagicMock()
        await tracker.end_run(status="FAILED")


class TestLogParams:
    async def test_success(self) -> None:
        tracker = _make_available_tracker()
        mock_client = MagicMock()
        tracker._mlflow.tracking.MlflowClient = MagicMock(return_value=mock_client)

        await tracker.log_params("run-1", {"lr": "0.001", "epochs": "10"})
        assert mock_client.log_param.call_count == 2

    async def test_failure(self) -> None:
        tracker = _make_available_tracker()
        tracker._mlflow.tracking.MlflowClient = MagicMock(side_effect=Exception("fail"))
        await tracker.log_params("run-1", {"lr": "0.001"})


class TestLogMetrics:
    async def test_success_no_step(self) -> None:
        tracker = _make_available_tracker()
        mock_client = MagicMock()
        tracker._mlflow.tracking.MlflowClient = MagicMock(return_value=mock_client)

        await tracker.log_metrics("run-1", {"acc": 0.95, "loss": 0.05})
        assert mock_client.log_metric.call_count == 2

    async def test_success_with_step(self) -> None:
        tracker = _make_available_tracker()
        mock_client = MagicMock()
        tracker._mlflow.tracking.MlflowClient = MagicMock(return_value=mock_client)

        await tracker.log_metrics("run-1", {"acc": 0.95}, step=5)
        assert mock_client.log_metric.call_count == 1

    async def test_failure(self) -> None:
        tracker = _make_available_tracker()
        tracker._mlflow.tracking.MlflowClient = MagicMock(side_effect=Exception("fail"))
        await tracker.log_metrics("run-1", {"acc": 0.95})


class TestLogModel:
    async def test_success(self) -> None:
        tracker = _make_available_tracker()
        tracker._mlflow.log_artifact = MagicMock()

        result = await tracker.log_model("run-1", "/model.pt", "model")
        assert result == "runs:/run-1//model.pt"

    async def test_failure(self) -> None:
        tracker = _make_available_tracker()
        tracker._mlflow.log_artifact = MagicMock(side_effect=Exception("fail"))

        result = await tracker.log_model("run-1", "/model.pt", "model")
        assert result is None


class TestRegisterModel:
    async def test_success(self) -> None:
        tracker = _make_available_tracker()
        mock_result = SimpleNamespace(version="3")
        tracker._mlflow.register_model = MagicMock(return_value=mock_result)

        result = await tracker.register_model("runs:/run-1/model", "mymodel")
        assert result == "3"

    async def test_failure(self) -> None:
        tracker = _make_available_tracker()
        tracker._mlflow.register_model = MagicMock(side_effect=Exception("fail"))

        result = await tracker.register_model("uri", "name")
        assert result is None


class TestTransitionStage:
    async def test_success(self) -> None:
        tracker = _make_available_tracker()
        mock_client = MagicMock()
        tracker._mlflow.tracking.MlflowClient = MagicMock(return_value=mock_client)

        result = await tracker.transition_stage("model", "1", "Production")
        assert result is True

    async def test_failure(self) -> None:
        tracker = _make_available_tracker()
        tracker._mlflow.tracking.MlflowClient = MagicMock(side_effect=Exception("fail"))

        result = await tracker.transition_stage("model", "1", "Production")
        assert result is False


class TestGetBestRun:
    async def test_success(self) -> None:
        tracker = _make_available_tracker()
        mock_client = MagicMock()
        mock_run = SimpleNamespace(
            info=SimpleNamespace(run_id="best-run"),
            data=SimpleNamespace(
                metrics={"acc": 0.99},
                params={"lr": "0.001"},
            ),
        )
        mock_client.search_runs = MagicMock(return_value=[mock_run])
        tracker._mlflow.tracking.MlflowClient = MagicMock(return_value=mock_client)

        result = await tracker.get_best_run("exp-1", "acc")
        assert result is not None
        assert result["run_id"] == "best-run"
        assert result["metrics"]["acc"] == 0.99
        assert result["params"]["lr"] == "0.001"

    async def test_no_runs(self) -> None:
        tracker = _make_available_tracker()
        mock_client = MagicMock()
        mock_client.search_runs = MagicMock(return_value=[])
        tracker._mlflow.tracking.MlflowClient = MagicMock(return_value=mock_client)

        result = await tracker.get_best_run("exp-1", "acc")
        assert result is None

    async def test_failure(self) -> None:
        tracker = _make_available_tracker()
        tracker._mlflow.tracking.MlflowClient = MagicMock(side_effect=Exception("fail"))

        result = await tracker.get_best_run("exp-1", "acc")
        assert result is None
