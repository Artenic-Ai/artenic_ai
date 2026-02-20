"""Async wrapper around MLflow for experiment tracking and model registry."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Async interface to MLflow tracking and model registry.

    All blocking MLflow SDK calls are dispatched via
    ``asyncio.to_thread`` so they do not block the event loop.

    If MLflow is not installed or connection fails, the tracker
    degrades gracefully — every method returns ``None`` instead of
    raising.
    """

    def __init__(self, tracking_uri: str = "", artifact_root: str = "") -> None:
        self._tracking_uri = tracking_uri
        self._artifact_root = artifact_root
        self._available = False
        self._mlflow: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup(self) -> bool:
        """Connect to MLflow.  Returns True if available."""
        try:
            import mlflow

            self._mlflow = mlflow
            if self._tracking_uri:
                await asyncio.to_thread(mlflow.set_tracking_uri, self._tracking_uri)
            self._available = True
            logger.info("MLflow connected (uri=%s)", self._tracking_uri or "default")
        except Exception:
            self._available = False
            logger.warning("MLflow unavailable — tracking disabled", exc_info=True)
        return self._available

    @property
    def available(self) -> bool:
        """Whether MLflow is connected and usable."""
        return self._available

    # ------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------

    async def create_experiment(self, domain: str, name: str) -> str | None:
        """Create or get an MLflow experiment.  Returns experiment_id."""
        if not self._available:
            return None
        try:
            full_name = f"{domain}/{name}"
            experiment_id: str = await asyncio.to_thread(
                self._mlflow.create_experiment,
                full_name,
                artifact_location=self._artifact_root or None,
            )
            return experiment_id
        except Exception:
            # Experiment may already exist — try to get it
            try:
                exp = await asyncio.to_thread(
                    self._mlflow.get_experiment_by_name, f"{domain}/{name}"
                )
                return str(exp.experiment_id) if exp else None
            except Exception:
                logger.warning("Failed to create/get experiment", exc_info=True)
                return None

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    async def start_run(
        self,
        experiment_id: str,
        run_name: str,
        tags: dict[str, str] | None = None,
    ) -> str | None:
        """Start a new MLflow run.  Returns run_id."""
        if not self._available:
            return None
        try:
            run = await asyncio.to_thread(
                self._mlflow.start_run,
                experiment_id=experiment_id,
                run_name=run_name,
                tags=tags or {},
            )
            return str(run.info.run_id)
        except Exception:
            logger.warning("Failed to start MLflow run", exc_info=True)
            return None

    async def end_run(self, status: str = "FINISHED") -> None:
        """End the current active MLflow run."""
        if not self._available:
            return
        try:
            await asyncio.to_thread(self._mlflow.end_run, status)
        except Exception:
            logger.warning("Failed to end MLflow run", exc_info=True)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    async def log_params(self, run_id: str, params: dict[str, Any]) -> None:
        """Log parameters to a run."""
        if not self._available:
            return
        try:
            client = self._mlflow.tracking.MlflowClient()
            for key, value in params.items():
                await asyncio.to_thread(client.log_param, run_id, key, str(value))
        except Exception:
            logger.warning("Failed to log params", exc_info=True)

    async def log_metrics(
        self,
        run_id: str,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Log metrics to a run."""
        if not self._available:
            return
        try:
            client = self._mlflow.tracking.MlflowClient()
            for key, value in metrics.items():
                kwargs: dict[str, Any] = {
                    "run_id": run_id,
                    "key": key,
                    "value": value,
                }
                if step is not None:
                    kwargs["step"] = step
                await asyncio.to_thread(client.log_metric, **kwargs)
        except Exception:
            logger.warning("Failed to log metrics", exc_info=True)

    # ------------------------------------------------------------------
    # Model registry
    # ------------------------------------------------------------------

    async def log_model(
        self,
        run_id: str,
        model_path: str,
        model_name: str,
    ) -> str | None:
        """Log a model artifact.  Returns model_uri."""
        if not self._available:
            return None
        try:
            uri = f"runs:/{run_id}/{model_path}"
            await asyncio.to_thread(self._mlflow.log_artifact, model_path)
            return uri
        except Exception:
            logger.warning("Failed to log model", exc_info=True)
            return None

    async def register_model(self, model_uri: str, name: str) -> str | None:
        """Register a model in the MLflow Model Registry.

        Returns the version string.
        """
        if not self._available:
            return None
        try:
            result = await asyncio.to_thread(self._mlflow.register_model, model_uri, name)
            return str(result.version)
        except Exception:
            logger.warning("Failed to register model", exc_info=True)
            return None

    async def transition_stage(self, name: str, version: str, stage: str) -> bool:
        """Transition a model version to a new stage.

        Returns True on success.
        """
        if not self._available:
            return False
        try:
            client = self._mlflow.tracking.MlflowClient()
            await asyncio.to_thread(
                client.transition_model_version_stage,
                name=name,
                version=version,
                stage=stage,
            )
            return True
        except Exception:
            logger.warning("Failed to transition stage", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def get_best_run(self, experiment_id: str, metric: str) -> dict[str, Any] | None:
        """Find the best run in an experiment by a given metric.

        Returns ``{"run_id": ..., "metrics": ..., "params": ...}``
        or ``None``.
        """
        if not self._available:
            return None
        try:
            client = self._mlflow.tracking.MlflowClient()
            runs = await asyncio.to_thread(
                client.search_runs,
                experiment_ids=[experiment_id],
                order_by=[f"metrics.{metric} DESC"],
                max_results=1,
            )
            if not runs:
                return None
            best = runs[0]
            return {
                "run_id": best.info.run_id,
                "metrics": dict(best.data.metrics),
                "params": dict(best.data.params),
            }
        except Exception:
            logger.warning("Failed to get best run", exc_info=True)
            return None
