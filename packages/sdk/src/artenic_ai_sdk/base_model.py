"""BaseModel ABC — the core contract every model must implement.

Uses the Template Method pattern: public methods handle state transitions,
logging, and metrics, then delegate to abstract `_do_*` hooks that each
concrete model implements.

Lifecycle::

    CREATED → warmup() → READY → predict() → READY → teardown() → SHUTDOWN
                           ↑                    |
                           └── health_check() ──┘
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

from artenic_ai_sdk.exceptions import ModelInferenceError, ModelLoadError
from artenic_ai_sdk.observability import ObservabilityMixin
from artenic_ai_sdk.schemas import (
    BasePrediction,
    EvalResult,
    FeatureSchema,
    HealthCheckResult,
    ModelConfig,
    ModelMetadata,
    TrainResult,
)
from artenic_ai_sdk.types import (
    ModelFramework,
    ModelId,
    ModelPhase,
    ModelVersion,
    SerializationFormat,
)


class BaseModel(ABC, ObservabilityMixin):
    """Abstract base class that every artenic_ai model must implement.

    Provides:
    - Lifecycle management (warmup → ready → inference → shutdown)
    - Automatic observability (metrics + structured logging) via mixin
    - State machine with phase tracking
    - Template Method hooks for concrete implementations

    Example::

        class MyModel(BaseModel):
            @property
            def model_id(self) -> str:
                return "my_model"

            @property
            def model_version(self) -> str:
                return "0.1.0"

            @property
            def model_type(self) -> str:
                return "lightgbm"

            @property
            def framework(self) -> ModelFramework:
                return ModelFramework.LIGHTGBM

            async def _do_warmup(self) -> None:
                self._model = load_my_model()

            async def predict(self, features: dict[str, Any]) -> BasePrediction:
                result = self._model.predict(features)
                return BasePrediction(...)
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self._phase: ModelPhase = ModelPhase.CREATED
        self._config: ModelConfig = config or ModelConfig()
        self._last_inference_at: datetime | None = None
        self._inference_count: int = 0
        self._error_count: int = 0
        self._init_observability()

    # =========================================================================
    # Abstract properties — identity
    # =========================================================================

    @property
    @abstractmethod
    def model_id(self) -> ModelId:
        """Unique identifier for this model (e.g. 'my_classifier_v1')."""
        ...  # pragma: no cover

    @property
    @abstractmethod
    def model_version(self) -> ModelVersion:
        """Semantic version string (e.g. '0.1.0')."""
        ...  # pragma: no cover

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Model type descriptor (e.g. 'lightgbm', 't_mamba')."""
        ...  # pragma: no cover

    @property
    @abstractmethod
    def framework(self) -> ModelFramework:
        """ML framework used by this model."""
        ...  # pragma: no cover

    # =========================================================================
    # Lifecycle — Template Method pattern
    # =========================================================================

    async def warmup(self) -> None:
        """Load model into memory/GPU. Called once at startup.

        Transitions: CREATED → WARMING_UP → READY (or ERROR on failure).
        """
        self._phase = ModelPhase.WARMING_UP
        self._logger.info(
            "Warmup started",
            model_id=self.model_id,
            event="lifecycle.warmup.start",
        )
        try:
            await self._do_warmup()
            self._phase = ModelPhase.READY
            self._logger.info(
                "Warmup completed — model ready",
                model_id=self.model_id,
                event="lifecycle.warmup.complete",
            )
        except Exception as e:
            self._phase = ModelPhase.ERROR
            self._logger.error(
                f"Warmup failed: {e}",
                model_id=self.model_id,
                event="lifecycle.warmup.error",
                error=str(e),
            )
            raise ModelLoadError(
                f"Warmup failed for {self.model_id}: {e}",
                details={"model_id": self.model_id, "error": str(e)},
            ) from e

    async def teardown(self) -> None:
        """Release resources. Called at shutdown.

        Transitions: * → SHUTDOWN.
        """
        self._logger.info(
            "Teardown started",
            model_id=self.model_id,
            event="lifecycle.teardown.start",
        )
        self._phase = ModelPhase.SHUTDOWN
        try:
            await self._do_teardown()
            self._logger.info(
                "Teardown completed",
                model_id=self.model_id,
                event="lifecycle.teardown.complete",
            )
        except Exception as e:
            self._logger.error(
                f"Teardown error: {e}",
                model_id=self.model_id,
                event="lifecycle.teardown.error",
                error=str(e),
            )

    async def health_check(self) -> HealthCheckResult:
        """Return current health status. Called periodically by the platform."""
        metrics_summary = self._metrics.get_summary()
        uptime = time.time() - self._created_at

        status: Literal["healthy", "degraded", "unhealthy"]
        if self._phase == ModelPhase.ERROR:
            status = "unhealthy"
        elif metrics_summary["error_rate"] > 0.1:
            status = "degraded"
        else:
            status = "healthy"

        return HealthCheckResult(
            status=status,
            phase=self._phase,
            uptime_seconds=uptime,
            last_inference_at=self._last_inference_at,
            inference_count=self._inference_count,
            error_count=self._error_count,
            details=metrics_summary,
        )

    # =========================================================================
    # Inference
    # =========================================================================

    @abstractmethod
    async def predict(self, features: dict[str, Any]) -> BasePrediction:
        """Run inference on a single input.

        Automatically tracked by ObservabilityMixin (latency, errors).

        Args:
            features: Pre-processed feature dictionary.

        Returns:
            A BasePrediction (or domain-specific subclass).
        """
        ...  # pragma: no cover

    async def predict_batch(self, batch: list[dict[str, Any]]) -> list[BasePrediction]:
        """Run inference on a batch of inputs.

        Default implementation loops over predict(). Override for
        frameworks that support native batching (PyTorch, ONNX).
        """
        results: list[BasePrediction] = []
        for features in batch:
            results.append(await self.predict(features))
        return results

    @abstractmethod
    async def preprocess(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        """Transform raw input into model features.

        Args:
            raw_input: Raw data from the platform/API.

        Returns:
            Processed feature dictionary ready for predict().
        """
        ...  # pragma: no cover

    async def postprocess(self, prediction: BasePrediction) -> BasePrediction:
        """Optional hook to post-process predictions.

        Override to add calibration, thresholding, or enrichment.
        Default: returns prediction unchanged.
        """
        return prediction

    # =========================================================================
    # Training (dispatched to remote providers)
    # =========================================================================

    @abstractmethod
    async def train(self, dataset: Any, config: ModelConfig) -> TrainResult:
        """Train the model. Executed on remote providers (OVH, GCP, etc.).

        Args:
            dataset: Training data (format depends on model).
            config: Hyperparameters and training config.

        Returns:
            TrainResult with metrics and artifact path.
        """
        ...  # pragma: no cover

    @abstractmethod
    async def evaluate(self, dataset: Any) -> EvalResult:
        """Evaluate model performance on a dataset.

        Args:
            dataset: Evaluation data.

        Returns:
            EvalResult with metrics.
        """
        ...  # pragma: no cover

    # =========================================================================
    # Persistence
    # =========================================================================

    @abstractmethod
    async def save(
        self,
        path: Path,
        format: SerializationFormat = SerializationFormat.SAFETENSORS,
    ) -> Path:
        """Save model weights and metadata to disk.

        Args:
            path: Target directory.
            format: Serialization format.

        Returns:
            Path to saved directory.
        """
        ...  # pragma: no cover

    @abstractmethod
    async def load(self, path: Path) -> None:
        """Load model weights from disk.

        Args:
            path: Directory containing saved model files.
        """
        ...  # pragma: no cover

    # =========================================================================
    # Metadata & Config
    # =========================================================================

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Return model metadata for the registry."""
        ...  # pragma: no cover

    def get_feature_schema(self) -> list[FeatureSchema]:
        """Return input feature schema. Override to declare features."""
        return []

    def validate_features(self, features: dict[str, Any]) -> None:
        """Validate input features against the declared schema.

        No-op if get_feature_schema() returns an empty list.
        Raises ModelInferenceError on missing or mistyped features.
        """
        schema = self.get_feature_schema()
        if not schema:
            return

        missing = [f.name for f in schema if f.required and f.name not in features]
        if missing:
            raise ModelInferenceError(
                f"Missing required features: {missing}",
                details={"missing": missing, "provided": list(features.keys())},
            )

        dtype_map: dict[str, type | tuple[type, ...]] = {
            "float32": (float, int),
            "float64": (float, int),
            "int32": (int,),
            "int64": (int,),
            "bool": (bool,),
            "string": (str,),
        }
        for f in schema:
            if f.name not in features:
                continue
            expected = dtype_map.get(f.dtype)
            if expected and not isinstance(features[f.name], expected):
                raise ModelInferenceError(
                    f"Feature '{f.name}' expected {f.dtype}, got {type(features[f.name]).__name__}",
                    details={"feature": f.name, "expected": f.dtype},
                )

    def get_config(self) -> ModelConfig:
        """Return the current model config."""
        return self._config

    async def update_config(self, config: ModelConfig) -> None:
        """Hot-reload config without restarting the model.

        Override if your model needs to re-initialize parameters
        when config changes.
        """
        self._config = config
        self._logger.info(
            "Config updated",
            model_id=self.model_id,
            event="config.updated",
            version=config.version,
        )

    # =========================================================================
    # State
    # =========================================================================

    @property
    def phase(self) -> ModelPhase:
        """Current lifecycle phase."""
        return self._phase

    @property
    def is_ready(self) -> bool:
        """Whether the model is ready for inference."""
        return self._phase == ModelPhase.READY

    # =========================================================================
    # Template Method hooks — implemented by concrete models
    # =========================================================================

    @abstractmethod
    async def _do_warmup(self) -> None:
        """Load model weights, initialize runtime.

        Called by warmup() after state transition.
        """
        ...  # pragma: no cover

    async def _do_teardown(self) -> None:
        """Release GPU memory, close connections.

        Called by teardown(). Override if cleanup is needed.
        Default: no-op.
        """

    # =========================================================================
    # Dunder
    # =========================================================================

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"id={self.model_id!r} "
            f"v={self.model_version!r} "
            f"phase={self._phase.value!r}>"
        )

    async def __aenter__(self) -> BaseModel:
        await self.warmup()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.teardown()
