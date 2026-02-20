"""Smart Data Splitting — automatic train/val/test splits with stratification."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

    from artenic_ai_sdk_training.config import DataSplitConfig

logger = logging.getLogger(__name__)


@dataclass
class DataSplitResult:
    """Result of a data split operation."""

    train: Any
    val: Any | None = None
    test: Any | None = None
    fold_index: int | None = None
    split_info: dict[str, Any] = field(default_factory=dict)


class SmartDataSplitter:
    """Auto-split datasets with stratification and cross-validation.

    Supports holdout, k-fold, stratified k-fold, and time-series splits.
    Uses sklearn internally (lazy import) with numpy fallback.
    """

    def __init__(self, config: DataSplitConfig) -> None:
        self._config = config

    def split(
        self,
        dataset: Any,
        target_column: str | None = None,
    ) -> DataSplitResult:
        """Perform a single train/val/test split."""
        n = len(dataset)
        rng = np.random.default_rng(self._config.random_seed)

        if self._config.strategy == "time_series":
            return self._time_series_split(dataset, n)

        if self._config.strategy in ("stratified_kfold", "holdout") and target_column:
            return self._stratified_holdout(dataset, target_column, n, rng)

        return self._random_holdout(dataset, n, rng)

    def kfold_splits(
        self,
        dataset: Any,
        target_column: str | None = None,
    ) -> Iterator[DataSplitResult]:
        """Generate k-fold cross-validation splits."""
        n = len(dataset)
        indices = np.arange(n)

        if self._config.strategy == "time_series":
            yield from self._time_series_kfold(dataset, n)
            return

        if self._config.strategy == "stratified_kfold" and target_column:
            yield from self._stratified_kfold(dataset, target_column, n)
            return

        rng = np.random.default_rng(self._config.random_seed)
        rng.shuffle(indices)
        folds = np.array_split(indices, self._config.n_folds)

        for i, test_idx in enumerate(folds):
            train_idx = np.concatenate([f for j, f in enumerate(folds) if j != i])
            yield DataSplitResult(
                train=self._index(dataset, train_idx),
                val=self._index(dataset, test_idx),
                fold_index=i,
                split_info={"train_size": len(train_idx), "val_size": len(test_idx)},
            )

    # ------------------------------------------------------------------
    # Internal split strategies
    # ------------------------------------------------------------------

    def _random_holdout(
        self,
        dataset: Any,
        n: int,
        rng: np.random.Generator,
    ) -> DataSplitResult:
        indices = np.arange(n)
        rng.shuffle(indices)
        train_end = int(n * self._config.train_ratio)
        val_end = train_end + int(n * self._config.val_ratio)
        return DataSplitResult(
            train=self._index(dataset, indices[:train_end]),
            val=(self._index(dataset, indices[train_end:val_end]) if val_end > train_end else None),
            test=(self._index(dataset, indices[val_end:]) if val_end < n else None),
            split_info={
                "strategy": "holdout",
                "train_size": train_end,
                "val_size": val_end - train_end,
                "test_size": n - val_end,
            },
        )

    def _stratified_holdout(
        self,
        dataset: Any,
        target_column: str,
        n: int,
        rng: np.random.Generator,
    ) -> DataSplitResult:
        try:
            from sklearn.model_selection import train_test_split

            targets = self._get_targets(dataset, target_column)
            train_ratio = self._config.train_ratio
            remaining_ratio = 1.0 - train_ratio

            train_idx, remaining_idx = train_test_split(
                np.arange(n),
                test_size=remaining_ratio,
                stratify=targets,
                random_state=self._config.random_seed,
            )

            if self._config.test_ratio > 0 and remaining_ratio > 0:
                test_frac = self._config.test_ratio / remaining_ratio
                test_frac = min(test_frac, 1.0)
                remaining_targets = targets[remaining_idx]
                val_idx, test_idx = train_test_split(
                    remaining_idx,
                    test_size=test_frac,
                    stratify=remaining_targets,
                    random_state=self._config.random_seed,
                )
            else:
                val_idx = remaining_idx
                test_idx = np.array([], dtype=int)

            return DataSplitResult(
                train=self._index(dataset, train_idx),
                val=self._index(dataset, val_idx) if len(val_idx) > 0 else None,
                test=self._index(dataset, test_idx) if len(test_idx) > 0 else None,
                split_info={
                    "strategy": "stratified_holdout",
                    "train_size": len(train_idx),
                    "val_size": len(val_idx),
                    "test_size": len(test_idx),
                },
            )
        except ImportError:
            logger.warning("sklearn not available, falling back to random holdout")
            return self._random_holdout(dataset, n, rng)

    def _time_series_split(self, dataset: Any, n: int) -> DataSplitResult:
        train_end = int(n * self._config.train_ratio)
        val_end = train_end + int(n * self._config.val_ratio)
        indices = np.arange(n)
        return DataSplitResult(
            train=self._index(dataset, indices[:train_end]),
            val=(self._index(dataset, indices[train_end:val_end]) if val_end > train_end else None),
            test=(self._index(dataset, indices[val_end:]) if val_end < n else None),
            split_info={
                "strategy": "time_series",
                "train_size": train_end,
                "val_size": val_end - train_end,
                "test_size": n - val_end,
            },
        )

    def _stratified_kfold(
        self,
        dataset: Any,
        target_column: str,
        n: int,
    ) -> Iterator[DataSplitResult]:
        try:
            from sklearn.model_selection import StratifiedKFold

            targets = self._get_targets(dataset, target_column)
            skf = StratifiedKFold(
                n_splits=self._config.n_folds,
                shuffle=True,
                random_state=self._config.random_seed,
            )
            for i, (train_idx, val_idx) in enumerate(skf.split(np.arange(n), targets)):
                yield DataSplitResult(
                    train=self._index(dataset, train_idx),
                    val=self._index(dataset, val_idx),
                    fold_index=i,
                    split_info={
                        "strategy": "stratified_kfold",
                        "train_size": len(train_idx),
                        "val_size": len(val_idx),
                    },
                )
        except ImportError:
            logger.warning("sklearn not available, falling back to regular kfold")
            yield from self.kfold_splits(dataset)

    def _time_series_kfold(
        self,
        dataset: Any,
        n: int,
    ) -> Iterator[DataSplitResult]:
        fold_size = n // (self._config.n_folds + 1)
        for i in range(self._config.n_folds):
            train_end = fold_size * (i + 1)
            val_end = min(train_end + fold_size, n)
            indices = np.arange(n)
            yield DataSplitResult(
                train=self._index(dataset, indices[:train_end]),
                val=self._index(dataset, indices[train_end:val_end]),
                fold_index=i,
                split_info={
                    "strategy": "time_series_kfold",
                    "train_size": train_end,
                    "val_size": val_end - train_end,
                },
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _index(dataset: Any, indices: np.ndarray) -> Any:
        """Index into a dataset (numpy array, pandas DataFrame, or list)."""
        if hasattr(dataset, "iloc"):
            return dataset.iloc[indices]
        if isinstance(dataset, np.ndarray):
            return dataset[indices]
        return [dataset[i] for i in indices]

    @staticmethod
    def _get_targets(dataset: Any, target_column: str) -> np.ndarray:
        """Extract target values for stratification."""
        if hasattr(dataset, "__getitem__") and hasattr(dataset, "iloc"):
            return np.asarray(dataset[target_column])
        return np.asarray(dataset)
