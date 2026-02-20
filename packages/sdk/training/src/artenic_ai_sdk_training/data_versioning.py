"""Data Versioning — hash-based dataset identity tracking."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from artenic_ai_sdk_training.config import DataVersioningConfig

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class DatasetVersion:
    """Immutable dataset identity."""

    hash: str
    algorithm: str
    size_bytes: int
    num_records: int | None = None
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "hash": self.hash,
            "algorithm": self.algorithm,
            "size_bytes": self.size_bytes,
            "num_records": self.num_records,
            "created_at": self.created_at,
        }


class DatasetVersioner:
    """Hash-based dataset identity tracking.

    Creates a content hash of a dataset (file, directory, or in-memory)
    that uniquely identifies the exact data used for training.
    """

    def __init__(self, config: DataVersioningConfig | None = None) -> None:
        self._config = config or DataVersioningConfig(enabled=True)

    def hash_file(self, path: Path) -> DatasetVersion:
        """Compute a content hash of a single file."""
        hasher = self._make_hasher()
        size = 0
        with open(path, "rb") as fh:
            while chunk := fh.read(8192):
                hasher.update(chunk)
                size += len(chunk)
        return DatasetVersion(
            hash=hasher.hexdigest(),
            algorithm=self._config.hash_algorithm,
            size_bytes=size,
            created_at=datetime.now(tz=UTC).isoformat(),
        )

    def hash_directory(self, path: Path) -> DatasetVersion:
        """Compute a combined hash of all files in a directory (sorted)."""
        hasher = self._make_hasher()
        total_size = 0
        file_count = 0
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file():
                rel = file_path.relative_to(path).as_posix()
                hasher.update(rel.encode())
                with open(file_path, "rb") as fh:
                    while chunk := fh.read(8192):
                        hasher.update(chunk)
                        total_size += len(chunk)
                file_count += 1
        return DatasetVersion(
            hash=hasher.hexdigest(),
            algorithm=self._config.hash_algorithm,
            size_bytes=total_size,
            num_records=file_count,
            created_at=datetime.now(tz=UTC).isoformat(),
        )

    def hash_dataframe(self, df: Any) -> DatasetVersion:
        """Compute a hash of a pandas DataFrame."""
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        data = buf.getvalue()
        hasher = self._make_hasher()

        if self._config.sample_size and len(df) > self._config.sample_size:
            sample = df.head(self._config.sample_size)
            sample_buf = io.BytesIO()
            sample.to_parquet(sample_buf, index=False)
            data = sample_buf.getvalue()

        hasher.update(data)
        return DatasetVersion(
            hash=hasher.hexdigest(),
            algorithm=self._config.hash_algorithm,
            size_bytes=len(data),
            num_records=len(df),
            created_at=datetime.now(tz=UTC).isoformat(),
        )

    def verify(self, path: Path, expected: DatasetVersion) -> bool:
        """Verify a file or directory matches an expected version."""
        actual = self.hash_directory(path) if path.is_dir() else self.hash_file(path)
        return actual.hash == expected.hash

    def _make_hasher(self) -> Any:
        if self._config.hash_algorithm == "xxhash":
            try:
                import xxhash

                return xxhash.xxh128()
            except ImportError:
                return hashlib.sha256()
        return hashlib.sha256()
