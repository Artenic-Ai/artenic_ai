"""Storage backends for dataset files.

Provides an abstract ``StorageBackend`` and concrete implementations:

* **FilesystemStorage** — default, stores files on local disk.
* **S3Storage** / **GCSStorage** / **AzureBlobStorage** / **OVHSwiftStorage** —
  cloud provider stubs (raise ``NotImplementedError`` until implemented).
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract base for dataset file storage."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short identifier for this backend (e.g. ``filesystem``, ``s3``)."""

    @abstractmethod
    async def save(self, key: str, data: bytes) -> str:
        """Persist *data* under *key*.  Returns the storage path."""

    @abstractmethod
    async def load(self, key: str) -> bytes:
        """Read and return the bytes stored under *key*."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove the object stored under *key*."""

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Return ``True`` if *key* exists in the store."""

    @abstractmethod
    async def list_keys(self, prefix: str) -> list[str]:
        """List all keys that start with *prefix*."""

    # Utility -----------------------------------------------------------

    @staticmethod
    def hash_bytes(data: bytes) -> str:
        """Compute SHA-256 hex digest of *data*."""
        return hashlib.sha256(data).hexdigest()


# ======================================================================
# Filesystem (default)
# ======================================================================


class FilesystemStorage(StorageBackend):
    """Store dataset files on the local filesystem."""

    @property
    def provider_name(self) -> str:
        return "filesystem"

    def __init__(self, base_dir: str = "data/datasets") -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        target = (self._base / key).resolve()
        base_resolved = self._base.resolve()
        if not target.is_relative_to(base_resolved):
            msg = f"Path traversal detected: {key}"
            raise ValueError(msg)
        return target

    async def save(self, key: str, data: bytes) -> str:
        target = self._resolve(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        logger.debug("Saved %d bytes → %s", len(data), target)
        return key

    async def load(self, key: str) -> bytes:
        target = self._resolve(key)
        if not target.is_file():
            msg = f"File not found: {key}"
            raise FileNotFoundError(msg)
        return target.read_bytes()

    async def delete(self, key: str) -> None:
        target = self._resolve(key)
        if target.is_file():
            target.unlink()
            logger.debug("Deleted %s", target)

    async def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    async def list_keys(self, prefix: str) -> list[str]:
        base = self._resolve(prefix)
        if not base.exists():
            return []
        root = self._base
        return sorted(str(p.relative_to(root)) for p in base.rglob("*") if p.is_file())


# ======================================================================
# Cloud stubs (raise NotImplementedError until implemented)
# ======================================================================


def _not_implemented(provider: str) -> NotImplementedError:
    return NotImplementedError(
        f"{provider} storage is not yet available. Use the filesystem backend."
    )


class S3Storage(StorageBackend):
    """AWS S3 storage backend (stub)."""

    @property
    def provider_name(self) -> str:
        return "s3"

    def __init__(
        self,
        bucket: str = "",
        prefix: str = "datasets/",
        *,
        endpoint_url: str = "",
        access_key: str = "",
        secret_key: str = "",
        region: str = "",
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix
        self.endpoint_url = endpoint_url
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region

    async def save(self, key: str, data: bytes) -> str:
        raise _not_implemented("S3")

    async def load(self, key: str) -> bytes:
        raise _not_implemented("S3")

    async def delete(self, key: str) -> None:
        raise _not_implemented("S3")

    async def exists(self, key: str) -> bool:
        raise _not_implemented("S3")

    async def list_keys(self, prefix: str) -> list[str]:
        raise _not_implemented("S3")


class GCSStorage(StorageBackend):
    """Google Cloud Storage backend (stub)."""

    @property
    def provider_name(self) -> str:
        return "gcs"

    def __init__(
        self,
        bucket: str = "",
        prefix: str = "datasets/",
        *,
        credentials_path: str = "",
        project_id: str = "",
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix
        self.credentials_path = credentials_path
        self.project_id = project_id

    async def save(self, key: str, data: bytes) -> str:
        raise _not_implemented("GCS")

    async def load(self, key: str) -> bytes:
        raise _not_implemented("GCS")

    async def delete(self, key: str) -> None:
        raise _not_implemented("GCS")

    async def exists(self, key: str) -> bool:
        raise _not_implemented("GCS")

    async def list_keys(self, prefix: str) -> list[str]:
        raise _not_implemented("GCS")


class AzureBlobStorage(StorageBackend):
    """Azure Blob Storage backend (stub)."""

    @property
    def provider_name(self) -> str:
        return "azure"

    def __init__(
        self,
        container: str = "",
        prefix: str = "datasets/",
        *,
        connection_string: str = "",
    ) -> None:
        self.container = container
        self.prefix = prefix
        self.connection_string = connection_string

    async def save(self, key: str, data: bytes) -> str:
        raise _not_implemented("Azure Blob")

    async def load(self, key: str) -> bytes:
        raise _not_implemented("Azure Blob")

    async def delete(self, key: str) -> None:
        raise _not_implemented("Azure Blob")

    async def exists(self, key: str) -> bool:
        raise _not_implemented("Azure Blob")

    async def list_keys(self, prefix: str) -> list[str]:
        raise _not_implemented("Azure Blob")


class OVHSwiftStorage(StorageBackend):
    """OVH Object Storage (OpenStack Swift) backend (stub)."""

    @property
    def provider_name(self) -> str:
        return "ovh"

    def __init__(
        self,
        container: str = "",
        prefix: str = "datasets/",
        *,
        endpoint_url: str = "",
        access_key: str = "",
        secret_key: str = "",
        region: str = "",
    ) -> None:
        self.container = container
        self.prefix = prefix
        self.endpoint_url = endpoint_url
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region

    async def save(self, key: str, data: bytes) -> str:
        raise _not_implemented("OVH Swift")

    async def load(self, key: str) -> bytes:
        raise _not_implemented("OVH Swift")

    async def delete(self, key: str) -> None:
        raise _not_implemented("OVH Swift")

    async def exists(self, key: str) -> bool:
        raise _not_implemented("OVH Swift")

    async def list_keys(self, prefix: str) -> list[str]:
        raise _not_implemented("OVH Swift")
