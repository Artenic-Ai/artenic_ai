"""Tests for artenic_ai_platform.datasets.storage — all backends."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from artenic_ai_platform.datasets.storage import (
    AzureBlobStorage,
    FilesystemStorage,
    GCSStorage,
    OVHSwiftStorage,
    S3Storage,
    StorageBackend,
)

# ======================================================================
# FilesystemStorage
# ======================================================================


class TestFilesystemStorageSave:
    """save() writes bytes and returns the storage key."""

    async def test_save_writes_bytes(self, tmp_path: Path) -> None:
        storage = FilesystemStorage(str(tmp_path / "store"))
        await storage.save("test.bin", b"hello world")
        data = await storage.load("test.bin")
        assert data == b"hello world"

    async def test_save_returns_key(self, tmp_path: Path) -> None:
        storage = FilesystemStorage(str(tmp_path / "store"))
        key = await storage.save("test.bin", b"data")
        assert isinstance(key, str)
        assert key == "test.bin"

    async def test_save_creates_subdirectories(self, tmp_path: Path) -> None:
        storage = FilesystemStorage(str(tmp_path / "store"))
        key = await storage.save("sub/dir/file.txt", b"nested")
        assert key == "sub/dir/file.txt"
        data = await storage.load("sub/dir/file.txt")
        assert data == b"nested"

    async def test_save_overwrites_existing(self, tmp_path: Path) -> None:
        storage = FilesystemStorage(str(tmp_path / "store"))
        await storage.save("file.txt", b"first")
        await storage.save("file.txt", b"second")
        data = await storage.load("file.txt")
        assert data == b"second"


class TestFilesystemStorageLoad:
    """load() reads saved bytes correctly."""

    async def test_load_returns_saved_bytes(self, tmp_path: Path) -> None:
        storage = FilesystemStorage(str(tmp_path / "store"))
        await storage.save("data.bin", b"\x00\x01\x02\xff")
        result = await storage.load("data.bin")
        assert result == b"\x00\x01\x02\xff"

    async def test_load_missing_key_raises_file_not_found(self, tmp_path: Path) -> None:
        storage = FilesystemStorage(str(tmp_path / "store"))
        with pytest.raises(FileNotFoundError):
            await storage.load("nonexistent.bin")


class TestFilesystemStorageDelete:
    """delete() removes file; no error on non-existent."""

    async def test_delete_removes_file(self, tmp_path: Path) -> None:
        storage = FilesystemStorage(str(tmp_path / "store"))
        await storage.save("to_delete.bin", b"bye")
        assert await storage.exists("to_delete.bin") is True
        await storage.delete("to_delete.bin")
        assert await storage.exists("to_delete.bin") is False

    async def test_delete_non_existent_no_error(self, tmp_path: Path) -> None:
        storage = FilesystemStorage(str(tmp_path / "store"))
        # Should not raise
        await storage.delete("does_not_exist.bin")


class TestFilesystemStorageExists:
    """exists() returns True for saved, False for missing."""

    async def test_exists_true_for_saved(self, tmp_path: Path) -> None:
        storage = FilesystemStorage(str(tmp_path / "store"))
        await storage.save("present.bin", b"here")
        assert await storage.exists("present.bin") is True

    async def test_exists_false_for_missing(self, tmp_path: Path) -> None:
        storage = FilesystemStorage(str(tmp_path / "store"))
        assert await storage.exists("absent.bin") is False


class TestFilesystemStorageListKeys:
    """list_keys() lists files under prefix."""

    async def test_list_keys_returns_files(self, tmp_path: Path) -> None:
        storage = FilesystemStorage(str(tmp_path / "store"))
        await storage.save("ds1/a.csv", b"a")
        await storage.save("ds1/b.csv", b"b")
        await storage.save("ds2/c.csv", b"c")
        keys = await storage.list_keys("ds1")
        assert len(keys) == 2
        assert any("a.csv" in k for k in keys)
        assert any("b.csv" in k for k in keys)

    async def test_list_keys_empty_prefix_returns_empty(self, tmp_path: Path) -> None:
        storage = FilesystemStorage(str(tmp_path / "store"))
        keys = await storage.list_keys("nonexistent_prefix")
        assert keys == []


class TestFilesystemStorageProviderName:
    """provider_name returns 'filesystem'."""

    def test_provider_name(self, tmp_path: Path) -> None:
        storage = FilesystemStorage(str(tmp_path / "store"))
        assert storage.provider_name == "filesystem"


class TestFilesystemStoragePathTraversal:
    """_resolve() blocks path traversal attempts."""

    async def test_traversal_blocked_on_save(self, tmp_path: Path) -> None:
        storage = FilesystemStorage(str(tmp_path / "store"))
        with pytest.raises(ValueError, match="Path traversal"):
            await storage.save("../../etc/passwd", b"hack")

    async def test_traversal_blocked_on_load(self, tmp_path: Path) -> None:
        storage = FilesystemStorage(str(tmp_path / "store"))
        with pytest.raises(ValueError, match="Path traversal"):
            await storage.load("../../../secret.txt")

    async def test_traversal_blocked_on_delete(self, tmp_path: Path) -> None:
        storage = FilesystemStorage(str(tmp_path / "store"))
        with pytest.raises(ValueError, match="Path traversal"):
            await storage.delete("../../important.dat")

    async def test_traversal_blocked_on_exists(self, tmp_path: Path) -> None:
        storage = FilesystemStorage(str(tmp_path / "store"))
        with pytest.raises(ValueError, match="Path traversal"):
            await storage.exists("../../../etc/shadow")


class TestHashBytes:
    """hash_bytes() computes correct SHA-256."""

    def test_hash_bytes_correct_sha256(self) -> None:
        data = b"test data for hashing"
        expected = hashlib.sha256(data).hexdigest()
        assert StorageBackend.hash_bytes(data) == expected

    def test_hash_bytes_empty(self) -> None:
        expected = hashlib.sha256(b"").hexdigest()
        assert StorageBackend.hash_bytes(b"") == expected


# ======================================================================
# S3Storage (stub)
# ======================================================================


class TestS3Storage:
    """S3Storage stub raises NotImplementedError for all operations."""

    def test_provider_name(self) -> None:
        s = S3Storage(bucket="b", prefix="p/")
        assert s.provider_name == "s3"

    def test_instantiate_with_config(self) -> None:
        s = S3Storage(
            bucket="my-bucket",
            prefix="data/",
            endpoint_url="https://s3.example.com",
            access_key="ak",
            secret_key="sk",
            region="eu-west-1",
        )
        assert s.bucket == "my-bucket"
        assert s.region == "eu-west-1"

    async def test_save_raises_not_implemented(self) -> None:
        s = S3Storage()
        with pytest.raises(NotImplementedError):
            await s.save("k", b"d")

    async def test_load_raises_not_implemented(self) -> None:
        s = S3Storage()
        with pytest.raises(NotImplementedError):
            await s.load("k")

    async def test_delete_raises_not_implemented(self) -> None:
        s = S3Storage()
        with pytest.raises(NotImplementedError):
            await s.delete("k")

    async def test_exists_raises_not_implemented(self) -> None:
        s = S3Storage()
        with pytest.raises(NotImplementedError):
            await s.exists("k")

    async def test_list_keys_raises_not_implemented(self) -> None:
        s = S3Storage()
        with pytest.raises(NotImplementedError):
            await s.list_keys("prefix")


# ======================================================================
# GCSStorage (stub)
# ======================================================================


class TestGCSStorage:
    """GCSStorage stub raises NotImplementedError for all operations."""

    def test_provider_name(self) -> None:
        s = GCSStorage(bucket="b")
        assert s.provider_name == "gcs"

    def test_instantiate_with_config(self) -> None:
        s = GCSStorage(
            bucket="my-gcs-bucket",
            prefix="datasets/",
            credentials_path="/path/to/creds.json",
            project_id="my-project",
        )
        assert s.bucket == "my-gcs-bucket"
        assert s.project_id == "my-project"

    async def test_save_raises_not_implemented(self) -> None:
        s = GCSStorage()
        with pytest.raises(NotImplementedError):
            await s.save("k", b"d")

    async def test_load_raises_not_implemented(self) -> None:
        s = GCSStorage()
        with pytest.raises(NotImplementedError):
            await s.load("k")

    async def test_delete_raises_not_implemented(self) -> None:
        s = GCSStorage()
        with pytest.raises(NotImplementedError):
            await s.delete("k")

    async def test_exists_raises_not_implemented(self) -> None:
        s = GCSStorage()
        with pytest.raises(NotImplementedError):
            await s.exists("k")

    async def test_list_keys_raises_not_implemented(self) -> None:
        s = GCSStorage()
        with pytest.raises(NotImplementedError):
            await s.list_keys("prefix")


# ======================================================================
# AzureBlobStorage (stub)
# ======================================================================


class TestAzureBlobStorage:
    """AzureBlobStorage stub raises NotImplementedError for all operations."""

    def test_provider_name(self) -> None:
        s = AzureBlobStorage(container="c")
        assert s.provider_name == "azure"

    def test_instantiate_with_config(self) -> None:
        s = AzureBlobStorage(
            container="my-container",
            prefix="data/",
            connection_string="DefaultEndpointsProtocol=https;...",
        )
        assert s.container == "my-container"
        assert s.connection_string == "DefaultEndpointsProtocol=https;..."

    async def test_save_raises_not_implemented(self) -> None:
        s = AzureBlobStorage()
        with pytest.raises(NotImplementedError):
            await s.save("k", b"d")

    async def test_load_raises_not_implemented(self) -> None:
        s = AzureBlobStorage()
        with pytest.raises(NotImplementedError):
            await s.load("k")

    async def test_delete_raises_not_implemented(self) -> None:
        s = AzureBlobStorage()
        with pytest.raises(NotImplementedError):
            await s.delete("k")

    async def test_exists_raises_not_implemented(self) -> None:
        s = AzureBlobStorage()
        with pytest.raises(NotImplementedError):
            await s.exists("k")

    async def test_list_keys_raises_not_implemented(self) -> None:
        s = AzureBlobStorage()
        with pytest.raises(NotImplementedError):
            await s.list_keys("prefix")


# ======================================================================
# OVHSwiftStorage (stub)
# ======================================================================


class TestOVHSwiftStorage:
    """OVHSwiftStorage stub raises NotImplementedError for all operations."""

    def test_provider_name(self) -> None:
        s = OVHSwiftStorage(container="c")
        assert s.provider_name == "ovh"

    def test_instantiate_with_config(self) -> None:
        s = OVHSwiftStorage(
            container="my-container",
            prefix="datasets/",
            endpoint_url="https://swift.ovh.example.com",
            access_key="ak",
            secret_key="sk",
            region="GRA",
        )
        assert s.container == "my-container"
        assert s.region == "GRA"

    async def test_save_raises_not_implemented(self) -> None:
        s = OVHSwiftStorage()
        with pytest.raises(NotImplementedError):
            await s.save("k", b"d")

    async def test_load_raises_not_implemented(self) -> None:
        s = OVHSwiftStorage()
        with pytest.raises(NotImplementedError):
            await s.load("k")

    async def test_delete_raises_not_implemented(self) -> None:
        s = OVHSwiftStorage()
        with pytest.raises(NotImplementedError):
            await s.delete("k")

    async def test_exists_raises_not_implemented(self) -> None:
        s = OVHSwiftStorage()
        with pytest.raises(NotImplementedError):
            await s.exists("k")

    async def test_list_keys_raises_not_implemented(self) -> None:
        s = OVHSwiftStorage()
        with pytest.raises(NotImplementedError):
            await s.list_keys("prefix")
