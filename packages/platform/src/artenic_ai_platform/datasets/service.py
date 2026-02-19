"""Dataset management service — CRUD, files, versioning, stats, lineage."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import mimetypes
import os
import re
import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, func, select

from artenic_ai_platform.db.models import (
    DatasetFileRecord,
    DatasetLineageRecord,
    DatasetRecord,
    DatasetVersionRecord,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from artenic_ai_platform.datasets.storage import StorageBackend

logger = logging.getLogger(__name__)

_MAX_PREVIEW_LIMIT = 500
_MAX_STAT_FILE_BYTES = 100 * 1024 * 1024  # 100 MB per file for stats/preview
_SAFE_FILENAME_RE = re.compile(r"[^\w\s\-.]", re.ASCII)


def _sanitize_filename(name: str) -> str:
    """Strip path components and dangerous characters from a filename."""
    # Take only the basename (no directory traversal)
    name = os.path.basename(name)
    # Replace unsafe characters
    name = _SAFE_FILENAME_RE.sub("_", name)
    # Collapse multiple underscores/spaces
    name = re.sub(r"[_\s]+", "_", name).strip("_. ")
    return name or "unnamed"


class DatasetService:
    """Service layer for dataset CRUD, file management, versioning, and lineage."""

    def __init__(self, session: AsyncSession, storage: StorageBackend) -> None:
        self._session = session
        self._storage = storage

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def create(self, metadata: dict[str, Any]) -> str:
        """Create a new dataset.  Returns the dataset_id."""
        dataset_id = str(uuid.uuid4())
        record = DatasetRecord(
            id=dataset_id,
            name=metadata["name"],
            format=metadata["format"],
            description=metadata.get("description", ""),
            storage_backend=metadata.get("storage_backend", "filesystem"),
            source=metadata.get("source", ""),
            tags=metadata.get("tags", {}),
        )
        self._session.add(record)
        await self._session.commit()
        logger.info("Created dataset %s (%s)", record.name, dataset_id)
        return dataset_id

    async def get(self, dataset_id: str) -> DatasetRecord | None:
        """Get a dataset by ID."""
        result = await self._session.execute(
            select(DatasetRecord).where(DatasetRecord.id == dataset_id)
        )
        return result.scalar_one_or_none()

    async def list_all(
        self, *, offset: int = 0, limit: int = 100
    ) -> list[DatasetRecord]:
        """List datasets, ordered by creation date descending."""
        limit = min(max(limit, 1), 1000)
        offset = max(offset, 0)
        result = await self._session.execute(
            select(DatasetRecord)
            .order_by(DatasetRecord.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def update(
        self, dataset_id: str, updates: dict[str, Any]
    ) -> DatasetRecord:
        """Update metadata fields on a dataset."""
        record = await self.get(dataset_id)
        if record is None:
            msg = f"Dataset not found: {dataset_id}"
            raise ValueError(msg)
        allowed = {"name", "description", "tags", "source"}
        for key, value in updates.items():
            if key in allowed:
                setattr(record, key, value)
        await self._session.commit()
        await self._session.refresh(record)
        return record

    async def delete(self, dataset_id: str) -> None:
        """Delete a dataset and all associated files, versions, and lineage."""
        record = await self.get(dataset_id)
        if record is None:
            msg = f"Dataset not found: {dataset_id}"
            raise ValueError(msg)

        # Delete storage files
        files = await self.list_files(dataset_id)
        for f in files:
            await self._storage.delete(f.storage_path)

        # Delete DB records in order (lineage → files → versions → dataset)
        await self._session.execute(
            delete(DatasetLineageRecord).where(
                DatasetLineageRecord.dataset_id == dataset_id
            )
        )
        await self._session.execute(
            delete(DatasetFileRecord).where(
                DatasetFileRecord.dataset_id == dataset_id
            )
        )
        await self._session.execute(
            delete(DatasetVersionRecord).where(
                DatasetVersionRecord.dataset_id == dataset_id
            )
        )
        await self._session.delete(record)
        await self._session.commit()
        logger.info("Deleted dataset %s", dataset_id)

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    async def upload_file(
        self, dataset_id: str, filename: str, data: bytes
    ) -> DatasetFileRecord:
        """Upload a file to a dataset."""
        record = await self.get(dataset_id)
        if record is None:
            msg = f"Dataset not found: {dataset_id}"
            raise ValueError(msg)

        filename = _sanitize_filename(filename)
        storage_key = f"{dataset_id}/{filename}"
        storage_path = await self._storage.save(storage_key, data)
        file_hash = self._storage.hash_bytes(data)
        mime = mimetypes.guess_type(filename)[0] or ""

        file_record = DatasetFileRecord(
            dataset_id=dataset_id,
            version=record.current_version,
            filename=filename,
            mime_type=mime,
            size_bytes=len(data),
            hash=file_hash,
            storage_path=storage_path,
        )
        self._session.add(file_record)

        # Update aggregates (autoflush includes the newly-added record)
        record.total_files = (
            await self._session.execute(
                select(func.count(DatasetFileRecord.id)).where(
                    DatasetFileRecord.dataset_id == dataset_id
                )
            )
        ).scalar_one()
        record.total_size_bytes = (
            await self._session.execute(
                select(func.coalesce(func.sum(DatasetFileRecord.size_bytes), 0)).where(
                    DatasetFileRecord.dataset_id == dataset_id
                )
            )
        ).scalar_one()

        await self._session.commit()
        await self._session.refresh(file_record)
        logger.info("Uploaded %s to dataset %s (%d bytes)", filename, dataset_id, len(data))
        return file_record

    async def list_files(
        self, dataset_id: str, *, version: int | None = None
    ) -> list[DatasetFileRecord]:
        """List files in a dataset, optionally filtered by version."""
        stmt = select(DatasetFileRecord).where(
            DatasetFileRecord.dataset_id == dataset_id
        )
        if version is not None:
            stmt = stmt.where(DatasetFileRecord.version <= version)
        stmt = stmt.order_by(DatasetFileRecord.filename)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def download_file(self, dataset_id: str, filename: str) -> bytes:
        """Download a file from a dataset."""
        result = await self._session.execute(
            select(DatasetFileRecord).where(
                DatasetFileRecord.dataset_id == dataset_id,
                DatasetFileRecord.filename == filename,
            )
        )
        file_record = result.scalar_one_or_none()
        if file_record is None:
            msg = f"File not found: {filename} in dataset {dataset_id}"
            raise FileNotFoundError(msg)
        return await self._storage.load(file_record.storage_path)

    async def delete_file(self, dataset_id: str, filename: str) -> None:
        """Delete a file from a dataset."""
        result = await self._session.execute(
            select(DatasetFileRecord).where(
                DatasetFileRecord.dataset_id == dataset_id,
                DatasetFileRecord.filename == filename,
            )
        )
        file_record = result.scalar_one_or_none()
        if file_record is None:
            msg = f"File not found: {filename} in dataset {dataset_id}"
            raise FileNotFoundError(msg)

        await self._storage.delete(file_record.storage_path)
        await self._session.delete(file_record)

        # Refresh aggregates
        dataset = await self.get(dataset_id)
        if dataset is not None:
            agg = await self._session.execute(
                select(
                    func.count(DatasetFileRecord.id),
                    func.coalesce(func.sum(DatasetFileRecord.size_bytes), 0),
                ).where(DatasetFileRecord.dataset_id == dataset_id)
            )
            row = agg.one()
            dataset.total_files = row[0]
            dataset.total_size_bytes = row[1]

        await self._session.commit()

    # ------------------------------------------------------------------
    # Versioning
    # ------------------------------------------------------------------

    async def create_version(
        self, dataset_id: str, change_summary: str = ""
    ) -> DatasetVersionRecord:
        """Create a new immutable version snapshot of the dataset."""
        dataset = await self.get(dataset_id)
        if dataset is None:
            msg = f"Dataset not found: {dataset_id}"
            raise ValueError(msg)

        new_version = dataset.current_version + 1
        files = await self.list_files(dataset_id)

        # Compute composite hash from all file hashes
        hasher = hashlib.sha256()
        for f in sorted(files, key=lambda x: x.filename):
            hasher.update(f.hash.encode())
        composite_hash = hasher.hexdigest()

        total_size = sum(f.size_bytes for f in files)

        version_record = DatasetVersionRecord(
            dataset_id=dataset_id,
            version=new_version,
            hash=composite_hash,
            size_bytes=total_size,
            num_files=len(files),
            change_summary=change_summary,
        )
        self._session.add(version_record)

        dataset.current_version = new_version
        await self._session.commit()
        await self._session.refresh(version_record)
        logger.info("Created version %d for dataset %s", new_version, dataset_id)
        return version_record

    async def list_versions(self, dataset_id: str) -> list[DatasetVersionRecord]:
        """List all versions of a dataset."""
        result = await self._session.execute(
            select(DatasetVersionRecord)
            .where(DatasetVersionRecord.dataset_id == dataset_id)
            .order_by(DatasetVersionRecord.version.desc())
        )
        return list(result.scalars().all())

    async def get_version(
        self, dataset_id: str, version: int
    ) -> DatasetVersionRecord | None:
        """Get a specific version of a dataset."""
        result = await self._session.execute(
            select(DatasetVersionRecord).where(
                DatasetVersionRecord.dataset_id == dataset_id,
                DatasetVersionRecord.version == version,
            )
        )
        return result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # Stats & Preview
    # ------------------------------------------------------------------

    async def compute_stats(self, dataset_id: str) -> dict[str, Any]:
        """Compute basic statistics for a dataset."""
        dataset = await self.get(dataset_id)
        if dataset is None:
            msg = f"Dataset not found: {dataset_id}"
            raise ValueError(msg)

        files = await self.list_files(dataset_id)
        format_breakdown: dict[str, int] = {}
        for f in files:
            ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else "unknown"
            format_breakdown[ext] = format_breakdown.get(ext, 0) + 1

        stats: dict[str, Any] = {
            "total_size_bytes": dataset.total_size_bytes,
            "total_files": dataset.total_files,
            "format_breakdown": format_breakdown,
        }

        # Attempt to count records for tabular formats
        if dataset.format in ("csv", "parquet", "json", "jsonl"):
            num_records = await self._count_tabular_records(dataset_id, dataset.format)
            if num_records is not None:
                stats["num_records"] = num_records

        if dataset.schema_info:
            stats["schema_info"] = dataset.schema_info

        return stats

    async def _count_tabular_records(
        self, dataset_id: str, fmt: str
    ) -> int | None:
        """Count records across tabular files in a dataset."""
        files = await self.list_files(dataset_id)
        total = 0
        for f in files:
            if f.size_bytes > _MAX_STAT_FILE_BYTES:
                continue
            try:
                data = await self._storage.load(f.storage_path)
                count = self._count_records_in_bytes(data, fmt)
                if count is not None:
                    total += count
            except (FileNotFoundError, ValueError):
                continue
        return total if total > 0 else None

    @staticmethod
    def _count_records_in_bytes(data: bytes, fmt: str) -> int | None:
        """Count records in raw bytes for supported tabular formats."""
        if fmt == "csv":
            text = data.decode("utf-8", errors="replace")
            reader = csv.reader(io.StringIO(text))
            rows = sum(1 for _ in reader)
            return max(rows - 1, 0)  # Subtract header
        if fmt in ("json", "jsonl"):
            text = data.decode("utf-8", errors="replace").strip()
            if not text:
                return 0
            if text.startswith("["):
                parsed = json.loads(text)
                return len(parsed) if isinstance(parsed, list) else None
            return sum(1 for line in text.splitlines() if line.strip())
        return None

    async def preview(
        self, dataset_id: str, *, limit: int = 50
    ) -> dict[str, Any]:
        """Return the first *limit* rows for tabular datasets."""
        limit = min(max(limit, 1), _MAX_PREVIEW_LIMIT)

        dataset = await self.get(dataset_id)
        if dataset is None:
            msg = f"Dataset not found: {dataset_id}"
            raise ValueError(msg)

        if dataset.format not in ("csv", "json", "jsonl"):
            return {
                "columns": [],
                "rows": [],
                "total_rows": 0,
                "truncated": False,
            }

        files = await self.list_files(dataset_id)
        if not files:
            return {"columns": [], "rows": [], "total_rows": 0, "truncated": False}

        # Preview from first file (skip files that are too large)
        first = files[0]
        if first.size_bytes > _MAX_STAT_FILE_BYTES:
            return {"columns": [], "rows": [], "total_rows": 0, "truncated": False}
        data = await self._storage.load(first.storage_path)
        return self._parse_preview(data, dataset.format, limit)

    @staticmethod
    def _parse_preview(
        data: bytes, fmt: str, limit: int
    ) -> dict[str, Any]:
        """Parse preview rows from raw bytes."""
        text = data.decode("utf-8", errors="replace")

        if fmt == "csv":
            reader = csv.DictReader(io.StringIO(text))
            columns = reader.fieldnames or []
            rows = []
            for i, row in enumerate(reader):
                if i >= limit:
                    break
                rows.append(dict(row))
            return {
                "columns": list(columns),
                "rows": rows,
                "total_rows": sum(1 for _ in csv.reader(io.StringIO(text))) - 1,
                "truncated": len(rows) >= limit,
            }

        if fmt in ("json", "jsonl"):
            text = text.strip()
            if text.startswith("["):
                parsed = json.loads(text)
                if not isinstance(parsed, list) or not parsed:
                    return {"columns": [], "rows": [], "total_rows": 0, "truncated": False}
                columns = list(parsed[0].keys()) if isinstance(parsed[0], dict) else []
                rows = parsed[:limit]
                return {
                    "columns": columns,
                    "rows": rows,
                    "total_rows": len(parsed),
                    "truncated": len(parsed) > limit,
                }
            # JSONL
            lines = [line for line in text.splitlines() if line.strip()]
            rows = []
            for line in lines[:limit]:
                rows.append(json.loads(line))
            columns = list(rows[0].keys()) if rows and isinstance(rows[0], dict) else []
            return {
                "columns": columns,
                "rows": rows,
                "total_rows": len(lines),
                "truncated": len(lines) > limit,
            }

        return {"columns": [], "rows": [], "total_rows": 0, "truncated": False}

    # ------------------------------------------------------------------
    # Lineage
    # ------------------------------------------------------------------

    async def add_lineage(
        self,
        dataset_id: str,
        dataset_version: int,
        entity_type: str,
        entity_id: str,
        role: str = "input",
    ) -> DatasetLineageRecord:
        """Link a dataset version to a model or training job."""
        dataset = await self.get(dataset_id)
        if dataset is None:
            msg = f"Dataset not found: {dataset_id}"
            raise ValueError(msg)

        record = DatasetLineageRecord(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            entity_type=entity_type,
            entity_id=entity_id,
            role=role,
        )
        self._session.add(record)
        await self._session.commit()
        await self._session.refresh(record)
        return record

    async def get_lineage(self, dataset_id: str) -> list[DatasetLineageRecord]:
        """Get all lineage records for a dataset."""
        result = await self._session.execute(
            select(DatasetLineageRecord)
            .where(DatasetLineageRecord.dataset_id == dataset_id)
            .order_by(DatasetLineageRecord.created_at.desc())
        )
        return list(result.scalars().all())
