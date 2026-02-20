"""Dataset service — CRUD, file management, stats, preview."""

from __future__ import annotations

import csv
import io
import json
import logging
import mimetypes
import os
import re
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, func, select

from artenic_ai_platform.db.models import MLDataset, MLDatasetFile
from artenic_ai_platform.entities.base_service import GenericEntityService
from artenic_ai_platform.entities.schemas import DATASET_TRANSITIONS

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from artenic_ai_platform.entities.datasets.storage import StorageBackend

logger = logging.getLogger(__name__)

_MAX_PREVIEW_LIMIT = 500
_MAX_STAT_FILE_BYTES = 100 * 1024 * 1024  # 100 MB
_SAFE_FILENAME_RE = re.compile(r"[^\w\s\-.]", re.ASCII)


def _sanitize_filename(name: str) -> str:
    """Strip path components and dangerous characters from a filename."""
    name = os.path.basename(name)
    name = _SAFE_FILENAME_RE.sub("_", name)
    name = re.sub(r"[_\s]+", "_", name).strip("_. ")
    return name or "unnamed"


class DatasetService(GenericEntityService[MLDataset]):
    """Extended service for datasets with file operations, stats, and preview."""

    _model_class = MLDataset

    def __init__(self, session: AsyncSession, storage: StorageBackend) -> None:
        super().__init__(session)
        self._storage = storage

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def change_status(self, entity_id: str, new_status: str) -> MLDataset:
        """Transition dataset to *new_status*."""
        record = await self.get_or_raise(entity_id)
        allowed = DATASET_TRANSITIONS.get(record.status, set())
        if new_status not in allowed:
            msg = f"Cannot transition from {record.status} to {new_status}"
            raise ValueError(msg)
        record.status = new_status
        await self._session.commit()
        await self._session.refresh(record)
        return record

    # ------------------------------------------------------------------
    # Delete (override — also removes physical files)
    # ------------------------------------------------------------------

    async def delete(self, entity_id: str) -> None:
        """Delete a dataset and all associated files."""
        record = await self.get_or_raise(entity_id)
        files = await self.list_files(entity_id)
        for f in files:
            await self._storage.delete(f.storage_path)
        await self._session.execute(
            delete(MLDatasetFile).where(MLDatasetFile.dataset_id == entity_id)
        )
        await self._session.delete(record)
        await self._session.commit()
        logger.info("Deleted dataset %s", entity_id)

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    async def upload_file(self, dataset_id: str, filename: str, data: bytes) -> MLDatasetFile:
        """Upload a file to a dataset."""
        record = await self.get_or_raise(dataset_id)
        filename = _sanitize_filename(filename)
        storage_key = f"{dataset_id}/{filename}"
        storage_path = await self._storage.save(storage_key, data)
        file_hash = self._storage.hash_bytes(data)
        mime = mimetypes.guess_type(filename)[0] or ""

        file_record = MLDatasetFile(
            dataset_id=dataset_id,
            path=filename,
            sha256=file_hash,
            size_bytes=len(data),
            storage_path=storage_path,
        )
        # Count records for tabular formats
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext in ("csv", "json", "jsonl"):
            file_record.num_records = self._count_records_in_bytes(data, ext)

        self._session.add(file_record)
        await self._refresh_aggregates(record)
        await self._session.commit()
        await self._session.refresh(file_record)
        logger.info(
            "Uploaded %s to dataset %s (%d bytes, mime=%s)",
            filename,
            dataset_id,
            len(data),
            mime,
        )
        return file_record

    async def list_files(self, dataset_id: str) -> list[MLDatasetFile]:
        """List all files in a dataset."""
        result = await self._session.execute(
            select(MLDatasetFile)
            .where(MLDatasetFile.dataset_id == dataset_id)
            .order_by(MLDatasetFile.path)
        )
        return list(result.scalars().all())

    async def download_file(self, dataset_id: str, path: str) -> bytes:
        """Download a file by path from a dataset."""
        file_record = await self._get_file(dataset_id, path)
        return await self._storage.load(file_record.storage_path)

    async def delete_file(self, dataset_id: str, path: str) -> None:
        """Delete a single file from a dataset."""
        file_record = await self._get_file(dataset_id, path)
        await self._storage.delete(file_record.storage_path)
        await self._session.delete(file_record)

        record = await self.get(dataset_id)
        if record is not None:
            await self._refresh_aggregates(record)
        await self._session.commit()

    async def _get_file(self, dataset_id: str, path: str) -> MLDatasetFile:
        result = await self._session.execute(
            select(MLDatasetFile).where(
                MLDatasetFile.dataset_id == dataset_id,
                MLDatasetFile.path == path,
            )
        )
        record = result.scalar_one_or_none()
        if record is None:
            msg = f"File not found: {path} in dataset {dataset_id}"
            raise FileNotFoundError(msg)
        return record

    async def _refresh_aggregates(self, record: MLDataset) -> None:
        agg = await self._session.execute(
            select(
                func.count(MLDatasetFile.id),
                func.coalesce(func.sum(MLDatasetFile.size_bytes), 0),
            ).where(MLDatasetFile.dataset_id == record.id)
        )
        row = agg.one()
        record.total_size_bytes = row[1]

    # ------------------------------------------------------------------
    # Stats & Preview
    # ------------------------------------------------------------------

    async def compute_stats(self, dataset_id: str) -> dict[str, Any]:
        """Compute basic statistics for a dataset."""
        record = await self.get_or_raise(dataset_id)
        files = await self.list_files(dataset_id)
        format_breakdown: dict[str, int] = {}
        for f in files:
            ext = f.path.rsplit(".", 1)[-1].lower() if "." in f.path else "unknown"
            format_breakdown[ext] = format_breakdown.get(ext, 0) + 1

        stats: dict[str, Any] = {
            "total_size_bytes": record.total_size_bytes,
            "total_files": len(files),
            "format_breakdown": format_breakdown,
        }

        if record.format in ("csv", "parquet", "json", "jsonl"):
            num_records = await self._count_tabular_records(dataset_id, record.format)
            if num_records is not None:
                stats["num_records"] = num_records

        return stats

    async def _count_tabular_records(self, dataset_id: str, fmt: str) -> int | None:
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
        if fmt == "csv":
            text = data.decode("utf-8", errors="replace")
            reader = csv.reader(io.StringIO(text))
            rows = sum(1 for _ in reader)
            return max(rows - 1, 0)
        if fmt in ("json", "jsonl"):
            text = data.decode("utf-8", errors="replace").strip()
            if not text:
                return 0
            if text.startswith("["):
                parsed = json.loads(text)
                return len(parsed) if isinstance(parsed, list) else None
            return sum(1 for line in text.splitlines() if line.strip())
        return None

    async def preview(self, dataset_id: str, *, limit: int = 50) -> dict[str, Any]:
        """Return the first *limit* rows for tabular datasets."""
        limit = min(max(limit, 1), _MAX_PREVIEW_LIMIT)
        record = await self.get_or_raise(dataset_id)

        if record.format not in ("csv", "json", "jsonl"):
            return {"columns": [], "rows": [], "total_rows": 0, "truncated": False}

        files = await self.list_files(dataset_id)
        if not files:
            return {"columns": [], "rows": [], "total_rows": 0, "truncated": False}

        first = files[0]
        if first.size_bytes > _MAX_STAT_FILE_BYTES:
            return {"columns": [], "rows": [], "total_rows": 0, "truncated": False}
        data = await self._storage.load(first.storage_path)
        return self._parse_preview(data, record.format, limit)

    @staticmethod
    def _parse_preview(data: bytes, fmt: str, limit: int) -> dict[str, Any]:
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
            lines = [line for line in text.splitlines() if line.strip()]
            rows = [json.loads(line) for line in lines[:limit]]
            columns = list(rows[0].keys()) if rows and isinstance(rows[0], dict) else []
            return {
                "columns": columns,
                "rows": rows,
                "total_rows": len(lines),
                "truncated": len(lines) > limit,
            }

        return {"columns": [], "rows": [], "total_rows": 0, "truncated": False}  # pragma: no cover
