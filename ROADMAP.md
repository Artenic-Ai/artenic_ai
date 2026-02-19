# Roadmap

## Datasets

### v1 (0.5.0) — Shipped

- CRUD: create, list, get, update, delete datasets
- File management: upload (multipart), download, list, delete
- Versioning: immutable snapshots with SHA-256 composite hashes
- Auto-stats: record counts for CSV/JSON/JSONL, format breakdown
- Tabular preview: first N rows for CSV/JSON/JSONL
- Lineage: link datasets to models and training jobs (input/output/evaluation)
- Storage: filesystem (default), cloud backend stubs (S3, GCS, Azure, OVH)
- CLI: 13 commands covering all operations
- Dashboard: list page, detail page with metadata/stats/schema/preview/files/versions/lineage

### v2 — Cloud Storage & Smart Features

**Cloud Storage Implementation**
- Implement S3Storage with `boto3` (async wrapper)
- Implement GCSStorage with `google-cloud-storage`
- Implement AzureBlobStorage with `azure-storage-blob`
- Implement OVHSwiftStorage with S3-compatible API
- Storage health check: verify connectivity and permissions on startup
- Storage usage endpoint: available/used space per provider
- Dashboard: show usage gauges, gray out providers without enough space

**Data Quality & Validation**
- Schema inference for CSV/Parquet (column names, dtypes, nullable)
- Schema validation on upload (reject files that don't match dataset schema)
- Data profiling: min/max/mean/std, null counts, unique counts, distributions
- Data quality score (completeness, consistency, timeliness)

**Search & Filtering**
- Full-text search on dataset name/description/tags
- Filter by format, storage backend, size range, date range
- Sort by any column (name, size, files, version, created)
- Pagination with cursor-based API

**Improved Versioning**
- Diff between versions (added/removed/modified files)
- Rollback to a previous version
- Version tags (e.g., "production", "staging", "candidate")
- Auto-version on upload (optional)

### v3 — Intelligence & Scale

**Smart Recommendations**
- Recommend storage backend based on dataset size and access patterns
- Recommend format conversion (e.g., CSV to Parquet for large datasets)
- Cost estimation per storage backend
- Automatic tiering: move cold data to cheaper storage

**Data Transformations**
- Format conversion pipeline (CSV -> Parquet, JSON -> JSONL, etc.)
- Sampling: create subset datasets for experimentation
- Split management: train/val/test splits with reproducible seeds
- Augmentation hooks: register transformation functions

**Collaboration**
- Dataset access control (owner, read, write permissions)
- Dataset sharing between teams/projects
- Comments and annotations on datasets
- Activity feed: who uploaded/modified/downloaded what

**Scale**
- Chunked/resumable upload for large files (>1 GB)
- Streaming download with range requests
- Background jobs for stats computation on large datasets
- Parquet support: columnar stats, row group metadata, predicate pushdown preview

---

## Platform

### Next Steps

- **Optimizer** (`packages/optimizer/`) — LTR-based training instance selection
- **Alembic migrations** — production DB schema management
- **Multi-tenancy** — workspace isolation, API key scoping
- **Webhooks v2** — event-driven notifications for dataset/model/training events
- **Audit trail** — comprehensive logging of all API operations

---

## Dashboard

### Next Steps

- **Real-time updates** — WebSocket integration for live job status, health alerts
- **User management** — login, roles, team workspaces
- **Responsive design** — mobile/tablet layouts
- **Export** — CSV/PDF export for reports, charts, and tables
- **Theming** — light mode option, custom color palettes
