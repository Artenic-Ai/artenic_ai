import { useParams } from "react-router";

import { Download } from "lucide-react";

import { PageShell } from "@/components/layout/page-shell";
import { Badge } from "@/components/ui/badge";
import { Breadcrumb } from "@/components/ui/breadcrumb";
import { Card } from "@/components/ui/card";
import { type Column, DataTable } from "@/components/ui/data-table";
import { DetailRow } from "@/components/ui/detail-row";
import { ErrorState } from "@/components/ui/error-state";
import { DetailSkeleton } from "@/components/ui/skeleton";
import {
  useDataset,
  useDatasetFiles,
  useDatasetLineage,
  useDatasetPreview,
  useDatasetStats,
  useDatasetVersions,
} from "@/hooks/use-datasets";
import { formatBytes, formatDateTime } from "@/lib/format";
import type { DatasetFile, DatasetLineage, DatasetVersion } from "@/types/api";

function truncateHash(hash: string): string {
  return hash.length > 12 ? `${hash.slice(0, 12)}...` : hash;
}

const TABULAR_FORMATS = new Set(["csv", "parquet", "json", "jsonl", "tsv"]);

function getFileColumns(datasetId: string): Column<DatasetFile>[] {
  return [
    {
      key: "filename",
      header: "Filename",
      sortable: true,
      sortValue: (r) => r.filename,
      render: (r) => (
        <span className="font-mono text-text-primary">{r.filename}</span>
      ),
    },
    {
      key: "mime_type",
      header: "Type",
      render: (r) => (
        <span className="text-text-secondary">{r.mime_type}</span>
      ),
    },
    {
      key: "size",
      header: "Size",
      sortable: true,
      sortValue: (r) => r.size_bytes,
      render: (r) => (
        <span className="text-text-secondary">
          {formatBytes(r.size_bytes)}
        </span>
      ),
    },
    {
      key: "hash",
      header: "Hash",
      render: (r) => (
        <span className="font-mono text-xs text-text-muted">
          {truncateHash(r.hash)}
        </span>
      ),
    },
    {
      key: "download",
      header: "",
      render: (r) => (
        <a
          href={`/datasets/${datasetId}/files/${r.id}/download`}
          className="inline-flex items-center gap-1 text-xs text-accent hover:underline"
          onClick={(e) => e.stopPropagation()}
          download
        >
          <Download size={12} />
          Download
        </a>
      ),
    },
  ];
}

export function DatasetDetailPage() {
  const { datasetId } = useParams();
  const id = datasetId ?? "";
  const { data: dataset, isLoading, isError, refetch } = useDataset(id);
  const { data: files } = useDatasetFiles(id);
  const { data: versions } = useDatasetVersions(id);
  const { data: stats } = useDatasetStats(id);
  const { data: preview } = useDatasetPreview(id);
  const { data: lineage } = useDatasetLineage(id);

  if (isLoading) {
    return (
      <PageShell title="">
        <DetailSkeleton />
      </PageShell>
    );
  }
  if (isError) {
    return (
      <ErrorState
        message="Failed to load dataset."
        onRetry={() => void refetch()}
      />
    );
  }
  if (!dataset) {
    return (
      <div className="flex h-64 items-center justify-center">
        <p className="text-text-muted">Dataset not found.</p>
      </div>
    );
  }

  const isTabular = TABULAR_FORMATS.has(dataset.format);
  const schemaInfo = dataset.schema_info ?? stats?.schema_info;

  const previewColumns: Column<Record<string, unknown>>[] = (
    preview?.columns ?? []
  ).map((col) => ({
    key: col,
    header: col,
    render: (row: Record<string, unknown>) => (
      <span className="text-text-secondary">
        {String(row[col] ?? "")}
      </span>
    ),
  }));

  return (
    <PageShell
      title={dataset.name}
      description={`${dataset.format} \u2014 v${dataset.current_version}`}
      breadcrumb={
        <Breadcrumb
          items={[
            { label: "Datasets", to: "/datasets" },
            { label: dataset.name },
          ]}
        />
      }
    >
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* ── Details Card ──────────────────────────────────────────────── */}
        <Card title="Details">
          <dl className="space-y-3">
            <DetailRow label="Dataset ID" value={dataset.id} />
            <DetailRow label="Name" value={dataset.name} />
            <DetailRow
              label="Format"
              value={<Badge value={dataset.format} />}
            />
            <DetailRow label="Storage" value={dataset.storage_backend} />
            <DetailRow label="Source" value={dataset.source || "\u2014"} />
            <DetailRow
              label="Created"
              value={formatDateTime(dataset.created_at)}
            />
            {dataset.updated_at && (
              <DetailRow
                label="Updated"
                value={formatDateTime(dataset.updated_at)}
              />
            )}
          </dl>
          {Object.keys(dataset.tags).length > 0 && (
            <div className="mt-4">
              <h4 className="text-xs font-medium text-text-muted">Tags</h4>
              <div className="mt-2 flex flex-wrap gap-2">
                {Object.entries(dataset.tags).map(([k, v]) => (
                  <span
                    key={k}
                    className="rounded-md border border-border bg-surface-2 px-2 py-1 text-xs text-text-secondary"
                  >
                    {k}: {v}
                  </span>
                ))}
              </div>
            </div>
          )}
        </Card>

        {/* ── Statistics Card ──────────────────────────────────────────── */}
        {stats && (
          <Card title="Statistics">
            <dl className="space-y-3">
              <DetailRow
                label="Total Size"
                value={formatBytes(stats.total_size_bytes)}
              />
              <DetailRow
                label="Total Files"
                value={stats.total_files.toLocaleString()}
              />
              {stats.num_records != null && (
                <DetailRow
                  label="Records"
                  value={stats.num_records.toLocaleString()}
                />
              )}
            </dl>
            {Object.keys(stats.format_breakdown).length > 0 && (
              <div className="mt-4">
                <h4 className="text-xs font-medium text-text-muted">
                  Format Breakdown
                </h4>
                <div className="mt-2 flex flex-wrap gap-2">
                  {Object.entries(stats.format_breakdown).map(
                    ([fmt, count]) => (
                      <span
                        key={fmt}
                        className="rounded-md border border-border bg-surface-2 px-2 py-1 text-xs text-text-secondary"
                      >
                        {fmt}: {count.toLocaleString()}
                      </span>
                    ),
                  )}
                </div>
              </div>
            )}
          </Card>
        )}

        {/* ── Schema Card ──────────────────────────────────────────────── */}
        {schemaInfo && schemaInfo.columns.length > 0 && (
          <Card title="Schema" className="lg:col-span-2">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-border">
                    <th className="px-4 py-2 text-xs font-medium text-text-secondary">
                      Column
                    </th>
                    <th className="px-4 py-2 text-xs font-medium text-text-secondary">
                      Type
                    </th>
                    <th className="px-4 py-2 text-xs font-medium text-text-secondary">
                      Nullable
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {schemaInfo.columns.map((col) => (
                    <tr
                      key={col.name}
                      className="border-b border-border last:border-0"
                    >
                      <td className="px-4 py-2 font-mono text-text-primary">
                        {col.name}
                      </td>
                      <td className="px-4 py-2 text-text-secondary">
                        {col.dtype}
                      </td>
                      <td className="px-4 py-2">
                        <Badge value={col.nullable ? "yes" : "no"} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        )}

        {/* ── Preview Card ─────────────────────────────────────────────── */}
        {isTabular && preview && preview.rows.length > 0 && (
          <Card title="Preview" className="lg:col-span-2">
            <DataTable
              columns={previewColumns}
              data={preview.rows.map((row, i) => ({
                ...row,
                __key: String(i),
              }))}
              keyFn={(r) => String(r.__key)}
              pageSize={5}
            />
            {preview.truncated && (
              <p className="mt-2 text-xs text-text-muted">
                Showing {preview.rows.length} of{" "}
                {preview.total_rows.toLocaleString()} rows (truncated)
              </p>
            )}
          </Card>
        )}

        {/* ── Files Card ───────────────────────────────────────────────── */}
        {files && files.length > 0 && (
          <Card title="Files" className="lg:col-span-2">
            <DataTable
              columns={getFileColumns(id)}
              data={files}
              keyFn={(r) => String(r.id)}
            />
          </Card>
        )}

        {/* ── Version History Card ─────────────────────────────────────── */}
        {versions && versions.length > 0 && (
          <Card title="Version History" className="lg:col-span-2">
            <div className="space-y-3">
              {versions.map((v: DatasetVersion) => (
                <div
                  key={v.id}
                  className="flex items-start justify-between rounded-md border border-border bg-surface-2 px-4 py-3"
                >
                  <div>
                    <span className="text-sm font-medium text-text-primary">
                      v{v.version}
                    </span>
                    <span className="ml-2 font-mono text-xs text-text-muted">
                      {truncateHash(v.hash)}
                    </span>
                    <p className="mt-1 text-xs text-text-secondary">
                      {v.change_summary}
                    </p>
                    <div className="mt-1 flex gap-3 text-xs text-text-muted">
                      <span>{formatBytes(v.size_bytes)}</span>
                      <span>
                        {v.num_files} file{v.num_files !== 1 ? "s" : ""}
                      </span>
                      {v.num_records != null && (
                        <span>{v.num_records.toLocaleString()} records</span>
                      )}
                    </div>
                  </div>
                  <span className="whitespace-nowrap text-xs text-text-muted">
                    {formatDateTime(v.created_at)}
                  </span>
                </div>
              ))}
            </div>
          </Card>
        )}

        {/* ── Lineage Card ─────────────────────────────────────────────── */}
        {lineage && lineage.length > 0 && (
          <Card title="Lineage" className="lg:col-span-2">
            <div className="space-y-3">
              {lineage.map((entry: DatasetLineage) => (
                <div
                  key={entry.id}
                  className="flex items-center justify-between rounded-md border border-border bg-surface-2 px-4 py-3"
                >
                  <div className="flex items-center gap-3">
                    <Badge value={entry.entity_type} />
                    <span className="font-mono text-sm text-accent">
                      {entry.entity_id}
                    </span>
                    <Badge value={entry.role} />
                    <span className="text-xs text-text-muted">
                      v{entry.dataset_version}
                    </span>
                  </div>
                  <span className="text-xs text-text-muted">
                    {formatDateTime(entry.created_at)}
                  </span>
                </div>
              ))}
            </div>
          </Card>
        )}
      </div>
    </PageShell>
  );
}
