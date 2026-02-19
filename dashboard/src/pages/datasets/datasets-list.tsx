import { useState } from "react";
import { useNavigate } from "react-router";

import { Database, Plus } from "lucide-react";

import { PageShell } from "@/components/layout/page-shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { type Column, DataTable } from "@/components/ui/data-table";
import { EmptyState } from "@/components/ui/empty-state";
import { ErrorState } from "@/components/ui/error-state";
import { TableSkeleton } from "@/components/ui/skeleton";
import { useDatasets } from "@/hooks/use-datasets";
import { formatBytes, formatDate } from "@/lib/format";
import type { Dataset } from "@/types/api";

import { DatasetCreateDialog } from "./dataset-create-dialog";

const COLUMNS: Column<Dataset>[] = [
  {
    key: "name",
    header: "Name",
    sortable: true,
    sortValue: (r) => r.name,
    render: (r) => (
      <div>
        <span className="font-medium text-text-primary">{r.name}</span>
        <p className="mt-0.5 text-xs text-text-muted line-clamp-1">
          {r.description}
        </p>
      </div>
    ),
  },
  {
    key: "format",
    header: "Format",
    sortable: true,
    sortValue: (r) => r.format,
    render: (r) => <Badge value={r.format} />,
  },
  {
    key: "storage",
    header: "Storage",
    render: (r) => (
      <span className="text-text-secondary">{r.storage_backend}</span>
    ),
  },
  {
    key: "files",
    header: "Files",
    sortable: true,
    sortValue: (r) => r.total_files,
    render: (r) => (
      <span className="text-text-secondary">
        {r.total_files.toLocaleString()}
      </span>
    ),
  },
  {
    key: "size",
    header: "Size",
    sortable: true,
    sortValue: (r) => r.total_size_bytes,
    render: (r) => (
      <span className="text-text-secondary">
        {formatBytes(r.total_size_bytes)}
      </span>
    ),
  },
  {
    key: "version",
    header: "Version",
    sortable: true,
    sortValue: (r) => r.current_version,
    render: (r) => (
      <span className="text-text-muted">v{r.current_version}</span>
    ),
  },
  {
    key: "created_at",
    header: "Created",
    sortable: true,
    sortValue: (r) => r.created_at,
    render: (r) => (
      <span className="text-text-muted">{formatDate(r.created_at)}</span>
    ),
  },
];

export function DatasetsListPage() {
  const { data, isLoading, isError, refetch } = useDatasets();
  const navigate = useNavigate();
  const [dialogOpen, setDialogOpen] = useState(false);

  if (isLoading) {
    return (
      <PageShell
        title="Datasets"
        description="Manage datasets, versions, and storage backends."
      >
        <TableSkeleton rows={6} cols={7} />
      </PageShell>
    );
  }
  if (isError)
    return (
      <PageShell title="Datasets">
        <ErrorState
          message="Failed to load datasets."
          onRetry={() => void refetch()}
        />
      </PageShell>
    );

  return (
    <PageShell
      title="Datasets"
      description="Manage datasets, versions, and storage backends."
      actions={
        <Button onClick={() => setDialogOpen(true)}>
          <Plus size={16} className="mr-1" />
          Create Dataset
        </Button>
      }
    >
      {data && data.length > 0 ? (
        <DataTable
          columns={COLUMNS}
          data={data}
          keyFn={(r) => r.id}
          onRowClick={(r) => navigate(`/datasets/${r.id}`)}
        />
      ) : (
        <EmptyState
          icon={<Database size={40} />}
          title="No datasets"
          description="Create your first dataset to get started."
          action={
            <Button onClick={() => setDialogOpen(true)}>Create Dataset</Button>
          }
        />
      )}

      <DatasetCreateDialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
      />
    </PageShell>
  );
}
