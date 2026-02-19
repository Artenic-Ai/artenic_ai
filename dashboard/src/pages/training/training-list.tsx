import { useState } from "react";
import { useNavigate } from "react-router";

import { Cpu, Plus } from "lucide-react";

import { PageShell } from "@/components/layout/page-shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { type Column, DataTable } from "@/components/ui/data-table";
import { EmptyState } from "@/components/ui/empty-state";
import { ErrorState } from "@/components/ui/error-state";
import { TableSkeleton } from "@/components/ui/skeleton";
import { useTrainingJobs } from "@/hooks/use-training";
import { formatDateTime, formatEUR } from "@/lib/format";
import type { TrainingJob } from "@/types/api";

import { TrainingDispatchDialog } from "./training-dispatch-dialog";

const COLUMNS: Column<TrainingJob>[] = [
  {
    key: "model",
    header: "Model",
    sortable: true,
    sortValue: (r) => r.model,
    render: (r) => (
      <div>
        <span className="font-medium text-text-primary">{r.model}</span>
        <span className="ml-2 text-xs text-text-muted">{r.service}</span>
      </div>
    ),
  },
  {
    key: "provider",
    header: "Provider",
    sortable: true,
    sortValue: (r) => r.provider,
    render: (r) => (
      <div className="text-text-secondary">
        <span>{r.provider}</span>
        {r.is_spot && (
          <span className="ml-1 text-xs text-warning">(spot)</span>
        )}
      </div>
    ),
  },
  {
    key: "status",
    header: "Status",
    sortable: true,
    sortValue: (r) => r.status,
    render: (r) => (
      <div className="flex items-center gap-2">
        <Badge value={r.status} />
        {r.progress != null && r.status === "running" && (
          <span className="text-xs text-text-muted">{r.progress}%</span>
        )}
      </div>
    ),
  },
  {
    key: "cost",
    header: "Cost",
    sortable: true,
    sortValue: (r) => r.cost_eur ?? 0,
    render: (r) => (
      <span className="text-text-secondary">
        {r.cost_eur != null ? formatEUR(r.cost_eur) : "—"}
      </span>
    ),
  },
  {
    key: "created_at",
    header: "Created",
    sortable: true,
    sortValue: (r) => r.created_at,
    render: (r) => (
      <span className="text-text-muted">{formatDateTime(r.created_at)}</span>
    ),
  },
];

export function TrainingListPage() {
  const { data, isLoading, isError, refetch } = useTrainingJobs();
  const navigate = useNavigate();
  const [dialogOpen, setDialogOpen] = useState(false);

  if (isLoading) {
    return (
      <PageShell title="Training Jobs" description="Manage and monitor training runs.">
        <TableSkeleton rows={8} cols={5} />
      </PageShell>
    );
  }
  if (isError) return <ErrorState message="Failed to load training jobs." onRetry={() => void refetch()} />;

  return (
    <PageShell
      title="Training Jobs"
      description="Manage and monitor training runs."
      actions={
        <Button onClick={() => setDialogOpen(true)}>
          <Plus size={16} className="mr-1" />
          Dispatch Job
        </Button>
      }
    >
      {data && data.length > 0 ? (
        <DataTable
          columns={COLUMNS}
          data={data}
          keyFn={(r) => r.job_id}
          onRowClick={(r) => navigate(`/training/${r.job_id}`)}
          pageSize={12}
        />
      ) : (
        <EmptyState
          icon={<Cpu size={40} />}
          title="No training jobs"
          description="Dispatch your first training job."
          action={
            <Button onClick={() => setDialogOpen(true)}>Dispatch Job</Button>
          }
        />
      )}

      <TrainingDispatchDialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
      />
    </PageShell>
  );
}
