import { useState } from "react";
import { useNavigate } from "react-router";

import { Box, Plus } from "lucide-react";

import { PageShell } from "@/components/layout/page-shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { type Column, DataTable } from "@/components/ui/data-table";
import { EmptyState } from "@/components/ui/empty-state";
import { ErrorState } from "@/components/ui/error-state";
import { TableSkeleton } from "@/components/ui/skeleton";
import { useModels } from "@/hooks/use-models";
import { formatDate } from "@/lib/format";
import type { Model } from "@/types/api";

import { ModelRegisterDialog } from "./model-register-dialog";

const COLUMNS: Column<Model>[] = [
  {
    key: "name",
    header: "Name",
    sortable: true,
    sortValue: (r) => r.name,
    render: (r) => (
      <div>
        <span className="font-medium text-text-primary">{r.name}</span>
        <span className="ml-2 text-xs text-text-muted">v{r.version}</span>
      </div>
    ),
  },
  {
    key: "type",
    header: "Type",
    sortable: true,
    sortValue: (r) => r.model_type,
    render: (r) => <span className="text-text-secondary">{r.model_type}</span>,
  },
  {
    key: "framework",
    header: "Framework",
    render: (r) => <span className="text-text-secondary">{r.framework}</span>,
  },
  {
    key: "stage",
    header: "Stage",
    sortable: true,
    sortValue: (r) => r.stage,
    render: (r) => <Badge value={r.stage} />,
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

export function ModelsListPage() {
  const { data, isLoading, isError, refetch } = useModels();
  const navigate = useNavigate();
  const [dialogOpen, setDialogOpen] = useState(false);

  if (isLoading) {
    return (
      <PageShell title="Model Registry" description="All registered models and their stages.">
        <TableSkeleton rows={8} cols={5} />
      </PageShell>
    );
  }
  if (isError) return (
    <PageShell title="Models">
      <ErrorState message="Failed to load models." onRetry={() => void refetch()} />
    </PageShell>
  );

  return (
    <PageShell
      title="Model Registry"
      description="All registered models and their stages."
      actions={
        <Button onClick={() => setDialogOpen(true)}>
          <Plus size={16} className="mr-1" />
          Register Model
        </Button>
      }
    >
      {data && data.length > 0 ? (
        <DataTable
          columns={COLUMNS}
          data={data}
          keyFn={(r) => r.model_id}
          onRowClick={(r) => navigate(`/models/${r.model_id}`)}
        />
      ) : (
        <EmptyState
          icon={<Box size={40} />}
          title="No models registered"
          description="Register your first model to get started."
          action={
            <Button onClick={() => setDialogOpen(true)}>
              Register Model
            </Button>
          }
        />
      )}

      <ModelRegisterDialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
      />
    </PageShell>
  );
}
