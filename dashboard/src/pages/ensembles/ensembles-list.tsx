import { useState } from "react";
import { useNavigate } from "react-router";

import { Layers, Plus } from "lucide-react";

import { PageShell } from "@/components/layout/page-shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { type Column, DataTable } from "@/components/ui/data-table";
import { EmptyState } from "@/components/ui/empty-state";
import { ErrorState } from "@/components/ui/error-state";
import { PageSpinner } from "@/components/ui/spinner";
import { useEnsembles } from "@/hooks/use-ensembles";
import { formatDate } from "@/lib/format";
import type { Ensemble } from "@/types/api";

import { EnsembleCreateDialog } from "./ensemble-create-dialog";

const COLUMNS: Column<Ensemble>[] = [
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
    key: "service",
    header: "Service",
    render: (r) => <span className="text-text-secondary">{r.service}</span>,
  },
  {
    key: "strategy",
    header: "Strategy",
    render: (r) => (
      <span className="text-text-secondary">{r.strategy.replace(/_/g, " ")}</span>
    ),
  },
  {
    key: "models",
    header: "Models",
    render: (r) => (
      <span className="text-text-muted">{r.model_ids.length} model(s)</span>
    ),
  },
  {
    key: "stage",
    header: "Stage",
    sortable: true,
    sortValue: (r) => r.stage,
    render: (r) => <Badge value={r.stage} />,
  },
  {
    key: "updated_at",
    header: "Updated",
    sortable: true,
    sortValue: (r) => r.updated_at,
    render: (r) => (
      <span className="text-text-muted">{formatDate(r.updated_at)}</span>
    ),
  },
];

export function EnsemblesListPage() {
  const { data, isLoading, isError, refetch } = useEnsembles();
  const navigate = useNavigate();
  const [dialogOpen, setDialogOpen] = useState(false);

  if (isLoading) return <PageSpinner />;
  if (isError) return <ErrorState message="Failed to load ensembles." onRetry={() => void refetch()} />;

  return (
    <PageShell
      title="Ensembles"
      description="Model ensembles and their configurations."
      actions={
        <Button onClick={() => setDialogOpen(true)}>
          <Plus size={16} className="mr-1" />
          Create Ensemble
        </Button>
      }
    >
      {data && data.length > 0 ? (
        <DataTable
          columns={COLUMNS}
          data={data}
          keyFn={(r) => r.ensemble_id}
          onRowClick={(r) => navigate(`/ensembles/${r.ensemble_id}`)}
        />
      ) : (
        <EmptyState
          icon={<Layers size={40} />}
          title="No ensembles"
          description="Create your first model ensemble."
          action={
            <Button onClick={() => setDialogOpen(true)}>
              Create Ensemble
            </Button>
          }
        />
      )}

      <EnsembleCreateDialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
      />
    </PageShell>
  );
}
