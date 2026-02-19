import { useState } from "react";
import { useNavigate } from "react-router";

import { GitBranch, Plus } from "lucide-react";

import { PageShell } from "@/components/layout/page-shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { type Column, DataTable } from "@/components/ui/data-table";
import { EmptyState } from "@/components/ui/empty-state";
import { ErrorState } from "@/components/ui/error-state";
import { TableSkeleton } from "@/components/ui/skeleton";
import { useABTests } from "@/hooks/use-ab-tests";
import { formatDate } from "@/lib/format";
import type { ABTest } from "@/types/api";

import { ABTestCreateDialog } from "./ab-test-create-dialog";

const COLUMNS: Column<ABTest>[] = [
  {
    key: "name",
    header: "Name",
    sortable: true,
    sortValue: (r) => r.name,
    render: (r) => (
      <span className="font-medium text-text-primary">{r.name}</span>
    ),
  },
  {
    key: "service",
    header: "Service",
    render: (r) => <span className="text-text-secondary">{r.service}</span>,
  },
  {
    key: "status",
    header: "Status",
    sortable: true,
    sortValue: (r) => r.status,
    render: (r) => <Badge value={r.status} />,
  },
  {
    key: "metric",
    header: "Primary Metric",
    render: (r) => (
      <span className="text-text-secondary">{r.primary_metric}</span>
    ),
  },
  {
    key: "winner",
    header: "Winner",
    render: (r) => (
      <span className={r.winner ? "text-success" : "text-text-muted"}>
        {r.winner ?? "—"}
      </span>
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

export function ABTestsListPage() {
  const { data, isLoading, isError, refetch } = useABTests();
  const navigate = useNavigate();
  const [dialogOpen, setDialogOpen] = useState(false);

  if (isLoading) {
    return (
      <PageShell title="A/B Tests" description="Manage model comparison experiments.">
        <TableSkeleton rows={6} cols={6} />
      </PageShell>
    );
  }
  if (isError) return (
    <PageShell title="A/B Tests">
      <ErrorState message="Failed to load A/B tests." onRetry={() => void refetch()} />
    </PageShell>
  );

  return (
    <PageShell
      title="A/B Tests"
      description="Manage model comparison experiments."
      actions={
        <Button onClick={() => setDialogOpen(true)}>
          <Plus size={16} className="mr-1" />
          Create Test
        </Button>
      }
    >
      {data && data.length > 0 ? (
        <DataTable
          columns={COLUMNS}
          data={data}
          keyFn={(r) => r.test_id}
          onRowClick={(r) => navigate(`/ab-tests/${r.test_id}`)}
        />
      ) : (
        <EmptyState
          icon={<GitBranch size={40} />}
          title="No A/B tests"
          description="Create your first experiment."
          action={
            <Button onClick={() => setDialogOpen(true)}>Create Test</Button>
          }
        />
      )}

      <ABTestCreateDialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
      />
    </PageShell>
  );
}
