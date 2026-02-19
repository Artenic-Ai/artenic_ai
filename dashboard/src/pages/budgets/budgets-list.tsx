import { useState } from "react";

import { AlertTriangle, Plus, Wallet } from "lucide-react";

import { BarChart } from "@/components/charts/bar-chart";
import { PageShell } from "@/components/layout/page-shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { EmptyState } from "@/components/ui/empty-state";
import { ErrorState } from "@/components/ui/error-state";
import { CardSkeleton, StatCardSkeleton } from "@/components/ui/skeleton";
import { useBudgets, useSpending, useSpendingHistory } from "@/hooks/use-budgets";
import { SPENDING_SERIES } from "@/lib/constants";
import { formatEUR } from "@/lib/format";

import { BudgetCreateDialog } from "./budget-create-dialog";

export function BudgetsListPage() {
  const budgets = useBudgets();
  const spending = useSpending();
  const history = useSpendingHistory();
  const [dialogOpen, setDialogOpen] = useState(false);

  if (budgets.isLoading || spending.isLoading) {
    return (
      <PageShell title="Budgets" description="Budget rules and spending overview.">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCardSkeleton />
          <StatCardSkeleton />
          <StatCardSkeleton />
          <StatCardSkeleton />
        </div>
        <CardSkeleton />
        <CardSkeleton />
      </PageShell>
    );
  }
  if (budgets.isError || spending.isError) {
    return (
      <PageShell title="Budgets">
        <ErrorState
          message="Failed to load budget data."
          onRetry={() => {
            void budgets.refetch();
            void spending.refetch();
          }}
        />
      </PageShell>
    );
  }

  return (
    <PageShell
      title="Budgets"
      description="Budget rules and spending overview."
      actions={
        <Button onClick={() => setDialogOpen(true)}>
          <Plus size={16} className="mr-1" />
          Create Rule
        </Button>
      }
    >
      {/* Spending overview cards */}
      {spending.data && spending.data.length > 0 && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {spending.data.map((s) => {
            const isAlert = s.pct_used >= 80;
            return (
              <div
                key={`${s.scope}-${s.scope_value}`}
                className={`rounded-lg border p-4 ${
                  isAlert
                    ? "border-warning/30 bg-warning/5"
                    : "border-border bg-surface-1"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-text-secondary">
                    {s.scope === "global"
                      ? "Global"
                      : s.scope_value.toUpperCase()}{" "}
                    {s.period}
                  </span>
                  {isAlert && (
                    <AlertTriangle size={14} className="text-warning" />
                  )}
                </div>
                <div className="mt-2 text-lg font-bold text-text-primary">
                  {formatEUR(s.spent_eur)}
                </div>
                <div className="mt-1 text-xs text-text-muted">
                  of {formatEUR(s.limit_eur)}
                </div>
                <div
                  className="mt-2 h-1.5 overflow-hidden rounded-full bg-surface-3"
                  role="progressbar"
                  aria-valuenow={Math.min(s.pct_used, 100)}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-label={`${s.scope === "global" ? "Global" : s.scope_value} budget usage`}
                >
                  <div
                    className={`h-full rounded-full transition-all ${
                      s.pct_used >= 90
                        ? "bg-danger"
                        : s.pct_used >= 75
                          ? "bg-warning"
                          : "bg-accent"
                    }`}
                    style={{ width: `${Math.min(s.pct_used, 100)}%` }}
                  />
                </div>
                <div className="mt-1 text-right text-xs text-text-muted">
                  {s.pct_used.toFixed(1)}%
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Spending history chart */}
      {history.data && (
        <Card title="Daily Spending by Provider (EUR)">
          <BarChart
            data={history.data}
            xKey="date"
            series={SPENDING_SERIES}
            stacked
            height={300}
            formatY={(v) => `€${v}`}
            yLabel="EUR"
          />
        </Card>
      )}

      {/* Budget rules */}
      <Card title="Budget Rules">
        {budgets.data && budgets.data.length > 0 ? (
          <div className="space-y-3">
            {budgets.data.map((b) => (
              <div
                key={b.id}
                className="flex items-center justify-between rounded-md border border-border bg-surface-2 px-4 py-3"
              >
                <div>
                  <span className="text-sm font-medium text-text-primary">
                    {b.scope === "global"
                      ? "Global"
                      : `${b.scope}: ${b.scope_value}`}
                  </span>
                  <span className="ml-2 text-xs text-text-muted">
                    {b.period}
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-sm text-text-secondary">
                    {formatEUR(b.limit_eur)}
                  </span>
                  {b.alert_threshold_pct && (
                    <span className="text-xs text-text-muted">
                      alert @ {b.alert_threshold_pct}%
                    </span>
                  )}
                  <Badge value={b.enabled ? "active" : "disabled"} />
                </div>
              </div>
            ))}
          </div>
        ) : (
          <EmptyState
            icon={<Wallet size={40} />}
            title="No budget rules"
            description="Create budget rules to monitor spending."
          />
        )}
      </Card>

      <BudgetCreateDialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
      />
    </PageShell>
  );
}
