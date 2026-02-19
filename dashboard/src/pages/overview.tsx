import { useMemo } from "react";

import {
  Activity,
  AlertTriangle,
  Box,
  Cpu,
  TrendingUp,
  Wallet,
} from "lucide-react";
import { Link } from "react-router";

import { BarChart } from "@/components/charts/bar-chart";
import { PageShell } from "@/components/layout/page-shell";
import { Badge } from "@/components/ui/badge";
import { Card, StatCard } from "@/components/ui/card";
import { ErrorState } from "@/components/ui/error-state";
import { CardSkeleton, StatCardSkeleton } from "@/components/ui/skeleton";
import { StatusDot } from "@/components/ui/status-dot";
import { useActivity } from "@/hooks/use-activity";
import { useModelHealth } from "@/hooks/use-health";
import { useModels } from "@/hooks/use-models";
import { useSpending, useSpendingHistory } from "@/hooks/use-budgets";
import { useTrainingJobs } from "@/hooks/use-training";
import { SPENDING_SERIES } from "@/lib/constants";
import { formatEUR, formatMs, formatRelative } from "@/lib/format";

const SEVERITY_ICON_CLASSES: Record<string, string> = {
  success: "text-success",
  warning: "text-warning",
  error: "text-danger",
  info: "text-info",
};

export function OverviewPage() {
  const models = useModels();
  const training = useTrainingJobs();
  const health = useModelHealth();
  const spending = useSpending();
  const spendingHistory = useSpendingHistory();
  const activity = useActivity();

  const isLoading =
    models.isLoading || training.isLoading || health.isLoading || spending.isLoading;
  const isError =
    models.isError || training.isError || health.isError || spending.isError;

  const runningJobs = useMemo(
    () => training.data?.filter((j) => j.status === "running") ?? [],
    [training.data],
  );
  const degradedCount = useMemo(
    () => health.data?.filter((h) => h.status !== "healthy").length ?? 0,
    [health.data],
  );
  const budgetAlerts = useMemo(
    () => spending.data?.filter((s) => s.pct_used >= 80) ?? [],
    [spending.data],
  );

  if (isLoading) {
    return (
      <PageShell title="Overview" description="Platform status and recent activity.">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCardSkeleton />
          <StatCardSkeleton />
          <StatCardSkeleton />
          <StatCardSkeleton />
        </div>
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <CardSkeleton className="lg:col-span-2" />
          <CardSkeleton />
        </div>
        <CardSkeleton />
      </PageShell>
    );
  }
  if (isError) {
    return (
      <ErrorState
        message="Failed to load dashboard data."
        onRetry={() => {
          void models.refetch();
          void training.refetch();
          void health.refetch();
          void spending.refetch();
        }}
      />
    );
  }

  const modelCount = models.data?.length ?? 0;
  const totalJobs = training.data?.length ?? 0;
  const totalSpent = spending.data?.find(
    (s) => s.scope === "global",
  )?.spent_eur;
  const totalLimit = spending.data?.find(
    (s) => s.scope === "global",
  )?.limit_eur;

  return (
    <PageShell
      title="Overview"
      description="Platform status and recent activity."
    >
      {/* Stat cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Models"
          value={modelCount}
          subtitle="registered"
          icon={<Box size={18} />}
        />
        <StatCard
          title="Training Jobs"
          value={`${runningJobs.length} running`}
          subtitle={`${totalJobs} total`}
          icon={<Cpu size={18} />}
        />
        <StatCard
          title="Health Alerts"
          value={degradedCount}
          subtitle={degradedCount > 0 ? "need attention" : "all healthy"}
          icon={
            degradedCount > 0 ? (
              <AlertTriangle size={18} className="text-warning" />
            ) : (
              <Activity size={18} />
            )
          }
        />
        <StatCard
          title="Monthly Spend"
          value={totalSpent != null ? formatEUR(totalSpent) : "\u2014"}
          subtitle={
            totalLimit != null ? `of ${formatEUR(totalLimit)} budget` : ""
          }
          icon={<Wallet size={18} />}
        />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Spending chart */}
        <Card title="Daily Spending (EUR)" className="lg:col-span-2">
          {spendingHistory.data ? (
            <BarChart
              data={spendingHistory.data}
              xKey="date"
              series={SPENDING_SERIES}
              stacked
              height={280}
              formatY={(v) => `\u20AC${v}`}
              yLabel="EUR"
            />
          ) : (
            <div className="flex h-[280px] items-center justify-center text-text-muted">
              No spending data
            </div>
          )}
        </Card>

        {/* Model health */}
        <Card title="Model Health">
          {health.data && health.data.length > 0 ? (
            <div className="space-y-3">
              {health.data.map((h) => (
                <div
                  key={h.model_id}
                  className="flex items-center justify-between rounded-md border border-border bg-surface-2 px-3 py-2"
                >
                  <div className="flex items-center gap-2">
                    <StatusDot
                      status={h.status}
                      pulse={h.status !== "healthy"}
                    />
                    <span className="text-sm text-text-primary">
                      {h.model_id.slice(0, 12)}
                    </span>
                  </div>
                  <div className="flex items-center gap-3 text-xs text-text-muted">
                    <span>{formatMs(h.avg_latency_ms)}</span>
                    <Badge value={h.status} />
                  </div>
                </div>
              ))}
              <Link
                to="/health"
                className="mt-2 block text-center text-xs text-accent hover:text-accent-hover"
              >
                View all health details
              </Link>
            </div>
          ) : (
            <p className="text-sm text-text-muted">No health data.</p>
          )}
        </Card>
      </div>

      {/* Budget alerts */}
      {budgetAlerts.length > 0 && (
        <Card title="Budget Alerts">
          <div className="space-y-2">
            {budgetAlerts.map((s) => (
              <div
                key={`${s.scope}-${s.scope_value}`}
                className="flex items-center justify-between rounded-md border border-warning/20 bg-warning/5 px-4 py-2"
              >
                <div className="flex items-center gap-2">
                  <AlertTriangle size={14} className="text-warning" />
                  <span className="text-sm text-text-primary">
                    {s.scope === "global" ? "Global" : s.scope_value.toUpperCase()}{" "}
                    {s.period}
                  </span>
                </div>
                <div className="flex items-center gap-3 text-sm">
                  <span className="text-text-secondary">
                    {formatEUR(s.spent_eur)} / {formatEUR(s.limit_eur)}
                  </span>
                  <span
                    className={
                      s.pct_used >= 90 ? "font-bold text-danger" : "text-warning"
                    }
                  >
                    {s.pct_used.toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Running jobs */}
      {runningJobs.length > 0 && (
        <Card title="Running Jobs">
          <div className="space-y-2">
            {runningJobs.map((j) => (
              <Link
                key={j.job_id}
                to={`/training/${j.job_id}`}
                className="flex items-center justify-between rounded-md border border-border bg-surface-2 px-4 py-2 transition-colors hover:border-border-hover"
              >
                <div className="flex items-center gap-3">
                  <StatusDot status="running" pulse />
                  <div>
                    <span className="text-sm font-medium text-text-primary">
                      {j.model}
                    </span>
                    <span className="ml-2 text-xs text-text-muted">
                      {j.provider}
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-4 text-sm">
                  {j.progress != null && (
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 w-24 overflow-hidden rounded-full bg-surface-3">
                        <div
                          className="h-full rounded-full bg-accent transition-all"
                          style={{ width: `${j.progress}%` }}
                        />
                      </div>
                      <span className="text-xs text-text-secondary">
                        {j.progress}%
                      </span>
                    </div>
                  )}
                  {j.cost_eur != null && (
                    <span className="text-text-muted">
                      {formatEUR(j.cost_eur)}
                    </span>
                  )}
                </div>
              </Link>
            ))}
          </div>
        </Card>
      )}

      {/* Activity feed */}
      <Card title="Recent Activity">
        {activity.data && activity.data.length > 0 ? (
          <div className="space-y-1">
            {activity.data.slice(0, 10).map((evt) => (
              <div
                key={evt.id}
                className="flex items-start gap-3 rounded-md px-2 py-2 transition-colors hover:bg-surface-2"
              >
                <div className="mt-0.5">
                  <TrendingUp
                    size={14}
                    className={
                      SEVERITY_ICON_CLASSES[evt.severity ?? "info"] ??
                      "text-text-muted"
                    }
                  />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-text-primary">
                      {evt.title}
                    </span>
                    <span className="text-xs text-text-muted">
                      {formatRelative(evt.timestamp)}
                    </span>
                  </div>
                  <p className="truncate text-xs text-text-secondary">
                    {evt.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-text-muted">No recent activity.</p>
        )}
      </Card>
    </PageShell>
  );
}
