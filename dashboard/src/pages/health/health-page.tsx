import { useMemo } from "react";

import { Activity, AlertTriangle, CheckCircle, XCircle } from "lucide-react";

import { PageShell } from "@/components/layout/page-shell";
import { Badge } from "@/components/ui/badge";
import { Card, StatCard } from "@/components/ui/card";
import { ErrorState } from "@/components/ui/error-state";
import { PageSpinner } from "@/components/ui/spinner";
import { StatusDot } from "@/components/ui/status-dot";
import { useHealthDetailed, useModelHealth } from "@/hooks/use-health";
import { useModels } from "@/hooks/use-models";
import { formatMs, formatNumber, formatPercent } from "@/lib/format";

export function HealthPage() {
  const system = useHealthDetailed();
  const modelHealth = useModelHealth();
  const models = useModels();

  const healthData = modelHealth.data ?? [];

  const { healthyCount, degradedCount, unhealthyCount } = useMemo(() => {
    let healthy = 0;
    let degraded = 0;
    let unhealthy = 0;
    for (const h of healthData) {
      if (h.status === "healthy") healthy++;
      else if (h.status === "degraded") degraded++;
      else unhealthy++;
    }
    return { healthyCount: healthy, degradedCount: degraded, unhealthyCount: unhealthy };
  }, [healthData]);

  if (system.isLoading || modelHealth.isLoading) return <PageSpinner />;
  if (system.isError || modelHealth.isError) {
    return (
      <ErrorState
        message="Failed to load health data."
        onRetry={() => {
          void system.refetch();
          void modelHealth.refetch();
        }}
      />
    );
  }

  function getModelName(modelId: string): string {
    return models.data?.find((m) => m.model_id === modelId)?.name ?? modelId;
  }

  return (
    <PageShell
      title="Health Monitoring"
      description="System health and model performance metrics."
    >
      {/* System status */}
      <Card title="System Status">
        <div className="flex items-center gap-3">
          <StatusDot
            status={system.data?.status ?? "unknown"}
            pulse={system.data?.status === "healthy"}
          />
          <span className="text-lg font-medium text-text-primary">
            {system.data?.status ?? "Unknown"}
          </span>
          {system.data?.version && (
            <span className="text-sm text-text-muted">
              v{system.data.version}
            </span>
          )}
        </div>
        {system.data?.components && (
          <div className="mt-3 flex gap-4">
            {Object.entries(system.data.components).map(([key, value]) => (
              <div
                key={key}
                className="rounded-md border border-border bg-surface-2 px-3 py-1.5 text-xs"
              >
                <span className="text-text-muted">{key}: </span>
                <span className="text-success">{String(value)}</span>
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* Summary stats */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <StatCard
          title="Healthy"
          value={healthyCount}
          icon={<CheckCircle size={18} className="text-success" />}
        />
        <StatCard
          title="Degraded"
          value={degradedCount}
          icon={<AlertTriangle size={18} className="text-warning" />}
        />
        <StatCard
          title="Unhealthy"
          value={unhealthyCount}
          icon={<XCircle size={18} className="text-danger" />}
        />
      </div>

      {/* Model health cards */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {healthData.map((h) => (
          <div
            key={h.model_id}
            className={`rounded-lg border p-5 ${
              h.status === "healthy"
                ? "border-border bg-surface-1"
                : h.status === "degraded"
                  ? "border-warning/30 bg-warning/5"
                  : "border-danger/30 bg-danger/5"
            }`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <StatusDot
                  status={h.status}
                  pulse={h.status !== "healthy"}
                />
                <span className="font-medium text-text-primary">
                  {getModelName(h.model_id)}
                </span>
              </div>
              <Badge value={h.status} />
            </div>

            <div className="mt-4 grid grid-cols-3 gap-4">
              <MetricItem
                label="Error Rate"
                value={formatPercent(h.error_rate)}
                alert={h.error_rate > 0.05}
              />
              <MetricItem
                label="Avg Latency"
                value={formatMs(h.avg_latency_ms)}
                alert={h.avg_latency_ms > 200}
              />
              <MetricItem
                label="Drift Score"
                value={h.drift_score.toFixed(2)}
                alert={h.drift_score > 0.1}
              />
            </div>

            <div className="mt-3 grid grid-cols-3 gap-4 border-t border-border pt-3">
              <MetricItem
                label="P95 Latency"
                value={formatMs(h.p95_latency_ms)}
              />
              <MetricItem
                label="P99 Latency"
                value={formatMs(h.p99_latency_ms)}
              />
              <MetricItem
                label="Total Predictions"
                value={formatNumber(h.total_predictions, 0)}
              />
            </div>
          </div>
        ))}
      </div>

      {healthData.length === 0 && (
        <div className="flex h-32 items-center justify-center text-sm text-text-muted">
          <Activity size={20} className="mr-2" />
          No model health data available.
        </div>
      )}
    </PageShell>
  );
}

function MetricItem({
  label,
  value,
  alert = false,
}: {
  label: string;
  value: string;
  alert?: boolean;
}) {
  return (
    <div>
      <span className="text-xs text-text-muted">{label}</span>
      <p
        className={`text-sm font-medium ${alert ? "text-danger" : "text-text-primary"}`}
      >
        {value}
      </p>
    </div>
  );
}
