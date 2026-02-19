import { useParams } from "react-router";

import { BarChart } from "@/components/charts/bar-chart";
import { PageShell } from "@/components/layout/page-shell";
import { Badge } from "@/components/ui/badge";
import { Breadcrumb } from "@/components/ui/breadcrumb";
import { Card } from "@/components/ui/card";
import { DetailRow } from "@/components/ui/detail-row";
import { ErrorState } from "@/components/ui/error-state";
import { DetailSkeleton } from "@/components/ui/skeleton";
import { useABTest, useABTestResults } from "@/hooks/use-ab-tests";
import { formatDateTime, formatMs } from "@/lib/format";

export function ABTestDetailPage() {
  const { testId } = useParams();
  const { data: test, isLoading, isError, refetch } = useABTest(testId ?? "");
  const { data: results } = useABTestResults(testId ?? "");

  if (isLoading) {
    return (
      <PageShell title="">
        <DetailSkeleton />
      </PageShell>
    );
  }
  if (isError) {
    return <ErrorState message="Failed to load A/B test." onRetry={() => void refetch()} />;
  }
  if (!test) {
    return (
      <div className="flex h-64 items-center justify-center">
        <p className="text-text-muted">A/B test not found.</p>
      </div>
    );
  }

  const chartData = results
    ? Object.entries(results.variants).map(([name, v]) => ({
        name,
        mean: Number((v.mean * 100).toFixed(1)),
        error_rate: Number((v.error_rate * 100).toFixed(2)),
        latency: v.avg_latency_ms,
      }))
    : [];

  return (
    <PageShell
      title={test.name}
      description={`${test.service} \u2014 ${test.primary_metric}`}
      breadcrumb={
        <Breadcrumb
          items={[
            { label: "A/B Tests", to: "/ab-tests" },
            { label: test.name },
          ]}
        />
      }
    >
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card title="Test Configuration">
          <dl className="space-y-3">
            <DetailRow label="Test ID" value={test.test_id} />
            <DetailRow label="Service" value={test.service} />
            <DetailRow
              label="Status"
              value={<Badge value={test.status} />}
            />
            <DetailRow label="Primary Metric" value={test.primary_metric} />
            <DetailRow
              label="Min Samples"
              value={test.min_samples.toLocaleString()}
            />
            {test.winner && (
              <DetailRow
                label="Winner"
                value={
                  <span className="font-medium text-success">
                    {test.winner}
                  </span>
                }
              />
            )}
            <DetailRow
              label="Created"
              value={formatDateTime(test.created_at)}
            />
            {test.concluded_at && (
              <DetailRow
                label="Concluded"
                value={formatDateTime(test.concluded_at)}
              />
            )}
          </dl>
        </Card>

        <Card title="Variants">
          <pre className="overflow-auto rounded-md border border-border bg-surface-2 p-4 font-mono text-sm text-text-primary">
            {JSON.stringify(test.variants, null, 2)}
          </pre>
        </Card>

        {results && (
          <>
            <Card title="Results Summary" className="lg:col-span-2">
              <div className="mb-4 text-sm text-text-secondary">
                Total samples: {results.total_samples.toLocaleString()}
              </div>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                {Object.entries(results.variants).map(([name, v]) => (
                  <div
                    key={name}
                    className={`rounded-lg border px-4 py-3 ${
                      test.winner === name
                        ? "border-success/30 bg-success/5"
                        : "border-border bg-surface-2"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-text-primary">
                        {name}
                      </span>
                      {test.winner === name && (
                        <Badge value="winner" className="bg-success/10 text-success" />
                      )}
                    </div>
                    <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
                      <div>
                        <span className="text-text-muted">Samples</span>
                        <p className="text-text-primary">{v.samples.toLocaleString()}</p>
                      </div>
                      <div>
                        <span className="text-text-muted">Mean</span>
                        <p className="text-text-primary">
                          {(v.mean * 100).toFixed(1)}%
                        </p>
                      </div>
                      <div>
                        <span className="text-text-muted">Error Rate</span>
                        <p className="text-text-primary">
                          {(v.error_rate * 100).toFixed(2)}%
                        </p>
                      </div>
                      <div>
                        <span className="text-text-muted">Avg Latency</span>
                        <p className="text-text-primary">
                          {formatMs(v.avg_latency_ms)}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </Card>

            {chartData.length > 0 && (
              <Card title="Metric Comparison" className="lg:col-span-2">
                <BarChart
                  data={chartData}
                  xKey="name"
                  series={[
                    {
                      key: "mean",
                      label: "Mean (%)",
                      color: "var(--color-chart-1)",
                    },
                    {
                      key: "latency",
                      label: "Latency (ms)",
                      color: "var(--color-chart-3)",
                    },
                  ]}
                  height={250}
                  yLabel="Value"
                />
              </Card>
            )}
          </>
        )}
      </div>
    </PageShell>
  );
}
