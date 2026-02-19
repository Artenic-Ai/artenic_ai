import { useParams } from "react-router";

import { LineChart } from "@/components/charts/line-chart";
import { PageShell } from "@/components/layout/page-shell";
import { Badge } from "@/components/ui/badge";
import { Breadcrumb } from "@/components/ui/breadcrumb";
import { Card } from "@/components/ui/card";
import { DetailRow } from "@/components/ui/detail-row";
import { ErrorState } from "@/components/ui/error-state";
import { DetailSkeleton } from "@/components/ui/skeleton";
import { useTrainingJob } from "@/hooks/use-training";
import { formatDateTime, formatEUR } from "@/lib/format";

export function TrainingDetailPage() {
  const { jobId } = useParams();
  const { data: job, isLoading, isError, refetch } = useTrainingJob(jobId ?? "");

  if (isLoading) {
    return (
      <PageShell title="">
        <DetailSkeleton />
      </PageShell>
    );
  }
  if (isError) {
    return <ErrorState message="Failed to load training job." onRetry={() => void refetch()} />;
  }
  if (!job) {
    return (
      <div className="flex h-64 items-center justify-center">
        <p className="text-text-muted">Training job not found.</p>
      </div>
    );
  }

  return (
    <PageShell
      title={job.model}
      description={`Job ${job.job_id} \u2014 ${job.service}`}
      breadcrumb={
        <Breadcrumb
          items={[
            { label: "Training", to: "/training" },
            { label: job.model },
          ]}
        />
      }
    >
      {/* Progress bar for running jobs */}
      {job.status === "running" && job.progress != null && (
        <div className="space-y-1">
          <div className="flex items-center justify-between text-sm">
            <span className="text-text-secondary">Progress</span>
            <span className="font-medium text-text-primary">
              {job.progress}%
            </span>
          </div>
          <div
            className="h-2 overflow-hidden rounded-full bg-surface-3"
            role="progressbar"
            aria-valuenow={job.progress}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label="Training progress"
          >
            <div
              className="h-full rounded-full bg-accent transition-all"
              style={{ width: `${job.progress}%` }}
            />
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card title="Job Details">
          <dl className="space-y-3">
            <DetailRow label="Job ID" value={job.job_id} />
            <DetailRow label="Service" value={job.service} />
            <DetailRow label="Model" value={job.model} />
            <DetailRow label="Provider" value={job.provider} />
            <DetailRow
              label="Status"
              value={<Badge value={job.status} />}
            />
            {job.instance_type && (
              <DetailRow label="Instance" value={job.instance_type} />
            )}
            {job.region && (
              <DetailRow label="Region" value={job.region} />
            )}
            <DetailRow
              label="Spot"
              value={job.is_spot ? "Yes" : "No"}
            />
          </dl>
        </Card>

        <Card title="Cost & Timing">
          <dl className="space-y-3">
            {job.cost_eur != null && (
              <DetailRow label="Total Cost" value={formatEUR(job.cost_eur)} />
            )}
            {job.cost_per_hour != null && (
              <DetailRow
                label="Cost/Hour"
                value={formatEUR(job.cost_per_hour)}
              />
            )}
            <DetailRow
              label="Created"
              value={formatDateTime(job.created_at)}
            />
            {job.started_at && (
              <DetailRow
                label="Started"
                value={formatDateTime(job.started_at)}
              />
            )}
            {job.completed_at && (
              <DetailRow
                label="Completed"
                value={formatDateTime(job.completed_at)}
              />
            )}
          </dl>
        </Card>

        {job.error_message && (
          <Card title="Error" className="lg:col-span-2">
            <div className="rounded-md border border-danger/20 bg-danger/5 px-4 py-3">
              <p className="text-sm text-danger">{job.error_message}</p>
            </div>
          </Card>
        )}

        {job.loss_history && job.loss_history.length > 0 && (
          <Card title="Loss Curve" className="lg:col-span-2">
            <LineChart
              data={job.loss_history}
              xKey="epoch"
              series={[
                {
                  key: "loss",
                  label: "Loss",
                  color: "var(--color-chart-1)",
                },
              ]}
              height={300}
              xLabel="Epoch"
              yLabel="Loss"
            />
          </Card>
        )}
      </div>
    </PageShell>
  );
}
