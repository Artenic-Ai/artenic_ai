import { useParams, Link } from "react-router";

import { ArrowLeft } from "lucide-react";

import { PageShell } from "@/components/layout/page-shell";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { DetailRow } from "@/components/ui/detail-row";
import { ErrorState } from "@/components/ui/error-state";
import { PageSpinner } from "@/components/ui/spinner";
import { useModel } from "@/hooks/use-models";
import { formatDateTime } from "@/lib/format";

export function ModelDetailPage() {
  const { modelId } = useParams();
  const { data: model, isLoading, isError, refetch } = useModel(modelId ?? "");

  if (isLoading) return <PageSpinner />;
  if (isError) {
    return <ErrorState message="Failed to load model." onRetry={() => void refetch()} />;
  }
  if (!model) {
    return (
      <div className="flex h-64 items-center justify-center">
        <p className="text-text-muted">Model not found.</p>
      </div>
    );
  }

  return (
    <PageShell
      title={model.name}
      description={`v${model.version} \u2014 ${model.model_type}`}
      actions={
        <Link
          to="/models"
          className="inline-flex items-center gap-1 text-sm text-text-secondary hover:text-text-primary"
        >
          <ArrowLeft size={16} />
          Back to models
        </Link>
      }
    >
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card title="Details">
          <dl className="space-y-3">
            <DetailRow label="Model ID" value={model.model_id} />
            <DetailRow label="Name" value={model.name} />
            <DetailRow label="Version" value={model.version} />
            <DetailRow label="Type" value={model.model_type} />
            <DetailRow label="Framework" value={model.framework} />
            <DetailRow
              label="Stage"
              value={<Badge value={model.stage} />}
            />
            <DetailRow
              label="Created"
              value={formatDateTime(model.created_at)}
            />
          </dl>
        </Card>

        <Card title="Description">
          <p className="text-sm leading-relaxed text-text-secondary">
            {model.description || "No description provided."}
          </p>
        </Card>

        {Object.keys(model.tags).length > 0 && (
          <Card title="Tags" className="lg:col-span-2">
            <div className="flex flex-wrap gap-2">
              {Object.entries(model.tags).map(([k, v]) => (
                <span
                  key={k}
                  className="rounded-md border border-border bg-surface-2 px-2 py-1 text-xs text-text-secondary"
                >
                  {k}: {v}
                </span>
              ))}
            </div>
          </Card>
        )}
      </div>
    </PageShell>
  );
}
