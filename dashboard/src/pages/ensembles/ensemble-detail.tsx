import { useParams, Link } from "react-router";

import { ArrowLeft } from "lucide-react";

import { PageShell } from "@/components/layout/page-shell";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { DetailRow } from "@/components/ui/detail-row";
import { ErrorState } from "@/components/ui/error-state";
import { PageSpinner } from "@/components/ui/spinner";
import { useEnsemble, useEnsembleVersions } from "@/hooks/use-ensembles";
import { formatDateTime } from "@/lib/format";

export function EnsembleDetailPage() {
  const { ensembleId } = useParams();
  const { data: ensemble, isLoading, isError, refetch } = useEnsemble(ensembleId ?? "");
  const { data: versions } = useEnsembleVersions(ensembleId ?? "");

  if (isLoading) return <PageSpinner />;
  if (isError) {
    return <ErrorState message="Failed to load ensemble." onRetry={() => void refetch()} />;
  }
  if (!ensemble) {
    return (
      <div className="flex h-64 items-center justify-center">
        <p className="text-text-muted">Ensemble not found.</p>
      </div>
    );
  }

  return (
    <PageShell
      title={ensemble.name}
      description={`${ensemble.service} \u2014 ${ensemble.strategy.replace(/_/g, " ")}`}
      actions={
        <Link
          to="/ensembles"
          className="inline-flex items-center gap-1 text-sm text-text-secondary hover:text-text-primary"
        >
          <ArrowLeft size={16} />
          Back to ensembles
        </Link>
      }
    >
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card title="Configuration">
          <dl className="space-y-3">
            <DetailRow label="Ensemble ID" value={ensemble.ensemble_id} />
            <DetailRow label="Service" value={ensemble.service} />
            <DetailRow
              label="Strategy"
              value={ensemble.strategy.replace(/_/g, " ")}
            />
            <DetailRow
              label="Stage"
              value={<Badge value={ensemble.stage} />}
            />
            <DetailRow label="Version" value={`v${ensemble.version}`} />
            <DetailRow
              label="Updated"
              value={formatDateTime(ensemble.updated_at)}
            />
          </dl>
        </Card>

        <Card title="Description">
          <p className="text-sm leading-relaxed text-text-secondary">
            {ensemble.description || "No description."}
          </p>
          <div className="mt-4">
            <h4 className="text-xs font-medium text-text-muted">Model IDs</h4>
            <div className="mt-2 space-y-1">
              {ensemble.model_ids.map((id) => (
                <div
                  key={id}
                  className="rounded-md border border-border bg-surface-2 px-3 py-1.5 text-xs font-mono text-text-secondary"
                >
                  {id}
                </div>
              ))}
            </div>
          </div>
        </Card>

        {ensemble.strategy_config &&
          Object.keys(ensemble.strategy_config).length > 0 && (
            <Card title="Strategy Config" className="lg:col-span-2">
              <pre className="overflow-auto rounded-md border border-border bg-surface-2 p-4 font-mono text-sm text-text-primary">
                {JSON.stringify(ensemble.strategy_config, null, 2)}
              </pre>
            </Card>
          )}

        {versions && versions.length > 0 && (
          <Card title="Version History" className="lg:col-span-2">
            <div className="space-y-3">
              {versions.map((v) => (
                <div
                  key={v.version}
                  className="flex items-start justify-between rounded-md border border-border bg-surface-2 px-4 py-3"
                >
                  <div>
                    <span className="text-sm font-medium text-text-primary">
                      v{v.version}
                    </span>
                    <span className="ml-2 text-xs text-text-muted">
                      {v.strategy.replace(/_/g, " ")}
                    </span>
                    <p className="mt-1 text-xs text-text-secondary">
                      {v.change_reason}
                    </p>
                  </div>
                  <span className="text-xs text-text-muted">
                    {formatDateTime(v.created_at)}
                  </span>
                </div>
              ))}
            </div>
          </Card>
        )}
      </div>
    </PageShell>
  );
}
