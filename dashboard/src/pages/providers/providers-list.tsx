import { useNavigate } from "react-router";
import { Cloud } from "lucide-react";

import { PageShell } from "@/components/layout/page-shell";
import { ErrorState } from "@/components/ui/error-state";
import { TableSkeleton } from "@/components/ui/skeleton";
import { useProviders } from "@/hooks/use-providers";
import type { ProviderStatus, ProviderSummary } from "@/types/api";

function statusColor(status: ProviderStatus): string {
  switch (status) {
    case "connected":
      return "bg-success";
    case "configured":
      return "bg-warning";
    case "error":
      return "bg-danger";
    default:
      return "bg-text-muted";
  }
}

function statusLabel(status: ProviderStatus): string {
  switch (status) {
    case "connected":
      return "Connected";
    case "configured":
      return "Configured";
    case "error":
      return "Error";
    default:
      return "Unconfigured";
  }
}

function ProviderCard({ provider }: { provider: ProviderSummary }) {
  const navigate = useNavigate();

  return (
    <button
      type="button"
      aria-label={`View provider ${provider.display_name}`}
      onClick={() => navigate(`/providers/${provider.id}`)}
      className="flex flex-col rounded-lg border border-border bg-surface-1 p-5 text-left transition-colors hover:border-accent/40 hover:bg-surface-2"
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent/10 text-accent">
            <Cloud size={20} />
          </div>
          <div>
            <h3 className="font-medium text-text-primary">
              {provider.display_name}
            </h3>
            <p className="mt-0.5 text-xs text-text-muted">
              {provider.description}
            </p>
          </div>
        </div>
      </div>

      <div className="mt-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span
            className={`inline-block h-2 w-2 rounded-full ${statusColor(provider.status)}`}
          />
          <span className="text-xs text-text-secondary">
            {statusLabel(provider.status)}
          </span>
        </div>
        <div className="flex gap-1.5">
          {provider.capabilities.map((cap) => (
            <span
              key={cap.type}
              className="rounded-full bg-surface-3 px-2 py-0.5 text-[10px] font-medium text-text-secondary"
            >
              {cap.type}
            </span>
          ))}
        </div>
      </div>
    </button>
  );
}

export function ProvidersListPage() {
  const { data, isLoading, isError, refetch } = useProviders();

  if (isLoading) {
    return (
      <PageShell
        title="Providers"
        description="Manage your cloud providers and their capabilities."
      >
        <TableSkeleton rows={4} cols={3} />
      </PageShell>
    );
  }

  if (isError) {
    return (
      <PageShell title="Providers">
        <ErrorState
          message="Failed to load providers."
          onRetry={() => void refetch()}
        />
      </PageShell>
    );
  }

  return (
    <PageShell
      title="Providers"
      description="Manage your cloud providers and their capabilities."
    >
      {(data ?? []).length === 0 ? (
        <div className="flex h-32 items-center justify-center text-sm text-text-muted">
          No providers found.
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {data!.map((provider) => (
            <ProviderCard key={provider.id} provider={provider} />
          ))}
        </div>
      )}
    </PageShell>
  );
}
