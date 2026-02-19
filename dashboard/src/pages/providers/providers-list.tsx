import { useNavigate } from "react-router";
import { Cloud, Plus } from "lucide-react";

import { PageShell } from "@/components/layout/page-shell";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/empty-state";
import { ErrorState } from "@/components/ui/error-state";
import { TableSkeleton } from "@/components/ui/skeleton";
import { useProviders } from "@/hooks/use-providers";
import type { ProviderStatus, ProviderSummary } from "@/types/api";

import { ProviderLogo } from "./provider-logos";

/* ── Status helpers ──────────────────────────────────────────────────────── */

function statusColor(status: ProviderStatus): string {
  switch (status) {
    case "connected":
      return "bg-success";
    case "configured":
      return "bg-warning";
    case "error":
      return "bg-danger";
    case "disabled":
      return "bg-text-muted";
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
    case "disabled":
      return "Disabled";
    default:
      return "Unconfigured";
  }
}

function statusPillClass(status: ProviderStatus): string {
  switch (status) {
    case "connected":
      return "bg-success/15 text-success";
    case "configured":
      return "bg-warning/15 text-warning";
    case "error":
      return "bg-danger/15 text-danger";
    default:
      return "bg-surface-3 text-text-muted";
  }
}

/* ── Card ─────────────────────────────────────────────────────────────────── */

function ProviderCard({ provider }: { provider: ProviderSummary }) {
  const navigate = useNavigate();

  return (
    <button
      type="button"
      aria-label={`View provider ${provider.display_name}`}
      onClick={() => navigate(`/providers/${provider.id}`)}
      className="flex flex-col rounded-lg border border-border bg-surface-1 p-5 text-left transition-all hover:border-accent/40 hover:bg-surface-2 hover:shadow-lg hover:shadow-accent/5"
    >
      {/* Header: logo + name */}
      <div className="flex items-center gap-3">
        <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-lg bg-surface-2">
          <ProviderLogo providerId={provider.id} size={32} />
        </div>
        <div className="min-w-0">
          <h3 className="font-medium text-text-primary">
            {provider.display_name}
          </h3>
          <p className="mt-0.5 truncate text-xs text-text-muted">
            {provider.description}
          </p>
        </div>
      </div>

      {/* Footer: status + capabilities */}
      <div className="mt-4 flex items-center justify-between">
        <span
          className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[11px] font-medium ${statusPillClass(provider.status)}`}
        >
          <span
            className={`inline-block h-1.5 w-1.5 rounded-full ${statusColor(provider.status)}`}
          />
          {statusLabel(provider.status)}
        </span>
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

/* ── Page ─────────────────────────────────────────────────────────────────── */

export function ProvidersListPage() {
  const navigate = useNavigate();
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

  const hasConfigured = (data ?? []).some(
    (p) => p.status !== "unconfigured",
  );

  return (
    <PageShell
      title="Providers"
      description="Manage your cloud providers and their capabilities."
      actions={
        <Button onClick={() => navigate("/providers/setup")}>
          <Plus size={14} className="mr-1.5" />
          Add Provider
        </Button>
      }
    >
      {(data ?? []).length === 0 ? (
        <EmptyState
          icon={<Cloud size={40} />}
          title="No providers available"
          description="No cloud providers are registered in the catalog."
        />
      ) : !hasConfigured ? (
        <EmptyState
          icon={<Cloud size={40} />}
          title="No providers configured"
          description="Set up your first cloud provider to unlock storage, compute, and training capabilities."
          action={
            <Button onClick={() => navigate("/providers/setup")}>
              <Plus size={14} className="mr-1.5" />
              Set up your first provider
            </Button>
          }
        />
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
