import { useState } from "react";
import { useParams } from "react-router";
import { Cloud, ExternalLink, HardDrive, Server } from "lucide-react";

import { PageShell } from "@/components/layout/page-shell";
import { Breadcrumb } from "@/components/ui/breadcrumb";
import { Card, StatCard } from "@/components/ui/card";
import { type Column, DataTable } from "@/components/ui/data-table";
import { DetailRow } from "@/components/ui/detail-row";
import { ErrorState } from "@/components/ui/error-state";
import { DetailSkeleton, TableSkeleton } from "@/components/ui/skeleton";
import {
  useProvider,
  useProviderCompute,
  useProviderRegions,
  useProviderStorage,
} from "@/hooks/use-providers";
import { formatBytes, formatDateTime, formatNumber } from "@/lib/format";
import type {
  ProviderComputeInstance,
  ProviderRegion,
  ProviderStorageOption,
} from "@/types/api";

const STORAGE_COLUMNS: Column<ProviderStorageOption>[] = [
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
    key: "type",
    header: "Type",
    render: (r) => <span className="text-text-secondary">{r.type}</span>,
  },
  {
    key: "region",
    header: "Region",
    render: (r) => <span className="text-text-secondary">{r.region}</span>,
  },
  {
    key: "bytes_used",
    header: "Used",
    sortable: true,
    sortValue: (r) => r.bytes_used ?? 0,
    render: (r) => (
      <span className="text-text-secondary">
        {r.bytes_used != null ? formatBytes(r.bytes_used) : "-"}
      </span>
    ),
  },
  {
    key: "object_count",
    header: "Objects",
    sortable: true,
    sortValue: (r) => r.object_count ?? 0,
    render: (r) => (
      <span className="text-text-muted">
        {r.object_count != null ? formatNumber(r.object_count) : "-"}
      </span>
    ),
  },
];

const COMPUTE_COLUMNS: Column<ProviderComputeInstance>[] = [
  {
    key: "name",
    header: "Instance",
    sortable: true,
    sortValue: (r) => r.name,
    render: (r) => (
      <span className="font-mono text-text-primary">{r.name}</span>
    ),
  },
  {
    key: "vcpus",
    header: "vCPUs",
    sortable: true,
    sortValue: (r) => r.vcpus,
    render: (r) => <span className="text-text-secondary">{r.vcpus}</span>,
  },
  {
    key: "memory",
    header: "Memory",
    sortable: true,
    sortValue: (r) => r.memory_gb,
    render: (r) => (
      <span className="text-text-secondary">{r.memory_gb} GB</span>
    ),
  },
  {
    key: "gpu",
    header: "GPU",
    sortable: true,
    sortValue: (r) => r.gpu_count,
    render: (r) =>
      r.gpu_type ? (
        <span className="text-text-primary">
          {r.gpu_count}x {r.gpu_type}
        </span>
      ) : (
        <span className="text-text-muted">-</span>
      ),
  },
  {
    key: "region",
    header: "Region",
    render: (r) => (
      <span className="text-text-secondary">{r.region || "-"}</span>
    ),
  },
];

const REGION_COLUMNS: Column<ProviderRegion>[] = [
  {
    key: "id",
    header: "ID",
    sortable: true,
    sortValue: (r) => r.id,
    render: (r) => (
      <span className="font-mono text-text-primary">{r.id}</span>
    ),
  },
  {
    key: "name",
    header: "Name",
    sortable: true,
    sortValue: (r) => r.name,
    render: (r) => <span className="text-text-secondary">{r.name}</span>,
  },
];

type Tab = "overview" | "storage" | "compute" | "regions";

export function ProviderDetailPage() {
  const { providerId } = useParams<{ providerId: string }>();
  const id = providerId ?? "";
  const { data: provider, isLoading, isError, refetch } = useProvider(id);
  const { data: storage, isLoading: storageLoading } = useProviderStorage(
    provider?.enabled ? id : "",
  );
  const { data: compute, isLoading: computeLoading } = useProviderCompute(
    provider?.enabled ? id : "",
  );
  const { data: regions, isLoading: regionsLoading } = useProviderRegions(
    provider?.enabled ? id : "",
  );

  const [tab, setTab] = useState<Tab>("overview");

  if (isLoading) {
    return (
      <PageShell title="Provider">
        <DetailSkeleton />
      </PageShell>
    );
  }
  if (isError || !provider) {
    return (
      <PageShell title="Provider">
        <ErrorState
          message="Failed to load provider."
          onRetry={() => void refetch()}
        />
      </PageShell>
    );
  }

  const statusColor =
    provider.status === "connected"
      ? "bg-success"
      : provider.status === "error"
        ? "bg-danger"
        : provider.status === "configured"
          ? "bg-warning"
          : "bg-text-muted";

  const tabs: { key: Tab; label: string }[] = [
    { key: "overview", label: "Overview" },
    ...(provider.enabled
      ? [
          { key: "storage" as Tab, label: "Storage" },
          { key: "compute" as Tab, label: "Compute" },
          { key: "regions" as Tab, label: "Regions" },
        ]
      : []),
  ];

  return (
    <PageShell
      title={provider.display_name}
      description={provider.description}
      breadcrumb={
        <Breadcrumb
          items={[
            { label: "Providers", href: "/providers" },
            { label: provider.display_name },
          ]}
        />
      }
    >
      {/* Tabs */}
      <div role="tablist" className="flex gap-1 rounded-lg bg-surface-2 p-1">
        {tabs.map((t) => (
          <button
            key={t.key}
            type="button"
            role="tab"
            aria-selected={tab === t.key}
            onClick={() => setTab(t.key)}
            className={`rounded-md px-3 py-1.5 text-sm transition-colors ${
              tab === t.key
                ? "bg-surface-1 font-medium text-text-primary shadow-sm"
                : "text-text-secondary hover:text-text-primary"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {tab === "overview" && (
        <div role="tabpanel" className="space-y-6">
          {/* Stats */}
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard
              title="Status"
              value={provider.status}
              icon={
                <span
                  className={`inline-block h-2.5 w-2.5 rounded-full ${statusColor}`}
                />
              }
            />
            <StatCard
              title="Connector"
              value={provider.connector_type}
              icon={<Cloud size={18} />}
            />
            <StatCard
              title="Capabilities"
              value={provider.capabilities.length}
              subtitle={provider.capabilities.map((c) => c.name).join(", ")}
            />
            <StatCard
              title="Last Checked"
              value={
                provider.last_checked_at
                  ? formatDateTime(provider.last_checked_at)
                  : "Never"
              }
            />
          </div>

          {/* Details */}
          <Card title="Configuration">
            <dl className="space-y-3">
              <DetailRow label="Provider ID" value={provider.id} />
              <DetailRow label="Enabled" value={provider.enabled ? "Yes" : "No"} />
              <DetailRow
                label="Credentials"
                value={provider.has_credentials ? "Configured" : "Not set"}
              />
              {provider.website && (
                <DetailRow
                  label="Documentation"
                  value={
                    <a
                      href={provider.website}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-accent hover:underline"
                    >
                      Visit
                      <ExternalLink size={12} />
                    </a>
                  }
                />
              )}
              {provider.status_message && (
                <DetailRow label="Message" value={provider.status_message} />
              )}
              {Object.keys(provider.config).length > 0 && (
                <>
                  <div className="border-t border-border pt-3">
                    <span className="text-xs font-medium uppercase tracking-wider text-text-muted">
                      Settings
                    </span>
                  </div>
                  {Object.entries(provider.config).map(([key, val]) => (
                    <DetailRow key={key} label={key} value={String(val)} />
                  ))}
                </>
              )}
            </dl>
          </Card>

          {/* Capabilities */}
          <Card title="Capabilities">
            <div className="space-y-3">
              {provider.capabilities.map((cap) => (
                <div
                  key={cap.type}
                  className="flex items-start gap-3 rounded-md bg-surface-2 p-3"
                >
                  <div className="mt-0.5 text-text-muted">
                    {cap.type === "storage" ? (
                      <HardDrive size={16} />
                    ) : (
                      <Server size={16} />
                    )}
                  </div>
                  <div>
                    <p className="text-sm font-medium text-text-primary">
                      {cap.name}
                    </p>
                    <p className="text-xs text-text-muted">{cap.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      {tab === "storage" && (
        <div role="tabpanel">
          <Card title="Storage Options">
            {storageLoading ? (
              <TableSkeleton rows={3} cols={5} />
            ) : storage && storage.length > 0 ? (
              <DataTable
                columns={STORAGE_COLUMNS}
                data={storage}
                keyFn={(r) => r.name}
              />
            ) : (
              <p className="text-sm text-text-muted">
                No storage containers found.
              </p>
            )}
          </Card>
        </div>
      )}

      {tab === "compute" && (
        <div role="tabpanel">
          <Card title="Compute Instances">
            {computeLoading ? (
              <TableSkeleton rows={4} cols={5} />
            ) : compute && compute.length > 0 ? (
              <DataTable
                columns={COMPUTE_COLUMNS}
                data={compute}
                keyFn={(r) => r.name}
              />
            ) : (
              <p className="text-sm text-text-muted">
                No compute instances found.
              </p>
            )}
          </Card>
        </div>
      )}

      {tab === "regions" && (
        <div role="tabpanel">
          <Card title="Available Regions">
            {regionsLoading ? (
              <TableSkeleton rows={3} cols={2} />
            ) : regions && regions.length > 0 ? (
              <DataTable
                columns={REGION_COLUMNS}
                data={regions}
                keyFn={(r) => r.id}
              />
            ) : (
              <p className="text-sm text-text-muted">No regions found.</p>
            )}
          </Card>
        </div>
      )}
    </PageShell>
  );
}
