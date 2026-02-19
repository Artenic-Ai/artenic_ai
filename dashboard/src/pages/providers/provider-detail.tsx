import { useState } from "react";
import { useNavigate, useParams } from "react-router";
import {
  Cloud,
  DollarSign,
  ExternalLink,
  HardDrive,
  Play,
  Power,
  PowerOff,
  Server,
  Settings,
  Trash2,
  Zap,
} from "lucide-react";

import { PageShell } from "@/components/layout/page-shell";
import { Breadcrumb } from "@/components/ui/breadcrumb";
import { Button } from "@/components/ui/button";
import { Card, StatCard } from "@/components/ui/card";
import { type Column, DataTable } from "@/components/ui/data-table";
import { DetailRow } from "@/components/ui/detail-row";
import { Dialog } from "@/components/ui/dialog";
import { ErrorState } from "@/components/ui/error-state";
import { DetailSkeleton, TableSkeleton } from "@/components/ui/skeleton";
import { toast } from "@/components/ui/toast";
import {
  useDeleteProvider,
  useDisableProvider,
  useEnableProvider,
  useProvider,
  useProviderCatalog,
  useProviderCompute,
  useProviderRegions,
  useProviderStorage,
  useTestProvider,
} from "@/hooks/use-providers";
import { formatBytes, formatDateTime, formatNumber } from "@/lib/format";
import type {
  CatalogComputeFlavor,
  CatalogStorageTier,
  ProviderComputeInstance,
  ProviderRegion,
  ProviderStatus,
  ProviderStorageOption,
} from "@/types/api";

import { ProviderConfigureDialog } from "./provider-configure-dialog";

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

/* ── Table columns ───────────────────────────────────────────────────────── */

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

const CATALOG_COMPUTE_COLUMNS: Column<CatalogComputeFlavor>[] = [
  {
    key: "name",
    header: "Flavor",
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
    key: "price",
    header: "Price/hr",
    sortable: true,
    sortValue: (r) => r.price_per_hour ?? 0,
    render: (r) => (
      <span className="text-text-secondary">
        {r.price_per_hour != null
          ? `${r.price_per_hour.toFixed(4)} ${r.currency}`
          : "-"}
      </span>
    ),
  },
  {
    key: "category",
    header: "Category",
    render: (r) => (
      <span
        className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${
          r.category === "gpu"
            ? "bg-accent/10 text-accent"
            : "bg-surface-2 text-text-secondary"
        }`}
      >
        {r.category || "general"}
      </span>
    ),
  },
];

const CATALOG_STORAGE_COLUMNS: Column<CatalogStorageTier>[] = [
  {
    key: "name",
    header: "Tier",
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
    key: "price",
    header: "Price/GB/month",
    sortable: true,
    sortValue: (r) => r.price_per_gb_month ?? 0,
    render: (r) => (
      <span className="text-text-secondary">
        {r.price_per_gb_month != null
          ? `${r.price_per_gb_month.toFixed(4)} ${r.currency}`
          : "-"}
      </span>
    ),
  },
  {
    key: "description",
    header: "Description",
    render: (r) => (
      <span className="text-text-muted">{r.description || "-"}</span>
    ),
  },
];

/* ── Tabs ─────────────────────────────────────────────────────────────────── */

type Tab = "overview" | "configuration" | "catalog" | "storage" | "compute" | "regions";

/* ── Page ─────────────────────────────────────────────────────────────────── */

export function ProviderDetailPage() {
  const navigate = useNavigate();
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
  const { data: catalog, isLoading: catalogLoading } = useProviderCatalog(id);

  const [tab, setTab] = useState<Tab>("overview");
  const [configOpen, setConfigOpen] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);

  const testMutation = useTestProvider(id);
  const enableMutation = useEnableProvider(id);
  const disableMutation = useDisableProvider(id);
  const deleteMutation = useDeleteProvider(id);

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

  function handleTest() {
    testMutation.mutate(undefined, {
      onSuccess: (result) => {
        if (result.success) {
          toast(
            `Connection OK (${result.latency_ms?.toFixed(0) ?? "?"}ms)`,
            "success",
          );
        } else {
          toast(`Connection failed: ${result.message}`, "error");
        }
        void refetch();
      },
      onError: () => toast("Test failed", "error"),
    });
  }

  function handleToggle() {
    if (provider!.enabled) {
      disableMutation.mutate(undefined, {
        onSuccess: () => {
          toast("Provider disabled", "info");
          void refetch();
        },
        onError: () => toast("Failed to disable", "error"),
      });
    } else {
      enableMutation.mutate(undefined, {
        onSuccess: () => {
          toast("Provider enabled", "success");
          void refetch();
        },
        onError: () => toast("Failed to enable", "error"),
      });
    }
  }

  function handleDelete() {
    deleteMutation.mutate(undefined, {
      onSuccess: () => {
        toast("Provider configuration deleted", "success");
        navigate("/providers");
      },
      onError: () => toast("Failed to delete", "error"),
    });
  }

  const tabs: { key: Tab; label: string; icon: React.ReactNode }[] = [
    { key: "overview", label: "Overview", icon: <Zap size={14} /> },
    {
      key: "configuration",
      label: "Configuration",
      icon: <Settings size={14} />,
    },
    { key: "catalog", label: "Catalog", icon: <DollarSign size={14} /> },
    ...(provider.enabled
      ? [
          {
            key: "storage" as Tab,
            label: "Storage",
            icon: <HardDrive size={14} />,
          },
          {
            key: "compute" as Tab,
            label: "Compute",
            icon: <Server size={14} />,
          },
          { key: "regions" as Tab, label: "Regions", icon: <Cloud size={14} /> },
        ]
      : []),
  ];

  const anyPending =
    testMutation.isPending ||
    enableMutation.isPending ||
    disableMutation.isPending;

  return (
    <PageShell
      title={provider.display_name}
      description={provider.description}
      breadcrumb={
        <Breadcrumb
          items={[
            { label: "Providers", to: "/providers" },
            { label: provider.display_name },
          ]}
        />
      }
      actions={
        <div className="flex items-center gap-2">
          {provider.has_credentials && (
            <Button
              variant="secondary"
              onClick={handleTest}
              disabled={anyPending}
            >
              <Play size={14} className="mr-1.5" />
              {testMutation.isPending ? "Testing..." : "Test"}
            </Button>
          )}
          <Button onClick={() => setConfigOpen(true)}>
            <Settings size={14} className="mr-1.5" />
            Configure
          </Button>
          {provider.has_credentials && (
            <Button
              variant={provider.enabled ? "ghost" : "secondary"}
              onClick={handleToggle}
              disabled={anyPending}
            >
              {provider.enabled ? (
                <>
                  <PowerOff size={14} className="mr-1.5" />
                  Disable
                </>
              ) : (
                <>
                  <Power size={14} className="mr-1.5" />
                  Enable
                </>
              )}
            </Button>
          )}
          {provider.has_credentials && (
            <Button
              variant="destructive"
              onClick={() => setDeleteOpen(true)}
              disabled={anyPending}
            >
              <Trash2 size={14} className="mr-1.5" />
              Delete
            </Button>
          )}
        </div>
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
            className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition-colors ${
              tab === t.key
                ? "bg-surface-1 font-medium text-text-primary shadow-sm"
                : "text-text-secondary hover:text-text-primary"
            }`}
          >
            {t.icon}
            {t.label}
          </button>
        ))}
      </div>

      {/* ── Overview ──────────────────────────────────────────────────────── */}
      {tab === "overview" && (
        <div role="tabpanel" className="space-y-6">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard
              title="Status"
              value={statusLabel(provider.status)}
              subtitle={provider.status_message || undefined}
              icon={
                <span
                  className={`inline-block h-2.5 w-2.5 rounded-full ${statusColor(provider.status)}`}
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

          {/* Info card */}
          <Card title="Details">
            <dl className="space-y-3">
              <DetailRow label="Provider ID" value={provider.id} />
              <DetailRow
                label="Enabled"
                value={provider.enabled ? "Yes" : "No"}
              />
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
              {provider.created_at && (
                <DetailRow
                  label="Created"
                  value={formatDateTime(provider.created_at)}
                />
              )}
              {provider.updated_at && (
                <DetailRow
                  label="Updated"
                  value={formatDateTime(provider.updated_at)}
                />
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

      {/* ── Configuration ─────────────────────────────────────────────────── */}
      {tab === "configuration" && (
        <div role="tabpanel" className="space-y-6">
          <Card title="Credentials">
            {provider.has_credentials ? (
              <dl className="space-y-3">
                {provider.credential_fields.map((field) => (
                  <DetailRow
                    key={field.key}
                    label={field.label}
                    value={field.secret ? "********" : "(set)"}
                  />
                ))}
              </dl>
            ) : (
              <div className="flex flex-col items-center gap-3 py-6 text-center">
                <p className="text-sm text-text-muted">
                  No credentials configured yet.
                </p>
                <Button onClick={() => setConfigOpen(true)}>
                  <Settings size={14} className="mr-1.5" />
                  Configure Now
                </Button>
              </div>
            )}
          </Card>

          {provider.config_fields.length > 0 && (
            <Card title="Settings">
              <dl className="space-y-3">
                {provider.config_fields.map((field) => (
                  <DetailRow
                    key={field.key}
                    label={field.label}
                    value={provider.config[field.key] ?? field.default ?? "-"}
                  />
                ))}
              </dl>
            </Card>
          )}

          {provider.has_credentials && (
            <div className="flex gap-2">
              <Button onClick={() => setConfigOpen(true)}>
                <Settings size={14} className="mr-1.5" />
                Edit Configuration
              </Button>
            </div>
          )}
        </div>
      )}

      {/* ── Catalog ───────────────────────────────────────────────────────── */}
      {tab === "catalog" && (
        <div role="tabpanel" className="space-y-6">
          {catalogLoading ? (
            <TableSkeleton rows={4} cols={6} />
          ) : catalog ? (
            <>
              <div className="flex items-center gap-2">
                <span
                  className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${
                    catalog.is_live
                      ? "bg-success/10 text-success"
                      : "bg-warning/10 text-warning"
                  }`}
                >
                  {catalog.is_live ? "Live" : "Static"}
                </span>
                {catalog.cached && (
                  <span className="inline-block rounded-full bg-surface-2 px-2 py-0.5 text-xs font-medium text-text-muted">
                    Cached
                  </span>
                )}
              </div>

              <Card title="Compute Flavors">
                {catalog.compute.length > 0 ? (
                  <DataTable
                    columns={CATALOG_COMPUTE_COLUMNS}
                    data={catalog.compute}
                    keyFn={(r) => r.name}
                  />
                ) : (
                  <p className="text-sm text-text-muted">
                    No compute flavors available.
                  </p>
                )}
              </Card>

              {catalog.storage.length > 0 && (
                <Card title="Storage Tiers">
                  <DataTable
                    columns={CATALOG_STORAGE_COLUMNS}
                    data={catalog.storage}
                    keyFn={(r) => r.name}
                  />
                </Card>
              )}
            </>
          ) : (
            <p className="text-sm text-text-muted">
              No catalog data available for this provider.
            </p>
          )}
        </div>
      )}

      {/* ── Storage ───────────────────────────────────────────────────────── */}
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

      {/* ── Compute ───────────────────────────────────────────────────────── */}
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

      {/* ── Regions ───────────────────────────────────────────────────────── */}
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

      {/* ── Dialogs ───────────────────────────────────────────────────────── */}
      <ProviderConfigureDialog
        provider={provider}
        open={configOpen}
        onClose={() => setConfigOpen(false)}
        onConfigured={() => void refetch()}
      />

      <Dialog
        open={deleteOpen}
        onClose={() => setDeleteOpen(false)}
        title="Delete Provider Configuration"
      >
        <div className="space-y-4">
          <p className="text-sm text-text-secondary">
            This will remove all stored credentials and settings for{" "}
            <strong className="text-text-primary">
              {provider.display_name}
            </strong>
            . The provider will return to an unconfigured state.
          </p>
          <div className="flex justify-end gap-2">
            <Button variant="ghost" onClick={() => setDeleteOpen(false)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleDelete}
              disabled={deleteMutation.isPending}
            >
              {deleteMutation.isPending ? "Deleting..." : "Delete"}
            </Button>
          </div>
        </div>
      </Dialog>
    </PageShell>
  );
}
