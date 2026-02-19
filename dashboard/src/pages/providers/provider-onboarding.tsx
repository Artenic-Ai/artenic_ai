import { useState } from "react";
import { useNavigate } from "react-router";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  CheckCircle,
  ExternalLink,
  Play,
  Power,
  Shield,
  XCircle,
} from "lucide-react";

import { PageShell } from "@/components/layout/page-shell";
import { Breadcrumb } from "@/components/ui/breadcrumb";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ErrorState } from "@/components/ui/error-state";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { TableSkeleton } from "@/components/ui/skeleton";
import { toast } from "@/components/ui/toast";
import {
  useConfigureProvider,
  useEnableProvider,
  useProviders,
  useTestProvider,
} from "@/hooks/use-providers";
import type { ConnectionTestResult, ProviderSummary } from "@/types/api";
import { MOCK_PROVIDER_DETAILS } from "@/mocks/providers";

import { ProviderLogo } from "./provider-logos";

/* ── Steps ───────────────────────────────────────────────────────────────── */

const STEPS = [
  { label: "Choose Provider", short: "Provider" },
  { label: "Credentials", short: "Credentials" },
  { label: "Settings", short: "Settings" },
  { label: "Test & Activate", short: "Activate" },
] as const;

type Step = 0 | 1 | 2 | 3;

/* ── Stepper ─────────────────────────────────────────────────────────────── */

function Stepper({ current }: { current: Step }) {
  return (
    <div className="mb-8 flex items-center justify-center gap-2">
      {STEPS.map((step, i) => {
        const done = i < current;
        const active = i === current;
        return (
          <div key={step.label} className="flex items-center gap-2">
            <div
              className={`flex h-8 w-8 items-center justify-center rounded-full text-xs font-bold transition-colors ${
                done
                  ? "bg-success text-white"
                  : active
                    ? "bg-accent text-white"
                    : "bg-surface-3 text-text-muted"
              }`}
            >
              {done ? <Check size={14} /> : i + 1}
            </div>
            <span
              className={`hidden text-sm sm:inline ${
                active
                  ? "font-medium text-text-primary"
                  : done
                    ? "text-success"
                    : "text-text-muted"
              }`}
            >
              {step.short}
            </span>
            {i < STEPS.length - 1 && (
              <div
                className={`h-px w-8 ${done ? "bg-success" : "bg-border"}`}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

/* ── Step 1: Choose Provider ─────────────────────────────────────────────── */

function StepChooseProvider({
  providers,
  selected,
  onSelect,
}: {
  providers: ProviderSummary[];
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="space-y-4">
      <div className="text-center">
        <h2 className="text-lg font-semibold text-text-primary">
          Choose a cloud provider
        </h2>
        <p className="mt-1 text-sm text-text-muted">
          Select the provider you want to configure for your ML infrastructure.
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        {providers.map((provider) => {
          const isSelected = selected === provider.id;
          return (
            <button
              key={provider.id}
              type="button"
              onClick={() => onSelect(provider.id)}
              className={`relative flex items-center gap-4 rounded-lg border-2 p-4 text-left transition-all ${
                isSelected
                  ? "border-accent bg-accent/5 shadow-md shadow-accent/10"
                  : "border-border bg-surface-1 hover:border-accent/30 hover:bg-surface-2"
              }`}
            >
              {/* Checkmark */}
              {isSelected && (
                <div className="absolute right-3 top-3 flex h-5 w-5 items-center justify-center rounded-full bg-accent text-white">
                  <Check size={12} />
                </div>
              )}

              {/* Logo */}
              <div className="flex h-14 w-14 flex-shrink-0 items-center justify-center rounded-xl bg-surface-2">
                <ProviderLogo providerId={provider.id} size={36} />
              </div>

              {/* Info */}
              <div className="min-w-0">
                <h3 className="font-semibold text-text-primary">
                  {provider.display_name}
                </h3>
                <p className="mt-0.5 text-xs text-text-muted">
                  {provider.description}
                </p>
                <div className="mt-2 flex gap-1.5">
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
        })}
      </div>
    </div>
  );
}

/* ── Step 2: Credentials ─────────────────────────────────────────────────── */

function StepCredentials({
  providerId,
  credentials,
  errors,
  onChange,
}: {
  providerId: string;
  credentials: Record<string, string>;
  errors: Record<string, string>;
  onChange: (key: string, value: string) => void;
}) {
  const detail = MOCK_PROVIDER_DETAILS[providerId];
  if (!detail) return null;

  return (
    <div className="space-y-4">
      <div className="text-center">
        <h2 className="text-lg font-semibold text-text-primary">
          Enter your credentials
        </h2>
        <p className="mt-1 text-sm text-text-muted">
          These credentials are encrypted at rest on the server.
        </p>
        {detail.website && (
          <a
            href={detail.website}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-2 inline-flex items-center gap-1 text-xs text-accent hover:underline"
          >
            {detail.display_name} documentation
            <ExternalLink size={10} />
          </a>
        )}
      </div>

      <Card>
        <div className="space-y-4 p-1">
          <div className="flex items-center gap-2 border-b border-border pb-3">
            <Shield size={14} className="text-accent" />
            <span className="text-xs font-medium uppercase tracking-wider text-text-muted">
              Credentials
            </span>
          </div>
          {detail.credential_fields.map((field) => (
            <Input
              key={field.key}
              label={field.label + (field.required ? " *" : "")}
              type={field.secret ? "password" : "text"}
              placeholder={field.placeholder || undefined}
              value={credentials[field.key] ?? ""}
              onChange={(e) => onChange(field.key, e.target.value)}
              error={errors[field.key]}
              autoComplete="off"
            />
          ))}
        </div>
      </Card>
    </div>
  );
}

/* ── Step 3: Configuration ───────────────────────────────────────────────── */

function StepConfiguration({
  providerId,
  config,
  onChange,
}: {
  providerId: string;
  config: Record<string, string>;
  onChange: (key: string, value: string) => void;
}) {
  const detail = MOCK_PROVIDER_DETAILS[providerId];
  if (!detail || detail.config_fields.length === 0) {
    return (
      <div className="space-y-4 text-center">
        <h2 className="text-lg font-semibold text-text-primary">
          Configuration
        </h2>
        <p className="text-sm text-text-muted">
          No additional settings needed for this provider. You can proceed to
          the next step.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="text-center">
        <h2 className="text-lg font-semibold text-text-primary">
          Additional settings
        </h2>
        <p className="mt-1 text-sm text-text-muted">
          Configure optional settings. Default values are pre-filled.
        </p>
      </div>

      <Card>
        <div className="space-y-4 p-1">
          {detail.config_fields.map((field) => (
            <div key={field.key}>
              <Input
                label={field.label}
                value={config[field.key] ?? ""}
                onChange={(e) => onChange(field.key, e.target.value)}
                placeholder={field.default || undefined}
              />
              {field.description && (
                <p className="mt-1 text-xs text-text-muted">
                  {field.description}
                </p>
              )}
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

/* ── Step 4: Test & Activate ─────────────────────────────────────────────── */

function StepTestActivate({
  providerId,
  testResult,
  testPending,
  enablePending,
  onTest,
  onEnable,
}: {
  providerId: string;
  testResult: ConnectionTestResult | null;
  testPending: boolean;
  enablePending: boolean;
  onTest: () => void;
  onEnable: () => void;
}) {
  const detail = MOCK_PROVIDER_DETAILS[providerId];

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-lg font-semibold text-text-primary">
          Test & Activate
        </h2>
        <p className="mt-1 text-sm text-text-muted">
          Test the connection to{" "}
          <strong>{detail?.display_name ?? providerId}</strong>, then activate
          it.
        </p>
      </div>

      {/* Test card */}
      <Card>
        <div className="flex flex-col items-center gap-4 py-4">
          {testResult === null && !testPending && (
            <>
              <div className="flex h-16 w-16 items-center justify-center rounded-full bg-surface-2">
                <Play size={24} className="text-accent" />
              </div>
              <p className="text-sm text-text-muted">
                Run a connection test to verify your credentials.
              </p>
              <Button onClick={onTest}>
                <Play size={14} className="mr-1.5" />
                Test Connection
              </Button>
            </>
          )}

          {testPending && (
            <>
              <Spinner />
              <p className="text-sm text-text-muted">Testing connection...</p>
            </>
          )}

          {testResult && !testPending && (
            <>
              <div
                className={`flex h-16 w-16 items-center justify-center rounded-full ${
                  testResult.success ? "bg-success/15" : "bg-danger/15"
                }`}
              >
                {testResult.success ? (
                  <CheckCircle size={28} className="text-success" />
                ) : (
                  <XCircle size={28} className="text-danger" />
                )}
              </div>
              <div className="text-center">
                <p
                  className={`text-sm font-medium ${
                    testResult.success ? "text-success" : "text-danger"
                  }`}
                >
                  {testResult.success
                    ? "Connection successful"
                    : "Connection failed"}
                </p>
                <p className="mt-1 text-xs text-text-muted">
                  {testResult.message}
                  {testResult.latency_ms != null &&
                    ` (${testResult.latency_ms.toFixed(0)}ms)`}
                </p>
              </div>
              {testResult.success ? (
                <Button onClick={onEnable} disabled={enablePending}>
                  <Power size={14} className="mr-1.5" />
                  {enablePending ? "Enabling..." : "Enable Provider"}
                </Button>
              ) : (
                <Button variant="secondary" onClick={onTest}>
                  <Play size={14} className="mr-1.5" />
                  Retry Test
                </Button>
              )}
            </>
          )}
        </div>
      </Card>
    </div>
  );
}

/* ── Main page ───────────────────────────────────────────────────────────── */

export function ProviderOnboardingPage() {
  const navigate = useNavigate();
  const { data: providers, isLoading, isError, refetch } = useProviders();

  const [step, setStep] = useState<Step>(0);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [credentials, setCredentials] = useState<Record<string, string>>({});
  const [config, setConfig] = useState<Record<string, string>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [testResult, setTestResult] = useState<ConnectionTestResult | null>(
    null,
  );

  const configureMutation = useConfigureProvider(selectedId ?? "");
  const testMutation = useTestProvider(selectedId ?? "");
  const enableMutation = useEnableProvider(selectedId ?? "");

  /* ── Navigation helpers ─────────────────────────────────────────── */

  function handleSelectProvider(id: string) {
    setSelectedId(id);
    // Initialize form fields from provider detail
    const detail = MOCK_PROVIDER_DETAILS[id];
    if (detail) {
      setCredentials(
        Object.fromEntries(detail.credential_fields.map((f) => [f.key, ""])),
      );
      setConfig(
        Object.fromEntries(
          detail.config_fields.map((f) => [f.key, f.default ?? ""]),
        ),
      );
    }
    setErrors({});
    setTestResult(null);
  }

  function validateCredentials(): boolean {
    if (!selectedId) return false;
    const detail = MOCK_PROVIDER_DETAILS[selectedId];
    if (!detail) return false;

    const newErrors: Record<string, string> = {};
    for (const field of detail.credential_fields) {
      if (field.required && !credentials[field.key]?.trim()) {
        newErrors[field.key] = `${field.label} is required`;
      }
    }
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }

  function goNext() {
    if (step === 0) {
      if (!selectedId) return;
      setStep(1);
    } else if (step === 1) {
      if (!validateCredentials()) return;
      setStep(2);
    } else if (step === 2) {
      // Save configuration via API, then go to step 3
      configureMutation.mutate(
        { credentials, config },
        {
          onSuccess: () => {
            toast("Configuration saved", "success");
            setStep(3);
          },
          onError: () => {
            toast("Failed to save configuration", "error");
          },
        },
      );
    }
  }

  function goBack() {
    if (step === 1) setStep(0);
    else if (step === 2) setStep(1);
    else if (step === 3) setStep(2);
  }

  function handleTest() {
    testMutation.mutate(undefined, {
      onSuccess: (result) => setTestResult(result),
      onError: () =>
        setTestResult({
          success: false,
          message: "Unexpected error during test",
          latency_ms: null,
        }),
    });
  }

  function handleEnable() {
    enableMutation.mutate(undefined, {
      onSuccess: () => {
        toast("Provider enabled!", "success");
        navigate(`/providers/${selectedId}`);
      },
      onError: () => toast("Failed to enable provider", "error"),
    });
  }

  function handleCredentialChange(key: string, value: string) {
    setCredentials((prev) => ({ ...prev, [key]: value }));
    if (errors[key]) {
      setErrors((prev) => {
        const next = { ...prev };
        delete next[key];
        return next;
      });
    }
  }

  /* ── Render ─────────────────────────────────────────────────────── */

  if (isLoading) {
    return (
      <PageShell title="Add Provider">
        <TableSkeleton rows={4} cols={2} />
      </PageShell>
    );
  }

  if (isError) {
    return (
      <PageShell title="Add Provider">
        <ErrorState
          message="Failed to load providers."
          onRetry={() => void refetch()}
        />
      </PageShell>
    );
  }

  return (
    <PageShell
      title="Add Provider"
      breadcrumb={
        <Breadcrumb
          items={[
            { label: "Providers", to: "/providers" },
            { label: "Setup" },
          ]}
        />
      }
    >
      <Stepper current={step} />

      <div className="mx-auto max-w-xl">
        {step === 0 && (
          <StepChooseProvider
            providers={providers ?? []}
            selected={selectedId}
            onSelect={handleSelectProvider}
          />
        )}

        {step === 1 && selectedId && (
          <StepCredentials
            providerId={selectedId}
            credentials={credentials}
            errors={errors}
            onChange={handleCredentialChange}
          />
        )}

        {step === 2 && selectedId && (
          <StepConfiguration
            providerId={selectedId}
            config={config}
            onChange={(key, value) =>
              setConfig((prev) => ({ ...prev, [key]: value }))
            }
          />
        )}

        {step === 3 && selectedId && (
          <StepTestActivate
            providerId={selectedId}
            testResult={testResult}
            testPending={testMutation.isPending}
            enablePending={enableMutation.isPending}
            onTest={handleTest}
            onEnable={handleEnable}
          />
        )}

        {/* Navigation buttons */}
        <div className="mt-8 flex justify-between border-t border-border pt-4">
          <div>
            {step > 0 && step < 3 && (
              <Button variant="ghost" onClick={goBack}>
                <ArrowLeft size={14} className="mr-1.5" />
                Back
              </Button>
            )}
            {step === 3 && testResult && !testResult.success && (
              <Button variant="ghost" onClick={() => setStep(1)}>
                <ArrowLeft size={14} className="mr-1.5" />
                Edit Credentials
              </Button>
            )}
          </div>
          <div>
            {step < 3 && (
              <Button
                onClick={goNext}
                disabled={
                  (step === 0 && !selectedId) || configureMutation.isPending
                }
              >
                {step === 2
                  ? configureMutation.isPending
                    ? "Saving..."
                    : "Save & Continue"
                  : "Next"}
                {!configureMutation.isPending && (
                  <ArrowRight size={14} className="ml-1.5" />
                )}
              </Button>
            )}
          </div>
        </div>
      </div>
    </PageShell>
  );
}
