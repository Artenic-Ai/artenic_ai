import { useState } from "react";
import { Shield } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { toast } from "@/components/ui/toast";
import { useConfigureProvider } from "@/hooks/use-providers";
import type { ProviderDetail } from "@/types/api";

interface Props {
  provider: ProviderDetail;
  open: boolean;
  onClose: () => void;
  onConfigured?: () => void;
}

export function ProviderConfigureDialog({
  provider,
  open,
  onClose,
  onConfigured,
}: Props) {
  const [credentials, setCredentials] = useState<Record<string, string>>(() =>
    Object.fromEntries(
      provider.credential_fields.map((f) => [f.key, ""]),
    ),
  );
  const [config, setConfig] = useState<Record<string, string>>(() =>
    Object.fromEntries(
      provider.config_fields.map((f) => [
        f.key,
        provider.config[f.key] ?? f.default ?? "",
      ]),
    ),
  );
  const [errors, setErrors] = useState<Record<string, string>>({});

  const configure = useConfigureProvider(provider.id);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();

    const newErrors: Record<string, string> = {};
    for (const field of provider.credential_fields) {
      if (field.required && !credentials[field.key]?.trim()) {
        newErrors[field.key] = `${field.label} is required`;
      }
    }
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }
    setErrors({});

    configure.mutate(
      { credentials, config },
      {
        onSuccess: () => {
          toast("Configuration saved", "success");
          onClose();
          onConfigured?.();
        },
        onError: () => {
          toast("Failed to save configuration", "error");
        },
      },
    );
  }

  function updateCredential(key: string, value: string) {
    setCredentials((prev) => ({ ...prev, [key]: value }));
    if (errors[key]) {
      setErrors((prev) => {
        const next = { ...prev };
        delete next[key];
        return next;
      });
    }
  }

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title={`Configure ${provider.display_name}`}
    >
      <form onSubmit={handleSubmit} className="space-y-5">
        {/* Credentials section */}
        <div>
          <div className="mb-3 flex items-center gap-2">
            <Shield size={14} className="text-accent" />
            <span className="text-xs font-medium uppercase tracking-wider text-text-muted">
              Credentials
            </span>
          </div>
          <p className="mb-3 text-xs text-text-muted">
            Credentials are encrypted at rest on the server.
          </p>
          <div className="space-y-3">
            {provider.credential_fields.map((field) => (
              <div key={field.key}>
                <Input
                  label={field.label + (field.required ? " *" : "")}
                  type={field.secret ? "password" : "text"}
                  placeholder={field.placeholder || undefined}
                  value={credentials[field.key] ?? ""}
                  onChange={(e) => updateCredential(field.key, e.target.value)}
                  error={errors[field.key]}
                  autoComplete="off"
                />
                {field.description && (
                  <p className="mt-1 text-xs text-text-muted">
                    {field.description}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Config section */}
        {provider.config_fields.length > 0 && (
          <div>
            <div className="mb-3">
              <span className="text-xs font-medium uppercase tracking-wider text-text-muted">
                Settings
              </span>
            </div>
            <div className="space-y-3">
              {provider.config_fields.map((field) => (
                <div key={field.key}>
                  <Input
                    label={field.label}
                    value={config[field.key] ?? ""}
                    onChange={(e) =>
                      setConfig((prev) => ({
                        ...prev,
                        [field.key]: e.target.value,
                      }))
                    }
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
          </div>
        )}

        {/* Actions */}
        <div className="flex justify-end gap-2 border-t border-border pt-4">
          <Button variant="ghost" type="button" onClick={onClose}>
            Cancel
          </Button>
          <Button type="submit" disabled={configure.isPending}>
            {configure.isPending ? "Saving..." : "Save Configuration"}
          </Button>
        </div>
      </form>
    </Dialog>
  );
}
