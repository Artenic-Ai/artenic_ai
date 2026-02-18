import { useState } from "react";

import { Save } from "lucide-react";

import { PageShell } from "@/components/layout/page-shell";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ErrorState } from "@/components/ui/error-state";
import { Input, Select } from "@/components/ui/input";
import { PageSpinner } from "@/components/ui/spinner";
import { useAuditLog, useSettingsSchema, useSettingsValues } from "@/hooks/use-settings";
import { formatDateTime } from "@/lib/format";

export function SettingsPage() {
  const schema = useSettingsSchema();
  const values = useSettingsValues("global");
  const audit = useAuditLog();
  const [editedValues, setEditedValues] = useState<Record<string, string>>({});

  if (schema.isLoading || values.isLoading) return <PageSpinner />;
  if (schema.isError || values.isError) {
    return (
      <ErrorState
        message="Failed to load settings."
        onRetry={() => {
          void schema.refetch();
          void values.refetch();
        }}
      />
    );
  }

  const currentValues = { ...values.data, ...editedValues };

  // Group fields by section (key prefix before first dot)
  const sections = new Map<string, typeof schema.data>();
  for (const field of schema.data ?? []) {
    const section = field.key.split(".")[0] ?? "general";
    const existing = sections.get(section);
    if (existing) {
      existing.push(field);
    } else {
      sections.set(section, [field]);
    }
  }

  function handleSave() {
    // Demo mode: just reset edits
    setEditedValues({});
  }

  return (
    <PageShell
      title="Settings"
      description="Runtime configuration and audit log."
      actions={
        Object.keys(editedValues).length > 0 ? (
          <Button onClick={handleSave}>
            <Save size={16} className="mr-1" />
            Save Changes
          </Button>
        ) : undefined
      }
    >
      {/* Settings sections */}
      {Array.from(sections.entries()).map(([section, fields]) => (
        <Card
          key={section}
          title={section.charAt(0).toUpperCase() + section.slice(1)}
        >
          <div className="space-y-4">
            {fields?.map((field) => {
              const value = currentValues?.[field.key] ?? field.default;

              if (field.is_secret) {
                return (
                  <Input
                    key={field.key}
                    label={field.key}
                    value={value}
                    disabled
                    type="password"
                  />
                );
              }

              if (field.choices && field.choices.length > 0) {
                return (
                  <Select
                    key={field.key}
                    label={field.key}
                    value={value}
                    onChange={(e) =>
                      setEditedValues((prev) => ({
                        ...prev,
                        [field.key]: e.target.value,
                      }))
                    }
                    options={field.choices.map((c) => ({
                      value: c,
                      label: c,
                    }))}
                  />
                );
              }

              if (field.type === "bool") {
                return (
                  <Select
                    key={field.key}
                    label={field.key}
                    value={value}
                    onChange={(e) =>
                      setEditedValues((prev) => ({
                        ...prev,
                        [field.key]: e.target.value,
                      }))
                    }
                    options={[
                      { value: "true", label: "true" },
                      { value: "false", label: "false" },
                    ]}
                  />
                );
              }

              return (
                <Input
                  key={field.key}
                  label={field.key}
                  value={value}
                  onChange={(e) =>
                    setEditedValues((prev) => ({
                      ...prev,
                      [field.key]: e.target.value,
                    }))
                  }
                  type={field.type === "int" || field.type === "float" ? "number" : "text"}
                />
              );
            })}
          </div>
        </Card>
      ))}

      {/* Audit log */}
      <Card title="Audit Log">
        {audit.data && audit.data.length > 0 ? (
          <div className="space-y-2">
            {audit.data.map((entry) => (
              <div
                key={entry.id}
                className="flex items-start justify-between rounded-md border border-border bg-surface-2 px-4 py-3"
              >
                <div>
                  <span className="text-sm font-medium text-text-primary">
                    {entry.key}
                  </span>
                  <div className="mt-1 flex items-center gap-2 text-xs">
                    <span className="text-text-muted line-through">
                      {entry.old_value || "(empty)"}
                    </span>
                    <span className="text-text-muted">&rarr;</span>
                    <span className="text-text-primary">
                      {entry.new_value}
                    </span>
                  </div>
                </div>
                <div className="text-right text-xs text-text-muted">
                  <div>{entry.changed_by}</div>
                  <div>{formatDateTime(entry.changed_at)}</div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-text-muted">No audit entries.</p>
        )}
      </Card>
    </PageShell>
  );
}
