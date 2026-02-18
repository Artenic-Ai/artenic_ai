import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import type { AuditLogEntry, SettingsField } from "@/types/api";

export function useSettingsSchema() {
  return useQuery({
    queryKey: queryKeys.settings.schema(),
    queryFn: () => apiFetch<SettingsField[]>("/settings/schema"),
  });
}

export function useSettingsValues(scope: string) {
  return useQuery({
    queryKey: queryKeys.settings.scope(scope),
    queryFn: () => apiFetch<Record<string, string>>(`/settings/${scope}`),
  });
}

export function useAuditLog() {
  return useQuery({
    queryKey: queryKeys.settings.audit(),
    queryFn: () => apiFetch<AuditLogEntry[]>("/settings/audit"),
  });
}
