import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import type { HealthDetailed, ModelHealth } from "@/types/api";

export function useHealthDetailed() {
  return useQuery({
    queryKey: queryKeys.health.detailed(),
    queryFn: () => apiFetch<HealthDetailed>("/health/detailed"),
  });
}

export function useModelHealth() {
  return useQuery({
    queryKey: queryKeys.monitoring.list(),
    queryFn: () => apiFetch<ModelHealth[]>("/monitoring"),
  });
}
