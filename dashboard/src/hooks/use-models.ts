import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import type { Model } from "@/types/api";

export function useModels() {
  return useQuery({
    queryKey: queryKeys.models.list(),
    queryFn: () => apiFetch<Model[]>("/models"),
  });
}

export function useModel(id: string) {
  return useQuery({
    queryKey: queryKeys.models.detail(id),
    queryFn: () => apiFetch<Model>(`/models/${id}`),
    enabled: !!id,
  });
}
