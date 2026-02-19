import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import type {
  ConnectionTestResult,
  ProviderComputeInstance,
  ProviderDetail,
  ProviderRegion,
  ProviderStorageOption,
  ProviderSummary,
} from "@/types/api";

export function useProviders() {
  return useQuery({
    queryKey: queryKeys.providers.list(),
    queryFn: () => apiFetch<ProviderSummary[]>("/providers"),
  });
}

export function useProvider(id: string) {
  return useQuery({
    queryKey: queryKeys.providers.detail(id),
    queryFn: () => apiFetch<ProviderDetail>(`/providers/${id}`),
    enabled: !!id,
  });
}

export function useProviderStorage(id: string) {
  return useQuery({
    queryKey: queryKeys.providers.storage(id),
    queryFn: () => apiFetch<ProviderStorageOption[]>(`/providers/${id}/storage`),
    enabled: !!id,
  });
}

export function useProviderCompute(id: string) {
  return useQuery({
    queryKey: queryKeys.providers.compute(id),
    queryFn: () =>
      apiFetch<ProviderComputeInstance[]>(`/providers/${id}/compute`),
    enabled: !!id,
  });
}

export function useProviderRegions(id: string) {
  return useQuery({
    queryKey: queryKeys.providers.regions(id),
    queryFn: () => apiFetch<ProviderRegion[]>(`/providers/${id}/regions`),
    enabled: !!id,
  });
}

export function useTestProvider(id: string) {
  return useQuery({
    queryKey: [...queryKeys.providers.all, id, "test"] as const,
    queryFn: () =>
      apiFetch<ConnectionTestResult>(`/providers/${id}/test`, {
        method: "POST",
      }),
    enabled: false,
  });
}
