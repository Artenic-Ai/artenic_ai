import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { apiFetch } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import type {
  CatalogResponse,
  ConfigureProviderRequest,
  ConnectionTestResult,
  ProviderComputeInstance,
  ProviderDetail,
  ProviderRegion,
  ProviderStorageOption,
  ProviderSummary,
} from "@/types/api";

/* ── Queries ─────────────────────────────────────────────────────────────── */

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

export function useProviderCatalog(id: string) {
  return useQuery({
    queryKey: queryKeys.providers.catalog(id),
    queryFn: () => apiFetch<CatalogResponse>(`/providers/${id}/catalog`),
    enabled: !!id,
  });
}

/* ── Mutations ───────────────────────────────────────────────────────────── */

export function useConfigureProvider(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: ConfigureProviderRequest) =>
      apiFetch<ProviderDetail>(`/providers/${id}/configure`, {
        method: "PUT",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: queryKeys.providers.detail(id) });
      void qc.invalidateQueries({ queryKey: queryKeys.providers.list() });
    },
  });
}

export function useEnableProvider(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () =>
      apiFetch<ProviderDetail>(`/providers/${id}/enable`, { method: "POST" }),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: queryKeys.providers.detail(id) });
      void qc.invalidateQueries({ queryKey: queryKeys.providers.list() });
    },
  });
}

export function useDisableProvider(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () =>
      apiFetch<ProviderDetail>(`/providers/${id}/disable`, { method: "POST" }),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: queryKeys.providers.detail(id) });
      void qc.invalidateQueries({ queryKey: queryKeys.providers.list() });
    },
  });
}

export function useDeleteProvider(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () =>
      apiFetch<void>(`/providers/${id}`, { method: "DELETE" }),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: queryKeys.providers.list() });
    },
  });
}

export function useTestProvider(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () =>
      apiFetch<ConnectionTestResult>(`/providers/${id}/test`, {
        method: "POST",
      }),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: queryKeys.providers.detail(id) });
    },
  });
}
