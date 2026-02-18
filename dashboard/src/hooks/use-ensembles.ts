import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import type { Ensemble, EnsembleVersion } from "@/types/api";

export function useEnsembles() {
  return useQuery({
    queryKey: queryKeys.ensembles.list(),
    queryFn: () => apiFetch<Ensemble[]>("/ensembles"),
  });
}

export function useEnsemble(id: string) {
  return useQuery({
    queryKey: queryKeys.ensembles.detail(id),
    queryFn: () => apiFetch<Ensemble>(`/ensembles/${id}`),
    enabled: !!id,
  });
}

export function useEnsembleVersions(id: string) {
  return useQuery({
    queryKey: queryKeys.ensembles.versions(id),
    queryFn: () => apiFetch<EnsembleVersion[]>(`/ensembles/${id}/versions`),
    enabled: !!id,
  });
}
