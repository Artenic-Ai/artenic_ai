import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import type { ABTest, ABTestResults } from "@/types/api";

export function useABTests() {
  return useQuery({
    queryKey: queryKeys.abTests.list(),
    queryFn: () => apiFetch<ABTest[]>("/ab-tests"),
  });
}

export function useABTest(id: string) {
  return useQuery({
    queryKey: queryKeys.abTests.detail(id),
    queryFn: () => apiFetch<ABTest>(`/ab-tests/${id}`),
    enabled: !!id,
  });
}

export function useABTestResults(id: string) {
  return useQuery({
    queryKey: queryKeys.abTests.results(id),
    queryFn: () => apiFetch<ABTestResults>(`/ab-tests/${id}/results`),
    enabled: !!id,
  });
}
