import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import type { Budget, SpendingSummary } from "@/types/api";

export function useBudgets() {
  return useQuery({
    queryKey: queryKeys.budgets.list(),
    queryFn: () => apiFetch<Budget[]>("/budgets"),
  });
}

export function useSpending() {
  return useQuery({
    queryKey: queryKeys.budgets.spending(),
    queryFn: () => apiFetch<SpendingSummary[]>("/budgets/spending"),
  });
}

export function useSpendingHistory() {
  return useQuery({
    queryKey: queryKeys.budgets.spendingHistory(),
    queryFn: () =>
      apiFetch<
        Array<Record<string, number | string>>
      >("/budgets/spending/history"),
  });
}
