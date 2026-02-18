import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import type { TrainingJob } from "@/types/api";

export function useTrainingJobs(params?: Record<string, string>) {
  const searchParams = params
    ? "?" + new URLSearchParams(params).toString()
    : "";

  return useQuery({
    queryKey: queryKeys.training.list(params),
    queryFn: () => apiFetch<TrainingJob[]>(`/training${searchParams}`),
  });
}

export function useTrainingJob(jobId: string) {
  return useQuery({
    queryKey: queryKeys.training.detail(jobId),
    queryFn: () => apiFetch<TrainingJob>(`/training/${jobId}`),
    enabled: !!jobId,
  });
}
