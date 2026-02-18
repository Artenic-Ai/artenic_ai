import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import type { ActivityEvent } from "@/types/api";

export function useActivity() {
  return useQuery({
    queryKey: queryKeys.activity.all,
    queryFn: () => apiFetch<ActivityEvent[]>("/activity"),
  });
}
