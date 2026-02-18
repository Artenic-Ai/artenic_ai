import { useMutation } from "@tanstack/react-query";

import { apiFetch } from "@/lib/api-client";
import type { PredictRequest, PredictResponse } from "@/types/api";

export function usePredict(service: string) {
  return useMutation({
    mutationFn: (request: PredictRequest) =>
      apiFetch<PredictResponse>(`/${service}/predict`, {
        method: "POST",
        body: JSON.stringify(request),
      }),
  });
}
