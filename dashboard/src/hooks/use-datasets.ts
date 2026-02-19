import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import type {
  Dataset,
  DatasetFile,
  DatasetLineage,
  DatasetPreview,
  DatasetStats,
  DatasetVersion,
  StorageOption,
} from "@/types/api";

export function useDatasets() {
  return useQuery({
    queryKey: queryKeys.datasets.list(),
    queryFn: () => apiFetch<Dataset[]>("/datasets"),
  });
}

export function useDataset(id: string) {
  return useQuery({
    queryKey: queryKeys.datasets.detail(id),
    queryFn: () => apiFetch<Dataset>(`/datasets/${id}`),
    enabled: !!id,
  });
}

export function useDatasetFiles(id: string) {
  return useQuery({
    queryKey: queryKeys.datasets.files(id),
    queryFn: () => apiFetch<DatasetFile[]>(`/datasets/${id}/files`),
    enabled: !!id,
  });
}

export function useDatasetVersions(id: string) {
  return useQuery({
    queryKey: queryKeys.datasets.versions(id),
    queryFn: () => apiFetch<DatasetVersion[]>(`/datasets/${id}/versions`),
    enabled: !!id,
  });
}

export function useDatasetStats(id: string) {
  return useQuery({
    queryKey: queryKeys.datasets.stats(id),
    queryFn: () => apiFetch<DatasetStats>(`/datasets/${id}/stats`),
    enabled: !!id,
  });
}

export function useDatasetPreview(id: string) {
  return useQuery({
    queryKey: queryKeys.datasets.preview(id),
    queryFn: () => apiFetch<DatasetPreview>(`/datasets/${id}/preview`),
    enabled: !!id,
  });
}

export function useDatasetLineage(id: string) {
  return useQuery({
    queryKey: queryKeys.datasets.lineage(id),
    queryFn: () => apiFetch<DatasetLineage[]>(`/datasets/${id}/lineage`),
    enabled: !!id,
  });
}

export function useStorageOptions() {
  return useQuery({
    queryKey: queryKeys.datasets.storageOptions(),
    queryFn: () => apiFetch<StorageOption[]>("/datasets/storage-options"),
  });
}
