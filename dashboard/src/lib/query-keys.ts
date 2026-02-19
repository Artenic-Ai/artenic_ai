export const queryKeys = {
  health: {
    all: ["health"] as const,
    liveness: () => [...queryKeys.health.all, "liveness"] as const,
    ready: () => [...queryKeys.health.all, "ready"] as const,
    detailed: () => [...queryKeys.health.all, "detailed"] as const,
  },
  monitoring: {
    all: ["monitoring"] as const,
    list: () => [...queryKeys.monitoring.all, "list"] as const,
    detail: (modelId: string) =>
      [...queryKeys.monitoring.all, "detail", modelId] as const,
  },
  models: {
    all: ["models"] as const,
    list: () => [...queryKeys.models.all, "list"] as const,
    detail: (id: string) => [...queryKeys.models.all, "detail", id] as const,
  },
  training: {
    all: ["training"] as const,
    list: (params?: Record<string, string>) =>
      [...queryKeys.training.all, "list", params ?? {}] as const,
    detail: (jobId: string) =>
      [...queryKeys.training.all, "detail", jobId] as const,
  },
  ensembles: {
    all: ["ensembles"] as const,
    list: (params?: Record<string, string>) =>
      [...queryKeys.ensembles.all, "list", params ?? {}] as const,
    detail: (id: string) =>
      [...queryKeys.ensembles.all, "detail", id] as const,
    versions: (id: string) =>
      [...queryKeys.ensembles.all, "versions", id] as const,
  },
  abTests: {
    all: ["ab-tests"] as const,
    list: (params?: Record<string, string>) =>
      [...queryKeys.abTests.all, "list", params ?? {}] as const,
    detail: (id: string) =>
      [...queryKeys.abTests.all, "detail", id] as const,
    results: (id: string) =>
      [...queryKeys.abTests.all, "results", id] as const,
  },
  budgets: {
    all: ["budgets"] as const,
    list: () => [...queryKeys.budgets.all, "list"] as const,
    spending: () => [...queryKeys.budgets.all, "spending"] as const,
    spendingHistory: () =>
      [...queryKeys.budgets.all, "spending", "history"] as const,
  },
  datasets: {
    all: ["datasets"] as const,
    list: () => [...queryKeys.datasets.all, "list"] as const,
    detail: (id: string) => [...queryKeys.datasets.all, id] as const,
    files: (id: string) => [...queryKeys.datasets.all, id, "files"] as const,
    versions: (id: string) =>
      [...queryKeys.datasets.all, id, "versions"] as const,
    stats: (id: string) => [...queryKeys.datasets.all, id, "stats"] as const,
    preview: (id: string) =>
      [...queryKeys.datasets.all, id, "preview"] as const,
    lineage: (id: string) =>
      [...queryKeys.datasets.all, id, "lineage"] as const,
    storageOptions: () =>
      [...queryKeys.datasets.all, "storage-options"] as const,
  },
  activity: {
    all: ["activity"] as const,
  },
  settings: {
    all: ["settings"] as const,
    schema: () => [...queryKeys.settings.all, "schema"] as const,
    scope: (scope: string) =>
      [...queryKeys.settings.all, "scope", scope] as const,
    section: (scope: string, section: string) =>
      [...queryKeys.settings.all, "section", scope, section] as const,
    audit: () => [...queryKeys.settings.all, "audit"] as const,
  },
} as const;
