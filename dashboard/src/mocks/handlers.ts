import { ApiError } from "@/lib/api-client";

import { MOCK_AB_RESULTS, MOCK_AB_TESTS } from "./ab-tests";
import { MOCK_ACTIVITY } from "./activity";
import { MOCK_BUDGETS, MOCK_SPENDING, MOCK_SPENDING_HISTORY } from "./budgets";
import {
  MOCK_DATASETS,
  MOCK_DATASET_FILES,
  MOCK_DATASET_LINEAGE,
  MOCK_DATASET_PREVIEW,
  MOCK_DATASET_STATS,
  MOCK_DATASET_VERSIONS,
  MOCK_STORAGE_OPTIONS,
} from "./datasets";
import { MOCK_ENSEMBLE_VERSIONS, MOCK_ENSEMBLES } from "./ensembles";
import { MOCK_MODEL_HEALTH } from "./health";
import { MOCK_MODELS } from "./models";
import {
  MOCK_AUDIT_LOG,
  MOCK_SETTINGS_SCHEMA,
  MOCK_SETTINGS_VALUES,
} from "./settings";
import { MOCK_TRAINING_JOBS } from "./training";

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function handleDemoRequest<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  await delay(50 + Math.random() * 150);

  const method = (init?.method ?? "GET").toUpperCase();
  const url = new URL(path, "http://localhost");
  const segments = url.pathname.split("/").filter(Boolean);

  // ── Health ──────────────────────────────────────────────────────────────────
  if (path === "/health" && method === "GET") {
    return { status: "healthy" } as T;
  }
  if (path === "/health/ready" && method === "GET") {
    return { status: "ready", database: "connected" } as T;
  }
  if (path === "/health/detailed" && method === "GET") {
    return {
      status: "healthy",
      components: {
        database: "connected",
        cache: "connected",
        storage: "connected",
      },
      version: "0.2.0",
    } as T;
  }

  // ── Models ──────────────────────────────────────────────────────────────────
  if (segments[0] === "models" && method === "GET") {
    if (segments.length === 1) return MOCK_MODELS as T;
    const model = MOCK_MODELS.find((m) => m.model_id === segments[1]);
    if (model) return model as T;
    throw new ApiError(404, `Model ${segments[1]} not found`);
  }

  // ── Training ────────────────────────────────────────────────────────────────
  if (segments[0] === "training" && method === "GET") {
    if (segments.length === 1) {
      const status = url.searchParams.get("status");
      if (status) {
        return MOCK_TRAINING_JOBS.filter((j) => j.status === status) as T;
      }
      return MOCK_TRAINING_JOBS as T;
    }
    const job = MOCK_TRAINING_JOBS.find((j) => j.job_id === segments[1]);
    if (job) return job as T;
    throw new ApiError(404, `Training job ${segments[1]} not found`);
  }

  // ── Datasets ──────────────────────────────────────────────────────────────────
  if (segments[0] === "datasets" && method === "GET") {
    if (segments.length === 1) return MOCK_DATASETS as T;
    if (segments[1] === "storage-options") return MOCK_STORAGE_OPTIONS as T;
    const dsId = segments[1] ?? "";
    if (segments.length === 3) {
      if (segments[2] === "files") {
        const files = MOCK_DATASET_FILES[dsId];
        if (files) return files as T;
        throw new ApiError(404, `Dataset ${dsId} not found`);
      }
      if (segments[2] === "versions") {
        const versions = MOCK_DATASET_VERSIONS[dsId];
        if (versions) return versions as T;
        throw new ApiError(404, `Dataset ${dsId} not found`);
      }
      if (segments[2] === "stats") {
        const stats = MOCK_DATASET_STATS[dsId];
        if (stats) return stats as T;
        throw new ApiError(404, `Dataset ${dsId} not found`);
      }
      if (segments[2] === "preview") {
        const preview = MOCK_DATASET_PREVIEW[dsId];
        if (preview) return preview as T;
        throw new ApiError(404, `Dataset ${dsId} has no preview`);
      }
      if (segments[2] === "lineage") {
        const lineage = MOCK_DATASET_LINEAGE[dsId];
        if (lineage) return lineage as T;
        throw new ApiError(404, `Dataset ${dsId} not found`);
      }
    }
    const dataset = MOCK_DATASETS.find((d) => d.id === dsId);
    if (dataset) return dataset as T;
    throw new ApiError(404, `Dataset ${dsId} not found`);
  }

  // ── Ensembles ───────────────────────────────────────────────────────────────
  if (segments[0] === "ensembles" && method === "GET") {
    if (segments.length === 1) return MOCK_ENSEMBLES as T;
    const ensId = segments[1] ?? "";
    if (segments.length === 3 && segments[2] === "versions") {
      const versions = MOCK_ENSEMBLE_VERSIONS[ensId];
      if (versions) return versions as T;
      throw new ApiError(404, `Ensemble ${ensId} not found`);
    }
    const ensemble = MOCK_ENSEMBLES.find(
      (e) => e.ensemble_id === ensId,
    );
    if (ensemble) return ensemble as T;
    throw new ApiError(404, `Ensemble ${ensId} not found`);
  }

  // ── A/B Tests ───────────────────────────────────────────────────────────────
  if (segments[0] === "ab-tests" && method === "GET") {
    if (segments.length === 1) return MOCK_AB_TESTS as T;
    const testId = segments[1] ?? "";
    if (segments.length === 3 && segments[2] === "results") {
      const results = MOCK_AB_RESULTS[testId];
      if (results) return results as T;
      throw new ApiError(404, `A/B test ${testId} not found`);
    }
    const test = MOCK_AB_TESTS.find((t) => t.test_id === testId);
    if (test) return test as T;
    throw new ApiError(404, `A/B test ${testId} not found`);
  }

  // ── Budgets ─────────────────────────────────────────────────────────────────
  if (segments[0] === "budgets" && method === "GET") {
    if (path === "/budgets/spending/history") return MOCK_SPENDING_HISTORY as T;
    if (path === "/budgets/spending") return MOCK_SPENDING as T;
    if (segments.length === 1) return MOCK_BUDGETS as T;
  }

  // ── Settings ────────────────────────────────────────────────────────────────
  if (segments[0] === "settings" && method === "GET") {
    if (path === "/settings/schema") return MOCK_SETTINGS_SCHEMA as T;
    if (path === "/settings/audit") return MOCK_AUDIT_LOG as T;
    if (segments.length >= 2) return MOCK_SETTINGS_VALUES as T;
    return MOCK_SETTINGS_VALUES as T;
  }

  // ── Health Monitoring ───────────────────────────────────────────────────────
  if (segments[0] === "monitoring" && method === "GET") {
    if (segments.length === 1) return MOCK_MODEL_HEALTH as T;
    const health = MOCK_MODEL_HEALTH.find(
      (h) => h.model_id === segments[1],
    );
    if (health) return health as T;
    throw new ApiError(404, `Health data for ${segments[1]} not found`);
  }

  // ── Activity ────────────────────────────────────────────────────────────────
  if (path === "/activity" && method === "GET") {
    return MOCK_ACTIVITY as T;
  }

  // ── Write operations (return success stubs) ─────────────────────────────────
  if (method === "POST" || method === "PUT" || method === "PATCH") {
    return { success: true } as T;
  }
  if (method === "DELETE") {
    return { deleted: true } as T;
  }

  throw new ApiError(404, `Demo: no mock for ${method} ${path}`);
}
