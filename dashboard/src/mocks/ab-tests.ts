import type { ABTest, ABTestResults } from "@/types/api";

export const MOCK_AB_TESTS: ABTest[] = [
  {
    test_id: "ab-001",
    name: "sentiment-bert-v3-vs-distilled",
    service: "sentiment",
    status: "running",
    variants: {
      control: { model_id: "mdl-a1b2c3d4", traffic_pct: 50 },
      challenger: { model_id: "mdl-distilled", traffic_pct: 50 },
    },
    primary_metric: "accuracy",
    min_samples: 10000,
    created_at: "2026-02-14T10:00:00Z",
  },
  {
    test_id: "ab-002",
    name: "fraud-lgbm-vs-stacking",
    service: "fraud",
    status: "concluded",
    variants: {
      control: { model_id: "mdl-u1v2w3x4", traffic_pct: 50 },
      challenger: { ensemble_id: "ens-002", traffic_pct: 50 },
    },
    primary_metric: "f1_score",
    min_samples: 5000,
    winner: "challenger",
    created_at: "2026-01-25T08:00:00Z",
    concluded_at: "2026-02-08T18:00:00Z",
  },
  {
    test_id: "ab-003",
    name: "ner-cascade-threshold",
    service: "ner",
    status: "paused",
    variants: {
      control: { ensemble_id: "ens-003", config: { threshold: 0.85 }, traffic_pct: 50 },
      challenger: { ensemble_id: "ens-003", config: { threshold: 0.80 }, traffic_pct: 50 },
    },
    primary_metric: "entity_f1",
    min_samples: 8000,
    created_at: "2026-02-11T12:00:00Z",
  },
];

export const MOCK_AB_RESULTS: Record<string, ABTestResults> = {
  "ab-001": {
    test_id: "ab-001",
    status: "running",
    total_samples: 6342,
    variants: {
      control: {
        samples: 3187,
        mean: 0.912,
        std: 0.042,
        min: 0.78,
        max: 0.99,
        error_rate: 0.003,
        avg_latency_ms: 45,
      },
      challenger: {
        samples: 3155,
        mean: 0.897,
        std: 0.051,
        min: 0.72,
        max: 0.98,
        error_rate: 0.005,
        avg_latency_ms: 12,
      },
    },
  },
  "ab-002": {
    test_id: "ab-002",
    status: "concluded",
    total_samples: 12840,
    variants: {
      control: {
        samples: 6420,
        mean: 0.938,
        std: 0.031,
        min: 0.85,
        max: 0.99,
        error_rate: 0.008,
        avg_latency_ms: 8,
      },
      challenger: {
        samples: 6420,
        mean: 0.961,
        std: 0.022,
        min: 0.89,
        max: 0.99,
        error_rate: 0.004,
        avg_latency_ms: 15,
      },
    },
  },
  "ab-003": {
    test_id: "ab-003",
    status: "paused",
    total_samples: 2150,
    variants: {
      control: {
        samples: 1080,
        mean: 0.874,
        std: 0.058,
        min: 0.71,
        max: 0.96,
        error_rate: 0.012,
        avg_latency_ms: 68,
      },
      challenger: {
        samples: 1070,
        mean: 0.881,
        std: 0.055,
        min: 0.73,
        max: 0.97,
        error_rate: 0.015,
        avg_latency_ms: 72,
      },
    },
  },
};
