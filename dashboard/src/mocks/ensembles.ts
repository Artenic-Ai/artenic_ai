import type { Ensemble, EnsembleVersion } from "@/types/api";

export const MOCK_ENSEMBLES: Ensemble[] = [
  {
    ensemble_id: "ens-001",
    name: "sentiment-production",
    service: "sentiment",
    strategy: "weighted_average",
    model_ids: ["mdl-a1b2c3d4"],
    description:
      "Production sentiment ensemble — weighted average of BERT v3 and distilled variant.",
    stage: "production",
    version: 3,
    strategy_config: {
      weights: { "mdl-a1b2c3d4": 0.7, "mdl-distilled": 0.3 },
    },
    created_at: "2026-01-20T10:00:00Z",
    updated_at: "2026-02-15T09:30:00Z",
  },
  {
    ensemble_id: "ens-002",
    name: "fraud-stacking",
    service: "fraud",
    strategy: "stacking",
    model_ids: ["mdl-u1v2w3x4", "mdl-q7r8s9t0"],
    description:
      "Stacking ensemble for fraud detection. LightGBM + XGBoost with logistic meta-learner.",
    stage: "staging",
    version: 1,
    strategy_config: { meta_learner: "logistic_regression" },
    created_at: "2026-02-10T14:00:00Z",
    updated_at: "2026-02-10T14:00:00Z",
  },
  {
    ensemble_id: "ens-003",
    name: "ner-multilingual-cascade",
    service: "ner",
    strategy: "cascade",
    model_ids: ["mdl-e5f6g7h8"],
    description:
      "Cascade NER: fast model filters easy cases, full model handles ambiguous tokens.",
    stage: "production",
    version: 5,
    strategy_config: { confidence_threshold: 0.85 },
    created_at: "2025-12-05T16:00:00Z",
    updated_at: "2026-02-12T11:20:00Z",
  },
];

export const MOCK_ENSEMBLE_VERSIONS: Record<string, EnsembleVersion[]> = {
  "ens-001": [
    {
      version: 1,
      strategy: "simple_average",
      model_ids: ["mdl-a1b2c3d4"],
      change_reason: "Initial setup with single model.",
      created_at: "2026-01-20T10:00:00Z",
    },
    {
      version: 2,
      strategy: "weighted_average",
      model_ids: ["mdl-a1b2c3d4", "mdl-distilled"],
      change_reason: "Added distilled variant for latency reduction.",
      created_at: "2026-02-01T15:00:00Z",
    },
    {
      version: 3,
      strategy: "weighted_average",
      model_ids: ["mdl-a1b2c3d4", "mdl-distilled"],
      change_reason: "Rebalanced weights after A/B test results (70/30).",
      created_at: "2026-02-15T09:30:00Z",
    },
  ],
  "ens-002": [
    {
      version: 1,
      strategy: "stacking",
      model_ids: ["mdl-u1v2w3x4", "mdl-q7r8s9t0"],
      change_reason: "Initial stacking ensemble for fraud detection.",
      created_at: "2026-02-10T14:00:00Z",
    },
  ],
  "ens-003": [
    {
      version: 5,
      strategy: "cascade",
      model_ids: ["mdl-e5f6g7h8"],
      change_reason: "Lowered confidence threshold from 0.90 to 0.85.",
      created_at: "2026-02-12T11:20:00Z",
    },
  ],
};
