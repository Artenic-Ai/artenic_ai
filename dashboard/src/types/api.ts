// ── Health ───────────────────────────────────────────────────────────────────

export interface HealthStatus {
  status: string;
}

export interface HealthReady {
  status: string;
  database: string;
}

export interface HealthDetailed {
  status: string;
  components: Record<string, unknown>;
  version?: string;
}

// ── Models ───────────────────────────────────────────────────────────────────

export interface Model {
  model_id: string;
  name: string;
  version: string;
  model_type: string;
  framework: string;
  description: string;
  tags: Record<string, string>;
  stage: string;
  created_at: string;
}

export interface RegisterModelRequest {
  name: string;
  version: string;
  model_type: string;
  framework?: string;
  description?: string;
  tags?: Record<string, string>;
}

// ── Training ─────────────────────────────────────────────────────────────────

export interface TrainingJob {
  job_id: string;
  service: string;
  model: string;
  provider: string;
  status: string;
  progress?: number;
  cost_eur?: number;
  cost_per_hour?: number;
  instance_type?: string;
  region?: string;
  is_spot: boolean;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
  config?: Record<string, unknown>;
  loss_history?: Array<{ epoch: number; loss: number }>;
}

export interface DispatchTrainingRequest {
  service: string;
  model: string;
  provider: string;
  config?: Record<string, unknown>;
  instance_type?: string;
  region?: string;
  is_spot?: boolean;
  max_runtime_hours?: number;
}

// ── Inference ────────────────────────────────────────────────────────────────

export interface PredictRequest {
  data: Record<string, unknown>;
  model_id?: string;
}

export interface PredictResponse {
  prediction: unknown;
  model_id: string;
  service: string;
  timestamp: string;
  inference_time_ms?: number;
}

export interface PredictBatchRequest {
  batch: Array<Record<string, unknown>>;
  model_id?: string;
}

// ── Ensembles ────────────────────────────────────────────────────────────────

export interface Ensemble {
  ensemble_id: string;
  name: string;
  service: string;
  strategy: string;
  model_ids: string[];
  description: string;
  stage: string;
  version: number;
  strategy_config?: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface CreateEnsembleRequest {
  name: string;
  service: string;
  strategy: string;
  model_ids: string[];
  description?: string;
  strategy_config?: Record<string, unknown>;
}

export interface UpdateEnsembleRequest {
  name?: string;
  model_ids?: string[];
  strategy?: string;
  strategy_config?: Record<string, unknown>;
  description?: string;
  change_reason?: string;
}

export interface EnsembleVersion {
  version: number;
  strategy: string;
  model_ids: string[];
  change_reason: string;
  created_at: string;
}

// ── A/B Tests ────────────────────────────────────────────────────────────────

export interface ABTest {
  test_id: string;
  name: string;
  service: string;
  status: string;
  variants: Record<string, unknown>;
  primary_metric: string;
  min_samples: number;
  winner?: string;
  created_at: string;
  concluded_at?: string;
}

export interface CreateABTestRequest {
  name: string;
  service: string;
  variants: Record<string, unknown>;
  primary_metric: string;
  min_samples?: number;
}

export interface ABTestResults {
  test_id: string;
  status: string;
  total_samples: number;
  variants: Record<
    string,
    {
      samples: number;
      mean: number;
      std: number;
      min: number;
      max: number;
      error_rate: number;
      avg_latency_ms: number;
    }
  >;
}

// ── Budgets ──────────────────────────────────────────────────────────────────

export interface Budget {
  id: string;
  scope: string;
  scope_value: string;
  period: string;
  limit_eur: number;
  alert_threshold_pct?: number;
  enabled: boolean;
  created_at: string;
}

export interface CreateBudgetRequest {
  scope: string;
  scope_value: string;
  period: string;
  limit_eur: number;
  alert_threshold_pct?: number;
}

export interface UpdateBudgetRequest {
  limit_eur?: number;
  alert_threshold_pct?: number;
  enabled?: boolean;
  period?: string;
}

export interface SpendingSummary {
  scope: string;
  scope_value: string;
  period: string;
  spent_eur: number;
  limit_eur: number;
  pct_used: number;
}

// ── Settings ─────────────────────────────────────────────────────────────────

export interface SettingsField {
  key: string;
  type: string;
  default: string;
  description: string;
  is_secret?: boolean;
  restart_required?: boolean;
  choices?: string[];
}

export interface AuditLogEntry {
  id: string;
  scope: string;
  section: string;
  key: string;
  old_value: string;
  new_value: string;
  changed_by: string;
  changed_at: string;
}

// ── Health Monitoring ────────────────────────────────────────────────────────

export interface ModelHealth {
  model_id: string;
  status: string;
  error_rate: number;
  avg_latency_ms: number;
  p95_latency_ms: number;
  p99_latency_ms: number;
  drift_score: number;
  total_predictions: number;
  last_checked: string;
}

// ── Activity Timeline ────────────────────────────────────────────────────────

export interface ActivityEvent {
  id: string;
  type: string;
  title: string;
  description: string;
  timestamp: string;
  severity?: "info" | "warning" | "error" | "success";
}

// ── Datasets ─────────────────────────────────────────────────────────────────

export interface Dataset {
  id: string;
  name: string;
  description: string;
  format: string;
  storage_backend: string;
  source: string;
  tags: Record<string, string>;
  current_version: number;
  total_size_bytes: number;
  total_files: number;
  schema_info?: {
    columns: Array<{ name: string; dtype: string; nullable: boolean }>;
  };
  created_at: string;
  updated_at?: string;
}

export interface StorageOption {
  id: string;
  label: string;
  available: boolean;
}

export interface DatasetVersion {
  id: number;
  dataset_id: string;
  version: number;
  hash: string;
  size_bytes: number;
  num_files: number;
  num_records?: number;
  change_summary: string;
  created_at: string;
}

export interface DatasetFile {
  id: number;
  dataset_id: string;
  version: number;
  filename: string;
  mime_type: string;
  size_bytes: number;
  hash: string;
  created_at: string;
}

export interface DatasetStats {
  total_size_bytes: number;
  total_files: number;
  num_records?: number;
  schema_info?: {
    columns: Array<{ name: string; dtype: string; nullable: boolean }>;
  };
  format_breakdown: Record<string, number>;
}

export interface DatasetLineage {
  id: number;
  dataset_id: string;
  dataset_version: number;
  entity_type: string;
  entity_id: string;
  role: string;
  created_at: string;
}

export interface DatasetPreview {
  columns: string[];
  rows: Array<Record<string, unknown>>;
  total_rows: number;
  truncated: boolean;
}

export interface CreateDatasetRequest {
  name: string;
  format: string;
  storage_backend?: string;
  description?: string;
  source?: string;
  tags?: Record<string, string>;
}

// ── Providers ────────────────────────────────────────────────────────────────

export interface ProviderCapability {
  type: string;
  name: string;
  description: string;
}

export interface CredentialField {
  key: string;
  label: string;
  required: boolean;
  secret: boolean;
  placeholder: string;
}

export interface ConfigField {
  key: string;
  label: string;
  default: string;
  description: string;
}

export type ProviderStatus =
  | "connected"
  | "configured"
  | "error"
  | "unconfigured"
  | "disabled";

export interface ProviderSummary {
  id: string;
  display_name: string;
  description: string;
  enabled: boolean;
  status: ProviderStatus;
  capabilities: ProviderCapability[];
}

export interface ProviderDetail {
  id: string;
  display_name: string;
  description: string;
  website: string;
  connector_type: string;
  enabled: boolean;
  status: ProviderStatus;
  status_message: string;
  has_credentials: boolean;
  config: Record<string, string>;
  capabilities: ProviderCapability[];
  credential_fields: CredentialField[];
  config_fields: ConfigField[];
  last_checked_at: string | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface ConnectionTestResult {
  success: boolean;
  message: string;
  latency_ms: number | null;
}

export interface ProviderStorageOption {
  provider_id: string;
  name: string;
  type: string;
  region: string;
  bytes_used: number | null;
  object_count: number | null;
}

export interface ProviderComputeInstance {
  provider_id: string;
  name: string;
  vcpus: number;
  memory_gb: number;
  disk_gb: number | null;
  gpu_type: string | null;
  gpu_count: number;
  region: string;
  available: boolean;
}

export interface ProviderRegion {
  provider_id: string;
  id: string;
  name: string;
}

export interface ConfigureProviderRequest {
  credentials: Record<string, string>;
  config: Record<string, string>;
}
