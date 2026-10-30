/**
 * The API contract, and the only place the app talks to the server.
 *
 * Types here mirror the Pydantic models in `backend/app/schemas.py` and the
 * dictionaries `backend/app/workspace.py` returns. They are written by hand rather
 * than generated, so the rule is: if a field is added on the server and not here, the
 * client cannot see it, and that is the failure we want — a silent `undefined`
 * spreading through the UI is worse than a type error at the edge.
 *
 * Every mutating call returns the whole `Workspace`. That is deliberate on the server
 * side and it shapes this file: there are no partial updates to merge, so there is no
 * client-side reducer guessing what a toggle implied. The server decides, the client
 * renders.
 */

// ---------------------------------------------------------------------------------
// Pipeline configuration
// ---------------------------------------------------------------------------------

export type StepId =
  | "data_types"
  | "missing_values"
  | "duplicates"
  | "outliers"
  | "encoding"
  | "scaling"
  | "feature_selection"
  | "imbalance";

export type ScalerMethod = "standard" | "minmax" | "robust";
export type OutlierMethod = "iqr" | "zscore";
export type OutlierAction = "cap" | "remove" | "winsorize" | "flag";

export type PipelineConfig = {
  target: string | null;
  test_size: number;
  seed: number;
  excluded_columns: string[];
  data_types: { enabled: true; parse_dates: boolean; coerce_numeric: boolean };
  missing_values: {
    enabled: boolean;
    numeric_strategy:
      | "median"
      | "mean"
      | "constant"
      | "forward_fill"
      | "backward_fill"
      | "drop_rows";
    categorical_strategy:
      | "most_frequent"
      | "constant"
      | "forward_fill"
      | "backward_fill"
      | "drop_rows";
    numeric_fill_value: number;
    categorical_fill_value: string;
    drop_column_threshold: number;
  };
  duplicates: { enabled: boolean; subset: string[]; keep: "first" | "last" };
  outliers: {
    enabled: boolean;
    method: OutlierMethod;
    action: OutlierAction;
    iqr_multiplier: number;
    zscore_threshold: number;
    winsorize_quantile: number;
    columns: string[];
  };
  encoding: {
    enabled: boolean;
    binary_method: "label" | "onehot";
    low_cardinality_method: "onehot" | "ordinal" | "label";
    high_cardinality_method: "frequency" | "target" | "ordinal" | "drop";
    max_onehot_cardinality: number;
    handle_unknown: "ignore" | "error";
  };
  scaling: { enabled: boolean; method: ScalerMethod; columns: string[] };
  feature_selection: {
    enabled: boolean;
    variance_threshold: number | null;
    correlation_threshold: number | null;
    mutual_information: boolean;
    top_k: number | null;
  };
  imbalance: {
    enabled: boolean;
    action: "report" | "oversample" | "undersample" | "class_weight";
    ratio_threshold: number;
  };
};

// ---------------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------------

export type HealthDimension = {
  key: string;
  label: string;
  description: string;
  score: number;
  weight: number;
  contribution: number;
  facts: Record<string, unknown>;
  summary: string;
  status: "good" | "concern";
};

export type ClassBalance = {
  ratio: number;
  imbalanced: boolean;
  classes: { value: string; count: number; share: number }[];
  summary: string;
};

export type Health = {
  score: number;
  rating: string;
  dimensions: HealthDimension[];
  issues: { dimension: string; label: string; score: number; text: string; severity: string }[];
  class_balance: ClassBalance | null;
  formula: {
    description: string;
    terms: { label: string; score: number; weight: number; contribution: number }[];
    total: number;
  };
  rows: number;
  columns: number;
};

// ---------------------------------------------------------------------------------
// Steps, impact, compilation
// ---------------------------------------------------------------------------------

export type StepFacts = {
  values_affected?: number;
  columns_affected?: number;
  rows_affected?: number;
  per_column?: Record<string, number>;
  columns_to_drop?: string[];
  constant_columns?: string[];
  redundant_pairs?: string[][];
  scale_disparity?: number;
  widest_column?: string | null;
  narrowest_column?: string | null;
  columns_created?: number;
  ratio?: number;
  majority?: string;
  minority?: string;
  share?: number;
};

export type StepView = {
  id: StepId;
  label: string;
  description: string;
  enabled: boolean;
  locked: boolean;
  explanation: string;
  summary: string;
  facts: StepFacts;
  options: Record<string, unknown>;
  status: "off" | "idle" | "active";
};

export type Severity = "none" | "low" | "moderate" | "high" | "blocking";

export type ImpactReport = {
  step: StepId;
  label: string;
  description: string;
  facts: StepFacts;
  consequences: string[];
  severity: Severity;
  reversible: boolean;
  enabling: boolean;
  current_score: number | null;
  projected_score: number | null;
  delta: number | null;
};

export type CompileError = {
  type: string;
  message: string;
  line: number | null;
  cause: string | null;
  suggestion: string | null;
  traceback: string;
};

export type StepReport = {
  step: string;
  action: string;
  rows: number;
  columns: number;
  detail: Record<string, unknown>;
  columns_affected: string[];
};

export type Comparison = {
  score: { before: number; after: number; delta: number };
  rows: { before: number; after: number; delta: number };
  columns: { before: number; after: number; delta: number };
  dimensions: { key: string; label: string; before: number; after: number; delta: number }[];
};

export type Compilation = {
  id: string;
  status: "success" | "error";
  source: "pipeline" | "editor";
  duration_ms: number;
  logs: { text: string }[];
  error: CompileError | null;
  result: {
    rows_before?: number;
    rows_after?: number;
    columns_before?: number;
    columns_after?: number;
    steps?: StepReport[];
    train_rows?: number | null;
    test_rows?: number | null;
    /** How the held-out rows were chosen. The evidence for the leakage claim. */
    split?: {
      train_rows: number | null;
      test_rows: number | null;
      stratified: boolean | null;
      test_size: number | null;
      seed: number | null;
    } | null;
    comparison?: Comparison;
  };
  health_before: Health | Record<string, never>;
  health_after: Health | Record<string, never>;
  sandbox: string;
  created_at: string;
};

export type SandboxStatus = {
  level: "os_sandbox" | "limits_only";
  label: string;
  detail: string;
  network_blocked: boolean;
  filesystem_isolated: boolean;
};

export type Workspace = {
  id: string;
  dataset: {
    id: string;
    filename: string;
    rows: number;
    columns: number;
    size_bytes: number;
    /** What ingestion detected — shown so a misread separator cannot stay silent. */
    delimiter: string;
    encoding: string;
  };
  config: PipelineConfig;
  code: string;
  generated_code: string;
  code_is_edited: boolean;
  code_is_authoritative: boolean;
  has_uncompiled_changes: boolean;
  steps: StepView[];
  step_order: StepId[];
  baseline_health: Health;
  latest_compilation: Compilation | null;
  sandbox: SandboxStatus;
  columns: string[];
  updated_at: string;
};

export type HistoryItem = {
  id: number;
  kind: string;
  summary: string;
  created_at: string;
  restorable: boolean;
};

export type PreviewCell = { value: unknown; missing: boolean; outlier: boolean };

export type PreviewPage = {
  which: "original" | "processed";
  columns: { name: string; dtype: string; kind: string; missing: number }[];
  rows: { index: number; cells: Record<string, PreviewCell> }[];
  page: number;
  page_size: number;
  total_rows: number;
  total_pages: number;
};

export type DatasetSummary = {
  id: string;
  filename: string;
  n_rows: number;
  n_columns: number;
  size_bytes: number;
  health_score: number;
  created_at: string;
};

// ---------------------------------------------------------------------------------
// Transport
// ---------------------------------------------------------------------------------

/**
 * An error carrying the server's sentence rather than a status code.
 *
 * Everything the backend refuses comes back as `{ "error": "..." }` written for a
 * person, so the UI can render `error.message` directly and never has to translate a
 * 422 into English of its own.
 */
export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}


// --------------------------------------------------------------------------------------
// The plan and the run
//
// The propose -> review -> execute cycle. A plan is a decision per column, versioned and
// never mutated; a run executes one version and keeps what it produced. These mirror
// `app/plan.py`, `app/planning_api.py` and `app/db.py` by hand — see `app/schemas.py`
// for why the contract is written down in two places rather than generated.
// --------------------------------------------------------------------------------------

export type PlanAction =
  | "keep"
  | "drop"
  | "impute_median"
  | "impute_mean"
  | "impute_mode"
  | "impute_constant"
  | "encode_binary"
  | "encode_onehot"
  | "encode_ordinal"
  | "encode_target"
  | "encode_frequency"
  | "datetime_expand"
  | "passthrough";

export type PlanWarning = {
  severity: "info" | "warn" | "critical";
  message: string;
  suggestion?: string;
  kind?: string;
  [key: string]: unknown;
};

export type ColumnPlan = {
  column: string;
  original_name: string | null;
  kind: string;
  dtype: string;
  action: PlanAction;
  params: Record<string, unknown>;
  reason: string;
  /** The measurements the decision was made from, so the reason can be checked. */
  stats: Record<string, number | string>;
  alternatives: PlanAction[];
  overridden: boolean;
  warning: PlanWarning | null;
  add_indicator: boolean;
  rare_categories: string[];
};

export type PlanOptions = {
  scaler: ScalerMethod | "none";
  outliers: "report" | "clip";
  missing_indicators: boolean;
  rare_category_rate: number;
  onehot_max_cardinality: number;
  power_transform_skewed: boolean;
  test_size: number;
  seed: number;
};

export type PlanVersion = {
  id: string;
  version: number;
  origin: string;
  n_overridden: number;
  created_at: string;
};

export type Plan = {
  id: string;
  dataset_id: string;
  version: number;
  parent_id: string | null;
  origin: "proposed" | "edited";
  target: string;
  task: string;
  n_columns: number;
  n_overridden: number;
  created_at: string;
  updated_at: string;
  columns: ColumnPlan[];
  options: PlanOptions;
  warnings: PlanWarning[];
  prepared: Record<string, number | boolean>;
  versions: PlanVersion[];
  /** Which actions each kind of column may take. Sent with the plan so the dropdowns
   *  cannot offer something the backend would refuse. */
  actions_for_kind: Record<string, PlanAction[]>;
};

export type DriftFeature = {
  column: string;
  statistic: number;
  p_value: number;
  severity: string;
  message: string;
  suggestion: string;
};

export type RunChecks = {
  drift?: {
    flagged: number;
    tested: number;
    skipped: number;
    threshold: number;
    features: DriftFeature[];
    note: string;
  };
  balance?: {
    classes: {
      value: string;
      train_count: number;
      test_count: number;
      train_share: number;
      test_share: number;
      delta: number;
    }[];
    warnings: PlanWarning[];
  } | null;
  baseline?: {
    task: string;
    metric: string;
    folds: number;
    baseline: number;
    model: string;
    model_score: number;
    lift: number;
    severity: string;
    verdict: string;
    note: string;
  } | null;
};

export type Run = {
  id: string;
  status: "running" | "succeeded" | "failed";
  dataset_id: string;
  plan_id: string;
  plan_version: number;
  created_at: string;
  completed_at: string | null;
  duration_ms: number;
  target: string;
  task: string;
  split: {
    test_size: number;
    seed: number;
    stratified: boolean;
    train_rows: number;
    test_rows: number;
  };
  n_features: number;
  output_hash: string | null;
  artifacts: string[];
  error: { type: string; message: string } | null;
  warnings: PlanWarning[];
  checks: RunChecks;
  feature_schema: Record<string, unknown>;
};


export type Histogram =
  | { kind: "numeric"; bins: number; edges: number[]; counts: number[] }
  | { kind: "categorical"; labels: string[]; counts: number[]; other: number };

export type ColumnProfile = {
  column: string;
  kind: string;
  kind_label: string;
  dtype: string;
  n_missing: number;
  missing_rate: number;
  n_unique: number;
  unique_rate: number;
  n_rows: number;
  outliers: { count: number; share: number; lower: number; upper: number } | null;
  stats: Record<string, number | boolean> | null;
  /** Bin counts computed server-side — the browser holds one page of rows and would
   *  draw a different distribution on every page. */
  histogram: Histogram | null;
  top_values: { value: string; count: number; share: number }[];
  flags: string[];
  is_target: boolean;
  used: boolean;
  excluded_reason: string | null;
};

export type DatasetProfile = {
  dataset_id: string;
  n_rows: number;
  n_columns: number;
  duplicate_rows: number;
  columns: ColumnProfile[];
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch {
    throw new ApiError(
      "Could not reach the server. Check that the API is running on port 8000.",
      0,
    );
  }

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail =
      body?.error ??
      body?.detail ??
      `The server returned ${response.status}. Try again, or check the API logs.`;
    throw new ApiError(
      typeof detail === "string" ? detail : JSON.stringify(detail),
      response.status,
    );
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

const put = (body: unknown): RequestInit => ({ ...json(body), method: "PUT" });

// ---------------------------------------------------------------------------------
// Calls
// ---------------------------------------------------------------------------------

export const api = {
  async uploadDataset(file: File, onProgress?: (fraction: number) => void): Promise<DatasetSummary> {
    const form = new FormData();
    form.append("file", file);

    // XMLHttpRequest rather than fetch: upload progress is the one thing fetch still
    // cannot report, and a large file with no progress bar looks like a hang.
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/datasets");
      xhr.upload.addEventListener("progress", (event) => {
        if (event.lengthComputable && onProgress) onProgress(event.loaded / event.total);
      });
      xhr.addEventListener("load", () => {
        let body: Record<string, unknown> | null = null;
        try {
          body = JSON.parse(xhr.responseText);
        } catch {
          body = null;
        }
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(body as unknown as DatasetSummary);
        } else {
          reject(
            new ApiError(
              (body?.error as string) ??
                (body?.detail as string) ??
                `Upload failed with status ${xhr.status}.`,
              xhr.status,
            ),
          );
        }
      });
      xhr.addEventListener("error", () =>
        reject(new ApiError("The upload failed before it reached the server.", 0)),
      );
      xhr.send(form);
    });
  },

  listDatasets: () => request<DatasetSummary[]>("/api/datasets"),

  // --- the plan and the run ---------------------------------------------------------

  proposePlan: (datasetId: string, goal: string, target?: string) =>
    request<Plan>(`/api/datasets/${datasetId}/plan`, json({ goal, target: target ?? null })),

  readPlan: (datasetId: string, version?: number) =>
    request<Plan>(
      `/api/datasets/${datasetId}/plan${version ? `?version=${version}` : ""}`,
    ),

  /** Applies the user's changes as a NEW version; the plan named here is unchanged. */
  editPlan: (
    planId: string,
    columns: Record<string, { action: PlanAction }>,
    options?: Partial<PlanOptions>,
  ) => request<Plan>(`/api/plans/${planId}`, put({ columns, options: options ?? null })),

  runPlan: (planId: string) =>
    request<Run>(`/api/plans/${planId}/run`, { method: "POST" }),

  listRuns: (datasetId: string) =>
    request<{ dataset_id: string; runs: Run[] }>(`/api/datasets/${datasetId}/runs`),

  readRun: (runId: string) => request<Run>(`/api/runs/${runId}`),

  readProfile: (datasetId: string) =>
    request<DatasetProfile>(`/api/datasets/${datasetId}/profile`),

  artifactUrl: (runId: string, name: string) => `/api/runs/${runId}/artifacts/${name}`,

  deleteDataset: (datasetId: string) =>
    request<{
      deleted: boolean;
      filename: string;
      artifacts_removed: number;
      artifacts_orphaned: string[];
    }>(`/api/datasets/${datasetId}`, { method: "DELETE" }),


  openWorkspace: (datasetId: string) =>
    request<Workspace>(`/api/workspaces/from-dataset/${datasetId}`, { method: "POST" }),

  readWorkspace: (id: string) => request<Workspace>(`/api/workspaces/${id}`),

  updateConfig: (id: string, config: PipelineConfig, summary = "") =>
    request<Workspace>(`/api/workspaces/${id}/config`, put({ config, summary })),

  updateCode: (id: string, code: string) =>
    request<Workspace>(`/api/workspaces/${id}/code`, put({ code })),

  resetCode: (id: string) =>
    request<Workspace>(`/api/workspaces/${id}/code/reset`, { method: "POST" }),

  previewImpact: (id: string, step: StepId, enabled: boolean) =>
    request<ImpactReport>(`/api/workspaces/${id}/impact`, json({ step, enabled })),

  compile: (id: string, source: "auto" | "pipeline" | "editor" = "auto") =>
    request<{ compilation: Compilation; workspace: Workspace }>(
      `/api/workspaces/${id}/compile`,
      json({ source }),
    ),

  compilations: (id: string) => request<Compilation[]>(`/api/workspaces/${id}/compilations`),

  health: (id: string) => request<Health>(`/api/workspaces/${id}/health`),

  history: (id: string) => request<HistoryItem[]>(`/api/workspaces/${id}/history`),

  restore: (id: string, entryId: number) =>
    request<Workspace>(`/api/workspaces/${id}/history/${entryId}/restore`, { method: "POST" }),

  preview: (
    id: string,
    options: {
      which?: "original" | "processed";
      page?: number;
      pageSize?: number;
      search?: string;
      sort?: string;
      descending?: boolean;
    } = {},
  ) => {
    const query = new URLSearchParams({
      which: options.which ?? "original",
      page: String(options.page ?? 1),
      page_size: String(options.pageSize ?? 25),
      search: options.search ?? "",
      sort: options.sort ?? "",
      descending: String(options.descending ?? false),
    });
    return request<PreviewPage>(`/api/workspaces/${id}/preview?${query}`);
  },

  downloadUrl: (id: string, what: string) => `/api/workspaces/${id}/download/${what}`,
};
