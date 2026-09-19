/** Types mirroring the FastAPI response, and the two calls the app makes. */

export type Strip = {
  column: string;
  before: string[];
  after: string[];
  changed: boolean[];
  rows: string[];
};

export type Step = {
  n: number;
  stage: "split" | "impute" | "encode" | "scale" | "derive";
  title: string;
  columns: string[];
  explanation: string;
  detail: Record<string, unknown>;
  fitted: boolean;
  strip: Strip | null;
  split?: { train: number; test: number; percent: number };
};

/** Values outside Tukey's fences for one numeric column. */
export type Outliers = {
  count: number;
  share: number;
  lower: number;
  upper: number;
};

/** One row of the per-column flag table. */
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
  outliers: Outliers | null;
  is_target: boolean;
  used: boolean;
  excluded_reason: string | null;
  leakage: { correlation: number; reason: string } | null;
  flags: string[];
};

export type CorrelatedPair = {
  columns: string[];
  correlation: number;
  reason: string;
};

export type ClassBalance = {
  ratio: number;
  majority_class: string;
  majority_share: number;
  minority_class: string;
  minority_count: number;
  minority_share: number;
  reason: string;
};

export type LeakageFlag = {
  kind: string;
  column: string;
  correlation: number;
  severity: string;
  reason: string;
};

/** One line of the health score's arithmetic. */
export type HealthDeduction = {
  factor: string;
  label: string;
  amount: number;
  weight: number;
  cap: number;
  points_lost: number;
};

export type Quality = {
  score: number;
  rows_used: number;
  missing_cells: number;
  duplicate_rows: number;
  issues: { severity: string; text: string }[];
  breakdown: HealthDeduction[];
};

export type Profile = {
  dataset_id: string;
  n_rows: number;
  n_columns: number;
  duplicate_rows: number;
  quality: Quality;
  columns: ColumnProfile[];
  correlated_pairs: CorrelatedPair[];
  leakage_flags: LeakageFlag[];
  class_balance: ClassBalance | null;
};

export type Analysis = {
  run_id: string;
  dataset_id: string;
  file: { name: string; rows: number; columns: number; encoding: string; delimiter: string };
  goal: string;
  target: {
    column: string;
    task: string;
    values: Record<string, number>;
    inferred: boolean;
    note: string;
  };
  columns: string[];
  cleaning: string[];
  quality: Quality;
  profile: Profile;
  options: { scaler: string; high_card_encoder: string };
  dropped: { column: string; reason: string }[];
  steps: Step[];
  result: { train_rows: number; test_rows: number; features: number };
  took_ms: number;
};

async function unwrap(response: Response): Promise<Analysis> {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(
      body?.error ?? "We couldn't reach the server. Check it is running and try again.",
    );
  }
  return body as Analysis;
}

export async function analyse(file: File, goal: string, target?: string): Promise<Analysis> {
  const form = new FormData();
  form.append("file", file);
  form.append("goal", goal);
  if (target) form.append("target", target);
  return unwrap(await fetch("/api/analyse", { method: "POST", body: form }));
}

export async function analyseExample(goal: string, target?: string): Promise<Analysis> {
  const form = new FormData();
  form.append("goal", goal);
  if (target) form.append("target", target);
  return unwrap(await fetch("/api/example", { method: "POST", body: form }));
}

export const downloadUrl = (runId: string, what: "data.csv" | "script.py") =>
  `/api/runs/${runId}/${what}`;
