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
  stage: "split" | "impute" | "encode" | "scale";
  title: string;
  columns: string[];
  explanation: string;
  detail: Record<string, unknown>;
  fitted: boolean;
  strip: Strip | null;
  split?: { train: number; test: number; percent: number };
};

export type Analysis = {
  run_id: string;
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
  quality: {
    score: number;
    rows_used: number;
    missing_cells: number;
    duplicate_rows: number;
    issues: { severity: string; text: string }[];
  };
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
