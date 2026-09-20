# ML-BenchX — Phase 1 specification

ML-BenchX: transparent AutoML web app for tabular data. Phase 1 = vertical slice across
logic, API, DB, UI: upload → profile → review/override preprocessing plan → execute →
see every transformation explained → download artifacts.

This document is the source of truth for Phase 1. Implementation status is tracked in
[AUDIT.md](../AUDIT.md); verification evidence in [VERIFICATION.md](../VERIFICATION.md).

---

## 1. Ingestion

- CSV (delimiter sniffing), XLSX (sheet select, default first). Encoding: UTF-8,
  fallback via charset-normalizer.
- Size cap 50MB (config). Reject empty, headerless, single-column files with clear
  errors.
- Sanitize column names (strip, dedupe); keep original→clean mapping.
- Type inference with coercion: numeric strings (`"1,234"`, `"$5"`, `"12%"`), booleans
  (yes/no/true/false/y/n), datetime strings (≥90% parse rate), categorical, free text
  (flag + exclude), ID-like (unique ratio >0.95), constant/near-constant (top value
  ≥99%).
- User can override any inferred type.

## 2. Profiling

- Per column: type, missing %, unique count, top values; numeric: mean/std/min/max/skew,
  IQR outlier %, histogram bins.
- Dataset: rows, cols, duplicate rows, high-corr pairs (|r|>0.9), target distribution +
  imbalance ratio.
- Health score 0–100 = documented weighted penalties (missing, duplicates, imbalance,
  outliers, high-corr, constant/ID cols). Return per-component breakdown.
- Warnings: `{severity: info|warn|critical, message, suggestion}`.

## 3. Task detection

Non-numeric or ≤20 unique integer-like target → classification (binary if 2) else
regression. User override. Rows with missing target dropped (logged). Class with <2
samples → warn, fall back to non-stratified split.

## 4. Preprocessing = Planner + Executor

Planner outputs `PreprocessingPlan` (per-column actions + plain-English reasons),
editable by user. Executor builds sklearn `Pipeline`/`ColumnTransformer` from plan.

Order (leakage-safe):

**a) Stateless, pre-split:** dedupe rows, drop missing-target rows, type coercion,
datetime extraction (year/month/dow/hour + sin/cos).

**b) Split 80/20**, stratified for classification, seed from config (default 42).

**c) All data-dependent decisions computed on TRAIN only:**
- Drop column if missing >60%. Impute median (num) / mode (cat). Missing-indicator if
  missing >5%.
- Rare categories (<1% freq) → `"Other"`.
- Encoding: binary → 0/1; ≤10 levels → one-hot (`handle_unknown="ignore"`); >10 →
  sklearn `TargetEncoder` (cross-fitted).
- Scaling: Standard default; MinMax/Robust/None selectable.
- Outliers: default report-only; option IQR clipping (bounds from train).
- Skew |>1|: log suggestion; optional Yeo-Johnson.
- Leakage: |r|>0.99 with target OR single-feature depth-2 tree CV score ≥0.98 → exclude
  by default + critical warning; user can re-include.

**d) Post-split checks:** per-feature KS test train vs test (flag p<0.01); class balance
in both splits.

**e) Sanity baseline** on train (5-fold CV): Dummy vs LogisticRegression/Ridge. Report
scores only.

## 5. Glass-Box log

Every action → record `{run_id, seq, stage, step, column|null, action, params, reason,
before, after, severity, overridden: bool}`. JSON-serializable.

**Invariant:** every input column appears in the log with a final fate (kept/transformed/
dropped + why).

## 6. Runs & artifacts

Each execution = versioned run (overrides create new run). Save: fitted pipeline
(joblib), `plan.json`, `schema.json` (input columns, types, categories, target, task —
used for inference in Phase 2), `log.json`, train/test parquet.

**Deterministic:** same input + plan + seed → identical output hash.

## 7. API (FastAPI, Pydantic)

```
POST   /datasets
GET    /datasets
GET    /datasets/{id}
DELETE /datasets/{id}
GET    /datasets/{id}/profile
GET    /datasets/{id}/preview
POST   /datasets/{id}/plan        (target, options → proposed plan, no execution)
POST   /datasets/{id}/runs        (plan + overrides → execute)
GET    /runs/{id}
GET    /runs/{id}/log
GET    /runs/{id}/preview         (before/after)
GET    /runs/{id}/artifacts/{name}
```

Errors: uniform `{code, message, detail}`. Upload safety: extension + content check, no
path traversal, size enforced while streaming.

## 8. DB

SQLAlchemy 2.x + Alembic, SQLite. Tables: `projects`, `datasets`, `dataset_columns`,
`profiles`, `preprocessing_runs`, `preprocessing_steps`. All have `project_id` (default
project) + `created_at`. JSON columns for params. Files via `StorageBackend` interface
(`LocalStorage` impl).

## 9. UI (React + Vite + TS)

- **Upload:** drag-drop, progress, validation errors.
- **Dataset:** summary cards, health gauge + breakdown, warnings, column table (type
  override dropdown, missing, unique, flags), histogram on column click.
- **Plan review:** target picker, detected task, per-column planned actions + reasons
  with override dropdowns, global options (scaler, outliers, missing indicators).
- **Run result:** Glass-Box timeline grouped by stage (severity colors, before→after
  stats), before/after data preview, split summary, baseline scores, artifact downloads.
- **Runs history** per dataset. Loading/error/empty states everywhere.

## 10. Config

All thresholds/seeds in one config module.

---

## Verification requirements

**Fixtures.** Script generating deterministic synthetic datasets in `tests/fixtures`:
`clean_binary`, `multiclass`, `regression`, `messy_types` (numeric strings, currency, %,
mixed dates, whitespace cats), `high_missing`, `id_and_constant`, `leakage` (noisy
target copy), `imbalanced` 95/5, `high_cardinality`, `tiny` (<20 rows), unicode +
duplicate headers, latin-1 CSV, xlsx, empty, malformed, `perf_10mb` (~50 cols).

**Tests (pytest):**
- Unit test per spec rule.
- Leakage invariance: altering test rows never changes fitted params.
- Determinism: repeated run → identical hash.
- Round-trip: reload pipeline + schema, transform raw rows → matches run output.
- Log invariant: every input column has a final fate.
- Output: all numeric, no NaN/inf, train/test same columns.
- API contract tests incl. bad files.
- Core coverage ≥80%.

**Frontend:** typecheck, lint, build; Playwright smoke: upload → profile → plan →
override one column → run → log + downloads visible.

**Performance:** `perf_10mb` profile ≤10s, run ≤30s. Report timings.

---

## Rules

- Preserve working code; refactor only what's broken or off-spec. Ask before changing
  stack or deleting modules.
- No stubs, TODOs, mocked results, or weakened tests to force passes. If blocked, say so.
- Pin dependency versions. Type hints on all Python.
