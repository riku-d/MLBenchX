# Mid-Sem demo guide — Stages 1 to 3

Every number in this document was produced by running the application against
`student_dropout_dataset_v3.csv` on the date below. Nothing here is estimated, and
nothing describes a feature that is not in the repository.

Verified: **2026-09-25**, branch `main`, against the working tree (uncommitted changes
included). Test suite: **339 passed, 0 failed**.

---

## 1. What is actually implemented

| Stage | Status |
|---|---|
| **1 — Ingestion** | Implemented and working |
| **2 — Data profiling** | Implemented and working |
| **3 — Glass-box preprocessing** | Implemented and working |
| 4 — Meta-learning engine | **Not implemented — future work** |
| 5 — Parallel training / Celery | **Not implemented — future work** |
| 6 — Explainability / SHAP | **Not implemented — future work** |
| 7 — Leaderboard & PDF report | **Not implemented — future work** |

Stages 4–7 have no code in this repository. There is no trainer, no model leaderboard,
no OpenML integration, no benchmark corpus, no recommender, no SHAP, no Celery and no
Redis. `openml`, `xgboost` and `shap` are not installed and are not in
`requirements.txt`. The interface exposes none of these, so nothing in the demo can
stray into them by accident.

---

## 2. Stage 1 — Ingestion

**Reads:** CSV, TSV, Excel (`.xlsx`/`.xls`), JSON (records, columns, line-delimited) and
Parquet. Excel is genuinely implemented — `app/services/ingest.py:_read_excel` reads a
named sheet or the first, and refuses a sheet name that does not exist rather than
silently falling back.

**Detects and validates:** encoding (UTF-8/BOM/latin-1 fallback), separator (`,` `;`
tab `|`), file size (50 MB cap, refused from `Content-Length` before the body is
parsed), empty files, headings with no rows, single-column files, ragged rows, and files
whose extension we do not read.

**Infers column types:** numeric, categorical, datetime and boolean, with placeholder
text (`N/A`, `-`, `unknown`, …) turned into genuine blanks first. A column converts only
when *every* non-blank value converts, so three stray words keep a column categorical
rather than destroying them.

**Metadata is preserved** on the `datasets` row (`encoding`, `delimiter`) and carried
into generated code — see Stage 3.

### On this dataset

```
10,000 rows × 19 columns      delimiter ","      encoding "utf-8"
Upload → stored + profiled:   0.22 s
Upload → workspace open:      0.43 s
Repair recorded: "Recognised 3 columns as true/false answers:
                  Internet_Access, Part_Time_Job, Scholarship"
```

---

## 3. Stage 2 — Data profiling

Measured per column: dtype, missing count and share, distinct count, cardinality,
numeric statistics, IQR outliers. Measured per dataset: duplicate rows, correlated
pairs, identifier-like columns, target leakage suspects, class balance.

### The health score — one number, one formula

There is **one** canonical health score, produced by `app/services/health.py:assess()`.
Six dimensions are measured independently, each multiplied by its weight and summed:

| Dimension | Weight | On this dataset |
|---|---|---|
| Completeness | 0.25 | 98.9 |
| Validity | 0.20 | 99.2 |
| Consistency | 0.15 | 100.0 |
| Uniqueness | 0.15 | 100.0 |
| Distribution | 0.10 | 67.0 |
| Feature Quality | 0.15 | 64.0 |
| | | **= 91** |

The same 91 appears on the dataset list, in the workspace header, and from
`GET /api/workspaces/{id}/health`. The interface shows the arithmetic under "How this
number was worked out".

A second, cheaper measurement exists at ingest time (blank cells, duplicates,
mostly-empty columns). It is **not** shown in the interface and is labelled in the API
as `"Ingest quality check (not the workspace health score)"` so it cannot be mistaken
for the health score.

### Issues found on this dataset

- Found 1 identifier-like column(s); 3 redundant pair(s).
- Found 1 heavily skewed column(s); the widest column varies 19,305× more than the narrowest.
- Class balance: 0 accounts for 76% of rows while 1 has 2,354.

---

## 4. Stage 3 — Glass-box preprocessing

Eight steps, each a switch: data types, missing values, duplicates, outliers, encoding,
scaling, feature selection, class imbalance. Seven are on by default for this dataset.

Every card states what the step will do **to this dataset**, in measured numbers. A step
with nothing to do says so ("No duplicate rows found.") rather than implying work.

### Turning a step off

Disabling opens a dialog first. For missing values on this dataset, measured live:

```
2,011 missing value(s) across 4 column(s), affecting 980 rows
  Parental_Education 511 · Family_Income 500 · Study_Hours_per_Day 500 · Stress_Index 500

Severity: blocking
  • Most scikit-learn estimators raise on the first blank they meet, so training
    will fail rather than degrade.
  • Any statistic computed downstream will silently skip those rows and be
    computed on a different subset per column.
```

### Leakage safety

The order is: **split → decide → fit on train → apply to test.** The split is the first
thing that happens, before any value is measured and before any decision is made.

Profiling the whole file and only *fitting* on the training rows is not enough: a
category appearing solely in the held-out rows could still push a column over the
one-hot threshold. Here it cannot.

Proven by `backend/tests/test_leakage.py` (12 tests), including:

- a spy on every `fit` asserting zero overlap with held-out row indices;
- the logged fill value equals the *training* median and differs from the whole-file median;
- **rewrite the held-out rows entirely and the resulting plan is byte-identical**;
- a category present only in the test fold cannot reach the encoder;
- target encoding never sees a held-out label.

### The split, on this dataset

```
Training set   8,000 rows
Test set       2,000 rows
Split          80 / 20
Strategy       Stratified
Random seed    42
```

Shown in the Insights panel after a compile. Stratification is applied for
classification targets and skipped where it is inappropriate.

### Code generation

The generated Python is the pipeline — the toggles write the configuration and the code
is generated from it, so the two cannot disagree. It is editable, and the app runs what
is written rather than what it generated.

The script reads the file **the way ingestion did**, carrying the detected separator and
encoding:

```python
df = pd.read_csv(INPUT_PATH, sep=',', encoding='utf-8')
```

### Sandbox

Edited code runs in an OS sandbox. `backend/tests/test_executor.py` (23 tests) genuinely
attempts to reach the network, delete repository files, fork-bomb and exhaust memory,
then asserts afterwards that each attempt failed. The sandbox was not weakened for this
work.

### Compile result on this dataset

```
status success · 1.3 s
10,000 × 19  →  10,000 rows × 29 columns
health 91 → 95
```

---

## 5. The demo — 4 to 5 minutes

Start both servers first, and **check the dev server is not suspended** (see Known
limitations).

| Time | Do | Say |
|---|---|---|
| **0:00–0:30** | Upload `student_dropout_dataset_v3.csv` | "ML-BenchX understands the dataset before it makes any preprocessing decision." |
| **0:30–1:15** | Point at the header: 10,000 × 19, health **91**. Open Insights → "How this number was worked out" | "Six dimensions, each measured, each weighted. Nothing here is a black box — the arithmetic is on screen." |
| **1:15–2:15** | Pipeline panel. Read two cards aloud. Then **toggle Missing Value Treatment off** | "Instead of silently transforming your data, it shows what it wants to do and why. Turn one off and it tells you the cost first — 2,011 values across 4 columns, and training would fail outright." **Cancel the dialog.** |
| **2:15–3:00** | Insights → Train / test split card | "We split before fitting *and before deciding*. Eight thousand training rows, two thousand held out, stratified, seed 42. Nothing in the held-out rows influenced this pipeline." |
| **3:00–4:00** | Code panel. Point at the `read_csv` line and a `.fit(` line. Press **Compile** | "This is the code that actually runs, not a description of it. Note it reads the file with the separator we detected, and every fit is on the training index only." Health moves 91 → 95. |
| **4:00–4:30** | Download menu → show the six artefacts | "Processed data, the original, the pipeline, the config, a report, or the whole project. Stages 1 to 3 are the transparent data-preparation foundation. Model training, meta-learning and automated selection are the later stages." |

### Do not click

- Nothing in the app exposes Stages 4–7, so there is nothing to avoid there.
- Avoid **Feature Selection** — it is off by default for this dataset and turning it on
  mid-demo changes the column count you have already quoted.
- Do not run the full test suite live; it takes 2 minutes 8 seconds.

---

## 6. Known limitations

These are real, and stating them is better than being caught by them.

1. **No model is trained.** The pipeline prepares data. A cross-validated sanity
   baseline (`services/checks.py:sanity_baseline`) reports whether the prepared features
   beat a dummy, which is a diagnostic about preparation — not a result, and not a
   leaderboard.
2. **`app/plan.py` and `app/runner.py` are unwired.** 1,173 statements that nothing
   imports — a partially-built alternative execution path. They are excluded from the
   coverage figure (see `.coveragerc`) so the number describes the shipped application.
   They are not deleted, because that is a direction decision, not a metric one.
3. **JSON re-reading in generated code is simplified.** Ingestion accepts three JSON
   shapes; the generated script emits a plain `pd.read_json`. CSV, TSV, Excel and
   Parquet round-trip exactly. Demo on a CSV.
4. **Single user, single process.** SQLite, local storage, no authentication. Workspace
   creation is idempotent and race-safe, but the app is not built for concurrent users.
5. **The recommender's KPIs cannot be measured yet**, because Stage 4 does not exist.

---

## 7. Stages 4–7 — future work

Not implemented. If asked:

> "Stages 1 to 3 are complete and demonstrable: ingestion, profiling and leakage-safe
> glass-box preprocessing. The meta-learning recommender needs a benchmark corpus with a
> winning-model label per dataset, and that label can only come from a trainer — so the
> trainer is the next thing to build, and the recommender follows it. We have stated zero
> downloaded datasets and zero benchmark runs rather than showing a placeholder number."

---

## 8. Test results

```
339 passed, 0 failed     coverage 88.37%     exit 0     2 min 6 s
```

| File | Tests | Covers |
|---|---|---|
| `test_rules.py` | 79 | One test per rule: each scaler, each encoder, dates, outliers, task detection |
| `test_codegen.py` | 50 | Every option combination, by running the generated script |
| `test_checks.py` | 37 | Drift, balance, sanity baseline |
| `test_workspace.py` | 33 | The workspace API as the interface drives it |
| `test_api.py` | 32 | Every dataset route, bad input, hostile filenames |
| `test_app.py` | 25 | Reading files, guessing the target |
| `test_glassbox.py` | 24 | Profiling, routing, the feature map |
| `test_executor.py` | 23 | Real attacks on the sandbox |
| `test_source_format.py` | 20 | Separator round-trip, one canonical health score |
| `test_leakage.py` | 12 | Held-out rows reach neither a fit nor a decision |

Coverage is **88.37%** against a 80% gate, so the command exits 0. `app/plan.py` and
`app/runner.py` are omitted from the measurement — they are unwired and unreachable, so
no test can cover them — with the reason recorded in `backend/.coveragerc`.

---

## 9. Verified metrics

Every figure below came from running the application on
`student_dropout_dataset_v3.csv`. The dataset was not modified.

| Metric | Value |
|---|---|
| Rows × columns | 10,000 × 19 |
| Delimiter / encoding detected | `,` / `utf-8` |
| Upload → stored and profiled | 0.22 s |
| Upload → workspace open | 0.43 s |
| Health score (canonical, all screens) | 91 |
| Target inferred | `Dropout` |
| Steps enabled by default | 7 of 8 |
| Missing values found | 2,011 across 4 columns, 980 rows |
| Compile duration | 1.3 s |
| Output shape | 10,000 rows × 29 columns |
| Health after preprocessing | 95 |
| Train / test | 8,000 / 2,000, stratified, seed 42 |
| Downloads | all six return HTTP 200 |
