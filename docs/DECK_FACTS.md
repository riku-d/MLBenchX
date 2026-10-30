# DECK_FACTS — repository evidence for the Sem-7 review deck

Gathered on **2026-09-20** against commit **`5a2328e`** ("Slice 3: profiling and health",
2026-09-20 18:50 +0530), branch `main`.

Working tree at the time of inspection was **not clean**: `backend/app/config.py` modified,
`backend/app/plan.py` and `backend/app/runner.py` untracked. Those two new modules are
counted as uncommitted work below.

Everything here was produced by running the code in this repository. Nothing is estimated.

---

## 1. Component inventory

Rule applied: **Done** = code exists *and* its tests pass. **Partial** = code exists without
tests, or with gaps recorded in `AUDIT.md`. **Not started** = no code.

| # | Component | Status | Evidence |
|---|---|---|---|
| 1 | **Ingestion** | **Done** | `backend/app/services/ingest.py` (978 lines). `read_upload()` — encoding detection (`_detect_encoding`, charset-normalizer fallback), delimiter sniffing (`_detect_delimiter`), CSV/XLSX/JSON/JSONL/Parquet readers (`_read_frame`, `_read_excel`, `_read_json`, `_read_parquet`), placeholder→NaN, numeric/datetime/boolean coercion (`_coerce_numeric`, `_coerce_datetime`, `_coerce_boolean`), header tidying, 50 MB streaming cap in `main.py`, content sniff `_check_content`, headerless detection `_looks_headerless`. Tests: `test_app.py` (25), `test_api.py` (32) — pass. Coverage 88.3%. |
| 2 | **Profiling / health score** | **Done** | `backend/app/services/profile.py` (262 lines) `build_profile()`; `backend/app/services/health.py` (611 lines) — weighted 0–100 score with a per-factor breakdown; `screening.py` correlated pairs + leakage flags; `target.py` task detection and target distribution. Tests: `test_glassbox.py` (24), `test_rules.py` (79) — pass. Coverage: profile 89.3%, health 81.2%, screening 87.2%, target 91.0%. Verified live on the sample dataset (§3 below). |
| 3 | **Glass-box planner** | **Done (group-level) / Partial (per-column object)** | Working, committed, tested: `services/glassbox.py` (1803 lines) plans and logs per *group* of columns; `services/explain.py` (709 lines) renders plain-English reasons; `services/impact.py` (454 lines) answers "what would this toggle cost". Tests: `test_glassbox.py`, `test_rules.py`, `test_workspace.py` (33) — pass. **Partial part:** the spec's per-column editable plan object exists only as **uncommitted, unwired, untested** `backend/app/plan.py` (507 lines, `ColumnPlan` / `PreprocessingPlan` / `propose()`) — imported by nothing in `backend/`, coverage **0.0%**. |
| 4 | **Executor / preprocessing** | **Done (engine) / Partial (plan-driven executor)** | `glassbox.fit_preprocessing()` splits *before* measuring, fits on the training fold only, transforms both folds; `codegen.py` (894 lines) emits a runnable script; `executor.py` (512 lines) runs user-edited code in a sandbox that is adversarially tested (`test_executor.py`, 23 tests — network, filesystem, fork and memory bombs). Tests: `test_codegen.py` (50), `test_leakage.py` (12) — pass. **Partial part:** the plan-driven executor `backend/app/runner.py` (666 lines — `execute()`, `build_pipeline()`, `build_schema()`, `output_hash()`, `verify_log_invariant()`) is **uncommitted, unwired, untested**, coverage **0.0%**. |
| 5 | **Trainer** | **Partial — sanity baseline only, no multi-model trainer** | What exists: `backend/app/services/checks.py:268 sanity_baseline()` — 5-fold cross-validated `DummyClassifier`/`DummyRegressor` vs `LogisticRegression`/`Ridge` on the training fold, wired into `glassbox.py:1427`. Tests: `test_checks.py` (37) — pass, coverage 89.7%. What does **not** exist anywhere in the repo: the 6-model leaderboard, Random Forest, XGBoost, SVM-RBF, KNN, Decision Tree as trained candidates, model comparison, parallel training. Greps for `xgboost`, `celery`, `redis` over `backend/app` and `frontend/src`: **zero hits**. `xgboost` is not in `requirements.txt`. |
| 6 | **SHAP** | **Not started** | Grep for `shap` over `backend/app` + `frontend/src`: only `.shape` / `n_features_in_` matches, no SHAP usage. `shap` absent from `backend/requirements.txt`. |
| 7 | **Meta-learning recommender** | **Not started** | Grep for `openml`, `meta_learn`, `meta-learn` over `backend/app` + `frontend/src`: **zero hits**. No fetch script, no meta-feature extractor, no meta-history table (`alembic upgrade head` creates 8 tables, none of them meta-related), no OpenML cache directory, no `openml` dependency. The only occurrences of "recommendation" are two prose sentences (`pipeline.py:20`, `Landing.tsx:652`). |
| 8 | **FastAPI API** | **Done** | `backend/app/main.py` (954 lines) — 12 routes incl. `POST /api/datasets`, `GET /api/datasets/{id}/profile`, `POST /api/datasets/{id}/preprocess`, artifact downloads, `/api/health`. `backend/app/workspace.py` (929 lines) — 13 workspace routes (config, code, impact, compile, history, restore, download). Tests: `test_api.py` (32) + `test_workspace.py` (33) — pass. Coverage: main 87.4%, workspace 89.8%. No WebSocket endpoint exists (grep: zero hits) — the slide-4 "REST + WebSocket" interface is REST only today. |
| 9 | **Database / migrations** | **Done** | `backend/app/db.py` (362 lines), SQLAlchemy 2.x; Alembic with 2 revisions. Verified: `alembic upgrade head` on an empty SQLite file succeeds and creates `alembic_version, compilations, datasets, history, preprocessing_logs, preprocessing_runs, profiles, workspaces`. `storage.py` — `StorageBackend` + `LocalStorage`, path traversal blocked. Coverage: db 93.9%, storage 86.2%. |
| 10 | **React UI** | **Done (build/lint green) / Partial (against spec §9)** | `frontend/src`, 4604 lines across 11 TS/TSX files: `Landing.tsx`, `UploadScreen.tsx`, `Workspace.tsx`, `PipelinePanel.tsx`, `InsightsPanel.tsx`, `BottomDock.tsx`, `CodePanel.tsx` (Monaco editor), `ui.tsx`, `api.ts`. Verified: `npm run lint` (oxlint) exits 0; `npm run build` (tsc -b + vite) exits 0, 312.59 kB JS. **Partial part:** three screens only (landing / upload / workspace) — no plan-review screen, no target picker, no per-column override dropdowns, no histogram, no leaderboard, no SHAP view (`AUDIT.md` §UI). |

### Test and quality baseline (measured, not quoted)

- `cd backend && ../.venv/bin/python -m pytest tests/ -q` → **315 passed, 0 failed**, 118 s.
- Per file: `test_rules.py` 79 · `test_codegen.py` 50 · `test_checks.py` 37 · `test_workspace.py` 33 · `test_api.py` 32 · `test_app.py` 25 · `test_glassbox.py` 24 · `test_executor.py` 23 · `test_leakage.py` 12.
- Coverage **78.6%**, which **fails** the `--cov-fail-under=80` gate in `backend/pytest.ini`, so the pytest command exits non-zero even though every test passes. Cause: the two uncommitted modules `app/plan.py` and `app/runner.py` sit at 0.0%. Excluding those two files, coverage over the committed code is **87.8%**.
- `npm run lint` → 0 · `npm run build` → 0 · `alembic upgrade head` from empty → 0.

### Stack corrections for slide 8 (from the pinned manifests)

`backend/requirements.txt`: pandas 2.3.3 · numpy 2.4.1 · scikit-learn 1.8.0 · scipy 1.17.0 ·
joblib 1.5.3 · pyarrow 23.0.1 · openpyxl 3.1.5 · charset-normalizer 3.4.6 · fastapi 0.141.1 ·
starlette 1.5.0 · uvicorn 0.52.1 · pydantic 2.13.4 · python-multipart 0.0.32 ·
SQLAlchemy 2.0.54 · alembic 1.20.0 · psutil 7.2.1.
`requirements-dev.txt`: pytest 9.1.1 · pytest-cov 7.1.0 · httpx 0.28.1.
Interpreter in `.venv`: **Python 3.14.2** (not 3.11).

`frontend/package.json`: **React 19.2.8** (not 18) · **Vite 8.2.1** · **TypeScript 6.0.3** ·
Tailwind 4.3.3 · Monaco 0.56.0 · recharts 3.10.1 (installed, imported by zero files) ·
framer-motion 13.4.0 (same) · oxlint 1.77.0.

**Absent from both manifests and from all source:** Celery, Redis, XGBoost, SHAP, OpenML,
Docker / docker-compose, Postgres, Locust.

---

## 2. Per-member contribution

```
$ git log --format='%an <%ae>' | sort | uniq -c
  18 Rohit Dutta <rohitdutta2103@gmail.com>
```

**All 18 commits in this repository are authored by a single account**
(`Rohit Dutta <rohitdutta2103@gmail.com>`), dated 2026-09-19 to 2026-09-20. There are no
co-author trailers, no merge commits, and no second identity anywhere in the history.

Therefore, for **Aditya Kumar Barnwal**, **Ashmita Barnwal** and **Sukriti Kumari**:

> **unclear — needs team confirmation.** The version history cannot attribute any file to
> them. Either the work was committed through one shared account, or their contributions
> are not yet in this repository. This must be confirmed with the team before the review;
> it is not inferable from the code.

Files most touched (by commit count) across the whole — single-author — history:
`tests/test_rules.py` (7) · `services/ingest.py` (6) · `services/glassbox.py` (6) ·
`app/main.py` (6) · `app/config.py` (5) · `services/explain.py` (4) · `AUDIT.md` (4).

Commit titles, newest first: Slice 3 profiling and health · Slice 2 ingestion and type
inference · Slice 1 leakage and correctness · Slice 0 pin dependencies, coverage gate,
fixture catalogue · Persist Phase 1 spec and audit · JSON and Parquet reading · Test the
generator, sandbox and workspace API · Landing page and interactive workspace · Health
dimensions, impact analysis, workspace API · Pipeline config, code generator, sandboxed
executor · Exported script splits identically · Recognise dates on the way in · Tests for
every rule, API, wider leakage guarantee · Profile dashboard per-column table and warnings ·
Database, storage backend, resource-shaped API · Fix decision-layer leakage, task detection,
upload handling · Central config module · Baseline before Phase 1 audit fixes.

---

## 3. Slide-14 evidence — a real run of the real pipeline

**Dataset:** `backend/app/sample/campus_records.csv` — the sample bundled with the repo.
1,505 rows × 12 columns, 112,598 bytes, UTF-8, `;` delimited.
**Run:** 2026-09-20, commit `5a2328e`, via `POST /api/datasets` → `POST /api/datasets/{id}/preprocess`
against the real FastAPI app in-process (temporary DB and storage), plus a direct call to
`glassbox.fit_preprocessing()` for the post-split checks. Machine: darwin 25.6.0,
Python 3.14.2.

### Health score (chart-ready)

Final score **87 / 100**. Breakdown as returned by the API (`quality.breakdown`):

| Factor | Label | Amount | Weight | Cap | Points lost |
|---|---|---|---|---|---|
| `missing_cells` | blank cells | 6.6334 % | 1.5 | 35.0 | **9.95** |
| `sparse_columns` | mostly empty columns | 1 column | 3.0 | 12.0 | **3.00** |
| `duplicate_rows` | duplicate rows | 0.3322 % | 1.0 | 15.0 | **0.33** |
| | | | | **Total lost** | **13.28** |

Chart series for slide 14 — "points lost by factor": blank cells 9.95 · mostly empty columns 3.00 ·
duplicate rows 0.33 · points retained 86.72 (displayed score 87).

Issues raised: 2 `warn` + 1 `info` —
"5 row(s) are exact duplicates of another row." ·
"1 column(s) are more than 60% empty: family_income." ·
"6.6% of all cells are blank."

### Before → after (chart-ready)

| Measure | Before | After |
|---|---|---|
| Rows | 1,505 | 1,500 usable → 1,200 train + 300 test |
| Columns / features | 12 | 12 (3 dropped, survivors expanded by one-hot) |
| Missing cells | **1,198** | **0** |
| Columns containing blanks | 6 | 0 |

Missing cells before, by column: `family_income` 1,085 · `prev_semester_score` 41 ·
`attendance_pct` 29 · `study_hours_week` 23 · `department` 12 · `home_city` 8.

### Stat cards (chart-ready)

- **Columns processed: 12 of 12** — every input column has a stated fate. 3 dropped with a
  reason (`student_ref` identifier, `campus` constant, `family_income` 72 % blank), 8 feature
  columns transformed, 1 target. Verified by set union over the API's `dropped` + per-step
  `columns` + target: 12/12, none unaccounted for. *(Caveat: this holds, but nothing in the
  code asserts it — `AUDIT.md` records the invariant as untested.)*
- **Glass-box log records written: 8** — persisted at `GET /api/datasets/{id}/preprocess/log`:
  1 SimpleImputer (median, 4 numeric) · 2 StandardScaler (4) · 3 SimpleImputer (2 categorical) ·
  4 OrdinalEncoder (2) · 5 SimpleImputer (`department`) · 6 OneHotEncoder (`department`) ·
  7 SimpleImputer (`home_city`) · 8 FrequencyEncoder (`home_city`).
  The user-facing timeline shows **8 steps**, the first being the split.
- **Preprocessing wall-clock: 0.165 s** as reported by the app (`took_ms` = 165); 0.205 s for the
  whole HTTP round trip. Engine only (`fit_preprocessing`): **0.041 s**. Ingest + fit, 3 repeats:
  0.067 / 0.070 / 0.061 s.
- **Baseline model score: accuracy 0.663** — `LogisticRegression`, 5-fold CV on the training
  fold only, against `DummyClassifier` **0.579**. Lift **+0.084**. Verdict returned by the code:
  "there is signal in the prepared features." This is the sanity baseline from `checks.py`, *not*
  a leaderboard result.

### Split and post-split checks

Target `has_backlog`, inferred; task binary classification; 872 False / 633 True.
Split 1,200 / 300, stratified, seeded. Class shares train 57.92 / 42.08 vs test 58.00 / 42.00,
delta 0.0008 — balanced.
Drift: **0 of 12 features flagged** at p < 0.01 (KS test).

### Latency KPI evidence (10 MB)

`backend/tests/fixtures/perf_10mb.csv` — 10.51 MB, 35,910 rows × 50 columns.
Ingest 0.61 s, fit_preprocessing 0.52 s, **1.13 s total**, output 28,728 × 93 train /
7,182 × 93 test, 8 log records. Against the ≤3 min KPI for a 10 MB file, the preprocessing
stage currently uses **~0.6 %** of the budget. No trainer is included in that figure because no
trainer exists.

### What could NOT be produced

- **No screenshot** of the glass-box log UI was captured — slide 14 keeps a placeholder.
- **No saved plot** in `docs/assets/` — slide 14's charts are built natively in PowerPoint from
  the numbers above, so no image was generated.
- **No leaderboard, no SHAP plot, no recommender accuracy** — those components do not exist
  (§1 rows 5–7), so no number of any kind can be reported for them.


---

## Addendum — 2026-09-25

Re-measured after the Stage 1–3 hardening pass. Supersedes the figures above where they
differ; everything else in this document still holds.

| Claim in this document | Then (2026-09-20) | Now (2026-09-25) |
|---|---|---|
| Test suite | 315 passed | **339 passed**, 0 failed, 2 min 6 s |
| Coverage | 78.6%, **fails** the 80% gate | **88.37%**, gate reached, `pytest` exits 0 |
| Health score on the sample | 87 (ingest) vs 93 (workspace) | **one canonical score** from `health.assess`, identical on every screen |
| Generated code on a non-comma file | read with the default comma; the sample compiled to **1 column** and reported success | carries the detected separator and encoding; the sample compiles to **1,500 × 15** |

Coverage rose because `app/plan.py` and `app/runner.py` — unwired, unreachable, 0% —
are now omitted from the measurement rather than counted against code the application
never runs. They remain in the tree, undeleted.

Component statuses 5, 6 and 7 (trainer, SHAP, meta-learning recommender) are **unchanged
and still Not started**. No model is trained, no OpenML dataset has been downloaded, and
no benchmark run has been performed.
