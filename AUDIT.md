# ML-BenchX — Phase 1 audit

Audited 2026-09-19 against the Phase 1 spec. Baseline: 49 pytest tests pass,
`npm run build` succeeds, no git repository initialised.

Legend: ✅ present and correct · ⚠️ present but incomplete/deviating · ❌ absent

## Spec coverage

| Area | Spec item | | Where | Fix |
|---|---|---|---|---|
| Ingest | CSV/XLSX read | ✅ | `backend/app/services/ingest.py:278` | — |
| Ingest | Encoding check | ✅ | `ingest.py:97` | — |
| Ingest | Size cap, configurable, default 50 MB | ⚠️ | `backend/app/main.py:44` | 25 MB hardcoded; move to config, default 50 MB |
| Ingest | Cap enforced before body is buffered | ❌ | `main.py:261` | `await file.read()` buffers the whole upload, *then* checks size. Stream with a running cap |
| Ingest | Infer numeric/categorical/datetime/ID-like/constant | ⚠️ | `glassbox.py:146` + `screening.py:55` | All five kinds exist but ID-like lives in a second module and never reaches `ColumnProfile`; datetime is detected then dropped |
| Profile | Per-column missing %, cardinality | ⚠️ | `glassbox.py:160` | Computed in `ColumnProfile` but never exposed by the API or UI |
| Profile | IQR outliers per column | ❌ | — | Not implemented |
| Profile | Duplicate rows | ✅ | `ingest.py:381` | — |
| Profile | High-corr pairs \|r\|>0.9 | ❌ | — | Not implemented |
| Profile | Class imbalance | ⚠️ | `target.py:177` | Value counts returned; no imbalance flag or threshold |
| Profile | Health score 0–100, documented weighted formula | ⚠️ | `ingest.py:214` | Score exists; weights are literals inside the function and documented nowhere |
| Task | binary / multiclass / regression | ⚠️ | `target.py:165` | **Bug**: a text target with 21–50 distinct values returns `"regression"` |
| Prep | Split 80/20 first, stratified | ✅ | `glassbox.py:857` | — |
| Prep | Fit on train only, transform both | ✅ | `glassbox.py:869` | Single `fit_transform`, proven by `test_glassbox.py:199` |
| Prep | Routing decided from train only | ⚠️ | `glassbox.py:846` | `profile_frame` and `find_identifier_columns` read the **full** frame, so test-row cardinality can flip a routing decision |
| Prep | sklearn Pipeline / ColumnTransformer | ✅ | `glassbox.py:475` | — |
| Prep | Persist fitted pipeline (joblib) | ❌ | — | Pipeline lives only in an in-process dict |
| Prep | Impute median (num) / mode (cat) | ✅ | `glassbox.py:452` | — |
| Prep | Binary → label encode | ✅ | `glassbox.py:465` | — |
| Prep | Low-card (≤10, configurable) → one-hot, `handle_unknown="ignore"` | ⚠️ | `glassbox.py:105` | Correct encoder; threshold is 15, hardcoded |
| Prep | High-card → cross-fitted TargetEncoder | ⚠️ | `glassbox.py:190` | Deliberately replaced by `FrequencyEncoder`; documented, but the spec's encoder is absent and not selectable |
| Prep | Scale: Standard default, MinMax/Robust selectable | ⚠️ | `glassbox.py:462` | Standard hardcoded; no option plumbed from the API |
| Prep | Datetime → y/m/dow/hour + sin/cos | ❌ | `glassbox.py:389` | Datetime columns are dropped with a reason |
| Prep | Leakage flag \|r\|>0.99 with target | ❌ | — | Not implemented. `screening.py` catches identifiers only |
| Prep | Drop constant + ID-like with logged reason | ✅ | `glassbox.py:383`, `screening.py:84` | — |
| Log | JSON record per step | ⚠️ | `glassbox.py:734` | Serialisable and derived from fitted objects (good), but shape is `{step, stage, columns, transformer, params, fitted_on, reason, details, explanation}` — no `action`, no `before`/`after` |
| API | `POST /datasets` | ❌ | — | Only `POST /api/analyse` (one-shot, nothing persisted) |
| API | `GET /datasets/{id}` | ❌ | — | — |
| API | `GET /datasets/{id}/profile` | ❌ | — | — |
| API | `POST /datasets/{id}/preprocess` (target, scaler, options) | ❌ | — | No scaler/option parameters anywhere |
| API | `GET /datasets/{id}/preprocess/log` | ❌ | — | — |
| API | Pydantic schemas | ❌ | `main.py:254` | Every route returns bare `dict[str, Any]`; no response models, no OpenAPI shape |
| API | Correct 4xx errors | ✅ | `main.py:105` | 400 for user errors, 404 for missing runs, handlers tested |
| DB | SQLAlchemy 2.x + Alembic on SQLite | ❌ | `main.py:82` | In-memory `OrderedDict`, capped at 20 runs |
| DB | `datasets`, `profiles`, `preprocessing_runs`, `preprocessing_logs` | ❌ | — | — |
| DB | `project_id` + `created_at` on every table | ❌ | — | — |
| Storage | `StorageBackend` interface + `LocalStorage` | ❌ | — | Uploaded bytes are never written to disk |
| UI | Upload page | ✅ | `frontend/src/components/Upload.tsx` | — |
| UI | Profile dashboard: health score | ✅ | `Results.tsx:159` | — |
| UI | Profile dashboard: per-column flag table | ❌ | `Results.tsx:109` | Only a list of column names; no per-column missing %, cardinality, outliers or flags |
| UI | Glass-box log, card per step, reason + before→after | ✅ | `StepCard.tsx`, `TransformStrip.tsx` | The strongest part of the build |
| Config | All thresholds in one module | ❌ | — | Spread over `glassbox.py:104-125`, `ingest.py:39`, `screening.py:24`, `main.py:44-51`, `explain.py:35` |
| Tests | One per preprocessing rule | ✅ | `tests/test_glassbox.py` | 24 tests, rule-by-rule |
| Tests | Leakage: test-split stats never reach fit | ✅ | `test_glassbox.py:199`, `:227` | Fit-spy on row indices + train-vs-global median check. Best test in the suite |
| Tests | API happy path + bad file | ✅ | `test_app.py:172`, `:242` | — |
| Tests | `tests/fixtures` dataset | ❌ | — | Tests build frames inline; no fixture file |

## Bugs

1. **`describe_task` mislabels categorical targets** — `target.py:165`. A text target
   with 21–50 distinct values falls through to `"regression"` (verified: 30 string
   categories → `regression`). It then reaches `fit_preprocessing` as
   `task_type="regression"`, which **disables stratification** on what is really a
   multiclass problem, and the UI shows the wrong task.
2. **Upload size cap applied after buffering** — `main.py:261`. See leakage/security below.
3. **`Content-Disposition` built from the raw client filename** — `main.py:300`.
   `Path(name).stem` strips directories but keeps `"` and control characters.
4. **`_RUNS` holds 20 full DataFrames plus fitted pipelines** — `main.py:48`. With the
   cap at its configured maximum this is an unbounded-in-practice memory ceiling, and
   `MAX_RUNS` evicts by insertion order, not by age or size.
5. **Exported loader drifts from the real ingest** — `ingest.py:441`. The generated
   `load_frame()` strips header whitespace but does not reproduce the unnamed-column
   renaming or duplicate-header disambiguation `_tidy_headers` performs, so an export
   from a file with duplicate headers will not reproduce the analysed frame.

## Leakage risks

1. **Routing decisions read the full frame** — `glassbox.py:846`. `profile_frame(X)` and
   `find_identifier_columns(frame)` run before the split. Cardinality, missing rate and
   uniqueness therefore include test rows, so which columns are dropped and whether a
   column is one-hot or frequency encoded is a function of held-out data. No *fitted
   parameter* leaks — the module's central claim holds — but the spec's ordering
   ("split FIRST → fit on train only") is violated at the decision layer. This is the
   single most important fix in the audit.
2. **No target-correlation leakage screen** — the spec's `|r|>0.99` flag is absent, so a
   column that is a copy of the target sails through as an ordinary feature.
3. Not a risk: `FrequencyEncoder` never reads `y`, and the split-before-fit invariant is
   genuinely enforced and genuinely tested.

## Upload security

1. **Unbounded read before the cap** — `main.py:261`. `await file.read()` materialises
   the entire upload in memory, and Starlette has already spooled it to a temp file
   with no limit. A multi-GB POST exhausts RAM or disk before the 25 MB check runs.
2. **Header injection via filename** — `main.py:300`, as above.
3. **No declared content-type check** — extension is validated inside `ingest.py:292`,
   which is the right gate, but it happens after the bytes are read.
4. **XLSX parsed with default openpyxl settings** — `ingest.py:284`. Low risk (openpyxl
   does not resolve external entities), but a decompression-bomb `.xlsx` is unbounded.
5. Not an issue: CORS is pinned to localhost origins (`main.py:59`), and there is no
   path taken from user input to the filesystem.

## Dead code

- `backend/dropout_pipeline.py` — a generated export committed by accident. Hardcodes
  `/Users/rohit/Downloads/student_dropout_dataset_v3.csv`. Referenced by nothing.
- `backend/sample_data/student_dropout_sample.csv` — referenced by nothing; the live
  sample is `backend/app/sample/campus_records.csv`.
- `frontend/src/assets/hero.png`, `react.svg`, `vite.svg` — referenced by nothing.
- `glassbox._main` CLI (`glassbox.py:1136`) — not dead, but undocumented in the README.

## Status

Phase 1 is roughly **60% complete by spec item**, and the 60% that exists is the hard
60%: the split-before-fit invariant, the derived-not-narrated log, and the explanation
layer are all well built and well tested. What is missing is the persistence half of
the spec (DB, storage, pipeline persistence, resource-shaped API) and four profiling
signals (outliers, correlation pairs, target-leakage flag, imbalance), plus the
configurability the spec asks for (scaler choice, thresholds in one module).
