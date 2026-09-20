# ML-BenchX — Phase 1 audit

Audited against [docs/PHASE1_SPEC.md](docs/PHASE1_SPEC.md) on 2026-09-20.
An earlier audit against a narrower spec is archived in
[docs/AUDIT_PREVIOUS.md](docs/AUDIT_PREVIOUS.md).

Baseline: 252 pytest tests pass, frontend lints and builds, migrations run from empty.

**Headline.** The engine underneath is strong — the leakage guarantee is *stricter* than
the spec asks and is genuinely tested. But the spec describes a
**propose → review → execute** product, and what exists is **configure → run**. The two
central spec objects — a per-column `PreprocessingPlan` the user edits, and a versioned
`run` with artifacts — do not exist. That accounts for most of the ❌ below and most of
the remaining work.

Status: ✅ done · ⚠️ partial · ❌ absent.

---

## Coverage

| Area | § | | Evidence | Gap / fix |
|---|---|---|---|---|
| **Ingestion** ||||
| CSV delimiter sniffing | 1 | ✅ | `ingest.py:117` | — |
| XLSX read | 1 | ✅ | `ingest.py:453` | — |
| XLSX **sheet select** | 1 | ❌ | `ingest.py:453` | `read_excel` takes no `sheet_name`; only sheet 1 reachable |
| Encoding fallback | 1 | ✅ | `ingest.py:102` | — |
| 50 MB cap, enforced while streaming | 1 | ✅ | `main.py:163`, `:135` | — |
| Reject empty / single-column | 1 | ✅ | `ingest.py:500,524` | — |
| Reject **headerless** | 1 | ❌ | — | A data-first row is silently promoted to headers |
| Column sanitisation | 1 | ✅ | `ingest.py:240` | — |
| **original→clean mapping** | 1 | ❌ | `ingest.py:240` | Returns a count; mapping discarded |
| Numeric `1,234` `$5` `12%` | 1 | ⚠️ | `ingest.py:150` | Works, but `12%`→`12` not `0.12`, undocumented |
| Booleans `true/false` | 1 | ✅ | pandas native | — |
| Booleans **`yes/no`, `y/n`** | 1 | ❌ | — | Stays `object`, later ordinal-encoded |
| Datetime ≥90% parse rate | 1 | ⚠️ | `ingest.py:202,226,232` | Gate is **100%**; 95%-valid column stays text |
| Free text flag + exclude | 1 | ⚠️ | `glassbox.py:184`, `:606` | Threshold `rate>0.5 AND n>50`, not spec's ratio |
| ID-like >0.95 | 1 | ⚠️ | `screening.py:74`, `config.py:126` | Threshold **0.98** |
| Constant / **near-constant ≥99%** | 1 | ⚠️ | `glassbox.py:175` | Exact constants only |
| **User type override** | 1 | ❌ | `schemas.py:51`, `pipeline.py:129` | Only global booleans; no per-column field |
| **Profiling** ||||
| Type, missing %, unique | 2 | ✅ | `profile.py:96` | — |
| **Top values** per column | 2 | ❌ | — | Only for the target |
| **Numeric mean/std/min/max** | 2 | ❌ | — | Absent from the profile payload |
| Skew | 2 | ⚠️ | `health.py:307` | Dataset-level only; threshold 2.0 not 1.0 |
| IQR outlier % | 2 | ✅ | `profile.py:38` | — |
| **Histogram bins** | 2 | ❌ | — | No backend source for the §9 histogram |
| Duplicate rows | 2 | ✅ | `profile.py:150` | — |
| High-corr pairs >0.9 | 2 | ✅ | `screening.py:186` | Computed on full frame, not train |
| Target distribution | 2 | ✅ | `target.py:241` | — |
| **Imbalance ratio always reported** | 2 | ⚠️ | `target.py:220` | Returns `None` unless ratio ≥3, so balanced targets report nothing |
| Health score + breakdown | 2 | ✅ | `ingest.py:273`, `health.py:443` | Two parallel scoring systems — worth unifying |
| Warnings `{severity, message, suggestion}` | 2 | ⚠️ | `ingest.py:66`, `health.py:479` | No `suggestion`; severity is `info\|warning`, spec wants `info\|warn\|critical` |
| **Task detection** ||||
| Classification/regression rule | 3 | ✅ | `target.py:173` | — |
| **User override of task** | 3 | ❌ | `main.py:425` | Target overridable, task is not |
| Missing-target rows dropped + logged | 3 | ✅ | `main.py:410` | Logged to `cleaning`, not the step log |
| Class <2 → **warn** + non-stratified | 3 | ⚠️ | `glassbox.py:1297` | Falls back **silently** |
| **Planner / Executor** ||||
| **Per-column plan + reasons** | 4 | ⚠️ | `glassbox.py:422,546` | Actions are per *group*, one shared reason per step |
| **Plan editable per column** | 4 | ❌ | `main.py:428` | `excluded=` exists but no route accepts it |
| Dedupe pre-split | 4a | ⚠️ | `codegen.py:211` | Codegen path only; glass-box path **counts but never removes** |
| Datetime y/m/dow/hour + sin/cos | 4a | ✅ | `glassbox.py:336` | — |
| Split 80/20 stratified, seeded | 4b | ✅ | `glassbox.py:1295`, `config.py:173` | — |
| **Decisions on TRAIN only** | 4c | ✅ | `glassbox.py:1247`; `test_leakage.py` | Stricter than spec — profiling is train-only too |
| Drop if missing >60% | 4c | ✅ | `glassbox.py:600` | — |
| Impute median / mode | 4c | ✅ | `glassbox.py:630` | — |
| **Missing-indicator when >5%** | 4c | ❌ | — | — |
| **Rare categories <1% → "Other"** | 4c | ❌ | — | — |
| Binary→0/1, ≤10→one-hot ignore | 4c | ✅ | `glassbox.py:697,699` | — |
| >10 → cross-fitted TargetEncoder | 4c | ⚠️ | `config.py:186` | Available; **default is frequency** |
| Scaling Standard/MinMax/Robust | 4c | ✅ | `glassbox.py:471` | — |
| Scaling **None** | 4c | ⚠️ | `schemas.py:38` | No `"none"` value in the glass-box path |
| Outliers **report-only default** | 4c | ⚠️ | `pipeline.py:187` | Default is `cap` — inverted |
| IQR clipping, bounds from train | 4c | ⚠️ | `codegen.py:347` | Codegen path only |
| **Skew >1 suggestion + Yeo-Johnson** | 4c | ❌ | — | — |
| Leakage \|r\|>0.99 **detected** | 4c | ✅ | `screening.py:134` | Runs on train fold only |
| **Leakage by depth-2 tree CV ≥0.98** | 4c | ❌ | — | — |
| **Leakage excluded by default + critical** | 4c | ❌ | `screening.py:183` | Returns `({}, flags)` — flagged columns stay features |
| **KS test train vs test** | 4d | ❌ | — | scipy is installed |
| **Class balance in both splits** | 4d | ❌ | `main.py:450` | Whole frame only |
| **Sanity baseline (Dummy vs LogReg/Ridge)** | 4e | ❌ | — | — |
| **Glass-box log** ||||
| Record shape | 5 | ⚠️ | `glassbox.py:1126` | Missing `seq`, `severity`, `overridden`; `run_id` added only at persist |
| JSON-serialisable | 5 | ✅ | tested | — |
| **Every column has a final fate** | 5 | ⚠️ | holds (12/12 probed) | **Never asserted** — no runtime check, no test |
| **Runs & artifacts** ||||
| Versioned runs | 6 | ⚠️ | `main.py:461`, `db.py:150` | Vacuous — no overrides exist to version; no parent-run column |
| Pipeline joblib | 6 | ✅ | `main.py:552` | — |
| **`plan.json` / `schema.json` / `log.json`** | 6 | ❌ | — | `schema.json` is the Phase 2 inference contract |
| **train/test parquet** | 6 | ❌ | `main.py:560` | Train as CSV; **test never persisted** |
| **Determinism hash** | 6 | ⚠️ | verified identical by probe | No hash computed, stored or asserted |
| **API** ||||
| `POST` / `GET /datasets` | 7 | ✅ | `main.py:634,682` | — |
| `GET /datasets/{id}` | 7 | ✅ | `main.py:696` | — |
| **`DELETE /datasets/{id}`** | 7 | ❌ | — | CORS also allows only GET/POST (`main.py:130`) |
| `GET /datasets/{id}/profile` | 7 | ✅ | `main.py:707` | — |
| **`GET /datasets/{id}/preview`** | 7 | ❌ | `workspace.py:559` | Only workspace-scoped |
| **`POST /datasets/{id}/plan`** | 7 | ❌ | — | No propose-without-execute route |
| **`POST /datasets/{id}/runs`** | 7 | ❌ | `main.py:735` | Exists as `/preprocess`, takes no plan |
| `GET /runs/{id}` | 7 | ⚠️ | `main.py:784` | Returns the old preprocess payload |
| **`GET /runs/{id}/log`** | 7 | ❌ | `main.py:758` | Dataset-scoped instead |
| **`GET /runs/{id}/preview`** | 7 | ❌ | — | — |
| **`GET /runs/{id}/artifacts/{name}`** | 7 | ❌ | `main.py:867,902` | Two fixed routes; joblib not downloadable |
| **Uniform `{code, message, detail}`** | 7 | ❌ | `schemas.py:41` | Shape is `{"error": "…"}` |
| Upload extension check | 7 | ✅ | `ingest.py:447` | — |
| **Upload content check** | 7 | ❌ | — | No MIME or magic-byte sniff |
| Path traversal blocked | 7 | ✅ | `storage.py:41` | — |
| **DB** ||||
| SQLAlchemy 2.x + Alembic + SQLite | 8 | ✅ | `db.py`, `migrations/` | — |
| **`projects`** | 8 | ❌ | `db.py:93` | `project_id` is a bare string, no table or FK |
| `datasets`, `profiles`, `preprocessing_runs` | 8 | ✅ | `db.py:101,129,150` | — |
| **`dataset_columns`** | 8 | ❌ | `db.py:145` | Per-column data lives in `profiles.payload` JSON |
| **`preprocessing_steps`** | 8 | ⚠️ | `db.py:181` | Named `preprocessing_logs` |
| `project_id` + `created_at` on all | 8 | ✅ | `db.py:90` | Verified on all 7 tables |
| `StorageBackend` + `LocalStorage` | 8 | ✅ | `storage.py:52,81` | — |
| **UI** ||||
| Upload drag-drop / progress / errors | 9 | ✅ | `UploadScreen.tsx:141,113,184` | — |
| Dataset summary cards | 9 | ⚠️ | `Workspace.tsx:425` | Header strip; size never shown |
| Health gauge + breakdown | 9 | ✅ | `InsightsPanel.tsx:85,181` | — |
| Warnings with severity | 9 | ⚠️ | `InsightsPanel.tsx:238` | Two visual levels, no suggestion |
| Column table | 9 | ⚠️ | `BottomDock.tsx:167` | No unique count, no flags |
| **Type override dropdown** | 9 | ❌ | — | — |
| **Histogram on column click** | 9 | ❌ | `BottomDock.tsx:178` | Click sorts; `recharts` installed and unused |
| **Plan review screen** | 9 | ❌ | `App.tsx:16` | Routes: landing / upload / workspace |
| **Target picker** | 9 | ❌ | `api.ts:35` | `config.target` never settable |
| **Detected task shown** | 9 | ❌ | — | — |
| Per-column actions + reasons | 9 | ⚠️ | `PipelinePanel.tsx:604` | Per-step |
| **Per-column override dropdowns** | 9 | ❌ | — | — |
| Scaler / outlier controls | 9 | ✅ | `PipelinePanel.tsx:274,135` | — |
| **Missing-indicator toggle** | 9 | ❌ | — | — |
| Timeline **grouped by stage** | 9 | ⚠️ | `InsightsPanel.tsx:347` | Flat numbered list |
| **Severity colours in timeline** | 9 | ❌ | `api.ts:190` | Log carries no severity |
| before→after per step | 9 | ⚠️ | `InsightsPanel.tsx:360` | Absolute, no delta |
| Before/after data preview | 9 | ✅ | `BottomDock.tsx:90` | — |
| **Split summary shown** | 9 | ❌ | `api.ts:219` | Data arrives, never rendered |
| **Baseline scores** | 9 | ❌ | — | — |
| Artifact downloads | 9 | ✅ | `Workspace.tsx:53` | — |
| Runs history per dataset | 9 | ⚠️ | `BottomDock.tsx:355` | Edit trail, not selectable runs |
| Loading / error / empty states | 9 | ✅ | throughout | — |
| **Config** ||||
| One config module | 10 | ✅ | `config.py` | — |
| **Verification** ||||
| **Fixture generator + 17 fixtures** | V | ❌ | only `titanic.csv` | — |
| Leakage-invariance test | V | ✅ | `test_leakage.py` | — |
| **Determinism / round-trip / log-invariant tests** | V | ❌ | — | — |
| Output numeric, no NaN | V | ✅ | `test_codegen.py` | — |
| API contract tests | V | ✅ | `test_api.py`, `test_workspace.py` | Need updating for new routes |
| **Coverage ≥80%** | V | ❌ | no `pytest-cov`, no config | — |
| **Playwright smoke** | V | ❌ | not installed | — |
| Frontend typecheck script | V | ⚠️ | `package.json:7` | Only inside `build` |
| **Pinned dependencies** | V | ❌ | — | **No Python manifest at all**; npm all `^` |
| Type hints (app) | V | ⚠️ | `main.py:100,136,225,230,235` | 5 functions unannotated; rest 100% |
| Type hints (tests) | V | ❌ | — | Largely unannotated |
| **Performance on 10 MB** | V | ❌ | — | — |

---

## Bugs

1. **Leakage columns are flagged but never excluded** (`screening.py:183`). Spec §4c
   says exclude by default, critical warning, user re-includes. A column that is a copy
   of the target currently enters the feature matrix with an advisory note. Most
   consequential gap in the audit.
2. **Class <2 samples falls back to non-stratified silently** (`glassbox.py:1297`). No
   warning record, nothing in the payload.
3. **Duplicates are counted but never removed** in the glass-box path
   (`ingest.py:560`). Spec §4a requires dedupe pre-split; only the codegen path does it.
4. **Headerless files are accepted**, first data row silently becoming column names.
5. **`12%` → `12`**, not `0.12` (`ingest.py:150`). Silent change of quantity.
6. **Datetime coercion needs a 100% parse rate** (`ingest.py:232`), spec says ≥90%.
7. **Near-constant columns are kept** (`glassbox.py:175`) — only exact constants drop.
8. **Imbalance ratio is withheld for balanced targets** (`target.py:220`), so the UI
   cannot show the ratio the spec asks for.
9. **Skew threshold is 2.0** (`health.py:314`), spec says 1.0.
10. **`GET /runs/{id}` returns the old preprocess payload** — same URL, different object
    from the run the spec describes.

## Leakage risks

1. **Already closed, and stricter than required.** The split precedes profiling *and*
   screening, not just fitting (`glassbox.py:1247`). `test_leakage.py` proves it by
   rewriting held-out rows and asserting a byte-identical plan.
2. **No train/test drift check** (§4d) — badly shifted splits pass unnoticed.
3. **Leakage detection exists but takes no action** — see Bug 1.

## Security

1. **Content is never checked, only the extension.** A renamed binary reaches the
   parser.
2. **No `DELETE /datasets/{id}`** — uploads accumulate with no API to remove them.
3. Not issues: path traversal is closed (`storage.py:41`); the code executor is
   sandboxed and adversarially tested (`test_executor.py`).

## Dead code and drift

1. **`recharts` and `framer-motion` installed, imported by zero files.** Histograms need
   a chart library — use one or drop both.
2. **`/api/analyse` and `/api/example` have no frontend caller.** Two API surfaces; the
   product uses one.
3. **`api.compilations()` defined and never called** (`frontend/src/api.ts:416`).
4. **`eslint-disable` comment with no ESLint installed** (`BottomDock.tsx:84`).
5. **Two parallel health scores** — `ingest._score_quality` and `health.assess`.
6. **`workspaces`/`compilations`/`history` vs `preprocessing_runs`/`preprocessing_steps`**
   — two overlapping models of the same idea.

---

## The two structural gaps

**1. There is no plan object.** `PipelineConfig` is per-*step* ("impute numerics with
the median"). The spec wants per-*column* ("`Age` → impute median, because 20% missing,
skew 0.4"), proposed by a planner, shown to the user, editable column by column, handed
to an executor. That changes the data model, adds `POST /datasets/{id}/plan`, and adds a
plan review screen.

**2. There is no run resource.** Execution produces a `compilation` tied to a
`workspace`. The spec wants a versioned `run` at `/runs/{id}` carrying `plan.json`,
`schema.json`, `log.json`, the joblib pipeline and train/test parquet — `schema.json`
specifically so Phase 2 can run inference.

Neither is a rewrite. The engine, the leakage guarantee, the sandbox, the health
dimensions and the storage layer all survive; this is a new layer over them plus an API
and UI reshaped around it.
