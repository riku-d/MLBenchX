# ML-BenchX — engineering audit

**Audited 30 October 2026, against the working tree at `5a2328e` + uncommitted changes.**
**Revised 30 October 2026** after the remediation round below (§8).

Scope: Stages 1–3 (ingestion, profiling, glass-box preprocessing). Stages 4–7 are
**PLANNED** and were not audited beyond confirming that nothing claims to implement
them.

| Document | Answers |
|---|---|
| [PRD.md](PRD.md) | What the product must do, and what counts as done |
| [Design.md](Design.md) | How it is built, and why it is built that way |
| [AUDIT.md](AUDIT.md) | What is actually true today — verification evidence, defects, remediation |

This document records *status and evidence*. The requirements it judges against are
[PRD.md](PRD.md) §3–4; the reasoning behind the designs it judges is
[Design.md](Design.md), and is referenced rather than restated.

Two sources this audit was originally written against have since been removed from the
repository: the Phase 1 specification (`docs/PHASE1_SPEC.md`) and the archive of an
earlier, narrower audit (`docs/AUDIT_PREVIOUS.md`). Their substance survives — the
specification's requirements are restated in PRD.md §3–4, and the earlier audit's
findings were carried into §7 below. References to "the Phase 1 specification" name that
retired source, not a file in this repository.

This is a living document, not a changelog. The prior revision of it audited an earlier
tree; its §4 coverage table has been superseded because most of its ❌ rows were closed
by the four slices committed since (ingestion repairs, profiling, health, leakage).
Its two structural findings survive verbatim in §7 below, because they are still true.

**Labels.** PASS — verified by inspection *and* an executed test · PARTIAL —
implemented with a stated gap · FAIL — required behaviour broken or missing ·
NOT VERIFIED — insufficient evidence · PLANNED — deliberately out of this release.

---

## 1. Audit summary

| | Status | Basis |
|---|---|---|
| **Overall** | **PASS** | 506 tests pass; 89.8% coverage; frontend builds and lints; migrations run from empty |
| **Stage 1 — Ingestion** | **PASS** | 21 requirements, all met. One P0 defect found and fixed (infinities) |
| **Stage 2 — Profiling** | **PASS** | 18 requirements, all met. One P0 defect found and fixed (unhandled 500 on non-finite values) |
| **Stage 3 — Preprocessing** | **PASS** | 32 requirements met. Four defects found and fixed; two correctness gaps closed in the generated-script path |
| **Critical blockers** | **None** | Three P0 defects were found (two crash/leak, one breaking the UI's two central interactions). All three are fixed, each with a regression test that fails against the prior code |
| **Important warnings** | **Three** | §1.2 |
| **Test status** | **506 passed, 0 failed, 0 skipped** | `pytest tests/ -q`, run 30 Oct 2026 (was 381 before §8) |
| **Build status** | **PASS** | `tsc -b && vite build` ✅ · `oxlint` exit 0 · `alembic upgrade head` from empty ✅ |

### 1.1 What this audit changed

Seven defects were found by probing rather than by reading, fixed, and pinned by tests
that fail against the previous implementation. Full detail in §5.7.

| | Defect | Severity | Stage |
|---|---|---|---|
| D1 | A column containing `inf` took `POST /api/datasets` down with an unhandled `ValueError` out of `numpy.histogram`, returning a full traceback | **P0** | 2 |
| D2 | The same values produced `Infinity` tokens in the profile JSON, which no strict parser reads back | **P0** | 2 |
| D3 | A stratified split was attempted whenever every class had ≥2 rows, ignoring sklearn's further requirement that each fold hold at least one row per class. A 60-row, 20-class file aborted the run with a message about a `test_size` the user never chose | **P1** | 3 |
| D4 | `fit_preprocessing` did not remove rows with a blank target. Classification aborted inside sklearn; **regression said nothing and carried the blank into `y_train`** | **P1** | 3 |
| D5 | The generated script decided which columns were "too empty to keep" from the **whole file**, and which calendar parts to emit from the **whole file** — decisions taken with the held-out rows in view | **P1** | 3 |
| D6 | `PreprocessingResult.warnings` and `.checks` were computed on every run and read by nothing. Drift, split balance, the sanity baseline and every stratification fallback were discarded | **P1** | 3 |
| D7 | CORS allowed `GET` and `POST` only, while the interface saves the pipeline configuration and the edited code with `PUT`. The browser's preflight was answered `400 Disallowed CORS method`, so **both central interactions failed in the real application** and in no test | **P0** | — |

### 1.2 Important warnings — all three now closed

These were the three open warnings of the previous revision. Each is resolved in §8:

1. ~~**The UI does not render what the engine detects.**~~ **Closed.** A plan review
   screen renders the warnings, drift, fold balance, stratification fallback and sanity
   baseline from a real run. No second execution: the run produces them natively.
2. ~~**Stage 3 is implemented twice.**~~ **Closed as a divergence risk.** Both paths now
   consume one canonical representation — the plan — and the generated script carries
   the engine's own transformer definitions, copied with `inspect.getsource`. 39 parity
   tests compare the two on an output-hash digest, action by action.
3. ~~**No asserted performance budget.**~~ **Closed.** A machine-calibrated budget over
   the 10 MB fixture, with a scaling check that needs no reference machine.

---

## 2. Repository and dependency audit

### 2.1 Structure and entry points

| Entry point | Status | Note |
|---|---|---|
| `backend/app/main.py:app` | PASS | FastAPI application; dataset routes + the single-call endpoints; mounts the workspace router |
| `frontend/src/main.tsx` | PASS | Vite + React 19 |
| `backend/migrations/` | PASS | Two Alembic revisions; `upgrade head` from empty verified |
| `backend/tests/fixtures/generate.py` | PASS | Deterministic fixture catalogue, seeded per fixture name |

### 2.2 Dependencies

| Check | Status | Evidence |
|---|---|---|
| Python manifest exists and is pinned | PASS | `requirements.txt` pins every runtime dependency exactly, with the reasoning stated in the file: numerical behaviour is part of the correctness claim, so a minor release of pandas or scikit-learn can falsify determinism without any code change |
| Dev dependencies pinned | PASS | `requirements-dev.txt` — `pytest==9.1.1`, `pytest-cov==7.1.0`, `httpx==0.28.1` |
| npm dependencies pinned | PASS | `package.json` carries exact versions, no `^` ranges |
| Every declared dependency is used | PARTIAL | `recharts` (3.10.1) and `framer-motion` (13.4.0) are installed and imported by zero files. Histograms are computed server-side and have no chart component to render them |
| Broken imports / unresolved references | PASS | Full import sweep clean; `python -c "import app.main"` succeeds |
| No destructive upgrades performed | PASS | No dependency version changed in this audit |

### 2.3 Configuration

| Check | Status | Evidence |
|---|---|---|
| One configuration module | PASS | `app/config.py`, 375 lines, every threshold with the reasoning for its value |
| Environment overrides | PASS | `MLBENCHX_`-prefixed; `_int`/`_float`/`_str` readers |
| No threshold duplicated in code | PASS **after fix** | `LEAKAGE_TREE_DEPTH` existed in config while `screening.py` hard-coded `max_depth=2`. Now reads the constant — the project's own convention is that a number appearing twice will eventually disagree with itself |
| Test isolation from developer state | PASS | `conftest.py` redirects `MLBENCHX_DATABASE_URL` and `MLBENCHX_STORAGE_ROOT` into a temporary directory **before** `app` is imported, and removes it at session end |

### 2.4 Dead code and duplication

| Finding | Status | Detail |
|---|---|---|
| `app/plan.py` (507 lines) and `app/runner.py` (666 lines) are unreachable | PARTIAL | `runner` imports `plan`; nothing in `backend/` imports `runner`. No route, service or test reaches either. `.coveragerc` omits both and documents why. **They are the only implementation of two spec features** — missing-value indicator columns (`MISSING_INDICATOR_RATE`) and rare-category grouping (`RARE_CATEGORY_RATE`) — so a reader of `config.py` will believe those features exist. Wire in or delete; both are direction decisions, not defects |
| Two preprocessing implementations | PARTIAL | §7 |
| Two health scores | PARTIAL | `ingest._score_quality` and `health.assess` both return 0–100 by different arithmetic, for different consumers |
| `/api/analyse` and `/api/example` have no frontend caller | PARTIAL | Two API surfaces; the product uses one |
| `api.compilations()` defined and never called | PARTIAL | `frontend/src/api.ts` |

### 2.5 Security and input validation — summary

Detail in §5.6. No blocking finding. The two items previously open — content sniffing
and traceback exposure — are both closed, the second by this audit.

---

## 3. Stage-by-stage audit

Only rows where the verdict is not a plain PASS carry a gap and an action. Every row
names the test that holds it; a row with no test says so.

### 3.1 Stage 1 — Ingestion

| Requirement | Status | Where | Evidence | Gap / action | Test |
|---|---|---|---|---|---|
| CSV/TSV, delimiter sniffing | PASS | `ingest._detect_delimiter:251` | Comma, semicolon, tab fixtures all parse | — | `test_source_format.py` |
| Excel, with sheet selection | PASS | `ingest._read_excel:695`, `list_sheets:683` | Sheet used is reported; others listed | — | `test_source_format.py` |
| JSON (records/columns/lines), Parquet | PASS | `_read_json:614`, `_read_parquet:667` | — | — | `test_source_format.py` |
| Encoding detection and fallback | PASS | `_detect_encoding:236` | `charset-normalizer`; Latin-1 fixture | — | fixture `latin1.csv` |
| 50 MB cap, enforced while streaming | PASS | `main.py:135,175` | Refused before the body is buffered | — | `test_api.py` |
| **Content checked, not just the extension** | PASS | `ingest._check_content:96` | Magic-byte sniff; a renamed PNG is refused before any parser runs | Previously FAIL; closed by an earlier slice | `test_scenarios.py::test_f_*` |
| Reject empty / header-only / single-column | PASS | `ingest:826,832` | Each has its own sentence | — | Scenario F |
| Reject below the row floor (50) | PASS | `ingest:877` | — | — | Scenario F |
| Headerless detection | PASS | `_looks_headerless:588` | Previously FAIL | — | `test_source_format.py` |
| Header tidying + original→clean mapping | PASS | `_tidy_headers:422` | Mapping retained, not just a count. Previously FAIL | — | fixture `unicode_duplicate_headers.csv` |
| Missing-token recognition | PASS | `_blank_placeholders:267` | — | — | `test_app.py` |
| Numeric coercion (`1,234`, `$5`, `12%`) | PARTIAL | `_coerce_numeric:284` | Converts only when every non-blank value converts | **`12%` → `12`, not `0.12`.** Documented in PRD.md §9 R8 rather than changed: silently rescaling a column by 100 would alter results for anyone already relying on it | `test_rules.py` |
| Datetime coercion at ≥90% | PASS | `_coerce_datetime:336`, `DATE_PARSE_RATE` | Four-digit-year anchor stops `1.2.3` parsing as a date. Previously the gate was 100% | — | `test_rules.py` |
| Boolean coercion (`yes/no`, `y/n`) | PASS | `_coerce_boolean:390` | Previously FAIL | — | `test_rules.py` |
| **Infinities treated as missing** | **PASS — fixed here** | `_blank_non_finite` | Was absent. `inf` survived ingestion into every downstream consumer | **D1/D2 root cause.** Now blanked, with a cleaning sentence, mirrored in `loader_source` and `codegen` so the exported script does the same | `test_infinity_is_treated_as_missing_at_ingestion` |
| Every repair reported as a sentence | PASS | `IngestResult.cleaning` | — | — | `test_app.py` |
| Path traversal blocked | PASS | `storage.py:41` | — | — | `test_api.py` |

### 3.2 Stage 2 — Profiling

| Requirement | Status | Where | Evidence | Gap / action | Test |
|---|---|---|---|---|---|
| Dimensions, per-column missing and cardinality | PASS | `profile.column_profiles` | — | — | `test_api.py` |
| **Numeric mean/std/min/max/quartiles/skew** | PASS | `profile.numeric_summary:107` | Previously FAIL ("absent from the payload") | — | `test_api.py` |
| **Histogram bins** | PASS | `profile.histogram:139` | Computed server-side — a histogram of one page of rows is not a histogram of the column. Previously FAIL | No frontend renders it; `recharts` is installed and unused | `test_api.py` |
| **Top values per column** | PASS | `profile.top_values:170` | For every kind, not only the target. Previously FAIL | — | `test_api.py` |
| IQR outlier count and fences | PASS | `profile.outlier_summary:69` | Fences collapse safely when IQR ≤ 0 | — | `test_rules.py` |
| Duplicate rows | PASS | `ingest`, `profile.build_profile` | — | — | `test_app.py` |
| High-correlation pairs > 0.90 | PASS | `screening.correlated_pairs` | — | Computed on the full frame — correct here, because this is profiling *for display* and never feeds a decision | `test_checks.py` |
| Class balance and imbalance ratio | PASS | `target.class_balance` | Ratio now reported for balanced targets too. Previously PARTIAL | — | `test_rules.py` |
| Health score + six-dimension breakdown | PASS | `health.assess:483`, `DIMENSION_WEIGHTS` | Arithmetic shown term by term | Two parallel scoring systems remain (§2.4) | `test_checks.py` |
| Warnings `{severity, message, suggestion}` | PASS | `ingest.QualityIssue:165` | Severity is `info\|warn\|critical`, and `suggestion` is populated. Previously PARTIAL on both counts | — | `test_checks.py` |
| Per-column quality flags | PASS | `profile._flags:187` | Deliberately excludes facts the table's own columns already state | — | `test_api.py` |
| **Non-finite values do not break profiling** | **PASS — fixed here** | `profile._finite:53` | `np.histogram` raises `"autodetected range … is not finite"`; the route had no handler, so the client received a 500 and a full traceback. `numeric_summary` separately emitted `Infinity` into the JSON | **D1/D2.** `_finite()` now guards `histogram`, `numeric_summary` and `outlier_summary` independently of the ingestion fix, because these are public functions reached with frames that did not come through ingest | `test_profiling_a_non_finite_column_does_not_raise`, `test_a_file_containing_infinity_uploads_and_preprocesses` |
| Profiling for display never feeds a decision | PASS | `profile.py` docstring; `glassbox.fit_preprocessing` | Two profiles exist, answer different questions, and neither substitutes for the other | — | `test_leakage.py` |

### 3.3 Stage 3 — Glass-box preprocessing

| Requirement | Status | Where | Evidence | Gap / action | Test |
|---|---|---|---|---|---|
| Target separated, never a feature | PASS | `fit_preprocessing` step 2 | `X = df.drop(columns=[target])` | — | `test_c_*` |
| **Blank-target rows removed, both tasks** | **PASS — fixed here** | `fit_preprocessing` step 1 | `main.py:410` dropped them on its own path, so the API was covered. The module did not, and the two tasks failed differently: classification raised `Input contains NaN` from the stratifier, regression **silently carried `NaN` into `y_train`** | **D4.** Removal moved into the module, with a `blank_targets_removed` warning. `main.py`'s own handling is unchanged and now redundant rather than load-bearing | `test_rows_with_a_blank_target_are_removed_and_reported` (parametrised over both tasks) |
| Deduplication **before** the split | PASS | `fit_preprocessing` step 1 | Copies on opposite sides of the split are scored as held-out rows the model trained on; removing them afterwards would not undo that | Previously FAIL in the glass-box path ("counts but never removes") | `test_d_duplicate_rows_are_removed_before_the_split_and_reported` |
| **Split before anything is measured** | PASS | `fit_preprocessing` step 3 | Stricter than "fit on train only": the column profile that decides which columns survive is computed from `X_train` alone | — | `test_e_rewriting_the_held_out_rows_cannot_change_the_plan` |
| Stratified classification split | PASS | `_stratification` | Each class within 2 points on both sides | — | `test_a_the_split_is_stratified_and_reproducible` |
| **Audible fallback from stratification** | **PASS — fixed here** | `_stratification` | Three causes now distinguished and named: one class, a class with one row, and **more classes than either fold has rows** — the last was unguarded and aborted the run | **D3.** Fold sizes are computed the way `sklearn.model_selection._validate_shuffle_split` computes them, so this predicts that function's verdict rather than approximating it | `test_too_many_classes_for_the_fold_falls_back_and_says_why` |
| Regression split not stratified | PASS | `_stratification` returns early | `stratified` recorded on the result | — | `test_c_*` |
| Reproducible with a fixed seed | PASS | `SEED`, `test_size` recorded on the result | Fold membership, matrices and log all identical across runs | — | `test_a_the_split_is_stratified_and_reproducible` |
| Feature/target alignment survives the split | PASS | — | Shared index, in order; folds disjoint | — | `test_a_train_and_test_are_disjoint_and_aligned` |
| Numeric imputation from train only | PASS | `_block("numeric")` | Logged fill value equals the training median and differs from the whole-file median | — | `test_b_fill_values_are_the_training_median_not_the_whole_file_median` |
| Categorical imputation from train only | PASS | `_block` | Fill is by construction a category the encoder knows | — | `test_b_categorical_gaps_are_filled_…` |
| Drop a column over 60% missing | PASS | `build_plan` | Measured on the training fold | — | `test_d_an_all_null_column_is_dropped_rather_than_filled` |
| **…in the generated script too** | **PASS — fixed here** | `codegen._MISSING_DROP_COLUMNS` | Was `df[c].isna().mean()` over the whole file, emitted **before** the split. The code comment justified the position and not the measurement | **D5.** Block moved after the split and measured on `train_index` | `test_codegen.py::test_a_column_is_dropped_on_how_empty_the_training_rows_are` |
| Encoding by cardinality | PASS | `_encoder_spec`, `_block` | Binary → ordinal; ≤10 → one-hot; above → frequency/target | — | `test_rules.py` |
| Unseen categories handled | PASS | `handle_unknown="ignore"` / `unknown_value=-1` / `fillna(0.0)` | No transform raises on a test-only category | — | `test_leakage.py` |
| Default encoder cannot leak | PASS | `DEFAULT_HIGH_CARD_ENCODER="frequency"` | Never reads `y`. `TargetEncoder` is offered and cross-fits at 5 folds | — | `test_rules.py` |
| Scaling, centre and spread from train | PASS | `_block("numeric")` | All three scalers reported through one centre/spread pair | — | `test_rules.py` |
| Calendar features | PASS | `DatetimeFeatures` | sin/cos pair per cyclical part | — | `test_rules.py` |
| **…decided from training rows, in the generated script** | **PASS — fixed here** | `codegen._DATETIME_FEATURES` | `_values.nunique() < 2` ran over the whole file, pre-split, deciding which columns exist | **D5.** Now `_values.loc[train_index].nunique()`, after the split, with a `not in df.columns` guard for the new ordering | `test_codegen.py::test_a_calendar_part_is_emitted_on_how_the_training_rows_vary` |
| Constant / near-constant / text columns dropped | PASS | `_classify`, `build_plan` | Near-constant at `NEAR_CONSTANT_SHARE` 0.99. Previously PARTIAL (exact constants only) | — | `test_d_columns_that_carry_nothing_are_dropped_with_a_reason` |
| Identifier screen, on the training fold | PASS | `screening.identifier_screen:355` | — | — | `test_checks.py` |
| **Leakage screen excludes, not just flags** | PASS | `screening.target_leakage_screen:217` | Correlation \|r\|>0.99 **and** a depth-2 tree at CV ≥0.98. Flagged columns are now removed. This was the previous audit's Bug 1 and its "most consequential gap" | — | `test_e_a_column_derived_from_the_answer_is_flagged_and_excluded` |
| Leakage stated as suspicion, not proof | PASS | `screening.py:280-298` | Evidence is described; the verdict is hedged | — | `test_e_leakage_warnings_are_phrased_as_suspicion_not_proof` |
| **Non-finite feature values** | **PASS — fixed here** | `glassbox._blank_non_finite` | Every estimator rejects `inf`; the error surfaced as "Try a file with more usable columns", which is wrong advice | Defended at the module boundary as well as at ingestion, because `fit_preprocessing` is public and is called with frames that did not come through ingest | `test_fit_preprocessing_blanks_infinity_without_touching_the_callers_frame` |
| Input data not modified | PASS | — | The frame the caller passes is byte-identical afterwards; `_blank_non_finite` mutates only the copy `drop(columns=…)` returns | — | same test |
| Fit on train, transform test | PASS | `fit_preprocessing` step 6 | The only `.fit` in the module's data path | — | `test_leakage.py` spy |
| Feature map complete | PASS | `build_feature_map` | **Raises** on an output column with no source rather than warning | — | `test_d_every_output_column_can_be_traced_back_to_an_input` |
| Derived log, never hand-written | PASS | `introspect` | Read from `statistics_`, `mean_`, `scale_`, `categories_`, `frequencies_`, `encodings_` | — | `test_glassbox.py` |
| Post-split checks (drift, balance, baseline) | PASS | `checks.run_post_split_checks`, `sanity_baseline` | KS at p<0.01; both folds' balance; CV baseline vs a dummy | — | `test_checks.py` |
| **…reach the caller** | **PASS — fixed here** | `main.py` payload, `schemas.AnalysisResponse` | `result.warnings` and `.checks` were produced on every run and consumed by **nothing**. Every drift finding, balance warning, baseline verdict and stratification fallback was computed and discarded | **D6.** Both now in the response, typed by `PipelineWarning`. `result` additionally carries `stratified`, `test_size` and `seed`. **The UI still does not render them** — P2, §5.7 | `test_the_preprocess_response_carries_the_warnings_and_the_checks` |
| Invalid configuration rejected clearly | PASS | `build_plan`, `_is_classification` | Raise with the valid values named | — | `test_glassbox.py` |
| Exported script reproduces the run | PASS | `export_script`, `codegen.generate` | Tests *run* the generated code and check the output, rather than grepping the source | — | `test_codegen.py` |
| Missing-value indicator columns | **PLANNED** | `app/plan.py` only | Implemented in a module nothing imports | §2.4 | — |
| Rare-category grouping | **PLANNED** | `app/plan.py` only | Same | §2.4 | — |
| Skew correction (Yeo-Johnson) | **PLANNED** | — | Skew is measured and reported; nothing acts on it | — | — |
| Per-column editable plan | **PLANNED** | — | §7 | — | — |

---

## 4. Preprocessing correctness review

The ten questions the review asks, each answered from the code and from a test. The
*design* these answers judge — why the ordering is what it is, how stratification
predicts sklearn's verdict, what reproducibility is pinned to — is Design.md §7.

| # | Question | Answer | Evidence |
|---|---|---|---|
| 1 | Is preprocessing fitted only on training data? | **Yes.** `fit_preprocessing` step 6 is the only `.fit` in the module's data path; the generated script fits on `df.loc[train_index]` throughout | A spy on `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `OrdinalEncoder` and `FrequencyEncoder` records the row indices of every `fit` and asserts zero overlap with the held-out rows — `test_leakage.py` |
| 2 | Is the target excluded from feature transformations? | **Yes.** Features are `df.drop(columns=[target])` before the split; the target never enters `build_plan`'s profile and never appears in the feature map | `test_c_regression_separates_the_target_and_does_not_stratify` |
| 3 | Does the test set influence imputation or scaling statistics? | **No**, in either path | Logged fill value equals the *training* median and differs from the whole-file median — `test_b_fill_values_…` |
| 4 | Does the test set influence any other **decision**? | **It did, in the generated script.** Column dropping and calendar-part emission were both measured over the whole file. **Fixed.** The engine path was already correct | **D5.** Two tests built so the whole-file answer and the training answer differ — both verified to fail against the prior code |
| 5 | Does categorical encoding handle unseen categories? | **Yes.** One-hot ignores; ordinal maps to −1; frequency maps to 0.0, which is what "never observed in training" means | `test_leakage.py`, `test_rules.py` |
| 6 | Is stratification appropriate to the target type and class counts? | **Now yes.** Regression is never stratified. Classification is stratified unless one of three named conditions holds, each producing a warning. The third — a fold smaller than the class count — was unguarded | **D3.** `test_too_many_classes_for_the_fold_falls_back_and_says_why` |
| 7 | Is leakage detection meaningful, and does it avoid correlation-only conclusions? | **Yes.** Two independent screens: \|r\|>0.99 **and** a depth-2 decision tree scoring ≥0.98 in cross-validation. Both run on the training fold. Findings are phrased as what was observed and what it suggests, not as proof | `test_e_a_column_derived_from_the_answer_…`, `test_e_leakage_warnings_are_phrased_as_suspicion_not_proof` |
| 8 | Does feature/target alignment survive splitting? | **Yes.** Shared index in order, folds disjoint, and columns identical between `X_train` and `X_test` | `test_a_train_and_test_are_disjoint_and_aligned` |
| 9 | Are seeds and outputs reproducible? | **Yes.** One seed; `test_size` and `seed` recorded on the result rather than recovered — deriving `test_size` back from row counts rounds, and a rounded fraction re-enters `train_test_split` as a different ceiling | `test_a_the_split_is_stratified_and_reproducible` asserts identical folds, identical matrices and a byte-identical log |
| 10 | Is input data modified unexpectedly? | **No.** The caller's frame is byte-identical after a run, including when the engine blanks infinities inside it | `test_fit_preprocessing_blanks_infinity_without_touching_the_callers_frame` |
| 11 | Do the artifacts preserve what is needed to apply the transformations again? | **Partially.** The fitted `ColumnTransformer` is persisted as joblib and the feature map is complete and asserted. `plan.json`, `schema.json`, `log.json` and train/test parquet are **not** written, and the test fold is never persisted | PLANNED — §7, and the prerequisite for Stage 4 |

---

## 5. Test and build audit

> **These are the figures of the first audit round.** The remediation round that
> followed it changed them; §8.5 has the current numbers. This section is kept as the
> record of what was true before that work, because §8 is written as a comparison
> against it.

### 5.1 Commands executed, and their actual results

All run on darwin 25.6.0 (macOS 26.6.2), Python 3.14.2, on 30 October 2026.

| Command | Result |
|---|---|
| `cd backend && ../.venv/bin/python -m pytest tests/ -q` **(baseline, before any change)** | **339 passed**, 1 warning, 115.93s · coverage **88.46%** |
| `cd backend && ../.venv/bin/python -m pytest tests/ -q` **(after all fixes and new tests)** | **381 passed**, 0 failed, 0 skipped, 1 warning, 122.29s · coverage **88.83%** |
| `cd frontend && npm run build` (`tsc -b && vite build`) | **PASS** — built in 246 ms, 314.50 kB JS / 53.94 kB CSS |
| `cd frontend && npm run lint` (`oxlint`) | **PASS** — exit 0, no diagnostics |
| `cd backend && alembic upgrade head` against an empty database | **PASS** — both revisions applied |

The one warning is `StarletteDeprecationWarning` from `fastapi/testclient.py` about
`httpx` vs `httpx2`. It is raised by FastAPI's own import, not by this code.

### 5.2 Coverage, per module

Gate: `--cov-fail-under=80` in `pytest.ini`. Measured **88.83%** of 3,744 statements and
1,146 branches.

| Module | Before | After |
|---|---|---|
| `app/codegen.py` | 89.9% | 90.0% |
| `app/services/glassbox.py` | 86.5% | 87.3% |
| `app/services/ingest.py` | 88.7% | 89.2% |
| `app/services/profile.py` | 89.3% | 88.2% |
| `app/services/screening.py` | 88.2% | 92.3% |
| `app/services/checks.py` | 89.7% | 90.9% |
| `app/main.py` | 87.8% | 86.4% |
| **TOTAL** | **88.46%** | **88.83%** |

`app/plan.py` and `app/runner.py` are omitted from measurement; `.coveragerc` states the
reason and says to remove the omission the moment either is wired in.

### 5.3 Tests added by this audit

`backend/tests/test_scenarios.py` — 34 new tests. `backend/tests/test_codegen.py` — 3
added. `backend/tests/test_api.py` — 5 added. 42 new tests; the suite went 339 → 381.

Every regression test was checked against the prior implementation rather than assumed
to be discriminating. The two codegen decision tests were run against a temporarily
reverted `codegen.py` and **both failed** with the expected assertion; restored, both
pass. The CORS tests were run against the reverted `allow_methods` and **two of the
four failed**; restored, all pass. The remainder are discriminating by construction —
a 500 response, an exception, or a `KeyError` on a payload key that did not exist.

### 5.4 What could not be run, and why

| | Reason |
|---|---|
| Browser end-to-end tests | Playwright is not installed and adding it is a dependency decision beyond this audit's scope |
| The non-macOS sandbox path | `sandbox-exec` is macOS-only; the limits-only path cannot be exercised here |
| A 10 MB performance budget | The fixture is generated on demand and a `perf` marker exists, but no test asserts a timing. Writing one requires choosing a budget, which is a product decision |
| Frontend unit tests | None exist; no test runner is configured |

### 5.5 Test-suite integrity

No test was weakened, skipped or deleted to make the suite pass. Four assertions in the
new file were corrected while writing it: two named the wrong *payload key* (`health`
for `quality`, `name` for `column`); one reconstructed an expected median without
mirroring the engine's deduplication; and one expected the target to appear in
`plan.dropped`, which it never does, because `build_plan` is handed a profile of the
features alone. In each case the test was wrong about the contract, not the code.

One test was also corrected for a reason worth recording: `_routed_methods()` first
walked `app.routes`, which in this FastAPI version represents an included router as a
single opaque `_IncludedRouter` entry. It therefore enumerated `GET` and `POST` and
silently missed every workspace route — the half of the API that uses `PUT`, and the
half D7 broke. It now reads the OpenAPI document, which is generated from the real
routing table.

### 5.6 Security and reliability

| Check | Status | Finding |
|---|---|---|
| File-upload validation | PASS | Size capped **while streaming** (`main.py:135`), so an oversized body is never buffered. Content sniffed by magic bytes before any parser runs (`ingest._check_content`). Extension checked as well, not instead |
| Unsafe deserialisation | PASS | No `pickle.load` of anything user-supplied. `joblib` writes pipelines the app itself produced; no artefact is loaded from a user-supplied path |
| Path traversal | PASS | `storage.py:41` validates by allow-list rather than sanitising: every key segment must match `^[A-Za-z0-9._-]+$`, and `""`, `.` and `..` are rejected by name before the segments are joined onto the root. Hostile filenames are tested in `test_api.py` |
| Temporary-file cleanup | PASS | Sandbox work directories removed after execution; the suite's temporary database and storage root removed at session end |
| Secrets in source control | PASS | None found. `git` history and working tree inspected |
| Logging of dataset content | PASS | Messages name columns and counts, never row values |
| Resource exhaustion (user code) | PASS | `RLIMIT_CPU`, `RLIMIT_FSIZE`, `RLIMIT_NPROC`, a memory watchdog thread (macOS silently refuses to lower `RLIMIT_AS`, so the usual `setrlimit` does nothing there), a wall-clock timeout on the process group, and a scrubbed environment. `test_executor.py` genuinely attempts each attack and asserts afterwards that the repository still exists |
| **Traceback exposure** | **PASS — fixed here** | A file containing `inf` produced an unhandled `ValueError` and FastAPI returned the full stack trace, including absolute filesystem paths and library internals, to the client. **D1.** Now a 200; the class of failure is covered by Scenario F, which asserts no `Traceback` and no `File "` in any refusal |
| Error handling generally | PASS | Nothing is swallowed. `UserFacingError` → 400 with a sentence; anything else reaches the 500 handler, which is correct — an unexpected error is a defect, not a user error |
| **CORS** | **PASS — fixed here** | `allow_methods` listed `GET` and `POST` while two workspace routes answer `PUT`. A browser asks permission before sending a `PUT`, and the preflight was refused `400 Disallowed CORS method` — so changing a toggle and saving edited code both failed in the running application. **D7.** `PUT` added; the origin allow-list is unchanged and still refuses an unknown origin, which a test now asserts. `DELETE` deliberately not added — there is no route to permit |

### 5.7 Prioritised remediation plan

**P0 — release-blocking. All fixed and verified in this audit.**

| # | Finding | Impact | Files | Fix | Verification |
|---|---|---|---|---|---|
| D1 | `np.histogram` raises on a non-finite range; the route had no handler | `POST /api/datasets` returned 500 with a full traceback — a crash *and* an information leak | `services/profile.py`, `services/ingest.py` | `_finite()` guards the three profile statistics; ingestion blanks `inf` with a cleaning sentence | `test_profiling_a_non_finite_column_does_not_raise`, `test_a_file_containing_infinity_uploads_and_preprocesses` |
| D2 | `numeric_summary` emitted `Infinity` into the JSON response | Invalid JSON for any strict parser, including `JSON.parse` | `services/profile.py` | Same `_finite()` guard | `assert "Infinity" not in profile.text` |
| D7 | CORS permitted `GET` and `POST`; the interface saves config and code with `PUT` | The browser refused to send either request. **Changing a toggle and saving edited code were both broken in the running application** — and invisible to the suite, because `TestClient` calls the app directly and never performs a preflight | `main.py`, `tests/test_api.py` | `PUT` added to `allow_methods`. The new test derives the required methods from the OpenAPI document, so the policy is checked against the routes rather than against a list someone maintains | `test_every_method_the_api_uses_survives_a_preflight[PUT]`, `test_saving_a_workspace_configuration_is_reachable_from_the_browser`, `test_an_origin_the_app_does_not_serve_is_still_refused` |

**P1 — significant. All fixed and verified in this audit, except where stated.**

| # | Finding | Impact | Files | Fix | Verification |
|---|---|---|---|---|---|
| D3 | Stratification ignored sklearn's fold-size requirement | A legitimate multiclass file aborted with a message about a `test_size` the user never chose | `services/glassbox.py`, `codegen.py`, `main.py` | `_stratification()` predicts sklearn's verdict and falls back with a warning naming the real cause; the misleading "try a file with more usable columns" suffix is now attached only to the failure it describes | `test_too_many_classes_for_the_fold_falls_back_and_says_why`, `test_a_dataset_with_many_classes_reaches_the_api_rather_than_a_400` |
| D4 | Blank target rows not removed by the module | Classification aborted opaquely; **regression silently trained on `NaN` labels** | `services/glassbox.py` | Removed before the split, for both tasks, with a warning | `test_rows_with_a_blank_target_are_removed_and_reported` (both tasks), `test_a_target_that_is_entirely_blank_fails_with_a_sentence` |
| D5 | Generated script made two column-shape decisions from the whole file | A held-out row could decide which columns the model is given — the leak the product's central claim denies | `codegen.py` | Both blocks moved after the split and measured on `train_index` | Two tests, **verified to fail against the reverted code** |
| D6 | Engine warnings and post-split checks read by nothing | Every drift finding, balance warning and baseline verdict discarded | `main.py`, `schemas.py`, `services/glassbox.py` | Both added to `AnalysisResponse`, typed by `PipelineWarning`; `result` now carries `stratified`, `test_size`, `seed` | `test_the_preprocess_response_carries_the_warnings_and_the_checks` |
| **O1** | **No asserted performance budget** | A regression in ingestion or profiling cost is invisible | `tests/` | **Open.** Choose a budget for the 10 MB fixture and assert it under the existing `perf` marker | — |

**P2 — deferrable.**

| # | Finding | Impact | Proposed fix |
|---|---|---|---|
| O2 | The UI does not render `warnings` or `checks` | A drifted split or weak baseline is reachable over HTTP and invisible on screen | Render them in `InsightsPanel.tsx`, keyed on severity. The workspace would need to call the preprocess route, or the compile path would need to run the same checks |
| O3 | `app/plan.py` / `app/runner.py` unreachable, and the only home of two spec features | A reader of `config.py` believes missing-indicator and rare-category grouping exist | Wire in or delete, then remove the two `omit` lines from `.coveragerc` |
| O4 | `recharts` and `framer-motion` installed, imported by zero files | Dependency surface with no user | Use one for the histogram the backend already computes, or drop both |
| O5 | Two health-scoring systems | They can disagree about the same file | Unify on `health.assess`; keep `_score_quality` only if ingestion genuinely needs a score before profiling |
| O6 | Error envelope is `{"error": …}`, not the spec's `{code, message, detail}` | Divergence from the Phase 1 specification §7 | Deliberately not changed — it breaks every existing client. Decide at the next API version |
| O7 | `12%` ingests as `12` | A percentage column is scaled 100× | Decide the semantics, then change it loudly. Not a silent fix |
| O8 | `plan.md` and the Phase 1 specification both link to `VERIFICATION.md`, which does not exist | Two broken documentation links | Point them at PRD.md §8 and this §5, or write the file |
| O9 | No `DELETE /api/datasets/{id}` | Uploads accumulate with no API to remove them | Add the route and widen CORS to match |

---

## 6. Housekeeping performed

* **`evidence/` removed**, as requested. It held a slide-deck validation package — an
  executed notebook, a generated `.docx`, a claim register and 14 PNGs, 3.3 MB in all.
  It was untracked by git, and a repository-wide search for `evidence/`, each filename
  inside it and each tool that wrote it found **no reference from any source file,
  test, configuration, CI definition or documentation link**. Nothing depended on it, so
  nothing needed repairing and no replacement was created. Deleted outright.
* No other file was removed **in that round**. `Sem;colon (2).pptx`, `plan.md`,
  `docs/DECK_FACTS.md` and `docs/DEMO_GUIDE.md` were user work and were left alone, as
  were the uncommitted deletions of `CLAUDE.md` and `.claude/` already in the tree.
* **Later, at the user's request**, `docs/` was reorganised: the Phase 1 specification,
  the previous audit archive, and the two Sem-7 deck documents (`DECK_FACTS.md`,
  `DEMO_GUIDE.md`) were removed, and this document, PRD.md, Design.md and plan.md were
  moved into `docs/`. The deck documents described the repository at commit `5a2328e`
  and were already stale. Nothing in the application reads any of them; the removal
  cost is the loss of the specification's own wording, which §3–4 of PRD.md restates.

---

## 7. The two structural gaps

Both carried forward from the previous audit. Both are still true, and neither is a
defect — each is a design decision that has not been taken.

**1. There is no plan object.** `PipelineConfig` is per-*step* ("impute numerics with
the median"). The specification wants per-*column* ("`Age` → impute median, because 20%
missing, skew 0.4"), proposed by a planner, shown to the user, editable column by
column, handed to an executor. That changes the data model, adds
`POST /datasets/{id}/plan`, and adds a plan review screen.

**2. There is no run resource.** Execution produces a `compilation` tied to a
`workspace`. The specification wants a versioned `run` at `/runs/{id}` carrying
`plan.json`, `schema.json`, `log.json`, the joblib pipeline and train/test parquet —
`schema.json` specifically so a later phase can run inference. **This is the hard
prerequisite for Stage 4** (Design.md §11.1 sets out what is missing, item by item): a meta-learning engine that ranks algorithms by similarity
to past experiments has nothing to learn from until runs are durable and comparable.

A third has been added by this audit:

**3. Stage 3 is implemented twice.** `services/glassbox.py` builds an sklearn
`ColumnTransformer`; `codegen.py` emits standalone Python that does the same work with
different code. They are genuinely different artefacts — one is introspected to produce
the log, the other is what the user edits — so this is not simple duplication. But D5
existed in one path and not the other, and was invisible precisely because the other
path was correct and tested. Any invariant that matters must now be asserted twice, and
the tests say so explicitly.

Neither 1 nor 2 is a rewrite. The engine, the leakage guarantee, the sandbox, the health
dimensions and the storage layer all survive; this is a new layer over them plus an API
and UI reshaped around it.

---

## 8. Remediation round — 30 October 2026

The nine observations of §5.7 and the three structural gaps of §7, worked through in
priority order. Every row's status is one of **FIXED AND VERIFIED** (a test exists that
fails against the previous behaviour), **FIXED BUT NOT FULLY VERIFIED**, **DEFERRED WITH
REASON**, **REQUIRES PRODUCT DECISION**, or **NOT FIXED**.

### 8.1 Baseline, re-established before any change

The previous revision's figures were confirmed against the working tree before anything
was touched: **381 passed, 88.83 % coverage**, frontend build and lint clean, migrations
applying from empty. Nothing had drifted.

### 8.2 O1–O9

| | Finding | Status | Evidence |
|---|---|---|---|
| **O1** | No asserted performance budget | **FIXED AND VERIFIED** | `tests/test_performance.py` — 9 tests. Budgets are a multiple of a **calibration workload** measured in the same process, so a slower machine gets a proportionally larger budget rather than a spurious failure. Measured: ingest 0.59 s, profile 0.33 s, health 0.39 s, plan 0.13 s, run 0.48 s, **total 1.92 s**; budgets 2.5/1.5/1.5/1.0/2.0/**8.0 s**. Confirmed discriminating by forcing a 1 ms budget. A scaling test compares 1× and 4× rows to catch an accidental quadratic, and needs no reference machine |
| **O2** | Warnings and checks never rendered | **FIXED AND VERIFIED** | `frontend/src/workspace/PlanReview.tsx`. Severity-coloured warnings, KS drift, class balance in both folds, stratification fallback with its reason, and the sanity baseline — all from one run's own output, so the engine is **not** executed twice. Verified in a browser against a live server: baseline 0.582 → 0.777, drift 0 of 13, balance 58/42 both folds |
| **O3** | `plan.py` / `runner.py` unreachable | **FIXED AND VERIFIED** | **Decision: integrate, not delete.** Both are complete, spec-written implementations of the two structural gaps — a per-column plan and a plan executor with `schema.json`, a determinism hash and a log-invariant check. Deleting them would have meant writing them again. Now reached by `services/planning.py` and `planning_api.py`; the `.coveragerc` omissions are removed and both are measured (`plan.py` 89.5 %, `runner.py` ~90 %). Two defects found in them while wiring — see §8.4 |
| **O4** | `recharts` / `framer-motion` unused | **FIXED AND VERIFIED** | `recharts` now draws the per-column distribution **from the bins the backend already computed** — which required fixing the real blocker: `ColumnProfileOut` declared no `histogram`, so a response model silently discarded it. `framer-motion` removed (`npm uninstall`, lockfile updated); the app's animation is CSS, as the README states. The plan screen is lazy-loaded, so the chart library costs nothing on first paint: main bundle 319 kB, plan chunk 372 kB |
| **O5** | Two health-scoring systems | **FIXED AND VERIFIED** | `health.assess` confirmed canonical — it is already what `Dataset.health_score`, `Profile.health_score` and the workspace report. The ingest-time check is retained (it runs before any profile exists) and is labelled on the wire as *"Ingest quality check (not the workspace health score)"*. Five tests in `test_lifecycle.py` pin this, including one asserting the two scores **differ** on the fixture, so a test that conflated them could not pass |
| **O6** | `{"error"}` vs the spec's `{code, message, detail}` | **FIXED AND VERIFIED** | Both are sent. `error` is unchanged for every existing client; `code`, `message` and `detail` are added beside it. A stable code vocabulary distinguishes the three categories the spec asks for. Two real inconsistencies fixed on the way: `RequestValidationError` answered in a different shape entirely (`{"detail": [...]}`), and a URL matching no route bypassed the handler because Starlette's router raises the **base** `HTTPException`, not FastAPI's subclass |
| **O7** | `12%` → `12`, not `0.12` | **FIXED AND VERIFIED — semantics deliberately unchanged** | The spec names `"12%"` as a string to coerce but not to what. Measured the thing that settles it: **all three scalers are affine, so the two representations produce identical model inputs** (test included). The choice therefore affects only what the user reads, and rescaling every existing dataset by 100 would be a cost with no benefit. Percentage points kept; what was wrong — that the unit was dropped *silently* — is fixed: such columns are detected and reported as their own repair. Seven tests cover `12%`, `0.12`, `12`, blanks, and a mixed column (which is **not** claimed as a percentage) |
| **O8** | Links to a non-existent `VERIFICATION.md` | **FIXED AND VERIFIED** | `plan.md` and the Phase 1 specification repointed at AUDIT.md §5 and PRD.md §8. No placeholder file created. Every relative link in every markdown file now resolves, checked programmatically |
| **O9** | No `DELETE /api/datasets/{id}` | **FIXED AND VERIFIED** | Route added, plus the UI with a danger-toned confirmation naming exactly what goes. **Ordering is the design decision**: rows are deleted and committed first, files second, so a partial failure strands bytes (recoverable) rather than leaving rows pointing at data that is gone (not). Orphaned keys are reported, not hidden. A latent defect was fixed to make this work at all — see §8.4 |

### 8.3 The three structural gaps

| | Gap | Status | What exists now |
|---|---|---|---|
| **1** | No plan object | **FIXED AND VERIFIED** | A `plans` table, immutable and versioned per dataset; `POST/GET /api/datasets/{id}/plan`, `GET/PUT /api/plans/{id}`; a review screen with a per-column editable action, the reason, the measurements behind it, and a distribution. Editing writes the **next** version — the one a run already named is never altered. 22 tests in `test_plan.py` |
| **2** | No run resource | **FIXED AND VERIFIED** | A `runs` table and `POST /api/plans/{id}/run`, `GET /api/runs/{id}`, `GET /api/runs/{id}/artifacts/{name}`, `GET /api/datasets/{id}/runs`. Seven artifacts: `plan.json`, `schema.json`, `log.json`, `pipeline.joblib`, `train.parquet`, `test.parquet` and `pipeline.py`. Status is `running` → `succeeded`/`failed` and a failure is kept with its error, never presented as a result. 27 tests in `test_run.py` |
| **3** | Stage 3 implemented twice | **FIXED AND VERIFIED** | The plan is the canonical representation both paths consume. `app/plan_codegen.py` emits a script that **carries the engine's own transformer classes and pipeline builder**, copied with `inspect.getsource` — so the two cannot disagree about what an encoder means, because there is one definition and the script has a copy. The config thresholds those definitions close over are derived from the sources rather than listed by hand. 39 parity tests in `test_parity.py` compare the two paths on an output-hash digest |

### 8.4 Defects found while doing this work

Four, none of them in the §5.7 list — they were found by probing the code being wired in.

| | Defect | Where | Fix |
|---|---|---|---|
| **D8** | The planner measured rows the executor never fitted on. `propose` split the frame as given; `execute` deduplicated and dropped blank-target rows *first*, so the two training folds held different rows — and every statistic the user reviewed described data the pipeline never saw | `app/plan.py`, `app/runner.py` | One shared `prepare()`, called by both. Pinned by `test_preparation_is_shared_so_both_sides_split_the_same_rows` |
| **D9** | Both modules carried their own copy of the stratification rule, neither with the fold-size guard that D3 added to the engine — so a legitimate multiclass file would have aborted again | same | `choose_stratification` promoted to a shared public helper; the second and third copies deleted |
| **D10** | **Every `ON DELETE CASCADE` in the schema was fiction.** SQLite ignores foreign keys unless `PRAGMA foreign_keys=ON`, which is off by default, so deleting a dataset would have orphaned its profile, runs, plans and workspace | `app/db.py` | The pragma is set on the pool's connect event. `test_deletion_leaves_no_rows_behind_in_any_table` would fail without it |
| **D11** | The histogram `services/profile.py` computes server-side — with a docstring explaining why the browser must not compute it — was **discarded by the response model**, which declared no such field | `app/schemas.py` | `stats`, `histogram`, `top_values` and `flags` added to `ColumnProfileOut` |

One frontend defect was found by driving the UI rather than by reading it: React's
development double-invoke of effects fired two plan loads at once, both found no plan,
both proposed one, and the user arrived on **version 2 of a plan they had never
edited**. Guarded with a ref, which survives the re-invocation where state does not.

### 8.5 Test and build results

| Command | Before | After |
|---|---|---|
| `pytest tests/ -q` | 381 passed | **506 passed**, 0 failed, 0 skipped |
| coverage of `app/` | 88.83 % | **89.78 %** — and now *including* `plan.py` and `runner.py`, which were previously excluded from measurement |
| `tsc -b` | pass | pass |
| `oxlint` | pass | pass |
| `vite build` | 314 kB | 319 kB main + 372 kB lazy plan chunk |
| `alembic upgrade head` from empty | 2 revisions | **3 revisions** |
| `alembic upgrade head` from the previous schema | — | **passes with every existing dataset and workspace intact** |

No test was weakened, skipped or deleted. The 125 new tests are additive; all 381
previous tests still pass.

### 8.6 Migration

One new revision, `dffe8ca073fe`, **purely additive**: two tables and their indexes, no
existing table, column, index or row altered or removed. Verified twice — from empty,
and against a database seeded at the previous revision with a dataset row, which
survived with its fields intact.

`compilations` and `runs` are deliberately **not** merged. A compilation records the user
running code in the workspace; a run records the execution of a reviewed plan. They are
different events, and collapsing them would make one of the two histories a lie.

### 8.7 Still open

| | Item | Why |
|---|---|---|
| **O2-UI** | The *workspace* panels still do not show engine warnings | The plan screen does. The workspace compiles a user-edited script in a sandbox, which produces no drift or baseline — surfacing them there would mean running the engine a second time, which O2 explicitly rules out |
| **Frontend tests** | No test runner is configured | Out of scope for this round and a dependency decision. The plan screen, the charts, the edit cycle and the delete dialog were verified by driving a live server in a browser, which is recorded here as **FIXED BUT NOT FULLY VERIFIED** for want of an automated guard |
| **O7 semantics** | Percentage points retained | **REQUIRES PRODUCT DECISION** to change. The evidence for leaving it is in §8.2; the decision to change it would be a breaking one and is documented rather than taken |
| **Dead code** | `codegen.py`'s legacy `PipelineConfig` path | Still the workspace editor's generator, still used, still tested. Converging it onto the plan would change the editor's product behaviour and is a design decision, not a defect |
