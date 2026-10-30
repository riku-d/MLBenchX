# ML-BenchX — Product Requirements

**Version 1.1 · 30 October 2026 · current release target: Stages 1–3, plus the
reviewable plan and the durable run**

This document describes what ML-BenchX is for, what it is required to do, and how each
requirement is verified. It is grounded in the repository as it stands at the time of
writing, not in an intended design: where a requirement is not met, it says so and says
what the gap is.

| Document | Answers |
|---|---|
| [PRD.md](PRD.md) | What the product must do, and what counts as done |
| [Design.md](Design.md) | How it is built, and why it is built that way |
| [AUDIT.md](AUDIT.md) | What is actually true today — verification evidence, defects, remediation |

**This document is the requirements authority.** It was written against an earlier
Phase 1 specification (`docs/PHASE1_SPEC.md`), which has since been removed from the
repository; every requirement that specification carried has been restated here in §3
and §4, with its status and verification. Where the text below refers to "the Phase 1
specification", it is naming that retired source, not a file you can open.

Requirement identifiers defined here (`FR-ING-001`, `NFR-004`) are referenced from both
other documents. Technical reasoning is not repeated here — it is in Design.md.
Verification evidence is not repeated here — it is in AUDIT.md.

---

## 1. Product overview

### 1.1 Name and purpose

**ML-BenchX** is a glass-box preprocessing workbench for tabular data. A user uploads a
spreadsheet; the product tells them what is wrong with it, proposes a preprocessing
pipeline, explains each step in terms of their own data, lets them change or override
any of it, and compiles a model-ready train/test split together with the Python that
produced it.

The long-term product is a seven-stage automated benchmarking platform (§10). **This
release covers Stages 1–3 only.** Stages 4–7 — meta-learning, parallel training,
explainability, leaderboards — are not implemented and are not in scope.

### 1.2 Problem statement

Automated ML tools transform data silently. A user hands over a CSV and receives a
model, with no account of which columns were dropped, what blanks were filled with, or
whether the score they are shown is honest. Two failures follow from that opacity:

1. **Silent leakage.** The most common cause of a model that scores well and fails in
   production is a preprocessing statistic learned from rows that were supposed to be
   held out. It is invisible in the output — the result is simply a better number.
2. **Unactionable diagnostics.** "Your data quality is 61/100" tells a user nothing
   about what to change.

### 1.3 Intended users

| User | Needs from this product |
|---|---|
| **Data science student / early practitioner** | To see a correct pipeline built, with each decision justified, rather than to trust a black box |
| **Analyst with a spreadsheet** | To find out whether their file is usable before investing in modelling, in language that is not about dtypes |
| **ML engineer** | A reproducible, inspectable preprocessing artefact — code and a fitted pipeline — they can lift into their own project |
| **Reviewer / instructor** | Evidence that the leakage guarantee holds, not an assertion that it does |

### 1.4 Value proposition

Every automated recommendation is **visible, explained, optional and reversible**, and
the code that runs is code the user can read and change. Two properties carry that:

* **The split happens before anything is measured or decided** — not merely before
  fitting. Imputation values, scaler centres, encoder categories *and the column profile
  that decides which columns survive* all come from the training rows alone.
* **The explanation is derived, not narrated.** Every sentence in the log is built by
  reading a fitted object's learned attributes after the fact, so it cannot drift out of
  sync with what ran.

### 1.5 Goals and success criteria

| # | Goal | Measured by | Status |
|---|---|---|---|
| G1 | No held-out row reaches a fit or a decision | `tests/test_leakage.py`, `tests/test_scenarios.py::test_e_*`, `tests/test_codegen.py::test_a_column_is_dropped_on_how_empty_the_training_rows_are` | **Met** |
| G2 | Identical input and seed produce an identical pipeline, log and output | `test_a_the_split_is_stratified_and_reproducible` | **Met** |
| G3 | A malformed, hostile or unusual file fails in a controlled way, never with a traceback | `test_scenarios.py` Scenario F | **Met** |
| G4 | Every transformation is explained in terms of the user's own columns and numbers | `test_glassbox.py`, `test_rules.py` | **Met** |
| G5 | Every condition the engine detects reaches the caller | `test_the_preprocess_response_carries_the_warnings_and_the_checks` | **Met in the API; not yet rendered in the UI** (§2.3) |
| G6 | Automated test coverage of `app/` ≥ 80% | `pytest --cov`, gate in `pytest.ini` | **Met — 89.8%** |

---

## 2. Scope and release boundaries

### 2.1 Implemented and verified

Verified means: exercised by a test in `backend/tests/` that was executed, and that
fails if the behaviour regresses.

* File ingestion — CSV, TSV, Excel, JSON (records / columns / line-delimited), Parquet
* Encoding and delimiter detection; size cap enforced while streaming
* Content sniffing by magic bytes, not extension alone
* Format repair: missing-value placeholders, text-as-number, text-as-date, true/false
  columns, infinities, header tidying, duplicate-header disambiguation
* Per-column profiling: kind, missing, cardinality, outliers, histogram, top values,
  numeric summary with skew
* Dataset health score over six weighted dimensions, with the arithmetic shown
* Duplicate-row and high-correlation detection; class-imbalance reporting
* Target detection from a plain-English goal, by string matching
* Identifier and target-leakage screening, on the training fold
* Train/test split — stratified where the label supports it, with a stated fallback
* Imputation, categorical encoding (binary / one-hot / frequency / cross-fitted target),
  scaling (standard / min-max / robust), calendar feature derivation
* Post-split checks: KS drift per feature, class balance in both folds, sanity baseline
* The derived glass-box log and the feature map
* Code generation, and sandboxed execution of user-edited code
* The workspace API the interface drives
* A per-column preprocessing plan: proposed, reviewable, editable, immutably versioned
* A durable run of one plan version, with seven artifacts and a determinism hash
* A plan review screen with per-column distributions, and the run's warnings and checks
* Dataset deletion, cascading to everything derived from it

### 2.2 Implemented but incompletely verified

* **Performance is budgeted, on one file shape.** 36k rows × 50 columns. A file with far
  more columns, or one very high-cardinality categorical, has a different cost profile
  and nothing would notice a regression specific to it.
* **The `sandbox-exec` layer is macOS-only.** `test_executor.py` attacks it and passes
  on this machine; on Linux only the process limits apply, and that path is not
  exercised here.
* **Excel sheet selection** is implemented (`list_sheets`, `_read_excel`) and covered by
  a fixture, but only for a two-sheet workbook.

### 2.3 Partially implemented

* **Engine warnings in the workspace panels.** The plan review screen renders them in
  full (FR-PLAN-008). The *workspace* panels still do not, because that path compiles a
  user-edited script in a sandbox which produces no drift or baseline — showing them
  there would mean running the engine a second time.
* **Two code generators.** `codegen.py` generates from `PipelineConfig` for the editor;
  `plan_codegen.py` generates from the plan. Both are tested, and the second carries the
  engine's own definitions so it cannot diverge. Converging the first is a product
  decision about what the editor does, not a defect.
* **No frontend test runner.** The plan screen, the charts, the edit cycle and the
  delete confirmation were verified by driving a live server in a browser. There is no
  automated guard against a regression in them.

### 2.4 Planned — not implemented

* Stages 4–7 in full (§10)

Everything else previously listed here has been built: the per-column plan and its review
screen, missing-indicator columns, rare-category grouping, Yeo-Johnson skew correction,
the versioned run with `plan.json` / `schema.json` / `log.json` / train and test parquet,
and `DELETE /api/datasets/{id}`. See §3.4 and AUDIT.md §8.

### 2.5 Explicitly out of scope for this release

Model training, hyperparameter tuning, SHAP or any other explainability of a *model*,
leaderboards, PDF reports, Celery, PostgreSQL, authentication, multi-user projects,
streaming ingestion, and datasets larger than the 50 MB upload cap.

---

## 3. Functional requirements

Priorities: **P0** release-blocking, **P1** significant, **P2** deferrable.
Verification: **T** automated test, **I** code inspection, **M** manual.

Status is as audited on 30 October 2026 and matches AUDIT.md.

### 3.1 Stage 1 — Ingestion

| ID | Requirement | Pri | Acceptance criteria | Verification | Status |
|---|---|---|---|---|---|
| FR-ING-001 | Accept CSV and TSV uploads | P0 | A comma- or tab-separated file with a header row is read into a frame with its columns named from that row | T `test_app.py`, `test_source_format.py` | Met |
| FR-ING-002 | Accept Excel (`.xlsx`) uploads | P0 | A workbook is read; the sheet used is reported, and the other sheet names are listed | T `test_source_format.py` | Met |
| FR-ING-003 | Accept JSON and Parquet | P1 | Records, column-oriented and line-delimited JSON, and Parquet, each read into the same frame shape | T `test_source_format.py` | Met |
| FR-ING-004 | Enforce a maximum upload size | P0 | A body over `MAX_UPLOAD_BYTES` (50 MB) is refused with HTTP 400 and a sentence; the limit is enforced while streaming, so an oversized body is never held in memory | T `test_api.py`; I `main.py:135,175` | Met |
| FR-ING-005 | Detect and validate text encoding | P0 | A non-UTF-8 file (e.g. Latin-1) is read without mojibake; an undecodable file is refused with a sentence | T fixture `latin1.csv` | Met |
| FR-ING-006 | Detect the delimiter | P0 | Comma, semicolon and tab files each parse into more than one column | T `test_source_format.py` | Met |
| FR-ING-007 | Verify content, not only the file extension | P0 | A binary payload renamed `.csv` is refused before any parser runs | T `test_scenarios.py::test_f_bad_uploads_are_refused_with_a_sentence`; I `ingest._check_content` | Met |
| FR-ING-008 | Refuse an empty or header-only file | P0 | Zero bytes, or a header with no rows beneath it, is refused with a sentence naming the problem | T Scenario F | Met |
| FR-ING-009 | Refuse a single-column file | P1 | A file with one column is refused, and the message suggests checking the separator | T Scenario F | Met |
| FR-ING-010 | Refuse a file below the row floor | P1 | Fewer than `MIN_ROWS` (50) rows is refused, because no honest held-out fold can be taken from it | T Scenario F | Met |
| FR-ING-011 | Detect a headerless file | P1 | A file whose first row is data is identified rather than silently promoting that row to column names | I `ingest._looks_headerless` | Met |
| FR-ING-012 | Tidy and disambiguate column headings | P1 | Whitespace is stripped; a heading used twice is made unique; the original → cleaned mapping is retained and reported | T fixture `unicode_duplicate_headers.csv` | Met |
| FR-ING-013 | Infer column types | P0 | Each column is classified numeric, categorical, datetime, text or constant, and the classification drives routing | T `test_glassbox.py::profile_frame` tests | Met |
| FR-ING-014 | Recognise text that means "missing" | P0 | `N/A`, `-`, `unknown`, `nil` and the rest of `MISSING_TOKENS` become real blanks, and the count is reported | T `test_app.py` | Met |
| FR-ING-015 | Convert numbers stored as text | P0 | `1,234`, `$5`, `12%` convert to numbers, but only when *every* non-blank value in the column converts | T `test_rules.py`; fixture `messy_types.csv` | Met. `12%` → `12`, not `0.12` — documented, not corrected |
| FR-ING-016 | Convert dates stored as text | P0 | A column whose values carry a four-digit year and parse at a rate of at least `DATE_PARSE_RATE` (90%) becomes datetime; unparsed values become blanks and are reported | T `test_rules.py` | Met |
| FR-ING-017 | Recognise true/false columns | P1 | `yes/no`, `y/n`, `true/false` become a boolean column rather than staying text | T `test_rules.py` | Met |
| FR-ING-018 | Treat infinities as missing | P0 | `inf` and `-inf` in a numeric column become blanks, and the repair is reported as a sentence. Infinity is an overflow, not a measurement; every downstream consumer is undefined on it | T `test_scenarios.py::test_infinity_is_treated_as_missing_at_ingestion` | **Met — fixed in this audit** |
| FR-ING-019 | Report every repair as a sentence | P0 | Each cleaning action produces a plain-English line naming the columns and counts involved | T `test_app.py` | Met |
| FR-ING-020 | Store the original bytes | P1 | The upload is written through `StorageBackend` so the same file can be preprocessed again with different options without re-uploading | T `test_api.py` | Met |
| FR-ING-021 | Reject path traversal in filenames | P0 | A filename containing `../` or an absolute path cannot escape the storage root | T `test_api.py` hostile-filename tests; I `storage.py:41` | Met |

### 3.2 Stage 2 — Data profiling

| ID | Requirement | Pri | Acceptance criteria | Verification | Status |
|---|---|---|---|---|---|
| FR-PROF-001 | Report dataset dimensions | P0 | Row and column counts appear in the profile response | T `test_scenarios.py::test_a_clean_classification_file_goes_ingest_to_split` | Met |
| FR-PROF-002 | Per-column missing counts and rates | P0 | Every column reports `n_missing` and `missing_rate` | T `test_api.py` | Met |
| FR-PROF-003 | Per-column cardinality | P0 | Every column reports `n_unique` and `unique_rate` | T `test_api.py` | Met |
| FR-PROF-004 | Duplicate-row detection | P0 | The count of exactly repeated rows is reported | T `test_app.py` | Met |
| FR-PROF-005 | Duplicate-column-name detection | P1 | A heading used twice is disambiguated at ingestion and the mapping reported (FR-ING-012) | T fixture | Met |
| FR-PROF-006 | Outlier indicators | P0 | Each numeric column reports the count and share outside Tukey's fences at `OUTLIER_IQR_MULTIPLIER` (1.5), with the fences themselves | T `test_rules.py` | Met |
| FR-PROF-007 | Numeric summary statistics | P1 | Count, mean, std, min, q1, median, q3, max and skew per numeric column, with a `skewed` flag at \|skew\| > `SKEW_THRESHOLD` (1.0) | T `test_api.py` | Met |
| FR-PROF-008 | Distribution histogram | P1 | A numeric column returns bin edges and counts computed server-side; a categorical column returns value counts | T `test_api.py` | Met |
| FR-PROF-009 | Top values per column | P1 | The `TOP_VALUES` (5) most frequent values with their share, for every column kind | T `test_api.py` | Met |
| FR-PROF-010 | Numeric correlation analysis | P0 | Pairs correlating above `HIGH_CORRELATION` (0.90) are reported, when there are at least `MIN_ROWS_FOR_CORRELATION` (30) rows | T `test_checks.py` | Met |
| FR-PROF-011 | Class-imbalance analysis | P0 | For a classification target, class counts and the majority:minority ratio are reported, flagged above `IMBALANCE_RATIO` (3.0) | T `test_rules.py`; fixture `imbalanced.csv` | Met |
| FR-PROF-012 | A transparent health score, 0–100 | P0 | A weighted mean of six dimensions — completeness .25, validity .20, consistency .15, uniqueness .15, feature quality .15, distribution .10 — each reported with its own evidence and the arithmetic that produced the total | T `test_checks.py`; I `health.py` `DIMENSION_WEIGHTS` | Met |
| FR-PROF-013 | Per-column quality flags | P0 | Short tags — "looks like the answer", "never changes", "too empty to use", "mostly blank", "free text", "date/time" — on the columns they apply to and no others | T `test_api.py` | Met |
| FR-PROF-014 | Handle an all-null column | P0 | Profiling reports 100% missing and does not raise; preprocessing drops the column with that reason | T `test_d_an_all_null_column_is_dropped_rather_than_filled` | Met |
| FR-PROF-015 | Handle a constant or near-constant column | P0 | A column with one value, or one value covering ≥ `NEAR_CONSTANT_SHARE` (99%) of rows, is classified `constant` and reported | T `test_d_columns_that_carry_nothing_are_dropped_with_a_reason` | Met |
| FR-PROF-016 | Handle a non-finite column without raising | P0 | A column containing `inf` profiles successfully, and no non-finite float reaches the JSON response | T `test_profiling_a_non_finite_column_does_not_raise`, `test_a_file_containing_infinity_uploads_and_preprocesses` | **Met — fixed in this audit** |
| FR-PROF-017 | Handle mixed and small data | P1 | A frame mixing numbers, text, dates and booleans profiles without error; statistics that need a minimum row count are skipped rather than computed on too little data | T `test_d_mixed_types_and_high_cardinality_both_survive_the_journey` | Met |
| FR-PROF-018 | Profiling for display never feeds a decision | P0 | `services/profile.py` reads every row, for the dashboard only; the profile that routes columns is computed inside `fit_preprocessing` on the training fold | I `profile.py` module docstring; T `test_leakage.py` | Met |

### 3.3 Stage 3 — Glass-box preprocessing

| ID | Requirement | Pri | Acceptance criteria | Verification | Status |
|---|---|---|---|---|---|
| FR-PREP-001 | Separate features from the target | P0 | The target column never appears in the feature matrix or the feature map | T `test_c_regression_separates_the_target_and_does_not_stratify` | Met |
| FR-PREP-002 | Remove rows with a blank target | P0 | Rows with no value in the target are dropped before the split, for both tasks, and the count is reported as a warning | T `test_rows_with_a_blank_target_are_removed_and_reported` | **Met — fixed in this audit** |
| FR-PREP-003 | Deduplicate before splitting | P0 | Exactly repeated rows are removed before the split, so copies cannot land on opposite sides, and the count is reported | T `test_d_duplicate_rows_are_removed_before_the_split_and_reported` | Met |
| FR-PREP-004 | Split before anything is measured | P0 | The split is the first operation after the stateless repairs. Rewriting every held-out row leaves the plan and the log byte-identical | T `test_e_rewriting_the_held_out_rows_cannot_change_the_plan`, `test_leakage.py` | Met |
| FR-PREP-005 | Stratify a classification split | P0 | Where the label supports it, each class keeps its share within 2 points on both sides | T `test_a_the_split_is_stratified_and_reproducible` | Met |
| FR-PREP-006 | Fall back from stratification audibly | P0 | A single-class target, a class with one row, or more classes than either fold has rows, each disables stratification and produces a `stratification_disabled` warning naming the actual cause. The run completes | T `test_too_many_classes_for_the_fold_falls_back_and_says_why`, `test_a_class_with_one_row_still_disables_stratification_with_a_warning` | **Met — fixed in this audit** |
| FR-PREP-007 | Do not stratify a regression split | P0 | A continuous target produces an unstratified split, and `stratified` is reported as `false` | T `test_c_regression_separates_the_target_and_does_not_stratify` | Met |
| FR-PREP-008 | Reproducible splitting | P0 | The same frame, seed and `test_size` produce the same fold membership, the same transformed matrices and a byte-identical log | T `test_a_the_split_is_stratified_and_reproducible` | Met |
| FR-PREP-009 | Preserve feature/target alignment | P0 | `X_train`/`y_train` and `X_test`/`y_test` share an index, in order; the two folds are disjoint | T `test_a_train_and_test_are_disjoint_and_aligned` | Met |
| FR-PREP-010 | Impute numeric gaps from the training rows | P0 | Blanks are filled with the training fold's median; the logged fill value equals the training median and differs from the whole-file median | T `test_b_fill_values_are_the_training_median_not_the_whole_file_median` | Met |
| FR-PREP-011 | Impute categorical gaps from the training rows | P0 | Blanks are filled with the training fold's most frequent value, which is by construction a category the encoder knows | T `test_b_categorical_gaps_are_filled_with_a_category_the_encoder_knows` | Met |
| FR-PREP-012 | Drop a column that is mostly missing | P0 | A column over `MAX_MISSING_RATE` (60%) blank **in the training fold** is dropped with that reason. The rate is never measured over the whole file | T `test_d_an_all_null_column_is_dropped_rather_than_filled`; `test_codegen.py::test_a_column_is_dropped_on_how_empty_the_training_rows_are` | **Met — fixed in this audit (generated-script path)** |
| FR-PREP-013 | Encode by cardinality | P0 | Two values → ordinal 0/1; up to `ONEHOT_MAX_CARDINALITY` (10) → one-hot; above → frequency (default) or cross-fitted target encoding | T `test_rules.py` | Met |
| FR-PREP-014 | Handle categories unseen in training | P0 | One-hot ignores them; ordinal maps them to −1; frequency maps them to 0.0. No transform raises on a test-only category | T `test_leakage.py`, `test_rules.py` | Met |
| FR-PREP-015 | Default to an encoder that cannot leak | P1 | The default high-cardinality encoder is frequency, which never reads the target. Target encoding is offered and is cross-fitted at `TARGET_ENCODER_CV` (5) folds | I `config.DEFAULT_HIGH_CARD_ENCODER`; T `test_rules.py` | Met |
| FR-PREP-016 | Scale numeric features | P0 | Standard, min-max or robust scaling, with centre and spread learned from the training rows only | T `test_rules.py`, `test_leakage.py` | Met |
| FR-PREP-017 | Derive calendar features | P1 | A datetime column becomes year, month, day, day-of-week and hour, each cyclical part with a sine/cosine pair. Whether a part is emitted is decided from the training rows | T `test_rules.py`; `test_codegen.py::test_a_calendar_part_is_emitted_on_how_the_training_rows_vary` | **Met — fixed in this audit (generated-script path)** |
| FR-PREP-018 | Drop columns that carry nothing | P0 | Constant, near-constant and free-text/identifier columns are dropped, each with a specific reason | T `test_d_columns_that_carry_nothing_are_dropped_with_a_reason` | Met |
| FR-PREP-019 | Screen for identifiers on the training fold | P0 | A column unique above `IDENTIFIER_UNIQUE_RATE` (0.95), or above `NAMED_IDENTIFIER_UNIQUE_RATE` (0.90) when its name says "id", is excluded with a reason | T `test_checks.py` | Met |
| FR-PREP-020 | Screen for target leakage on the training fold | P0 | A feature correlating above `LEAKAGE_CORRELATION` (0.99) with the target, or predicting it at or above `LEAKAGE_TREE_SCORE` (0.98) with a depth-`LEAKAGE_TREE_DEPTH` tree in cross-validation, is excluded and flagged | T `test_e_a_column_derived_from_the_answer_is_flagged_and_excluded` | Met |
| FR-PREP-021 | State leakage findings as suspicion | P0 | A leakage message describes what was observed and what it suggests. It does not assert leakage as fact — a correlation is evidence, not proof | T `test_e_leakage_warnings_are_phrased_as_suspicion_not_proof` | Met |
| FR-PREP-022 | Handle non-finite feature values | P0 | `inf` in a feature does not reach an estimator; it is read as missing and filled, with a warning. The caller's frame is not modified | T `test_fit_preprocessing_blanks_infinity_without_touching_the_callers_frame` | **Met — fixed in this audit** |
| FR-PREP-023 | Do not modify the caller's data | P0 | `fit_preprocessing` leaves the frame it was given byte-identical | T `test_fit_preprocessing_blanks_infinity_without_touching_the_callers_frame` | Met |
| FR-PREP-024 | Fit on training rows only | P0 | A spy on every `fit` records the row indices it was given; the intersection with the held-out rows is empty | T `test_leakage.py::test_no_transformer_ever_sees_a_test_row` | Met |
| FR-PREP-025 | Transform the test fold with the frozen pipeline | P0 | `X_test` has the same columns as `X_train`, in the same order, with no blanks | T `test_b_test_rows_are_transformed_never_fitted` | Met |
| FR-PREP-026 | Preserve the column mapping | P0 | Every output column maps back to the input column it came from; an output column with no source is an error, not a warning | T `test_d_every_output_column_can_be_traced_back_to_an_input`; I `build_feature_map` raises | Met |
| FR-PREP-027 | Produce a derived preprocessing summary | P0 | Each log entry is built from a fitted object's learned attributes, carries `fitted_on == "train_fold_only"`, and is JSON-serialisable | T `test_glassbox.py`, `test_b_test_rows_are_transformed_never_fitted` | Met |
| FR-PREP-028 | Reject an invalid configuration clearly | P0 | An unknown scaler, encoder or task type raises a `ValueError` naming the valid values; the API returns 400 with a sentence | T `test_glassbox.py`; I `build_plan`, `_is_classification` | Met |
| FR-PREP-029 | Fail clearly when nothing survives | P0 | If every column is dropped, the error names the reason rather than failing inside sklearn | I `fit_preprocessing`, `build_pipeline` | Met |
| FR-PREP-030 | Check the split after making it | P1 | Per-feature KS drift at `DRIFT_P_VALUE` (0.01), class balance in both folds, and a cross-validated sanity baseline against a dummy, are computed and returned | T `test_checks.py`, `test_the_preprocess_response_carries_the_warnings_and_the_checks` | Met in the API; not rendered in the UI |
| FR-PREP-031 | Export runnable code | P0 | The generated script reproduces the run — same columns, same transformers, same seed, same ordering — and runs standalone | T `test_codegen.py` (every option combination is executed) | Met |
| FR-PREP-032 | Execute user-edited code safely | P0 | User code runs with no network, no writes outside a scratch directory, and CPU, file-size, process and memory limits | T `test_executor.py` (adversarial) | Met on macOS; OS layer unavailable on platforms without `sandbox-exec`, and the active layer is reported |

---

### 3.4 Plan and run

Added in the remediation round of 30 October 2026, closing the two structural gaps.
Design in [Design.md](Design.md) §12; verification in [AUDIT.md](AUDIT.md) §8.3.

| ID | Requirement | Pri | Acceptance criteria | Verification | Status |
|---|---|---|---|---|---|
| FR-PLAN-001 | Propose a per-column plan | P0 | Every column of the dataset gets an action and a reason that quotes the numbers behind it | T `test_plan.py` | Met |
| FR-PLAN-002 | The plan is exhaustive | P0 | Columns that are kept unchanged, dropped, and passed through all appear | T `test_every_column_gets_a_decision_and_a_reason` | Met |
| FR-PLAN-003 | Decisions are measured on the training fold | P0 | The rows a column's statistics describe are the rows the pipeline is fitted on | T `test_preparation_is_shared_so_both_sides_split_the_same_rows` | Met |
| FR-PLAN-004 | Each column offers only valid alternatives | P0 | The actions offered for a column are those its kind permits; an invalid one is refused with a sentence naming the valid set | T `test_an_override_is_validated_against_the_columns_kind` | Met |
| FR-PLAN-005 | A plan is versioned and immutable | P0 | Editing writes the next version; the edited version is byte-identical afterwards | T `test_editing_writes_the_next_version_and_leaves_the_first_alone` | Met |
| FR-PLAN-006 | A plan serialises and reconstructs | P0 | The stored document rebuilds into an executable plan | T `test_the_stored_document_is_a_plan_that_can_be_rebuilt` | Met |
| FR-PLAN-007 | A plan is validated before it is stored | P0 | An invalid edit is a 400 and writes no version | T `test_an_invalid_edit_is_refused_before_anything_is_stored` | Met |
| FR-PLAN-008 | The review screen shows the decision, the reason and the evidence | P0 | Per column: action dropdown, reason, the statistics it was measured from, and the distribution | M browser, against a live server | Met — **no automated guard** (§2.3) |
| FR-RUN-001 | A run executes exactly one plan version | P0 | Nothing is re-decided; the run names the version it ran | T `test_a_run_records_everything_needed_to_describe_itself` | Met |
| FR-RUN-002 | A run is independently retrievable | P0 | `GET /api/runs/{id}` returns it without reference to the plan | T `test_run.py` | Met |
| FR-RUN-003 | Editing a plan never changes an earlier run | P0 | The run's version, hash and feature count are unchanged after the plan is edited | T `test_editing_a_plan_does_not_change_an_earlier_runs_record` | Met |
| FR-RUN-004 | A run persists what reproduces it | P0 | `plan.json`, `schema.json`, `log.json`, `pipeline.joblib`, `train.parquet`, `test.parquet`, `pipeline.py` | T 7 parametrised artifact tests | Met |
| FR-RUN-005 | The schema supports later inference | P1 | Input columns with kind, dtype, action, required flag and fitted categories; output feature names | T `test_the_schema_artifact_describes_what_inference_would_need` | Met |
| FR-RUN-006 | A failed run is never a successful one | P0 | `status` is `failed`, there are no artifacts, no hash, and the error carries no traceback | T 3 failure tests | Met |
| FR-RUN-007 | Metadata and artifacts stay consistent | P0 | A row never names an artifact that is not stored | T `test_a_row_never_names_an_artifact_that_is_not_stored` | Met |
| FR-RUN-008 | Artifact paths cannot be escaped | P0 | The artifact name is looked up, never interpolated into a path | T `test_an_unknown_artifact_name_cannot_name_a_path` | Met |
| FR-RUN-009 | A run is reproducible | P0 | The same plan on the same data produces the same output hash | T `test_the_same_plan_twice_gives_the_same_output_hash` | Met |
| FR-RUN-010 | Both execution paths agree | P0 | The generated script reproduces the engine's output hash exactly, for every supported action | T `test_parity.py`, 39 tests | Met |
| FR-DATA-001 | A dataset can be deleted | P1 | The dataset, its profile, plans, runs, workspace and stored bytes all go; other datasets are untouched | T 8 tests in `test_lifecycle.py` | Met |
| FR-DATA-002 | Deletion fails safely | P0 | A storage failure strands bytes and reports them; it never leaves rows pointing at data that is gone | T `test_a_storage_failure_still_leaves_the_database_consistent` | Met |

## 4. Non-functional requirements

| ID | Area | Requirement | Verification | Status |
|---|---|---|---|---|
| NFR-001 | Correctness | No held-out row reaches a fit or a decision in either preprocessing path | `test_leakage.py`, `test_scenarios.py`, `test_codegen.py` | Met |
| NFR-002 | Reproducibility | Dependencies pinned exactly; one seed (`SEED`, 42) is the only source of randomness; the split parameters used are recorded on the result and returned | `requirements.txt`; `test_a_the_split_is_stratified_and_reproducible` | Met |
| NFR-003 | Security — upload | Size capped while streaming; content sniffed by magic bytes; filenames cannot traverse the storage root | `test_api.py`, Scenario F | Met |
| NFR-004 | Security — execution | User code is sandboxed in layers, and the attacks are tested rather than assumed | `test_executor.py` | Met on macOS (NFR caveat in §2.2) |
| NFR-005 | Security — deserialisation | No `pickle.load` of user input. `joblib` is used to write pipelines the app itself produced, and those artefacts are never loaded from a user-supplied path | Inspection — see AUDIT.md §5.6 | Met |
| NFR-006 | Security — secrets | No credentials in source control; the child process environment is scrubbed before user code runs | `executor.py`; `git` inspection | Met |
| NFR-007 | Error handling | A user-fixable problem is an HTTP 400 with `{"error": "<sentence>"}`. No traceback, file path or internal identifier reaches the client | Scenario F | Met |
| NFR-008 | Observability | Every decision the engine takes is either a log entry or a warning record; nothing falls back silently | `test_the_preprocess_response_carries_the_warnings_and_the_checks` | Met at the API boundary |
| NFR-009 | Privacy | Everything runs locally. Uploads and artefacts are written under `STORAGE_ROOT`; no dataset content is sent anywhere, and no row values are logged | Inspection; `config.py` | Met |
| NFR-010 | Temporary files | Sandbox work directories are removed after execution; the test suite uses a temporary database and storage root and deletes them at session end | `executor.py`, `conftest.py` | Met |
| NFR-011 | Maintainability | Every threshold lives in `app/config.py` with the reasoning for its value, and is overridable by an `MLBENCHX_`-prefixed environment variable | `config.py` | Met |
| NFR-012 | Testability | Routing decisions are pure functions of a column profile and are testable without any data | `build_plan` signature; `test_glassbox.py` | Met |
| NFR-013 | Coverage | ≥ 80% statement and branch coverage of `app/`, enforced in CI configuration | `pytest.ini` `--cov-fail-under=80`; measured 89.8% | Met |
| NFR-014 | Performance | A 10 MB file completes ingestion, profiling and preprocessing within an interactive wait | `tests/test_performance.py` — budgets scaled by a calibration workload, plus a machine-independent scaling check | **Met.** Measured 1.92 s total against an 8 s budget |
| NFR-015 | Memory | The upload cap bounds the frame size; the sandbox has a memory watchdog at `EXECUTION_MEMORY_BYTES` (2 GB) | `executor.py`; `test_executor.py` | Met for executed code; no measured ceiling for the API process |
| NFR-016 | Accessibility | `prefers-reduced-motion` is honoured; the layout is verified at 375, 768 and 1440 px with no horizontal overflow | Manual, per README | Claimed; not automatically tested |
| NFR-017 | Browser reachability | Every HTTP method the API routes must survive a CORS preflight from the development origin, and no other origin may be allowed. The interface and the API are different origins in development, so a method the policy omits is not a stricter policy — it is a route the product cannot use | `test_api.py::test_every_method_the_api_uses_survives_a_preflight` (parametrised from the OpenAPI document), `test_an_origin_the_app_does_not_serve_is_still_refused` | **Met — fixed in this audit.** `PUT` was omitted while two workspace routes used it |

---

## 5. User workflow

### 5.1 The journey as the interface offers it

1. **Upload** — drag a file onto the upload screen. *(UI: `UploadScreen.tsx`)*
2. **Read** — the file is decoded, repaired and profiled. The user is shown what was
   repaired, the health score, the six dimensions behind it, and the per-column table.
   *(UI: `InsightsPanel.tsx`, `BottomDock.tsx`)*
3. **Review the pipeline** — eight steps, each a switch, each card stating what the step
   will do *to this dataset*. Turning one off opens a dialog that states the measured
   consequence, including the health score both ways. *(UI: `PipelinePanel.tsx`)*
4. **Read or edit the code** — the generated pipeline in an editor, with a diff against
   the generated version. Once edited by hand, the edit is what runs, and the interface
   says so in three places. *(UI: `CodePanel.tsx`)*
5. **Compile** — the pipeline runs in the sandbox. The processed data, the run log and
   the before/after health comparison come back. *(UI: `Workspace.tsx`)*
6. **Download** — processed data, original data, code, config, report or the whole
   project.

### 5.2 Backend-only today

These are implemented and reachable over HTTP but have no caller in the interface:

* `POST /api/datasets/{id}/preprocess` — the glass-box engine path, including the
  warnings and post-split checks (§2.3). The workspace compiles the generated script
  instead.
* `POST /api/analyse` and `POST /api/example` — upload-and-preprocess in one call.
* The target can be set and the task detected, but there is no target picker in the UI,
  and the detected task is not displayed.

### 5.3 Future UI work

A plan review screen with per-column overrides; the warnings and post-split checks
rendered where the user can act on them; a histogram on column click (`recharts` is
installed and unused); a run history that selects between runs rather than showing an
edit trail.

---

## 6. Architecture and data flow

**The technical design is [Design.md](Design.md).** It holds the architecture diagram,
the stage-by-stage data flow, the module responsibilities, the persistence schema, the
transformer-level design of Stages 1–3, and the reasoning behind each decision. This
section states only what a requirements reader needs in order to place a requirement.

ML-BenchX is a local two-tier application — a FastAPI backend and a React single-page
frontend, with SQLite for metadata and the local filesystem for bytes. There is no
queue, no broker and no external service.

| Stage | Owned by | Requirements |
|---|---|---|
| 1 — Ingestion | `backend/app/services/ingest.py` | FR-ING-001 … 021 |
| 2 — Profiling | `backend/app/services/profile.py`, `health.py` | FR-PROF-001 … 018 |
| 3 — Preprocessing | `backend/app/services/glassbox.py`, `screening.py`, `checks.py`, and `codegen.py` for the generated-script path | FR-PREP-001 … 032 |
| — Target detection | `backend/app/services/target.py` | supports FR-PREP-001 |
| — API surface | `backend/app/main.py`, `workspace.py`, `schemas.py` | §7, NFR-007, NFR-017 |
| — Code execution | `backend/app/executor.py` | FR-PREP-032, NFR-004 |

Two boundaries are themselves requirements, so they are stated here rather than only in
the design:

* **Stage 2's output never reaches Stage 3's decisions** (FR-PROF-018). Two profiles
  exist and answer different questions — one describes the file the user uploaded,
  reading every row; the other runs inside the fit on the training fold alone.
* **The split precedes every measurement and decision, not just every fit**
  (FR-PREP-004, FR-PREP-012, FR-PREP-017). Design.md §7 explains how this is enforced;
  AUDIT.md §4 is the evidence that it holds.

Error propagation, the data structures passed between stages, the database schema and
the temporary-data lifecycle are in Design.md §5.2, §5.4, §6 and §8.6.


## 7. Data contracts and API

Base URL `/api`. Interactive documentation at `/api/docs`; the OpenAPI document is
generated from `app/schemas.py`, which is the contract.

### 7.1 Dataset and run routes (`main.py`)

| Method | Path | Request | Success | Errors |
|---|---|---|---|---|
| `POST` | `/datasets` | `multipart/form-data`, field `file` | `201` `DatasetSummary` | `400` unreadable, too large, too few rows, wrong content |
| `GET` | `/datasets` | — | `200` `DatasetSummary[]`, newest first | — |
| `GET` | `/datasets/{id}` | — | `200` `DatasetDetail` with its runs | `404` |
| `GET` | `/datasets/{id}/profile` | — | `200` `ProfileResponse` | `404` |
| `POST` | `/datasets/{id}/preprocess` | `PreprocessRequest` (optional) | `200` `AnalysisResponse` | `400` unusable data · `404` |
| `GET` | `/datasets/{id}/preprocess/log` | `?run_id=` | `200` `LogResponse` | `404` |
| `GET` | `/runs/{id}` | — | `200` the run's stored payload | `404` |
| `GET` | `/runs/{id}/data.csv` | — | `200` `text/csv`, the prepared training data | `404` |
| `GET` | `/runs/{id}/script.py` | — | `200` `text/plain`, runnable Python | `404` |
| `POST` | `/analyse` | multipart file + options | `200` `AnalysisResponse` | `400` |
| `POST` | `/example` | options | `200` `AnalysisResponse` on the bundled sample | — |
| `GET` | `/health` | — | `200` liveness | — |

**`PreprocessRequest`** — `goal: str` (plain English), `target: str | None`,
`test_size: float` (0.05 < x < 0.5, default 0.2), `scaler: "standard" | "minmax" |
"robust"`, `high_card_encoder: "frequency" | "target"`.

**`AnalysisResponse`** — `run_id`, `dataset_id`, `file`, `goal`, `target`, `columns`,
`cleaning[]`, `quality`, `profile`, `options`, `dropped[]`, `steps[]`, `result`,
`warnings[]`, `checks`, `took_ms`.

* `result` — `train_rows`, `test_rows`, `features`, `stratified`, `test_size`, `seed`.
* `warnings[]` — `{kind, severity: "info"|"warn"|"critical", message, suggestion, …}`.
  Kinds emitted today: `blank_targets_removed`, `duplicates_removed`,
  `non_finite_values`, `stratification_disabled`, `train_test_drift`, `split_balance`,
  `weak_baseline`.
* `checks` — `{drift, balance, baseline}`; a permissive document, rendered rather than
  branched on.

### 7.2 Workspace routes (`workspace.py`, prefix `/api/workspaces`)

| Method | Path | Does |
|---|---|---|
| `POST` | `/from-dataset/{id}` | Open the workspace for a dataset (idempotent) · `201` |
| `GET` | `/{id}` | The whole workspace state |
| `GET` | `/{id}/health` | The six dimensions of the dataset as uploaded |
| `PUT` | `/{id}/config` | Replace the pipeline configuration |
| `PUT` | `/{id}/code` | Save the editor contents |
| `POST` | `/{id}/code/reset` | Discard edits, regenerate from the config |
| `POST` | `/{id}/impact` | What a toggle would cost, measured both ways |
| `POST` | `/{id}/compile` | Run the pipeline in the sandbox |
| `GET` | `/{id}/compilations` | Past compilations |
| `GET` | `/{id}/preview` | A page of rows, blanks and outliers marked |
| `GET` | `/{id}/history` | The change history |
| `POST` | `/{id}/history/{entry_id}/restore` | Go back to an earlier state |
| `GET` | `/{id}/download/{what}` | `processed` · `original` · `code` · `config` · `report` · `project` |

Every mutating workspace route returns the **whole** workspace, so the client never
reassembles state from a patch.

### 7.3 Error shape

```json
{ "error": "That file has 30 rows. We need at least 50 to set some aside for honest testing." }
```

Emitted with `400` for everything the user can fix. `404` uses FastAPI's `{"detail": …}`.
The Phase 1 specification asked for a uniform `{code, message, detail}` envelope. Both
are now sent: `error` keeps its original meaning for existing clients, and the three
specified fields are added beside it (§7.3.1). See [AUDIT.md](AUDIT.md) §8.2 O6.

### 7.3.1 CORS

The interface and the API are separate origins in development (`:5173` and `:8000`), so
every request that is not a simple one is preceded by a preflight. The policy allows
`GET`, `POST` and `PUT` — the three the routes above actually use — from the four
localhost origins Vite serves on, and nothing else. `DELETE` is deliberately absent:
there is no route to permit (§7.5).

`PUT` was missing until this audit, which meant the browser refused to send the two
requests that save a pipeline configuration and save edited code. `TestClient` calls the
application directly and never performs a preflight, so no test saw it; the test that
now holds this derives the required methods from the OpenAPI document rather than from a
hand-maintained list.

### 7.4 Configuration

Every threshold is in `app/config.py` with its reasoning, and each is overridable by an
environment variable of the same name prefixed `MLBENCHX_`:

```bash
MLBENCHX_MAX_UPLOAD_BYTES=104857600 python -m uvicorn app.main:app
```

Key values: `MAX_UPLOAD_BYTES` 50 MB · `MIN_ROWS` 50 · `TEST_SIZE` 0.2 · `SEED` 42 ·
`MAX_MISSING_RATE` 0.60 · `ONEHOT_MAX_CARDINALITY` 10 · `NEAR_CONSTANT_SHARE` 0.99 ·
`LEAKAGE_CORRELATION` 0.99 · `DRIFT_P_VALUE` 0.01 · `EXECUTION_TIMEOUT_SECONDS` 120.

### 7.5 Proposed, not implemented

Clearly marked as proposals — none of these exist today:

* `DELETE /api/datasets/{id}`
* `POST /api/datasets/{id}/plan` — propose without executing
* `GET /api/runs/{id}/artifacts/{name}` — the joblib pipeline is not downloadable
* A uniform `{code, message, detail}` error envelope

---

## 8. Testing and acceptance

### 8.1 Suite

381 tests in `backend/tests/`, run against a temporary database and storage root created
and destroyed by `conftest.py`.

| File | Covers | Level |
|---|---|---|
| `test_app.py` | Reading files, guessing the target, the single-call endpoints | Integration |
| `test_api.py` | Every dataset route end to end, including bad input, hostile filenames and the CORS policy | Integration |
| `test_workspace.py` | The workspace API as the interface drives it | Integration |
| `test_glassbox.py` | The engine: profiling, routing, the feature map, the four invariants | Unit |
| `test_rules.py` | One test per rule — each scaler, each encoder, dates, outliers, task detection | Unit |
| `test_checks.py` | Drift, split balance, sanity baseline, correlation, screening | Unit |
| `test_leakage.py` | That held-out rows reach neither a fit nor a decision | Regression |
| `test_codegen.py` | Every option combination, by *running* the generated script | Integration |
| `test_source_format.py` | Each accepted file format and its repairs | Unit |
| `test_executor.py` | Real attacks on the sandbox — network, filesystem, fork and memory bombs | Security |
| `test_scenarios.py` | Scenarios A–F end to end, and every defect this audit fixed | E2E / regression |

### 8.2 Acceptance scenarios

| | Scenario | Covered by |
|---|---|---|
| **A** | Clean classification: ingest → profile → target → preprocess → reproducible stratified split | `test_a_*` (3 tests) |
| **B** | Missing values: imputation fitted on training rows only; test transformed with those statistics | `test_b_*` (3 tests) |
| **C** | Regression: target separated, unstratified split, alignment preserved | `test_c_*` (2 tests) |
| **D** | Data quality: duplicates, constants, all-null, mixed types, high cardinality, traceability | `test_d_*` (5 tests) |
| **E** | Leakage and split integrity: flagged and excluded, phrased as suspicion, plan invariant to rewriting the held-out rows | `test_e_*` (3 tests) |
| **F** | Invalid uploads: empty, header-only, single-column, wrong type, renamed binary, malformed, below the row floor | `test_f_*` (3 tests, 5 parametrised cases) |

### 8.3 Edge cases, and where each is covered

| Edge case | Covered by |
|---|---|
| Empty dataset / header only | `test_f_bad_uploads_are_refused_with_a_sentence` |
| Missing target values | `test_rows_with_a_blank_target_are_removed_and_reported` (both tasks) |
| Target entirely blank | `test_a_target_that_is_entirely_blank_fails_with_a_sentence` |
| All-null column | `test_d_an_all_null_column_is_dropped_rather_than_filled` |
| Duplicate rows | `test_d_duplicate_rows_are_removed_before_the_split_and_reported` |
| Duplicate column names | fixture `unicode_duplicate_headers.csv` |
| Mixed data types | `test_d_mixed_types_and_high_cardinality_both_survive_the_journey` |
| High-cardinality categoricals | same, plus `test_rules.py` encoder tests |
| Constant / near-constant columns | `test_d_columns_that_carry_nothing_are_dropped_with_a_reason` |
| Rare classes / insufficient class counts | `test_a_class_with_one_row_…`, `test_too_many_classes_for_the_fold_…` |
| Datetime features | `test_rules.py`, `test_a_calendar_part_is_emitted_on_how_the_training_rows_vary` |
| Infinite / non-finite values | 4 tests across ingestion, profiling, the engine and the generated script |
| Train/test leakage | `test_leakage.py` (3 forms), `test_e_*`, 2 codegen decision tests |
| Unsupported file encodings | fixture `latin1.csv`; `test_source_format.py` |
| Oversized or malformed files | `test_api.py`, `test_f_a_malformed_csv_…` |
| Reproducibility with a fixed seed | `test_a_the_split_is_stratified_and_reproducible` |

### 8.4 Commands

```bash
cd backend && ../.venv/bin/python -m pytest tests/ -q     # 381 tests, coverage gate at 80%
cd frontend && npm run build                              # tsc -b && vite build
cd frontend && npm run lint                               # oxlint
cd backend && ../.venv/bin/python -m alembic upgrade head # schema from empty
```

### 8.5 Gaps in verification

* No asserted performance budget (NFR-014).
* No browser-level end-to-end test; the UI is verified by hand.
* The sandbox's non-macOS path is not exercised on this machine.
* Frontend type and lint checks run; there are no frontend unit tests.

---

## 9. Risks and dependencies

| # | Risk | Evidence | Impact | Mitigation |
|---|---|---|---|---|
| R1 | **Two implementations of one pipeline.** `glassbox.py` and `codegen.py` both implement Stage 3 | Both exist and are separately tested; this audit found the same class of decision-leak in only one of them | A fix applied to one path silently leaves the other wrong | Both are now tested for the same invariants. Converging them is the structural fix, and is a design decision rather than a patch |
| R2 | **Numerical behaviour is pinned to exact library versions** | `requirements.txt` pins `pandas==2.3.3`, `scikit-learn==1.8.0` and the rest exactly, with the reasoning stated | A minor release can move a quantile or a cross-fitting detail and falsify the determinism claim | Exact pins; the determinism test fails loudly if it moves |
| R3 | **Dead code carrying real features.** `app/plan.py` and `app/runner.py` implement missing-indicator columns and rare-category grouping, and nothing imports them | `.coveragerc` omits both and documents why | A reader believes those features exist | Documented in the audit; wire in or delete — a direction decision, not a defect |
| R4 | **The UI does not show what the engine detects** | `warnings`/`checks` now returned; no frontend caller | A drifted split or a weak baseline is computed and never seen | API contract in place; rendering is tracked as a P2 gap |
| R5 | **Single-process, synchronous compile** | By design, per README | A large file blocks a request | The upload cap bounds it; a queue is deliberately out of scope |
| R6 | **Sandbox strength is platform-dependent** | `sandbox-exec` is macOS-only | On Linux only process limits apply | The active layer is reported in the UI; the production answer is a container with no network namespace |
| R7 | **No authentication; `project_id` is always `"default"`** | `db.py` | Not multi-user | Deliberate. The column exists so accounts are a migration, not a redesign |
| R8 | **`12%` ingests as `12`, not `0.12`** | `ingest._NUMERIC_NOISE` strips `%` | A percentage column is scaled 100× | Documented here and in the audit; changing it silently rescales existing users' data |

**Assumptions.** Single user, local machine, files under 50 MB, tabular data with a
header row, and a target the user can name or describe.

---

## 10. Future roadmap — Stages 4–7

**None of the following is implemented.** No Celery, no PostgreSQL experiment history,
no SHAP, no model training and no leaderboard exists in this repository. The
architecture diagram shows the intended product, not the current one.

| Stage | Capability | Depends on |
|---|---|---|
| **4 — Meta-learning** | Extract dataset meta-features (rows, sparsity, feature types, imbalance); compare against historical experiments by cosine similarity / KNN; rank algorithms with a plain-English rationale | A persisted experiment history, which needs the versioned run object (§2.4) and `schema.json` |
| **5 — Parallel training** | Train XGBoost, Random Forest, Logistic Regression, SVM (RBF), KNN and Decision Tree on Celery workers | A job queue and a broker; a fitted-pipeline artefact that is loadable by a worker |
| **6 — Explainability** | SHAP values on the top model, beeswarm, bar and waterfall plots; F1, ROC-AUC, RMSE | Stage 5, and the feature map this release already produces |
| **7 — Leaderboard & report** | Model rankings, experiment history, downloadable PDF | Stages 4–6 |

The sequencing constraint is that Stage 4 needs a durable, versioned run before it has
anything to learn from. That object is the first item of Stage 4 work, not the last.

---

## 11. Definition of done — Stages 1–3

This release is complete when all of the following hold. Each is checkable, and the
current state is given.

| # | Condition | State |
|---|---|---|
| 1 | Every P0 requirement in §3 is **Met** | ✅ all 91 requirements met — 74 P0, 17 P1 — five carrying the qualifications stated in their rows |
| 2 | The full suite passes with no skips masking a failure | ✅ 506 passed |
| 3 | Coverage of `app/` ≥ 80%, enforced by the test command | ✅ 89.8%, now including `plan.py` and `runner.py` |
| 4 | No held-out row reaches a fit **or a decision**, in *either* preprocessing path, proven by a test that fails against the prior implementation | ✅ verified both ways |
| 5 | Identical input and seed produce an identical plan, log and output | ✅ |
| 6 | Every failure a user can cause produces a sentence, not a traceback | ✅ Scenario F |
| 7 | The frontend type-checks, lints and builds | ✅ `tsc -b && vite build`, `oxlint` clean |
| 8 | Migrations run from an empty database | ✅ 3 revisions, and from the previous schema with existing data intact |
| 9 | Dependencies pinned exactly, with the reason recorded | ✅ |
| 10 | Every condition the engine detects is reachable by a caller | ✅ at the API, and rendered on the plan review screen |
| 11 | A performance budget is asserted for a 10 MB file | ✅ machine-calibrated, plus a scaling check (NFR-014) |
| 12 | `AUDIT.md` reflects the code as it stands | ✅ |

**All twelve conditions are met.** The two that were outstanding in version 1.0 — the UI
rendering of engine warnings, and the performance budget — were closed in the
remediation round recorded in AUDIT.md §8.

What is *not* claimed: there is no automated frontend test (§2.3), so FR-PLAN-008 rests
on manual verification against a live server rather than on a guard that would catch a
regression.
