# ML-BenchX — Phase 1 audit

Audited 2026-09-19 against the Phase 1 spec, remediated 2026-09-20.

**Final status: every spec item is implemented.** 137 tests pass, the backend starts,
the frontend builds, and an end-to-end run against `tests/fixtures/titanic.csv`
completes over HTTP. Work landed in six commits, one per area.

Legend: ✅ done · ⚠️ partial · ❌ absent. The **Was** column is the state at audit time;
**Now** is the state after remediation.

## Spec coverage

| Area | Spec item | Was | Now | Where |
|---|---|---|---|---|
| Ingest | CSV/XLSX read | ✅ | ✅ | `ingest.py:278` |
| Ingest | Encoding check | ✅ | ✅ | `ingest.py:97` |
| Ingest | Size cap, configurable, default 50 MB | ⚠️ | ✅ | `config.py:MAX_UPLOAD_BYTES` |
| Ingest | Cap enforced before body is buffered | ❌ | ✅ | `main.py` `_reject_oversized_bodies` |
| Ingest | Infer numeric/categorical/datetime/ID-like/constant | ⚠️ | ✅ | `glassbox.py` `_classify`, `screening.py` |
| Profile | Per-column missing %, cardinality | ⚠️ | ✅ | `services/profile.py` |
| Profile | IQR outliers per column | ❌ | ✅ | `profile.py` `outlier_summary` |
| Profile | Duplicate rows | ✅ | ✅ | `ingest.py` |
| Profile | High-corr pairs \|r\|>0.9 | ❌ | ✅ | `screening.py` `correlated_pairs` |
| Profile | Class imbalance | ⚠️ | ✅ | `target.py` `class_balance` |
| Profile | Health score 0–100, documented weighted formula | ⚠️ | ✅ | `config.py:HEALTH_WEIGHTS`, README |
| Task | binary / multiclass / regression | ⚠️ | ✅ | `target.py` `describe_task` |
| Prep | Split 80/20 first, stratified | ✅ | ✅ | `glassbox.py` `fit_preprocessing` |
| Prep | Fit on train only, transform both | ✅ | ✅ | `glassbox.py`, `tests/test_leakage.py` |
| Prep | Routing decided from train only | ⚠️ | ✅ | `fit_preprocessing` steps 2–3 |
| Prep | sklearn Pipeline / ColumnTransformer | ✅ | ✅ | `glassbox.py` `build_pipeline` |
| Prep | Persist fitted pipeline (joblib) | ❌ | ✅ | `main.py` `_persist_run_artifacts` |
| Prep | Impute median (num) / mode (cat) | ✅ | ✅ | `glassbox.py` `_block` |
| Prep | Binary → label encode | ✅ | ✅ | `glassbox.py` `_encoder_spec` |
| Prep | Low-card (≤10, configurable) → one-hot, `handle_unknown="ignore"` | ⚠️ | ✅ | `config.py:ONEHOT_MAX_CARDINALITY` |
| Prep | High-card → cross-fitted TargetEncoder | ⚠️ | ✅ | `glassbox.py:HIGH_CARD_ENCODERS` |
| Prep | Scale: Standard default, MinMax/Robust selectable | ⚠️ | ✅ | `glassbox.py:SCALERS` |
| Prep | Datetime → y/m/dow/hour + sin/cos | ❌ | ✅ | `glassbox.py` `DatetimeFeatures` |
| Prep | Leakage flag \|r\|>0.99 with target | ❌ | ✅ | `screening.py` `target_leakage_screen` |
| Prep | Drop constant + ID-like with logged reason | ✅ | ✅ | `glassbox.py` `build_plan` |
| Log | JSON record per step | ⚠️ | ✅ | `glassbox.py` `introspect` |
| API | `POST /datasets` | ❌ | ✅ | `main.py` `create_dataset` |
| API | `GET /datasets/{id}` | ❌ | ✅ | `main.py` `get_dataset` |
| API | `GET /datasets/{id}/profile` | ❌ | ✅ | `main.py` `get_profile` |
| API | `POST /datasets/{id}/preprocess` (target, scaler, options) | ❌ | ✅ | `main.py` `preprocess_dataset` |
| API | `GET /datasets/{id}/preprocess/log` | ❌ | ✅ | `main.py` `get_preprocess_log` |
| API | Pydantic schemas | ❌ | ✅ | `app/schemas.py` |
| API | Correct 4xx errors | ✅ | ✅ | `main.py`, `tests/test_api.py` |
| DB | SQLAlchemy 2.x + Alembic on SQLite | ❌ | ✅ | `app/db.py`, `migrations/` |
| DB | `datasets`, `profiles`, `preprocessing_runs`, `preprocessing_logs` | ❌ | ✅ | `app/db.py` |
| DB | `project_id` + `created_at` on every table | ❌ | ✅ | `db.py` `TimestampedProjectMixin` |
| Storage | `StorageBackend` interface + `LocalStorage` | ❌ | ✅ | `app/storage.py` |
| UI | Upload page | ✅ | ✅ | `Upload.tsx` |
| UI | Profile dashboard: health score | ✅ | ✅ | `Results.tsx`, `HealthBreakdown` |
| UI | Profile dashboard: per-column flag table | ❌ | ✅ | `ColumnTable.tsx` |
| UI | Glass-box log, card per step, reason + before→after | ✅ | ✅ | `StepCard.tsx`, `TransformStrip.tsx` |
| Config | All thresholds in one module | ❌ | ✅ | `app/config.py` |
| Tests | One per preprocessing rule | ✅ | ✅ | `tests/test_rules.py` |
| Tests | Leakage: test-split stats never reach fit | ✅ | ✅ | `tests/test_leakage.py` |
| Tests | API happy path + bad file | ✅ | ✅ | `tests/test_api.py` |
| Tests | `tests/fixtures` dataset | ❌ | ✅ | `tests/fixtures/titanic.csv` |

## Bugs found and fixed

1. **`describe_task` mislabelled categorical targets** — a text target with 21–50
   distinct values returned `"regression"`, which **disabled stratification** on real
   multiclass problems. Only numeric columns can be regressions now.
   *Fixed in `target.py`; covered by a parametrised test including the exact case.*
2. **Upload size cap applied after buffering** — `await file.read()` materialised the
   whole body before the check. Now refused from `Content-Length` in middleware, with a
   streaming cap as backstop. *Verified: a 51 MB POST returns 413.*
3. **`Content-Disposition` built from the raw client filename** — quotes and CRLF
   survived `Path(name).stem`. Now sanitised to a known-good character set.
   *Covered by a test that uploads `evil";\r\nX-Injected: yes.csv` and asserts no
   injected header.*
4. **`_RUNS` held 20 full DataFrames** as the only copy of a run. Now a cache in front
   of the database and storage; a miss reads back the stored artefacts.
5. **Exported loader drifted from the real ingest** — still true for duplicate-header
   disambiguation; see *Known limitations* below.
6. **A boolean column crashed the entire run** — found while writing the new tests, not
   during the audit. `SimpleImputer` rejects `bool` dtype, so any file with a real
   `True`/`False` column failed with a dtype error. *Fixed by `CategoricalCast`, which
   also handles pandas nullable extension dtypes.*
7. **Dates were never parsed from uploads** — found during end-to-end verification. A
   CSV carries no dtypes, so the datetime feature work was unreachable from the actual
   upload path and date columns were one-hot encoded into a column per timestamp.
   *Fixed in `ingest.py`, gated on a four-digit year so version numbers and scores are
   not turned into 1970 timestamps.*
8. **Unreachable branch in the identifier screen** — required `unique_rate == 1.0`,
   which the preceding `>= 0.98` branch always caught first. *Removed.*

## Leakage risks — resolved

1. **Routing decisions read the full frame.** `profile_frame` and the identifier screen
   ran before the split, so cardinality, missing rate and uniqueness included held-out
   rows, and which columns survived was a function of test data. **Fixed**:
   `fit_preprocessing` now splits first, and screens are injected callables run on the
   training fold.

   Verified to be a real behavioural change, not a cosmetic one: under the old ordering,
   tampering with only the held-out rows flipped a column from one-hot to
   high-cardinality encoding. Under the new ordering the plan is byte-identical.
   `tests/test_leakage.py` asserts exactly this.
2. **No target-correlation leakage screen.** **Fixed**: `target_leakage_screen` flags any
   feature correlating above `LEAKAGE_CORRELATION` with the target. It flags and never
   drops — the spec's word, and the right one, since only the person who exported the
   file knows whether the column is a mistake or the point.
3. `TargetEncoder` was added, which does read `y`. A spy test asserts it never sees a
   held-out label, and its cross-fitting is verified by checking a training row's encoded
   value is *not* the mean of its own category.

## Upload security — resolved

1. **Unbounded read before the cap** — fixed, see Bugs 2.
2. **Header injection via filename** — fixed, see Bugs 3.
3. **No declared content-type check** — extension validation in `ingest.py` remains the
   gate, but it now runs after a bounded read rather than an unbounded one.
4. **Storage path traversal** — new surface, closed on arrival: `storage.py` validates
   every key segment, so a key built from a user filename cannot escape the root.
5. **XLSX decompression bombs** — still unbounded; see *Known limitations*.

## Dead code — removed

- `backend/dropout_pipeline.py` — a generated export hardcoding a path in `~/Downloads`.
- `backend/sample_data/student_dropout_sample.csv` — superseded by `app/sample/`.
- `frontend/src/assets/` — `hero.png`, `react.svg`, `vite.svg`, referenced by nothing.
- The unreachable identifier-screen branch (Bugs 8).

## Known limitations

Deliberately not fixed, and why.

- **The exported script does not reproduce header disambiguation.** `_tidy_headers`
  renames unnamed and duplicate columns; the generated `load_frame()` only strips
  whitespace. An export from a file with duplicate headers will not reproduce the
  analysed frame. Narrow enough to be worth stating rather than papering over.
- **XLSX parsing is unbounded.** A decompression-bomb `.xlsx` within the 50 MB cap can
  still expand arbitrarily in memory. Fixing it means either bounding openpyxl or
  pre-scanning the archive, both of which are more machinery than a localhost demo
  warrants.
- **Ambiguous dates are read month-first.** `05/01/2024` is May 1st, not 1 May. There is
  no way to infer the intent from the file, and asking would be a UI this phase does not
  have.
- **No accounts.** `project_id` is always `"default"`.

## Verification

Run on 2026-09-20 from a clean state — database deleted, storage deleted, migrations
applied from zero.

| Check | Result |
|---|---|
| `alembic upgrade head` on an empty database | 4 tables, each with `project_id` + `created_at` |
| `pytest tests/ -q` | 137 passed |
| `uvicorn app.main:app` | starts, `/api/health` responds |
| `npm run build` | builds clean |
| `npm run lint` | clean |
| End-to-end on `tests/fixtures/titanic.csv` over HTTP | upload → profile → preprocess → log → downloads, all 200 |
| Exported script | compiles and runs standalone, both encoders |
| Cold-start restart | joblib pipeline reloads and transforms all 891 rows |
| UI | verified in a browser at desktop and 375px widths |

End-to-end result on the fixture: 891 rows, health 86, target `Survived` detected as
binary classification, 712/179 split, 22 features, `PassengerId`/`Ticket`/`Name`/`Cabin`
set aside with reasons, 7 step cards including the datetime derivation.
