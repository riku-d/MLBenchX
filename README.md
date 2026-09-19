# ML-BenchX — glass-box preprocessing

Upload a spreadsheet, say what you want to predict, and see every change made to your
data in plain English, with the reason for each one.

Other AutoML tools transform your data silently. This is the counter-argument: one
screen that shows the whole sequence, names the real numbers it learned, and proves the
test rows were set aside before anything was measured.

---

## Running it

Two terminals, one command each. The virtual environment and `node_modules` are already
set up; the database creates itself on first run.

**Terminal 1 — the API**

```bash
cd backend && ../.venv/bin/python -m uvicorn app.main:app --reload --port 8000
```

**Terminal 2 — the interface**

```bash
cd frontend && npm run dev
```

Then open **http://localhost:5173** and click **Try an example**.

The example runs entirely on your machine from a bundled file, so it works with the
network disconnected. Interactive API documentation is at
**http://localhost:8000/api/docs**.

### Tests

```bash
cd backend && ../.venv/bin/python -m pytest tests/ -q
```

137 tests. They run against a temporary database and storage directory, never the ones
the app uses, so running them is always safe.

| File | What it covers |
|---|---|
| `test_glassbox.py` | The preprocessing engine: profiling, routing, the feature map |
| `test_rules.py` | One test per rule — each scaler, each encoder, dates, outliers, task detection |
| `test_leakage.py` | That held-out rows reach neither a fit nor a decision |
| `test_api.py` | Every route end to end, including bad input and hostile filenames |
| `test_app.py` | Reading files, guessing the target, the single-call endpoints |

### Database

The schema is managed by Alembic and lives in `backend/mlbenchx.db`.

```bash
cd backend && ../.venv/bin/python -m alembic upgrade head
```

The app also creates any missing tables at startup, so this is only needed after
pulling a schema change. To add one, edit `backend/app/db.py` and then:

```bash
cd backend && ../.venv/bin/python -m alembic revision --autogenerate -m "what changed"
```

### If the environment is ever rebuilt

```bash
python3 -m venv .venv && .venv/bin/python -m pip install pandas numpy "scikit-learn>=1.4" fastapi "uvicorn[standard]" python-multipart openpyxl charset-normalizer joblib "sqlalchemy>=2.0" alembic pytest httpx
```

```bash
cd frontend && npm install
```

---

## What it does

```
your file ──▶ ingest ──▶ store ──▶ profile ──▶ SPLIT ──▶ screen ──▶ plan ──▶ fit ──▶ explain
             repairs    keeps     describes    ↑         judges     decides  train   plain
             format     bytes     columns      └─ before anything is measured or decided
```

1. **Reads the file.** Detects encoding and separator, treats `N/A`, `-`, `unknown` and
   friends as genuinely missing, converts columns stored as text into numbers or dates,
   and records every repair as a sentence.
2. **Keeps the original bytes** through a storage backend, so the same upload can be
   preprocessed again with different options without re-uploading it.
3. **Profiles every column** — missing share, cardinality, IQR outliers — plus duplicate
   rows, highly correlated pairs and class imbalance, and scores the file out of 100.
4. **Guesses the target** from your plain-English goal by string matching — no LLM, no
   network. The guess is always shown with a way to change it.
5. **Sets aside 20% of the rows** before anything at all is measured.
6. **Prepares the columns** — fills gaps, encodes labels, builds calendar features, puts
   numbers on one scale — learning every value from the training rows alone.
7. **Explains itself** by reading the fitted objects, not by narrating alongside them.

### The correctness rule

The split happens before any value is learned **and before any decision is made**.
Imputation medians, scaler centres, encoder categories and frequency tables all come
from the training rows — and so does the column profile that decides which columns
survive and how each one is encoded.

That second half is the stricter claim, and it is the one most tools get wrong.
Profiling the whole file and only *fitting* on the training rows still lets a category
that appears exclusively in your held-out rows push a column over the one-hot threshold.
Here, the split is the first thing that happens.

It is tested three ways in `tests/test_leakage.py`: a spy that records the row index of
every `fit` call and asserts zero overlap with the held-out rows; a check that the logged
fill value equals the *training* median and differs from the whole-file median; and the
strongest form — rewrite the held-out rows entirely, and assert the resulting plan is
byte-identical. Every step card in the UI carries the badge **"learned from training
rows only"**, and the split is shown as step 1, because that is the order it happened in.

### The log cannot lie

No explanation string is written next to the code that does the work. Each one is built
by reading the fitted object's learned attributes — `statistics_`, `mean_`, `scale_`,
`center_`, `categories_`, `frequencies_`, `encodings_` — after the fact. A hand-maintained
log drifts out of sync with the code; a derived one cannot.

---

## Choices you can make

Passed to `POST /api/datasets/{id}/preprocess`, defaults in `backend/app/config.py`.

| Option | Values | Default |
|---|---|---|
| `scaler` | `standard`, `minmax`, `robust` | `standard` |
| `high_card_encoder` | `frequency`, `target` | `frequency` |
| `test_size` | 0.05 – 0.5 | `0.2` |
| `target` | any column name | inferred from `goal` |

**Why frequency encoding is the default.** `FrequencyEncoder` replaces each value with
how often it occurs in the training rows. It never reads the target, so it cannot leak
under any circumstances. `TargetEncoder` is offered because it is often more predictive:
it does read the target, but cross-fits internally, so the value a training row receives
is computed from folds that exclude that row. The naive version of target encoding — the
one that does leak — is among the most common causes of silently optimistic AutoML
scores, and is not what sklearn's implementation does.

---

## The API

| Route | Does |
|---|---|
| `POST /api/datasets` | Store an upload and record what it is |
| `GET /api/datasets` | List datasets, newest first |
| `GET /api/datasets/{id}` | One dataset and its runs |
| `GET /api/datasets/{id}/profile` | Per-column measurements and the health score |
| `POST /api/datasets/{id}/preprocess` | Split, screen, fit and explain |
| `GET /api/datasets/{id}/preprocess/log` | The ordered glass-box log |
| `GET /api/runs/{id}` | A completed run, exactly as first computed |
| `GET /api/runs/{id}/data.csv` | The prepared training data |
| `GET /api/runs/{id}/script.py` | Runnable Python that reproduces the run |
| `POST /api/analyse` | Upload and preprocess in one call |
| `POST /api/example` | The same, on the bundled sample |

`/api/analyse` and `/api/example` are what the single-screen UI uses. They are wrappers
over the routes above, not a second implementation, and leave the same durable trace.

---

## The health score

Starts at 100 and loses weighted, capped points for each defect. The weights live in
`HEALTH_WEIGHTS` in `backend/app/config.py`, and the app shows the arithmetic itemised
under "How was this worked out?".

```
score = 100
  − min(blank_cell_percentage      × 1.5, 35)
  − min(duplicate_row_percentage   × 1.0, 15)
  − min(empty_column_count         × 4.0, 12)
  − min(sparse_column_count        × 3.0, 12)
  − min(leakage_flag_count         × 10.0, 20)
  − 8 if rows < 200
```

Blank cells weigh most because they are the most common and the most quietly damaging.
Each cap stops one defect from sinking the score alone. A small sample is a flat penalty
because it is a fact about the file rather than a matter of degree.

---

## Layout

```
backend/
  app/
    config.py                every threshold, with the reasoning for its value
    main.py                  the routes
    db.py                    SQLAlchemy models and sessions
    schemas.py               Pydantic request and response shapes
    storage.py               StorageBackend interface + LocalStorage
    sample/campus_records.csv  the bundled example — deliberately messy
    services/
      ingest.py              read the file, repair the format, score quality
      target.py              plain-English goal -> column, no LLM
      screening.py           identifier and target-leakage screens
      profile.py             per-column measurements for the dashboard
      glassbox.py            split, plan, fit, introspect, export
      explain.py             fitted objects -> cards a non-technical reader can follow
  migrations/                Alembic
  tests/                     137 tests, fixtures/titanic.csv
frontend/
  src/
    App.tsx                  three states: upload, working, results
    api.ts                   response types and the calls
    components/              Upload, Working, Results, ColumnTable, StepCard, TransformStrip
    index.css                the design tokens
```

All thresholds are in `backend/app/config.py`, each with the reasoning for its value. Any
of them can be overridden by an environment variable of the same name prefixed
`MLBENCHX_`:

```bash
MLBENCHX_MAX_UPLOAD_BYTES=104857600 ../.venv/bin/python -m uvicorn app.main:app
```

### Design notes

Six named colours: `bone` (surface), `paper` (cards), `ink` (text), `slate`
(structure), `xray` (the one saturated accent), `ochre` (warnings only). `xray` appears
in exactly two places — a value that changed, and the active control. If it showed up
anywhere else the highlight would stop meaning anything.

Monospace is not decoration. Every column name, value and number is set in IBM Plex
Mono; everything the app says about them is in IBM Plex Sans. That boundary is visible
at a glance and is the one typographic decision doing real work.

---

## Deliberate limits

There is no authentication and no model training. Every table carries a `project_id`
that is always `"default"` — the column exists so that adding accounts later is a
migration rather than a redesign.

Uploads and fitted pipelines are written to the local filesystem through
`StorageBackend`. `LocalStorage` is the only implementation; the interface exists so the
code that stores a pipeline is not the code that knows about file permissions.

Runs are also held in a small in-memory cache, which means a restart makes the next
download slower, not impossible: the database and storage are the source of truth, and a
cache miss reads back the stored artefacts.

The app is single-worker by design in development. Nothing about the storage or database
layer prevents more than one; the in-memory cache simply stops being useful.
