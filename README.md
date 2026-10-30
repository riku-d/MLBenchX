# ML-BenchX: glass-box preprocessing

Upload a spreadsheet, see exactly what is wrong with it, switch any preprocessing step
on or off with the consequences spelled out first, edit the generated Python by hand,
and compile a production-ready dataset — in one workspace.

Other AutoML tools transform your data silently. This is the counter-argument: every
automated recommendation is **visible, explained, optional and reversible**, and the
code that runs is code you can read and change.

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

Then open **http://localhost:5173** and click **Start preprocessing**.

Everything runs on your machine. Interactive API documentation is at
**http://localhost:8000/api/docs**.

### Tests

```bash
cd backend && ../.venv/bin/python -m pytest tests/ -q
```

244 tests. They run against a temporary database and storage directory, never the ones
the app uses, so running them is always safe.

| File | What it covers |
|---|---|
| `test_glassbox.py` | The preprocessing engine: profiling, routing, the feature map |
| `test_rules.py` | One test per rule — each scaler, each encoder, dates, outliers, task detection |
| `test_leakage.py` | That held-out rows reach neither a fit nor a decision |
| `test_codegen.py` | Every option combination, by running the generated script and checking its output |
| `test_executor.py` | Real attacks on the sandbox — network, filesystem, fork and memory bombs |
| `test_workspace.py` | The workspace API as the interface drives it |
| `test_api.py` | Every dataset route end to end, including bad input and hostile filenames |
| `test_app.py` | Reading files, guessing the target, the single-call endpoints |

The sandbox tests genuinely try to delete this repository and reach the network, then
check afterwards that they failed. They take about twenty seconds.

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
python3 -m venv .venv && .venv/bin/python -m pip install pandas numpy "scikit-learn>=1.4" fastapi "uvicorn[standard]" python-multipart openpyxl pyarrow charset-normalizer joblib psutil "sqlalchemy>=2.0" alembic pytest httpx
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

1. **Reads the file.** CSV, TSV, Excel, JSON (records, columns or line-delimited) and
   Parquet. Detects encoding and separator, treats `N/A`, `-`, `unknown` and friends as
   genuinely missing, converts columns stored as text into numbers or dates, and records
   every repair as a sentence.
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

## The workspace

The interface is three panels over a dock.

**Pipeline** — eight steps, each a switch. A card says what the step is doing *to this
dataset*: "885 missing value(s) in 3 column(s) will be filled", or "No duplicate rows
found." Turning one off opens a dialog first, and that dialog is the point of the
product:

```
Disable Missing Value Treatment?                This will stop the pipeline working

  • 885 missing value(s) will remain, across 3 column(s).
  • Most scikit-learn estimators raise on the first blank they meet, so training
    will fail rather than degrade.
  • 1 column(s) over 60% empty will be kept: Cabin

  DATASET HEALTH   92 → 90    Measured by running the pipeline both ways —
                              not estimated.
```

Every number there is measured from the dataset in front of you. A step that is doing
nothing for your data says so instead — "No duplicate rows were found, so this step is
currently doing nothing" — which is just as useful and far rarer to be told.

**Code** — the generated pipeline in a real editor: find and replace, folding, an error
marker on the failing line, a jump-to-line from the error panel, a diff against the
generated version, and ⌘↵ to compile.

**Insights** — the health score, the six dimensions that produced it, the arithmetic
behind it, and after a compile, every dimension's before and after.

**Dock** — the data itself with blanks and outliers marked in place and an
original/processed switch, the run log with the structured error, and a change history
where any earlier state can be restored.

### One source of truth, and one honest exception

The toggles write the configuration. The code is generated from the configuration.
They cannot disagree, because there is nothing for them to disagree between.

The exception is the reason the editor exists. **Once you edit the code by hand, the
code is what runs** — and the app says so, in the top bar, on the file tab, and in a
banner over the pipeline panel:

> You have edited the code by hand, so it is what runs. These switches describe the
> pipeline they were generated from, not what will execute.

Changing a toggle after that regenerates the *generated* version, which stays available
to diff against and restore from, but **never overwrites your edit**.

---

## Running your code safely

The editor means the backend executes Python that a user wrote. That is the most
dangerous thing this application does, and it is defended in layers:

| Layer | What it stops |
|---|---|
| macOS `sandbox-exec` profile | Network access outright; writes anywhere but a scratch directory |
| `RLIMIT_CPU`, `RLIMIT_FSIZE`, `RLIMIT_NPROC` | Busy loops, disk filling, fork bombs |
| Memory watchdog thread | Runaway allocation |
| Wall-clock timeout on the process group | Anything that blocks |
| Scrubbed environment | Credentials and proxy settings reaching the child |

`tests/test_executor.py` attacks all of it: it tries to reach the network, resolve a
hostname, write outside the work directory, `rmtree` this repository, shell out to `rm`
against a canary file, fork without limit, allocate without limit, and loop forever —
then asserts afterwards that the repository is still there.

**The memory limit is a watchdog, not an `RLIMIT_AS`.** macOS silently refuses to lower
`RLIMIT_AS` at any value, so the `setrlimit` call that looks like a memory cap in most
examples does nothing there. A thread polls the process group's resident size and kills
it on breach, which works on both platforms.

**What this is not.** On a platform without `sandbox-exec` the OS layer is unavailable
and only the process limits apply; the app reports which layer is active and the
workspace shows it in the top bar. The production answer is a container with no network
namespace, and the executor is shaped so that swapping one in means replacing a single
function.

---

## Choices you can make

Every step, and every option on it. Defined in `backend/app/pipeline.py`, with the
thresholds in `backend/app/config.py`.

| Step | Options | Default |
|---|---|---|
| **Data Type Detection** | `coerce_numeric`, `parse_dates` | both on — cannot be disabled |
| **Missing Values** | numeric: `median`, `mean`, `constant`, `forward_fill`, `backward_fill`, `drop_rows` | `median` |
| | categorical: `most_frequent`, `constant`, `forward_fill`, `backward_fill`, `drop_rows` | `most_frequent` |
| | `drop_column_threshold` | `0.60` |
| **Duplicates** | `subset`, `keep` = `first` / `last` | whole row, keep first |
| **Outliers** | method `iqr` / `zscore`; action `cap`, `remove`, `winsorize`, `flag` | `iqr` + `cap` |
| **Encoding** | binary `label` / `onehot`; low-card `onehot` / `ordinal` / `label`; high-card `frequency` / `target` / `ordinal` / `drop` | `label`, `onehot`, `frequency` |
| | `max_onehot_cardinality` | `10` |
| **Scaling** | `standard`, `minmax`, `robust` | `standard` |
| **Feature Selection** | `variance_threshold`, `correlation_threshold`, `mutual_information` + `top_k` | **off** |
| **Class Imbalance** | `report`, `oversample`, `undersample`, `class_weight` | `report` |

Plus `target`, `test_size` (0.05–0.5, default `0.2`), `seed`, and `excluded_columns`.

**Step order is fixed and not configurable.** Each step assumes the previous ones have
run: encoding needs the blanks filled or it learns "missing" as a category, scaling
needs the categories already numeric, feature selection needs everything numeric to
measure correlation at all. Letting a user reorder them would let them build pipelines
that fail in ways that are hard to explain.

**Two defaults are deliberately conservative.** Outliers are *capped* rather than
removed, because an unusually large real value and a mistyped one look identical from
here and removal is the only action that loses data outright. Feature selection is
*off*, because every other step changes how a column is represented while that one
deletes it, and which of two correlated columns to keep is a judgement about meaning
rather than something a threshold settles.

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

### Workspace routes

These are what the interactive workspace talks to.

| Route | Does |
|---|---|
| `POST /api/workspaces/from-dataset/{id}` | Open the workspace for a dataset (idempotent) |
| `GET /api/workspaces/{id}` | The whole workspace state |
| `PUT /api/workspaces/{id}/config` | Replace the pipeline configuration |
| `PUT /api/workspaces/{id}/code` | Save the editor contents |
| `POST /api/workspaces/{id}/code/reset` | Discard edits, regenerate from the config |
| `POST /api/workspaces/{id}/impact` | What a toggle would cost, measured |
| `POST /api/workspaces/{id}/compile` | Run the pipeline in the sandbox |
| `GET /api/workspaces/{id}/preview` | A page of rows, blanks and outliers marked |
| `GET /api/workspaces/{id}/history` | The change history |
| `POST /api/workspaces/{id}/history/{n}/restore` | Go back to an earlier state |
| `GET /api/workspaces/{id}/download/{what}` | `processed`, `original`, `code`, `config`, `report`, `project` |

Every mutating route returns the **whole** workspace, so the client never reassembles
state from a patch and the panels cannot drift apart.

---

## Dataset health

One score out of a hundred is easy to display and impossible to act on. "You scored 61"
says nothing about what to do next; "your data is 87% complete but only 54% unique"
sends you looking for duplicates. So the score is a weighted mean of six independently
measured dimensions, each shown with the evidence behind it.

| Dimension | Weight | Measures |
|---|---|---|
| Completeness | 0.25 | How much of the data is actually present |
| Validity | 0.20 | Whether values sit in ranges that make sense |
| Consistency | 0.15 | Whether each column holds one kind of thing, written one way |
| Uniqueness | 0.15 | Whether rows are distinct observations |
| Feature Quality | 0.15 | Whether columns carry independent signal |
| Distribution | 0.10 | Whether numeric columns are shaped usefully |

Completeness and validity weigh most because they are the two that stop a model
training at all. Distribution weighs least: a skewed column is a reason to pick a
different scaler, not a reason to distrust the data.

The weights live in `DIMENSION_WEIGHTS` in `backend/app/services/health.py`, and the
workspace shows the arithmetic term by term under "How this number was worked out".

The same six are computed on the data you uploaded and on the data that came out, and
the difference is the honest answer to "did this help?" — including when it is no.

---

## Layout

```
backend/
  app/
    config.py                every threshold, with the reasoning for its value
    pipeline.py              PipelineConfig — the single source of truth
    codegen.py               config -> runnable Python, the only such path
    executor.py              runs user code in a sandbox
    workspace.py             the routes the interactive UI talks to
    main.py                  dataset routes and the single-call endpoints
    db.py                    SQLAlchemy models and sessions
    schemas.py               Pydantic request and response shapes
    storage.py               StorageBackend interface + LocalStorage
    sample/campus_records.csv  the bundled example — deliberately messy
    services/
      ingest.py              read the file, repair the format, score quality
      target.py              plain-English goal -> column, no LLM
      screening.py           identifier and target-leakage screens
      profile.py             per-column measurements for the dashboard
      health.py              the six dimensions
      impact.py              what turning a step off would cost
      glassbox.py            split, plan, fit, introspect, export
      explain.py             fitted objects -> cards a non-technical reader can follow
  migrations/                Alembic
  tests/                     244 tests, fixtures/titanic.csv
frontend/
  src/
    App.tsx                  routing: landing, upload, workspace
    api.ts                   the API contract and the only place we call it
    components/ui.tsx        buttons, toggles, dialogs, meters, the score ring
    landing/Landing.tsx      the marketing page
    workspace/
      Workspace.tsx          the three-panel shell and all workspace state
      PipelinePanel.tsx      step cards, options, the impact dialog
      CodePanel.tsx          Monaco, compile, reset, diff
      InsightsPanel.tsx      health, dimensions, issues, changes
      BottomDock.tsx         data table, logs, history
      UploadScreen.tsx       drag-and-drop and narrated analysis
    index.css                the design tokens
```

All thresholds are in `backend/app/config.py`, each with the reasoning for its value. Any
of them can be overridden by an environment variable of the same name prefixed
`MLBENCHX_`:

```bash
MLBENCHX_MAX_UPLOAD_BYTES=104857600 ../.venv/bin/python -m uvicorn app.main:app
```

### Design notes

Dark-first, because the product is an instrument panel read for long stretches. Four
surface levels (`void`, `surface`, `raised`, `sunken`), three ink levels (`text`,
`muted`, `faint`), and one saturated accent reserved for exactly three things: a value
that changed, the active control, and the compile action. If it appeared anywhere else
the highlight would stop meaning anything. `warn` and `danger` carry everything that
needs attention.

Monospace is not decoration. Every column name, value and measured number is set in IBM
Plex Mono; everything the app *says* about them is in IBM Plex Sans. That boundary is
visible at a glance and is the one typographic decision doing real work.

Every animation reports a state change — something arrived, something is running,
something succeeded or failed. None of it is ornament, and all of it is short enough
not to sit between you and your next action. `prefers-reduced-motion` turns it off.

Three columns on a desktop, one column with a tab bar below 1024px. Verified at 375,
768 and 1440 with no horizontal overflow at any of them.

---

## Deliberate limits

There is no authentication and no model training. Every table carries a `project_id`
that is always `"default"` — the column exists so that adding accounts later is a
migration rather than a redesign.

Compiling is synchronous. A pipeline on the file sizes this accepts finishes in about a
second, and a job queue would be machinery with nothing to do; a dataset large enough to
need one would need streaming ingestion first.

Uploads and fitted pipelines are written to the local filesystem through
`StorageBackend`. `LocalStorage` is the only implementation; the interface exists so the
code that stores a pipeline is not the code that knows about file permissions.

Runs are also held in a small in-memory cache, which means a restart makes the next
download slower, not impossible: the database and storage are the source of truth, and a
cache miss reads back the stored artefacts.

The app is single-worker by design in development. Nothing about the storage or database
layer prevents more than one; the in-memory cache simply stops being useful.
