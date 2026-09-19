# ML-BenchX — glass-box preprocessing

Upload a spreadsheet, say what you want to predict, and see every change made to your
data in plain English, with the reason for each one.

Other AutoML tools transform your data silently. This is the counter-argument: one
screen that shows the whole sequence, names the real numbers it learned, and proves the
test rows were set aside before anything was measured.

---

## Running it

Two terminals, one command each. Nothing else to install — the virtual environment and
`node_modules` are already set up.

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
network disconnected.

### Tests

```bash
cd backend && ../.venv/bin/python -m pytest tests/ -q
```

49 tests: 24 on the preprocessing engine, 25 on reading files, guessing the target and
the API.

### If the environment is ever rebuilt

```bash
python3 -m venv .venv && .venv/bin/python -m pip install pandas numpy "scikit-learn>=1.4" fastapi "uvicorn[standard]" python-multipart openpyxl charset-normalizer pytest httpx
```

```bash
cd frontend && npm install
```

---

## What it does

```
your file ──▶ ingest ──▶ profile ──▶ plan ──▶ SPLIT ──▶ fit ──▶ explain ──▶ screen
             repairs    describes   decides   ↑          train    plain      cards
             format     columns     steps     └─ before anything is measured
```

1. **Reads the file.** Detects encoding and separator, treats `N/A`, `-`, `unknown` and
   friends as genuinely missing, converts columns stored as text into numbers, and
   records every repair as a sentence.
2. **Guesses the target** from your plain-English goal by string matching — no LLM, no
   network. The guess is always shown with a way to change it.
3. **Sets aside 20% of the rows** before fitting anything.
4. **Prepares the columns** — fills gaps, encodes labels, puts numbers on one scale —
   learning every value from the training rows alone.
5. **Explains itself** by reading the fitted objects, not by narrating alongside them.

### The correctness rule

The split happens before any value is learned. Imputation medians, scaler centres,
encoder categories and frequency tables all come from the training rows only.

This is enforced in `backend/app/services/glassbox.py` and tested two ways: a spy that
records the row index of every `fit` call and asserts zero overlap with the held-out
rows, and a check that the logged fill value equals the *training* median and differs
from the whole-file median. Every step card in the UI carries the badge
**"learned from training rows only"**, and the split is shown as step 1 — because that
is the order it happened in.

### The log cannot lie

No explanation string is written next to the code that does the work. Each one is built
by reading the fitted object's learned attributes — `statistics_`, `mean_`, `scale_`,
`categories_`, `frequencies_` — after the fact. A hand-maintained log drifts out of sync
with the code; a derived one cannot.

---

## Layout

```
backend/
  app/
    main.py                  three endpoints, in-memory run store
    sample/campus_records.csv  the bundled example — deliberately messy
    services/
      ingest.py              read the file, repair the format, score quality
      target.py              plain-English goal -> column, no LLM
      screening.py           spot identifier columns before they become features
      glassbox.py            profile, plan, split, fit, introspect, export
      explain.py             fitted objects -> cards a non-technical reader can follow
  tests/                     49 tests
frontend/
  src/
    App.tsx                  three states: upload, working, results
    api.ts                   response types and the two calls
    components/              Upload, Working, Results, StepCard, TransformStrip
    index.css                the design tokens
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

This is a demonstration of one screen, so there is no authentication, no database, no
job queue and no model training. Runs are held in a dict in memory, which means results
are lost on restart and the API must run as a single worker — correct here, wrong in
production, where that becomes a table.

Dates are dropped rather than turned into calendar features; target encoding is not
offered because making it leakage-safe needs out-of-fold fitting, and frequency
encoding gets the same job done without ever reading the target.
