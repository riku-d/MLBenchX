"""ML-BenchX API — one screen's worth of endpoints.

Three routes: analyse an upload, download the cleaned data, download the equivalent
Python. Everything is synchronous because everything finishes in well under a second on
the file sizes this is built for; a job queue here would be machinery with nothing to do.

Runs are held in a module-level dict. That is deliberate for a single-machine demo: it
keeps the whole app to two processes with nothing to install. It also means results are
lost on restart and the app cannot be run with more than one worker — both fine here,
both wrong for production, where this becomes a table.
"""

from __future__ import annotations

import io
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from app.services import explain
from app.services.glassbox import PreprocessingResult, export_script, fit_preprocessing
from app.services.ingest import (
    MIN_ROWS,
    IngestError,
    IngestResult,
    loader_source,
    read_upload,
)
from app.services.screening import find_identifier_columns
from app.services.target import describe_task, detect_target, summarise_values

log = logging.getLogger("mlbenchx")

#: Upload ceiling. Generous for a demo, small enough that a mis-drag cannot exhaust RAM.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

#: How many runs to keep. Old ones are evicted so a long session cannot grow without
#: bound; the downloads for the current run always work, which is all this needs.
MAX_RUNS = 20

#: The only source of randomness in the app.
SEED = 42

SAMPLE_PATH = Path(__file__).parent / "sample" / "campus_records.csv"

app = FastAPI(title="ML-BenchX", version="0.1.0", docs_url="/api/docs")

# The Vite dev server runs on a different port, so the browser needs permission to call
# this one. Localhost only — this app is never meant to be exposed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:4173", "http://127.0.0.1:4173",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@dataclass
class Run:
    """Everything needed to serve the downloads for one analysis."""

    payload: dict[str, Any]
    result: PreprocessingResult
    raw: pd.DataFrame
    filename: str
    target: str
    ingested: IngestResult


_RUNS: "OrderedDict[str, Run]" = OrderedDict()


def _store(run_id: str, run: Run) -> None:
    _RUNS[run_id] = run
    while len(_RUNS) > MAX_RUNS:
        _RUNS.popitem(last=False)


def _fetch(run_id: str) -> Run:
    run = _RUNS.get(run_id)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail="That result has expired. Analyse your file again to download it.",
        )
    return run


class UserFacingError(Exception):
    """An error whose message is written for the user and can be shown as-is."""


@app.exception_handler(UserFacingError)
async def _user_facing(_request, exc: UserFacingError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.exception_handler(HTTPException)
async def _http_error(_request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(Exception)
async def _unexpected(_request, exc: Exception) -> JSONResponse:
    # The traceback goes to the terminal; the user gets a sentence they can act on.
    log.exception("unhandled error while analysing", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Something went wrong reading that file. Try the example file to "
                     "check the app is working, then re-save yours as CSV and retry."
        },
    )


# --------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------


def _check_target(frame: pd.DataFrame, column: str) -> None:
    """Reject targets that cannot be predicted, with a sentence saying why."""
    series = frame[column]
    n_unique = int(series.nunique(dropna=True))
    if n_unique < 2:
        raise UserFacingError(
            f"Every row has the same value in {column}, so there is nothing to predict. "
            "Pick a different column."
        )
    if series.isna().mean() > 0.5:
        raise UserFacingError(
            f"More than half of {column} is blank, so there is not enough to learn from. "
            "Pick a different column."
        )
    if not pd.api.types.is_numeric_dtype(series) and n_unique > 50:
        raise UserFacingError(
            f"{column} has {n_unique:,} different values, which looks like a label rather "
            "than an outcome. Pick a column with fewer distinct values."
        )


def _analyse(raw_bytes: bytes, filename: str, goal: str, target: str | None) -> dict[str, Any]:
    """Run the whole pipeline and shape the response. Raises UserFacingError on bad input."""
    started = time.perf_counter()

    try:
        ingested: IngestResult = read_upload(raw_bytes, filename)
    except IngestError as exc:
        raise UserFacingError(str(exc)) from exc

    frame = ingested.frame

    try:
        guess = detect_target(frame, goal or "", target)
    except ValueError as exc:
        raise UserFacingError(str(exc)) from exc

    _check_target(frame, guess.column)

    # A row with no answer in the target teaches nothing, and carrying it forward breaks
    # the stratified split. Drop those rows and say so — it is a change to their data.
    blank_targets = int(frame[guess.column].isna().sum())
    if blank_targets:
        frame = frame[frame[guess.column].notna()].reset_index(drop=True)
        ingested.cleaning.append(
            f"Left out {blank_targets:,} row{'' if blank_targets == 1 else 's'} where "
            f"{guess.column} was blank — with no answer recorded there is nothing to "
            "learn from them"
        )
        if len(frame) < MIN_ROWS:
            raise UserFacingError(
                f"Only {len(frame)} rows have a value in {guess.column}. We need at "
                f"least {MIN_ROWS} to set some aside for honest testing."
            )

    task = describe_task(frame[guess.column])
    task_type = "regression" if task == "regression" else "classification"

    # Numeric primary keys profile as ordinary numbers, so they are screened out here and
    # passed through the same `excluded` channel a leakage screen would use.
    identifiers = find_identifier_columns(frame, target=guess.column)

    try:
        result = fit_preprocessing(
            frame,
            target=guess.column,
            task_type=task_type,
            excluded=identifiers,
            test_size=0.2,
            seed=SEED,
        )
    except ValueError as exc:
        raise UserFacingError(
            f"We couldn't prepare that data: {exc}".split("Reasons:")[0].strip()
            + " Try a file with more usable columns."
        ) from exc

    steps = explain.build_steps(frame, result)
    dropped = explain.build_dropped(result, frame, identifiers)

    run_id = uuid.uuid4().hex
    payload = {
        "run_id": run_id,
        "file": {
            "name": ingested.name,
            "rows": ingested.rows,
            "columns": ingested.columns,
            "encoding": ingested.encoding,
            "delimiter": ingested.delimiter,
        },
        "goal": goal or "",
        "target": {
            "column": guess.column,
            "task": task,
            "values": summarise_values(frame[guess.column]),
            "inferred": guess.inferred,
            "note": guess.note,
        },
        "columns": [str(c) for c in frame.columns],
        "cleaning": ingested.cleaning,
        "quality": {
            "score": ingested.quality_score,
            "rows_used": len(frame),
            "missing_cells": ingested.missing_cells,
            "duplicate_rows": ingested.duplicate_rows,
            "issues": [{"severity": i.severity, "text": i.text} for i in ingested.issues],
        },
        "dropped": dropped,
        "steps": steps,
        "result": {
            "train_rows": len(result.X_train),
            "test_rows": len(result.X_test),
            "features": int(result.X_train.shape[1]),
        },
        "took_ms": round((time.perf_counter() - started) * 1000),
    }

    _store(run_id, Run(payload, result, frame, ingested.name, guess.column, ingested))
    return payload


@app.post("/api/analyse")
async def analyse(
    file: UploadFile = File(...),
    goal: str = Form(""),
    target: str | None = Form(None),
) -> dict[str, Any]:
    """Analyse an uploaded spreadsheet and return the full explanation."""
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise UserFacingError(
            f"That file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB. "
            "Try a smaller extract of it."
        )
    return _analyse(raw, file.filename or "upload.csv", goal, target or None)


@app.post("/api/example")
async def example(goal: str = Form("which students will fail"),
                  target: str | None = Form(None)) -> dict[str, Any]:
    """Analyse the bundled sample file.

    Read from disk on the server so the demo works with the network unplugged and with
    nothing in the browser's cache.
    """
    if not SAMPLE_PATH.exists():  # pragma: no cover - the file ships with the app
        raise UserFacingError("The example file is missing from this install.")
    return _analyse(SAMPLE_PATH.read_bytes(), SAMPLE_PATH.name, goal, target or None)


# --------------------------------------------------------------------------------------
# Downloads
# --------------------------------------------------------------------------------------


@app.get("/api/runs/{run_id}/data.csv")
async def download_data(run_id: str) -> PlainTextResponse:
    """The cleaned training data: every prepared feature, plus the column being predicted."""
    run = _fetch(run_id)
    frame = run.result.X_train.copy()
    frame[run.target] = run.result.y_train
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    stem = Path(run.filename).stem or "data"
    return PlainTextResponse(
        buffer.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{stem}_prepared.csv"'},
    )


@app.get("/api/runs/{run_id}/script.py")
async def download_script(run_id: str) -> PlainTextResponse:
    """Python that reproduces this preparation exactly, split included."""
    run = _fetch(run_id)
    # The script must reproduce the *cleaned* frame, so it carries the same reader
    # settings and the same format repairs the upload needed.
    block, call = loader_source(run.ingested)
    source = export_script(
        run.result,
        csv=run.filename,
        exp_id=run_id[:8],
        seed=SEED,
        loader_block=block,
        loader_call=call,
    )
    return PlainTextResponse(
        source,
        media_type="text/x-python",
        headers={"Content-Disposition": 'attachment; filename="prepare_data.py"'},
    )


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "runs_held": len(_RUNS)}
