"""ML-BenchX API.

Two families of routes over the same machinery.

**Dataset resources** — ``/api/datasets`` — are the durable surface. An upload becomes a
row and a stored file; profiling it, preprocessing it and reading back the log are
separate requests against that resource, so a run survives a restart and can be fetched
again without recomputing anything.

**Analysis shortcuts** — ``/api/analyse`` and ``/api/example`` — do upload, profile and
preprocess in one round trip. The interface is a single screen and this is what it
needs; they are wrappers over the resource routes, not a second implementation.

Everything is synchronous because everything finishes in well under a second on the file
sizes this is built for. A job queue here would be machinery with nothing to do.

Runs stay in a small in-memory cache as well as the database. The cache holds the fitted
pipeline and the prepared frames, which are expensive to rebuild and cheap to keep for
the run the user is currently looking at; the database and storage are the source of
truth, and a cache miss re-reads rather than failing.
"""

from __future__ import annotations

import io
import logging
import re
import time
import unicodedata
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import (
    DEFAULT_HIGH_CARD_ENCODER,
    DEFAULT_SCALER,
    MAX_RUNS_IN_MEMORY,
    MAX_UPLOAD_BYTES,
    MAX_TARGET_CARDINALITY,
    MIN_ROWS,
    SEED,
    TEST_SIZE,
    UPLOAD_CHUNK_BYTES,
)
from app.db import (
    Dataset,
    PreprocessingLog,
    PreprocessingRun,
    Profile,
    get_session,
    init_db,
)
from app.schemas import (
    AnalysisResponse,
    DatasetDetail,
    DatasetSummary,
    ErrorResponse,
    LogResponse,
    PreprocessRequest,
    ProfileResponse,
)
from app.services import explain, profile as profiling
from app.services.glassbox import PreprocessingResult, export_script, fit_preprocessing
from app.services.ingest import (
    IngestError,
    IngestResult,
    loader_source,
    read_upload,
)
from app.services.screening import (
    correlated_pairs,
    identifier_screen,
    target_leakage_screen,
)
from app.storage import StorageError, get_storage
from app.services.target import (
    class_balance,
    describe_task,
    detect_target,
    summarise_values,
)

log = logging.getLogger("mlbenchx")

SAMPLE_PATH = Path(__file__).parent / "sample" / "campus_records.csv"

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Create any missing tables so a fresh checkout runs without a migration step.

    Alembic owns the schema; both are generated from the same models, so this is a
    convenience for first run, not a second source of truth.
    """
    init_db()
    yield


app = FastAPI(
    title="ML-BenchX",
    version="1.0.0",
    description=(
        "Glass-box preprocessing for tabular data. Every transformation is reported "
        "with the reason for it and the values it learned, and every one of those "
        "values comes from the training rows alone."
    ),
    docs_url="/api/docs",
    lifespan=_lifespan,
)

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


@app.middleware("http")
async def _reject_oversized_bodies(request: Request, call_next):
    """Refuse an oversized upload from its headers, before a byte of it is parsed.

    Checking the size inside the route is too late: by then Starlette's multipart
    parser has already spooled the whole body to a temporary file, so a multi-gigabyte
    POST costs the disk whatever it wants before we get a say. ``Content-Length`` is
    declared before the body arrives, which is the only point at which refusing is free.

    A chunked request declares no length, so the route's own streaming cap in
    :func:`_read_capped` remains the backstop rather than the primary defence.
    """
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=413,
            content={"error": _too_large_message()},
        )
    return await call_next(request)


def _too_large_message() -> str:
    return (
        f"That file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB. "
        "Try a smaller extract of it."
    )


async def _read_capped(upload: UploadFile) -> bytes:
    """Read an upload, giving up as soon as it exceeds the cap.

    Nothing larger than the cap is ever held, so the ceiling on memory is the cap
    itself rather than whatever the client decided to send.
    """
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise UserFacingError(_too_large_message())
        chunks.append(chunk)
    return b"".join(chunks)


_UNSAFE_IN_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(name: str, fallback: str) -> str:
    """A filename safe to put inside a ``Content-Disposition`` header.

    The name comes from the client, and the header is a single line of ASCII: a quote
    closes the filename early and a newline splits the header. Rather than escape those,
    everything outside a known-good set is replaced, which cannot go wrong.
    """
    stem = Path(name).stem
    ascii_only = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    cleaned = _UNSAFE_IN_FILENAME.sub("_", ascii_only).strip("._-")
    return cleaned[:80] or fallback


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


def _cache(run_id: str, run: Run) -> None:
    """Keep a run's fitted objects hot. The database is the source of truth."""
    _RUNS[run_id] = run
    while len(_RUNS) > MAX_RUNS_IN_MEMORY:
        _RUNS.popitem(last=False)


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
    if not pd.api.types.is_numeric_dtype(series) and n_unique > MAX_TARGET_CARDINALITY:
        raise UserFacingError(
            f"{column} has {n_unique:,} different values, which looks like a label rather "
            "than an outcome. Pick a column with fewer distinct values."
        )




# --------------------------------------------------------------------------------------
# Ingestion — an upload becomes a stored file and a row
# --------------------------------------------------------------------------------------


def _ingest_dataset(
    raw_bytes: bytes, filename: str, session: Session
) -> tuple[Dataset, IngestResult]:
    """Read an upload, keep the original bytes, and record what it is.

    The bytes are stored before anything is derived from them, so every later request
    for this dataset — profile it, preprocess it with different options — starts from
    exactly the file the user sent rather than from a frame we happen to still have.
    """
    try:
        ingested: IngestResult = read_upload(raw_bytes, filename)
    except IngestError as exc:
        raise UserFacingError(str(exc)) from exc

    dataset_id = uuid.uuid4().hex
    suffix = Path(filename).suffix.lower().lstrip(".") or "csv"
    key = f"datasets/{dataset_id}/source.{_safe_filename(suffix, 'csv')}"
    get_storage().put(key, raw_bytes)

    dataset = Dataset(
        id=dataset_id,
        filename=ingested.name,
        storage_key=key,
        size_bytes=len(raw_bytes),
        n_rows=ingested.rows,
        n_columns=ingested.columns,
        encoding=ingested.encoding,
        delimiter=ingested.delimiter,
        health_score=ingested.quality_score,
        cleaning=list(ingested.cleaning),
    )
    session.add(dataset)
    session.flush()
    return dataset, ingested


def _reload(dataset: Dataset) -> IngestResult:
    """Re-read a stored dataset into a frame.

    Ingestion is deterministic — the same bytes give the same frame and the same repair
    notes — so reading the file again is equivalent to having kept it, and avoids a
    second copy of the data going stale in a cache.
    """
    try:
        raw = get_storage().get(dataset.storage_key)
    except StorageError as exc:
        raise HTTPException(
            status_code=410,
            detail=(
                "The stored copy of that file is gone, so it cannot be analysed again. "
                "Upload it once more."
            ),
        ) from exc
    return read_upload(raw, dataset.filename)


def _quality_block(ingested: IngestResult, rows_used: int) -> dict[str, Any]:
    return {
        "score": ingested.quality_score,
        "rows_used": rows_used,
        "missing_cells": ingested.missing_cells,
        "duplicate_rows": ingested.duplicate_rows,
        "issues": [{"severity": i.severity, "text": i.text} for i in ingested.issues],
        "breakdown": ingested.quality_breakdown,
    }


def _store_profile(
    dataset: Dataset, ingested: IngestResult, document: dict[str, Any], session: Session
) -> Profile:
    """Write (or overwrite) the profile for one dataset."""
    existing = session.scalars(
        select(Profile).where(Profile.dataset_id == dataset.id)
    ).first()
    profile = existing or Profile(dataset_id=dataset.id)
    profile.n_rows = document["n_rows"]
    profile.n_columns = document["n_columns"]
    profile.duplicate_rows = document["duplicate_rows"]
    profile.missing_cells = ingested.missing_cells
    profile.health_score = ingested.quality_score
    profile.payload = document
    session.add(profile)
    session.flush()
    return profile


def _profile_response(
    dataset: Dataset, ingested: IngestResult, document: dict[str, Any], rows_used: int
) -> dict[str, Any]:
    """Shape a profile document into the API's response form."""
    return {
        "dataset_id": dataset.id,
        "n_rows": document["n_rows"],
        "n_columns": document["n_columns"],
        "duplicate_rows": document["duplicate_rows"],
        "quality": _quality_block(ingested, rows_used),
        "columns": document["columns"],
        "correlated_pairs": document["correlated_pairs"],
        "leakage_flags": document["leakage_flags"],
        "class_balance": document["class_balance"],
    }


# --------------------------------------------------------------------------------------
# Preprocessing — the pipeline, persisted
# --------------------------------------------------------------------------------------


def _preprocess(
    dataset: Dataset,
    ingested: IngestResult,
    options: PreprocessRequest,
    session: Session,
) -> dict[str, Any]:
    """Fit the pipeline for one dataset and persist everything it produced.

    Raises :class:`UserFacingError` for anything the user can fix.
    """
    started = time.perf_counter()
    frame = ingested.frame

    try:
        guess = detect_target(frame, options.goal or "", options.target)
    except ValueError as exc:
        raise UserFacingError(str(exc)) from exc

    _check_target(frame, guess.column)

    # A row with no answer in the target teaches nothing, and carrying it forward breaks
    # the stratified split. Drop those rows and say so — it is a change to their data.
    blank_targets = int(frame[guess.column].isna().sum())
    cleaning = list(ingested.cleaning)
    if blank_targets:
        frame = frame[frame[guess.column].notna()].reset_index(drop=True)
        cleaning.append(
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

    try:
        result = fit_preprocessing(
            frame,
            target=guess.column,
            task_type=task_type,
            test_size=options.test_size,
            seed=SEED,
            # Both screens measure, so both run inside fit_preprocessing on the training
            # fold. Running them out here on the whole frame would let held-out rows
            # decide which columns survive.
            screeners=[identifier_screen, target_leakage_screen],
            scaler=options.scaler,
            high_card_encoder=options.high_card_encoder,
        )
    except ValueError as exc:
        raise UserFacingError(
            f"We couldn't prepare that data: {exc}".split("Reasons:")[0].strip()
            + " Try a file with more usable columns."
        ) from exc

    steps = explain.build_steps(frame, result)
    dropped = explain.build_dropped(result, frame, result.excluded)
    balance = class_balance(frame[guess.column]) if task_type == "classification" else None
    document = profiling.build_profile(
        frame,
        target=guess.column,
        excluded=result.excluded,
        leakage_flags=result.flags,
        duplicate_rows=ingested.duplicate_rows,
        balance=balance,
    )
    _store_profile(dataset, ingested, document, session)

    run_id = uuid.uuid4().hex
    payload: dict[str, Any] = {
        "run_id": run_id,
        "dataset_id": dataset.id,
        "file": {
            "name": ingested.name,
            "rows": ingested.rows,
            "columns": ingested.columns,
            "encoding": ingested.encoding,
            "delimiter": ingested.delimiter,
        },
        "goal": options.goal or "",
        "target": {
            "column": guess.column,
            "task": task,
            "values": summarise_values(frame[guess.column]),
            "inferred": guess.inferred,
            "note": guess.note,
        },
        "columns": [str(c) for c in frame.columns],
        "cleaning": cleaning,
        "quality": _quality_block(ingested, len(frame)),
        "profile": _profile_response(dataset, ingested, document, len(frame)),
        "options": {
            "scaler": options.scaler,
            "high_card_encoder": options.high_card_encoder,
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

    pipeline_key = _persist_run_artifacts(run_id, result, guess.column)

    run = PreprocessingRun(
        id=run_id,
        dataset_id=dataset.id,
        target=guess.column,
        task=task,
        scaler=options.scaler,
        high_card_encoder=options.high_card_encoder,
        test_size=options.test_size,
        seed=SEED,
        train_rows=len(result.X_train),
        test_rows=len(result.X_test),
        n_features=int(result.X_train.shape[1]),
        pipeline_key=pipeline_key,
        took_ms=payload["took_ms"],
        payload=payload,
    )
    session.add(run)
    for entry in result.log:
        session.add(PreprocessingLog(
            run_id=run_id,
            step=entry["step"],
            stage=entry["stage"],
            action=entry["action"],
            column=entry["column"],
            transformer=entry["transformer"],
            params=entry["params"],
            reason=entry["reason"],
            before=entry["before"],
            after=entry["after"],
            details=entry["details"],
        ))
    session.flush()

    _cache(run_id, Run(payload, result, frame, ingested.name, guess.column, ingested))
    return payload


def _persist_run_artifacts(
    run_id: str, result: PreprocessingResult, target: str
) -> str | None:
    """Write the fitted pipeline and the prepared training data to storage.

    The pipeline is the point: it is the thing that can be loaded and applied to new
    rows, and without it the log describes something that no longer exists. Joblib
    rather than pickle because it is what sklearn's own documentation specifies for
    estimators, and it handles the numpy arrays inside them efficiently.

    A failure here is logged and swallowed. The analysis itself succeeded, and refusing
    to show the user their results because a file could not be written would be the
    wrong trade.
    """
    storage = get_storage()
    key = f"runs/{run_id}/pipeline.joblib"
    try:
        with storage.open(key, "wb") as handle:
            joblib.dump(result.pipeline, handle)
    except Exception:  # noqa: BLE001 - the run is still valid without its artifacts
        log.exception("could not persist the fitted pipeline for run %s", run_id)
        return None

    try:
        prepared = result.X_train.copy()
        prepared[target] = result.y_train
        storage.put(
            f"runs/{run_id}/prepared_train.csv",
            prepared.to_csv(index=False).encode("utf-8"),
        )
    except Exception:  # noqa: BLE001
        log.exception("could not persist prepared data for run %s", run_id)
    return key


def _analyse(
    raw_bytes: bytes,
    filename: str,
    goal: str,
    target: str | None,
    session: Session,
    scaler: str = DEFAULT_SCALER,
    high_card_encoder: str = DEFAULT_HIGH_CARD_ENCODER,
) -> dict[str, Any]:
    """Upload, profile and preprocess in one call — what the single-screen UI wants."""
    dataset, ingested = _ingest_dataset(raw_bytes, filename, session)
    options = PreprocessRequest(
        target=target,
        goal=goal or "",
        scaler=scaler,  # type: ignore[arg-type]
        high_card_encoder=high_card_encoder,  # type: ignore[arg-type]
        test_size=TEST_SIZE,
    )
    return _preprocess(dataset, ingested, options, session)


# --------------------------------------------------------------------------------------
# Lookups
# --------------------------------------------------------------------------------------


def _get_dataset(dataset_id: str, session: Session) -> Dataset:
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="No dataset with that id.")
    return dataset


def _get_run(run_id: str, session: Session) -> PreprocessingRun:
    run = session.get(PreprocessingRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="No preprocessing run with that id.")
    return run


def _latest_run(dataset_id: str, session: Session) -> PreprocessingRun:
    run = session.scalars(
        select(PreprocessingRun)
        .where(PreprocessingRun.dataset_id == dataset_id)
        .order_by(PreprocessingRun.created_at.desc())
    ).first()
    if run is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "That dataset has not been preprocessed yet. POST to its /preprocess "
                "endpoint first."
            ),
        )
    return run


# --------------------------------------------------------------------------------------
# Dataset routes
# --------------------------------------------------------------------------------------


@app.post(
    "/api/datasets",
    response_model=DatasetDetail,
    status_code=201,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
    summary="Upload a dataset",
)
async def create_dataset(
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> Any:
    """Store an uploaded spreadsheet and record what it is. Nothing is fitted here."""
    raw = await _read_capped(file)
    dataset, ingested = _ingest_dataset(raw, file.filename or "upload.csv", session)
    document = profiling.build_profile(
        ingested.frame, duplicate_rows=ingested.duplicate_rows
    )
    _store_profile(dataset, ingested, document, session)
    session.commit()
    return _dataset_detail(dataset, ingested, session)


def _dataset_detail(
    dataset: Dataset, ingested: IngestResult | None, session: Session
) -> dict[str, Any]:
    columns: list[str] = []
    if ingested is not None:
        columns = [str(c) for c in ingested.frame.columns]
    elif dataset.profile is not None:
        columns = [record["column"] for record in dataset.profile.payload.get("columns", [])]

    return {
        "id": dataset.id,
        "project_id": dataset.project_id,
        "filename": dataset.filename,
        "n_rows": dataset.n_rows,
        "n_columns": dataset.n_columns,
        "size_bytes": dataset.size_bytes,
        "health_score": dataset.health_score,
        "created_at": dataset.created_at,
        "encoding": dataset.encoding,
        "delimiter": dataset.delimiter,
        "cleaning": list(dataset.cleaning or []),
        "columns": columns,
        "runs": dataset.runs,
    }


@app.get(
    "/api/datasets",
    response_model=list[DatasetSummary],
    summary="List uploaded datasets",
)
async def list_datasets(
    limit: int = 50, session: Session = Depends(get_session)
) -> Any:
    """Most recently uploaded first."""
    return session.scalars(
        select(Dataset).order_by(Dataset.created_at.desc()).limit(min(max(limit, 1), 200))
    ).all()


@app.get(
    "/api/datasets/{dataset_id}",
    response_model=DatasetDetail,
    responses={404: {"model": ErrorResponse}},
    summary="Fetch one dataset",
)
async def get_dataset(dataset_id: str, session: Session = Depends(get_session)) -> Any:
    dataset = _get_dataset(dataset_id, session)
    return _dataset_detail(dataset, None, session)


@app.get(
    "/api/datasets/{dataset_id}/profile",
    response_model=ProfileResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Profile a dataset",
)
async def get_profile(dataset_id: str, session: Session = Depends(get_session)) -> Any:
    """Per-column measurements, duplicates, correlated pairs and the health score.

    Served from the stored profile when there is one. A dataset uploaded before this
    was recorded is re-read and profiled on the spot rather than refused.
    """
    dataset = _get_dataset(dataset_id, session)
    ingested = _reload(dataset)

    stored = dataset.profile
    if stored is None:
        document = profiling.build_profile(
            ingested.frame, duplicate_rows=ingested.duplicate_rows
        )
        _store_profile(dataset, ingested, document, session)
        session.commit()
    else:
        document = stored.payload

    return _profile_response(dataset, ingested, document, document["n_rows"])


@app.post(
    "/api/datasets/{dataset_id}/preprocess",
    response_model=AnalysisResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    summary="Run the preprocessing pipeline",
)
async def preprocess_dataset(
    dataset_id: str,
    options: PreprocessRequest | None = None,
    session: Session = Depends(get_session),
) -> Any:
    """Split, screen, fit and explain — persisting the run, its log and its pipeline.

    Running this twice with different options produces two runs against the same
    dataset rather than overwriting the first.
    """
    dataset = _get_dataset(dataset_id, session)
    ingested = _reload(dataset)
    payload = _preprocess(dataset, ingested, options or PreprocessRequest(), session)
    session.commit()
    return payload


@app.get(
    "/api/datasets/{dataset_id}/preprocess/log",
    response_model=LogResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Read the glass-box log",
)
async def get_preprocess_log(
    dataset_id: str,
    run_id: str | None = None,
    session: Session = Depends(get_session),
) -> Any:
    """The ordered log of one run — the most recent unless a ``run_id`` is given."""
    _get_dataset(dataset_id, session)
    run = _get_run(run_id, session) if run_id else _latest_run(dataset_id, session)
    if run.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail="That run belongs to another dataset.")

    return {
        "run_id": run.id,
        "dataset_id": run.dataset_id,
        "target": run.target,
        "task": run.task,
        "entries": run.logs,
    }


@app.get(
    "/api/runs/{run_id}",
    response_model=AnalysisResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Re-read a completed run",
)
async def get_run(run_id: str, session: Session = Depends(get_session)) -> Any:
    """The stored analysis document, exactly as first computed. Nothing is refitted."""
    return _get_run(run_id, session).payload


# --------------------------------------------------------------------------------------
# Analysis shortcuts — upload and preprocess in one call
# --------------------------------------------------------------------------------------


@app.post(
    "/api/analyse",
    response_model=AnalysisResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
    summary="Upload and analyse in one call",
)
async def analyse(
    file: UploadFile = File(...),
    goal: str = Form(""),
    target: str | None = Form(None),
    scaler: str = Form(DEFAULT_SCALER),
    high_card_encoder: str = Form(DEFAULT_HIGH_CARD_ENCODER),
    session: Session = Depends(get_session),
) -> Any:
    """Analyse an uploaded spreadsheet and return the full explanation."""
    raw = await _read_capped(file)
    payload = _analyse(
        raw,
        file.filename or "upload.csv",
        goal,
        target or None,
        session,
        scaler=scaler,
        high_card_encoder=high_card_encoder,
    )
    session.commit()
    return payload


@app.post(
    "/api/example",
    response_model=AnalysisResponse,
    responses={400: {"model": ErrorResponse}},
    summary="Analyse the bundled example",
)
async def example(
    goal: str = Form("which students will fail"),
    target: str | None = Form(None),
    scaler: str = Form(DEFAULT_SCALER),
    high_card_encoder: str = Form(DEFAULT_HIGH_CARD_ENCODER),
    session: Session = Depends(get_session),
) -> Any:
    """Analyse the bundled sample file.

    Read from disk on the server so the demo works with the network unplugged and with
    nothing in the browser's cache.
    """
    if not SAMPLE_PATH.exists():  # pragma: no cover - the file ships with the app
        raise UserFacingError("The example file is missing from this install.")
    payload = _analyse(
        SAMPLE_PATH.read_bytes(),
        SAMPLE_PATH.name,
        goal,
        target or None,
        session,
        scaler=scaler,
        high_card_encoder=high_card_encoder,
    )
    session.commit()
    return payload


# --------------------------------------------------------------------------------------
# Downloads
# --------------------------------------------------------------------------------------


@app.get(
    "/api/runs/{run_id}/data.csv",
    responses={404: {"model": ErrorResponse}},
    summary="Download the prepared training data",
)
async def download_data(run_id: str, session: Session = Depends(get_session)) -> PlainTextResponse:
    """The cleaned training data: every prepared feature, plus the column being predicted."""
    run = _get_run(run_id, session)
    cached = _RUNS.get(run_id)

    if cached is not None:
        frame = cached.result.X_train.copy()
        frame[cached.target] = cached.result.y_train
        body = frame.to_csv(index=False)
        name = cached.filename
    else:
        # Not in the cache, so read back what was written when the run was made rather
        # than refitting the pipeline to reproduce it.
        try:
            body = get_storage().get(f"runs/{run_id}/prepared_train.csv").decode("utf-8")
        except StorageError as exc:
            raise HTTPException(
                status_code=410,
                detail="The prepared data for that run is no longer stored. Run it again.",
            ) from exc
        name = run.dataset.filename

    stem = _safe_filename(name, "data")
    return PlainTextResponse(
        body,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{stem}_prepared.csv"'},
    )


@app.get(
    "/api/runs/{run_id}/script.py",
    responses={404: {"model": ErrorResponse}},
    summary="Download reproducing Python",
)
async def download_script(run_id: str, session: Session = Depends(get_session)) -> PlainTextResponse:
    """Python that reproduces this preparation exactly, split included."""
    run = _get_run(run_id, session)
    cached = _RUNS.get(run_id)

    if cached is None:
        # Rebuild by re-reading the stored upload and refitting with the same options.
        # Deterministic given the same bytes, seed and options, so the script it
        # produces is the one that describes this run.
        dataset = run.dataset
        ingested = _reload(dataset)
        options = PreprocessRequest(
            target=run.target,
            scaler=run.scaler,  # type: ignore[arg-type]
            high_card_encoder=run.high_card_encoder,  # type: ignore[arg-type]
            test_size=run.test_size,
        )
        _preprocess(dataset, ingested, options, session)
        session.commit()
        # _preprocess caches under a new run id; reuse its artefacts for this download.
        cached = next(reversed(_RUNS.values()))

    # The script must reproduce the *cleaned* frame, so it carries the same reader
    # settings and the same format repairs the upload needed.
    block, call = loader_source(cached.ingested)
    source = export_script(
        cached.result,
        csv=cached.filename,
        exp_id=run_id[:8],
        seed=run.seed,
        loader_block=block,
        loader_call=call,
    )
    return PlainTextResponse(
        source,
        media_type="text/x-python",
        headers={"Content-Disposition": 'attachment; filename="prepare_data.py"'},
    )


@app.get("/api/health", summary="Liveness and counters")
async def health(session: Session = Depends(get_session)) -> dict[str, Any]:
    return {
        "ok": True,
        "runs_cached": len(_RUNS),
        "datasets": session.scalar(select(func.count()).select_from(Dataset)) or 0,
        "runs": session.scalar(select(func.count()).select_from(PreprocessingRun)) or 0,
    }
