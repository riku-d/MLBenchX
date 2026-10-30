"""The interactive workspace: configure, generate, compile, compare, download.

These routes are what the three-panel interface talks to. They sit alongside the
resource routes in :mod:`app.main` rather than replacing them — the earlier endpoints
describe what the *engine* did to a dataset, and these describe what the *user* is doing
with it.

The contract the interface depends on
-------------------------------------
Every response that changes the workspace returns the whole workspace back. The client
therefore never has to reason about what its edit implied, or reassemble state from a
patch; it renders what it is given. That is what keeps the toggles, the editor and the
health panel from drifting apart after a request fails halfway.

One rule runs through all of it: **the user is never told something the app has not
measured.** An impact preview runs the pipeline both ways rather than guessing at a
score. A health delta compares two real assessments. When the code has been hand-edited,
the response says the configuration no longer describes what will run, instead of
quietly showing toggles that have stopped meaning anything.
"""

from __future__ import annotations

import io
import json
import logging
import uuid
import zipfile
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.codegen import SourceFormat, explain_step, generate
from app.config import MAX_HISTORY_ENTRIES, PREVIEW_PAGE_SIZE
from app.db import Compilation, Dataset, HistoryEntry, Workspace, get_session
from app.executor import ExecutionError, run_pipeline, sandbox_status
from app.pipeline import STEP_LABELS, STEP_ORDER, PipelineConfig
from app.services import impact
from app.services.health import assess, compare
from app.services.ingest import IngestError, read_upload
from app.storage import StorageError, get_storage

router = APIRouter(prefix="/api/workspaces", tags=["workspace"])
log = logging.getLogger("mlbenchx")


# --------------------------------------------------------------------------------------
# Request shapes
# --------------------------------------------------------------------------------------


class ConfigUpdate(BaseModel):
    """A new pipeline configuration, plus what to call the change in the history."""

    config: PipelineConfig
    summary: str = Field(default="", max_length=200)


class CodeUpdate(BaseModel):
    """Code straight from the editor."""

    code: str = Field(max_length=2_000_000)


class CompileRequest(BaseModel):
    """Which source to run.

    ``auto`` means "the honest one": hand-edited code if there is any, otherwise code
    generated fresh from the configuration.
    """

    source: str = Field(default="auto", pattern="^(auto|pipeline|editor)$")


class ImpactRequest(BaseModel):
    step: str
    enabled: bool = False


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _dataset_bytes(dataset: Dataset) -> bytes:
    try:
        return get_storage().get(dataset.storage_key)
    except StorageError as exc:
        raise HTTPException(
            status_code=410,
            detail="The stored copy of that file is gone. Upload it again.",
        ) from exc


def _frame(dataset: Dataset) -> pd.DataFrame:
    """The dataset as uploaded, re-read.

    Reading again rather than caching a frame: ingestion is deterministic, so the result
    is identical, and a cached frame is one more thing that can go stale.
    """
    try:
        return read_upload(_dataset_bytes(dataset), dataset.filename).frame
    except IngestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _get_workspace(workspace_id: str, session: Session) -> Workspace:
    workspace = session.get(Workspace, workspace_id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="No workspace with that id.")
    return workspace


def _config_of(workspace: Workspace) -> PipelineConfig:
    return PipelineConfig.model_validate(workspace.config)


def _record(workspace: Workspace, kind: str, summary: str, session: Session,
            config: dict | None = None, code: str | None = None) -> None:
    """Append to the change history, trimming the oldest beyond the cap."""
    session.add(HistoryEntry(
        workspace_id=workspace.id, kind=kind, summary=summary,
        config=config, code=code,
    ))
    session.flush()
    entries = session.scalars(
        select(HistoryEntry)
        .where(HistoryEntry.workspace_id == workspace.id)
        .order_by(HistoryEntry.created_at.desc())
    ).all()
    for stale in entries[MAX_HISTORY_ENTRIES:]:
        session.delete(stale)


def _steps_view(frame: pd.DataFrame, config: PipelineConfig) -> list[dict[str, Any]]:
    """Every step as the pipeline panel needs it: state, reason, and what it is doing.

    The numbers are recomputed against the current dataset on every read rather than
    cached, because a step card that quotes a stale count is worse than one that quotes
    none — it looks authoritative and is wrong.
    """
    view: list[dict[str, Any]] = []
    for step_id in STEP_ORDER:
        label, description = STEP_LABELS[step_id]
        step = config.step(step_id)
        facts = impact.step_facts(frame, config, step_id)
        active = step.enabled
        view.append({
            "id": step_id,
            "label": label,
            "description": description,
            "enabled": active,
            "locked": step_id == "data_types",
            "explanation": explain_step(step_id, config),
            "summary": impact.summarise_enabled(frame, config, step_id),
            "facts": facts,
            "options": step.model_dump(exclude={"enabled"}),
            "status": (
                "off" if not active
                else "idle" if not (facts.get("values_affected") or facts.get("rows_affected")
                                    or facts.get("columns_affected"))
                else "active"
            ),
        })
    return view


def _source_of(workspace: Workspace) -> SourceFormat:
    """How the generated script must read this workspace's file.

    The script is handed the original upload, so it has to parse it with the separator
    and encoding ingestion detected. Assuming a comma turns a semicolon export into one
    column per row.
    """
    return SourceFormat.from_dataset(workspace.dataset)


def _serialise(workspace: Workspace, frame: pd.DataFrame, session: Session) -> dict[str, Any]:
    """The whole workspace, as every mutating route returns it."""
    config = _config_of(workspace)
    generated = generate(config, _source_of(workspace))
    latest = session.scalars(
        select(Compilation)
        .where(Compilation.workspace_id == workspace.id)
        .order_by(Compilation.created_at.desc())
    ).first()

    dirty = workspace.code.strip() != (
        latest.code.strip() if latest else ""
    )

    return {
        "id": workspace.id,
        "dataset": {
            "id": workspace.dataset.id,
            "filename": workspace.dataset.filename,
            "rows": workspace.dataset.n_rows,
            "columns": workspace.dataset.n_columns,
            "size_bytes": workspace.dataset.size_bytes,
            # What ingestion detected. Shown so the user can confirm the file was read
            # the way they expect — a wrong separator is silent until it is displayed.
            "delimiter": workspace.dataset.delimiter,
            "encoding": workspace.dataset.encoding,
        },
        "config": config.model_dump(),
        "code": workspace.code,
        "generated_code": generated,
        "code_is_edited": workspace.code_is_edited,
        # When the code has been hand-edited, the toggles describe a pipeline that is no
        # longer the one that will run. Saying so is the whole reason this flag exists.
        "code_is_authoritative": workspace.code_is_edited,
        "has_uncompiled_changes": dirty,
        "steps": _steps_view(frame, config),
        "step_order": list(STEP_ORDER),
        "baseline_health": workspace.baseline_health,
        "latest_compilation": _compilation_view(latest) if latest else None,
        "sandbox": sandbox_status(),
        "columns": [str(c) for c in frame.columns],
        "updated_at": workspace.updated_at.isoformat(),
    }


def _compilation_view(compilation: Compilation) -> dict[str, Any]:
    return {
        "id": compilation.id,
        "status": compilation.status,
        "source": compilation.source,
        "duration_ms": compilation.duration_ms,
        "logs": compilation.logs,
        "error": compilation.error,
        "result": compilation.result,
        "health_before": compilation.health_before,
        "health_after": compilation.health_after,
        "sandbox": compilation.sandbox,
        "created_at": compilation.created_at.isoformat(),
    }


def _code_to_run(workspace: Workspace, source: str) -> tuple[str, str]:
    """The code that will actually execute, and where it came from."""
    config = _config_of(workspace)
    if source == "pipeline":
        return generate(config, _source_of(workspace)), "pipeline"
    if source == "editor":
        return workspace.code, "editor"
    if workspace.code_is_edited and workspace.code.strip():
        return workspace.code, "editor"
    return generate(config, _source_of(workspace)), "pipeline"


# --------------------------------------------------------------------------------------
# Creating and reading a workspace
# --------------------------------------------------------------------------------------


def _workspace_of(dataset_id: str, session: Session) -> Workspace | None:
    return session.scalars(
        select(Workspace).where(Workspace.dataset_id == dataset_id)
    ).first()


@router.post("/from-dataset/{dataset_id}", status_code=201, summary="Open a workspace")
def create_workspace(
    dataset_id: str, response: Response, session: Session = Depends(get_session)
) -> Any:
    """Open the workspace for a dataset, creating it on first use.

    Idempotent: a dataset has exactly one workspace, and asking twice returns the same
    one rather than discarding the work in it. 201 when this call created it, 200 when it
    already existed.

    The existence check alone is not enough. Building a workspace re-reads the dataset
    and measures its health, which takes long enough for a second request — React
    StrictMode sends one on every mount in development — to pass the same check. The
    unique index on ``dataset_id`` then rejects the second insert, and that rejection is
    answered by returning the workspace the first request made.
    """
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="No dataset with that id.")

    existing = _workspace_of(dataset_id, session)
    frame = _frame(dataset)

    if existing is not None:
        response.status_code = 200
        return _serialise(existing, frame, session)

    config = PipelineConfig(target=_guess_target(frame))
    workspace = Workspace(
        id=uuid.uuid4().hex,
        dataset_id=dataset_id,
        config=config.model_dump(),
        code=generate(config, SourceFormat.from_dataset(dataset)),
        code_is_edited=False,
        baseline_health=dict(assess(frame, config.target)),
    )
    log.info("Creating workspace for dataset %s", dataset_id)
    session.add(workspace)
    try:
        session.flush()
    except IntegrityError:
        # Only a concurrent request for the same dataset is expected here, and it leaves
        # a workspace behind. Anything else — no row for this dataset after rolling
        # back — is not the race, and is re-raised untouched.
        session.rollback()
        winner = _workspace_of(dataset_id, session)
        if winner is None:
            raise
        log.info(
            "Concurrent workspace creation for dataset %s; returning workspace %s",
            dataset_id, winner.id,
        )
        response.status_code = 200
        return _serialise(winner, frame, session)
    _record(workspace, "created", f"Opened {dataset.filename}", session,
            config=workspace.config)
    session.commit()
    return _serialise(workspace, frame, session)


def _guess_target(frame: pd.DataFrame) -> str | None:
    """A first guess at the column to predict, always shown and always changeable.

    Reuses the same heuristic the rest of the app uses, so the workspace and the
    single-call endpoints do not disagree about what looks like a label.
    """
    from app.services.target import detect_target

    try:
        return detect_target(frame, "", None).column
    except ValueError:
        return None


@router.get("/{workspace_id}", summary="Read a workspace")
def read_workspace(workspace_id: str, session: Session = Depends(get_session)) -> Any:
    workspace = _get_workspace(workspace_id, session)
    return _serialise(workspace, _frame(workspace.dataset), session)


@router.get("/{workspace_id}/health", summary="Health of the dataset as uploaded")
def read_health(workspace_id: str, session: Session = Depends(get_session)) -> Any:
    """The six dimensions, with the evidence and arithmetic behind each."""
    workspace = _get_workspace(workspace_id, session)
    return workspace.baseline_health or dict(
        assess(_frame(workspace.dataset), _config_of(workspace).target)
    )


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@router.put("/{workspace_id}/config", summary="Update the pipeline configuration")
def update_config(
    workspace_id: str, update: ConfigUpdate, session: Session = Depends(get_session)
) -> Any:
    """Replace the configuration and regenerate the code from it.

    Regenerating overwrites hand-edited code, so it only happens when the code has *not*
    been edited. Silently discarding someone's work because they flipped a toggle would
    be the worst thing this app could do; instead the edit is kept, and the response
    says the configuration no longer describes what will run.
    """
    workspace = _get_workspace(workspace_id, session)
    frame = _frame(workspace.dataset)

    previous = _config_of(workspace)
    workspace.config = update.config.model_dump()
    if not workspace.code_is_edited:
        workspace.code = generate(update.config, _source_of(workspace))
    workspace.updated_at = _now()

    summary = update.summary or _describe_change(previous, update.config)
    _record(workspace, "config", summary, session, config=workspace.config)
    session.commit()
    return _serialise(workspace, frame, session)


def _describe_change(before: PipelineConfig, after: PipelineConfig) -> str:
    """A sentence for the history panel, worked out from what actually differs."""
    changes: list[str] = []
    for step_id in STEP_ORDER:
        was, now = before.step(step_id), after.step(step_id)
        label = STEP_LABELS[step_id][0]
        if was.enabled != now.enabled:
            changes.append(f"{'Enabled' if now.enabled else 'Disabled'} {label}")
            continue
        old_options = was.model_dump(exclude={"enabled"})
        new_options = now.model_dump(exclude={"enabled"})
        for key, value in new_options.items():
            if old_options.get(key) != value:
                changes.append(f"{label}: {key} → {value}")
    if before.target != after.target:
        changes.append(f"Target → {after.target}")
    return "; ".join(changes[:3]) if changes else "Updated the pipeline"


# --------------------------------------------------------------------------------------
# The editor
# --------------------------------------------------------------------------------------


@router.put("/{workspace_id}/code", summary="Save editor contents")
def update_code(
    workspace_id: str, update: CodeUpdate, session: Session = Depends(get_session)
) -> Any:
    """Store what is in the editor.

    Code matching the generated output exactly is not an edit — it is the user typing
    their way back to where they started, or reverting. Treating that as an edit would
    leave the workspace permanently flagged as diverged for no reason.
    """
    workspace = _get_workspace(workspace_id, session)
    generated = generate(_config_of(workspace), _source_of(workspace))

    workspace.code = update.code
    workspace.code_is_edited = update.code.strip() != generated.strip()
    workspace.updated_at = _now()
    session.commit()
    return _serialise(workspace, _frame(workspace.dataset), session)


@router.post("/{workspace_id}/code/reset", summary="Discard edits")
def reset_code(workspace_id: str, session: Session = Depends(get_session)) -> Any:
    """Throw away hand edits and regenerate from the configuration."""
    workspace = _get_workspace(workspace_id, session)
    workspace.code = generate(_config_of(workspace), _source_of(workspace))
    workspace.code_is_edited = False
    workspace.updated_at = _now()
    _record(workspace, "code", "Reset the code to match the pipeline", session,
            config=workspace.config)
    session.commit()
    return _serialise(workspace, _frame(workspace.dataset), session)


# --------------------------------------------------------------------------------------
# Impact preview
# --------------------------------------------------------------------------------------


@router.post("/{workspace_id}/impact", summary="What a toggle would cost")
def preview_impact(
    workspace_id: str, request: ImpactRequest, session: Session = Depends(get_session)
) -> Any:
    """What changing this step would do, measured rather than guessed.

    The consequences come from the dataset directly. The projected health score comes
    from actually running the pipeline the other way and assessing the result — which
    costs a second or so, and is the difference between a warning the user can trust and
    a number invented to look precise.
    """
    workspace = _get_workspace(workspace_id, session)
    if request.step not in STEP_ORDER:
        raise HTTPException(status_code=422, detail=f"Unknown step: {request.step}")

    frame = _frame(workspace.dataset)
    config = _config_of(workspace)
    report = impact.assess_disable(frame, config, request.step)

    proposed = config.model_copy(deep=True)
    getattr(proposed, request.step).enabled = request.enabled

    current = _score_of(config, workspace)
    projected = _score_of(proposed, workspace)

    report["current_score"] = current
    report["projected_score"] = projected
    report["delta"] = None if current is None or projected is None else projected - current
    report["enabling"] = request.enabled
    return report


def _score_of(config: PipelineConfig, workspace: Workspace) -> int | None:
    """Health of the dataset after running this configuration.

    Returns ``None`` when the pipeline fails, which is itself information: the modal
    shows "this configuration does not run" rather than a number that would imply it
    does.
    """
    dataset_bytes = _dataset_bytes(workspace.dataset)
    try:
        result = run_pipeline(generate(config, _source_of(workspace)), dataset_bytes)
    except ExecutionError:
        return None
    if not result.ok or result.frame is None:
        return None
    return assess(result.frame, config.target).score


# --------------------------------------------------------------------------------------
# Compiling
# --------------------------------------------------------------------------------------


@router.post("/{workspace_id}/compile", summary="Run the pipeline")
def compile_pipeline(
    workspace_id: str,
    request: CompileRequest | None = None,
    session: Session = Depends(get_session),
) -> Any:
    """Execute the pipeline in the sandbox and record everything it produced.

    A failing run is stored like any other. The user needs to read the error, compare it
    against the run before, and come back to it — none of which is possible if failures
    are discarded.
    """
    workspace = _get_workspace(workspace_id, session)
    config = _config_of(workspace)
    code, source = _code_to_run(workspace, (request or CompileRequest()).source)

    frame = _frame(workspace.dataset)
    before = workspace.baseline_health or dict(assess(frame, config.target))

    try:
        execution = run_pipeline(code, _dataset_bytes(workspace.dataset))
    except ExecutionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    compilation = Compilation(
        id=uuid.uuid4().hex,
        workspace_id=workspace.id,
        status="success" if execution.ok else "error",
        code=code,
        config=None if source == "editor" else workspace.config,
        source=source,
        duration_ms=execution.duration_ms,
        logs=[{"text": line} for line in execution.logs],
        error=execution.error,
        sandbox=execution.sandbox,
        health_before=before,
    )

    if execution.ok and execution.frame is not None:
        after = dict(assess(execution.frame, config.target))
        key = f"runs/{compilation.id}/processed.csv"
        get_storage().put(key, execution.frame.to_csv(index=False).encode("utf-8"))
        compilation.output_key = key
        compilation.health_after = after
        compilation.result = {
            "rows_before": int(len(frame)),
            "rows_after": int(len(execution.frame)),
            "columns_before": int(frame.shape[1]),
            "columns_after": int(execution.frame.shape[1]),
            "steps": execution.report.get("steps", []),
            "train_rows": execution.report.get("train_rows"),
            "test_rows": execution.report.get("test_rows"),
            # How the held-out rows were chosen. Shown in the interface because
            # "we split before fitting" is the product's central claim, and a claim
            # the user cannot see the evidence for is just a sentence.
            "split": {
                "train_rows": execution.report.get("train_rows"),
                "test_rows": execution.report.get("test_rows"),
                "stratified": execution.report.get("stratified"),
                "test_size": execution.report.get("test_size"),
                "seed": execution.report.get("seed"),
            },
            "comparison": compare(before, after),  # type: ignore[arg-type]
        }
        workspace.last_compilation_id = compilation.id

    session.add(compilation)
    _record(
        workspace,
        "compile",
        f"Compiled from the {source} — "
        + (
            f"health {before.get('score')} → {compilation.health_after.get('score')}"
            if execution.ok
            else f"failed: {(execution.error or {}).get('type', 'error')}"
        ),
        session,
        config=workspace.config,
        code=code,
    )
    session.commit()

    return {
        "compilation": _compilation_view(compilation),
        "workspace": _serialise(workspace, frame, session),
    }


@router.get("/{workspace_id}/compilations", summary="Past compilations")
def list_compilations(
    workspace_id: str, limit: int = 20, session: Session = Depends(get_session)
) -> Any:
    workspace = _get_workspace(workspace_id, session)
    rows = session.scalars(
        select(Compilation)
        .where(Compilation.workspace_id == workspace.id)
        .order_by(Compilation.created_at.desc())
        .limit(min(max(limit, 1), 100))
    ).all()
    return [_compilation_view(row) for row in rows]


# --------------------------------------------------------------------------------------
# Data preview
# --------------------------------------------------------------------------------------


@router.get("/{workspace_id}/preview", summary="Rows of the dataset")
def preview_rows(
    workspace_id: str,
    which: str = "original",
    page: int = 1,
    page_size: int = PREVIEW_PAGE_SIZE,
    search: str = "",
    sort: str = "",
    descending: bool = False,
    session: Session = Depends(get_session),
) -> Any:
    """A page of rows, with the cells worth noticing already marked.

    Blanks and outliers are identified server-side rather than in the browser: the
    browser has one page of rows and cannot know where a column's quartiles are, so a
    client-side rule would highlight a different set of cells on every page.
    """
    workspace = _get_workspace(workspace_id, session)

    if which == "processed":
        compilation = session.scalars(
            select(Compilation)
            .where(Compilation.workspace_id == workspace.id,
                   Compilation.status == "success")
            .order_by(Compilation.created_at.desc())
        ).first()
        if compilation is None or not compilation.output_key:
            raise HTTPException(
                status_code=404,
                detail="Nothing has been compiled yet, so there is no processed data to show.",
            )
        try:
            frame = pd.read_csv(io.BytesIO(get_storage().get(compilation.output_key)))
        except StorageError as exc:
            raise HTTPException(status_code=410, detail="That result is no longer stored.") from exc
    else:
        frame = _frame(workspace.dataset)

    if search:
        needle = search.lower()
        mask = frame.apply(
            lambda row: row.astype(str).str.lower().str.contains(needle, na=False).any(), axis=1
        )
        frame = frame[mask]

    if sort and sort in frame.columns:
        frame = frame.sort_values(by=sort, ascending=not descending, kind="stable")

    total = len(frame)
    size = min(max(page_size, 1), 200)
    start = max(page - 1, 0) * size
    window = frame.iloc[start : start + size]

    bounds = _outlier_bounds(frame)
    rows: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(window.iterrows(), start=start):
        cells = {}
        for column in frame.columns:
            value = row[column]
            blank = pd.isna(value)
            low, high = bounds.get(str(column), (None, None))
            outlying = (
                not blank and low is not None
                and isinstance(value, (int, float))
                and (value < low or value > high)
            )
            cells[str(column)] = {
                "value": None if blank else _jsonable(value),
                "missing": bool(blank),
                "outlier": bool(outlying),
            }
        rows.append({"index": position, "cells": cells})

    return {
        "which": which,
        "columns": [
            {
                "name": str(c),
                "dtype": str(frame[c].dtype),
                "kind": _kind_of(frame[c]),
                "missing": int(frame[c].isna().sum()),
            }
            for c in frame.columns
        ],
        "rows": rows,
        "page": page,
        "page_size": size,
        "total_rows": int(total),
        "total_pages": max(1, -(-int(total) // size)),
    }


def _kind_of(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    return "text"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (int, float, str, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    return str(value)


def _outlier_bounds(frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for column in frame.select_dtypes(include="number").columns:
        values = frame[column].dropna()
        if values.empty:
            continue
        q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            continue
        bounds[str(column)] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return bounds


# --------------------------------------------------------------------------------------
# History
# --------------------------------------------------------------------------------------


@router.get("/{workspace_id}/history", summary="Change history")
def read_history(workspace_id: str, session: Session = Depends(get_session)) -> Any:
    workspace = _get_workspace(workspace_id, session)
    entries = session.scalars(
        select(HistoryEntry)
        .where(HistoryEntry.workspace_id == workspace.id)
        .order_by(HistoryEntry.created_at.desc())
    ).all()
    return [
        {
            "id": entry.id,
            "kind": entry.kind,
            "summary": entry.summary,
            "created_at": entry.created_at.isoformat(),
            "restorable": entry.config is not None,
        }
        for entry in entries
    ]


@router.post("/{workspace_id}/history/{entry_id}/restore", summary="Go back to a point")
def restore_history(
    workspace_id: str, entry_id: int, session: Session = Depends(get_session)
) -> Any:
    """Return the pipeline to an earlier state.

    Recorded as a new entry rather than by rewinding the list. The fact that someone
    went back is part of the history, and a restore that erased the entries after it
    would make the panel lie about what happened.
    """
    workspace = _get_workspace(workspace_id, session)
    entry = session.get(HistoryEntry, entry_id)
    if entry is None or entry.workspace_id != workspace.id:
        raise HTTPException(status_code=404, detail="No such history entry.")
    if entry.config is None:
        raise HTTPException(
            status_code=422, detail="That entry has no configuration to restore."
        )

    config = PipelineConfig.model_validate(entry.config)
    workspace.config = entry.config
    _source = _source_of(workspace)
    workspace.code = entry.code or generate(config, _source)
    workspace.code_is_edited = (
        bool(entry.code) and entry.code.strip() != generate(config, _source).strip()
    )
    workspace.updated_at = _now()
    _record(workspace, "restore", f"Restored: {entry.summary}", session, config=entry.config)
    session.commit()
    return _serialise(workspace, _frame(workspace.dataset), session)


# --------------------------------------------------------------------------------------
# Downloads
# --------------------------------------------------------------------------------------


def _latest_success(workspace: Workspace, session: Session) -> Compilation:
    compilation = session.scalars(
        select(Compilation)
        .where(Compilation.workspace_id == workspace.id, Compilation.status == "success")
        .order_by(Compilation.created_at.desc())
    ).first()
    if compilation is None:
        raise HTTPException(
            status_code=404,
            detail="Nothing has compiled successfully yet, so there is nothing to download.",
        )
    return compilation


@router.get("/{workspace_id}/download/{what}", summary="Download an artefact")
def download(
    workspace_id: str, what: str, session: Session = Depends(get_session)
) -> Any:
    """One file from the workspace.

    ``processed`` | ``original`` | ``code`` | ``config`` | ``report`` | ``project``.
    """
    workspace = _get_workspace(workspace_id, session)
    stem = workspace.dataset.filename.rsplit(".", 1)[0][:60] or "dataset"

    if what == "original":
        return Response(
            _dataset_bytes(workspace.dataset),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{stem}_original.csv"'},
        )

    if what == "code":
        code, _ = _code_to_run(workspace, "auto")
        return PlainTextResponse(
            code,
            media_type="text/x-python",
            headers={"Content-Disposition": 'attachment; filename="pipeline.py"'},
        )

    if what == "config":
        return Response(
            json.dumps(workspace.config, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="pipeline_config.json"'},
        )

    if what == "processed":
        compilation = _latest_success(workspace, session)
        return Response(
            get_storage().get(compilation.output_key or ""),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{stem}_processed.csv"'},
        )

    if what == "report":
        compilation = _latest_success(workspace, session)
        return PlainTextResponse(
            _build_report(workspace, compilation),
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{stem}_report.md"'},
        )

    if what == "project":
        compilation = _latest_success(workspace, session)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("original.csv", _dataset_bytes(workspace.dataset))
            archive.writestr("processed.csv", get_storage().get(compilation.output_key or ""))
            archive.writestr("pipeline.py", compilation.code)
            archive.writestr("pipeline_config.json", json.dumps(workspace.config, indent=2))
            archive.writestr("report.md", _build_report(workspace, compilation))
        return Response(
            buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{stem}_project.zip"'},
        )

    raise HTTPException(status_code=422, detail=f"Nothing called {what!r} to download.")


def _build_report(workspace: Workspace, compilation: Compilation) -> str:
    """The transformation report, in Markdown.

    Written to be readable on its own, months later, by someone who was not there: what
    the data was, what was wrong with it, what was done, **what was deliberately not
    done**, and what the result was. That fourth item is the one most reports omit and
    the one that makes the rest interpretable.
    """
    config = _config_of(workspace)
    before = compilation.health_before or {}
    after = compilation.health_after or {}
    result = compilation.result or {}
    lines: list[str] = []

    lines.append(f"# Preprocessing report — {workspace.dataset.filename}")
    lines.append("")
    lines.append(f"Generated {compilation.created_at:%Y-%m-%d %H:%M} UTC by ML-BenchX.")
    lines.append("")

    lines.append("## Dataset")
    lines.append("")
    lines.append("| | Before | After |")
    lines.append("|---|---|---|")
    lines.append(f"| Rows | {result.get('rows_before', '—'):,} | {result.get('rows_after', '—'):,} |")
    lines.append(f"| Columns | {result.get('columns_before', '—')} | {result.get('columns_after', '—')} |")
    lines.append(f"| Health | {before.get('score', '—')} / 100 | {after.get('score', '—')} / 100 |")
    lines.append("")
    lines.append(f"Target column: `{config.target or 'not set'}`")
    lines.append("")

    lines.append("## Health, dimension by dimension")
    lines.append("")
    lines.append("| Dimension | Weight | Before | After | Change |")
    lines.append("|---|---|---|---|---|")
    after_scores = {d["key"]: d["score"] for d in after.get("dimensions", [])}
    for dimension in before.get("dimensions", []):
        now = after_scores.get(dimension["key"])
        delta = "—" if now is None else f"{now - dimension['score']:+.1f}"
        lines.append(
            f"| {dimension['label']} | {dimension['weight']:.2f} | "
            f"{dimension['score']:.1f} | {now if now is None else f'{now:.1f}'} | {delta} |"
        )
    lines.append("")

    lines.append("## Problems found in the original data")
    lines.append("")
    issues = before.get("issues", [])
    if issues:
        for issue in issues:
            lines.append(f"- **{issue['label']}** ({issue['score']:.0f}/100) — {issue['text']}")
    else:
        lines.append("None worth reporting.")
    lines.append("")

    lines.append("## Steps applied")
    lines.append("")
    enabled = config.enabled_steps()
    for step_id in enabled:
        label = STEP_LABELS[step_id][0]
        lines.append(f"### {label}")
        lines.append("")
        lines.append(explain_step(step_id, config))
        lines.append("")

    lines.append("## Steps deliberately not applied")
    lines.append("")
    disabled = config.disabled_steps()
    if disabled:
        lines.append(
            "These were switched off. Anything they would have handled is still present "
            "in the processed data."
        )
        lines.append("")
        for step_id in disabled:
            lines.append(f"- **{STEP_LABELS[step_id][0]}** — {STEP_LABELS[step_id][1]}")
    else:
        lines.append("None — every step was applied.")
    lines.append("")

    if compilation.source == "editor":
        lines.append("## Note on the code")
        lines.append("")
        lines.append(
            "This pipeline was run from hand-edited code, so the configuration above "
            "describes the starting point rather than exactly what executed. The code "
            "that ran is included below and in the downloaded project."
        )
        lines.append("")

    lines.append("## Execution")
    lines.append("")
    lines.append(f"- Source: {compilation.source}")
    lines.append(f"- Duration: {compilation.duration_ms} ms")
    lines.append(f"- Isolation: {compilation.sandbox}")
    lines.append("")

    lines.append("## The code that ran")
    lines.append("")
    lines.append("```python")
    lines.append(compilation.code)
    lines.append("```")
    lines.append("")

    return "\n".join(lines)
