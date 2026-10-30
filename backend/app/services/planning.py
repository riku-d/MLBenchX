"""Propose, store, edit and execute a preprocessing plan.

:mod:`app.plan` decides and :mod:`app.runner` executes. Neither knows about databases,
storage or HTTP, and that is deliberate — it is what lets the planner be tested with a
DataFrame and nothing else. This module is the seam: it turns a stored dataset into a
plan, keeps the versions, runs one of them, and writes down what came out.

Three rules shape it.

**A stored plan is immutable.** Editing one writes a new row with the next version
rather than changing the row a run may already point at. "This run used that plan" then
survives every later edit, which is the whole reason the run records a version number
instead of a foreign key alone.

**A run is consistent or it is failed.** Artifacts are written first and the row is
marked ``succeeded`` only once every one of them is stored, in a single commit. A row
therefore never names a file that is not there. If anything raises along the way the run
is recorded as ``failed`` with the error — kept, not deleted, because a user needs to
read why.

**Nothing is re-decided at execution time.** The run executes the plan document exactly
as stored. Re-proposing from the dataset at execution would mean the user approved one
thing and another ran.
"""

from __future__ import annotations

import io
import json
import uuid
from datetime import datetime, timezone
from typing import Any

import joblib
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import Dataset, Plan, Run
from app.plan import GlobalOptions, PreprocessingPlan, propose
from app.runner import RunResult, execute
from app.services.ingest import IngestResult
from app.services.screening import identifier_screen, target_leakage_screen
from app.services.target import describe_task, detect_target
from app.storage import get_storage

__all__ = [
    "ARTIFACT_NAMES",
    "PlanningError",
    "propose_for_dataset",
    "store_plan",
    "current_plan",
    "plan_object",
    "apply_overrides",
    "execute_plan",
    "artifact_key",
    "delete_run_artifacts",
]

#: Artifact name -> the filename it is stored under. The names are the contract: §6 of
#: the spec asks for these by name, and `GET /api/runs/{id}/artifacts/{name}` serves
#: exactly this set and nothing else, so a traversal attempt cannot name a file outside
#: it — the name is looked up here rather than interpolated into a path.
ARTIFACT_NAMES: dict[str, str] = {
    "plan": "plan.json",
    "schema": "schema.json",
    "log": "log.json",
    "pipeline": "pipeline.joblib",
    "train": "train.parquet",
    "test": "test.parquet",
    "script": "pipeline.py",
}

#: Which media type each artifact is served as.
ARTIFACT_MEDIA: dict[str, str] = {
    "plan": "application/json",
    "schema": "application/json",
    "log": "application/json",
    "pipeline": "application/octet-stream",
    "train": "application/vnd.apache.parquet",
    "test": "application/vnd.apache.parquet",
    "script": "text/x-python",
}


class PlanningError(Exception):
    """Something the user can fix, phrased as a sentence to show them."""


# --------------------------------------------------------------------------------------
# Proposing
# --------------------------------------------------------------------------------------


def propose_for_dataset(
    ingested: IngestResult,
    goal: str = "",
    target: str | None = None,
    options: GlobalOptions | None = None,
) -> PreprocessingPlan:
    """Work out the target, then propose a plan for every column of the dataset.

    The screens run inside :func:`~app.plan.propose`, on its training fold, for the
    reason that function's docstring gives: a screen is a measurement, and a measurement
    taken over the held-out rows is a decision informed by data the model will be tested
    on.
    """
    frame = ingested.frame
    try:
        guess = detect_target(frame, goal or "", target)
    except ValueError as exc:
        raise PlanningError(str(exc)) from exc

    if frame[guess.column].notna().sum() == 0:
        raise PlanningError(
            f"Every row is blank in {guess.column}, so there is nothing to predict. "
            "Choose a different column."
        )

    task = describe_task(frame[guess.column])

    try:
        return propose(
            frame,
            target=guess.column,
            task=task,
            options=options,
            column_mapping=ingested.column_mapping,
            screeners=[identifier_screen, target_leakage_screen],
        )
    except ValueError as exc:
        raise PlanningError(str(exc)) from exc


# --------------------------------------------------------------------------------------
# Storing and versioning
# --------------------------------------------------------------------------------------


def _next_version(session: Session, dataset_id: str) -> int:
    rows = session.scalars(
        select(Plan.version).where(Plan.dataset_id == dataset_id)
    ).all()
    return (max(rows) + 1) if rows else 1


def store_plan(
    session: Session,
    dataset: Dataset,
    plan: PreprocessingPlan,
    parent: Plan | None = None,
    origin: str = "proposed",
) -> Plan:
    """Write a plan as a new, immutable version of this dataset's plan.

    ``version`` is one past the highest this dataset has, so versions are dense and
    ordered per dataset rather than globally — a user reading "version 3" is counting
    their own edits, not everyone's.
    """
    document = plan.model_dump(mode="json")
    now = datetime.now(timezone.utc)
    row = Plan(
        id=uuid.uuid4().hex,
        dataset_id=dataset.id,
        version=_next_version(session, dataset.id),
        parent_id=parent.id if parent else None,
        target=plan.target,
        task=plan.task,
        origin=origin,
        n_columns=len(plan.columns),
        n_overridden=sum(1 for column in plan.columns if column.overridden),
        document=document,
        updated_at=now,
    )
    session.add(row)
    session.flush()
    return row


def current_plan(session: Session, dataset_id: str) -> Plan | None:
    """The newest version of this dataset's plan, or ``None`` if none was proposed."""
    return session.scalars(
        select(Plan)
        .where(Plan.dataset_id == dataset_id)
        .order_by(Plan.version.desc())
        .limit(1)
    ).first()


def plan_versions(session: Session, dataset_id: str) -> list[Plan]:
    """Every version of this dataset's plan, oldest first."""
    return list(
        session.scalars(
            select(Plan).where(Plan.dataset_id == dataset_id).order_by(Plan.version)
        ).all()
    )


def plan_object(row: Plan) -> PreprocessingPlan:
    """Reconstruct the domain object from a stored row.

    Validated on the way back rather than trusted: a document that cannot be rebuilt
    into a plan must fail here, where the error names the field, rather than halfway
    through an execution.
    """
    try:
        return PreprocessingPlan.model_validate(row.document)
    except Exception as exc:  # pragma: no cover - a corrupt row is not reachable by API
        raise PlanningError(
            f"Plan {row.id} (version {row.version}) could not be read back: {exc}"
        ) from exc


def apply_overrides(
    plan: PreprocessingPlan,
    columns: dict[str, dict[str, Any]] | None = None,
    options: dict[str, Any] | None = None,
) -> PreprocessingPlan:
    """Return a new plan with the user's changes applied, or raise on an invalid one.

    Works on a copy. The caller stores the result as a new version, and the plan it was
    derived from is left exactly as it was — which is what makes an earlier run's
    reference to it still true.

    Every override is validated by :meth:`~app.plan.PreprocessingPlan.override`, which
    refuses an action that makes no sense for the column's kind. Validation happens here,
    before anything is stored, so an invalid edit is a 400 rather than a stored plan that
    fails at execution with a message about dtypes.
    """
    edited = plan.model_copy(deep=True)

    for name, change in (columns or {}).items():
        action = change.get("action")
        if not action:
            raise PlanningError(f"No action given for column {name!r}.")
        try:
            edited.override(name, action, change.get("params"))
        except KeyError as exc:
            raise PlanningError(str(exc).strip("'")) from exc
        except ValueError as exc:
            raise PlanningError(str(exc)) from exc

    if options:
        try:
            edited.options = GlobalOptions.model_validate({
                **edited.options.model_dump(),
                **options,
            })
        except Exception as exc:
            raise PlanningError(f"Those pipeline options are not valid: {exc}") from exc

    return edited


# --------------------------------------------------------------------------------------
# Executing
# --------------------------------------------------------------------------------------


def artifact_key(run_id: str, name: str) -> str:
    """The storage key for one artifact of one run.

    Both halves are constrained: ``run_id`` is a 32-character hex uuid this module
    generated, and ``name`` is looked up in :data:`ARTIFACT_NAMES` rather than
    interpolated, so a caller cannot name a path. ``storage`` validates the key again on
    the way in, which is the layer that actually enforces it.
    """
    if name not in ARTIFACT_NAMES:
        raise PlanningError(f"There is no artifact called {name!r}.")
    return f"runs/{run_id}/{ARTIFACT_NAMES[name]}"


def _to_parquet(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    # Column names reaching parquet must be strings; one-hot output can produce
    # non-string labels when the source categories were numeric.
    frame.columns = [str(c) for c in frame.columns]
    frame.to_parquet(buffer, index=True)
    return buffer.getvalue()


def _script_for(result: RunResult, ingested: IngestResult | None) -> bytes:
    """The standalone script that reproduces this run.

    Stored as an artifact rather than offered as a separate call, because it is part of
    what the run *is*: the run's claim is that these decisions produced this data, and
    the script is the form of that claim a user can execute themselves. It carries the
    engine's own definitions, so running it reproduces the run's output hash exactly —
    see :mod:`app.plan_codegen`.
    """
    from app.codegen import SourceFormat
    from app.plan_codegen import generate_from_plan

    source = SourceFormat()
    numeric_columns: list[str] = []
    date_columns: list[str] = []
    if ingested is not None:
        source = SourceFormat(
            kind="csv" if not ingested.delimiter.startswith("n/a") else "excel",
            delimiter=ingested.delimiter,
            encoding=ingested.encoding,
        )
        numeric_columns = ingested.numeric_columns
        date_columns = ingested.datetime_columns

    return generate_from_plan(
        result.plan,
        source=source,
        run_id=result.schema.get("run_id", "") or "this run",
        numeric_columns=numeric_columns,
        date_columns=date_columns,
    ).encode("utf-8")


def _write_artifacts(
    run_id: str, result: RunResult, ingested: IngestResult | None = None
) -> dict[str, str]:
    """Store every artifact, and return ``{name: key}``.

    Raises before returning if any one of them fails, so the caller never records a
    partial set. Nothing is written to the database until this has returned.
    """
    storage = get_storage()
    train = result.X_train.copy()
    train[result.plan.target] = result.y_train
    test = result.X_test.copy()
    test[result.plan.target] = result.y_test

    payloads: dict[str, bytes] = {
        "plan": json.dumps(result.plan.model_dump(mode="json"), indent=2).encode("utf-8"),
        "schema": json.dumps(result.schema, indent=2).encode("utf-8"),
        "log": json.dumps(result.log, indent=2, default=str).encode("utf-8"),
        "pipeline": _dump_pipeline(result),
        "train": _to_parquet(train),
        "test": _to_parquet(test),
        "script": _script_for(result, ingested),
    }

    written: dict[str, str] = {}
    for name, payload in payloads.items():
        key = artifact_key(run_id, name)
        storage.put(key, payload)
        written[name] = key
    return written


def _dump_pipeline(result: RunResult) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(result.pipeline, buffer)
    return buffer.getvalue()


def execute_plan(
    session: Session,
    plan_row: Plan,
    frame: pd.DataFrame,
    ingested: IngestResult | None = None,
) -> Run:
    """Execute one stored plan version and record the run and its artifacts.

    The run row is created ``running`` before execution starts, so a crash leaves a row
    that plainly says it never finished rather than no evidence at all. It becomes
    ``succeeded`` only after every artifact is stored.
    """
    plan = plan_object(plan_row)
    run_id = uuid.uuid4().hex

    run = Run(
        id=run_id,
        dataset_id=plan_row.dataset_id,
        plan_id=plan_row.id,
        plan_version=plan_row.version,
        status="running",
        target=plan.target,
        task=plan.task,
        seed=plan.options.seed,
        test_size=plan.options.test_size,
    )
    session.add(run)
    session.flush()

    try:
        result = execute(frame, plan, run_id=run_id)
        artifacts = _write_artifacts(run_id, result, ingested)
    except Exception as exc:
        run.status = "failed"
        run.completed_at = datetime.now(timezone.utc)
        run.error = {
            "type": type(exc).__name__,
            # The message only. A traceback names filesystem paths and library
            # internals, and this field is served to the browser.
            "message": str(exc)[:2000],
        }
        session.flush()
        return run

    run.status = "succeeded"
    run.completed_at = datetime.now(timezone.utc)
    run.duration_ms = result.duration_ms
    run.stratified = result.stratified
    run.train_rows = int(len(result.X_train))
    run.test_rows = int(len(result.X_test))
    run.n_features = int(result.X_train.shape[1])
    run.output_hash = result.output_hash
    run.feature_schema = result.schema
    run.warnings = result.warnings
    run.checks = result.checks
    run.artifacts = artifacts
    session.flush()
    return run


def delete_run_artifacts(run: Run) -> list[str]:
    """Remove a run's stored artifacts. Returns the keys that could not be removed.

    Missing files are not failures — the point is that the key is gone afterwards, and
    a key that was never written is already in that state.
    """
    storage = get_storage()
    failed: list[str] = []
    for key in (run.artifacts or {}).values():
        try:
            storage.delete(key)
        except Exception:
            if storage.exists(key):
                failed.append(key)
    return failed
