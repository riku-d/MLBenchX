"""Routes for the reviewable plan and the durable run.

These are the spec's §4 and §6 resources. They sit beside the two surfaces that already
exist rather than replacing either: :mod:`app.main`'s routes describe what the glass-box
*engine* did to a dataset in one call, :mod:`app.workspace`'s describe what the user is
doing with it in the editor, and these describe the **propose → review → execute** cycle
the specification is built around.

The shape of the cycle is the point:

``POST /api/datasets/{id}/plan``   propose, from the dataset — version 1
``GET  /api/datasets/{id}/plan``   the current version, with the list of versions
``GET  /api/plans/{plan_id}``      one version, exactly as it was stored
``PUT  /api/plans/{plan_id}``      review and edit — writes the **next** version
``POST /api/plans/{plan_id}/run``  execute that version, producing a run
``GET  /api/runs/{run_id}``        the run
``GET  /api/runs/{id}/artifacts/{name}``  one of its six artifacts

A ``PUT`` that writes a new row rather than updating one is the unusual choice here, and
it is what makes a run's claim about its plan true. If editing mutated the plan, a run
recorded last week would silently start describing this week's decisions, and the
artifact trail would be the only honest record left. Versions are cheap; a run that
cannot be explained is not.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import (
    Compilation,
    Dataset,
    Plan,
    PreprocessingRun,
    Run,
    Workspace,
    get_session,
)
from app.plan import ACTIONS_FOR_KIND
from app.services import planning
from app.services.ingest import IngestError, IngestResult, read_upload
from app.storage import StorageError, get_storage

router = APIRouter(tags=["plan"])


# --------------------------------------------------------------------------------------
# Request shapes
# --------------------------------------------------------------------------------------


class ProposeRequest(BaseModel):
    """What to plan for, in the user's own words or by naming the column."""

    goal: str = Field(default="", description="Plain-English description of the goal.")
    target: str | None = Field(default=None, description="Name the column directly.")
    options: dict[str, Any] | None = Field(
        default=None, description="Overrides for the pipeline-wide options."
    )


class ColumnOverride(BaseModel):
    action: str
    params: dict[str, Any] | None = None


class EditRequest(BaseModel):
    """The user's changes to a plan. Produces the next version; changes nothing."""

    columns: dict[str, ColumnOverride] = Field(default_factory=dict)
    options: dict[str, Any] | None = None


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _get_dataset(dataset_id: str, session: Session) -> Dataset:
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="No dataset with that id.")
    return dataset


def _ingest(dataset: Dataset) -> IngestResult:
    """Re-read the stored bytes. Ingestion is deterministic, so this is a cache-free read."""
    try:
        raw = get_storage().get(dataset.storage_key)
    except StorageError as exc:
        raise HTTPException(
            status_code=410,
            detail="The stored copy of that file is gone. Upload it again.",
        ) from exc
    try:
        return read_upload(raw, dataset.filename)
    except IngestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _get_plan(plan_id: str, session: Session) -> Plan:
    row = session.get(Plan, plan_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No plan with that id.")
    return row


def plan_view(row: Plan, versions: list[Plan] | None = None) -> dict[str, Any]:
    """One plan, as the review screen renders it.

    ``actions_for_kind`` travels with the plan on purpose. The interface offers a
    dropdown per column and must offer exactly the actions that column can take — asking
    the client to hold that table would mean two copies of the rule, and the one in the
    browser would be the stale one.
    """
    view: dict[str, Any] = {
        "id": row.id,
        "dataset_id": row.dataset_id,
        "version": row.version,
        "parent_id": row.parent_id,
        "origin": row.origin,
        "target": row.target,
        "task": row.task,
        "n_columns": row.n_columns,
        "n_overridden": row.n_overridden,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
        "actions_for_kind": ACTIONS_FOR_KIND,
        **row.document,
    }
    if versions is not None:
        view["versions"] = [
            {
                "id": other.id,
                "version": other.version,
                "origin": other.origin,
                "n_overridden": other.n_overridden,
                "created_at": other.created_at,
            }
            for other in versions
        ]
    return view


def run_view(row: Run, detail: bool = True) -> dict[str, Any]:
    """One run. The shape Stage 4 will read across many runs — see Design.md §11.

    ``status`` is first because every other field is conditional on it: a failed run has
    no artifacts and no hash, and a consumer that reads them without checking would be
    treating a failure as a result.
    """
    view: dict[str, Any] = {
        "id": row.id,
        "status": row.status,
        "dataset_id": row.dataset_id,
        "plan_id": row.plan_id,
        "plan_version": row.plan_version,
        "created_at": row.created_at,
        "completed_at": row.completed_at,
        "duration_ms": row.duration_ms,
        "target": row.target,
        "task": row.task,
        "split": {
            "test_size": row.test_size,
            "seed": row.seed,
            "stratified": row.stratified,
            "train_rows": row.train_rows,
            "test_rows": row.test_rows,
        },
        "n_features": row.n_features,
        "output_hash": row.output_hash,
        "artifacts": sorted(row.artifacts or {}),
        "error": row.error,
    }
    if detail:
        view["warnings"] = row.warnings or []
        view["checks"] = row.checks or {}
        view["feature_schema"] = row.feature_schema or {}
    return view


# --------------------------------------------------------------------------------------
# Plan routes
# --------------------------------------------------------------------------------------


@router.post(
    "/api/datasets/{dataset_id}/plan",
    status_code=201,
    summary="Propose a preprocessing plan",
)
def propose_plan(
    dataset_id: str,
    request: ProposeRequest | None = None,
    session: Session = Depends(get_session),
) -> Any:
    """Decide what to do with every column, and say why — without executing anything.

    This is the route the specification's propose→review→execute cycle starts from, and
    the reason it exists separately from ``/preprocess``: a user is owed the chance to
    disagree with a decision before it is applied to their data.

    Proposing again writes a further version rather than replacing the current one, so a
    user who has edited a plan and then re-proposes still has their edits in the history.
    """
    dataset = _get_dataset(dataset_id, session)
    body = request or ProposeRequest()

    options = None
    if body.options:
        from app.plan import GlobalOptions

        try:
            options = GlobalOptions.model_validate(body.options)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Those pipeline options are not valid: {exc}"
            ) from exc

    ingested = _ingest(dataset)
    try:
        plan = planning.propose_for_dataset(
            ingested, goal=body.goal, target=body.target, options=options
        )
    except planning.PlanningError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    row = planning.store_plan(session, dataset, plan)
    session.commit()
    return plan_view(row, planning.plan_versions(session, dataset_id))


@router.get("/api/datasets/{dataset_id}/plan", summary="The current plan")
def get_current_plan(
    dataset_id: str,
    version: int | None = None,
    session: Session = Depends(get_session),
) -> Any:
    """The newest version of this dataset's plan, or the one ``?version=`` names."""
    _get_dataset(dataset_id, session)
    versions = planning.plan_versions(session, dataset_id)
    if not versions:
        raise HTTPException(
            status_code=404,
            detail=(
                "No plan has been proposed for that dataset yet. POST to its /plan "
                "route first."
            ),
        )

    if version is None:
        row = versions[-1]
    else:
        row = next((p for p in versions if p.version == version), None)  # type: ignore[assignment]
        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"That dataset has no plan version {version}.",
            )
    return plan_view(row, versions)


@router.get("/api/plans/{plan_id}", summary="One plan version")
def get_plan(plan_id: str, session: Session = Depends(get_session)) -> Any:
    row = _get_plan(plan_id, session)
    return plan_view(row, planning.plan_versions(session, row.dataset_id))


@router.put("/api/plans/{plan_id}", status_code=201, summary="Edit a plan")
def edit_plan(
    plan_id: str,
    request: EditRequest,
    session: Session = Depends(get_session),
) -> Any:
    """Apply the user's changes, as a **new version**.

    ``201`` rather than ``200``: this creates a resource at a new id. The plan named in
    the URL is returned unchanged by a subsequent ``GET`` of it, which is what lets a
    run that used it keep meaning what it said.

    An action that makes no sense for a column's kind is a ``400`` here, before anything
    is stored — rather than a failure at execution with a message about dtypes.
    """
    parent = _get_plan(plan_id, session)
    dataset = _get_dataset(parent.dataset_id, session)

    try:
        edited = planning.apply_overrides(
            planning.plan_object(parent),
            columns={name: change.model_dump() for name, change in request.columns.items()},
            options=request.options,
        )
    except planning.PlanningError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    row = planning.store_plan(session, dataset, edited, parent=parent, origin="edited")
    session.commit()
    return plan_view(row, planning.plan_versions(session, dataset.id))


# --------------------------------------------------------------------------------------
# Run routes
# --------------------------------------------------------------------------------------


@router.post("/api/plans/{plan_id}/run", status_code=201, summary="Execute a plan")
def run_plan(plan_id: str, session: Session = Depends(get_session)) -> Any:
    """Execute exactly this plan version, and store the run with its artifacts.

    Nothing is re-decided: the executor carries out the plan's column decisions and has
    no freedom to choose differently. That is what makes the review meaningful.

    A run that fails is stored as ``failed`` with its error and still returns ``201`` —
    the run resource was created, and the user needs to be able to read why it did not
    finish. Callers check ``status``; a failed run has no artifacts to read.
    """
    plan_row = _get_plan(plan_id, session)
    dataset = _get_dataset(plan_row.dataset_id, session)
    ingested = _ingest(dataset)

    run = planning.execute_plan(session, plan_row, ingested.frame, ingested)
    session.commit()
    return run_view(run)


@router.get("/api/datasets/{dataset_id}/runs", summary="Runs for a dataset")
def list_runs(dataset_id: str, session: Session = Depends(get_session)) -> Any:
    """Every run of this dataset, newest first — successes and failures alike."""
    _get_dataset(dataset_id, session)
    rows = session.scalars(
        select(Run).where(Run.dataset_id == dataset_id).order_by(Run.created_at.desc())
    ).all()
    return {"dataset_id": dataset_id, "runs": [run_view(row, detail=False) for row in rows]}


@router.get("/api/runs/{run_id}/artifacts/{name}", summary="Download one artifact")
def get_artifact(run_id: str, name: str, session: Session = Depends(get_session)) -> Any:
    """One of the six artifacts, by name.

    The name is looked up in :data:`~app.services.planning.ARTIFACT_NAMES` rather than
    placed into a path, so an unknown name is a 404 and a path cannot be named at all.
    """
    run = session.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="No run with that id.")
    if name not in planning.ARTIFACT_NAMES:
        raise HTTPException(
            status_code=404,
            detail=(
                f"There is no artifact called {name!r}. This run has: "
                + ", ".join(sorted(run.artifacts or {}))
            ),
        )
    key = (run.artifacts or {}).get(name)
    if not key:
        raise HTTPException(
            status_code=404,
            detail=(
                f"That run produced no {name} artifact."
                + (" It did not finish: " + (run.error or {}).get("message", "")
                   if run.status == "failed" else "")
            ),
        )

    try:
        payload = get_storage().get(key)
    except StorageError as exc:
        raise HTTPException(
            status_code=410, detail="That artifact is no longer stored."
        ) from exc

    return Response(
        content=payload,
        media_type=planning.ARTIFACT_MEDIA[name],
        headers={
            "Content-Disposition": (
                f'attachment; filename="{planning.ARTIFACT_NAMES[name]}"'
            )
        },
    )


# --------------------------------------------------------------------------------------
# Deleting a dataset
# --------------------------------------------------------------------------------------


def _storage_keys_of(dataset: Dataset, session: Session) -> list[str]:
    """Every stored object that belongs to this dataset and nothing else.

    Collected *before* the rows are deleted, because the rows are the only record of
    where the bytes are. Anything missed here becomes an orphaned file — wasted space,
    not a correctness problem — which is the right way round for this to fail.
    """
    keys: list[str] = [dataset.storage_key]

    for run in session.scalars(select(Run).where(Run.dataset_id == dataset.id)).all():
        keys.extend((run.artifacts or {}).values())

    for legacy in session.scalars(
        select(PreprocessingRun).where(PreprocessingRun.dataset_id == dataset.id)
    ).all():
        if legacy.pipeline_key:
            keys.append(legacy.pipeline_key)
        keys.append(f"runs/{legacy.id}/prepared_train.csv")

    workspace_ids = session.scalars(
        select(Workspace.id).where(Workspace.dataset_id == dataset.id)
    ).all()
    if workspace_ids:
        for compilation in session.scalars(
            select(Compilation).where(Compilation.workspace_id.in_(workspace_ids))
        ).all():
            if compilation.output_key:
                keys.append(compilation.output_key)

    # Stable and de-duplicated, so the response is the same every time and a key shared
    # by two rows is not reported twice.
    return sorted(set(keys))


@router.delete("/api/datasets/{dataset_id}", summary="Delete a dataset")
def delete_dataset(dataset_id: str, session: Session = Depends(get_session)) -> Any:
    """Remove a dataset, everything derived from it, and its stored bytes.

    **Ordering is the design decision here.** The database rows go first and are
    committed; the files are removed afterwards. The other order is worse in a way that
    matters: a storage failure after the rows are gone leaves unreferenced bytes on
    disk, which costs space, while a database failure after the files are gone leaves
    rows pointing at data that no longer exists — a dataset that lists, profiles and
    then fails at the moment someone tries to use it.

    So a partial failure here always lands on the recoverable side, and the response
    says which keys survived rather than reporting a clean delete it did not perform.

    Cascades do the rest: the profile, the legacy preprocessing runs and their logs, the
    workspace with its compilations and history, and every plan with its runs all go
    with it. Nothing belonging to another dataset is touched — each cascade is keyed on
    this dataset's id.
    """
    dataset = _get_dataset(dataset_id, session)
    keys = _storage_keys_of(dataset, session)
    filename = dataset.filename

    try:
        session.delete(dataset)
        session.commit()
    except Exception as exc:
        session.rollback()
        raise HTTPException(
            status_code=500,
            detail=(
                "That dataset could not be removed, so nothing was changed. "
                f"({type(exc).__name__})"
            ),
        ) from exc

    storage = get_storage()
    orphaned: list[str] = []
    for key in keys:
        try:
            storage.delete(key)
        except Exception:
            if storage.exists(key):
                orphaned.append(key)

    return {
        "deleted": True,
        "dataset_id": dataset_id,
        "filename": filename,
        "artifacts_removed": len(keys) - len(orphaned),
        # Named rather than hidden: these are files the application has lost track of,
        # and a caller that is cleaning up needs to know they are still there.
        "artifacts_orphaned": orphaned,
    }
