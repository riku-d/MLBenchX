"""Database models and session handling — SQLAlchemy 2.x on SQLite.

Nine tables, one per thing the app produces. The first four record what the glass-box
engine did to a dataset; three more (``workspaces``, ``compilations``, ``history``)
record what the user is doing with it interactively; and two (``plans``, ``runs``)
record the spec's reviewable plan and the durable, versioned execution of it.

``datasets``
    One uploaded file. The bytes themselves live in storage; this row records what it
    was, where it went, and what reading it cost.
``profiles``
    What we measured about a dataset. One per dataset, rewritten if it is re-profiled.
``preprocessing_runs``
    One execution of the pipeline against a dataset, with the options it was given and
    the key of the fitted pipeline it produced.
``preprocessing_logs``
    One row per step of a run, in order. A table rather than a JSON blob on the run,
    because the spec asks for the log as a resource and because "show me every
    one-hot step across every run" should be a query, not a scan.

Two conventions hold everywhere, both from the spec:

* Every table carries ``project_id``. Phase 1 has no accounts and everything lands in
  the default project, but a column that exists from the start is a filter later rather
  than a migration of live data.
* Every table carries ``created_at``, set server-side in UTC.

JSON columns
------------
Profiles and log details are stored as JSON text. They are documents — deeply nested,
read whole, never joined on — and normalising them into columns would mean a migration
every time an explanation gains a field. What *is* promoted to real columns is anything
worth querying or sorting by.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    func,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from app.config import DATABASE_URL, DEFAULT_PROJECT_ID

__all__ = [
    "Base",
    "Dataset",
    "Profile",
    "PreprocessingRun",
    "PreprocessingLog",
    "Workspace",
    "Compilation",
    "HistoryEntry",
    "engine",
    "SessionLocal",
    "session_scope",
    "get_session",
    "init_db",
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base. ``dict`` maps to SQLite JSON via SQLAlchemy's JSON type."""

    type_annotation_map = {dict[str, Any]: JSON, list[dict[str, Any]]: JSON}


class TimestampedProjectMixin:
    """The two columns the spec requires on every table."""

    project_id: Mapped[str] = mapped_column(
        String(64), default=DEFAULT_PROJECT_ID, index=True, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), nullable=False
    )


class Dataset(TimestampedProjectMixin, Base):
    """One uploaded file."""

    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    #: Storage key for the original bytes, not a filesystem path.
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    n_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    n_columns: Mapped[int] = mapped_column(Integer, nullable=False)
    encoding: Mapped[str] = mapped_column(String(64), nullable=False)
    delimiter: Mapped[str] = mapped_column(String(32), nullable=False)
    #: Health score 0-100. A column rather than part of the profile JSON because it is
    #: the one number anyone wants to sort a list of datasets by.
    health_score: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    #: Sentences describing each format repair applied on the way in.
    cleaning: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)

    profile: Mapped["Profile | None"] = relationship(
        back_populates="dataset", cascade="all, delete-orphan", uselist=False
    )
    runs: Mapped[list["PreprocessingRun"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan", order_by="PreprocessingRun.created_at"
    )


class Profile(TimestampedProjectMixin, Base):
    """What we measured about one dataset."""

    __tablename__ = "profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("datasets.id", ondelete="CASCADE"), index=True, nullable=False
    )
    n_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    n_columns: Mapped[int] = mapped_column(Integer, nullable=False)
    duplicate_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    missing_cells: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    health_score: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    #: Per-column records, the correlated pairs, the leakage flags and the class
    #: balance — the whole document the profile dashboard renders.
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    dataset: Mapped[Dataset] = relationship(back_populates="profile")


class PreprocessingRun(TimestampedProjectMixin, Base):
    """One execution of the pipeline over one dataset."""

    __tablename__ = "preprocessing_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("datasets.id", ondelete="CASCADE"), index=True, nullable=False
    )
    target: Mapped[str] = mapped_column(String(255), nullable=False)
    task: Mapped[str] = mapped_column(String(32), nullable=False)
    scaler: Mapped[str] = mapped_column(String(32), nullable=False)
    high_card_encoder: Mapped[str] = mapped_column(String(32), nullable=False)
    test_size: Mapped[float] = mapped_column(Float, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    train_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    test_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    n_features: Mapped[int] = mapped_column(Integer, nullable=False)
    #: Storage key of the joblib-serialised fitted ColumnTransformer.
    pipeline_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    took_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: The full response document, so a run can be re-served exactly as first computed
    #: without refitting anything.
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    dataset: Mapped[Dataset] = relationship(back_populates="runs")
    logs: Mapped[list["PreprocessingLog"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="PreprocessingLog.step"
    )


class PreprocessingLog(TimestampedProjectMixin, Base):
    """One step of one run — the glass-box record, one row at a time."""

    __tablename__ = "preprocessing_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("preprocessing_runs.id", ondelete="CASCADE"),
        index=True, nullable=False,
    )
    step: Mapped[int] = mapped_column(Integer, nullable=False)
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    #: Comma-separated column list. The spec names this field ``column``; the machine
    #: readable list is in ``params`` alongside everything else structured.
    column: Mapped[str] = mapped_column(Text, nullable=False, default="")
    transformer: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    #: Plain English, written for someone with no ML background.
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    before: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    after: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    run: Mapped[PreprocessingRun] = relationship(back_populates="logs")


# --------------------------------------------------------------------------------------
# Engine and sessions
# --------------------------------------------------------------------------------------

#: ``check_same_thread=False`` because Starlette runs sync routes in a worker thread,
#: which is not the thread that created the connection. Safe here: SQLAlchemy's pool
#: hands one connection to one caller at a time.
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    future=True,
)

if DATABASE_URL.startswith("sqlite"):

    @event.listens_for(engine, "connect")
    def _enforce_foreign_keys(dbapi_connection, _record) -> None:  # pragma: no cover
        """Turn on SQLite's foreign-key enforcement, per connection.

        SQLite accepts ``ON DELETE CASCADE`` in a schema and then ignores it unless this
        pragma is set, and it is off by default — so every cascade declared above was
        documentation rather than behaviour. Deleting a dataset would have left its
        profile, runs, plans and workspace behind, pointing at a row that no longer
        exists. The pragma is per-connection, not per-database, which is why it is set
        on the pool's connect event rather than once at startup.
        """
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    """A session that commits on success and rolls back on any exception."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Iterator[Session]:
    """FastAPI dependency. One session per request, closed when the response is done."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    """Create any missing tables.

    Alembic owns the schema; this exists for tests and for a first run against an empty
    file, where waiting on a migration to create four tables helps nobody. The two agree
    because both are generated from the models above.
    """
    Base.metadata.create_all(engine)


# --------------------------------------------------------------------------------------
# The interactive workspace
# --------------------------------------------------------------------------------------
#
# The tables above record what the glass-box engine did to a dataset. These three record
# what the *user* is doing with it: the pipeline they have configured, the code they have
# edited, every compile they have run, and the trail of changes that lets them go back.


class Workspace(TimestampedProjectMixin, Base):
    """One user's working state over one dataset.

    Exactly one per dataset. A workspace holds the configuration and, separately, the
    code — because the moment a user edits the code by hand the two can diverge, and
    which one is authoritative becomes a fact the interface has to state rather than
    guess. ``code_is_edited`` is that fact.
    """

    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("datasets.id", ondelete="CASCADE"),
        index=True, unique=True, nullable=False,
    )
    #: The PipelineConfig, as written by the toggles.
    config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    #: The code as it currently stands in the editor.
    code: Mapped[str] = mapped_column(Text, nullable=False, default="")
    #: True once the user's code differs from what the config would generate. From then
    #: on the code is what runs, and the workspace says so rather than letting the
    #: toggles imply otherwise.
    code_is_edited: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    #: The id of the last compile that succeeded, so "unsaved changes" is answerable.
    last_compilation_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    #: Health of the dataset as uploaded, measured once.
    baseline_health: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    dataset: Mapped[Dataset] = relationship()
    compilations: Mapped[list["Compilation"]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan",
        order_by="Compilation.created_at.desc()",
    )
    history: Mapped[list["HistoryEntry"]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan",
        order_by="HistoryEntry.created_at.desc()",
    )


class Compilation(TimestampedProjectMixin, Base):
    """One execution of a pipeline — what ran, what it produced, what it cost.

    Failures are rows here too. A compile that raised is something the user needs to
    read, come back to, and compare against the one before it; discarding it because it
    did not succeed would throw away the more interesting half of the history.
    """

    __tablename__ = "compilations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("workspaces.id", ondelete="CASCADE"), index=True, nullable=False
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)  # success | error
    #: Exactly what was executed, so a run can always be reproduced.
    code: Mapped[str] = mapped_column(Text, nullable=False)
    #: The config at the time, or null when the code had been hand-edited.
    config: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="pipeline")
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    logs: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    error: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    #: Shape before and after, and the per-step report the script emitted.
    result: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    health_before: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    health_after: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    #: Storage key of the processed CSV, when one was produced.
    output_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    sandbox: Mapped[str] = mapped_column(String(32), nullable=False, default="none")

    workspace: Mapped[Workspace] = relationship(back_populates="compilations")


class HistoryEntry(TimestampedProjectMixin, Base):
    """One change the user made, and the state it produced.

    Stores the whole configuration rather than a diff. A diff is smaller and would make
    "restore this point" a replay of every entry since — which is fragile, and wrong the
    first time an entry fails to apply. A config is a few hundred bytes.
    """

    __tablename__ = "history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    workspace_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("workspaces.id", ondelete="CASCADE"), index=True, nullable=False
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    config: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    code: Mapped[str | None] = mapped_column(Text, nullable=True)

    workspace: Mapped[Workspace] = relationship(back_populates="history")


# --------------------------------------------------------------------------------------
# The plan and the run
# --------------------------------------------------------------------------------------
#
# The tables above record a *configuration* applied to a dataset and the compilations it
# produced. These two record the spec's §4 and §6 objects, which are different things:
#
# ``plans``
#     A per-column decision document — what happens to each column and why — that the
#     user reviews and edits before anything executes. Versioned, and **immutable once
#     written**: an edit writes a new row with the next version rather than changing the
#     one a run may already point at. That is what makes "this run used that plan" a
#     fact rather than a hope, and it is why the table's primary key is a plan id while
#     its identity is ``(dataset_id, version)``.
#
# ``runs``
#     One execution of one plan version, with the artifacts it produced. A run records
#     the plan id *and* the version it used, and additionally stores the whole plan
#     document as its own ``plan.json`` artifact — belt and braces, so a run can be
#     explained from its artifacts alone even if the database is unavailable.
#
# A run is never updated after it finishes. ``status`` moves ``running`` → ``succeeded``
# or ``failed`` exactly once, and a failed run keeps its error rather than disappearing:
# the user needs to read it, and a run that vanished on failure would make the history
# of a dataset a record of only its successes.


class Plan(TimestampedProjectMixin, Base):
    """One version of the preprocessing plan for one dataset."""

    __tablename__ = "plans"
    __table_args__ = (
        UniqueConstraint("dataset_id", "version", name="uq_plans_dataset_version"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("datasets.id", ondelete="CASCADE"), index=True, nullable=False
    )
    #: 1 for the proposal, incremented by every edit. Unique per dataset.
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    #: The plan this one was derived from, so an edit trail can be followed back to the
    #: original proposal.
    parent_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("plans.id", ondelete="SET NULL"), nullable=True
    )

    target: Mapped[str] = mapped_column(String(255), nullable=False)
    task: Mapped[str] = mapped_column(String(32), nullable=False)
    #: ``proposed`` until a user edits it, then ``edited``. Promoted out of the document
    #: because "show me the plans a user changed" should be a query.
    origin: Mapped[str] = mapped_column(String(16), nullable=False, default="proposed")
    n_columns: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_overridden: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    #: The whole :class:`~app.plan.PreprocessingPlan`, exactly as it will be executed.
    document: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    dataset: Mapped[Dataset] = relationship()
    runs: Mapped[list["Run"]] = relationship(
        back_populates="plan", cascade="all, delete-orphan", passive_deletes=True
    )


class Run(TimestampedProjectMixin, Base):
    """One execution of one plan version, and the artifacts it produced."""

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("datasets.id", ondelete="CASCADE"), index=True, nullable=False
    )
    plan_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("plans.id", ondelete="CASCADE"), index=True, nullable=False
    )
    #: Copied rather than joined. A run names the exact version it used, and that answer
    #: must survive the plan table being queried differently later.
    plan_version: Mapped[int] = mapped_column(Integer, nullable=False)

    #: ``running`` | ``succeeded`` | ``failed``. A failed run is never presented as a
    #: result: every consumer checks this before reading the artifacts.
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="running")
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    target: Mapped[str] = mapped_column(String(255), nullable=False)
    task: Mapped[str] = mapped_column(String(32), nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    test_size: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    stratified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    train_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    test_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_features: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    #: A digest of the prepared data. Two runs of the same plan on the same dataset
    #: match on it or the determinism guarantee is false, which is a checkable claim
    #: rather than an argued one.
    output_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    #: The inference contract (`schema.json`), promoted here as well as stored as an
    #: artifact because Stage 4 will read it across many runs.
    feature_schema: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    warnings: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    checks: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    #: ``{artifact name: storage key}``. Written only after every artifact is stored, so
    #: a row never names a file that is not there.
    artifacts: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)
    error: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    dataset: Mapped[Dataset] = relationship()
    plan: Mapped[Plan] = relationship(back_populates="runs")
