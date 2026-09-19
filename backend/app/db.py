"""Database models and session handling — SQLAlchemy 2.x on SQLite.

Four tables, one per thing the app produces:

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
    create_engine,
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
