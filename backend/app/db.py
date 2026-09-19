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
