"""Request and response shapes for the API.

These are the contract. The frontend's ``api.ts`` types mirror them by hand, and the
OpenAPI document at ``/api/docs`` is generated from them, so a field added here and
nowhere else is visible in three places at once.

Nested documents that the UI renders but never branches on — the per-column profile
records, a step's ``details`` — are typed as permissive dicts on purpose. Pinning every
key of an explanation's details block would mean editing this file to add a sentence,
and the value of a schema is in the shape callers depend on, not in every leaf.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.config import (
    DEFAULT_HIGH_CARD_ENCODER,
    DEFAULT_SCALER,
    MAX_TARGET_CARDINALITY,
)

__all__ = [
    "ErrorResponse",
    "PreprocessRequest",
    "DatasetSummary",
    "DatasetDetail",
    "ProfileResponse",
    "LogEntry",
    "LogResponse",
    "AnalysisResponse",
    "PipelineWarning",
]

ScalerName = Literal["standard", "minmax", "robust"]
HighCardEncoder = Literal["frequency", "target"]


class ErrorResponse(BaseModel):
    """Every 4xx and 5xx body. One sentence, written for the person reading it."""

    error: str


# --------------------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------------------


class PreprocessRequest(BaseModel):
    """Options for one preprocessing run.

    Every field has a working default, so ``POST .../preprocess`` with an empty body is
    a valid request that does the sensible thing.
    """

    target: str | None = Field(
        default=None,
        max_length=255,
        description="Column to predict. Inferred from `goal` when omitted.",
    )
    goal: str = Field(
        default="",
        max_length=500,
        description="Plain-English description of what to predict, used to infer the target.",
    )
    scaler: ScalerName = Field(
        default=DEFAULT_SCALER,  # type: ignore[assignment]
        description="How numeric columns are put on a common scale.",
    )
    high_card_encoder: HighCardEncoder = Field(
        default=DEFAULT_HIGH_CARD_ENCODER,  # type: ignore[assignment]
        description=(
            "How categoricals with many distinct values are encoded. `frequency` never "
            "reads the target; `target` does, but cross-fits so no row informs its own "
            "encoded value."
        ),
    )
    test_size: float = Field(
        default=0.2, gt=0.05, lt=0.5, description="Share of rows held out for testing."
    )


# --------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------


class DatasetSummary(BaseModel):
    """One row in a list of datasets."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    filename: str
    n_rows: int
    n_columns: int
    size_bytes: int
    health_score: int
    created_at: datetime


class DatasetDetail(DatasetSummary):
    """Everything known about one dataset without computing anything new."""

    encoding: str
    delimiter: str
    cleaning: list[str] = Field(
        default_factory=list, description="One sentence per format repair applied on read."
    )
    columns: list[str] = Field(default_factory=list)
    runs: list[RunSummary] = Field(default_factory=list)


class RunSummary(BaseModel):
    """One preprocessing run, without its payload."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    target: str
    task: str
    scaler: str
    high_card_encoder: str
    train_rows: int
    test_rows: int
    n_features: int
    created_at: datetime


# --------------------------------------------------------------------------------------
# Profile
# --------------------------------------------------------------------------------------


class OutlierSummary(BaseModel):
    """Values outside Tukey's fences for one numeric column."""

    count: int
    share: float
    lower: float
    upper: float


class ColumnProfileOut(BaseModel):
    """One row of the per-column flag table."""

    column: str
    kind: str
    kind_label: str
    dtype: str
    n_missing: int
    missing_rate: float
    n_unique: int
    unique_rate: float
    n_rows: int
    outliers: OutlierSummary | None = None
    #: Descriptive statistics, bin counts and the most frequent values.
    #:
    #: These were computed by `services/profile.py` and then discarded here, because a
    #: response model drops what it does not declare — so the histogram the backend goes
    #: to the trouble of computing server-side (a histogram of one page of rows is not a
    #: histogram of the column) never reached the client that was meant to draw it.
    #: Typed as permissive documents for the reason this file's docstring gives: the
    #: interface renders them and does not branch on their leaves.
    stats: dict[str, Any] | None = None
    histogram: dict[str, Any] | None = None
    top_values: list[dict[str, Any]] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    is_target: bool = False
    used: bool = True
    excluded_reason: str | None = None
    leakage: dict[str, Any] | None = None
    flags: list[str] = Field(default_factory=list)


class CorrelatedPair(BaseModel):
    """Two columns carrying nearly the same information."""

    columns: list[str]
    correlation: float
    reason: str


class QualityIssueOut(BaseModel):
    severity: str
    text: str


class HealthDeduction(BaseModel):
    """One line of the health score's arithmetic."""

    factor: str
    label: str
    amount: float
    weight: float
    cap: float
    points_lost: float


class QualityOut(BaseModel):
    """The ingest-time quality check — *not* the canonical health score.

    A cheap arithmetic check run while the file is being read: blank cells, duplicate
    rows, mostly-empty columns. The number shown throughout the interface is the
    six-dimension :func:`~app.services.health.assess` score, which is measured
    differently and will usually differ from this one. Both are kept because they answer
    different questions, and ``label`` exists so neither can be mistaken for the other.
    """

    label: str = Field(
        default="Ingest quality check (not the workspace health score)",
        description="Names this measurement so it cannot be read as the health score.",
    )
    score: int = Field(ge=0, le=100)
    rows_used: int
    missing_cells: int
    duplicate_rows: int
    issues: list[QualityIssueOut] = Field(default_factory=list)
    breakdown: list[HealthDeduction] = Field(default_factory=list)


class ProfileResponse(BaseModel):
    """Everything the profile dashboard renders."""

    dataset_id: str
    n_rows: int
    n_columns: int
    duplicate_rows: int
    quality: QualityOut
    columns: list[ColumnProfileOut] = Field(default_factory=list)
    correlated_pairs: list[CorrelatedPair] = Field(default_factory=list)
    leakage_flags: list[dict[str, Any]] = Field(default_factory=list)
    class_balance: dict[str, Any] | None = None


# --------------------------------------------------------------------------------------
# Preprocessing log
# --------------------------------------------------------------------------------------


class LogEntry(BaseModel):
    """One step of the glass-box log, as stored."""

    model_config = ConfigDict(from_attributes=True)

    step: int
    stage: str
    action: str
    column: str
    transformer: str
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str
    before: dict[str, Any] = Field(default_factory=dict)
    after: dict[str, Any] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)


class LogResponse(BaseModel):
    """The ordered log for one run."""

    run_id: str
    dataset_id: str
    target: str
    task: str
    fitted_on: Literal["train_fold_only"] = "train_fold_only"
    entries: list[LogEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# The analysis document
# --------------------------------------------------------------------------------------


class FileInfo(BaseModel):
    name: str
    rows: int
    columns: int
    encoding: str
    delimiter: str


class TargetInfo(BaseModel):
    column: str
    task: str
    values: dict[str, int] = Field(default_factory=dict)
    inferred: bool
    note: str


class DroppedColumn(BaseModel):
    column: str
    reason: str


class Strip(BaseModel):
    """Three real values from the user's file, before and after one step."""

    column: str
    before: list[str]
    after: list[str]
    changed: list[bool]
    rows: list[str]


class SplitInfo(BaseModel):
    train: int
    test: int
    percent: int


class StepCard(BaseModel):
    """One card in the glass-box log."""

    n: int
    stage: str
    title: str
    columns: list[str] = Field(default_factory=list)
    explanation: str
    detail: dict[str, Any] = Field(default_factory=dict)
    fitted: bool
    strip: Strip | None = None
    split: SplitInfo | None = None


class ResultInfo(BaseModel):
    train_rows: int
    test_rows: int
    features: int
    #: Whether the split kept each class's share the same on both sides. ``False`` is
    #: always accompanied by a ``stratification_disabled`` warning saying why.
    stratified: bool = False
    #: The split parameters actually used, so a caller can reproduce the run without
    #: having to assume the defaults were in force.
    test_size: float = 0.0
    seed: int = 0


class PipelineWarning(BaseModel):
    """One condition the user needs to know about that is not tied to a single column.

    A split that could not be stratified, a fold whose features are distributed
    differently from the other, a target that was blank on some rows. These are the
    records that would otherwise be a silent fallback, which is the one thing this
    application is built not to do.

    Extra keys are allowed because each ``kind`` carries its own evidence — the classes
    that were too rare, the count of rows removed — and pinning them here would mean
    editing this file to add a number to a sentence.
    """

    model_config = ConfigDict(extra="allow")

    kind: str
    severity: Literal["info", "warn", "critical"]
    message: str
    suggestion: str = ""


class OptionsInfo(BaseModel):
    scaler: str
    high_card_encoder: str


class AnalysisResponse(BaseModel):
    """The whole document one analysis produces — what the results screen renders."""

    run_id: str
    dataset_id: str
    file: FileInfo
    goal: str
    target: TargetInfo
    columns: list[str] = Field(default_factory=list)
    cleaning: list[str] = Field(default_factory=list)
    quality: QualityOut
    profile: ProfileResponse
    options: OptionsInfo
    dropped: list[DroppedColumn] = Field(default_factory=list)
    steps: list[StepCard] = Field(default_factory=list)
    result: ResultInfo
    #: Conditions raised by the engine itself. Produced on every run; the workspace UI
    #: does not render them yet, so today this is a backend contract that the
    #: single-call clients and ``/api/docs`` can read.
    warnings: list[PipelineWarning] = Field(default_factory=list)
    #: The post-split checks: per-feature drift between the folds, class balance in
    #: each, and the sanity baseline. A permissive document for the reason this file's
    #: docstring gives — callers render it, they do not branch on its leaves.
    checks: dict[str, Any] = Field(default_factory=dict)
    took_ms: int


#: ``DatasetDetail`` names ``RunSummary`` before it is defined, which Pydantic resolves
#: only once both exist.
DatasetDetail.model_rebuild()

#: Refused targets are explained with this number, so the message and the check cannot
#: disagree about where the line is.
MAX_TARGET_VALUES = MAX_TARGET_CARDINALITY
