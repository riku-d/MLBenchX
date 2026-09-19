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
    """The health score, what it cost, and why."""

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
    took_ms: int


#: ``DatasetDetail`` names ``RunSummary`` before it is defined, which Pydantic resolves
#: only once both exist.
DatasetDetail.model_rebuild()

#: Refused targets are explained with this number, so the message and the check cannot
#: disagree about where the line is.
MAX_TARGET_VALUES = MAX_TARGET_CARDINALITY
