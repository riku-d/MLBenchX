"""The pipeline configuration — the single source of truth for what happens to a dataset.

Everything downstream is derived from a :class:`PipelineConfig`:

    PipelineConfig
        ├── codegen.generate()  ->  Python the user can read, edit and run
        ├── impact.assess()     ->  what turning a step off would cost
        └── the visual pipeline ->  the cards and toggles in the UI

The configuration is never derived *back* from any of those. That is the point: a
toggle in the UI and a line in the editor cannot disagree, because the toggle writes
the config and the code is generated from it.

There is one deliberate exception, and it is the whole reason the editor exists. Once a
user edits the generated code by hand, the code — not the config — is what runs. The
app says so plainly rather than pretending the toggles still describe what will happen;
see ``CompileSource`` and the ``code_is_authoritative`` flag the workspace shows.

Every step here is *optional* by design. A preprocessing tool that decides for you is
the thing this product exists to argue against: the recommendation is visible, the
reason is stated, and the switch is yours.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.config import (
    DEFAULT_SCALER,
    IMBALANCE_RATIO,
    MAX_MISSING_RATE,
    ONEHOT_MAX_CARDINALITY,
    OUTLIER_IQR_MULTIPLIER,
    SEED,
    TEST_SIZE,
)

__all__ = [
    "StepId",
    "StepConfig",
    "MissingValuesStep",
    "DuplicatesStep",
    "OutliersStep",
    "EncodingStep",
    "ScalingStep",
    "FeatureSelectionStep",
    "ImbalanceStep",
    "PipelineConfig",
    "STEP_ORDER",
    "STEP_LABELS",
]

#: The order steps run in, and the order the visual pipeline draws them.
#:
#: This is not arbitrary and it is not configurable. Each step assumes the previous ones
#: have run: encoding needs the blanks filled or it learns "missing" as a category;
#: scaling needs the categories already numeric; feature selection needs everything
#: numeric to measure correlation at all. Letting a user reorder these would let them
#: build a pipeline that fails in ways that are hard to explain.
STEP_ORDER: tuple[str, ...] = (
    "data_types",
    "missing_values",
    "duplicates",
    "outliers",
    "encoding",
    "scaling",
    "feature_selection",
    "imbalance",
)

StepId = Literal[
    "data_types",
    "missing_values",
    "duplicates",
    "outliers",
    "encoding",
    "scaling",
    "feature_selection",
    "imbalance",
]

#: What each step is called in the interface, and the one-line description under it.
STEP_LABELS: dict[str, tuple[str, str]] = {
    "data_types": (
        "Data Type Detection",
        "Work out what each column actually holds before touching any of it.",
    ),
    "missing_values": (
        "Missing Value Treatment",
        "Fill the gaps, or remove the rows and columns that have too many.",
    ),
    "duplicates": (
        "Duplicate Removal",
        "Drop rows that repeat another row exactly.",
    ),
    "outliers": (
        "Outlier Treatment",
        "Find values far outside the usual range, and decide what to do with them.",
    ),
    "encoding": (
        "Categorical Encoding",
        "Turn text categories into numbers a model can work with.",
    ),
    "scaling": (
        "Feature Scaling",
        "Put every number on a comparable range.",
    ),
    "feature_selection": (
        "Feature Selection",
        "Remove columns that carry no signal or repeat another column.",
    ),
    "imbalance": (
        "Class Imbalance",
        "Report how lopsided the target is, and optionally rebalance the training rows.",
    ),
}


class StepConfig(BaseModel):
    """Fields every step has."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True


class DataTypesStep(StepConfig):
    """Detection only — it changes no values, it decides how later steps read them.

    Cannot be disabled. Every other step needs to know which columns are numbers and
    which are categories; without that there is no pipeline, only a table.
    """

    enabled: Literal[True] = True
    parse_dates: bool = Field(
        default=True, description="Recognise text columns that hold dates."
    )
    coerce_numeric: bool = Field(
        default=True, description="Recognise text columns that hold numbers."
    )


class MissingValuesStep(StepConfig):
    """How blanks are filled, or removed."""

    numeric_strategy: Literal[
        "median", "mean", "constant", "forward_fill", "backward_fill", "drop_rows"
    ] = "median"
    categorical_strategy: Literal[
        "most_frequent", "constant", "forward_fill", "backward_fill", "drop_rows"
    ] = "most_frequent"
    numeric_fill_value: float = Field(
        default=0.0, description="Used only when numeric_strategy is 'constant'."
    )
    categorical_fill_value: str = Field(
        default="Unknown", description="Used only when categorical_strategy is 'constant'."
    )
    #: Columns emptier than this are dropped rather than imputed: past this point the
    #: column records our fill value more than it records the user's data.
    drop_column_threshold: float = Field(
        default=MAX_MISSING_RATE, ge=0.0, le=1.0,
        description="Drop any column missing more than this share of its values.",
    )


class DuplicatesStep(StepConfig):
    """Exact repeats, or repeats within a chosen set of columns."""

    subset: list[str] = Field(
        default_factory=list,
        description="Columns that define a duplicate. Empty means the whole row.",
    )
    keep: Literal["first", "last"] = "first"


class OutliersStep(StepConfig):
    """Values far outside the usual range, and what to do about them.

    Defaults to ``cap`` rather than ``remove``. An outlier is not automatically an
    error — a genuinely large value and a mistyped one look identical from here — and
    deleting rows is the one action in this pipeline that loses data outright.
    """

    method: Literal["iqr", "zscore"] = "iqr"
    action: Literal["cap", "remove", "winsorize", "flag"] = "cap"
    iqr_multiplier: float = Field(default=OUTLIER_IQR_MULTIPLIER, gt=0.0)
    zscore_threshold: float = Field(default=3.0, gt=0.0)
    winsorize_quantile: float = Field(default=0.05, gt=0.0, lt=0.5)
    columns: list[str] = Field(
        default_factory=list, description="Empty means every numeric column."
    )


class EncodingStep(StepConfig):
    """How text categories become numbers.

    Two thresholds rather than one method: a two-value column and a forty-value column
    are different problems, and using one encoder for both either wastes columns or
    invents an ordering.
    """

    binary_method: Literal["label", "onehot"] = "label"
    low_cardinality_method: Literal["onehot", "ordinal", "label"] = "onehot"
    high_cardinality_method: Literal["frequency", "target", "ordinal", "drop"] = "frequency"
    max_onehot_cardinality: int = Field(default=ONEHOT_MAX_CARDINALITY, ge=2, le=200)
    handle_unknown: Literal["ignore", "error"] = "ignore"


class ScalingStep(StepConfig):
    """Putting numbers on a comparable range."""

    method: Literal["standard", "minmax", "robust"] = DEFAULT_SCALER  # type: ignore[assignment]
    columns: list[str] = Field(
        default_factory=list, description="Empty means every numeric column."
    )


class FeatureSelectionStep(StepConfig):
    """Dropping columns that carry nothing, or that repeat another column.

    Off by default. Every other step changes how a column is *represented*; this one
    removes it, and which of two correlated columns to keep is a judgement about what
    the data means rather than something a threshold settles.
    """

    enabled: bool = False
    variance_threshold: float | None = Field(
        default=0.0, ge=0.0, description="Drop numeric columns with variance at or below this."
    )
    correlation_threshold: float | None = Field(
        default=0.95, gt=0.0, le=1.0,
        description="Of two columns correlated above this, drop the second.",
    )
    mutual_information: bool = Field(
        default=False, description="Rank features by mutual information with the target."
    )
    top_k: int | None = Field(
        default=None, ge=1, description="Keep only this many features, by mutual information."
    )


class ImbalanceStep(StepConfig):
    """Reporting a lopsided target, and optionally doing something about it.

    ``report`` is the default and does not touch the data. Resampling changes the rows a
    model trains on, which is a modelling decision rather than a cleaning one, so it is
    opt-in and applied to the training rows only.
    """

    enabled: bool = True
    action: Literal["report", "oversample", "undersample", "class_weight"] = "report"
    ratio_threshold: float = Field(default=IMBALANCE_RATIO, gt=1.0)


class PipelineConfig(BaseModel):
    """Everything that will happen to one dataset, as data."""

    model_config = ConfigDict(extra="forbid")

    target: str | None = Field(
        default=None, description="Column being predicted. Never transformed as a feature."
    )
    test_size: float = Field(default=TEST_SIZE, gt=0.05, lt=0.5)
    seed: int = Field(default=SEED)
    #: Columns the user removed by hand. Kept separate from the steps because it is a
    #: stated preference, not a rule with a threshold behind it.
    excluded_columns: list[str] = Field(default_factory=list)

    data_types: DataTypesStep = Field(default_factory=DataTypesStep)
    missing_values: MissingValuesStep = Field(default_factory=MissingValuesStep)
    duplicates: DuplicatesStep = Field(default_factory=DuplicatesStep)
    outliers: OutliersStep = Field(default_factory=OutliersStep)
    encoding: EncodingStep = Field(default_factory=EncodingStep)
    scaling: ScalingStep = Field(default_factory=ScalingStep)
    feature_selection: FeatureSelectionStep = Field(default_factory=FeatureSelectionStep)
    imbalance: ImbalanceStep = Field(default_factory=ImbalanceStep)

    def step(self, step_id: str) -> StepConfig:
        """One step by id, in the order :data:`STEP_ORDER` declares."""
        return getattr(self, step_id)

    def enabled_steps(self) -> list[str]:
        return [s for s in STEP_ORDER if self.step(s).enabled]

    def disabled_steps(self) -> list[str]:
        return [s for s in STEP_ORDER if not self.step(s).enabled]


Annotated_PipelineConfig = Annotated[PipelineConfig, Field(description="Pipeline configuration")]
