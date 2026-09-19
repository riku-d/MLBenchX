"""Glass-box preprocessing engine for ML-BenchX.

This module turns a clean ``pandas.DataFrame`` into model-ready train/test matrices
*and* a plain-English record of every transformation that was applied. It sits between
dataset ingestion (upstream) and model training (downstream).

Four invariants define the module:

I1 — Split before fit.
    The train/test split happens before any transformer is fitted. Every learned
    parameter (imputation medians, scaler means, encoder categories, frequency tables)
    comes from the training partition alone. See :func:`fit_preprocessing`, where the
    ordering is explicit and commented.

I2 — The log is derived, never hand-written.
    :func:`introspect` reads the *fitted* sklearn objects and reports their learned
    attributes (``SimpleImputer.statistics_``, ``StandardScaler.mean_``/``.scale_``,
    ``OneHotEncoder.categories_``, ``FrequencyEncoder.frequencies_``). No number in the
    log is computed a second time alongside the transformation, so the log cannot drift
    out of sync with what actually happened.

I3 — Decisions are separated from execution.
    :func:`build_plan` decides which transformer applies to which column. It is a pure
    function of the column profile: it accepts no DataFrame and returns a plain data
    structure. :func:`build_pipeline` turns that plan into sklearn objects. This makes
    every routing decision unit-testable without data, and lets the UI show the user a
    plan before anything runs.

I4 — Fully deterministic.
    No LLM calls, no network, no randomness beyond the caller-supplied seed. Explanation
    strings come from templates keyed on transformer type and parameters, and every
    learned table is emitted in a sorted, stable order. Identical input therefore
    produces a byte-identical log.

Design note — why frequency encoding and not target encoding.
    High-cardinality categoricals are encoded by :class:`FrequencyEncoder`, which
    replaces each value by how often it occurs in the training rows. Target (mean)
    encoding is deliberately *not* implemented: it reads ``y``, so making it
    leakage-safe requires nested out-of-fold fitting, and a naive implementation is one
    of the most common sources of silently optimistic AutoML scores. Frequency encoding
    never touches ``y`` and so cannot leak by construction.

Design note — where dropped columns are recorded.
    Dropped columns are recorded in :attr:`PreprocessingPlan.dropped` (column -> human
    readable reason) rather than as entries in the introspection log. The log describes
    *fitted* transformers, and every entry it emits carries
    ``fitted_on == "train_fold_only"``; a drop is decided from the schema before any
    data is read, so putting it in the same list would mean labelling it with a fold it
    was never fitted on. :func:`format_report` prints the drops ahead of the steps so
    the user still sees one continuous narrative.

Out of scope by design: target encoding, datetime feature decomposition, polynomial
features, PCA, feature selection, class-imbalance resampling, hyperparameter tuning,
model training, SHAP, persistence. This module returns data structures; storing them is
the caller's job.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import textwrap
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

__all__ = [
    "ONEHOT_MAX_CARDINALITY",
    "MAX_MISSING_RATE",
    "ColumnProfile",
    "FrequencyEncoder",
    "StepPlan",
    "PreprocessingPlan",
    "PreprocessingResult",
    "profile_frame",
    "build_plan",
    "build_pipeline",
    "introspect",
    "build_feature_map",
    "fit_preprocessing",
    "export_script",
    "format_log",
    "format_report",
]


# --------------------------------------------------------------------------------------
# Tunable thresholds. Named constants rather than literals so the plan, the explanations
# and the tests all quote the same number.
# --------------------------------------------------------------------------------------

#: A categorical column with at most this many distinct values is one-hot encoded.
#: Above it, one-hot would add more columns than the model can usefully absorb, so the
#: column is frequency encoded instead.
ONEHOT_MAX_CARDINALITY = 15

#: Columns missing more than this fraction of their values are dropped: whatever is
#: imputed would be mostly invented data.
MAX_MISSING_RATE = 0.60

#: A column whose distinct values exceed this share of the rows, *and* which has more
#: than :data:`TEXT_MIN_UNIQUE` of them, is treated as free text or an identifier.
TEXT_UNIQUE_RATE = 0.5
TEXT_MIN_UNIQUE = 50

#: Fixed order of the transformer blocks. ``ColumnTransformer`` emits output columns in
#: the order its transformers are listed, so the pipeline builder and the feature-map
#: builder must agree on this tuple.
GROUP_ORDER: tuple[str, ...] = ("numeric", "binary", "low_card", "high_card")

#: Explanations name at most this many columns inline before switching to a summary.
#: Every value is always present in the machine-readable ``details`` block.
_MAX_COLUMNS_IN_EXPLANATION = 6

_STAGE_TO_STEP_NAME = {"impute": "impute", "encode": "encode", "scale": "scale"}


# --------------------------------------------------------------------------------------
# 1. Profiling — pure description, no decisions
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnProfile:
    """What a single column *is*, with no opinion about what to do with it."""

    name: str
    kind: Literal["numeric", "categorical", "datetime", "text", "constant"]
    dtype: str
    n_missing: int
    missing_rate: float
    n_unique: int
    unique_rate: float


def _classify(series: pd.Series, n_unique: int, unique_rate: float) -> str:
    """Assign a ``kind`` to one column. Rules are applied in this order, first match wins."""
    if n_unique <= 1:
        return "constant"
    if pd.api.types.is_bool_dtype(series):
        # Booleans are numeric to numpy but behave like two-level categories.
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if unique_rate > TEXT_UNIQUE_RATE and n_unique > TEXT_MIN_UNIQUE:
        return "text"
    return "categorical"


def profile_frame(df: pd.DataFrame) -> dict[str, ColumnProfile]:
    """Describe every column of ``df``.

    Returns a mapping of column name to :class:`ColumnProfile`, in the frame's column
    order. This function makes no decisions; it only measures. All downstream routing is
    a function of its output, which is what makes routing testable without data.
    """
    n_rows = len(df)
    profiles: dict[str, ColumnProfile] = {}
    for name in df.columns:
        series = df[name]
        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))
        missing_rate = (n_missing / n_rows) if n_rows else 0.0
        unique_rate = (n_unique / n_rows) if n_rows else 0.0
        profiles[str(name)] = ColumnProfile(
            name=str(name),
            kind=_classify(series, n_unique, unique_rate),  # type: ignore[arg-type]
            dtype=str(series.dtype),
            n_missing=n_missing,
            missing_rate=missing_rate,
            n_unique=n_unique,
            unique_rate=unique_rate,
        )
    return profiles


# --------------------------------------------------------------------------------------
# 2. Custom transformer for high-cardinality categoricals
# --------------------------------------------------------------------------------------


class FrequencyEncoder(TransformerMixin, BaseEstimator):
    """Replace each category by how often it occurs in the training rows.

    One output column per input column, so a 400-city column costs one feature instead
    of 400. Values not seen during ``fit`` map to ``0.0`` — genuinely novel categories
    are rare by definition, which is what a zero frequency encodes.

    This encoder never reads ``y``, so it cannot leak target information. See the module
    docstring for why target encoding is not offered as an alternative.

    Mixins are listed before :class:`~sklearn.base.BaseEstimator` because sklearn's tag
    system resolves ``__sklearn_tags__`` through the MRO; the reverse order silently
    drops the transformer tags.

    Attributes set by ``fit``:
        frequencies_: column -> {category: relative frequency in the training fold}.
        feature_names_in_: input column names, in order.
        n_features_in_: number of input columns.
    """

    def _as_frame(self, X: Any, fitting: bool = False) -> pd.DataFrame:
        """Accept a DataFrame (the normal path) or a bare array, always return a frame."""
        if isinstance(X, pd.DataFrame):
            return X
        array = np.asarray(X, dtype=object)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if fitting:
            columns = [f"x{i}" for i in range(array.shape[1])]
        else:
            columns = list(self.feature_names_in_)
        return pd.DataFrame(array, columns=columns)

    def fit(self, X: Any, y: Any = None) -> "FrequencyEncoder":
        """Learn each column's relative value frequencies from the rows it is given."""
        frame = self._as_frame(X, fitting=True)
        self.feature_names_in_ = np.asarray(list(frame.columns), dtype=object)
        self.n_features_in_ = frame.shape[1]
        self.frequencies_ = {}
        for column in frame.columns:
            counts = frame[column].value_counts(normalize=True, dropna=True)
            # Sort by descending frequency, then by string form, so equal-frequency
            # categories always come out in the same order (invariant I4).
            ordered = sorted(counts.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
            self.frequencies_[column] = {key: float(value) for key, value in ordered}
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        """Map every value to its learned frequency; unseen categories become ``0.0``."""
        check_is_fitted(self, "frequencies_")
        frame = self._as_frame(X)
        out_names = list(self.get_feature_names_out())
        columns = {}
        for source, out_name in zip(self.feature_names_in_, out_names):
            table = self.frequencies_[source]
            mapped = frame[source].map(table)
            columns[out_name] = pd.to_numeric(mapped, errors="coerce").fillna(0.0).astype(float)
        return pd.DataFrame(columns, index=frame.index)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return ``"{column}_freq"`` per input column.

        ``ColumnTransformer.set_output(transform="pandas")`` relies on this to label the
        block's output columns, and :func:`build_feature_map` relies on it to trace each
        output column back to its source.
        """
        check_is_fitted(self, "frequencies_")
        names = list(self.feature_names_in_) if input_features is None else list(input_features)
        return np.asarray([f"{name}_freq" for name in names], dtype=object)


# --------------------------------------------------------------------------------------
# 3. The plan — decisions, as data
# --------------------------------------------------------------------------------------


@dataclass
class StepPlan:
    """One planned transformation: what will run, on which columns, and why."""

    step_id: int
    stage: Literal["drop", "impute", "encode", "scale"]
    columns: list[str]
    transformer: str
    params: dict[str, Any]
    reason: str


@dataclass
class PreprocessingPlan:
    """Everything that will happen to the feature columns, decided before any fitting."""

    target: str
    task_type: str
    dropped: dict[str, str] = field(default_factory=dict)
    steps: list[StepPlan] = field(default_factory=list)
    numeric: list[str] = field(default_factory=list)
    binary: list[str] = field(default_factory=list)
    low_card: list[str] = field(default_factory=list)
    high_card: list[str] = field(default_factory=list)

    @property
    def feature_columns(self) -> list[str]:
        """Surviving input columns, in the order the pipeline consumes them."""
        return [*self.numeric, *self.binary, *self.low_card, *self.high_card]


def _pct(rate: float) -> str:
    """Format a 0..1 rate as a percentage, dropping a trailing ``.0``."""
    text = f"{rate * 100:.1f}"
    if text.endswith(".0"):
        text = text[:-2]
    return f"{text}%"


_REASON_IMPUTE_MEDIAN = (
    "Numeric columns can have blanks, and the median is used rather than the mean "
    "because a few extreme values cannot drag it around."
)
_REASON_SCALE = (
    "Distance- and gradient-based models compare columns directly, so every number is "
    "put on a comparable range first."
)
_REASON_IMPUTE_MODE = (
    "The most common value is the safest stand-in for a missing label, and it is "
    "guaranteed to be a category the encoder already knows."
)
_REASON_BINARY = "Exactly two distinct values, so a single 0/1 column carries all of it."
_REASON_LOW_CARD = (
    f"At most {ONEHOT_MAX_CARDINALITY} distinct values, so one column per value is "
    "affordable and avoids implying the categories have an order."
)
_REASON_HIGH_CARD = (
    f"More than {ONEHOT_MAX_CARDINALITY} distinct values, so one column per value "
    "would swamp the model. Frequency encoding costs one column and reads only the "
    "feature itself, never the target."
)

_ENCODERS: dict[str, tuple[str, dict[str, Any], str]] = {
    "binary": (
        "OrdinalEncoder",
        {"handle_unknown": "use_encoded_value", "unknown_value": -1},
        _REASON_BINARY,
    ),
    "low_card": ("OneHotEncoder", {"handle_unknown": "ignore"}, _REASON_LOW_CARD),
    "high_card": ("FrequencyEncoder", {"unseen_value": 0.0}, _REASON_HIGH_CARD),
}


def build_plan(
    profiles: dict[str, ColumnProfile],
    target: str,
    task_type: str,
    excluded: dict[str, str] | None = None,
) -> PreprocessingPlan:
    """Decide what to do with every column, from the profile alone.

    Pure and deterministic: no DataFrame is accepted and nothing is fitted, so two calls
    with the same profile return equal plans. This is what lets the UI show the user the
    plan before it runs, and lets the routing rules be tested without any data.

    Args:
        profiles: output of :func:`profile_frame`, normally on the features only.
        target: name of the label column; never becomes a feature.
        task_type: ``"classification"`` or ``"regression"``. Recorded on the plan and
            used downstream to decide stratification.
        excluded: column -> reason, supplied by the caller (leakage screen, or a column
            the user deselected in the UI). These reasons take precedence over the
            generic ones so the more specific message is what the user sees.

    Drop precedence, first match wins: target, caller exclusion, mostly-missing,
    constant, datetime, free text. Mostly-missing is checked before constant so that an
    all-blank column reports "100% of values are missing" rather than the less
    actionable "constant".
    """
    excluded = dict(excluded or {})
    plan = PreprocessingPlan(target=target, task_type=task_type)

    for name, profile in profiles.items():
        if name == target:
            plan.dropped[name] = "this is the target column, never a feature"
            continue
        if name in excluded:
            plan.dropped[name] = excluded[name]
            continue
        if profile.missing_rate > MAX_MISSING_RATE:
            plan.dropped[name] = f"{_pct(profile.missing_rate)} of values are missing"
            continue
        if profile.kind == "constant":
            plan.dropped[name] = "constant — a single value carries no information"
            continue
        if profile.kind == "datetime":
            # Building calendar features from a timestamp is out of scope for this
            # version. Dropping it loudly beats letting it vanish unexplained.
            plan.dropped[name] = (
                "date/time column — this version does not build calendar features, "
                "so it cannot be used as-is"
            )
            continue
        if profile.kind == "text":
            plan.dropped[name] = (
                f"free text / identifier — {profile.n_unique} distinct values in "
                f"{_pct(profile.unique_rate)} of rows"
            )
            continue

        if profile.kind == "numeric":
            plan.numeric.append(name)
        elif profile.n_unique == 2:
            plan.binary.append(name)
        elif profile.n_unique <= ONEHOT_MAX_CARDINALITY:
            plan.low_card.append(name)
        else:
            plan.high_card.append(name)

    step_id = 1
    for group in GROUP_ORDER:
        columns: list[str] = list(getattr(plan, group))
        if not columns:
            continue
        if group == "numeric":
            plan.steps.append(
                StepPlan(step_id, "impute", columns, "SimpleImputer",
                         {"strategy": "median", "keep_empty_features": True},
                         _REASON_IMPUTE_MEDIAN)
            )
            step_id += 1
            plan.steps.append(
                StepPlan(step_id, "scale", columns, "StandardScaler", {}, _REASON_SCALE)
            )
            step_id += 1
        else:
            plan.steps.append(
                StepPlan(step_id, "impute", columns, "SimpleImputer",
                         {"strategy": "most_frequent", "keep_empty_features": True},
                         _REASON_IMPUTE_MODE)
            )
            step_id += 1
            transformer, params, reason = _ENCODERS[group]
            plan.steps.append(
                StepPlan(step_id, "encode", columns, transformer, dict(params), reason)
            )
            step_id += 1

    return plan


# --------------------------------------------------------------------------------------
# 4. Building the sklearn pipeline
# --------------------------------------------------------------------------------------


def _block(group: str) -> Pipeline:
    """The two-step Pipeline for one column group.

    Inner step names (``impute``/``encode``/``scale``) are part of the contract:
    :func:`introspect` and :func:`build_feature_map` look transformers up by them.
    """
    if group == "numeric":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ])

    imputer = SimpleImputer(strategy="most_frequent", keep_empty_features=True)
    if group == "binary":
        encoder: Any = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    elif group == "low_card":
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    elif group == "high_card":
        encoder = FrequencyEncoder()
    else:  # pragma: no cover - guarded by GROUP_ORDER
        raise ValueError(f"unknown column group: {group!r}")
    return Pipeline([("impute", imputer), ("encode", encoder)])


def build_pipeline(plan: PreprocessingPlan) -> ColumnTransformer:
    """Turn a plan into a fitted-ready ``ColumnTransformer``.

    One named block per non-empty group, named exactly after the group.
    ``set_output(transform="pandas")`` is essential: without it the transformed frame
    loses its column names and the feature map degrades into string-parsing guesswork.
    """
    transformers = [
        (group, _block(group), list(getattr(plan, group)))
        for group in GROUP_ORDER
        if getattr(plan, group)
    ]
    if not transformers:
        raise ValueError(
            "No usable feature columns remain, so there is nothing to preprocess. "
            "Dropped columns and reasons: " + json.dumps(plan.dropped, indent=2)
        )
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


# --------------------------------------------------------------------------------------
# 5. Introspection — the log is read off the fitted objects
# --------------------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    """Convert numpy/pandas scalars into something ``json.dumps`` accepts."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return None if (math.isnan(number) or math.isinf(number)) else number
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        return value
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(_jsonable(key)): _jsonable(item) for key, item in value.items()}
    return str(value)


def _fmt_number(value: Any) -> str:
    """Format a learned number for a sentence a non-technical reader will see."""
    number = float(value)
    if math.isnan(number):
        return "not available"
    if number == int(number) and abs(number) < 1e15:
        return str(int(number))
    return f"{number:.6g}"


def _fmt_value(value: Any) -> str:
    """Format a category value for display, quoting anything non-numeric."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return _fmt_number(value)
    return f"'{value}'"


def _join_clauses(clauses: list[str]) -> str:
    """Join per-column clauses, summarising the tail when there are many columns."""
    if len(clauses) <= _MAX_COLUMNS_IN_EXPLANATION:
        return "; ".join(clauses)
    shown = "; ".join(clauses[:_MAX_COLUMNS_IN_EXPLANATION])
    hidden = len(clauses) - _MAX_COLUMNS_IN_EXPLANATION
    return f"{shown}; and {hidden} more (full values in the details)"


def _group_of(plan: PreprocessingPlan, step: StepPlan) -> str:
    """Which pipeline block a planned step belongs to.

    Recovered by matching the step's column list against the plan's groups. The groups
    are disjoint and each step's columns are exactly one group's list, so the match is
    exact — this keeps :class:`StepPlan` to the fields the contract specifies.
    """
    for group in GROUP_ORDER:
        if list(getattr(plan, group)) == list(step.columns):
            return group
    raise ValueError(f"step {step.step_id} does not correspond to any column group")


def _fitted_step(ct: ColumnTransformer, group: str, stage: str) -> Any:
    """Pull one fitted inner transformer out of the fitted ColumnTransformer."""
    block = ct.named_transformers_[group]
    return block.named_steps[_STAGE_TO_STEP_NAME[stage]]


def _impute_entry(step: StepPlan, imputer: Any, train_raw: pd.DataFrame) -> dict[str, Any]:
    """Log entry for an imputation step, read from ``imputer.statistics_``."""
    strategy = step.params["strategy"]
    columns = list(imputer.feature_names_in_)
    statistics = list(imputer.statistics_)

    fill_values: dict[str, Any] = {}
    blanks_filled: dict[str, int] = {}
    clauses: list[str] = []
    for column, statistic in zip(columns, statistics):
        fill_values[column] = _jsonable(statistic)
        n_blank = int(train_raw[column].isna().sum())
        blanks_filled[column] = n_blank
        if n_blank == 0:
            continue
        is_empty = isinstance(statistic, (float, np.floating)) and math.isnan(float(statistic))
        if is_empty:
            clauses.append(
                f"`{column}` — {n_blank} blank(s) set to 0, because the training rows "
                "held no value at all to learn from"
            )
        else:
            clauses.append(f"`{column}` — {n_blank} blank(s) set to {_fmt_value(statistic)}")

    if not clauses:
        explanation = "No blanks to fill in these columns."
    elif strategy == "median":
        explanation = (
            "Filled blanks with the median of the training rows only — the median is "
            "used because a handful of extreme values cannot drag it around: "
            f"{_join_clauses(clauses)}."
        )
    else:
        explanation = (
            "Filled blanks with the most common value in the training rows only, so "
            "every filled-in cell is a category the model already knows: "
            f"{_join_clauses(clauses)}."
        )

    return {
        "details": {"strategy": strategy, "fill_values": fill_values, "blanks_filled": blanks_filled},
        "explanation": explanation,
    }


def _scale_entry(scaler: Any) -> dict[str, Any]:
    """Log entry for standardisation, read from ``scaler.mean_`` and ``scaler.scale_``."""
    columns = list(scaler.feature_names_in_)
    means = {c: _jsonable(m) for c, m in zip(columns, scaler.mean_)}
    scales = {c: _jsonable(s) for c, s in zip(columns, scaler.scale_)}
    clauses = [
        f"`{c}` — centred by subtracting {_fmt_number(scaler.mean_[i])}, then divided by "
        f"{_fmt_number(scaler.scale_[i])}"
        for i, c in enumerate(columns)
    ]
    explanation = (
        "Rescaled each number to mean 0, spread 1, so a column measured in thousands "
        "does not outweigh one measured in single digits. The centre and spread come "
        f"from the training rows only: {_join_clauses(clauses)}."
    )
    return {"details": {"mean": means, "scale": scales}, "explanation": explanation}


def _onehot_entry(encoder: Any) -> dict[str, Any]:
    """Log entry for one-hot encoding, read from ``encoder.categories_``."""
    columns = list(encoder.feature_names_in_)
    categories = {c: _jsonable(cats) for c, cats in zip(columns, encoder.categories_)}
    widths = {c: len(cats) for c, cats in zip(columns, encoder.categories_)}
    total = sum(widths.values())
    clauses = [f"`{c}` -> {widths[c]} columns" for c in columns]
    noun = "category column" if len(columns) == 1 else "category columns"
    explanation = (
        f"Turned {len(columns)} {noun} into {total} yes/no columns so the "
        "model does not assume the categories have an order: "
        f"{_join_clauses(clauses)}. A value not seen during training turns every one of "
        "its yes/no columns off rather than causing an error."
    )
    return {
        "details": {"categories": categories, "columns_created": widths, "total_columns_created": total},
        "explanation": explanation,
    }


def _ordinal_entry(encoder: Any) -> dict[str, Any]:
    """Log entry for the two-value ordinal encoding, read from ``encoder.categories_``."""
    columns = list(encoder.feature_names_in_)
    categories = {c: _jsonable(cats) for c, cats in zip(columns, encoder.categories_)}
    codes = {
        c: {str(_jsonable(v)): i for i, v in enumerate(cats)}
        for c, cats in zip(columns, encoder.categories_)
    }
    clauses = [
        f"`{c}` — " + ", ".join(f"{_fmt_value(v)} -> {i}" for i, v in enumerate(cats))
        for c, cats in zip(columns, encoder.categories_)
    ]
    explanation = (
        "Replaced each two-value column with a single 0/1 number, which is all the "
        f"information there is in two values: {_join_clauses(clauses)}. A value never "
        f"seen in training becomes {encoder.unknown_value}, which marks it as unknown "
        "instead of guessing."
    )
    return {"details": {"categories": categories, "codes": codes}, "explanation": explanation}


def _frequency_entry(encoder: Any) -> dict[str, Any]:
    """Log entry for frequency encoding, read from ``encoder.frequencies_``."""
    columns = list(encoder.feature_names_in_)
    frequencies = {c: _jsonable(encoder.frequencies_[c]) for c in columns}
    counts = {c: len(encoder.frequencies_[c]) for c in columns}
    clauses = [f"`{c}` — {counts[c]} distinct values, now 1 column" for c in columns]
    explanation = (
        "Too many categories to one-hot, so each value was replaced by how often it "
        "appears in the training rows. This uses no information about the target, so it "
        f"cannot leak: {_join_clauses(clauses)}. A value not seen during training gets "
        "0, which correctly says 'never observed'."
    )
    return {
        "details": {"frequencies": frequencies, "n_categories": counts},
        "explanation": explanation,
    }


def introspect(
    ct: ColumnTransformer,
    plan: PreprocessingPlan,
    train_raw: pd.DataFrame,
) -> list[dict]:
    """Read the fitted pipeline and describe what it learned, in plain English.

    This is invariant I2 in code. Nothing here recomputes a statistic: every number is
    read off a fitted attribute (``statistics_``, ``mean_``, ``scale_``, ``categories_``,
    ``frequencies_``), with blank counts taken from ``train_raw``. A log built this way
    cannot disagree with the pipeline it describes.

    Args:
        ct: the fitted ``ColumnTransformer`` from :func:`build_pipeline`.
        plan: the plan it was built from.
        train_raw: the training frame *before* transformation, used only to count how
            many blanks each imputation actually filled.

    Returns:
        One dict per step, ``step`` ids contiguous from 1, JSON-serialisable.
    """
    check_is_fitted(ct)
    log: list[dict] = []
    for position, step in enumerate(plan.steps, start=1):
        group = _group_of(plan, step)
        fitted = _fitted_step(ct, group, step.stage)

        if step.stage == "impute":
            body = _impute_entry(step, fitted, train_raw)
        elif step.stage == "scale":
            body = _scale_entry(fitted)
        elif step.transformer == "OneHotEncoder":
            body = _onehot_entry(fitted)
        elif step.transformer == "OrdinalEncoder":
            body = _ordinal_entry(fitted)
        elif step.transformer == "FrequencyEncoder":
            body = _frequency_entry(fitted)
        else:  # pragma: no cover - unreachable while _ENCODERS is the only source
            raise ValueError(f"no explanation template for {step.transformer!r}")

        log.append({
            "step": position,
            "stage": step.stage,
            "columns": list(step.columns),
            "transformer": step.transformer,
            "params": _jsonable(step.params),
            "fitted_on": "train_fold_only",
            "reason": step.reason,
            "details": body["details"],
            "explanation": body["explanation"],
        })
    return log


# --------------------------------------------------------------------------------------
# 6. Feature map — transformed column back to its source column
# --------------------------------------------------------------------------------------


def build_feature_map(ct: ColumnTransformer, plan: PreprocessingPlan) -> dict[str, str]:
    """Map each transformed feature name to the original column it came from.

    Downstream SHAP explains transformed features (``department_CSE``), but users think
    in source columns (``department``). This mapping is what lets per-feature SHAP
    magnitudes be summed back into one number per column the user recognises.

    Every column of the transformed frame is guaranteed to appear as a key; a missing
    one is a bug in this module rather than something to paper over, so it raises.
    """
    check_is_fitted(ct)
    mapping: dict[str, str] = {}

    for group in GROUP_ORDER:
        columns = list(getattr(plan, group))
        if not columns:
            continue
        block = ct.named_transformers_[group]

        if group in ("numeric", "binary"):
            # One output column per input column, in order.
            out_names = list(block.get_feature_names_out(columns))
            for out_name, source in zip(out_names, columns):
                mapping[str(out_name)] = source
        elif group == "low_card":
            encoder = block.named_steps["encode"]
            out_names = list(encoder.get_feature_names_out(columns))
            index = 0
            for source, categories in zip(columns, encoder.categories_):
                for _ in categories:
                    mapping[str(out_names[index])] = source
                    index += 1
        else:  # high_card
            encoder = block.named_steps["encode"]
            out_names = list(encoder.get_feature_names_out(columns))
            for out_name, source in zip(out_names, columns):
                mapping[str(out_name)] = source

    produced = [str(name) for name in ct.get_feature_names_out()]
    orphans = [name for name in produced if name not in mapping]
    if orphans:
        raise RuntimeError(
            f"feature map is incomplete; these transformed columns have no source: {orphans}"
        )
    return {name: mapping[name] for name in produced}


# --------------------------------------------------------------------------------------
# 7. Orchestration
# --------------------------------------------------------------------------------------


@dataclass
class PreprocessingResult:
    """Model-ready data, plus everything needed to explain and reproduce it."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    plan: PreprocessingPlan
    pipeline: ColumnTransformer
    log: list[dict]
    feature_map: dict[str, str]


def _is_classification(task_type: str) -> bool:
    """Interpret the caller's task label; unrecognised values fail loudly."""
    normalised = task_type.strip().lower()
    if "class" in normalised:
        return True
    if "regress" in normalised:
        return False
    raise ValueError(
        f"task_type must describe classification or regression, got {task_type!r}"
    )


def fit_preprocessing(
    df: pd.DataFrame,
    target: str,
    task_type: str,
    excluded: dict[str, str] | None = None,
    test_size: float = 0.2,
    seed: int = 42,
) -> PreprocessingResult:
    """Profile, plan, split, fit, and explain — in that order.

    The ordering below *is* invariant I1. The split happens at step 4 and the single
    ``fit_transform`` call at step 5 sees ``X_train`` only, so no medians, means,
    categories or frequency tables can carry information from the test rows.

    Args:
        df: clean input frame, features and target together.
        target: label column, present in ``df``.
        task_type: ``"classification"`` or ``"regression"``; classification splits are
            stratified on the label.
        excluded: column -> reason for columns the caller wants removed (leakage screen,
            user deselection). The reason is surfaced to the user verbatim.
        test_size: fraction held out.
        seed: the only source of randomness in the module.

    Returns:
        :class:`PreprocessingResult`. ``X_train``/``X_test`` are transformed frames with
        identical columns and no missing values.
    """
    if target not in df.columns:
        raise ValueError(f"target column {target!r} is not in the frame")
    classification = _is_classification(task_type)

    # 1. Separate the label from the features.
    y = df[target]
    X = df.drop(columns=[target])

    # 2. Decide everything from the schema. Nothing is fitted, nothing is measured on
    #    values that will end up in the test fold beyond the column profile itself.
    plan = build_plan(profile_frame(X), target=target, task_type=task_type, excluded=excluded)

    # 3. Remove the columns the plan rejected.
    X = X.drop(columns=[c for c in plan.dropped if c in X.columns])
    if X.shape[1] == 0:
        raise ValueError(
            "Every column was dropped, so there is nothing to train on. Reasons: "
            + json.dumps(plan.dropped, indent=2)
        )

    # 4. Split BEFORE any transformer exists. Stratify so a rare class is represented in
    #    both folds; fall back when a class is too small to appear on both sides.
    stratify = None
    if classification:
        counts = y.value_counts(dropna=False)
        if len(counts) > 1 and int(counts.min()) >= 2:
            stratify = y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, shuffle=True, stratify=stratify
    )

    # 5. Fit on the training partition only, then apply the frozen pipeline to the test
    #    partition. This is the one and only `.fit` in the module's data path.
    pipeline = build_pipeline(plan)
    X_train_out = pipeline.fit_transform(X_train)
    X_test_out = pipeline.transform(X_test)

    # 6. Explain what was learned, and record where each output column came from.
    log = introspect(pipeline, plan, X_train)
    feature_map = build_feature_map(pipeline, plan)

    return PreprocessingResult(
        X_train=X_train_out,
        X_test=X_test_out,
        y_train=y_train,
        y_test=y_test,
        plan=plan,
        pipeline=pipeline,
        log=log,
        feature_map=feature_map,
    )


# --------------------------------------------------------------------------------------
# 8. Script export
# --------------------------------------------------------------------------------------


_SCRIPT_TEMPLATE = '''"""Preprocessing for experiment {exp_id}, exported by ML-BenchX.

Runnable as-is. It reproduces exactly the pipeline whose decisions were shown to you in
the app: the same columns, the same transformers, the same seed, and the same ordering —
split first, then fit on the training rows only.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
{extra_imports}
CSV_PATH = {csv!r}
TARGET = {target!r}
TASK_TYPE = {task_type!r}
TEST_SIZE = {test_size!r}
SEED = {seed!r}

# Columns dropped before modelling, and why.
DROPPED = {dropped}

NUMERIC = {numeric}
BINARY = {binary}
LOW_CARD = {low_card}
HIGH_CARD = {high_card}
{loader_block}{frequency_encoder}

def build_pipeline() -> ColumnTransformer:
    """The same ColumnTransformer the app built for this experiment."""
    blocks = []
{blocks}    return ColumnTransformer(
        transformers=blocks,
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def main() -> None:
    df = {loader_call}
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    X = X.drop(columns=[c for c in DROPPED if c in X.columns])

    stratify = y if TASK_TYPE == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, shuffle=True, stratify=stratify
    )

    pipeline = build_pipeline()
    # --- Fitted on the TRAINING partition only. Everything the transformers learn
    # --- (medians, means, categories, frequencies) comes from these rows alone.
    X_train_out = pipeline.fit_transform(X_train)
    X_test_out = pipeline.transform(X_test)

    print("train:", X_train_out.shape, " test:", X_test_out.shape)
    print("features:", list(X_train_out.columns))


if __name__ == "__main__":
    main()
'''

_BLOCK_SOURCE = {
    "numeric": (
        '    blocks.append(("numeric", Pipeline([\n'
        '        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),\n'
        '        ("scale", StandardScaler()),\n'
        '    ]), NUMERIC))\n'
    ),
    "binary": (
        '    blocks.append(("binary", Pipeline([\n'
        '        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),\n'
        '        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),\n'
        '    ]), BINARY))\n'
    ),
    "low_card": (
        '    blocks.append(("low_card", Pipeline([\n'
        '        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),\n'
        '        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),\n'
        '    ]), LOW_CARD))\n'
    ),
    "high_card": (
        '    blocks.append(("high_card", Pipeline([\n'
        '        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),\n'
        '        ("encode", FrequencyEncoder()),\n'
        '    ]), HIGH_CARD))\n'
    ),
}


def export_script(
    result: PreprocessingResult,
    csv: str,
    exp_id: str,
    seed: int = 42,
    loader_block: str = "",
    loader_call: str = "pd.read_csv(CSV_PATH)",
) -> str:
    """Emit standalone Python that reproduces this preprocessing exactly.

    The generated file carries the same column lists, transformer parameters and seed,
    splits before it fits, and marks the fit-on-train-only line with a comment. When the
    plan uses :class:`FrequencyEncoder`, its source is embedded via :mod:`inspect` rather
    than copied into a string, so the export cannot drift from the real implementation.

    Args:
        result: the outcome of :func:`fit_preprocessing`.
        csv: path the generated script should read the dataset from.
        exp_id: experiment identifier, recorded in the file's docstring.
        seed: the seed passed to :func:`fit_preprocessing`. It is not recoverable from
            :class:`PreprocessingResult`, whose fields are fixed by the module contract,
            so callers that override the default must pass the same value here for the
            export to reproduce their split.
        loader_block: optional Python source defining how to read the file — separator,
            encoding, and any repairs applied before preprocessing. Without it the
            script assumes a plain comma-separated file, which will not reproduce the
            cleaned data for an upload that needed repairing.
        loader_call: the expression the script uses to obtain the frame. Must pair with
            ``loader_block``.

    Returns the source text; writing it to disk is the caller's job.
    """
    plan = result.plan
    uses_frequency = bool(plan.high_card)

    if uses_frequency:
        encoder_source = (
            "\n\n"
            + textwrap.dedent(inspect.getsource(FrequencyEncoder)).rstrip()
            + "\n"
        )
        extra_imports = (
            "from sklearn.base import BaseEstimator, TransformerMixin\n"
            "from sklearn.utils.validation import check_is_fitted\n"
            "from typing import Any\n"
        )
    else:
        encoder_source = ""
        extra_imports = ""

    blocks = "".join(
        _BLOCK_SOURCE[group] for group in GROUP_ORDER if getattr(plan, group)
    )

    source = _SCRIPT_TEMPLATE.format(
        exp_id=exp_id,
        csv=csv,
        target=plan.target,
        task_type="classification" if _is_classification(plan.task_type) else "regression",
        test_size=round(len(result.X_test) / (len(result.X_train) + len(result.X_test)), 4),
        seed=seed,
        dropped=_pretty_dict(plan.dropped),
        numeric=_pretty_list(plan.numeric),
        binary=_pretty_list(plan.binary),
        low_card=_pretty_list(plan.low_card),
        high_card=_pretty_list(plan.high_card),
        frequency_encoder=encoder_source,
        extra_imports=extra_imports,
        blocks=blocks,
        loader_block=loader_block,
        loader_call=loader_call,
    )
    compile(source, f"<exported:{exp_id}>", "exec")  # never hand the user broken Python
    return source


def _pretty_list(values: list[str]) -> str:
    if not values:
        return "[]"
    body = "".join(f"    {value!r},\n" for value in values)
    return f"[\n{body}]"


def _pretty_dict(mapping: dict[str, str]) -> str:
    if not mapping:
        return "{}"
    body = "".join(f"    {key!r}: {value!r},\n" for key, value in mapping.items())
    return f"{{\n{body}}}"


# --------------------------------------------------------------------------------------
# 9. Human-readable rendering
# --------------------------------------------------------------------------------------


def format_log(log: list[dict]) -> str:
    """Render the log as ``[step] STAGE explanation`` lines."""
    return "\n".join(
        f"[{entry['step']}] {entry['stage'].upper():<7} {entry['explanation']}"
        for entry in log
    )


def format_report(result: PreprocessingResult) -> str:
    """Render the whole story: what was dropped, what was done, and what came out."""
    plan = result.plan
    lines: list[str] = []
    lines.append(f"Target: {plan.target}   Task: {plan.task_type}")
    lines.append(
        f"Rows: {len(result.X_train)} train / {len(result.X_test)} test   "
        f"Features out: {result.X_train.shape[1]}"
    )
    lines.append("")

    if plan.dropped:
        lines.append(f"Columns removed ({len(plan.dropped)}):")
        for column, reason in plan.dropped.items():
            lines.append(f"  - {column}: {reason}")
    else:
        lines.append("Columns removed: none.")
    lines.append("")

    lines.append("Column groups:")
    for group in GROUP_ORDER:
        columns = getattr(plan, group)
        lines.append(f"  {group:<10} {', '.join(columns) if columns else '(none)'}")
    lines.append("")

    lines.append("What was done:")
    lines.append(format_log(result.log))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# 10. Command-line entry point — point it at a CSV and read the output
# --------------------------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="glassbox",
        description="Run the glass-box preprocessing pipeline over a CSV and print the log.",
    )
    parser.add_argument("csv", help="path to the dataset")
    parser.add_argument("target", help="name of the label column")
    parser.add_argument(
        "--task",
        default="classification",
        choices=["classification", "regression"],
        help="task type (default: classification)",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="COLUMN[:REASON]",
        help=(
            "Drop a column before modelling, with an optional reason shown to the user. "
            "Repeat per column. Needed for numeric identifiers such as a 1..N row id, "
            "which profile as numeric and would otherwise be scaled into a feature."
        ),
    )
    parser.add_argument("--json", action="store_true", help="print the log as JSON")
    parser.add_argument("--export", metavar="PATH", help="also write the exported script here")
    args = parser.parse_args(argv)

    excluded: dict[str, str] = {}
    for item in args.exclude:
        column, _, reason = item.partition(":")
        excluded[column] = reason or "excluded by the user"

    df = pd.read_csv(args.csv)
    result = fit_preprocessing(
        df,
        target=args.target,
        task_type=args.task,
        excluded=excluded,
        test_size=args.test_size,
        seed=args.seed,
    )

    if args.json:
        print(json.dumps(result.log, indent=2))
    else:
        print(format_report(result))
        print()
        print("Feature map (transformed -> source):")
        for out_name, source in result.feature_map.items():
            print(f"  {out_name} -> {source}")

    if args.export:
        source = export_script(result, csv=args.csv, exp_id="cli")
        with open(args.export, "w", encoding="utf-8") as handle:
            handle.write(source)
        print(f"\nExported runnable script to {args.export}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
