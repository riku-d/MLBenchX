"""Glass-box preprocessing engine for ML-BenchX.

This module turns a clean ``pandas.DataFrame`` into model-ready train/test matrices
*and* a plain-English record of every transformation that was applied. It sits between
dataset ingestion (upstream) and model training (downstream).

Four invariants define the module:

I1 — Split before anything is measured.
    The train/test split is the first thing that happens — before profiling, before
    screening, before any transformer exists. Every learned parameter (imputation
    medians, scaler means, encoder categories, frequency tables) and every *decision*
    (which columns are dropped, which encoder each survivor gets) is a function of the
    training partition alone. See :func:`fit_preprocessing`, where the ordering is
    explicit and commented.

    The stricter form matters: profiling the whole frame and only fitting on train is
    the usual compromise, and it still lets a category that appears exclusively in the
    held-out rows push a column over the one-hot threshold. Splitting first closes that.

I2 — The log is derived, never hand-written.
    :func:`introspect` reads the *fitted* sklearn objects and reports their learned
    attributes (``SimpleImputer.statistics_``, ``StandardScaler.mean_``/``.scale_``,
    ``OneHotEncoder.categories_``, ``FrequencyEncoder.frequencies_``,
    ``TargetEncoder.encodings_``). No number in the
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

Design note — two encoders for crowded categoricals, and why frequency is the default.
    :class:`FrequencyEncoder` replaces each value by how often it occurs in the training
    rows. It never reads ``y``, so it cannot leak under any circumstances, and that is
    why it is the default. ``sklearn.preprocessing.TargetEncoder`` is offered as the
    alternative: it does read ``y``, but it cross-fits internally, so the encoded value
    a training row receives is computed from folds that exclude that row. A naive mean
    encoding — the kind that does leak, and one of the most common sources of silently
    optimistic AutoML scores — is what sklearn's implementation exists to avoid. Both
    are selectable per run; see :data:`HIGH_CARD_ENCODERS`.

Design note — where dropped columns are recorded.
    Dropped columns are recorded in :attr:`PreprocessingPlan.dropped` (column -> human
    readable reason) rather than as entries in the introspection log. The log describes
    *fitted* transformers, and every entry it emits carries
    ``fitted_on == "train_fold_only"``; a drop is decided from the training fold's
    schema before any transformer exists, so putting it in the same list would mean
    labelling it with a fitting that never happened. :func:`format_report` prints the drops ahead of the steps so
    the user still sees one continuous narrative.

Out of scope by design: polynomial features, PCA, feature selection, class-imbalance
resampling, hyperparameter tuning, model training and SHAP. This module returns data
structures; persisting them is :mod:`app.services.storage`'s job, not this module's.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)
from sklearn.utils.validation import check_is_fitted

from app.services.checks import run_post_split_checks, sanity_baseline
from app.config import (
    DEFAULT_HIGH_CARD_ENCODER,
    DEFAULT_SCALER,
    MAX_COLUMNS_IN_EXPLANATION,
    MAX_MISSING_RATE,
    NEAR_CONSTANT_SHARE,
    ONEHOT_MAX_CARDINALITY,
    SEED,
    TARGET_ENCODER_CV,
    TEST_SIZE,
    TEXT_MIN_UNIQUE,
    TEXT_UNIQUE_RATE,
)

#: A screen that inspects the training fold and reports what should be removed and what
#: merely deserves a warning. Returns ``(column -> drop reason, flag records)``.
#: Screens are injected rather than imported so this module keeps no opinion about what
#: counts as suspicious; see :mod:`app.services.screening` for the implementations.
Screener = Callable[[pd.DataFrame, pd.Series], tuple[dict[str, str], list[dict[str, Any]]]]

__all__ = [
    "ONEHOT_MAX_CARDINALITY",
    "MAX_MISSING_RATE",
    "SCALERS",
    "HIGH_CARD_ENCODERS",
    "ColumnProfile",
    "Screener",
    "FrequencyEncoder",
    "DatetimeFeatures",
    "CategoricalCast",
    "StepPlan",
    "PreprocessingPlan",
    "PreprocessingResult",
    "profile_frame",
    "build_plan",
    "build_pipeline",
    "introspect",
    "build_feature_map",
    "fit_preprocessing",
    "choose_stratification",
    "export_script",
    "format_log",
    "format_report",
]


# --------------------------------------------------------------------------------------
# Thresholds all live in app.config. They are re-exported here under their historical
# names so the plan, the explanations and the tests keep quoting one number.
# --------------------------------------------------------------------------------------

#: Fixed order of the transformer blocks. ``ColumnTransformer`` emits output columns in
#: the order its transformers are listed, so the pipeline builder and the feature-map
#: builder must agree on this tuple.
GROUP_ORDER: tuple[str, ...] = ("numeric", "binary", "low_card", "high_card", "datetime")

_MAX_COLUMNS_IN_EXPLANATION = MAX_COLUMNS_IN_EXPLANATION

_STAGE_TO_STEP_NAME = {
    "impute": "impute", "encode": "encode", "scale": "scale", "derive": "derive",
}


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


def _top_share(series: pd.Series) -> float:
    """Share of the non-blank values taken by the most common one."""
    values = series.dropna()
    if values.empty:
        return 0.0
    return float(values.value_counts(normalize=True).iloc[0])


def _classify(series: pd.Series, n_unique: int, unique_rate: float) -> str:
    """Assign a ``kind`` to one column. Rules are applied in this order, first match wins."""
    if n_unique <= 1:
        return "constant"
    # Near-constant counts as constant. A column that is 99.5% one value carries almost
    # nothing, and treating it as an ordinary category spends a one-hot column — or a
    # whole encoder — on a distinction that occurs twice in a thousand rows.
    if _top_share(series) >= NEAR_CONSTANT_SHARE:
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


class CategoricalCast(TransformerMixin, BaseEstimator):
    """Present categorical columns to the encoders as plain objects.

    ``SimpleImputer`` refuses a ``bool`` column outright, and pandas' nullable extension
    dtypes (``boolean``, ``Int64``, ``string``) reach sklearn as masked arrays it does
    not expect. A file with a real ``True``/``False`` column would otherwise fail the
    whole run with a dtype error that says nothing about the user's data.

    Casting to ``object`` is lossless for the encoders downstream: every one of them —
    ordinal, one-hot, frequency, target — works from distinct values, and ``True`` as an
    object is exactly as distinct from ``False`` as it was before. Numeric columns never
    reach this transformer; they have their own block.

    It learns nothing, so it cannot leak. It is still inside the fitted pipeline so the
    exported script and any later ``transform`` apply the same cast.
    """

    def fit(self, X: Any, y: Any = None) -> "CategoricalCast":
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(list(frame.columns), dtype=object)
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(
            X, columns=list(self.feature_names_in_)
        )
        out = frame.copy()
        for column in out.columns:
            if not pd.api.types.is_numeric_dtype(out[column]) or pd.api.types.is_bool_dtype(
                out[column]
            ):
                # `where(notna)` keeps real blanks as NaN for the imputer to find,
                # rather than turning them into the string "nan" or "<NA>".
                out[column] = out[column].astype(object).where(out[column].notna(), np.nan)
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        names = list(self.feature_names_in_) if input_features is None else list(input_features)
        return np.asarray(names, dtype=object)


class DatetimeFeatures(TransformerMixin, BaseEstimator):
    """Split a timestamp into the parts of it a model can actually use.

    A datetime is a single enormous integer, and its magnitude means nothing: two orders
    placed a year apart in the same week of December are far apart as numbers and close
    together in every way that matters. This breaks each column into ``year``, ``month``,
    ``day``, ``dayofweek`` and ``hour``, then adds a sine/cosine pair for each of the
    three cyclical parts.

    The sine/cosine pair is the point. Month 12 and month 1 are adjacent in the calendar
    but eleven apart as numbers, and no model can discover otherwise from the integer
    alone. Projecting the month onto a circle puts December next to January by
    construction, which is what ``sin(2π·m/12)`` and ``cos(2π·m/12)`` together encode —
    both are needed, because either one alone maps two different months to the same
    value.

    This transformer learns nothing from the data: the output for a row is a function of
    that row only. It is still fitted on the training fold, and still reported in the
    log, because it changes the feature matrix and the user is owed that.

    Attributes set by ``fit``:
        feature_names_in_: input column names, in order.
        n_features_in_: number of input columns.
    """

    #: ``(part, period)`` — period ``None`` means the part is not cyclical, so it is
    #: emitted as a plain number with no sine/cosine pair.
    PARTS: tuple[tuple[str, int | None], ...] = (
        ("year", None),
        ("month", 12),
        ("day", 31),
        ("dayofweek", 7),
        ("hour", 24),
    )

    def _as_frame(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        array = np.asarray(X, dtype=object)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        return pd.DataFrame(array, columns=list(self.feature_names_in_))

    def fit(self, X: Any, y: Any = None) -> "DatetimeFeatures":
        frame = X if isinstance(X, pd.DataFrame) else self._as_frame(X)
        self.feature_names_in_ = np.asarray(list(frame.columns), dtype=object)
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        frame = self._as_frame(X)
        columns: dict[str, pd.Series] = {}

        for source in self.feature_names_in_:
            stamps = pd.to_datetime(frame[source], errors="coerce")
            for part, period in self.PARTS:
                values = getattr(stamps.dt, part).astype("float64")
                # A blank timestamp has no parts. Zero is the neutral value here: it is
                # what the sine/cosine pair of an unknown angle averages to anyway.
                columns[f"{source}_{part}"] = values.fillna(0.0)
                if period is None:
                    continue
                angle = 2 * math.pi * values.fillna(0.0) / period
                columns[f"{source}_{part}_sin"] = np.sin(angle)
                columns[f"{source}_{part}_cos"] = np.cos(angle)

        return pd.DataFrame(columns, index=frame.index)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        sources = list(self.feature_names_in_) if input_features is None else list(input_features)
        names: list[str] = []
        for source in sources:
            for part, period in self.PARTS:
                names.append(f"{source}_{part}")
                if period is not None:
                    names.extend([f"{source}_{part}_sin", f"{source}_{part}_cos"])
        return np.asarray(names, dtype=object)


# --------------------------------------------------------------------------------------
# 3. The plan — decisions, as data
# --------------------------------------------------------------------------------------


@dataclass
class StepPlan:
    """One planned transformation: what will run, on which columns, and why."""

    step_id: int
    stage: Literal["drop", "impute", "encode", "scale", "derive"]
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
    datetime: list[str] = field(default_factory=list)
    #: ``"standard"`` | ``"minmax"`` | ``"robust"``.
    scaler: str = DEFAULT_SCALER
    #: ``"frequency"`` | ``"target"``.
    high_card_encoder: str = DEFAULT_HIGH_CARD_ENCODER

    @property
    def feature_columns(self) -> list[str]:
        """Surviving input columns, in the order the pipeline consumes them."""
        return [*self.numeric, *self.binary, *self.low_card, *self.high_card, *self.datetime]


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
#: One entry per offered scaler: the sklearn class, and why someone would pick it.
SCALERS: dict[str, tuple[str, str]] = {
    "standard": (
        "StandardScaler",
        "Distance- and gradient-based models compare columns directly, so every number "
        "is put on a comparable range first. Each column is centred on its mean and "
        "divided by its spread.",
    ),
    "minmax": (
        "MinMaxScaler",
        "Every number is squeezed into the range 0 to 1, keeping the shape of the "
        "column exactly as it was. Useful when a model needs bounded inputs, but one "
        "extreme value compresses everything else against the opposite end.",
    ),
    "robust": (
        "RobustScaler",
        "Each column is centred on its median and divided by the distance between its "
        "quartiles, so a handful of extreme values cannot set the scale for the rest. "
        "The right choice when the column has genuine outliers worth keeping.",
    ),
}

#: One entry per offered high-cardinality encoder.
HIGH_CARD_ENCODERS: dict[str, tuple[str, str]] = {
    "frequency": (
        "FrequencyEncoder",
        f"More than {ONEHOT_MAX_CARDINALITY} distinct values, so one column per value "
        "would swamp the model. Frequency encoding costs one column and reads only the "
        "feature itself, never the target, so it cannot leak under any circumstances.",
    ),
    "target": (
        "TargetEncoder",
        f"More than {ONEHOT_MAX_CARDINALITY} distinct values, so one column per value "
        "would swamp the model. Each value is replaced by the average outcome for that "
        "value — computed with cross-fitting, so no row contributes to its own encoded "
        "value and the training rows do not get an unfairly good number.",
    ),
}

_REASON_DATETIME = (
    "A timestamp is useless to a model as a single number, but the parts of it are not: "
    "the month, the day of the week and the hour are where seasonal and weekly patterns "
    "live. Each is split out, and the cyclical ones are written as a sine/cosine pair so "
    "that December sits next to January rather than eleven months away."
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


def _encoder_spec(group: str, plan: PreprocessingPlan) -> tuple[str, dict[str, Any], str]:
    """The transformer name, parameters and reason for one categorical group."""
    if group == "binary":
        return (
            "OrdinalEncoder",
            {"handle_unknown": "use_encoded_value", "unknown_value": -1},
            _REASON_BINARY,
        )
    if group == "low_card":
        return "OneHotEncoder", {"handle_unknown": "ignore"}, _REASON_LOW_CARD

    name, reason = HIGH_CARD_ENCODERS[plan.high_card_encoder]
    params: dict[str, Any] = (
        {"unseen_value": 0.0}
        if plan.high_card_encoder == "frequency"
        else {"cv": TARGET_ENCODER_CV, "target_type": "auto", "random_state": SEED}
    )
    return name, params, reason


def build_plan(
    profiles: dict[str, ColumnProfile],
    target: str,
    task_type: str,
    excluded: dict[str, str] | None = None,
    scaler: str = DEFAULT_SCALER,
    high_card_encoder: str = DEFAULT_HIGH_CARD_ENCODER,
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
        excluded: column -> reason, supplied by the caller (a screen, or a column the
            user deselected in the UI). These reasons take precedence over the generic
            ones so the more specific message is what the user sees.
        scaler: which scaler numeric columns get — see :data:`SCALERS`.
        high_card_encoder: which encoder crowded categoricals get — see
            :data:`HIGH_CARD_ENCODERS`.

    Drop precedence, first match wins: target, caller exclusion, mostly-missing,
    constant, free text. Mostly-missing is checked before constant so that an all-blank
    column reports "100% of values are missing" rather than the less actionable
    "constant".
    """
    if scaler not in SCALERS:
        raise ValueError(f"scaler must be one of {sorted(SCALERS)}, got {scaler!r}")
    if high_card_encoder not in HIGH_CARD_ENCODERS:
        raise ValueError(
            f"high_card_encoder must be one of {sorted(HIGH_CARD_ENCODERS)}, "
            f"got {high_card_encoder!r}"
        )

    excluded = dict(excluded or {})
    plan = PreprocessingPlan(
        target=target,
        task_type=task_type,
        scaler=scaler,
        high_card_encoder=high_card_encoder,
    )

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
            plan.dropped[name] = (
                "constant — a single value carries no information"
                if profile.n_unique <= 1
                else (
                    f"near-constant — one value covers "
                    f"{NEAR_CONSTANT_SHARE:.0%} or more of the rows, so the column "
                    "barely varies"
                )
            )
            continue
        if profile.kind == "text":
            plan.dropped[name] = (
                f"free text / identifier — {profile.n_unique} distinct values in "
                f"{_pct(profile.unique_rate)} of rows"
            )
            continue

        if profile.kind == "datetime":
            plan.datetime.append(name)
        elif profile.kind == "numeric":
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
            scaler_name, scaler_reason = SCALERS[plan.scaler]
            plan.steps.append(
                StepPlan(step_id, "scale", columns, scaler_name, {}, scaler_reason)
            )
            step_id += 1
        elif group == "datetime":
            plan.steps.append(
                StepPlan(step_id, "derive", columns, "DatetimeFeatures",
                         {"parts": [part for part, _ in DatetimeFeatures.PARTS]},
                         _REASON_DATETIME)
            )
            step_id += 1
        else:
            plan.steps.append(
                StepPlan(step_id, "impute", columns, "SimpleImputer",
                         {"strategy": "most_frequent", "keep_empty_features": True},
                         _REASON_IMPUTE_MODE)
            )
            step_id += 1
            transformer, params, reason = _encoder_spec(group, plan)
            plan.steps.append(
                StepPlan(step_id, "encode", columns, transformer, dict(params), reason)
            )
            step_id += 1

    return plan


# --------------------------------------------------------------------------------------
# 4. Building the sklearn pipeline
# --------------------------------------------------------------------------------------


_SCALER_CLASSES = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def _block(group: str, plan: PreprocessingPlan) -> Pipeline:
    """The Pipeline for one column group.

    Inner step names (``impute``/``encode``/``scale``/``derive``) are part of the
    contract: :func:`introspect` and :func:`build_feature_map` look transformers up by
    them.
    """
    if group == "numeric":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", _SCALER_CLASSES[plan.scaler]()),
        ])

    if group == "datetime":
        # No imputer: a blank timestamp has no parts, and DatetimeFeatures already
        # emits zeros for one. Filling it with the most common date first would invent
        # a specific day that the calendar features would then treat as real.
        return Pipeline([("derive", DatetimeFeatures())])

    imputer = SimpleImputer(strategy="most_frequent", keep_empty_features=True)
    if group == "binary":
        encoder: Any = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    elif group == "low_card":
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    elif group == "high_card":
        encoder = (
            FrequencyEncoder()
            if plan.high_card_encoder == "frequency"
            else TargetEncoder(cv=TARGET_ENCODER_CV, random_state=SEED)
        )
    else:  # pragma: no cover - guarded by GROUP_ORDER
        raise ValueError(f"unknown column group: {group!r}")
    return Pipeline([("cast", CategoricalCast()), ("impute", imputer), ("encode", encoder)])


def build_pipeline(plan: PreprocessingPlan) -> ColumnTransformer:
    """Turn a plan into a fitted-ready ``ColumnTransformer``.

    One named block per non-empty group, named exactly after the group.
    ``set_output(transform="pandas")`` is essential: without it the transformed frame
    loses its column names and the feature map degrades into string-parsing guesswork.
    """
    transformers = [
        (group, _block(group, plan), list(getattr(plan, group)))
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
    """Log entry for a scaling step, read from whichever attributes it fitted.

    All three scalers reduce to the same arithmetic — subtract a centre, divide by a
    spread — but each learns those two numbers differently and stores them under
    different names. The centre and spread are recovered here under one pair of keys so
    the log, the UI card and the exported script describe every scaler the same way.
    """
    columns = list(scaler.feature_names_in_)
    kind = type(scaler).__name__

    if kind == "StandardScaler":
        centres, spreads = list(scaler.mean_), list(scaler.scale_)
        summary = (
            "Rescaled each number to mean 0, spread 1, so a column measured in "
            "thousands does not outweigh one measured in single digits."
        )
        centre_word, spread_word = "mean", "standard deviation"
    elif kind == "MinMaxScaler":
        centres = list(scaler.data_min_)
        spreads = [float(s) if s else 1.0 for s in (scaler.data_max_ - scaler.data_min_)]
        summary = (
            "Squeezed each column into the range 0 to 1, so every column contributes "
            "over the same interval regardless of the units it was measured in."
        )
        centre_word, spread_word = "smallest training value", "training range"
    else:  # RobustScaler
        centres, spreads = list(scaler.center_), list(scaler.scale_)
        summary = (
            "Centred each column on its median and divided by the spread of its middle "
            "half, so a handful of extreme values cannot set the scale for the rest."
        )
        centre_word, spread_word = "median", "interquartile range"

    centre_map = {c: _jsonable(v) for c, v in zip(columns, centres)}
    spread_map = {c: _jsonable(v) for c, v in zip(columns, spreads)}
    clauses = [
        f"`{c}` — shifted by {_fmt_number(centres[i])}, then divided by "
        f"{_fmt_number(spreads[i])}"
        for i, c in enumerate(columns)
    ]
    explanation = (
        f"{summary} The {centre_word} and {spread_word} come from the training rows "
        f"only: {_join_clauses(clauses)}."
    )
    return {
        "details": {"scaler": kind, "mean": centre_map, "scale": spread_map},
        "explanation": explanation,
    }


def _target_entry(encoder: Any) -> dict[str, Any]:
    """Log entry for target encoding, read from ``encoder.encodings_``."""
    columns = list(encoder.feature_names_in_)
    encodings: dict[str, Any] = {}
    counts: dict[str, int] = {}
    for column, categories, values in zip(columns, encoder.categories_, encoder.encodings_):
        table = {str(_jsonable(cat)): _jsonable(val) for cat, val in zip(categories, np.ravel(values))}
        encodings[column] = table
        counts[column] = len(table)

    clauses = [f"`{c}` — {counts[c]} distinct values, now 1 column" for c in columns]
    explanation = (
        "Too many categories to one-hot, so each value was replaced by the average "
        f"outcome recorded for it. The overall training average is "
        f"{_fmt_number(np.ravel(encoder.target_mean_)[0])}, and a value the model has "
        "not seen falls back to it. The encoded numbers the model trains on are "
        f"cross-fitted across {encoder.cv} folds, so no row helped compute its own "
        f"value: {_join_clauses(clauses)}."
    )
    return {
        "details": {
            "encodings": encodings,
            "n_categories": counts,
            "target_mean": _jsonable(np.ravel(encoder.target_mean_)[0]),
            "cv": int(encoder.cv),
        },
        "explanation": explanation,
    }


def _derive_entry(step: StepPlan, transformer: Any) -> dict[str, Any]:
    """Log entry for datetime decomposition, read from the fitted transformer's outputs."""
    columns = list(transformer.feature_names_in_)
    produced = [str(name) for name in transformer.get_feature_names_out()]
    per_column = {
        c: [name for name in produced if name.startswith(f"{c}_")] for c in columns
    }
    cyclical = [part for part, period in transformer.PARTS if period is not None]

    clauses = [f"`{c}` -> {len(per_column[c])} columns" for c in columns]
    explanation = (
        f"Split {len(columns)} date/time column(s) into {len(produced)} numbers a model "
        f"can use: {', '.join(part for part, _ in transformer.PARTS)}. "
        f"{', '.join(cyclical).capitalize()} are each also written as a sine/cosine "
        "pair, which is what puts December next to January instead of eleven months "
        f"away: {_join_clauses(clauses)}."
    )
    return {
        "details": {
            "parts": [part for part, _ in transformer.PARTS],
            "cyclical_parts": cyclical,
            "columns_created": {c: len(per_column[c]) for c in columns},
            "feature_names": per_column,
        },
        "explanation": explanation,
    }


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


#: Short machine-readable verb for each step, keyed on ``(stage, transformer)``. The
#: record's ``reason`` is the sentence a person reads; this is what code switches on, so
#: the two never have to be parsed out of each other.
_ACTIONS: dict[tuple[str, str], str] = {
    ("impute", "SimpleImputer"): "impute",
    ("scale", "StandardScaler"): "scale_standard",
    ("scale", "MinMaxScaler"): "scale_minmax",
    ("scale", "RobustScaler"): "scale_robust",
    ("encode", "OrdinalEncoder"): "encode_label",
    ("encode", "OneHotEncoder"): "encode_onehot",
    ("encode", "FrequencyEncoder"): "encode_frequency",
    ("encode", "TargetEncoder"): "encode_target",
    ("derive", "DatetimeFeatures"): "derive_datetime",
}


def _before_after(
    step: StepPlan, fitted: Any, train_raw: pd.DataFrame
) -> tuple[dict[str, Any], dict[str, Any]]:
    """What the data looked like going into this step, and coming out of it.

    Both sides are measured, not asserted: the input columns come from the training
    frame as it stood, and the output names come from the fitted transformer itself.
    Every step therefore states its own effect on the shape of the data, which is the
    part a reader can check against the table in front of them.
    """
    columns = list(step.columns)
    present = [c for c in columns if c in train_raw.columns]
    blanks = int(train_raw[present].isna().sum().sum()) if present else 0

    try:
        out_names = [str(name) for name in fitted.get_feature_names_out(columns)]
    except Exception:  # noqa: BLE001 - not every transformer names its output
        out_names = list(columns)

    before: dict[str, Any] = {
        "n_columns": len(columns),
        "columns": columns,
        "missing_values": blanks,
    }
    after: dict[str, Any] = {
        "n_columns": len(out_names),
        "columns": out_names,
        "missing_values": 0 if step.stage == "impute" else blanks,
    }

    if step.stage == "impute":
        before["dtype"] = "as uploaded"
        after["dtype"] = "as uploaded, no blanks"
    elif step.stage in ("encode", "derive"):
        before["dtype"] = "text or timestamp"
        after["dtype"] = "numeric"
    else:
        before["dtype"] = "numeric, original units"
        after["dtype"] = "numeric, rescaled"

    return before, after


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
        elif step.stage == "derive":
            body = _derive_entry(step, fitted)
        elif step.transformer == "OneHotEncoder":
            body = _onehot_entry(fitted)
        elif step.transformer == "OrdinalEncoder":
            body = _ordinal_entry(fitted)
        elif step.transformer == "FrequencyEncoder":
            body = _frequency_entry(fitted)
        elif step.transformer == "TargetEncoder":
            body = _target_entry(fitted)
        else:  # pragma: no cover - unreachable while the plan is the only source
            raise ValueError(f"no explanation template for {step.transformer!r}")

        before, after = _before_after(step, fitted, train_raw)
        log.append({
            "step": position,
            "stage": step.stage,
            "action": _ACTIONS[(step.stage, step.transformer)],
            "columns": list(step.columns),
            "column": ", ".join(step.columns),
            "transformer": step.transformer,
            "params": _jsonable(step.params),
            "fitted_on": "train_fold_only",
            "reason": step.reason,
            "before": before,
            "after": after,
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
        elif group == "datetime":
            # Many output columns per input column, each named "{source}_{part}".
            derive = block.named_steps["derive"]
            for source in columns:
                for out_name in derive.get_feature_names_out([source]):
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
    #: Every column the caller or a screener removed, mapped to the reason shown to the
    #: user. Screener entries are produced from the training fold only, so this is the
    #: authoritative record of what was set aside and why.
    excluded: dict[str, str] = field(default_factory=dict)
    #: Records from the screens — what was flagged, why, and whether it was excluded.
    flags: list[dict[str, Any]] = field(default_factory=list)
    #: Conditions the user needs to know about that are not tied to one column, such as
    #: a split that could not be stratified. Shaped like the spec's §2 warnings:
    #: ``{severity, message, suggestion}``.
    warnings: list[dict[str, Any]] = field(default_factory=list)
    #: Post-split checks (§4d): per-feature drift and class balance in both folds.
    checks: dict[str, Any] = field(default_factory=dict)
    #: The split parameters actually used. Recorded rather than recovered: deriving
    #: test_size back from the row counts rounds, and a rounded fraction re-enters
    #: `train_test_split` as a different ceiling — an 891-row file exported at 0.2009
    #: instead of 0.2 splits 711/180 where the app split 712/179. One row is enough to
    #: make "this script reproduces exactly what you were shown" false.
    test_size: float = TEST_SIZE
    seed: int = SEED
    #: Whether the split kept each class's share the same on both sides. Recorded
    #: rather than inferred: "no stratification" and "stratification that happened to
    #: change nothing" look identical in the resulting folds, and only one of them is
    #: something the user needs told.
    stratified: bool = False


def _blank_non_finite(frame: pd.DataFrame) -> int:
    """Replace ``inf``/``-inf`` with ``NaN`` in place. Returns how many were replaced.

    Mutates the frame it is given, which is safe here because the caller owns it: it
    is the feature frame produced by ``df.drop(columns=[target])``, a copy that never
    leaves this function. The user's frame is not touched.
    """
    replaced = 0
    for column in frame.columns:
        series = frame[column]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        mask = np.isinf(series.to_numpy(dtype="float64", na_value=np.nan))
        count = int(mask.sum())
        if count:
            frame.loc[mask, column] = np.nan
            replaced += count
    return replaced


def choose_stratification(
    y: pd.Series, classification: bool, test_size: float
) -> tuple[pd.Series | None, dict[str, Any] | None]:
    """Decide whether the split can be stratified, and say why when it cannot.

    Stratifying keeps each class's share the same on both sides of the split, which is
    the only way a rare class is represented in the held-out rows at all. Three things
    stop it, and each one is reported rather than absorbed: a target with one class,
    a class with a single row, and a test fold smaller than the number of classes.

    That third condition is the one that is easy to miss. ``train_test_split`` has to
    place at least one row of every class on each side, so it refuses outright when
    either fold has fewer rows than there are classes — a 60-row file with 20 classes
    holds out 12 rows and cannot cover them. Left unchecked that surfaces as a
    ``ValueError`` from deep inside sklearn, phrased in terms of ``test_size`` and
    counts the user never chose. Checking it here turns an aborted run into a split
    that happens, with a warning that names the real problem.

    The fold sizes are computed the way ``sklearn.model_selection._validate_shuffle_split``
    computes them, so this predicts that function's verdict rather than approximating it.

    Returns ``(stratify_argument, warning_or_None)``.
    """
    if not classification:
        return None, None

    counts = y.value_counts(dropna=False)
    n_classes = int(len(counts))

    if n_classes <= 1:
        return None, {
            "kind": "stratification_disabled",
            "severity": "warn",
            "message": "The split was not stratified: the target has only one class.",
            "suggestion": "A single-class target cannot be predicted; pick another column.",
            "classes": {},
        }

    singletons = counts[counts < 2]
    if len(singletons):
        # Falling back silently would leave the user believing their rare class is
        # represented on both sides of the split when it cannot be. The whole point of
        # stratifying is the rare class, so losing it is exactly what needs saying.
        named = ", ".join(str(v) for v in singletons.index[:5])
        return None, {
            "kind": "stratification_disabled",
            "severity": "warn",
            "message": (
                f"{len(singletons)} class(es) have fewer than 2 rows ({named}), so "
                "the split could not be stratified."
            ),
            "suggestion": (
                "The train and test folds may not contain the same classes. Collect "
                "more examples of the rare class, or merge it into a neighbouring "
                "one, before trusting any score."
            ),
            "classes": {str(k): int(v) for k, v in singletons.items()},
        }

    n_samples = int(len(y))
    n_test = math.ceil(test_size * n_samples)
    n_train = n_samples - n_test
    if min(n_test, n_train) < n_classes:
        smaller = "held-out" if n_test <= n_train else "training"
        return None, {
            "kind": "stratification_disabled",
            "severity": "warn",
            "message": (
                f"{n_classes} classes across {n_samples:,} rows leave a {smaller} fold "
                f"of {min(n_test, n_train):,} rows, too few to hold one of every class, "
                "so the split could not be stratified."
            ),
            "suggestion": (
                "Some classes will be missing from one side of the split. Collect more "
                "rows, or group the rarer classes together, before trusting any score."
            ),
            "classes": {str(k): int(v) for k, v in counts.items()},
        }

    return y, None


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
    test_size: float = TEST_SIZE,
    seed: int = SEED,
    screeners: Sequence[Screener] = (),
    scaler: str = DEFAULT_SCALER,
    high_card_encoder: str = DEFAULT_HIGH_CARD_ENCODER,
) -> PreprocessingResult:
    """Split, screen, plan, fit, and explain — in that order.

    The ordering below *is* invariant I1, and it is stricter than "fit on train only".
    The split happens at step 2, before anything at all is measured: the column profile
    that decides which columns are dropped and how each survivor is encoded is computed
    from ``X_train`` alone, so a value that only ever appears in the test fold cannot
    change the shape of the pipeline. Every later step inherits that guarantee.

    Args:
        df: clean input frame, features and target together.
        target: label column, present in ``df``.
        task_type: ``"classification"`` or ``"regression"``; classification splits are
            stratified on the label.
        excluded: column -> reason for columns the caller wants removed outright, such as
            one the user deselected in the UI. Caller exclusions are a stated preference
            rather than a measurement, so they are honoured as given.
        test_size: fraction held out.
        seed: the only source of randomness in the module.
        screeners: callables run on the *training fold only*, each returning
            ``(to_drop, flags)``. This is how measurement-based screens — identifier
            detection, target-leakage correlation — get their verdict without reading a
            single held-out row. See :mod:`app.services.screening`.
        scaler: ``"standard"`` | ``"minmax"`` | ``"robust"``; see :data:`SCALERS`.
        high_card_encoder: ``"frequency"`` | ``"target"``; see
            :data:`HIGH_CARD_ENCODERS`.

    Returns:
        :class:`PreprocessingResult`. ``X_train``/``X_test`` are transformed frames with
        identical columns and no missing values.
    """
    if target not in df.columns:
        raise ValueError(f"target column {target!r} is not in the frame")
    classification = _is_classification(task_type)
    warnings: list[dict[str, Any]] = []

    # 1. Stateless repairs that must precede the split (spec §4a).
    #
    #    A row with no value in the target teaches nothing — there is no answer in it
    #    to learn — and carrying one forward fails in two different ways depending on
    #    the task. Stratification reads the label, so on a classification target a
    #    blank arrives at ``train_test_split`` as a class of its own and stops the run
    #    with a message about NaN that says nothing about the user's file. On a
    #    regression target nothing raises at all: the blank travels into ``y_train``
    #    and quietly corrupts whatever is fitted on it. Dropping the rows here makes
    #    both paths behave the same way, and says so.
    blank_targets = int(df[target].isna().sum())
    if blank_targets:
        df = df[df[target].notna()].reset_index(drop=True)
        warnings.append({
            "kind": "blank_targets_removed",
            "severity": "info",
            "message": (
                f"Left out {blank_targets:,} row(s) with no value in {target}."
            ),
            "suggestion": (
                "A row with no answer recorded cannot teach the model anything, and "
                "cannot be scored against either."
            ),
            "count": blank_targets,
        })

    #    Deduplication belongs here and nowhere else. A repeated row left in place is
    #    counted twice when the model fits, and — worse — its copies can land on
    #    opposite sides of the split, so the model is scored on rows it has already
    #    trained on. Removing them after the split would not undo that.
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        df = df.drop_duplicates().reset_index(drop=True)
        warnings.append({
            "kind": "duplicates_removed",
            "severity": "info",
            "message": f"Removed {duplicate_rows:,} duplicate row(s) before splitting.",
            "suggestion": (
                "Copies of one row can otherwise land on both sides of the split, which "
                "makes any score on the held-out rows optimistic."
            ),
            "count": duplicate_rows,
        })

    if df.empty:
        raise ValueError(
            f"No rows are left once rows with no value in {target!r} are set aside, so "
            "there is nothing to learn from."
        )

    # 2. Separate the label from the features.
    y = df[target]
    X = df.drop(columns=[target])

    #    Infinity is an overflow rather than a measurement, and every estimator below
    #    rejects it outright. Read as a blank, it is filled from the training rows like
    #    any other gap — which is what "the real value is unknown" means. Ingestion
    #    already does this for an uploaded file; it is repeated here because this
    #    function is called with frames that did not come through it.
    non_finite = _blank_non_finite(X)
    if non_finite:
        warnings.append({
            "kind": "non_finite_values",
            "severity": "warn",
            "message": (
                f"Treated {non_finite:,} infinite value(s) as missing before splitting."
            ),
            "suggestion": (
                "Infinity usually means an overflow or a division by zero upstream. "
                "Check how the column was calculated — the fill value used in its place "
                "is in the imputation step below."
            ),
            "count": non_finite,
        })

    # 3. Split FIRST — before any column is profiled, screened or fitted. Stratify so a
    #    rare class is represented in both folds; fall back when the label cannot
    #    support it, and say why rather than falling back silently.
    stratify, stratify_warning = choose_stratification(y, classification, test_size)
    if stratify_warning is not None:
        warnings.append(stratify_warning)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, shuffle=True, stratify=stratify
    )

    # 4. Screen and profile the TRAINING fold only. Both are measurements, so both
    #    belong on this side of the split: uniqueness, missing rate and cardinality all
    #    decide which columns survive and how they are encoded.
    removals = dict(excluded or {})
    flags: list[dict[str, Any]] = []
    for screener in screeners:
        screened, screen_flags = screener(X_train_raw, y_train)
        for column, reason in screened.items():
            removals.setdefault(column, reason)
        flags.extend(screen_flags)

    plan = build_plan(
        profile_frame(X_train_raw),
        target=target,
        task_type=task_type,
        excluded=removals,
        scaler=scaler,
        high_card_encoder=high_card_encoder,
    )

    # 5. Remove the columns the plan rejected, from both folds identically.
    to_drop = [c for c in plan.dropped if c in X_train_raw.columns]
    X_train_raw = X_train_raw.drop(columns=to_drop)
    X_test_raw = X_test_raw.drop(columns=to_drop)
    if X_train_raw.shape[1] == 0:
        raise ValueError(
            "Every column was dropped, so there is nothing to train on. Reasons: "
            + json.dumps(plan.dropped, indent=2)
        )

    # 6. Fit on the training partition only, then apply the frozen pipeline to the test
    #    partition. This is the one and only `.fit` in the module's data path.
    pipeline = build_pipeline(plan)
    X_train_out = pipeline.fit_transform(X_train_raw, y_train)
    X_test_out = pipeline.transform(X_test_raw)

    # 7. Check the split itself (spec §4d). These measure the division of the data
    #    rather than the data, and a pipeline can be correct in every step and still
    #    produce folds that do not resemble each other.
    checks = run_post_split_checks(X_train_out, X_test_out, y_train, y_test)
    checks["baseline"] = sanity_baseline(X_train_out, y_train, task_type, seed)
    warnings.extend(checks["warnings"])
    if checks["baseline"] and checks["baseline"]["severity"] == "warn":
        checks["baseline"]["kind"] = "weak_baseline"
        warnings.append({
            "kind": "weak_baseline",
            "severity": "warn",
            "message": checks["baseline"]["verdict"],
            "suggestion": (
                "Check the plan above: a step may be removing the signal, or the target "
                "may not be predictable from these columns."
            ),
        })

    # 8. Explain what was learned, and record where each output column came from.
    log = introspect(pipeline, plan, X_train_raw)
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
        excluded=removals,
        flags=flags,
        warnings=warnings,
        checks=checks,
        test_size=test_size,
        seed=seed,
        stratified=stratify is not None,
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
from sklearn.preprocessing import (
    MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler, TargetEncoder,
)
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
DATETIME = {datetime}
{loader_block}{custom_transformers}

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
    # --- (medians, centres, spreads, categories, frequencies) comes from these rows
    # --- alone. y_train is passed because TargetEncoder needs it; it cross-fits
    # --- internally, so no row contributes to its own encoded value.
    X_train_out = pipeline.fit_transform(X_train, y_train)
    X_test_out = pipeline.transform(X_test)

    print("train:", X_train_out.shape, " test:", X_test_out.shape)
    print("features:", list(X_train_out.columns))


if __name__ == "__main__":
    main()
'''

_SCALER_SOURCE = {
    "standard": "StandardScaler()",
    "minmax": "MinMaxScaler()",
    "robust": "RobustScaler()",
}


def _block_source(plan: PreprocessingPlan) -> str:
    """Python source for every pipeline block this plan uses, in GROUP_ORDER."""
    high_card_encoder = (
        "FrequencyEncoder()"
        if plan.high_card_encoder == "frequency"
        else f"TargetEncoder(cv={TARGET_ENCODER_CV}, random_state={SEED})"
    )
    sources = {
        "numeric": (
            '    blocks.append(("numeric", Pipeline([\n'
            '        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),\n'
            f'        ("scale", {_SCALER_SOURCE[plan.scaler]}),\n'
            '    ]), NUMERIC))\n'
        ),
        "binary": (
            '    blocks.append(("binary", Pipeline([\n'
            '        ("cast", CategoricalCast()),\n'
            '        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),\n'
            '        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),\n'
            '    ]), BINARY))\n'
        ),
        "low_card": (
            '    blocks.append(("low_card", Pipeline([\n'
            '        ("cast", CategoricalCast()),\n'
            '        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),\n'
            '        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),\n'
            '    ]), LOW_CARD))\n'
        ),
        "high_card": (
            '    blocks.append(("high_card", Pipeline([\n'
            '        ("cast", CategoricalCast()),\n'
            '        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),\n'
            f'        ("encode", {high_card_encoder}),\n'
            '    ]), HIGH_CARD))\n'
        ),
        "datetime": (
            '    blocks.append(("datetime", Pipeline([\n'
            '        ("derive", DatetimeFeatures()),\n'
            '    ]), DATETIME))\n'
        ),
    }
    return "".join(sources[group] for group in GROUP_ORDER if getattr(plan, group))


def export_script(
    result: PreprocessingResult,
    csv: str,
    exp_id: str,
    seed: int = SEED,
    loader_block: str = "",
    loader_call: str = "pd.read_csv(CSV_PATH)",
) -> str:
    """Emit standalone Python that reproduces this preprocessing exactly.

    The generated file carries the same column lists, transformer parameters and seed,
    splits before it fits, and marks the fit-on-train-only line with a comment. Any
    transformer this app defines itself — :class:`FrequencyEncoder`,
    :class:`DatetimeFeatures` — has its source embedded via :mod:`inspect` rather than
    copied into a string, so the export cannot drift from the real implementation.

    The script drops its columns before splitting while the app splits first. The two
    still produce identical folds: ``train_test_split`` chooses rows from the row count,
    the seed and the test size, none of which the column list affects.

    Args:
        result: the outcome of :func:`fit_preprocessing`.
        csv: path the generated script should read the dataset from.
        exp_id: experiment identifier, recorded in the file's docstring.
        seed: the seed to write into the script. Defaults to the module's, but
            :attr:`PreprocessingResult.seed` records what the run actually used and is
            what callers should pass.
        loader_block: optional Python source defining how to read the file — separator,
            encoding, and any repairs applied before preprocessing. Without it the
            script assumes a plain comma-separated file, which will not reproduce the
            cleaned data for an upload that needed repairing.
        loader_call: the expression the script uses to obtain the frame. Must pair with
            ``loader_block``.

    Returns the source text; writing it to disk is the caller's job.
    """
    plan = result.plan

    # Transformers this app defines rather than imports are embedded via `inspect`,
    # never copied into a string literal, so the export cannot drift from the code that
    # actually ran.
    embedded: list[Any] = []
    if plan.binary or plan.low_card or plan.high_card:
        embedded.append(CategoricalCast)
    if plan.high_card and plan.high_card_encoder == "frequency":
        embedded.append(FrequencyEncoder)
    if plan.datetime:
        embedded.append(DatetimeFeatures)

    if embedded:
        custom_transformers = "".join(
            "\n\n" + textwrap.dedent(inspect.getsource(cls)).rstrip() + "\n"
            for cls in embedded
        )
        extra_imports = (
            "import math\n"
            "from typing import Any\n"
            "from sklearn.base import BaseEstimator, TransformerMixin\n"
            "from sklearn.utils.validation import check_is_fitted\n"
        )
    else:
        custom_transformers = ""
        extra_imports = ""

    blocks = _block_source(plan)

    source = _SCRIPT_TEMPLATE.format(
        exp_id=exp_id,
        csv=csv,
        target=plan.target,
        task_type="classification" if _is_classification(plan.task_type) else "regression",
        test_size=result.test_size,
        seed=seed,
        dropped=_pretty_dict(plan.dropped),
        numeric=_pretty_list(plan.numeric),
        binary=_pretty_list(plan.binary),
        low_card=_pretty_list(plan.low_card),
        high_card=_pretty_list(plan.high_card),
        datetime=_pretty_list(plan.datetime),
        custom_transformers=custom_transformers,
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
