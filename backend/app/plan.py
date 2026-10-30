"""The preprocessing plan: one decision per column, with the reason for it.

This is the object the spec's §4 is built around, and the difference between it and a
set of global options is the whole product. "Impute numeric columns with the median" is
a policy; "``Age`` → fill 184 gaps with the median, because 20% is missing and the
column is skewed" is a decision about a particular column that a person can read,
disagree with, and change — one column at a time, without touching the other forty.

Three properties make that work:

**Every column appears.** Including the ones nothing happens to, and the ones that are
dropped. A plan that lists only the interesting columns leaves the reader unable to tell
"nothing needed doing" from "we forgot about it", and §5's invariant — every input column
has a stated fate — is enforceable only because the plan is exhaustive.

**Every action carries its alternatives.** A dropdown with no options is a label. The
planner records what else it could reasonably have chosen for that column, so the
interface can offer exactly those and nothing nonsensical — no "one-hot encode" on a
float, no "take the median" on a string.

**Data-dependent decisions are made on the training fold only.** :func:`propose` splits
before it measures, for the same reason :func:`~app.services.glassbox.fit_preprocessing`
does: a missing rate or a cardinality computed over rows the model will later be tested
on is a decision informed by held-out data. The plan therefore carries the seed and
test size it was proposed under, and executing it re-creates exactly that split.

A plan is data, not behaviour. It serialises to JSON, round-trips, and is stored with
the run that used it — so a run can be explained months later without re-deriving
anything.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.config import (
    DEFAULT_SCALER,
    MAX_MISSING_RATE,
    MISSING_INDICATOR_RATE,
    NEAR_CONSTANT_SHARE,
    ONEHOT_MAX_CARDINALITY,
    RARE_CATEGORY_RATE,
    SEED,
    SKEW_THRESHOLD,
    TEST_SIZE,
)

#: A screen that inspects the training fold and reports what should be removed and
#: what merely deserves a warning. Same shape as the engine's, so the two share
#: implementations rather than each having its own.
Screener = Callable[[pd.DataFrame, pd.Series], tuple[dict[str, str], list[dict[str, Any]]]]

__all__ = [
    "Action",
    "Screener",
    "prepare",
    "ColumnPlan",
    "GlobalOptions",
    "PreprocessingPlan",
    "propose",
    "ACTIONS_FOR_KIND",
]

#: Every action a column can be assigned, and what it means to a reader.
#:
#: The vocabulary is deliberately small. Each entry is something a person can picture
#: happening to a column; there is no "transform" or "process" that would need a second
#: field to explain it.
Action = Literal[
    "keep",                 # numeric, nothing needed
    "drop",                 # removed, with a reason
    "impute_median",
    "impute_mean",
    "impute_mode",
    "impute_constant",
    "encode_binary",        # two values -> 0/1
    "encode_onehot",
    "encode_ordinal",
    "encode_target",        # cross-fitted
    "encode_frequency",
    "datetime_expand",      # -> year/month/dow/hour + sin/cos
    "passthrough",          # the target, and anything explicitly untouched
]

#: What may reasonably be chosen for a column of each kind. The interface offers these
#: and nothing else, so a user cannot ask for the median of a string column.
ACTIONS_FOR_KIND: dict[str, list[str]] = {
    "numeric": ["keep", "impute_median", "impute_mean", "impute_constant", "drop"],
    "categorical": [
        "encode_onehot", "encode_ordinal", "encode_target", "encode_frequency",
        "encode_binary", "impute_mode", "drop",
    ],
    "boolean": ["encode_binary", "impute_mode", "drop"],
    "datetime": ["datetime_expand", "drop"],
    "text": ["drop", "encode_frequency"],
    "constant": ["drop"],
}


class ColumnPlan(BaseModel):
    """What happens to one column, why, and what else could have happened.

    ``reason`` is written for someone who does not know what an imputer is, and quotes
    the numbers that drove the decision — a reason without them ("this column has
    missing values") is a restatement of the action, not a justification for it.
    """

    model_config = ConfigDict(extra="forbid")

    column: str
    #: The name as the user wrote it, when sanitisation changed it.
    original_name: str | None = None
    kind: str
    dtype: str

    action: Action
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str

    #: Facts the decision was made from, so the reason can be checked against them.
    stats: dict[str, Any] = Field(default_factory=dict)

    #: Other actions that make sense for this column.
    alternatives: list[str] = Field(default_factory=list)

    #: True once a user has changed this column's action away from the proposal.
    overridden: bool = False

    #: Set when the planner wants to say something beyond the action itself — a skew
    #: worth transforming, a leak, an identifier.
    warning: dict[str, Any] | None = None

    #: Whether a missing-indicator column is added alongside this one.
    add_indicator: bool = False
    #: Categories below the rare threshold, collapsed to "Other".
    rare_categories: list[str] = Field(default_factory=list)


class GlobalOptions(BaseModel):
    """Choices that apply to the pipeline rather than to one column."""

    model_config = ConfigDict(extra="forbid")

    scaler: Literal["standard", "minmax", "robust", "none"] = DEFAULT_SCALER  # type: ignore[assignment]
    #: ``report`` measures outliers and changes nothing; ``clip`` bounds them using
    #: limits taken from the training fold. Report is the default per §4c: an outlier is
    #: not automatically an error.
    outliers: Literal["report", "clip"] = "report"
    missing_indicators: bool = True
    rare_category_rate: float = Field(default=RARE_CATEGORY_RATE, ge=0.0, lt=0.5)
    onehot_max_cardinality: int = Field(default=ONEHOT_MAX_CARDINALITY, ge=2, le=200)
    #: Applies Yeo-Johnson to columns the planner flagged as skewed. Off by default:
    #: it makes a column harder to reason about, and the flag alone is often enough.
    power_transform_skewed: bool = False
    test_size: float = Field(default=TEST_SIZE, gt=0.05, lt=0.5)
    seed: int = SEED


class PreprocessingPlan(BaseModel):
    """Every decision for one dataset, ready to be shown, edited, and executed."""

    model_config = ConfigDict(extra="forbid")

    target: str
    task: str
    columns: list[ColumnPlan]
    options: GlobalOptions = Field(default_factory=GlobalOptions)
    #: Conditions found while planning: leakage, near-constants, a target with blanks.
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    #: Rows dropped before the split, and why.
    prepared: dict[str, Any] = Field(default_factory=dict)

    def for_column(self, name: str) -> ColumnPlan | None:
        return next((c for c in self.columns if c.column == name), None)

    def kept(self) -> list[ColumnPlan]:
        return [c for c in self.columns if c.action not in ("drop", "passthrough")]

    def dropped(self) -> list[ColumnPlan]:
        return [c for c in self.columns if c.action == "drop"]

    def override(self, column: str, action: str, params: dict[str, Any] | None = None) -> None:
        """Change one column's action, marking it as the user's decision.

        Refuses an action that makes no sense for the column's kind. The alternatives
        were computed for a reason, and silently accepting "take the median" on a string
        column would fail at execution with a message about dtypes rather than here with
        one about the choice.
        """
        plan = self.for_column(column)
        if plan is None:
            raise KeyError(f"no column called {column!r} in this plan")
        allowed = ACTIONS_FOR_KIND.get(plan.kind, [])
        if action not in allowed:
            raise ValueError(
                f"{action!r} is not a valid action for {column!r}, which holds "
                f"{plan.kind} values. Choose one of: {', '.join(allowed)}."
            )
        plan.action = action  # type: ignore[assignment]
        plan.params = params or {}
        plan.overridden = True
        plan.reason = f"You chose {_ACTION_LABELS.get(action, action)} for this column."


_ACTION_LABELS: dict[str, str] = {
    "keep": "no change",
    "drop": "removing it",
    "impute_median": "filling gaps with the median",
    "impute_mean": "filling gaps with the mean",
    "impute_mode": "filling gaps with the most common value",
    "impute_constant": "filling gaps with a fixed value",
    "encode_binary": "a single 0/1 column",
    "encode_onehot": "one column per value",
    "encode_ordinal": "one number per value",
    "encode_target": "the average outcome per value",
    "encode_frequency": "how often each value occurs",
    "datetime_expand": "splitting the date into parts",
    "passthrough": "leaving it untouched",
}


# --------------------------------------------------------------------------------------
# The planner
# --------------------------------------------------------------------------------------


def _pct(value: float) -> str:
    return f"{value:.0%}" if value >= 0.01 else f"{value:.1%}"


def _kind_of(series: pd.Series, n_unique: int, top_share: float) -> str:
    """The column's kind, by the §1 rules, first match winning."""
    if n_unique <= 1 or top_share >= NEAR_CONSTANT_SHARE:
        return "constant"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    n_rows = max(len(series), 1)
    if n_unique / n_rows > 0.5 and n_unique > 50:
        return "text"
    return "categorical"


def _rare_categories(series: pd.Series, rate: float) -> list[str]:
    """Values occurring in less than ``rate`` of the rows.

    Collapsing these into "Other" is not tidying. A category seen four times in ten
    thousand rows gives the model a column that is almost always zero, and whatever it
    learns from those four rows will not generalise — while a value that appears *only*
    in the held-out fold would otherwise be unseen at prediction time.
    """
    if rate <= 0:
        return []
    counts = series.value_counts(normalize=True, dropna=True)
    return [str(value) for value, share in counts.items() if share < rate]


def prepare(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """The stateless repairs that must happen before the split, and their record.

    Both the planner and the executor need this, and they need *the same* result. The
    planner measures a column's missing rate and cardinality on the training fold; the
    executor fits on the training fold. If one of them deduplicates first and the other
    does not, the two folds contain different rows and every statistic the user reviewed
    describes data the pipeline never saw. The split is seeded, so identical input gives
    identical folds — which makes this function, and not the seed alone, what keeps the
    plan and the run talking about the same rows.

    Returns the prepared frame and one record per repair, in the ``{kind, severity,
    message, suggestion}`` shape the rest of the application uses for warnings.
    """
    records: list[dict[str, Any]] = []

    blank_targets = int(df[target].isna().sum())
    if blank_targets:
        df = df[df[target].notna()].reset_index(drop=True)
        records.append({
            "kind": "blank_targets_removed",
            "severity": "info",
            "message": f"Left out {blank_targets:,} row(s) with no value in {target}.",
            "suggestion": (
                "A row with no answer recorded cannot teach the model anything, and "
                "cannot be scored against either."
            ),
            "count": blank_targets,
        })

    duplicates = int(df.duplicated().sum())
    if duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
        records.append({
            "kind": "duplicates_removed",
            "severity": "info",
            "message": f"Removed {duplicates:,} duplicate row(s) before splitting.",
            "suggestion": (
                "Copies of one row can otherwise land on both sides of the split, which "
                "makes any score on the held-out rows optimistic."
            ),
            "count": duplicates,
        })

    return df, records


def propose(
    df: pd.DataFrame,
    target: str,
    task: str,
    options: GlobalOptions | None = None,
    excluded: dict[str, str] | None = None,
    warnings: list[dict[str, Any]] | None = None,
    column_mapping: dict[str, Any] | None = None,
    screeners: "Sequence[Screener] | None" = None,
) -> PreprocessingPlan:
    """Decide what to do with every column, and say why.

    Args:
        df: the cleaned frame, after ingestion's stateless repairs.
        target: the column being predicted.
        task: ``binary classification`` | ``multiclass classification`` | ``regression``.
        options: global choices; defaults are the spec's.
        excluded: columns the caller wants removed outright, with the reason to show
            verbatim — a stated preference rather than a measurement.
        screeners: measurement-based screens, run here rather than by the caller so they
            see the **same training fold** the statistics below are measured on. A
            caller running them itself would have to reproduce this function's split to
            do that, and a second copy of a split is a second thing to get wrong.
        warnings: conditions found before planning, carried onto the plan.
        column_mapping: ``{original: clean}`` from ingestion, so a plan can show the
            name the user actually typed.

    Returns:
        A :class:`PreprocessingPlan` covering every column in ``df``.

    The decisions here are measured on the **training fold only**. The split is taken
    inside this function using ``options.seed`` and ``options.test_size``, and executing
    the plan reproduces it exactly.
    """
    options = options or GlobalOptions()
    excluded = dict(excluded or {})
    plan_warnings = list(warnings or [])

    # Reverse the ingestion mapping so a cleaned name can report what it was called.
    original_of: dict[str, str] = {}
    for original, cleaned in (column_mapping or {}).items():
        for name in (cleaned if isinstance(cleaned, list) else [cleaned]):
            original_of[str(name)] = str(original)

    # The stateless repairs first, so the rows measured below are the rows that will be
    # fitted on. See :func:`prepare`.
    df, prepared_records = prepare(df, target)
    plan_warnings.extend(prepared_records)

    if df.empty:
        raise ValueError(
            f"No rows are left once rows with no value in {target!r} are set aside, so "
            "there is nothing to plan."
        )

    # The split, before anything is measured. The decision whether to stratify is the
    # shared one in :func:`~app.services.glassbox.choose_stratification`: a second copy
    # of that rule is what let a fold smaller than the class count abort a run.
    from sklearn.model_selection import train_test_split

    from app.services.glassbox import choose_stratification

    classification = "class" in task
    y = df[target]
    stratify, stratify_warning = choose_stratification(y, classification, options.test_size)
    if stratify_warning is not None:
        plan_warnings.append(stratify_warning)
    train_index, _ = train_test_split(
        df.index, test_size=options.test_size, random_state=options.seed,
        shuffle=True, stratify=stratify,
    )
    train = df.loc[train_index]

    # Screens measure, so they run on the training fold like every other measurement
    # here. A caller's own exclusions take precedence: those are a stated preference,
    # and the more specific message is the one the user should read.
    features_train = train.drop(columns=[target])
    for screener in screeners or ():
        screened, _flags = screener(features_train, train[target])
        for column, reason in screened.items():
            excluded.setdefault(column, reason)

    columns: list[ColumnPlan] = []
    for name in df.columns:
        column = str(name)
        series = train[column]
        full = df[column]

        n_missing = int(series.isna().sum())
        missing_rate = float(series.isna().mean())
        n_unique = int(series.nunique(dropna=True))
        values = series.dropna()
        top_share = float(values.value_counts(normalize=True).iloc[0]) if not values.empty else 0.0
        kind = _kind_of(series, n_unique, top_share)

        stats: dict[str, Any] = {
            "n_missing": n_missing,
            "missing_rate": round(missing_rate, 4),
            "n_unique": n_unique,
            "top_share": round(top_share, 4),
            "rows_measured": int(len(series)),
        }
        if kind == "numeric" and not values.empty:
            skew = float(values.astype("float64").skew()) if n_unique > 1 and len(values) > 2 else 0.0
            stats["skew"] = round(skew, 4)
            stats["median"] = round(float(values.median()), 6)
            stats["mean"] = round(float(values.mean()), 6)

        entry = ColumnPlan(
            column=column,
            original_name=original_of.get(column),
            kind=kind,
            dtype=str(full.dtype),
            action="passthrough",
            reason="",
            stats=stats,
            alternatives=ACTIONS_FOR_KIND.get(kind, []),
        )

        # ---- the target is never a feature -------------------------------------
        if column == target:
            entry.action = "passthrough"
            entry.alternatives = []
            entry.reason = (
                f"This is the column being predicted, so it is never transformed as a "
                f"feature. The task is {task}."
            )
            columns.append(entry)
            continue

        # ---- a screen already rejected it --------------------------------------
        if column in excluded:
            entry.action = "drop"
            entry.reason = excluded[column]
            entry.warning = {
                "severity": "critical" if "answer" in excluded[column] else "warn",
                "message": excluded[column],
                "suggestion": "You can put this column back if you disagree.",
            }
            columns.append(entry)
            continue

        # ---- structural drops ---------------------------------------------------
        if missing_rate > MAX_MISSING_RATE:
            entry.action = "drop"
            entry.reason = (
                f"{_pct(missing_rate)} of this column is empty in the training rows. "
                f"Filling that much would record our fill value more than your data."
            )
            columns.append(entry)
            continue

        if kind == "constant":
            entry.action = "drop"
            entry.reason = (
                "Every row holds the same value, so the column cannot explain anything "
                "that varies."
                if n_unique <= 1
                else (
                    f"One value covers {_pct(top_share)} of the rows, so the column "
                    "barely varies and carries almost nothing."
                )
            )
            columns.append(entry)
            continue

        if kind == "text":
            entry.action = "drop"
            entry.reason = (
                f"{n_unique:,} different values across {len(series):,} rows — this "
                "identifies the row rather than describing it."
            )
            columns.append(entry)
            continue

        # ---- the kinds we keep --------------------------------------------------
        if kind == "datetime":
            entry.action = "datetime_expand"
            entry.reason = (
                "A timestamp is one large number whose size means nothing. This splits "
                "it into year, month, day, weekday and hour, with a sine/cosine pair "
                "for the repeating parts so December sits next to January."
            )

        elif kind == "numeric":
            if n_missing:
                entry.action = "impute_median"
                entry.reason = (
                    f"{n_missing:,} of {len(series):,} training rows have no value "
                    f"({_pct(missing_rate)}). The median is used rather than the mean "
                    "because a few extreme values cannot drag it around."
                )
            else:
                entry.action = "keep"
                entry.reason = "Numeric with no gaps — nothing needs doing before scaling."

            skew = stats.get("skew", 0.0)
            if abs(skew) > SKEW_THRESHOLD:
                entry.warning = {
                    "severity": "info",
                    "message": (
                        f"This column is skewed ({skew:+.2f}), so most values sit on one "
                        "side of a long tail."
                    ),
                    "suggestion": (
                        "A log or Yeo-Johnson transform would make it more symmetric. "
                        "Enable 'power transform skewed columns' to apply one."
                    ),
                }

        elif kind == "boolean":
            entry.action = "encode_binary"
            entry.reason = "Two possible answers, so a single 0/1 column carries all of it."

        else:  # categorical
            rare = _rare_categories(series, options.rare_category_rate)
            entry.rare_categories = rare
            effective_levels = n_unique - len(rare) + (1 if rare else 0)

            if n_unique == 2:
                entry.action = "encode_binary"
                entry.reason = "Two distinct values, so a single 0/1 column says it all."
            elif effective_levels <= options.onehot_max_cardinality:
                entry.action = "encode_onehot"
                entry.reason = (
                    f"{n_unique} distinct values, so one column per value is affordable "
                    "and avoids implying the categories have an order."
                )
            else:
                entry.action = "encode_target"
                entry.reason = (
                    f"{n_unique} distinct values — one column each would swamp the "
                    "model. Each value becomes the average outcome recorded for it, "
                    "cross-fitted so no row informs its own encoded value."
                )

            if rare:
                covered = float(series.isin(rare).mean())
                entry.reason += (
                    f" {len(rare)} value(s) occurring in under "
                    f"{options.rare_category_rate:.0%} of rows "
                    f"({_pct(covered)} of the column) are grouped as 'Other'."
                )

            if n_missing:
                entry.reason += (
                    f" {n_missing:,} blank(s) are filled with the most common value "
                    "first."
                )

        # An indicator marks which rows were filled, so the model can tell an imputed
        # value from a real one — and so "missing" can itself be predictive.
        if options.missing_indicators and missing_rate > MISSING_INDICATOR_RATE:
            entry.add_indicator = True
            entry.reason += (
                f" A separate yes/no column records which rows were blank, because at "
                f"{_pct(missing_rate)} the fact of being missing may itself carry "
                "information."
            )

        columns.append(entry)

    return PreprocessingPlan(
        target=target,
        task=task,
        columns=columns,
        options=options,
        warnings=plan_warnings,
        prepared={
            "rows": int(len(df)),
            "train_rows": int(len(train_index)),
            "test_rows": int(len(df) - len(train_index)),
            "stratified": stratify is not None,
        },
    )


AnnotatedPlan = Annotated[PreprocessingPlan, Field(description="A preprocessing plan")]
