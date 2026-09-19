"""Dataset health, as six measured dimensions rather than one opaque number.

A single score out of 100 is easy to display and impossible to act on. "You scored 61"
tells a user nothing about what to do next; "your data is 87% complete but only 54%
unique" tells them to go looking for duplicates. So the score here is a weighted mean of
six dimensions, each independently measured, each shown with the evidence behind it:

``completeness``   How much of the data is actually there.
``consistency``    Whether columns hold one kind of thing, in one format.
``validity``       Whether values fall in ranges that make sense.
``uniqueness``     Whether rows are distinct observations rather than repeats.
``distribution``   Whether numeric columns are shaped in a way models can use.
``feature_quality`` Whether the columns carry signal, independently of each other.

Every dimension returns a 0–100 score, the facts that produced it, and a sentence a
non-specialist can read. Nothing is a magic constant hidden in a function: the weights
live in :data:`DIMENSION_WEIGHTS` and the whole calculation is handed to the interface
so the user can take it apart.

This module measures and never decides. It is called on the dataset as uploaded and
again on the processed result, and the difference between the two is the honest answer
to "did the pipeline help?".
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.config import (
    HIGH_CORRELATION,
    IMBALANCE_RATIO,
    OUTLIER_IQR_MULTIPLIER,
)

__all__ = ["DIMENSION_WEIGHTS", "DIMENSION_LABELS", "assess", "HealthReport"]

#: How much each dimension contributes to the headline score.
#:
#: Completeness and validity weigh most because they are the two that stop a model
#: training at all — a blank or a non-numeric value in a numeric column is a hard
#: failure, not a degradation. Distribution weighs least: a skewed column is a reason to
#: pick a different scaler, not a reason to distrust the data.
DIMENSION_WEIGHTS: dict[str, float] = {
    "completeness": 0.25,
    "validity": 0.20,
    "consistency": 0.15,
    "uniqueness": 0.15,
    "feature_quality": 0.15,
    "distribution": 0.10,
}

DIMENSION_LABELS: dict[str, tuple[str, str]] = {
    "completeness": ("Completeness", "How much of the data is actually present."),
    "validity": ("Validity", "Whether values sit in ranges that make sense."),
    "consistency": ("Consistency", "Whether each column holds one kind of thing."),
    "uniqueness": ("Uniqueness", "Whether rows are distinct observations."),
    "feature_quality": ("Feature Quality", "Whether columns carry independent signal."),
    "distribution": ("Distribution", "Whether numeric columns are shaped usefully."),
}

#: Below this a dimension is called a problem rather than a note.
_CONCERN = 75.0


class HealthReport(dict):
    """The assessment. A dict so it serialises directly, with the score to hand."""

    @property
    def score(self) -> int:
        return int(self["score"])


def _rating(score: float) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 55:
        return "Usable"
    if score >= 35:
        return "Poor"
    return "Critical"


def _feature_columns(df: pd.DataFrame, target: str | None) -> list[str]:
    return [c for c in df.columns if c != target]


# --------------------------------------------------------------------------------------
# The six dimensions
# --------------------------------------------------------------------------------------


def _completeness(df: pd.DataFrame, target: str | None) -> dict[str, Any]:
    """The share of cells that hold a value.

    Cells rather than rows: one blank in a 50-column row does not make the row useless,
    and scoring by row would call a dataset with one sparse column catastrophic.
    """
    total = max(df.shape[0] * df.shape[1], 1)
    missing = int(df.isna().sum().sum())
    filled_share = 1.0 - (missing / total)

    per_column = {
        str(c): round(float(df[c].isna().mean()), 4) for c in df.columns if df[c].isna().any()
    }
    worst = sorted(per_column.items(), key=lambda kv: -kv[1])[:5]

    return {
        "score": round(filled_share * 100, 1),
        "facts": {
            "missing_cells": missing,
            "total_cells": total,
            "complete_share": round(filled_share, 4),
            "columns_with_gaps": len(per_column),
            "worst_columns": [{"column": c, "missing_rate": r} for c, r in worst],
        },
        "summary": (
            "Every cell holds a value."
            if missing == 0
            else (
                f"{missing:,} of {total:,} cells are blank "
                f"({missing / total:.1%}), across {len(per_column)} column(s)."
            )
        ),
    }


def _validity(df: pd.DataFrame, target: str | None) -> dict[str, Any]:
    """Whether values fall where they plausibly should.

    Two things count as invalid here, and both are cases where the value is almost
    certainly wrong rather than merely unusual: infinities, and numbers outside Tukey's
    fences.

    An earlier version also flagged negative values in columns whose *name* suggested a
    non-negative quantity — ``age``, ``price``, ``fare``. That check is removed rather
    than fixed. It inferred meaning from a column name, which is guesswork, and it
    misfired on exactly the data this app produces: StandardScaler centres a column on
    its mean, so half of every scaled column is negative by construction. It reported
    "Age should not go below zero" about a correctly scaled Age column, which is the
    kind of confident wrongness that costs a tool its credibility.
    """
    numeric = df[_feature_columns(df, target)].select_dtypes(include="number")
    if numeric.empty:
        return {
            "score": 100.0,
            "facts": {"checked_columns": 0},
            "summary": "No numeric columns to check.",
        }

    total_values = int(numeric.notna().sum().sum()) or 1
    infinite = int(np.isinf(numeric.to_numpy(dtype="float64", na_value=np.nan)).sum())

    outliers = 0
    per_column: dict[str, int] = {}
    for column in numeric.columns:
        values = numeric[column].dropna()
        if values.empty:
            continue
        q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            continue
        low = q1 - OUTLIER_IQR_MULTIPLIER * iqr
        high = q3 + OUTLIER_IQR_MULTIPLIER * iqr
        count = int(((values < low) | (values > high)).sum())
        if count:
            per_column[str(column)] = count
            outliers += count

    invalid = infinite + outliers
    score = max(0.0, 100.0 * (1 - invalid / total_values))

    notes: list[str] = []
    if outliers:
        notes.append(f"{outliers:,} value(s) outside the usual range")
    if infinite:
        notes.append(f"{infinite} infinite value(s)")

    return {
        "score": round(score, 1),
        "facts": {
            "checked_columns": int(numeric.shape[1]),
            "outliers": outliers,
            "infinite": infinite,
            "per_column_outliers": dict(sorted(per_column.items(), key=lambda kv: -kv[1])[:5]),
        },
        "summary": (
            "Every numeric value sits in a plausible range."
            if not notes
            else "Found " + "; ".join(notes) + "."
        ),
    }


def _consistency(df: pd.DataFrame, target: str | None) -> dict[str, Any]:
    """Whether a column holds one kind of thing, written one way.

    Catches the two failures that look fine in a spreadsheet and break a pipeline: a
    column of numbers with a few words in it, and a category column where ``"Yes"``,
    ``"yes"`` and ``" Yes "`` are three different values.
    """
    columns = _feature_columns(df, target)
    if not columns:
        return {"score": 100.0, "facts": {}, "summary": "No feature columns to check."}

    mixed: dict[str, dict[str, Any]] = {}
    casing: dict[str, int] = {}

    for column in columns:
        series = df[column].dropna()
        if series.empty or series.dtype != object:
            continue

        as_text = series.astype("string")
        numeric_like = pd.to_numeric(as_text.str.strip(), errors="coerce").notna()
        share = float(numeric_like.mean())
        # A column that is mostly-but-not-entirely numeric is the dangerous case: fully
        # numeric is fine, fully textual is fine, 97% numeric is a broken export.
        if 0.5 < share < 1.0:
            mixed[str(column)] = {
                "numeric_share": round(share, 3),
                "non_numeric_examples": [
                    str(v) for v in series[~numeric_like.to_numpy()].unique()[:3]
                ],
            }

        normalised = as_text.str.strip().str.lower()
        collapsed = int(series.nunique() - normalised.nunique())
        if collapsed > 0:
            casing[str(column)] = collapsed

    penalty = min(len(mixed) * 18 + len(casing) * 10, 100)
    notes: list[str] = []
    if mixed:
        notes.append(f"{len(mixed)} column(s) mix numbers and text")
    if casing:
        notes.append(f"{len(casing)} column(s) have values differing only by case or spacing")

    return {
        "score": round(max(0.0, 100.0 - penalty), 1),
        "facts": {"mixed_type_columns": mixed, "inconsistent_casing": casing},
        "summary": (
            "Each column holds one kind of value, written consistently."
            if not notes
            else "Found " + "; ".join(notes) + "."
        ),
    }


def _uniqueness(df: pd.DataFrame, target: str | None) -> dict[str, Any]:
    """Whether rows are distinct observations.

    Duplicate rows are the headline. Constant columns are counted here too: a column
    with one value is the degenerate case of non-uniqueness, and it contributes nothing
    to any model while taking up a slot.
    """
    rows = max(len(df), 1)
    duplicates = int(df.duplicated().sum())
    constant = [str(c) for c in _feature_columns(df, target) if df[c].nunique(dropna=True) <= 1]

    duplicate_share = duplicates / rows
    # Duplicates are weighted 2x: one duplicate in a hundred rows is a real problem,
    # while one constant column in a hundred is untidy.
    penalty = min(duplicate_share * 200, 70) + min(len(constant) * 6, 30)

    return {
        "score": round(max(0.0, 100.0 - penalty), 1),
        "facts": {
            "duplicate_rows": duplicates,
            "duplicate_share": round(duplicate_share, 4),
            "constant_columns": constant,
            "distinct_rows": int(rows - duplicates),
        },
        "summary": (
            "Every row is a distinct observation."
            if not duplicates and not constant
            else "; ".join(
                part
                for part in (
                    f"{duplicates:,} duplicate row(s)" if duplicates else "",
                    f"{len(constant)} column(s) that never change" if constant else "",
                )
                if part
            )
            + "."
        ),
    }


def _distribution(df: pd.DataFrame, target: str | None) -> dict[str, Any]:
    """Whether numeric columns are shaped in a way a model can use.

    Two measures: skew, because a heavily one-sided column makes linear models fit the
    tail rather than the bulk; and scale disparity, because a column in millions next to
    one in single digits dominates any distance calculation until it is scaled.
    """
    numeric = df[_feature_columns(df, target)].select_dtypes(include="number")
    if numeric.shape[1] == 0:
        return {"score": 100.0, "facts": {"numeric_columns": 0},
                "summary": "No numeric columns to assess."}

    skews = {}
    for column in numeric.columns:
        values = numeric[column].dropna()
        if len(values) < 3 or values.nunique() < 2:
            continue
        skews[str(column)] = round(float(values.skew()), 3)

    heavily_skewed = {c: s for c, s in skews.items() if abs(s) > 2.0}

    # A column that is constant, or constant to within floating-point noise, has no
    # scale to compare. Including one makes the ratio astronomical and says nothing:
    # an earlier version reported a disparity of 8e31 because one derived column's
    # "spread" was rounding error.
    spreads = {}
    for column in numeric.columns:
        values = numeric[column].dropna()
        if values.empty or values.nunique() < 2:
            continue
        spread = float(values.std())
        magnitude = max(abs(float(values.mean())), 1.0)
        if spread <= 1e-9 * magnitude:
            continue
        spreads[str(column)] = spread

    disparity = (max(spreads.values()) / min(spreads.values())) if len(spreads) > 1 else 1.0

    penalty = min(len(heavily_skewed) * 8, 40)
    if disparity > 1000:
        penalty += 25
    elif disparity > 100:
        penalty += 12

    notes: list[str] = []
    if heavily_skewed:
        notes.append(f"{len(heavily_skewed)} heavily skewed column(s)")
    if disparity > 100:
        notes.append(f"the widest column varies {disparity:,.0f}x more than the narrowest")

    return {
        "score": round(max(0.0, 100.0 - penalty), 1),
        "facts": {
            "numeric_columns": int(numeric.shape[1]),
            "skew": dict(sorted(skews.items(), key=lambda kv: -abs(kv[1]))[:5]),
            "heavily_skewed": heavily_skewed,
            "scale_disparity": round(disparity, 2),
        },
        "summary": (
            "Numeric columns are reasonably shaped and comparably scaled."
            if not notes
            else "Found " + "; ".join(notes) + "."
        ),
    }


def _feature_quality(df: pd.DataFrame, target: str | None) -> dict[str, Any]:
    """Whether the columns carry independent signal.

    Penalises three things: columns that are near-unique and so identify the row rather
    than describing it; pairs that repeat each other; and — when a target is known — a
    column correlated with it so strongly that it is almost certainly derived from the
    answer.
    """
    columns = _feature_columns(df, target)
    if not columns:
        return {"score": 100.0, "facts": {}, "summary": "No feature columns to assess."}

    rows = max(len(df), 1)
    identifiers = [
        str(c) for c in columns
        if df[c].nunique(dropna=True) / rows > 0.98 and not pd.api.types.is_float_dtype(df[c])
    ]

    numeric = df[columns].select_dtypes(include="number")
    redundant: list[dict[str, Any]] = []
    if numeric.shape[1] > 1:
        matrix = numeric.corr().abs()
        names = list(matrix.columns)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                value = matrix.loc[left, right]
                if pd.notna(value) and float(value) >= HIGH_CORRELATION:
                    redundant.append(
                        {"columns": [str(left), str(right)], "correlation": round(float(value), 3)}
                    )

    leaking: list[dict[str, Any]] = []
    if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
        for column in numeric.columns:
            pair = pd.concat([numeric[column], df[target]], axis=1).dropna()
            if len(pair) < 30 or pair.iloc[:, 0].nunique() < 2:
                continue
            value = pair.iloc[:, 0].corr(pair.iloc[:, 1])
            if pd.notna(value) and abs(float(value)) >= 0.99:
                leaking.append({"column": str(column), "correlation": round(float(value), 3)})

    penalty = min(len(identifiers) * 12, 30) + min(len(redundant) * 8, 30) + min(len(leaking) * 20, 40)

    notes: list[str] = []
    if leaking:
        notes.append(f"{len(leaking)} column(s) that look like a copy of the answer")
    if identifiers:
        notes.append(f"{len(identifiers)} identifier-like column(s)")
    if redundant:
        notes.append(f"{len(redundant)} redundant pair(s)")

    return {
        "score": round(max(0.0, 100.0 - penalty), 1),
        "facts": {
            "identifier_columns": identifiers,
            "redundant_pairs": redundant[:8],
            "leakage_suspects": leaking,
            "feature_count": len(columns),
        },
        "summary": (
            "Columns carry independent signal."
            if not notes
            else "Found " + "; ".join(notes) + "."
        ),
    }


_DIMENSIONS = {
    "completeness": _completeness,
    "validity": _validity,
    "consistency": _consistency,
    "uniqueness": _uniqueness,
    "distribution": _distribution,
    "feature_quality": _feature_quality,
}


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------


def assess(df: pd.DataFrame, target: str | None = None) -> HealthReport:
    """Measure all six dimensions and combine them into one score.

    Args:
        df: the frame to assess — as uploaded, or after a pipeline has run.
        target: the column being predicted, excluded from feature-level checks and used
            for the leakage check.

    Returns:
        A :class:`HealthReport`: the headline score, every dimension with its evidence,
        the issues worth acting on, and the arithmetic that produced the score.
    """
    dimensions: dict[str, Any] = {}
    for name, measure in _DIMENSIONS.items():
        result = measure(df, target)
        label, description = DIMENSION_LABELS[name]
        dimensions[name] = {
            "key": name,
            "label": label,
            "description": description,
            "score": result["score"],
            "weight": DIMENSION_WEIGHTS[name],
            "contribution": round(result["score"] * DIMENSION_WEIGHTS[name], 2),
            "facts": result["facts"],
            "summary": result["summary"],
            "status": "good" if result["score"] >= _CONCERN else "concern",
        }

    score = sum(d["contribution"] for d in dimensions.values())

    issues = [
        {
            "dimension": d["key"],
            "label": d["label"],
            "score": d["score"],
            "text": d["summary"],
            "severity": "warning" if d["score"] < 55 else "info",
        }
        for d in sorted(dimensions.values(), key=lambda d: d["score"])
        if d["status"] == "concern"
    ]

    balance = _class_balance(df, target)

    return HealthReport({
        "score": int(round(score)),
        "rating": _rating(score),
        "dimensions": list(dimensions.values()),
        "issues": issues,
        "class_balance": balance,
        "formula": {
            "description": (
                "Each dimension is measured independently, then multiplied by its weight "
                "and summed. Nothing else contributes."
            ),
            "terms": [
                {
                    "label": d["label"],
                    "score": d["score"],
                    "weight": d["weight"],
                    "contribution": d["contribution"],
                }
                for d in dimensions.values()
            ],
            "total": round(score, 2),
        },
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
    })


def _class_balance(df: pd.DataFrame, target: str | None) -> dict[str, Any] | None:
    """How lopsided a classification target is. ``None`` when there is nothing to say."""
    if not target or target not in df.columns:
        return None
    series = df[target].dropna()
    if series.empty:
        return None
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 20:
        return None  # a regression target has no classes to balance

    counts = series.value_counts()
    if len(counts) < 2:
        return None

    total = int(counts.sum())
    largest, smallest = int(counts.iloc[0]), int(counts.iloc[-1])
    ratio = largest / max(smallest, 1)

    return {
        "ratio": round(ratio, 2),
        "imbalanced": ratio >= IMBALANCE_RATIO,
        "classes": [
            {"value": str(k), "count": int(v), "share": round(int(v) / total, 4)}
            for k, v in counts.items()
        ],
        "summary": (
            f"{counts.index[0]} accounts for {largest / total:.0%} of rows while "
            f"{counts.index[-1]} has {smallest:,}. Judge a model on precision and recall "
            "for the rarer class, not on accuracy."
            if ratio >= IMBALANCE_RATIO
            else f"Classes are reasonably balanced at {ratio:.1f}:1."
        ),
    }


def compare(before: HealthReport, after: HealthReport) -> dict[str, Any]:
    """What changed between two assessments, dimension by dimension."""
    after_by_key = {d["key"]: d for d in after["dimensions"]}
    return {
        "score": {
            "before": before["score"],
            "after": after["score"],
            "delta": after["score"] - before["score"],
        },
        "rows": {"before": before["rows"], "after": after["rows"],
                 "delta": after["rows"] - before["rows"]},
        "columns": {"before": before["columns"], "after": after["columns"],
                    "delta": after["columns"] - before["columns"]},
        "dimensions": [
            {
                "key": d["key"],
                "label": d["label"],
                "before": d["score"],
                "after": after_by_key[d["key"]]["score"],
                "delta": round(after_by_key[d["key"]]["score"] - d["score"], 1),
            }
            for d in before["dimensions"]
            if d["key"] in after_by_key
        ],
    }
