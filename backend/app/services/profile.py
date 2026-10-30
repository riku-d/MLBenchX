"""Describe the uploaded dataset, column by column, for the profile dashboard.

This module only measures. It makes no decision about what to keep, encode or drop —
that is :mod:`app.services.glassbox`'s job, and the two are deliberately separate:

* **Profiling for display** describes the file the user uploaded, so it reads every row.
  Telling someone "12% of ``income`` is blank" about only 80% of their file would be a
  strange kind of honesty.
* **Profiling for decisions** happens inside ``fit_preprocessing`` on the training fold
  alone, because a decision informed by held-out rows is leakage however indirect.

Both exist, they answer different questions, and neither is a substitute for the other.
Nothing computed here is ever fed back into the pipeline.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

import numpy as np

from app.config import (
    HISTOGRAM_BINS,
    MAX_MISSING_RATE,
    OUTLIER_IQR_MULTIPLIER,
    SKEW_THRESHOLD,
    TOP_VALUES,
)
from app.services.glassbox import ColumnProfile, profile_frame
from app.services.screening import correlated_pairs

__all__ = [
    "column_profiles",
    "outlier_summary",
    "numeric_summary",
    "histogram",
    "top_values",
    "build_profile",
]

#: How each internal ``kind`` is described to someone who does not think in dtypes.
_KIND_LABEL = {
    "numeric": "number",
    "categorical": "category",
    "datetime": "date/time",
    "text": "free text",
    "constant": "single value",
}


def _finite(values: pd.Series) -> pd.Series:
    """The float values of a numeric column that can actually be summarised.

    Ingestion reads an infinity as a blank, so a frame that came through
    :mod:`app.services.ingest` has none left. These functions are public and are also
    handed frames built elsewhere, and every statistic below is undefined on a
    non-finite value: a mean goes infinite, ``numpy.histogram`` refuses the range
    outright, and both would reach the browser as a bare ``Infinity`` token that no
    strict JSON parser reads back. Dropping them here describes the part of the column
    that can be described; the count of what was dropped is the validity dimension's
    job, in :mod:`app.services.health`.
    """
    numbers = values.astype("float64")
    return numbers[np.isfinite(numbers)]


def outlier_summary(series: pd.Series) -> dict[str, Any] | None:
    """Count values outside Tukey's fences, or ``None`` for a non-numeric column.

    An outlier is a value more than ``OUTLIER_IQR_MULTIPLIER`` interquartile ranges
    below the first quartile or above the third — the rule every box plot draws. It is
    reported and never acted on: an implausible value and an unusually large real one
    look identical from here, and only the person who collected the data can tell them
    apart.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return None
    values = _finite(series.dropna())
    if values.empty:
        return None

    q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
    iqr = q3 - q1
    if iqr <= 0:
        # More than half the column is one value, so the fences collapse onto it and
        # would condemn every other value in the column.
        return {"count": 0, "share": 0.0, "lower": q1, "upper": q3}

    lower = q1 - OUTLIER_IQR_MULTIPLIER * iqr
    upper = q3 + OUTLIER_IQR_MULTIPLIER * iqr
    outside = values[(values < lower) | (values > upper)]
    return {
        "count": int(outside.size),
        "share": round(float(outside.size) / float(values.size), 4),
        "lower": round(lower, 4),
        "upper": round(upper, 4),
    }


def numeric_summary(series: pd.Series) -> dict[str, Any] | None:
    """Descriptive statistics for one numeric column, or ``None`` for anything else.

    Skew is included because it is the statistic that changes what you should *do*: a
    column with |skew| above 1 is a candidate for a log or Yeo-Johnson transform, and
    is also the reason a mean and a median disagree. Both are reported so the reader
    can see that disagreement rather than take the skew number on trust.
    """
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return None
    values = _finite(series.dropna())
    if values.empty:
        return None

    skew = float(values.skew()) if len(values) > 2 and values.nunique() > 1 else 0.0
    return {
        "count": int(values.size),
        "mean": round(float(values.mean()), 6),
        "std": round(float(values.std()), 6) if len(values) > 1 else 0.0,
        "min": round(float(values.min()), 6),
        "q1": round(float(values.quantile(0.25)), 6),
        "median": round(float(values.median()), 6),
        "q3": round(float(values.quantile(0.75)), 6),
        "max": round(float(values.max()), 6),
        "skew": round(skew, 4),
        "skewed": bool(abs(skew) > SKEW_THRESHOLD),
    }


def histogram(series: pd.Series, bins: int = HISTOGRAM_BINS) -> dict[str, Any] | None:
    """Bin counts for a numeric column, computed server-side.

    Computed here rather than in the browser because the browser holds one page of rows
    and would draw a different distribution on every page — a histogram of 25 rows is
    not a histogram of the column.

    Categorical columns get their value counts instead, which is the same question
    ("how is this distributed?") answered the way that column can answer it.
    """
    values = series.dropna()
    if values.empty:
        return None

    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        counts = values.value_counts().head(bins)
        return {
            "kind": "categorical",
            "labels": [str(v) for v in counts.index],
            "counts": [int(v) for v in counts.values],
            "other": int(len(values) - int(counts.sum())),
        }

    numbers = _finite(values)
    if numbers.empty:
        return None
    if numbers.nunique() == 1:
        only = float(numbers.iloc[0])
        return {
            "kind": "numeric", "bins": 1, "edges": [only, only],
            "counts": [int(numbers.size)],
        }

    counts, edges = np.histogram(numbers.to_numpy(), bins=min(bins, max(numbers.nunique(), 2)))
    return {
        "kind": "numeric",
        "bins": int(len(counts)),
        "edges": [round(float(e), 6) for e in edges],
        "counts": [int(c) for c in counts],
    }


def top_values(series: pd.Series, limit: int = TOP_VALUES) -> list[dict[str, Any]]:
    """The most frequent values, with their share.

    Shown for every column kind, not only categoricals: the most common value of a
    numeric column is how you notice that 40% of it is zero, which no mean or quartile
    reveals.
    """
    values = series.dropna()
    if values.empty:
        return []
    counts = values.value_counts().head(limit)
    total = int(values.size)
    return [
        {"value": str(index), "count": int(count), "share": round(int(count) / total, 4)}
        for index, count in counts.items()
    ]


def _flags(profile: ColumnProfile, outliers: dict | None, leaking: bool) -> list[str]:
    """Short tags for the per-column table. Ordered most to least serious.

    Only things the table's own numbers do not already say. The blank percentage and
    the outlier count each have a column, so "has blanks" and "18 outliers" would be
    the same fact twice — and a tag on every row is a tag that means nothing. What is
    left is the reason a column's number matters: that it never varies, that it is too
    empty to use, that it is an identifier, that it looks like the answer.
    """
    tags: list[str] = []
    if leaking:
        tags.append("looks like the answer")
    if profile.kind == "constant":
        tags.append("never changes")
    if profile.missing_rate > MAX_MISSING_RATE:
        tags.append("too empty to use")
    elif profile.missing_rate >= 0.5:
        tags.append("mostly blank")
    if profile.kind == "text":
        tags.append("free text")
    if profile.kind == "datetime":
        tags.append("date/time")
    return tags


def column_profiles(
    df: pd.DataFrame,
    target: str | None = None,
    excluded: dict[str, str] | None = None,
    leakage_flags: list[dict] | None = None,
) -> list[dict[str, Any]]:
    """One record per column: what it is, what is wrong with it, and what we did.

    Args:
        df: the cleaned frame as uploaded.
        target: the column being predicted, marked as such rather than flagged.
        excluded: column -> reason for everything the pipeline set aside.
        leakage_flags: records from the target-leakage screen.
    """
    excluded = excluded or {}
    leaking = {flag["column"]: flag for flag in (leakage_flags or [])}
    profiles = profile_frame(df)
    n_rows = len(df)

    records: list[dict[str, Any]] = []
    for name, profile in profiles.items():
        series = df[name]
        outliers = outlier_summary(series)
        is_target = name == target
        records.append({
            "column": name,
            "kind": profile.kind,
            "kind_label": _KIND_LABEL.get(profile.kind, profile.kind),
            "dtype": profile.dtype,
            "n_missing": profile.n_missing,
            "missing_rate": round(profile.missing_rate, 4),
            "n_unique": profile.n_unique,
            "unique_rate": round(profile.unique_rate, 4),
            "n_rows": n_rows,
            "outliers": outliers,
            "stats": numeric_summary(series),
            "histogram": histogram(series),
            "top_values": top_values(series),
            "is_target": is_target,
            "used": not is_target and name not in excluded,
            "excluded_reason": excluded.get(name),
            "leakage": leaking.get(name),
            "flags": [] if is_target else _flags(profile, outliers, name in leaking),
        })
    return records


def build_profile(
    df: pd.DataFrame,
    target: str | None = None,
    excluded: dict[str, str] | None = None,
    leakage_flags: list[dict] | None = None,
    duplicate_rows: int = 0,
    balance: dict | None = None,
) -> dict[str, Any]:
    """The whole profile payload: per-column records plus the dataset-wide signals."""
    return {
        "columns": column_profiles(df, target, excluded, leakage_flags),
        "duplicate_rows": duplicate_rows,
        "correlated_pairs": correlated_pairs(df),
        "leakage_flags": list(leakage_flags or []),
        "class_balance": balance,
        "n_rows": len(df),
        "n_columns": int(df.shape[1]),
    }
