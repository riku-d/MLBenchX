"""Spot columns that identify a row rather than describe it.

``glassbox`` classifies a column by dtype before it checks how unique the values are, so
a primary key stored as ``int64`` (1, 2, 3, … n) profiles as a perfectly ordinary number
and would be scaled into a feature. Text keys are caught by the free-text rule; numeric
ones are not.

Rather than change the profiling rules, this module finds those columns and hands them
to ``fit_preprocessing`` through its ``screeners`` argument. The reason travels with the
column, so the user still sees why it went.

Every screen here is a *measurement*, which is why they run inside
:func:`~app.services.glassbox.fit_preprocessing` on the training fold rather than on the
uploaded frame. A uniqueness rate computed over rows the model will later be tested on
is a decision informed by held-out data, however indirectly.

A screen returns two things, and the difference is deliberate:

* **drops** — columns that cannot be features, with a reason.
* **flags** — columns that are suspicious but stay. The target-leakage screen only
  flags: a column almost perfectly correlated with the target is usually a mistake in
  the export, but sometimes it is the whole point of the dataset, and silently deleting
  the user's most predictive column is the kind of help nobody asked for.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from app.config import (
    HIGH_CORRELATION,
    IDENTIFIER_UNIQUE_RATE,
    LEAKAGE_CORRELATION,
    MIN_ROWS_FOR_CORRELATION,
    NAMED_IDENTIFIER_UNIQUE_RATE,
)

__all__ = [
    "find_identifier_columns",
    "identifier_screen",
    "target_leakage_screen",
    "correlated_pairs",
]

_ID_NAME = re.compile(
    r"(^|[_\s])(id|ids|ref|reference|no|num|number|code|key|uuid|guid|index|idx|"
    r"sr|serial|roll|registration|regno|enrollment|admission)([_\s]|$)",
    re.IGNORECASE,
)


def _is_continuous(series: pd.Series) -> bool:
    """True for a float column holding genuine fractions.

    Whole numbers stored as floats (an id column that picked up a blank) are not
    continuous — those still count as candidate keys.
    """
    if not pd.api.types.is_float_dtype(series):
        return False
    values = series.dropna()
    if values.empty:
        return False
    return bool((values != values.round()).any())


def _looks_like_a_key_name(name: str) -> bool:
    cleaned = str(name).strip()
    return bool(_ID_NAME.search(cleaned)) or cleaned.lower().endswith("id")


def find_identifier_columns(df: pd.DataFrame, target: str | None = None) -> dict[str, str]:
    """Return ``{column: reason}`` for every column that identifies rather than explains.

    The reason is written for the user and is shown verbatim in the "columns we set
    aside" list.
    """
    n_rows = len(df)
    if n_rows == 0:
        return {}

    found: dict[str, str] = {}
    for name in df.columns:
        column = str(name)
        if target is not None and column == target:
            continue

        series = df[column]
        n_unique = int(series.nunique(dropna=True))
        if n_unique < 2:
            continue                       # a constant, handled by glassbox's own rule
        unique_rate = n_unique / n_rows
        named_like_a_key = _looks_like_a_key_name(column)

        # A continuous measurement is naturally almost all-distinct — a price, a weight, a
        # score with decimals. Uniqueness alone must never condemn one, or the screen eats
        # the most informative columns in the file. Keys are whole numbers or text.
        if _is_continuous(series):
            continue

        if unique_rate >= IDENTIFIER_UNIQUE_RATE:
            found[column] = (
                f"Every value in {column} is different, so it identifies the row rather "
                "than telling us anything that could explain the outcome"
            )
        elif named_like_a_key and unique_rate >= NAMED_IDENTIFIER_UNIQUE_RATE:
            found[column] = (
                f"{column} is named like a reference number and almost every value is "
                "unique, so it labels the row rather than describing it"
            )
    return found


def _numeric_view(series: pd.Series) -> pd.Series | None:
    """A numeric stand-in for a column, so correlation can be taken against anything.

    Numbers are used as-is. A two-value column becomes 0/1. Anything else has no
    meaningful linear correlation and returns ``None`` — inventing an ordering for
    unordered categories would manufacture a correlation that is not there.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    values = series.dropna().unique()
    if len(values) == 2:
        first = sorted(values, key=str)[0]
        return (series == first).astype(float).where(series.notna())
    return None


def target_leakage_screen(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[dict[str, str], list[dict]]:
    """Flag features that are almost a copy of the target.

    A correlation above :data:`LEAKAGE_CORRELATION` nearly always means the column was
    derived from the answer — a ``final_grade`` next to a ``passed`` target, an exit
    date next to ``churned``. A model trained on it scores beautifully and is useless,
    because the column will not exist at prediction time.

    Nothing is dropped. The spec's word is *flag*, and it is the right one: only the
    person who exported the file knows whether the column is a mistake or the point.
    """
    target_values = _numeric_view(y_train)
    if target_values is None:
        return {}, []

    flags: list[dict] = []
    for name in X_train.columns:
        column = str(name)
        feature_values = _numeric_view(X_train[column])
        if feature_values is None:
            continue

        pair = pd.concat([feature_values, target_values], axis=1).dropna()
        if len(pair) < MIN_ROWS_FOR_CORRELATION:
            continue
        if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
            continue

        correlation = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
        if np.isnan(correlation) or abs(correlation) < LEAKAGE_CORRELATION:
            continue

        flags.append({
            "kind": "target_leakage",
            "column": column,
            "correlation": round(correlation, 4),
            "severity": "warning",
            "reason": (
                f"{column} moves almost exactly in step with {y_train.name} "
                f"(correlation {correlation:+.3f}). That usually means it was worked out "
                f"from the answer rather than known beforehand — a model trained on it "
                f"will look excellent here and fail on real cases, because the column "
                f"will not exist yet when you need a prediction. It has been kept as a "
                f"feature: check whether you would really have it in advance."
            ),
        })

    return {}, flags


def correlated_pairs(df: pd.DataFrame, threshold: float = HIGH_CORRELATION) -> list[dict]:
    """Pairs of numeric columns that carry nearly the same information.

    Reported for the profile only — neither column is removed. Which of a redundant pair
    to keep is a judgement about the data's meaning, not something a threshold settles.
    """
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return []

    matrix = numeric.corr(numeric_only=True)
    found: list[dict] = []
    columns = list(matrix.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            value = matrix.loc[left, right]
            if pd.isna(value) or abs(float(value)) < threshold:
                continue
            found.append({
                "columns": [str(left), str(right)],
                "correlation": round(float(value), 4),
                "reason": (
                    f"{left} and {right} tell you almost the same thing "
                    f"(correlation {float(value):+.3f}). Keeping both is not harmful, but "
                    f"it splits one signal across two columns and makes each look less "
                    f"important than it is."
                ),
            })
    return sorted(found, key=lambda item: -abs(item["correlation"]))


def identifier_screen(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[dict[str, str], list[dict]]:
    """:data:`~app.services.glassbox.Screener` wrapper around :func:`find_identifier_columns`.

    ``y_train`` is unused — an identifier is recognisable from the feature alone — but
    the signature is shared by every screen so they are interchangeable in the list
    ``fit_preprocessing`` is given.
    """
    return find_identifier_columns(X_train), []
