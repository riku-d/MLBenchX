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
* **flags** — columns worth warning about, whether or not they were dropped.

Target leakage produces **both**: the column is excluded and a ``critical`` flag is
raised. An earlier version only flagged, on the reasoning that deleting someone's most
predictive column is unwelcome help. That reasoning was wrong, and the asymmetry is
why. If the column really is legitimate, excluding it costs one click to put back — the
flag carries ``can_reinclude`` and names the column. If it is derived from the answer
and we merely mention it, the model scores beautifully, ships, and fails on live data,
and the warning that would have prevented it was one line in a panel nobody re-read.
The cheap mistake is recoverable; the expensive one is not.

Two independent detectors run, because they catch different shapes of the same problem:

* **Correlation** finds a linear copy — a rescaled or rounded target.
* **A depth-2 decision tree, cross-validated** finds a non-linear one: a column that
  splits the target almost perfectly without correlating with it at all, such as a
  status string or a bucketed score.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.config import (
    HIGH_CORRELATION,
    IDENTIFIER_UNIQUE_RATE,
    LEAKAGE_CORRELATION,
    LEAKAGE_TREE_CV_FOLDS,
    LEAKAGE_TREE_DEPTH,
    LEAKAGE_TREE_SCORE,
    MIN_ROWS_FOR_CORRELATION,
    NAMED_IDENTIFIER_UNIQUE_RATE,
    SEED,
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


def _is_classification_target(y: pd.Series) -> bool:
    """Whether to probe with a classifier or a regressor."""
    if not pd.api.types.is_numeric_dtype(y):
        return True
    return int(y.nunique(dropna=True)) <= 20


def _probe_score(feature: pd.Series, y: pd.Series) -> float | None:
    """How well this one column predicts the target, alone, in cross-validation.

    A depth-2 tree on a single feature. Depth two can express "this column separates the
    answer" — the question being asked — while being far too small to memorise the rows,
    which is what would make every column look like a leak.

    Scored against what a constant guess achieves, not against zero: on a 95/5 target a
    useless column still scores 0.95 accuracy, and comparing to the majority-class
    baseline is what separates "predicts the target" from "the target is lopsided".
    Returns a value on that adjusted scale, or ``None`` when the column cannot be
    probed.
    """
    frame = pd.concat([feature, y], axis=1).dropna()
    if len(frame) < MIN_ROWS_FOR_CORRELATION:
        return None

    values, labels = frame.iloc[:, [0]], frame.iloc[:, 1]
    if values.iloc[:, 0].nunique() < 2 or labels.nunique() < 2:
        return None

    classification = _is_classification_target(labels)
    if classification:
        # Every class needs at least one row per fold, or the split is impossible.
        folds = min(LEAKAGE_TREE_CV_FOLDS, int(labels.value_counts().min()))
        if folds < 2:
            return None
        model = DecisionTreeClassifier(max_depth=LEAKAGE_TREE_DEPTH, random_state=SEED)
        scoring = "accuracy"
    else:
        folds = LEAKAGE_TREE_CV_FOLDS
        model = DecisionTreeRegressor(max_depth=LEAKAGE_TREE_DEPTH, random_state=SEED)
        scoring = "r2"

    try:
        score = float(cross_val_score(model, values, labels, cv=folds, scoring=scoring).mean())
    except ValueError:
        # Degenerate splits — not enough of some class to cross-validate. Not a leak,
        # just not answerable.
        return None

    if not classification:
        return score  # r2 is already "better than predicting the mean"

    baseline = float(
        cross_val_score(
            DummyClassifier(strategy="most_frequent"), values, labels,
            cv=folds, scoring=scoring,
        ).mean()
    )
    if baseline >= 1.0:
        return None
    # Rescale so 0 is "no better than always guessing the majority" and 1 is perfect.
    return (score - baseline) / (1.0 - baseline)


def target_leakage_screen(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[dict[str, str], list[dict]]:
    """Find features that are copies of the target, exclude them, and say so loudly.

    Two detectors, because they catch different shapes:

    * **Correlation** above :data:`LEAKAGE_CORRELATION` — a linear copy: the target
      rescaled, rounded, or offset.
    * **A depth-2 tree scoring above** :data:`LEAKAGE_TREE_SCORE` **in cross-validation**
      — a non-linear one: a status string, a bucketed score, anything that splits the
      answer cleanly while correlating with it barely at all.

    Either is enough. The column is **excluded** and flagged ``critical``, with
    ``can_reinclude`` set so the interface can offer it back. Excluding a legitimate
    column costs one click; keeping a leaked one costs a model that works in testing and
    fails in production.

    Runs on the training fold only, like every other screen here.
    """
    flags: list[dict] = []
    drops: dict[str, str] = {}
    target_values = _numeric_view(y_train)
    target_name = y_train.name or "the target"

    for name in X_train.columns:
        column = str(name)
        series = X_train[column]

        # --- linear copy ---------------------------------------------------------
        correlation: float | None = None
        feature_values = _numeric_view(series)
        if feature_values is not None and target_values is not None:
            pair = pd.concat([feature_values, target_values], axis=1).dropna()
            if (
                len(pair) >= MIN_ROWS_FOR_CORRELATION
                and pair.iloc[:, 0].nunique() >= 2
                and pair.iloc[:, 1].nunique() >= 2
            ):
                value = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
                correlation = None if np.isnan(value) else value

        by_correlation = correlation is not None and abs(correlation) >= LEAKAGE_CORRELATION

        # --- non-linear copy -----------------------------------------------------
        # The tree needs numbers. Use the 0/1 view when the column has one, otherwise
        # category codes — the tree only needs to be able to split on the values and
        # does not care that the ordering is arbitrary. Passing the raw column through
        # made the probe silently return None for every two-value string column, so the
        # tree detector never ran on exactly the shape it was best at catching.
        if pd.api.types.is_numeric_dtype(series):
            probe_input = series
        elif feature_values is not None:
            probe_input = feature_values
        else:
            probe_input = series.astype("category").cat.codes.where(series.notna())
        score = _probe_score(probe_input.rename(column), y_train)
        by_tree = score is not None and score >= LEAKAGE_TREE_SCORE

        if not (by_correlation or by_tree):
            continue

        if by_correlation and by_tree:
            evidence = (
                f"it moves almost exactly in step with {target_name} "
                f"(correlation {correlation:+.3f}) and, on its own, predicts it in "
                f"cross-validation ({score:.3f})"
            )
        elif by_correlation:
            evidence = (
                f"it moves almost exactly in step with {target_name} "
                f"(correlation {correlation:+.3f})"
            )
        else:
            evidence = (
                f"on its own it predicts {target_name} almost perfectly in "
                f"cross-validation ({score:.3f}), even though it does not correlate "
                f"with it"
            )

        reason = (
            f"{column} looks like it was derived from the answer: {evidence}. "
            f"A model trained on it will score excellently here and fail on real cases, "
            f"because the value will not exist yet at the moment you need a prediction. "
            f"It has been left out. If you will genuinely have it in advance, put it back."
        )

        drops[column] = reason
        flags.append({
            "kind": "target_leakage",
            "column": column,
            "correlation": None if correlation is None else round(correlation, 4),
            "probe_score": None if score is None else round(score, 4),
            "detected_by": (
                "correlation and single-feature tree" if by_correlation and by_tree
                else "correlation" if by_correlation
                else "single-feature tree"
            ),
            "severity": "critical",
            "excluded": True,
            "can_reinclude": True,
            "reason": reason,
        })

    return drops, flags


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
