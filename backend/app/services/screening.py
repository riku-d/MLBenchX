"""Spot columns that identify a row rather than describe it.

``glassbox`` classifies a column by dtype before it checks how unique the values are, so
a primary key stored as ``int64`` (1, 2, 3, … n) profiles as a perfectly ordinary number
and would be scaled into a feature. Text keys are caught by the free-text rule; numeric
ones are not.

Rather than change the profiling rules, this module finds those columns up front and
hands them to ``fit_preprocessing`` through its ``excluded`` argument — the same channel
a leakage screen or a user deselection would use. The reason travels with the column, so
the user still sees why it went.
"""

from __future__ import annotations

import re

import pandas as pd

__all__ = ["IDENTIFIER_UNIQUE_RATE", "find_identifier_columns"]

#: A column this close to entirely distinct is an identifier, whatever its dtype. Set
#: high on purpose: a genuine measurement like income can be 60% distinct.
IDENTIFIER_UNIQUE_RATE = 0.98

#: A weaker bar applies when the column is *named* like a key.
NAMED_IDENTIFIER_UNIQUE_RATE = 0.90

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
        elif (
            named_like_a_key
            and pd.api.types.is_integer_dtype(series)
            and series.is_monotonic_increasing
            and n_unique == n_rows
        ):
            found[column] = (
                f"{column} counts up one row at a time, so it is a row number rather "
                "than a measurement"
            )

    return found
