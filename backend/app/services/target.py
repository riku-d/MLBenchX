"""Work out which column the user wants to predict, from what they typed.

No model and no API call — a four-tier cascade of string matching, tried in order of how
much confidence each tier deserves. The result is always shown back to the user with a
way to change it, because a silent wrong guess is worse than an obvious one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

__all__ = ["TargetGuess", "detect_target", "describe_task", "summarise_values"]

#: Words a user is likely to type, mapped to fragments that appear in column names.
_KEYWORD_MAP: dict[str, tuple[str, ...]] = {
    "fail": ("fail", "backlog", "arrear", "pass", "result", "outcome", "grade"),
    "backlog": ("backlog", "arrear", "fail"),
    "pass": ("pass", "result", "outcome", "grade", "fail"),
    "drop": ("dropout", "drop_out", "dropped", "attrition", "withdraw"),
    "dropout": ("dropout", "drop_out", "dropped", "attrition"),
    "churn": ("churn", "attrition", "left", "cancelled", "retained"),
    "leave": ("churn", "attrition", "left", "exit"),
    "default": ("default", "repaid", "repay", "delinquent", "charge_off"),
    "repay": ("repaid", "repay", "default"),
    "place": ("placed", "placement", "hired", "job", "offer"),
    "placed": ("placed", "placement", "hired", "offer"),
    "hire": ("hired", "placed", "offer", "recruit"),
    "survive": ("survived", "survival", "alive"),
    "price": ("price", "cost", "value", "amount", "fare"),
    "cost": ("cost", "price", "amount", "charge"),
    "salary": ("salary", "wage", "pay", "income", "compensation"),
    "amount": ("amount", "total", "value", "sum"),
    "score": ("score", "mark", "grade", "rating", "gpa"),
    "revenue": ("revenue", "sales", "turnover", "income"),
    "buy": ("purchase", "bought", "converted", "order"),
    "click": ("click", "clicked", "converted"),
    "spam": ("spam", "is_spam", "label"),
    "risk": ("risk", "default", "dropout"),
}

_STOPWORDS = frozenset({
    "predict", "predicting", "prediction", "forecast", "estimate", "find", "out",
    "which", "who", "what", "whether", "will", "would", "the", "a", "an", "of", "for",
    "is", "are", "be", "going", "to", "my", "their", "his", "her", "students",
    "student", "customers", "customer", "people", "users", "user", "and", "or", "in",
    "on", "at", "by", "from", "with", "want", "i", "we", "likely", "chance", "risk",
    "not", "each", "every", "them", "it", "that", "this", "how", "much", "many",
})


@dataclass
class TargetGuess:
    column: str
    inferred: bool
    confidence: str          # "exact" | "normalised" | "keyword" | "fallback" | "user"
    note: str                # short sentence explaining the choice, shown in the UI


def _normalise(text: str) -> str:
    """Lowercase and strip anything that is not a letter or digit."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t and t not in _STOPWORDS]


def _candidate_score(series: pd.Series, position: int, total: int) -> float:
    """How plausible a column is as a prediction target, absent any hint from the user.

    Rewards a small set of distinct values, few blanks, and sitting near the end of the
    file — the three things that are true of almost every real label column.
    """
    n_unique = int(series.nunique(dropna=True))
    if n_unique < 2:
        return -1.0
    score = 0.0
    if 2 <= n_unique <= 20:
        score += 3.0
        if n_unique == 2:
            score += 1.5          # a binary label is the most common case by far
    elif pd.api.types.is_numeric_dtype(series):
        score += 1.0              # a continuous target is plausible, just less likely
    score -= float(series.isna().mean()) * 3.0
    score += (position / max(total - 1, 1)) * 2.0    # labels usually sit last
    return score


def detect_target(
    df: pd.DataFrame,
    goal: str,
    explicit: str | None = None,
) -> TargetGuess:
    """Resolve a free-text goal to a column name.

    Tiers, tried in order: an explicit choice from the user, an exact column name in the
    text, a match once underscores and case are ignored, a keyword association, and
    finally the most label-shaped column in the file.
    """
    columns = [str(c) for c in df.columns]
    if not columns:
        raise ValueError("the file has no columns")

    if explicit:
        if explicit in columns:
            return TargetGuess(explicit, False, "user", "You chose this column.")
        match = next((c for c in columns if _normalise(c) == _normalise(explicit)), None)
        if match:
            return TargetGuess(match, False, "user", "You chose this column.")
        raise ValueError(
            f"We couldn't find a column called '{explicit}'. Pick one from the list."
        )

    goal_text = goal or ""
    lowered = goal_text.lower()

    # Tier 1 — a column name appears verbatim in what the user typed.
    verbatim = [c for c in columns if c.lower() in lowered and len(c) > 2]
    if verbatim:
        best = max(verbatim, key=len)
        return TargetGuess(best, True, "exact", "You named this column in your goal.")

    # Tier 2 — same, ignoring underscores, spaces and case.
    normalised_goal = _normalise(goal_text)
    for column in sorted(columns, key=len, reverse=True):
        token = _normalise(column)
        if len(token) > 3 and token in normalised_goal:
            return TargetGuess(
                column, True, "normalised", "This column matches the words in your goal."
            )

    # Tier 3 — a keyword in the goal points at a fragment of a column name.
    for word in _tokens(goal_text):
        fragments = _KEYWORD_MAP.get(word)
        if not fragments:
            continue
        for fragment in fragments:
            for column in columns:
                if fragment in _normalise(column):
                    return TargetGuess(
                        column, True, "keyword",
                        f"'{word}' in your goal points at this column.",
                    )

    # Tier 4 — nothing matched, so pick the most label-shaped column.
    total = len(columns)
    ranked = sorted(
        ((_candidate_score(df[c], i, total), i, c) for i, c in enumerate(columns)),
        key=lambda item: (-item[0], item[1]),
    )
    best_score, _, best_column = ranked[0]
    if best_score < 0:
        raise ValueError(
            "We couldn't work out what to predict from that. Pick a column from the list."
        )
    return TargetGuess(
        best_column, True, "fallback",
        "We couldn't match your wording to a column, so we picked the most likely one.",
    )


def describe_task(series: pd.Series) -> str:
    """Name the prediction task in words the user will recognise."""
    n_unique = int(series.nunique(dropna=True))
    if n_unique == 2:
        return "binary classification"
    if pd.api.types.is_numeric_dtype(series) and n_unique > 20:
        return "regression"
    if 3 <= n_unique <= 20:
        return "multiclass classification"
    return "regression"


def summarise_values(series: pd.Series, limit: int = 8) -> dict[str, int]:
    """Value counts for a label column, for the "what we found" band."""
    counts = series.value_counts(dropna=True).head(limit)
    return {str(key): int(value) for key, value in counts.items()}
