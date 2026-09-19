"""Turn the fitted pipeline into cards a person with no ML background can read.

``glassbox.introspect`` already produces an accurate log, but it is written for someone
who knows what an imputer is. This module rewrites the same facts for the person the
product is actually for, and adds the thing that makes it concrete: a before/after strip
showing three real values from the user's own file changing.

The rules it follows:

* Every number shown is read from a fitted attribute or from the user's frame. Nothing
  is recomputed alongside the transformation, so the words cannot drift from the deed.
* Every card names the user's columns and says why, never what the library is called.
* Same input, same words — every sentence comes from a template keyed on the step.

One ordering decision worth stating. The split is shown as the **first** card, not the
last. The whole claim of the product is that nothing was learned before the data was
split, and a sequence that shows the split happening first is the honest depiction of
that. A closing card saying "everything above used only training rows" would read as an
afterthought and misstate the order.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from app.config import STRIP_WIDTH
from app.services.glassbox import PreprocessingResult

__all__ = ["build_steps", "build_dropped", "row_noun"]

_ROW_NOUNS: tuple[tuple[str, str], ...] = (
    ("student", "students"),
    ("pupil", "students"),
    ("customer", "customers"),
    ("client", "clients"),
    ("patient", "patients"),
    ("employee", "employees"),
    ("staff", "staff members"),
    ("applicant", "applicants"),
    ("passenger", "passengers"),
    ("order", "orders"),
    ("property", "properties"),
    ("house", "properties"),
    ("loan", "loans"),
    ("account", "accounts"),
)

_COUNT_WORDS = {
    1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
    6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten",
}


def row_noun(df: pd.DataFrame) -> str:
    """What one row of this file is, so sentences read naturally.

    Guessed from column names — a file with ``student_id`` and ``study_hours`` is about
    students. Falls back to "rows", which is always true if less warm.
    """
    joined = " ".join(str(c).lower() for c in df.columns)
    for needle, noun in _ROW_NOUNS:
        if needle in joined:
            return noun
    return "rows"


# --------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    """Format one data value for display. Blanks become an em dash."""
    if value is None:
        return "—"
    if isinstance(value, float) and math.isnan(value):
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if number == int(number) and abs(number) < 1e15:
            return f"{int(number):,}"
        if abs(number) >= 1000:
            return f"{number:,.0f}"
        if abs(number) >= 1:
            return f"{number:,.2f}".rstrip("0").rstrip(".")
        return f"{number:.3g}"
    return str(value)


def _count(n: int) -> str:
    """Small counts read better as words at the start of a sentence."""
    return _COUNT_WORDS.get(n, f"{n:,}")


def _quantity(n: int, singular: str, plural: str | None = None) -> str:
    return f"{n:,} {singular if n == 1 else (plural or singular + 's')}"


# --------------------------------------------------------------------------------------
# Picking the three rows for the strip
# --------------------------------------------------------------------------------------


def _pick_rows(
    series: pd.Series,
    prefer_missing: bool = False,
    prefer_variety: bool = False,
) -> list[Any]:
    """Choose up to three row labels for the before/after strip.

    When a step fills blanks, the strip has to contain a blank or it demonstrates
    nothing — so one missing row is placed in the middle, flanked by rows that did have
    a value. When a step encodes categories, three identical values would demonstrate
    just as little, so distinct ones are preferred. Selection is by position, so the same
    file always produces the same strip.
    """
    present = series.dropna().index.tolist()

    if prefer_variety:
        picked: list[Any] = []
        seen: set[str] = set()
        for label in present:
            key = str(series.at[label])
            if key in seen:
                continue
            seen.add(key)
            picked.append(label)
            if len(picked) == STRIP_WIDTH:
                return picked
        # Fewer distinct values than strip slots: pad with whatever is left.
        picked += [label for label in present if label not in picked]
        return picked[:STRIP_WIDTH]

    if not prefer_missing:
        return present[:STRIP_WIDTH] if present else series.index[:STRIP_WIDTH].tolist()

    absent = series[series.isna()].index.tolist()
    if not absent:
        return present[:STRIP_WIDTH]
    if len(present) < 2:
        return (present + absent)[:STRIP_WIDTH]
    return [present[0], absent[0], present[1]]


def _strip(
    labels: list[Any],
    before: list[str],
    after: list[str],
    changed: list[bool],
    column: str,
) -> dict[str, Any]:
    return {
        "column": column,
        "before": before,
        "after": after,
        "changed": changed,
        "rows": [str(label) for label in labels],
    }


# --------------------------------------------------------------------------------------
# One builder per stage
# --------------------------------------------------------------------------------------


def _fitted(result: PreprocessingResult, group: str, name: str) -> Any:
    return result.pipeline.named_transformers_[group].named_steps[name]


def _impute_numeric_card(
    entry: dict, train_raw: pd.DataFrame, noun: str
) -> dict[str, Any] | None:
    fills = entry["details"]["fill_values"]
    blanks = entry["details"]["blanks_filled"]
    affected = [c for c in entry["columns"] if blanks.get(c, 0) > 0]
    if not affected:
        return None  # nothing happened; a card saying so would waste the reader's time

    if len(affected) == 1:
        column = affected[0]
        sentence = (
            f"{_count(blanks[column])} {noun} had no {column} recorded. We filled those "
            f"gaps with {_fmt(fills[column])} — the middle value among the {noun} that "
            "did have one. The middle value is safer than the average, because a handful "
            "of unusually high or low numbers would drag an average off course."
        )
    else:
        parts = "; ".join(
            f"{column} — {_quantity(blanks[column], 'gap')} filled with {_fmt(fills[column])}"
            for column in affected
        )
        sentence = (
            f"{_count(len(affected))} of your columns had gaps. We filled each one with "
            f"its middle value, worked out from the training rows only: {parts}. The "
            "middle value is safer than the average, because a handful of unusually high "
            "or low numbers would drag an average off course."
        )

    lead = max(affected, key=lambda c: blanks[c])
    labels = _pick_rows(train_raw[lead], prefer_missing=True)
    raw_values = [train_raw.at[label, lead] for label in labels]
    fill = fills[lead]
    return {
        "affected": affected,
        "title": "Filled in missing numbers",
        "explanation": sentence,
        "detail": {c: {"filled": blanks[c], "value": fills[c]} for c in affected},
        "strip": _strip(
            labels,
            [_fmt(v) for v in raw_values],
            [_fmt(fill) if pd.isna(v) else _fmt(v) for v in raw_values],
            [bool(pd.isna(v)) for v in raw_values],
            lead,
        ),
    }


def _impute_categorical_card(
    entry: dict, train_raw: pd.DataFrame, noun: str
) -> dict[str, Any] | None:
    fills = entry["details"]["fill_values"]
    blanks = entry["details"]["blanks_filled"]
    affected = [c for c in entry["columns"] if blanks.get(c, 0) > 0]
    if not affected:
        return None

    if len(affected) == 1:
        column = affected[0]
        sentence = (
            f"{_count(blanks[column])} {noun} had no {column}. We used "
            f"{_fmt(fills[column])}, the most common one among the training rows — so "
            "every filled-in gap holds a label that really occurs in your data."
        )
    else:
        parts = "; ".join(
            f"{column} — {_quantity(blanks[column], 'gap')} filled with {_fmt(fills[column])}"
            for column in affected
        )
        sentence = (
            f"{_count(len(affected))} label columns had gaps. We filled each with its "
            f"most common value from the training rows: {parts}."
        )

    lead = max(affected, key=lambda c: blanks[c])
    labels = _pick_rows(train_raw[lead], prefer_missing=True)
    raw_values = [train_raw.at[label, lead] for label in labels]
    fill = fills[lead]
    return {
        "affected": affected,
        "title": "Filled in missing labels",
        "explanation": sentence,
        "detail": {c: {"filled": blanks[c], "value": fills[c]} for c in affected},
        "strip": _strip(
            labels,
            [_fmt(v) for v in raw_values],
            [_fmt(fill) if pd.isna(v) else _fmt(v) for v in raw_values],
            [bool(pd.isna(v)) for v in raw_values],
            lead,
        ),
    }


def _scale_card(
    entry: dict, result: PreprocessingResult, train_raw: pd.DataFrame
) -> dict[str, Any]:
    means = entry["details"]["mean"]
    scales = entry["details"]["scale"]
    columns = entry["columns"]

    # Name the widest and narrowest columns, because the contrast between them is the
    # reason this step exists. Both spreads are read from the fitted scaler.
    widest = max(columns, key=lambda c: abs(scales[c]))
    narrowest = min(columns, key=lambda c: abs(scales[c]))
    if widest != narrowest and scales[narrowest] > 0:
        ratio = scales[widest] / scales[narrowest]
        contrast = (
            f"{widest} varies about {ratio:,.0f} times more widely than {narrowest}. "
            "Left alone, a model would assume the bigger numbers matter more. "
        )
    else:
        contrast = "Columns measured on different scales pull a model around. "

    imputer = _fitted(result, "numeric", "impute")
    fills = dict(zip(imputer.feature_names_in_, imputer.statistics_))
    lead = widest
    labels = _pick_rows(train_raw[lead], prefer_missing=False)
    raw_values = [train_raw.at[label, lead] for label in labels]
    filled = [fills[lead] if pd.isna(v) else v for v in raw_values]
    centred = [(float(v) - means[lead]) / scales[lead] for v in filled]

    kind = entry["details"].get("scaler", "StandardScaler")
    what_it_did = {
        "StandardScaler": (
            "We shifted and squeezed each column so they all sit around zero and vary "
            "by a similar amount."
        ),
        "MinMaxScaler": (
            "We squeezed each column into the range 0 to 1, so the smallest value you "
            "recorded became 0 and the largest became 1."
        ),
        "RobustScaler": (
            "We centred each column on its middle value and scaled it by the spread of "
            "its middle half, so a few unusually large numbers cannot set the scale for "
            "everything else."
        ),
    }[kind]

    return {
        "title": "Put every number on the same footing",
        "explanation": (
            contrast
            + what_it_did
            + " The ordering inside each column is untouched — only the units change. "
            "The centre and spread were measured on the training rows alone."
        ),
        "detail": {c: {"centre": means[c], "spread": scales[c]} for c in columns},
        "strip": _strip(
            labels,
            [_fmt(v) for v in filled],
            [f"{value:+.2f}" for value in centred],
            [True] * len(labels),
            lead,
        ),
    }


def _binary_card(entry: dict, train_raw: pd.DataFrame) -> dict[str, Any]:
    categories = entry["details"]["categories"]
    columns = entry["columns"]

    def mapping_text(column: str) -> str:
        values = categories[column]
        return " and ".join(f"{_fmt(v)} became {i}" for i, v in enumerate(values))

    if len(columns) == 1:
        column = columns[0]
        sentence = (
            f"{column} only ever holds two answers, and models work with numbers, so "
            f"{mapping_text(column)}. Nothing is lost — two answers need only one column."
        )
    else:
        parts = "; ".join(f"{column} ({mapping_text(column)})" for column in columns)
        sentence = (
            f"{_count(len(columns))} columns hold just two answers each, and models work "
            f"with numbers, so each became a single 1-or-0 column: {parts}."
        )

    lead = columns[0]
    labels = _pick_rows(train_raw[lead], prefer_variety=True)
    codes = {str(value): index for index, value in enumerate(categories[lead])}
    raw_values = [train_raw.at[label, lead] for label in labels]
    return {
        "title": "Turned yes/no answers into 1 and 0",
        "explanation": sentence,
        "detail": {c: {"mapping": {str(v): i for i, v in enumerate(categories[c])}} for c in columns},
        "strip": _strip(
            labels,
            [_fmt(v) for v in raw_values],
            [str(codes.get(str(v), "—")) for v in raw_values],
            [True] * len(labels),
            lead,
        ),
    }


def _onehot_card(
    entry: dict, result: PreprocessingResult, train_raw: pd.DataFrame
) -> dict[str, Any]:
    categories = entry["details"]["categories"]
    widths = entry["details"]["columns_created"]
    columns = entry["columns"]
    total = entry["details"]["total_columns_created"]

    if len(columns) == 1:
        column = columns[0]
        listed = ", ".join(_fmt(v) for v in categories[column][:4])
        more = "" if len(categories[column]) <= 4 else ", and so on"
        sentence = (
            f"{column} holds words ({listed}{more}), and models work with numbers. "
            f"We made {widths[column]} yes/no columns, one per value, so the model can "
            f"tell them apart without wrongly treating them as ranked — nothing here "
            f"says one is bigger or better than another."
        )
    else:
        parts = "; ".join(f"{c} into {widths[c]}" for c in columns)
        sentence = (
            f"{_count(len(columns))} columns hold words rather than numbers. We turned "
            f"them into {total} yes/no columns — {parts} — so the model can tell the "
            "values apart without wrongly treating them as ranked."
        )

    imputer = _fitted(result, "low_card", "impute")
    fills = dict(zip(imputer.feature_names_in_, imputer.statistics_))
    lead = max(columns, key=lambda c: widths[c])
    labels = _pick_rows(train_raw[lead], prefer_variety=True)
    raw_values = [train_raw.at[label, lead] for label in labels]
    filled = [fills[lead] if pd.isna(v) else v for v in raw_values]

    return {
        "title": "Made a yes/no column for each value",
        "explanation": sentence,
        "detail": {c: {"values": categories[c], "columns_made": widths[c]} for c in columns},
        "strip": _strip(
            labels,
            [_fmt(v) for v in filled],
            [f"{_fmt(v)} = 1 · rest 0" for v in filled],
            [True] * len(labels),
            lead,
        ),
    }


def _frequency_card(
    entry: dict, result: PreprocessingResult, train_raw: pd.DataFrame
) -> dict[str, Any]:
    frequencies = entry["details"]["frequencies"]
    counts = entry["details"]["n_categories"]
    columns = entry["columns"]
    lead = max(columns, key=lambda c: counts[c])

    sentence = (
        f"{lead} has {counts[lead]} different values — far too many to give each one its "
        "own column. Instead we replaced every value with how often it appears in the "
        "training rows. This uses nothing at all about the thing you are predicting, so "
        "it cannot accidentally hand the model the answer."
    )
    if len(columns) > 1:
        others = ", ".join(f"{c} ({counts[c]} values)" for c in columns if c != lead)
        sentence += f" The same was done for {others}."

    imputer = _fitted(result, "high_card", "impute")
    fills = dict(zip(imputer.feature_names_in_, imputer.statistics_))
    labels = _pick_rows(train_raw[lead], prefer_variety=True)
    raw_values = [train_raw.at[label, lead] for label in labels]
    filled = [fills[lead] if pd.isna(v) else v for v in raw_values]
    table = frequencies[lead]

    return {
        "title": "Replaced crowded labels with how often they appear",
        "explanation": sentence,
        "detail": {c: {"values": counts[c], "example": dict(list(frequencies[c].items())[:5])}
                   for c in columns},
        "strip": _strip(
            labels,
            [_fmt(v) for v in filled],
            [f"{table.get(str(v), 0.0):.3f}" for v in filled],
            [True] * len(labels),
            lead,
        ),
    }


def _target_encode_card(
    entry: dict, result: PreprocessingResult, train_raw: pd.DataFrame
) -> dict[str, Any]:
    counts = entry["details"]["n_categories"]
    encodings = entry["details"]["encodings"]
    average = entry["details"]["target_mean"]
    columns = entry["columns"]
    lead = max(columns, key=lambda c: counts[c])

    sentence = (
        f"{lead} has {counts[lead]} different values — far too many to give each one "
        f"its own column. Instead each value was replaced by the average outcome "
        f"recorded against it, next to an overall average of {_fmt(average)}. Because "
        f"this does read the answer column, it is worked out in "
        f"{entry['details']['cv']} rotating folds: the number a row is given is "
        f"computed from other rows only, never from itself. A value never seen before "
        f"falls back to the overall average."
    )
    if len(columns) > 1:
        others = ", ".join(f"{c} ({counts[c]} values)" for c in columns if c != lead)
        sentence += f" The same was done for {others}."

    imputer = _fitted(result, "high_card", "impute")
    fills = dict(zip(imputer.feature_names_in_, imputer.statistics_))
    labels = _pick_rows(train_raw[lead], prefer_variety=True)
    raw_values = [train_raw.at[label, lead] for label in labels]
    filled = [fills[lead] if pd.isna(v) else v for v in raw_values]
    table = encodings[lead]

    return {
        "title": "Replaced crowded labels with their average outcome",
        "explanation": sentence,
        "detail": {
            c: {"values": counts[c], "example": dict(list(encodings[c].items())[:5])}
            for c in columns
        },
        "strip": _strip(
            labels,
            [_fmt(v) for v in filled],
            [_fmt(table.get(str(v), average)) for v in filled],
            [True] * len(labels),
            lead,
        ),
    }


def _datetime_card(entry: dict, train_raw: pd.DataFrame) -> dict[str, Any]:
    columns = entry["columns"]
    created = entry["details"]["columns_created"]
    total = sum(created.values())
    lead = max(columns, key=lambda c: created[c])

    if len(columns) == 1:
        opening = f"{lead} holds dates, which a model cannot use as they are."
    else:
        opening = (
            f"{_count(len(columns))} of your columns hold dates, which a model cannot "
            "use as they are."
        )
    sentence = (
        f"{opening} A date is one huge number, and its size means nothing: two events "
        f"in the same week of December a year apart look far apart, while nothing marks "
        f"them as both being just before the holidays. We split each date into its "
        f"parts — year, month, day, day of the week and hour — giving {total} columns "
        f"the model can actually learn from. The repeating parts are written twice over, "
        f"as a pair of wave values, which is what puts December next to January instead "
        f"of eleven months away. Nothing here is learned from your data: every value "
        f"depends only on the date in that row."
    )

    labels = _pick_rows(train_raw[lead], prefer_variety=True)
    raw_values = [train_raw.at[label, lead] for label in labels]
    stamps = pd.to_datetime(pd.Series(raw_values), errors="coerce")

    def described(stamp: Any) -> str:
        if pd.isna(stamp):
            return "—"
        return f"{stamp.strftime('%b')} · {stamp.strftime('%a')} · {stamp.year}"

    return {
        "title": "Broke dates into parts a model can use",
        "explanation": sentence,
        "detail": {
            c: {"columns_made": created[c], "parts": entry["details"]["parts"]}
            for c in columns
        },
        "strip": _strip(
            labels,
            [_fmt(v) if pd.isna(v) else str(pd.Timestamp(v).date()) for v in raw_values],
            [described(stamp) for stamp in stamps],
            [True] * len(labels),
            lead,
        ),
    }


def _split_card(result: PreprocessingResult, noun: str) -> dict[str, Any]:
    train_rows = len(result.X_train)
    test_rows = len(result.X_test)
    share = round(test_rows / max(train_rows + test_rows, 1) * 100)
    return {
        "title": "Set aside rows we would not look at",
        "explanation": (
            f"Before changing anything, we put {test_rows:,} {noun} — {share}% of your "
            f"file — to one side and did not look at them again. Every number in the "
            f"steps below was worked out from the other {train_rows:,} {noun} only. "
            "That is what makes a later test honest: the model gets marked on rows it "
            "has genuinely never seen."
        ),
        "detail": {"train_rows": train_rows, "test_rows": test_rows, "test_percent": share},
        "strip": None,
        "split": {"train": train_rows, "test": test_rows, "percent": share},
    }


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------

_STAGE_LABEL = {
    "split": "split",
    "impute": "impute",
    "scale": "scale",
    "encode": "encode",
    "derive": "derive",
}


def build_steps(raw: pd.DataFrame, result: PreprocessingResult) -> list[dict[str, Any]]:
    """Build the ordered step cards shown to the user.

    Imputation steps that filled nothing are left out: a card reading "we changed no
    values" costs the reader four seconds and tells them nothing. The technical log
    still records them, and the exported script still contains the step.
    """
    train_raw = raw.loc[result.X_train.index]
    noun = row_noun(raw)

    cards: list[dict[str, Any]] = []
    split = _split_card(result, noun)
    split.update({"stage": "split", "columns": [], "fitted": False})
    cards.append(split)

    for entry in result.log:
        stage = entry["stage"]
        transformer = entry["transformer"]

        if stage == "impute":
            numeric = entry["params"].get("strategy") == "median"
            card = (_impute_numeric_card if numeric else _impute_categorical_card)(
                entry, train_raw, noun
            )
        elif stage == "scale":
            card = _scale_card(entry, result, train_raw)
        elif stage == "derive":
            card = _datetime_card(entry, train_raw)
        elif transformer == "TargetEncoder":
            card = _target_encode_card(entry, result, train_raw)
        elif transformer == "OrdinalEncoder":
            card = _binary_card(entry, train_raw)
        elif transformer == "OneHotEncoder":
            card = _onehot_card(entry, result, train_raw)
        elif transformer == "FrequencyEncoder":
            card = _frequency_card(entry, result, train_raw)
        else:  # pragma: no cover - every transformer above is accounted for
            continue

        if card is None:
            continue
        # An impute card names only the columns it actually filled; listing a column
        # that had no gaps would have the user hunting for a change that never happened.
        shown = list(card.pop("affected", entry["columns"]))
        card.update({
            "stage": _STAGE_LABEL[stage],
            "columns": shown,
            # The badge means "this step learned something, and only from training
            # rows". Splitting a date into its parts learns nothing — every output
            # depends on that row alone — so claiming the badge there would dilute it
            # everywhere it is true.
            "fitted": stage != "derive",
        })
        cards.append(card)

    for number, card in enumerate(cards, start=1):
        card["n"] = number
    return cards


def build_dropped(
    result: PreprocessingResult,
    raw: pd.DataFrame,
    identifier_reasons: dict[str, str],
) -> list[dict[str, str]]:
    """Rewrite each dropped column's reason for a non-technical reader."""
    n_rows = len(raw)
    out: list[dict[str, str]] = []

    for column, reason in result.plan.dropped.items():
        if column == result.plan.target:
            continue

        if column in identifier_reasons:
            text = identifier_reasons[column] + ". Removed."
        elif reason.startswith("constant"):
            series = raw[column] if column in raw.columns else None
            value = _fmt(series.dropna().iloc[0]) if series is not None and series.notna().any() else "the same value"
            text = (
                f"{column} holds {value} in all {n_rows:,} rows. A column that never "
                "changes cannot explain anything that does. Removed."
            )
        elif reason.startswith("free text"):
            distinct = int(raw[column].nunique()) if column in raw.columns else 0
            text = (
                f"Every value in {column} is different ({distinct:,} of them), so it "
                "identifies the row rather than describing it. Removed."
            )
        elif "missing" in reason:
            share = raw[column].isna().mean() * 100 if column in raw.columns else 0.0
            text = (
                f"{share:.0f}% of {column} is blank. Filling in that much would be "
                "inventing data rather than using yours. Removed."
            )
        else:
            text = reason if reason.endswith(".") else reason + "."

        out.append({"column": column, "reason": text})

    return out
