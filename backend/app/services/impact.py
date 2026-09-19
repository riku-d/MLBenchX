"""What turning a step off would actually cost.

The toggle warning is the most important piece of writing in this product. A user about
to disable outlier treatment is asking a real question — "do I need this?" — and the
answer depends entirely on their data. "This may affect your results" answers nothing
and teaches nothing; it is the kind of warning people learn to click through.

So every consequence here is computed from the dataset in front of the user and stated
with its number: *1,284 blanks will remain in 3 columns, and most scikit-learn
estimators raise on the first one*. If the dataset has no blanks, the warning says the
step is doing nothing and disabling it is free — which is just as useful and far rarer
to be told.

Two kinds of number appear in an impact report, and they are not equally certain:

* **Facts** — how many rows, columns and values the step currently touches. These are
  measured directly from the dataset and are exact.
* **Projection** — what the health score would become. That requires running the
  pipeline both ways, which the caller does; this module states the facts and leaves a
  slot for the measured result rather than guessing at one.

Nothing here is rhetorical. A step that genuinely does not matter for this dataset is
reported as not mattering.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.config import (
    HIGH_CORRELATION,
    MAX_MISSING_RATE,
    OUTLIER_IQR_MULTIPLIER,
)
from app.pipeline import STEP_LABELS, PipelineConfig

__all__ = ["assess_disable", "step_facts", "ImpactReport"]

ImpactReport = dict[str, Any]

#: How serious it is to switch a step off, given what it is currently doing.
#:
#: ``blocking`` is reserved for the case where downstream code will actually raise —
#: not "results may be worse", but "this will not run". Using it for anything softer
#: would make it worthless the first time a user disabled a step and nothing broke.
_SEVERITIES = ("none", "low", "moderate", "high", "blocking")


def _numeric_columns(df: pd.DataFrame, target: str | None) -> list[str]:
    return [
        c for c in df.select_dtypes(include="number").columns if c != target
    ]


def _outlier_counts(df: pd.DataFrame, target: str | None, multiplier: float) -> dict[str, int]:
    """Values outside Tukey's fences, per numeric column."""
    counts: dict[str, int] = {}
    for column in _numeric_columns(df, target):
        values = df[column].dropna().astype(float)
        if values.empty:
            continue
        q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            continue
        low, high = q1 - multiplier * iqr, q3 + multiplier * iqr
        count = int(((values < low) | (values > high)).sum())
        if count:
            counts[str(column)] = count
    return counts


def step_facts(df: pd.DataFrame, config: PipelineConfig, step_id: str) -> dict[str, Any]:
    """Exactly what this step is doing to this dataset, right now.

    Measured, not estimated. These are the numbers the step card shows while the step is
    enabled, and the numbers the warning quotes when the user moves to disable it.
    """
    target = config.target

    if step_id == "missing_values":
        gaps = {
            str(c): int(df[c].isna().sum())
            for c in df.columns
            if c != target and df[c].isna().any()
        }
        too_empty = [
            str(c) for c in df.columns
            if c != target and df[c].isna().mean() > config.missing_values.drop_column_threshold
        ]
        rows_with_gaps = int(df.drop(columns=[target], errors="ignore").isna().any(axis=1).sum())
        return {
            "values_affected": sum(gaps.values()),
            "columns_affected": len(gaps),
            "rows_affected": rows_with_gaps,
            "per_column": dict(sorted(gaps.items(), key=lambda kv: -kv[1])[:8]),
            "columns_to_drop": too_empty,
        }

    if step_id == "duplicates":
        subset = config.duplicates.subset or None
        mask = df.duplicated(subset=subset, keep=config.duplicates.keep)
        return {
            "rows_affected": int(mask.sum()),
            "columns_affected": 0,
            "values_affected": int(mask.sum()) * int(df.shape[1]),
            "share": round(float(mask.mean()), 4),
        }

    if step_id == "outliers":
        counts = _outlier_counts(df, target, config.outliers.iqr_multiplier)
        columns = config.outliers.columns or _numeric_columns(df, target)
        counts = {c: n for c, n in counts.items() if c in columns}
        return {
            "values_affected": sum(counts.values()),
            "columns_affected": len(counts),
            "rows_affected": sum(counts.values()),
            "per_column": dict(sorted(counts.items(), key=lambda kv: -kv[1])[:8]),
        }

    if step_id == "encoding":
        categorical = [
            str(c) for c in df.columns
            if c != target
            and not pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_datetime64_any_dtype(df[c])
        ]
        widths = {c: int(df[c].nunique(dropna=True)) for c in categorical}
        onehot = {c: n for c, n in widths.items() if 2 < n <= config.encoding.max_onehot_cardinality}
        return {
            "columns_affected": len(categorical),
            "values_affected": int(sum(df[c].notna().sum() for c in categorical)) if categorical else 0,
            "rows_affected": len(df) if categorical else 0,
            "per_column": dict(sorted(widths.items(), key=lambda kv: -kv[1])[:8]),
            "columns_created": sum(onehot.values()),
        }

    if step_id == "scaling":
        columns = [
            c for c in (config.scaling.columns or _numeric_columns(df, target))
            if c in df.columns
        ]
        spreads = {
            str(c): float(df[c].std())
            for c in columns
            if df[c].notna().any() and float(df[c].std() or 0) > 0
        }
        disparity = (max(spreads.values()) / min(spreads.values())) if len(spreads) > 1 else 1.0
        widest = max(spreads, key=lambda k: spreads[k]) if spreads else None
        narrowest = min(spreads, key=lambda k: spreads[k]) if spreads else None
        return {
            "columns_affected": len(columns),
            "values_affected": int(sum(df[c].notna().sum() for c in columns)) if columns else 0,
            "rows_affected": len(df) if columns else 0,
            "scale_disparity": round(disparity, 1),
            "widest_column": widest,
            "narrowest_column": narrowest,
        }

    if step_id == "feature_selection":
        numeric = df[_numeric_columns(df, target)]
        constant = [str(c) for c in numeric.columns if numeric[c].nunique(dropna=True) <= 1]
        redundant: list[list[str]] = []
        if numeric.shape[1] > 1:
            matrix = numeric.corr().abs()
            names = list(matrix.columns)
            for i, left in enumerate(names):
                for right in names[i + 1:]:
                    value = matrix.loc[left, right]
                    threshold = config.feature_selection.correlation_threshold or HIGH_CORRELATION
                    if pd.notna(value) and float(value) >= threshold:
                        redundant.append([str(left), str(right)])
        return {
            "columns_affected": len(constant) + len(redundant),
            "values_affected": 0,
            "rows_affected": 0,
            "constant_columns": constant,
            "redundant_pairs": redundant[:8],
        }

    if step_id == "imbalance":
        if not target or target not in df.columns:
            return {"columns_affected": 0, "rows_affected": 0, "values_affected": 0}
        counts = df[target].value_counts()
        if len(counts) < 2:
            return {"columns_affected": 0, "rows_affected": 0, "values_affected": 0}
        ratio = float(counts.iloc[0] / max(counts.iloc[-1], 1))
        return {
            "columns_affected": 1,
            "rows_affected": int(counts.iloc[-1]),
            "values_affected": 0,
            "ratio": round(ratio, 2),
            "majority": str(counts.index[0]),
            "minority": str(counts.index[-1]),
        }

    if step_id == "data_types":
        text_numeric = 0
        text_dates = 0
        for column in df.columns:
            if df[column].dtype != object:
                continue
            values = df[column].dropna().astype("string")
            if values.empty:
                continue
            if pd.to_numeric(values.str.strip(), errors="coerce").notna().all():
                text_numeric += 1
        return {
            "columns_affected": int(df.shape[1]),
            "rows_affected": len(df),
            "values_affected": 0,
            "text_numeric_columns": text_numeric,
            "text_date_columns": text_dates,
        }

    return {"columns_affected": 0, "rows_affected": 0, "values_affected": 0}


def _consequences(step_id: str, facts: dict[str, Any], config: PipelineConfig) -> tuple[list[str], str]:
    """The sentences shown in the warning, and how serious this is.

    Written from the numbers. Each returns the consequences in the order a user would
    care about them: what breaks, then what degrades, then what merely gets untidy.
    """
    if step_id == "missing_values":
        blanks = facts["values_affected"]
        if blanks == 0:
            return (
                ["This dataset has no missing values, so this step is currently doing nothing."],
                "none",
            )
        columns = facts["columns_affected"]
        lines = [
            f"{blanks:,} missing value(s) will remain, across {columns} column(s).",
            "Most scikit-learn estimators raise on the first blank they meet, so training "
            "will fail rather than degrade.",
            "Any statistic computed downstream — a mean, a correlation — will silently "
            "skip those rows and be computed on a different subset per column.",
        ]
        if facts["columns_to_drop"]:
            lines.append(
                f"{len(facts['columns_to_drop'])} column(s) over "
                f"{config.missing_values.drop_column_threshold:.0%} empty will be kept: "
                + ", ".join(facts["columns_to_drop"][:4])
            )
        return lines, "blocking"

    if step_id == "duplicates":
        rows = facts["rows_affected"]
        if rows == 0:
            return (["No duplicate rows were found, so this step is currently doing nothing."], "none")
        return (
            [
                f"{rows:,} duplicate row(s) will stay in the dataset "
                f"({facts['share']:.1%} of all rows).",
                "Each repeated observation is counted twice when the model fits, weighting "
                "it above rows that appear once.",
                "Copies of one row can land on both sides of the train/test split, so the "
                "model is tested on rows it has already seen and scores higher than it should.",
            ],
            "high" if facts["share"] > 0.02 else "moderate",
        )

    if step_id == "outliers":
        values = facts["values_affected"]
        if values == 0:
            return (["No values fall outside the usual range, so this step is doing nothing."], "none")
        worst = list(facts["per_column"].items())[:3]
        detail = ", ".join(f"{c} ({n:,})" for c, n in worst)
        return (
            [
                f"{values:,} outlying value(s) will be left as they are, in "
                f"{facts['columns_affected']} column(s): {detail}.",
                "Scaling is computed from the full range, so one extreme value compresses "
                "every ordinary value into a narrow band.",
                "Models that minimise squared error will fit the extremes at the expense of "
                "the bulk of the data.",
            ],
            "moderate",
        )

    if step_id == "encoding":
        columns = facts["columns_affected"]
        if columns == 0:
            return (["Every column is already numeric, so this step is doing nothing."], "none")
        worst = list(facts["per_column"].items())[:3]
        detail = ", ".join(f"{c} ({n} values)" for c, n in worst)
        return (
            [
                f"{columns} text column(s) will be left as text: {detail}.",
                "scikit-learn cannot fit on text at all — this will raise a "
                "ValueError on the first estimator that touches them.",
                f"The {facts['columns_created']} numeric column(s) this step would have "
                "created will not exist, so that information is simply absent from the model.",
            ],
            "blocking",
        )

    if step_id == "scaling":
        columns = facts["columns_affected"]
        if columns == 0:
            return (["There are no numeric columns to scale."], "none")
        disparity = facts["scale_disparity"]
        lines = [
            f"{columns} numeric column(s) will keep their original units.",
        ]
        if disparity > 10 and facts["widest_column"]:
            lines.append(
                f"{facts['widest_column']} varies about {disparity:,.0f} times more widely "
                f"than {facts['narrowest_column']}. Any model that measures distance — "
                "k-NN, SVM, k-means — will be dominated by the larger column."
            )
        lines.append(
            "Gradient descent converges slowly or not at all when features differ by "
            "orders of magnitude."
        )
        lines.append(
            "Tree-based models are unaffected by scaling, so if that is all you plan to "
            "train, turning this off costs nothing."
        )
        return lines, "high" if disparity > 100 else "moderate"

    if step_id == "feature_selection":
        affected = facts["columns_affected"]
        if affected == 0:
            return (["No columns are constant or redundant, so this step would remove nothing."], "none")
        lines = []
        if facts["constant_columns"]:
            lines.append(
                f"{len(facts['constant_columns'])} column(s) that never change will be "
                "kept: " + ", ".join(facts["constant_columns"][:4])
            )
        if facts["redundant_pairs"]:
            pairs = "; ".join(" + ".join(p) for p in facts["redundant_pairs"][:3])
            lines.append(
                f"{len(facts['redundant_pairs'])} near-duplicate pair(s) will both be "
                f"kept: {pairs}."
            )
        lines.append(
            "Neither is harmful to most models. Redundant columns split one signal across "
            "two features, which makes each look less important than it is."
        )
        return lines, "low"

    if step_id == "imbalance":
        if not facts.get("ratio"):
            return (["There is no classification target to assess."], "none")
        ratio = facts["ratio"]
        if ratio < config.imbalance.ratio_threshold:
            return (
                [f"Classes are balanced at {ratio:.1f}:1, so there is nothing to report."],
                "none",
            )
        return (
            [
                f"The imbalance will not be reported: {facts['majority']} outnumbers "
                f"{facts['minority']} by {ratio:.1f} to 1.",
                "Accuracy will look high for a model that never predicts the smaller class.",
            ],
            "low",
        )

    if step_id == "data_types":
        return (
            [
                "Data type detection cannot be turned off — every other step depends on "
                "knowing which columns hold numbers and which hold categories."
            ],
            "blocking",
        )

    return ([], "none")


def assess_disable(
    df: pd.DataFrame, config: PipelineConfig, step_id: str
) -> ImpactReport:
    """What would happen if ``step_id`` were switched off for this dataset.

    Args:
        df: the dataset as uploaded.
        config: the pipeline as currently configured.
        step_id: the step the user is about to disable.

    Returns:
        The step's label, the exact facts about what it currently does, the consequences
        of it not doing them, and a severity. ``projected_score`` is left ``None`` — the
        caller fills it by running the pipeline both ways, because a guessed score would
        be the one number in this report that was not measured.
    """
    label, description = STEP_LABELS[step_id]
    facts = step_facts(df, config, step_id)
    consequences, severity = _consequences(step_id, facts, config)

    return {
        "step": step_id,
        "label": label,
        "description": description,
        "facts": facts,
        "consequences": consequences,
        "severity": severity,
        "reversible": True,
        "projected_score": None,
        "current_score": None,
    }


def summarise_enabled(df: pd.DataFrame, config: PipelineConfig, step_id: str) -> str:
    """The one-line "what this step is doing" shown on an enabled card."""
    facts = step_facts(df, config, step_id)
    if step_id == "missing_values":
        n = facts["values_affected"]
        return (
            "No missing values in this dataset."
            if n == 0
            else f"{n:,} missing value(s) in {facts['columns_affected']} column(s) will be filled."
        )
    if step_id == "duplicates":
        n = facts["rows_affected"]
        return "No duplicate rows found." if n == 0 else f"{n:,} duplicate row(s) will be removed."
    if step_id == "outliers":
        n = facts["values_affected"]
        verb = {
            "cap": "capped", "remove": "removed with their rows",
            "winsorize": "pulled to the nearest percentile", "flag": "flagged",
        }[config.outliers.action]
        return "No outliers detected." if n == 0 else f"{n:,} outlying value(s) will be {verb}."
    if step_id == "encoding":
        n = facts["columns_affected"]
        return (
            "Every column is already numeric."
            if n == 0
            else f"{n} text column(s) will become numbers."
        )
    if step_id == "scaling":
        n = facts["columns_affected"]
        return "No numeric columns to scale." if n == 0 else f"{n} numeric column(s) will be rescaled."
    if step_id == "feature_selection":
        n = facts["columns_affected"]
        return "Nothing to remove." if n == 0 else f"{n} column(s) would be removed."
    if step_id == "imbalance":
        ratio = facts.get("ratio")
        if not ratio:
            return "No classification target."
        return (
            f"Classes are balanced at {ratio:.1f}:1."
            if ratio < config.imbalance.ratio_threshold
            else f"Largest class is {ratio:.1f}x the smallest."
        )
    if step_id == "data_types":
        return f"{facts['columns_affected']} column(s) will be typed."
    return ""
