"""Post-split checks: did the split itself go wrong?

Everything else in this application asks whether the *data* is sound. These two ask
whether the *division* of it is — because a pipeline can be correct in every step and
still hand you a test fold that does not resemble the training fold, and every score
measured against it will be misleading in a way no amount of inspecting the transforms
would reveal.

Both run after the split and before anything is reported, and neither changes the data.
A drifted feature is a fact about the split, and the remedy — a different seed, more
rows, a stratified variable you had not thought to stratify on — is the user's call.

Why a KS test rather than comparing means
-----------------------------------------
Two folds can have identical means and completely different shapes: bimodal against
uniform, tight against heavy-tailed. The Kolmogorov–Smirnov statistic compares the whole
empirical distribution, so it notices that. It is also distribution-free, which matters
here because the features have already been transformed and nothing guarantees they are
normal.

On p-values
-----------
The threshold is ``p < 0.01`` per the spec, and the honest caveat is recorded with the
result: testing fifty features means roughly one false flag per run at that level even
when nothing is wrong. :func:`drift_report` therefore returns the count of features
tested alongside the flags, and the interface says "1 of 50" rather than "1 feature
drifted", so a single flag reads as the noise it probably is.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score

from app.config import (
    BASELINE_CV_FOLDS,
    DRIFT_P_VALUE,
    MIN_ROWS_FOR_BASELINE,
    MIN_ROWS_FOR_DRIFT,
    SEED,
)

__all__ = ["drift_report", "split_balance", "sanity_baseline", "run_post_split_checks"]


def _numeric(series: pd.Series) -> pd.Series | None:
    """A numeric view, or ``None`` for something a KS test cannot read."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return None


def drift_report(
    X_train: pd.DataFrame, X_test: pd.DataFrame, p_threshold: float = DRIFT_P_VALUE
) -> dict[str, Any]:
    """Per-feature Kolmogorov–Smirnov test, train against test.

    Args:
        X_train: the training fold, after transformation.
        X_test: the test fold, transformed by the same fitted pipeline.
        p_threshold: flag a feature below this p-value.

    Returns:
        ``{flagged, tested, skipped, threshold, features}`` — ``features`` holds one
        record per flagged column with its statistic and p-value.

    A feature that is constant in either fold is skipped rather than flagged: the test
    is undefined there, and a one-hot column that happens to be all-zero in the test
    fold is a cardinality artefact, not drift.
    """
    flagged: list[dict[str, Any]] = []
    tested = 0
    skipped = 0

    for name in X_train.columns:
        if name not in X_test.columns:
            skipped += 1
            continue

        train_values = _numeric(X_train[name])
        test_values = _numeric(X_test[name])
        if train_values is None or test_values is None:
            skipped += 1
            continue

        train_clean = train_values.dropna()
        test_clean = test_values.dropna()
        if len(train_clean) < MIN_ROWS_FOR_DRIFT or len(test_clean) < MIN_ROWS_FOR_DRIFT:
            skipped += 1
            continue
        if train_clean.nunique() < 2 and test_clean.nunique() < 2:
            skipped += 1
            continue

        tested += 1
        result = stats.ks_2samp(train_clean.to_numpy(), test_clean.to_numpy())
        statistic, p_value = float(result.statistic), float(result.pvalue)
        if np.isnan(p_value) or p_value >= p_threshold:
            continue

        flagged.append({
            "column": str(name),
            "statistic": round(statistic, 4),
            "p_value": float(f"{p_value:.4g}"),
            "severity": "warn",
            "message": (
                f"{name} is distributed differently in the training and test folds "
                f"(KS={statistic:.3f}, p={p_value:.3g})."
            ),
            "suggestion": (
                "A score measured on this fold may not transfer. Try a different seed, "
                "or check whether the rows were ordered by something meaningful before "
                "the split."
            ),
        })

    return {
        "flagged": len(flagged),
        "tested": tested,
        "skipped": skipped,
        "threshold": p_threshold,
        "features": sorted(flagged, key=lambda f: f["p_value"]),
        # Stated so the interface can put a single flag in proportion rather than
        # presenting it as a finding. Rounding the expectation up to 1 would overstate
        # it badly on a narrow dataset — with two features tested the expected number
        # of false flags is 0.02, and claiming "roughly 1 of 2" would turn a real
        # finding into something the reader dismisses.
        "note": _false_positive_note(tested, p_threshold),
    }


def _false_positive_note(tested: int, threshold: float) -> str:
    """How many of these flags to expect from chance alone, stated honestly."""
    if tested == 0:
        return "No features could be tested."
    expected = tested * threshold
    if expected < 0.5:
        return (
            f"At p<{threshold}, fewer than one of these {tested} features would be "
            "flagged by chance, so a flag here is worth reading."
        )
    return (
        f"At p<{threshold}, about {expected:.1f} of {tested} features would be flagged "
        "by chance alone — treat a single flag with caution."
    )


def split_balance(y_train: pd.Series, y_test: pd.Series) -> dict[str, Any] | None:
    """Class proportions on each side of the split, and whether they agree.

    Returns ``None`` for a regression target. For a classification one, reports the
    share of every class in both folds and flags two distinct problems:

    * a class **missing entirely** from one fold — the model is scored on something it
      never saw, or trained on something it is never scored against;
    * a class whose **share differs materially** between folds, which stratification is
      supposed to prevent and which signals it did not run.
    """
    if pd.api.types.is_numeric_dtype(y_train) and y_train.nunique(dropna=True) > 20:
        return None

    train_counts = y_train.value_counts(dropna=False)
    test_counts = y_test.value_counts(dropna=False)
    classes = sorted(set(train_counts.index) | set(test_counts.index), key=str)

    train_total = max(int(train_counts.sum()), 1)
    test_total = max(int(test_counts.sum()), 1)

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    skewed: list[str] = []

    for value in classes:
        in_train = int(train_counts.get(value, 0))
        in_test = int(test_counts.get(value, 0))
        train_share = in_train / train_total
        test_share = in_test / test_total

        if in_train == 0 or in_test == 0:
            missing.append(str(value))
        elif abs(train_share - test_share) > 0.05:
            skewed.append(str(value))

        rows.append({
            "value": str(value),
            "train_count": in_train,
            "test_count": in_test,
            "train_share": round(train_share, 4),
            "test_share": round(test_share, 4),
            "delta": round(test_share - train_share, 4),
        })

    warnings: list[dict[str, Any]] = []
    if missing:
        warnings.append({
            "severity": "critical",
            "message": (
                f"{len(missing)} class(es) appear in only one fold: {', '.join(missing[:5])}."
            ),
            "suggestion": (
                "The model is either trained on a class it is never scored against, or "
                "scored on one it never saw. Collect more rows of that class."
            ),
        })
    if skewed:
        warnings.append({
            "severity": "warn",
            "message": (
                f"{len(skewed)} class(es) differ by more than 5 percentage points "
                f"between the folds: {', '.join(skewed[:5])}."
            ),
            "suggestion": (
                "Stratification should prevent this. Check that it ran — a class with "
                "fewer than two rows disables it."
            ),
        })

    return {
        "classes": rows,
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "balanced": not missing and not skewed,
        "warnings": warnings,
    }


def run_post_split_checks(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Both §4d checks, and the warnings they jointly produce."""
    drift = drift_report(X_train, X_test)
    balance = split_balance(y_train, y_test)

    warnings: list[dict[str, Any]] = []
    if drift["flagged"]:
        warnings.append({
            "kind": "train_test_drift",
            "severity": "warn",
            "message": (
                f"{drift['flagged']} of {drift['tested']} features are distributed "
                "differently in the training and test folds."
            ),
            "suggestion": drift["note"],
        })
    for warning in (balance or {}).get("warnings", []):
        warnings.append({"kind": "split_balance", **warning})

    return {"drift": drift, "balance": balance, "warnings": warnings}


# --------------------------------------------------------------------------------------
# Sanity baseline (spec §4e)
# --------------------------------------------------------------------------------------


def sanity_baseline(
    X_train: pd.DataFrame, y_train: pd.Series, task: str, seed: int = SEED
) -> dict[str, Any] | None:
    """Cross-validated scores for a trivial model and a simple one, on the training fold.

    This is the cheapest useful question about a prepared dataset: *is there anything in
    here at all?* Two models answer it together, and neither is the point on its own.

    ``Dummy`` ignores the features entirely — majority class, or the mean. It is the
    score to beat, and on a 95/5 target it beats 0.95 accuracy while knowing nothing,
    which is precisely why a bare accuracy figure is worthless without it.

    ``LogisticRegression`` / ``Ridge`` is the simplest model that can use the features.
    If it does not clear the dummy, either the features carry no signal or the
    preprocessing destroyed it — and the user should know that before spending an
    afternoon tuning a gradient booster.

    Scores only, on the **training fold** via cross-validation. The held-out rows are not
    touched: this is a diagnostic about preparation, not a result, and spending the test
    fold on it would leave nothing honest to report later.
    """
    if len(X_train) < MIN_ROWS_FOR_BASELINE:
        return None

    numeric = X_train.select_dtypes(include="number")
    if numeric.empty or not np.isfinite(numeric.to_numpy(dtype="float64")).all():
        return None

    classification = task.strip().lower().startswith(("binary", "multiclass", "class"))

    if classification:
        counts = y_train.value_counts()
        folds = min(BASELINE_CV_FOLDS, int(counts.min()))
        if folds < 2 or len(counts) < 2:
            return None
        models = {
            "dummy": DummyClassifier(strategy="most_frequent"),
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        }
        metric = "accuracy"
    else:
        folds = BASELINE_CV_FOLDS
        models = {
            "dummy": DummyRegressor(strategy="mean"),
            "ridge": Ridge(random_state=seed),
        }
        metric = "r2"

    scores: dict[str, float] = {}
    for name, model in models.items():
        try:
            values = cross_val_score(model, numeric, y_train, cv=folds, scoring=metric)
        except (ValueError, ConvergenceWarning):
            return None
        scores[name] = round(float(values.mean()), 4)

    dummy = scores["dummy"]
    model_name = "logistic_regression" if classification else "ridge"
    model_score = scores[model_name]
    lift = round(model_score - dummy, 4)

    if lift <= 0:
        verdict = (
            "The simple model does no better than guessing. Either the features carry "
            "no signal for this target, or the preprocessing removed it."
        )
        severity = "warn"
    elif classification and lift < 0.02:
        verdict = (
            f"The simple model beats guessing by {lift:.1%}, which is within noise on a "
            "dataset this size. Treat any headline accuracy with suspicion."
        )
        severity = "warn"
    else:
        verdict = (
            f"The simple model beats the trivial one by {lift:+.3f} {metric}, so there "
            "is signal in the prepared features."
        )
        severity = "info"

    return {
        "task": "classification" if classification else "regression",
        "metric": metric,
        "folds": folds,
        "scores": scores,
        "baseline": dummy,
        "model": model_name,
        "model_score": model_score,
        "lift": lift,
        "severity": severity,
        "verdict": verdict,
        "note": (
            "Cross-validated on the training rows only. The held-out fold is untouched, "
            "so these are a check on preparation, not a result."
        ),
    }
