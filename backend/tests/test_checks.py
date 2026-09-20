"""The correctness rules added in slice 1: leakage exclusion, drift, balance, baseline.

These defend claims the product makes to the user, so each test is written to fail if
the claim stops being true — not merely if the code stops running. Where a rule is
supposed to fire, there is a matching test that it stays quiet on clean data, because a
detector that fires on everything is worth no more than one that fires on nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config import DRIFT_P_VALUE, LEAKAGE_CORRELATION, LEAKAGE_TREE_SCORE
from app.services.checks import (
    drift_report,
    sanity_baseline,
    split_balance,
)
from app.services.glassbox import fit_preprocessing
from app.services.screening import identifier_screen, target_leakage_screen
from tests.fixtures import generate as fixtures


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(fixtures.ensure(name))


# --------------------------------------------------------------------------------------
# Target leakage: detect, exclude, and say so
# --------------------------------------------------------------------------------------


def test_a_linear_copy_of_the_target_is_excluded() -> None:
    frame = load("leakage")
    target = frame.pop("churned")

    dropped, flags = target_leakage_screen(frame, target)

    assert "leaked_score" in dropped
    flag = next(f for f in flags if f["column"] == "leaked_score")
    assert abs(flag["correlation"]) >= LEAKAGE_CORRELATION
    assert flag["severity"] == "critical"
    assert flag["excluded"] is True
    assert flag["can_reinclude"] is True


def test_a_non_linear_copy_is_caught_by_the_tree_alone() -> None:
    """The case correlation cannot see, and the reason there are two detectors.

    A status string that maps one-to-one onto a three-class target has no linear
    relationship with it at all — correlation is undefined — but a depth-2 tree splits
    it perfectly.
    """
    rng = np.random.default_rng(0)
    n = 400
    target = rng.choice([0, 1, 2], n)
    frame = pd.DataFrame({
        "status": np.where(target == 0, "new", np.where(target == 1, "open", "closed")),
        "noise": rng.normal(size=n),
    })

    dropped, flags = target_leakage_screen(frame, pd.Series(target, name="state"))

    assert "status" in dropped
    assert "noise" not in dropped
    flag = next(f for f in flags if f["column"] == "status")
    assert flag["detected_by"] == "single-feature tree"
    assert flag["correlation"] is None, "correlation should not have fired here"
    assert flag["probe_score"] >= LEAKAGE_TREE_SCORE


def test_a_two_value_string_column_is_probed_by_the_tree() -> None:
    """Regression guard.

    An earlier version handed the tree the raw strings whenever a column also had a
    numeric view, so the probe silently returned ``None`` for every two-value string —
    exactly the shape the tree was best at catching.
    """
    frame = load("leakage")
    target = frame.pop("churned")

    _, flags = target_leakage_screen(frame, target)
    flag = next(f for f in flags if f["column"] == "leaked_flag")
    assert flag["probe_score"] is not None, "the tree probe did not run"
    assert flag["probe_score"] >= LEAKAGE_TREE_SCORE


def test_honest_features_survive() -> None:
    """A predictive column is not a leaked one; the screen must tell them apart."""
    frame = load("leakage")
    target = frame.pop("churned")

    dropped, _ = target_leakage_screen(frame, target)

    assert "honest_feature" not in dropped
    assert "noise" not in dropped


def test_a_clean_dataset_triggers_nothing() -> None:
    frame = load("clean_binary")
    target = frame.pop("converted")
    assert target_leakage_screen(frame, target) == ({}, [])


def test_an_imbalanced_target_does_not_make_useless_columns_look_leaked() -> None:
    """On a 95/5 target a column that knows nothing still scores 0.95 accuracy.

    Scoring against the majority-class baseline rather than against zero is what stops
    the detector condemning every column in an imbalanced dataset.
    """
    frame = load("imbalanced")
    target = frame.pop("fraud")
    dropped, _ = target_leakage_screen(frame, target)
    assert dropped == {}


def test_leaked_columns_never_reach_the_feature_matrix() -> None:
    """End to end: exclusion has to survive the whole pipeline, not just the screen."""
    frame = load("leakage")
    result = fit_preprocessing(
        frame, target="churned", task_type="binary classification",
        screeners=[identifier_screen, target_leakage_screen],
    )

    assert "leaked_score" in result.excluded
    assert "leaked_flag" in result.excluded
    assert not [c for c in result.X_train.columns if c.startswith("leaked")]
    assert {"honest_feature", "noise"} <= set(result.plan.feature_columns)


# --------------------------------------------------------------------------------------
# Train/test drift
# --------------------------------------------------------------------------------------


def test_drift_is_detected_when_the_folds_really_differ() -> None:
    rng = np.random.default_rng(1)
    train = pd.DataFrame({"stable": rng.normal(0, 1, 400), "shifted": rng.normal(0, 1, 400)})
    test = pd.DataFrame({"stable": rng.normal(0, 1, 100), "shifted": rng.normal(2.5, 1, 100)})

    report = drift_report(train, test)

    assert report["flagged"] == 1
    assert report["tested"] == 2
    flagged = report["features"][0]
    assert flagged["column"] == "shifted"
    assert flagged["p_value"] < DRIFT_P_VALUE
    assert flagged["suggestion"]


def test_matching_folds_are_not_flagged() -> None:
    rng = np.random.default_rng(2)
    train = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    test = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
    assert drift_report(train, test)["flagged"] == 0


def test_constant_columns_are_skipped_not_flagged() -> None:
    """A one-hot column that is all-zero in the test fold is cardinality, not drift."""
    train = pd.DataFrame({"const": [1.0] * 200})
    test = pd.DataFrame({"const": [1.0] * 50})
    report = drift_report(train, test)
    assert report["flagged"] == 0
    assert report["skipped"] == 1


def test_the_false_positive_note_does_not_overstate_itself() -> None:
    """With two features tested, 0.02 false flags are expected — not 'roughly 1 of 2'.

    Rounding that up would turn a real finding into something a reader dismisses.
    """
    rng = np.random.default_rng(3)
    train = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    test = pd.DataFrame({"a": rng.normal(size=80), "b": rng.normal(size=80)})

    note = drift_report(train, test)["note"]
    assert "fewer than one" in note
    assert "roughly 1 of 2" not in note


# --------------------------------------------------------------------------------------
# Class balance across the split
# --------------------------------------------------------------------------------------


def test_a_class_missing_from_one_fold_is_critical() -> None:
    balance = split_balance(
        pd.Series(["a"] * 80 + ["b"] * 20), pd.Series(["a"] * 25)
    )
    assert balance is not None
    assert balance["balanced"] is False
    assert any(w["severity"] == "critical" for w in balance["warnings"])


def test_materially_different_shares_are_warned_about() -> None:
    balance = split_balance(
        pd.Series(["a"] * 90 + ["b"] * 10), pd.Series(["a"] * 50 + ["b"] * 50)
    )
    assert balance is not None
    assert any(w["severity"] == "warn" for w in balance["warnings"])


def test_a_stratified_split_reports_balanced() -> None:
    frame = load("clean_binary")
    result = fit_preprocessing(frame, target="converted", task_type="binary classification")
    balance = result.checks["balance"]
    assert balance["balanced"] is True
    assert balance["train_rows"] + balance["test_rows"] == len(frame.drop_duplicates())


def test_a_regression_target_has_no_class_balance() -> None:
    assert split_balance(pd.Series(np.linspace(0, 100, 400)), pd.Series(np.linspace(0, 100, 100))) is None


# --------------------------------------------------------------------------------------
# Stratification fallback
# --------------------------------------------------------------------------------------


def test_a_singleton_class_disables_stratification_loudly() -> None:
    """Spec §3. Falling back silently leaves the user believing the rare class is in both
    folds when it cannot be — and the rare class is the entire reason to stratify."""
    frame = load("tiny")
    result = fit_preprocessing(frame, target="g", task_type="multiclass classification",
                               test_size=0.25)

    warning = next(w for w in result.warnings if w["kind"] == "stratification_disabled")
    assert warning["severity"] == "warn"
    assert "fewer than 2 rows" in warning["message"]
    assert warning["suggestion"]
    assert "c" in warning["classes"]


def test_a_healthy_target_produces_no_stratification_warning() -> None:
    frame = load("clean_binary")
    result = fit_preprocessing(frame, target="converted", task_type="binary classification")
    assert not [w for w in result.warnings if w["kind"] == "stratification_disabled"]


# --------------------------------------------------------------------------------------
# Deduplication
# --------------------------------------------------------------------------------------


def test_duplicate_rows_are_removed_before_the_split() -> None:
    """Spec §4a. Removing them afterwards cannot undo copies landing in both folds."""
    frame = load("clean_binary")
    with_repeats = pd.concat([frame, frame.head(40)], ignore_index=True)

    result = fit_preprocessing(
        with_repeats, target="converted", task_type="binary classification"
    )

    assert len(result.X_train) + len(result.X_test) == len(frame)
    warning = next(w for w in result.warnings if w["kind"] == "duplicates_removed")
    assert warning["count"] == 40
    assert warning["severity"] == "info"


def test_no_duplicate_warning_when_there_are_none() -> None:
    frame = load("clean_binary")
    result = fit_preprocessing(frame, target="converted", task_type="binary classification")
    assert not [w for w in result.warnings if w["kind"] == "duplicates_removed"]


# --------------------------------------------------------------------------------------
# Sanity baseline
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture,target,task,metric",
    [
        ("clean_binary", "converted", "binary classification", "accuracy"),
        ("multiclass", "species", "multiclass classification", "accuracy"),
        ("regression", "price", "regression", "r2"),
    ],
)
def test_the_baseline_finds_signal_where_there_is_some(fixture, target, task, metric) -> None:
    frame = load(fixture)
    result = fit_preprocessing(frame, target=target, task_type=task)

    baseline = result.checks["baseline"]
    assert baseline is not None
    assert baseline["metric"] == metric
    assert baseline["lift"] > 0, "a fixture built with real structure shows no lift"
    assert baseline["severity"] == "info"
    assert "dummy" in baseline["scores"]


def test_the_baseline_reports_absence_of_signal() -> None:
    """Pure noise must be called pure noise, not given a flattering accuracy."""
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({
        "a": rng.normal(size=300), "b": rng.normal(size=300),
        "y": rng.choice([0, 1], 300),
    })

    result = fit_preprocessing(frame, target="y", task_type="binary classification")
    baseline = result.checks["baseline"]

    assert baseline["lift"] <= 0.02
    assert baseline["severity"] == "warn"
    assert any(w["kind"] == "weak_baseline" for w in result.warnings)


def test_the_baseline_exposes_the_imbalance_trap() -> None:
    """On a 95/5 target the dummy scores ~0.95. Reporting that without the dummy beside
    it is how a useless model gets shipped."""
    frame = load("imbalanced")
    result = fit_preprocessing(frame, target="fraud", task_type="binary classification")

    baseline = result.checks["baseline"]
    assert baseline["baseline"] > 0.9, "the dummy should score high on a 95/5 target"
    assert baseline["lift"] < 0.05


def test_the_baseline_uses_only_the_training_fold() -> None:
    """It is a diagnostic about preparation. Spending the test fold on it would leave
    nothing honest to report later."""
    frame = load("clean_binary")
    result = fit_preprocessing(frame, target="converted", task_type="binary classification")

    baseline = sanity_baseline(
        result.X_train, result.y_train, "binary classification"
    )
    assert baseline is not None
    assert baseline["folds"] <= len(result.X_train)
    assert "training rows only" in baseline["note"]


def test_no_baseline_on_too_few_rows() -> None:
    """A cross-validated score on 12 rows says more about the split than the data."""
    frame = load("tiny")
    assert sanity_baseline(frame[["x"]], frame["y"], "binary classification") is None


# --------------------------------------------------------------------------------------
# Profiling additions from slice 3 (spec §2)
# --------------------------------------------------------------------------------------


def test_every_numeric_column_carries_descriptive_statistics() -> None:
    from app.services.profile import column_profiles

    frame = load("regression")
    records = {r["column"]: r for r in column_profiles(frame, target="price")}

    stats = records["area_sqm"]["stats"]
    assert stats is not None
    for key in ("count", "mean", "std", "min", "q1", "median", "q3", "max", "skew"):
        assert key in stats, f"{key} missing from the numeric summary"
    assert stats["min"] <= stats["q1"] <= stats["median"] <= stats["q3"] <= stats["max"]


def test_a_categorical_column_has_no_numeric_statistics() -> None:
    from app.services.profile import column_profiles

    frame = load("regression")
    records = {r["column"]: r for r in column_profiles(frame, target="price")}
    assert records["district"]["stats"] is None


def test_numeric_columns_get_histogram_bins() -> None:
    from app.services.profile import histogram

    frame = load("regression")
    result = histogram(frame["area_sqm"])

    assert result is not None
    assert result["kind"] == "numeric"
    assert len(result["counts"]) == result["bins"]
    assert len(result["edges"]) == result["bins"] + 1
    assert sum(result["counts"]) == int(frame["area_sqm"].notna().sum())


def test_categorical_columns_get_value_counts_instead() -> None:
    """Same question — how is this distributed — answered the way the column can."""
    from app.services.profile import histogram

    result = histogram(load("regression")["district"])
    assert result is not None
    assert result["kind"] == "categorical"
    assert len(result["labels"]) == len(result["counts"])


def test_a_constant_column_does_not_break_the_histogram() -> None:
    from app.services.profile import histogram

    result = histogram(pd.Series([7.0] * 50))
    assert result is not None
    assert result["counts"] == [50]


def test_top_values_are_reported_for_every_column_kind() -> None:
    """The most common value of a numeric column is how you notice 40% of it is zero."""
    from app.services.profile import top_values

    values = top_values(pd.Series([0.0] * 40 + list(range(1, 61))))
    assert values[0]["value"] == "0.0"
    assert values[0]["count"] == 40
    assert 0 < values[0]["share"] <= 1


def test_skew_is_flagged_at_the_spec_threshold() -> None:
    """|skew| > 1, not the 2.0 the health engine had been using."""
    from app.config import SKEW_THRESHOLD
    from app.services.profile import numeric_summary

    assert SKEW_THRESHOLD == 1.0
    heavy = numeric_summary(pd.Series(np.random.default_rng(0).exponential(1, 500)))
    assert heavy is not None
    assert heavy["skewed"] is True

    symmetric = numeric_summary(pd.Series(np.random.default_rng(0).normal(0, 1, 500)))
    assert symmetric is not None
    assert symmetric["skewed"] is False


@pytest.mark.parametrize("fixture", ["high_missing", "id_and_constant", "clean_binary"])
def test_every_warning_carries_a_suggestion(fixture) -> None:
    """Spec §2. A warning that names a problem and stops makes the reader guess."""
    from app.services.ingest import read_upload

    path = fixtures.ensure(fixture)
    result = read_upload(path.read_bytes(), path.name)

    for issue in result.issues:
        record = issue.as_dict()
        assert record["severity"] in ("info", "warn", "critical")
        assert record["message"].strip()
        assert record["suggestion"].strip(), f"{record['message']} has no suggestion"


def test_health_issues_use_the_spec_severity_vocabulary() -> None:
    from app.services.health import assess

    report = assess(load("high_missing"), target="target")
    for issue in report["issues"]:
        assert issue["severity"] in ("info", "warn", "critical")
        assert issue["suggestion"].strip()
