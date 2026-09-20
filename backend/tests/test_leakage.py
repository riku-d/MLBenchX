"""Proof that held-out rows never reach anything that decides or learns.

The existing suite already proves the narrow claim — no transformer is *fitted* on a
test row. These tests defend the wider one the audit added: that the *decisions* are
made from the training fold too, so a value appearing only in the held-out rows cannot
change which columns survive or how they are encoded.

Each test is written so the pre-audit implementation fails it. A test that passes both
before and after a fix proves nothing about the fix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from app.services.glassbox import (
    FrequencyEncoder,
    fit_preprocessing,
)
from app.services.screening import identifier_screen, target_leakage_screen

N = 400


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(31)
    return pd.DataFrame({
        "amount": np.round(rng.gamma(5, 20, N), 3),
        "tier": rng.choice(["bronze", "silver", "gold"], N),
        "label": rng.choice([0, 1], N, p=[0.4, 0.6]),
    })


# --------------------------------------------------------------------------------------
# Nothing is fitted on a held-out row
# --------------------------------------------------------------------------------------


def test_no_transformer_ever_sees_a_test_row(frame, monkeypatch) -> None:
    """A spy on every fit records which rows it was given; none may be held out."""
    seen: list[set] = []

    def spy(cls):
        original = cls.fit

        def recording_fit(self, X, y=None, **kwargs):
            if isinstance(X, pd.DataFrame):
                seen.append(set(X.index))
            return original(self, X, y, **kwargs)

        monkeypatch.setattr(cls, "fit", recording_fit, raising=False)

    for cls in (SimpleImputer, StandardScaler, OneHotEncoder, OrdinalEncoder, FrequencyEncoder):
        spy(cls)

    result = fit_preprocessing(frame, "label", "classification")
    held_out = set(result.X_test.index)

    assert seen, "the spy recorded nothing, so this test proves nothing"
    for rows in seen:
        assert not (rows & held_out), "a transformer was fitted on held-out rows"


def test_the_fill_value_is_the_training_median_not_the_whole_files(frame) -> None:
    holes = frame.copy()
    rng = np.random.default_rng(3)
    holes.loc[rng.choice(N, 80, replace=False), "amount"] = np.nan

    result = fit_preprocessing(holes, "label", "classification")
    entry = next(e for e in result.log if e["stage"] == "impute" and "amount" in e["columns"])
    logged = entry["details"]["fill_values"]["amount"]

    # Index against the deduplicated frame: dedupe runs before the split and resets the
    # index, so positions in `result` refer to that frame, not the one passed in.
    deduped = holes.drop_duplicates().reset_index(drop=True)
    train_median = float(deduped.loc[result.X_train.index, "amount"].median())
    whole_median = float(deduped["amount"].median())

    assert logged == pytest.approx(train_median)
    # If the two coincided the assertion above would be satisfied by a leaking
    # implementation too, so the fixture is only useful when they differ.
    assert train_median != pytest.approx(whole_median), "fixture no longer proves anything"


# --------------------------------------------------------------------------------------
# Nothing is *decided* from a held-out row
# --------------------------------------------------------------------------------------


def test_a_category_only_in_the_held_out_rows_cannot_change_the_encoder() -> None:
    """The audit's central fix, as a behavioural difference.

    A column with few enough distinct values to be one-hot encoded is given many novel
    categories — but only in the held-out rows. Profiling the whole frame, which is what
    the code did before, sees a cardinality over the threshold and routes the column to
    the high-cardinality encoder instead. Profiling the training fold keeps it one-hot.

    The held-out rows are identified by running the split once first, so the scenario is
    constructed rather than hoped for.
    """
    from app.config import ONEHOT_MAX_CARDINALITY

    rng = np.random.default_rng(99)
    n = 500
    frame = pd.DataFrame({
        "city": rng.choice([f"city_{i}" for i in range(6)], n),
        "amount": rng.normal(size=n),
        "label": rng.choice([0, 1], n),
    })

    baseline = fit_preprocessing(frame, "label", "classification")
    assert baseline.plan.low_card == ["city"]
    held_out = baseline.X_test.index

    # Every held-out row gets its own never-before-seen city.
    tampered = frame.copy()
    tampered.loc[held_out, "city"] = [f"only_in_test_{i}" for i in range(len(held_out))]

    train_distinct = int(tampered.loc[baseline.X_train.index, "city"].nunique())
    whole_distinct = int(tampered["city"].nunique())
    assert train_distinct <= ONEHOT_MAX_CARDINALITY < whole_distinct, (
        "fixture must straddle the threshold for this test to mean anything"
    )

    after = fit_preprocessing(tampered, "label", "classification")
    assert after.plan.low_card == ["city"], "held-out rows changed the encoder choice"
    assert after.plan.high_card == []


def test_changing_only_held_out_rows_leaves_the_plan_identical(frame) -> None:
    """The strongest form: rewrite the test fold entirely, get the same decisions.

    Any decision that read a held-out row would change when those rows change.
    """
    baseline = fit_preprocessing(
        frame, "label", "classification",
        screeners=[identifier_screen, target_leakage_screen],
    )
    held_out = baseline.X_test.index

    tampered = frame.copy()
    # Make the held-out rows extreme, near-constant and full of novel categories.
    tampered.loc[held_out, "amount"] = 1e9
    tampered.loc[held_out, "tier"] = [f"novel_{i}" for i in range(len(held_out))]

    after = fit_preprocessing(
        tampered, "label", "classification",
        screeners=[identifier_screen, target_leakage_screen],
    )

    assert after.plan.numeric == baseline.plan.numeric
    assert after.plan.binary == baseline.plan.binary
    assert after.plan.low_card == baseline.plan.low_card
    assert after.plan.high_card == baseline.plan.high_card
    assert after.plan.dropped == baseline.plan.dropped
    assert after.excluded == baseline.excluded


def test_the_learned_values_are_unchanged_when_only_test_rows_change(frame) -> None:
    """Same idea, one level down: the fitted numbers must be identical too."""
    baseline = fit_preprocessing(frame, "label", "classification")
    held_out = baseline.X_test.index

    tampered = frame.copy()
    tampered.loc[held_out, "amount"] = tampered.loc[held_out, "amount"] * 1000

    after = fit_preprocessing(tampered, "label", "classification")

    before_scale = next(e for e in baseline.log if e["stage"] == "scale")
    after_scale = next(e for e in after.log if e["stage"] == "scale")
    assert after_scale["details"]["mean"] == before_scale["details"]["mean"]
    assert after_scale["details"]["scale"] == before_scale["details"]["scale"]

    # And the training rows themselves come out byte-identical.
    pd.testing.assert_frame_equal(after.X_train, baseline.X_train)


def test_an_identifier_hidden_in_the_test_fold_is_not_screened_out(frame) -> None:
    """The screens run on the training fold, so they judge it and nothing else."""
    withcol = frame.copy()
    withcol["code"] = "SAME"  # constant in every fold

    baseline = fit_preprocessing(
        withcol, "label", "classification", screeners=[identifier_screen]
    )
    held_out = baseline.X_test.index

    tampered = withcol.copy()
    tampered.loc[held_out, "code"] = [f"UNIQUE{i}" for i in range(len(held_out))]

    after = fit_preprocessing(
        tampered, "label", "classification", screeners=[identifier_screen]
    )
    # In the training fold `code` is still constant, so it is dropped for being
    # constant — never for the uniqueness that exists only in the held-out rows.
    assert "constant" in after.plan.dropped["code"]
    assert "code" not in after.excluded


# --------------------------------------------------------------------------------------
# Target encoding, the one transformer that does read y
# --------------------------------------------------------------------------------------


def test_target_encoding_never_sees_a_held_out_label(frame, monkeypatch) -> None:
    """``TargetEncoder`` reads y by design. It must only ever read the training y."""
    from sklearn.preprocessing import TargetEncoder

    crowded = frame.copy()
    rng = np.random.default_rng(12)
    crowded["region"] = rng.choice([f"r{i:02d}" for i in range(40)], len(crowded))

    seen_rows: list[set] = []
    original = TargetEncoder.fit_transform

    def recording(self, X, y=None, **kwargs):
        if isinstance(X, pd.DataFrame):
            seen_rows.append(set(X.index))
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(TargetEncoder, "fit_transform", recording, raising=False)

    result = fit_preprocessing(
        crowded, "label", "classification", high_card_encoder="target"
    )
    held_out = set(result.X_test.index)

    assert seen_rows, "the spy recorded nothing, so this test proves nothing"
    for rows in seen_rows:
        assert not (rows & held_out)


def test_test_rows_are_transformed_but_never_refitted(frame) -> None:
    """``transform`` on the test fold must not update anything the encoder learned."""
    crowded = frame.copy()
    rng = np.random.default_rng(8)
    crowded["region"] = rng.choice([f"r{i:02d}" for i in range(40)], len(crowded))

    result = fit_preprocessing(crowded, "label", "classification")
    encoder = result.pipeline.named_transformers_["high_card"].named_steps["encode"]
    learned = {k: dict(v) for k, v in encoder.frequencies_.items()}

    result.pipeline.transform(crowded.drop(columns=["label"]))

    assert {k: dict(v) for k, v in encoder.frequencies_.items()} == learned


# --------------------------------------------------------------------------------------
# What the user is told
# --------------------------------------------------------------------------------------


def test_every_fitted_step_declares_the_fold_it_learned_from(frame) -> None:
    result = fit_preprocessing(frame, "label", "classification")
    assert result.log
    for entry in result.log:
        assert entry["fitted_on"] == "train_fold_only"


def test_the_split_is_reported_as_the_first_step(frame) -> None:
    """The order shown has to be the order that happened."""
    from app.services import explain

    result = fit_preprocessing(frame, "label", "classification")
    cards = explain.build_steps(frame, result)

    assert cards[0]["stage"] == "split"
    assert cards[0]["n"] == 1
    assert "set aside" in cards[0]["title"].lower() or "aside" in cards[0]["explanation"]


def test_train_and_test_come_out_aligned_and_complete(frame) -> None:
    result = fit_preprocessing(frame, "label", "classification")

    assert list(result.X_train.columns) == list(result.X_test.columns)
    assert not result.X_train.isna().any().any()
    assert not result.X_test.isna().any().any()
    # Duplicates are removed before the split (spec §4a), so the folds sum to the
    # deduplicated row count rather than the raw one. Comparing against len(frame)
    # would make this test fail whenever the fixture happens to repeat a row.
    assert len(result.X_train) + len(result.X_test) == len(frame.drop_duplicates())
    assert not set(result.X_train.index) & set(result.X_test.index)


def test_stratification_preserves_the_class_balance() -> None:
    rng = np.random.default_rng(17)
    skewed = pd.DataFrame({
        "x": rng.normal(size=N),
        "label": rng.choice([0, 1], N, p=[0.9, 0.1]),
    })
    result = fit_preprocessing(skewed, "label", "classification")

    train_rate = float(result.y_train.mean())
    test_rate = float(result.y_test.mean())
    assert train_rate == pytest.approx(test_rate, abs=0.03)
