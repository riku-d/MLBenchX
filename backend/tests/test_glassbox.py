"""Tests for the glass-box preprocessing engine.

The suite is organised around the four module invariants. The leakage tests
(:func:`test_no_transformer_ever_saw_a_test_row` and
:func:`test_imputation_values_come_from_train_only`) are the ones that matter most: they
are the difference between an honest score and an optimistic one, and both are written so
that a broken test fails loudly rather than passing vacuously.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from app.services.glassbox import (
    ONEHOT_MAX_CARDINALITY,
    ColumnProfile,
    FrequencyEncoder,
    build_feature_map,
    build_plan,
    export_script,
    fit_preprocessing,
    format_log,
    profile_frame,
)

N_ROWS = 200
TARGET = "placed"
TASK = "classification"


@pytest.fixture
def frame() -> pd.DataFrame:
    """A mixed-type frame exercising every routing branch.

    Contents, and what each column is here to prove:
      student_id     unique identifier          -> kind "text", dropped
      campus         single value               -> kind "constant", dropped
      enrolled_on    timestamps                 -> kind "datetime", calendar features
      mostly_missing 70% blank                  -> dropped for missingness
      department     5 categories               -> low_card, one-hot
      gender         2 categories               -> binary, ordinal
      city           40 categories              -> high_card, frequency
      attendance     numeric with blanks        -> numeric
      prev_score     numeric with blanks        -> numeric
      placed         0/1 label                  -> target
    """
    # Seed chosen so that the training-fold median of `attendance` genuinely differs
    # from the whole-frame median; test 6 asserts that gap and would be vacuous without
    # it. Some seeds make the two coincide by chance.
    rng = np.random.default_rng(11)

    department = np.array(["CSE", "ECE", "ME", "CE", "IT"])
    city = np.array([f"city_{i:02d}" for i in range(40)])

    # Deliberately skewed and high-resolution so that the training-fold median and the
    # whole-frame median differ; test 6 asserts that difference, and a fixture where the
    # two happened to coincide would make that test prove nothing.
    attendance = np.round(rng.beta(6.0, 2.0, N_ROWS) * 100.0, 3)
    prev_score = np.round(rng.gamma(shape=9.0, scale=6.0, size=N_ROWS), 3)

    df = pd.DataFrame({
        "student_id": [f"S{i:05d}" for i in range(N_ROWS)],
        "campus": ["MAIN"] * N_ROWS,
        "enrolled_on": pd.to_datetime("2022-01-01") + pd.to_timedelta(
            rng.integers(0, 900, N_ROWS), unit="D"
        ),
        "mostly_missing": np.where(rng.random(N_ROWS) < 0.70, np.nan, rng.normal(size=N_ROWS)),
        "department": rng.choice(department, N_ROWS),
        "gender": rng.choice(np.array(["F", "M"]), N_ROWS),
        "city": rng.choice(city, N_ROWS),
        "attendance": attendance,
        "prev_score": prev_score,
        TARGET: rng.choice([0, 1], N_ROWS, p=[0.4, 0.6]),
    })

    # Inject blanks into the two numerics and one categorical.
    df.loc[rng.choice(N_ROWS, 29, replace=False), "attendance"] = np.nan
    df.loc[rng.choice(N_ROWS, 12, replace=False), "prev_score"] = np.nan
    df.loc[rng.choice(N_ROWS, 6, replace=False), "department"] = np.nan
    return df


@pytest.fixture
def result(frame: pd.DataFrame):
    return fit_preprocessing(frame, target=TARGET, task_type=TASK)


# --------------------------------------------------------------------------------------
# Profiling and planning
# --------------------------------------------------------------------------------------


def test_1_profile_frame_assigns_the_correct_kind(frame: pd.DataFrame) -> None:
    profiles = profile_frame(frame)
    expected = {
        "student_id": "text",
        "campus": "constant",
        "enrolled_on": "datetime",
        "mostly_missing": "numeric",
        "department": "categorical",
        "gender": "categorical",
        "city": "categorical",
        "attendance": "numeric",
        "prev_score": "numeric",
        # An integer 0/1 label profiles as numeric: the rules check dtype before
        # cardinality. It never reaches routing anyway, because build_plan drops the
        # target first.
        TARGET: "numeric",
    }
    assert {name: p.kind for name, p in profiles.items()} == expected

    # Spot-check the measurements the routing rules depend on.
    assert profiles["attendance"].n_missing == 29
    assert profiles["city"].n_unique == 40
    assert profiles["student_id"].unique_rate == pytest.approx(1.0)
    assert profiles["mostly_missing"].missing_rate > 0.60


def test_2_build_plan_routes_by_cardinality(frame: pd.DataFrame) -> None:
    profiles = profile_frame(frame.drop(columns=[TARGET]))
    plan = build_plan(profiles, target=TARGET, task_type=TASK)

    assert plan.numeric == ["attendance", "prev_score"]
    assert plan.binary == ["gender"]
    assert plan.low_card == ["department"]
    assert plan.high_card == ["city"]
    assert plan.datetime == ["enrolled_on"]

    assert set(plan.dropped) == {"student_id", "campus", "mostly_missing"}
    assert "constant" in plan.dropped["campus"]
    assert "identifier" in plan.dropped["student_id"]

    # Every surviving column is routed exactly once.
    routed = plan.feature_columns
    assert len(routed) == len(set(routed))
    assert set(routed) | set(plan.dropped) == set(profiles)


def test_2b_cardinality_boundaries_are_exact() -> None:
    """The one-hot ceiling is inclusive, so 15 one-hots and 16 frequency-encodes."""
    def profile(name: str, n_unique: int) -> ColumnProfile:
        return ColumnProfile(
            name=name, kind="categorical", dtype="object", n_missing=0,
            missing_rate=0.0, n_unique=n_unique, unique_rate=n_unique / 1000,
        )

    profiles = {
        "two": profile("two", 2),
        "three": profile("three", 3),
        "at_ceiling": profile("at_ceiling", ONEHOT_MAX_CARDINALITY),
        "over_ceiling": profile("over_ceiling", ONEHOT_MAX_CARDINALITY + 1),
    }
    plan = build_plan(profiles, target="y", task_type="regression")
    assert plan.binary == ["two"]
    assert plan.low_card == ["three", "at_ceiling"]
    assert plan.high_card == ["over_ceiling"]


def test_3_build_plan_is_pure(frame: pd.DataFrame) -> None:
    profiles = profile_frame(frame.drop(columns=[TARGET]))
    first = build_plan(profiles, target=TARGET, task_type=TASK)
    second = build_plan(profiles, target=TARGET, task_type=TASK)
    assert first == second
    assert first.steps == second.steps


def test_4_mostly_missing_columns_are_dropped_with_a_percentage(frame: pd.DataFrame) -> None:
    profiles = profile_frame(frame.drop(columns=[TARGET]))
    plan = build_plan(profiles, target=TARGET, task_type=TASK)

    reason = plan.dropped["mostly_missing"]
    assert "missing" in reason
    assert "%" in reason
    # The percentage quoted is the real one, not a rounded-to-nothing placeholder.
    actual = profiles["mostly_missing"].missing_rate * 100
    assert f"{actual:.0f}"[:2] in reason


def test_4b_caller_exclusions_win_over_generic_reasons(frame: pd.DataFrame) -> None:
    profiles = profile_frame(frame.drop(columns=[TARGET]))
    plan = build_plan(
        profiles,
        target=TARGET,
        task_type=TASK,
        excluded={"department": "leaks the outcome: recorded after placement"},
    )
    assert plan.dropped["department"] == "leaks the outcome: recorded after placement"
    assert "department" not in plan.low_card


# --------------------------------------------------------------------------------------
# Leakage — the critical tests
# --------------------------------------------------------------------------------------


def test_5_no_transformer_ever_saw_a_test_row(
    frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spy on every ``fit`` and prove none of them touched a held-out row."""
    seen_indices: list[pd.Index] = []

    def spy(original):
        def wrapper(self, X, y=None, **kwargs):
            if isinstance(X, pd.DataFrame):
                seen_indices.append(pd.Index(X.index))
            return original(self, X, y, **kwargs)
        return wrapper

    monkeypatch.setattr(SimpleImputer, "fit", spy(SimpleImputer.fit))
    monkeypatch.setattr(StandardScaler, "fit", spy(StandardScaler.fit))

    result = fit_preprocessing(frame, target=TARGET, task_type=TASK)

    # A silently broken spy would make the assertions below vacuous.
    assert len(seen_indices) >= 2, "no fit calls were intercepted; the spy is broken"

    test_index = pd.Index(result.X_test.index)
    assert len(test_index) > 0
    for index in seen_indices:
        overlap = index.intersection(test_index)
        assert len(overlap) == 0, f"a transformer was fitted on held-out rows: {list(overlap)[:5]}"


def test_6_imputation_values_come_from_train_only(frame: pd.DataFrame) -> None:
    """The logged fill value is the *training* median, and that differs from the whole-frame one."""
    result = fit_preprocessing(frame, target=TARGET, task_type=TASK)

    raw = frame.drop(columns=[TARGET])
    train_rows = raw.loc[result.X_train.index]

    train_median = float(train_rows["attendance"].median())
    full_median = float(raw["attendance"].median())

    # Without this, the assertion below would hold even for a leaky implementation.
    assert train_median != pytest.approx(full_median), (
        "fixture is degenerate: train and full medians coincide, so this test cannot "
        "distinguish a correct implementation from a leaky one"
    )

    entry = next(e for e in result.log if e["stage"] == "impute" and "attendance" in e["columns"])
    logged = entry["details"]["fill_values"]["attendance"]

    assert logged == pytest.approx(train_median)
    assert logged != pytest.approx(full_median)
    assert entry["details"]["blanks_filled"]["attendance"] == int(
        train_rows["attendance"].isna().sum()
    )


# --------------------------------------------------------------------------------------
# Log integrity
# --------------------------------------------------------------------------------------


def test_7_logged_scaler_values_match_the_fitted_scaler(result) -> None:
    scaler = result.pipeline.named_transformers_["numeric"].named_steps["scale"]
    entry = next(e for e in result.log if e["stage"] == "scale")

    for i, column in enumerate(scaler.feature_names_in_):
        assert entry["details"]["mean"][column] == pytest.approx(scaler.mean_[i], abs=1e-6)
        assert entry["details"]["scale"][column] == pytest.approx(scaler.scale_[i], abs=1e-6)


def test_8_every_entry_is_explained_and_fitted_on_train_only(result) -> None:
    assert result.log, "the log is empty"
    for entry in result.log:
        assert entry["fitted_on"] == "train_fold_only"
        assert isinstance(entry["explanation"], str)
        assert entry["explanation"].strip()
        assert entry["reason"].strip()
        assert entry["columns"]


def test_9_step_ids_are_contiguous_from_one(result) -> None:
    assert [e["step"] for e in result.log] == list(range(1, len(result.log) + 1))


def test_9b_log_is_json_serialisable_and_deterministic(frame: pd.DataFrame) -> None:
    import json

    first = fit_preprocessing(frame, target=TARGET, task_type=TASK)
    second = fit_preprocessing(frame, target=TARGET, task_type=TASK)
    assert json.dumps(first.log) == json.dumps(second.log)


def test_9c_a_group_without_blanks_says_so(result) -> None:
    """`gender` is the whole binary group and has no blanks, so the entry must say so."""
    entry = next(e for e in result.log if e["stage"] == "impute" and e["columns"] == ["gender"])
    assert entry["explanation"] == "No blanks to fill in these columns."
    assert entry["details"]["blanks_filled"] == {"gender": 0}


# --------------------------------------------------------------------------------------
# Feature map
# --------------------------------------------------------------------------------------


def test_10_every_transformed_column_is_in_the_feature_map(result) -> None:
    assert set(result.X_train.columns) == set(result.feature_map)
    assert set(result.X_test.columns) == set(result.feature_map)
    assert set(result.feature_map.values()) <= set(result.plan.feature_columns)


def test_11_one_hot_children_map_back_to_one_source(result) -> None:
    children = [name for name, source in result.feature_map.items() if source == "department"]
    encoder = result.pipeline.named_transformers_["low_card"].named_steps["encode"]
    n_categories = len(encoder.categories_[0])

    assert n_categories == 5, "fixture should give department 5 categories"
    assert len(children) == n_categories
    assert all(name.startswith("department_") for name in children)

    # And the single-column encodings are not accidentally renamed.
    assert result.feature_map["city_freq"] == "city"
    assert result.feature_map["gender"] == "gender"
    assert result.feature_map["attendance"] == "attendance"


# --------------------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------------------


def test_12_train_and_test_align_and_contain_no_blanks(result) -> None:
    assert list(result.X_train.columns) == list(result.X_test.columns)
    assert not result.X_train.isna().any().any()
    assert not result.X_test.isna().any().any()
    assert len(result.X_train) == len(result.y_train)
    assert len(result.X_test) == len(result.y_test)
    # Held-out rows must not appear in training.
    assert not set(result.X_train.index) & set(result.X_test.index)


def test_13_unseen_categories_are_handled_without_changing_width(result, frame) -> None:
    fresh = frame.loc[:9, list(result.pipeline.feature_names_in_)].copy()
    fresh["department"] = "ARCH"        # never seen: one-hot -> all zeros
    fresh["city"] = "city_99"           # never seen: frequency -> 0.0
    fresh["gender"] = "X"               # never seen: ordinal -> -1

    out = result.pipeline.transform(fresh)

    assert list(out.columns) == list(result.X_train.columns)
    assert out.shape[1] == result.X_train.shape[1]
    assert not out.isna().any().any()

    onehot_columns = [c for c, src in result.feature_map.items() if src == "department"]
    assert (out[onehot_columns].to_numpy() == 0).all()
    assert (out["city_freq"] == 0.0).all()
    assert (out["gender"] == -1).all()


def test_13b_frequency_encoder_learns_and_generalises() -> None:
    train = pd.DataFrame({"c": ["a", "a", "a", "b"]})
    encoder = FrequencyEncoder().fit(train)

    assert encoder.frequencies_["c"] == {"a": 0.75, "b": 0.25}
    assert list(encoder.get_feature_names_out()) == ["c_freq"]

    out = encoder.transform(pd.DataFrame({"c": ["a", "b", "zzz"]}))
    assert list(out["c_freq"]) == [0.75, 0.25, 0.0]


def test_14_same_seed_gives_the_same_split_and_keeps_class_balance(frame: pd.DataFrame) -> None:
    first = fit_preprocessing(frame, target=TARGET, task_type=TASK, seed=42)
    second = fit_preprocessing(frame, target=TARGET, task_type=TASK, seed=42)

    assert list(first.X_train.index) == list(second.X_train.index)
    assert list(first.X_test.index) == list(second.X_test.index)

    overall = frame[TARGET].mean()
    assert first.y_train.mean() == pytest.approx(overall, abs=0.03)
    assert first.y_test.mean() == pytest.approx(overall, abs=0.03)


def test_15_exported_script_compiles_and_splits_before_fitting(result) -> None:
    source = export_script(result, csv="data/students.csv", exp_id="exp-0042")

    compile(source, "<exported>", "exec")

    assert source.index("train_test_split") < source.index("fit_transform")
    assert "random_state=SEED" in source
    assert "FrequencyEncoder" in source, "the plan uses it, so the export must define it"
    assert "data/students.csv" in source
    for column in result.plan.feature_columns:
        assert repr(column) in source


def test_16_regression_task_skips_stratification(frame: pd.DataFrame) -> None:
    numeric_target = frame.assign(salary=np.linspace(3.0, 30.0, len(frame))).drop(columns=[TARGET])
    result = fit_preprocessing(numeric_target, target="salary", task_type="regression")

    assert result.plan.task_type == "regression"
    assert len(result.X_test) == 40
    assert not result.X_train.isna().any().any()
    with pytest.raises(ValueError, match="classification or regression"):
        fit_preprocessing(numeric_target, target="salary", task_type="clustering")


def test_17_a_frame_with_only_numerics_still_works(frame: pd.DataFrame) -> None:
    """Groups with no columns must be omitted rather than producing empty blocks."""
    numeric_only = frame[["attendance", "prev_score", TARGET]]
    result = fit_preprocessing(numeric_only, target=TARGET, task_type=TASK)

    assert list(result.pipeline.named_transformers_) == ["numeric"]
    assert list(result.X_train.columns) == ["attendance", "prev_score"]
    assert [e["stage"] for e in result.log] == ["impute", "scale"]
    assert set(result.feature_map) == {"attendance", "prev_score"}


def test_18_a_frame_with_no_usable_columns_fails_with_the_reasons(frame: pd.DataFrame) -> None:
    unusable = frame[["student_id", "campus", TARGET]]
    with pytest.raises(ValueError) as excinfo:
        fit_preprocessing(unusable, target=TARGET, task_type=TASK)
    message = str(excinfo.value)
    assert "campus" in message and "student_id" in message


def test_15b_report_reads_as_english(result) -> None:
    text = format_log(result.log)
    lines = text.splitlines()
    assert len(lines) == len(result.log)
    assert all(line.startswith("[") for line in lines)
    assert "median" in text
    assert "cannot leak" in text
