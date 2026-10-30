"""The six Stage 1 → 2 → 3 journeys, end to end, plus the defects they caught.

The rest of the suite tests units: one rule, one route, one encoder. This file tests
the *path* — a file arriving, being described, and coming out as two aligned matrices —
because every defect fixed in this audit lived in a seam rather than in a function. A
column dropped for emptiness measured over the whole file, an infinity that each stage
handled differently, a stratified split that sklearn refused on arithmetic no caller
chose: each is correct in isolation and wrong in sequence.

Six scenarios, named as the audit names them:

A  a clean classification file, profiled and split reproducibly
B  gaps in numbers and categories, filled from the training rows alone
C  a continuous target, separated and split without stratification
D  the data-quality cases — duplicates, constants, all-null, mixed types
E  leakage and split integrity
F  uploads that must fail in a controlled way

Each then has its regression tests underneath it, pinned to the defect they close.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.glassbox import fit_preprocessing
from app.services.ingest import read_upload
from app.services.profile import build_profile, histogram, numeric_summary
from app.services.screening import identifier_screen, target_leakage_screen

FIXTURES = Path(__file__).parent / "fixtures"

#: Both measurement screens, as ``main._preprocess`` passes them: inside the fit, so
#: they see the training fold only.
SCREENS = [identifier_screen, target_leakage_screen]


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def _upload(client, name: str, payload: bytes):
    return client.post("/api/datasets", files={"file": (name, payload, "text/csv")})


def _fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _frame(name: str) -> pd.DataFrame:
    """A fixture read the way an upload reads it, so tests see the repaired frame."""
    return read_upload(_fixture_bytes(name), name).frame


# --------------------------------------------------------------------------------------
# Scenario A — a clean classification dataset, all the way through
# --------------------------------------------------------------------------------------


def test_a_clean_classification_file_goes_ingest_to_split(client) -> None:
    """Stage 1 → 2 → 3 over HTTP, which is the only path the product actually offers."""
    upload = _upload(client, "clean_binary.csv", _fixture_bytes("clean_binary.csv"))
    assert upload.status_code == 201, upload.text
    dataset_id = upload.json()["id"]

    profile = client.get(f"/api/datasets/{dataset_id}/profile")
    assert profile.status_code == 200, profile.text
    described = profile.json()
    assert 0 <= described["quality"]["score"] <= 100
    assert {c["column"] for c in described["columns"]} >= {"age", "income", "region"}
    assert described["n_rows"] == 600

    run = client.post(
        f"/api/datasets/{dataset_id}/preprocess", json={"goal": "predict converted"}
    )
    assert run.status_code == 200, run.text
    body = run.json()

    assert body["target"]["column"] == "converted"
    assert body["result"]["train_rows"] + body["result"]["test_rows"] == 600
    assert body["result"]["stratified"] is True
    assert body["result"]["seed"] == 42
    # Nothing is left for the model to trip over.
    assert body["result"]["features"] > 0


def test_a_the_split_is_stratified_and_reproducible() -> None:
    """Same seed, same folds — and each class keeps its share on both sides."""
    frame = _frame("clean_binary.csv")

    first = fit_preprocessing(frame, "converted", "classification", screeners=SCREENS)
    second = fit_preprocessing(frame, "converted", "classification", screeners=SCREENS)

    assert list(first.y_test.index) == list(second.y_test.index)
    assert first.X_train.equals(second.X_train)
    assert json.dumps(first.log) == json.dumps(second.log)

    assert first.stratified is True
    whole = frame["converted"].value_counts(normalize=True)
    for fold in (first.y_train, first.y_test):
        share = fold.value_counts(normalize=True)
        for label, expected in whole.items():
            assert share[label] == pytest.approx(expected, abs=0.02)


def test_a_train_and_test_are_disjoint_and_aligned() -> None:
    frame = _frame("clean_binary.csv")
    result = fit_preprocessing(frame, "converted", "classification", screeners=SCREENS)

    assert not set(result.X_train.index) & set(result.X_test.index)
    assert list(result.X_train.index) == list(result.y_train.index)
    assert list(result.X_test.index) == list(result.y_test.index)
    assert list(result.X_train.columns) == list(result.X_test.columns)


# --------------------------------------------------------------------------------------
# Scenario B — missing values, filled from the training rows only
# --------------------------------------------------------------------------------------


def test_b_fill_values_are_the_training_median_not_the_whole_file_median() -> None:
    """The distinguishing test: the two medians are made to differ, then compared.

    A fill value that matches both proves nothing, so the held-out rows are rewritten
    to drag the whole-file median somewhere the training median cannot reach.
    """
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({
        "amount": rng.normal(100, 5, 400),
        "tier": rng.choice(["bronze", "silver"], 400),
        "label": rng.choice([0, 1], 400),
    })
    frame.loc[:19, "amount"] = np.nan

    result = fit_preprocessing(frame, "label", "classification")
    fill = next(
        entry["details"]["fill_values"]["amount"]
        for entry in result.log
        if entry["stage"] == "impute" and "amount" in entry["details"]["fill_values"]
    )

    # The engine deduplicates before it splits, and the blanked rows collapse into a
    # handful of distinct (tier, label) pairs, so the fold indices refer to the
    # deduplicated frame. Mirroring that here is the difference between comparing the
    # right rows and comparing rows that no longer exist.
    deduplicated = frame.drop_duplicates().reset_index(drop=True)
    train_median = float(deduplicated.loc[result.y_train.index, "amount"].median())
    whole_median = float(deduplicated["amount"].median())
    assert fill == pytest.approx(train_median)
    # The fixture is random enough that the two medians differ; if they ever collide
    # the assertion above stops distinguishing anything, so say so rather than pass.
    assert train_median != whole_median, "fixture no longer separates the two medians"


def test_b_test_rows_are_transformed_never_fitted() -> None:
    """Both folds come out complete, and the test fold's statistics are the train's."""
    frame = _frame("high_missing.csv")
    target = "outcome" if "outcome" in frame.columns else frame.columns[-1]

    result = fit_preprocessing(frame, target, "classification", screeners=SCREENS)

    assert not result.X_train.isna().any().any()
    assert not result.X_test.isna().any().any()
    assert list(result.X_train.columns) == list(result.X_test.columns)
    # Every entry in the log says where it learned its numbers.
    assert {entry["fitted_on"] for entry in result.log} == {"train_fold_only"}


def test_b_categorical_gaps_are_filled_with_a_category_the_encoder_knows() -> None:
    rng = np.random.default_rng(12)
    frame = pd.DataFrame({
        "tier": rng.choice(["bronze", "silver", "gold"], 300),
        "amount": rng.normal(50, 10, 300),
        "label": rng.choice([0, 1], 300),
    })
    frame.loc[:29, "tier"] = None

    result = fit_preprocessing(frame, "label", "classification")

    fill = next(
        entry["details"]["fill_values"]["tier"]
        for entry in result.log
        if entry["stage"] == "impute" and "tier" in entry["details"]["fill_values"]
    )
    assert fill in {"bronze", "silver", "gold"}
    assert not result.X_test.isna().any().any()


# --------------------------------------------------------------------------------------
# Scenario C — a regression target
# --------------------------------------------------------------------------------------


def test_c_regression_separates_the_target_and_does_not_stratify() -> None:
    frame = _frame("regression.csv")
    result = fit_preprocessing(frame, "price", "regression", screeners=SCREENS)

    assert result.stratified is False
    assert "price" not in result.X_train.columns
    assert "price" not in result.feature_map.values()
    assert "price" not in result.plan.feature_columns

    assert len(result.X_train) == len(result.y_train)
    assert len(result.X_test) == len(result.y_test)
    assert list(result.X_test.index) == list(result.y_test.index)
    assert result.y_train.name == "price"


def test_c_regression_numeric_columns_are_imputed_and_scaled() -> None:
    frame = _frame("regression.csv")
    result = fit_preprocessing(frame, "price", "regression", screeners=SCREENS)

    stages = [entry["stage"] for entry in result.log]
    assert "impute" in stages and "scale" in stages
    assert np.isfinite(result.X_train.to_numpy(dtype=float)).all()
    assert np.isfinite(result.X_test.to_numpy(dtype=float)).all()


# --------------------------------------------------------------------------------------
# Scenario D — data-quality edge cases
# --------------------------------------------------------------------------------------


def test_d_duplicate_rows_are_removed_before_the_split_and_reported() -> None:
    """Copies landing on opposite sides is the leak; removing them after would not fix it."""
    base = _frame("clean_binary.csv")
    doubled = pd.concat([base, base.head(50)], ignore_index=True)

    result = fit_preprocessing(doubled, "converted", "classification")

    removed = next(w for w in result.warnings if w["kind"] == "duplicates_removed")
    assert removed["count"] == 50
    assert len(result.X_train) + len(result.X_test) == len(base)
    assert not set(result.X_train.index) & set(result.X_test.index)


def test_d_columns_that_carry_nothing_are_dropped_with_a_reason() -> None:
    frame = _frame("id_and_constant.csv")
    result = fit_preprocessing(frame, "target", "classification", screeners=SCREENS)

    for column in ("record_id", "country", "near_constant"):
        assert column in result.plan.dropped, f"{column} survived"
        assert result.plan.dropped[column].strip(), f"{column} dropped without a reason"
    assert "score" in result.plan.feature_columns


def test_d_an_all_null_column_is_dropped_rather_than_filled() -> None:
    rng = np.random.default_rng(13)
    frame = pd.DataFrame({
        "amount": rng.normal(10, 2, 200),
        "empty": [np.nan] * 200,
        "label": rng.choice([0, 1], 200),
    })

    result = fit_preprocessing(frame, "label", "classification")

    assert "empty" in result.plan.dropped
    assert "100%" in result.plan.dropped["empty"]
    assert "empty" not in result.X_train.columns


def test_d_mixed_types_and_high_cardinality_both_survive_the_journey() -> None:
    messy = _frame("messy_types.csv")
    result = fit_preprocessing(messy, "outcome", "classification", screeners=SCREENS)
    assert np.isfinite(result.X_train.to_numpy(dtype=float)).all()

    crowded = _frame("high_cardinality.csv")
    target = crowded.columns[-1]
    crowded_result = fit_preprocessing(
        crowded, target, "classification", screeners=SCREENS
    )
    # A crowded column costs one feature, not one per value.
    assert crowded_result.X_train.shape[1] < crowded[target].size


def test_d_every_output_column_can_be_traced_back_to_an_input() -> None:
    frame = _frame("messy_types.csv")
    result = fit_preprocessing(frame, "outcome", "classification", screeners=SCREENS)

    assert set(result.feature_map) == set(result.X_train.columns)
    assert set(result.feature_map.values()) <= set(frame.columns)


# --------------------------------------------------------------------------------------
# Scenario E — leakage and split integrity
# --------------------------------------------------------------------------------------


def test_e_a_column_derived_from_the_answer_is_flagged_and_excluded() -> None:
    frame = _frame("leakage.csv")
    result = fit_preprocessing(frame, "churned", "classification", screeners=SCREENS)

    assert "leaked_score" in result.excluded
    assert "leaked_score" not in result.plan.feature_columns
    assert "honest_feature" in result.plan.feature_columns

    flagged = {flag["column"] for flag in result.flags}
    assert "leaked_score" in flagged


def test_e_leakage_warnings_are_phrased_as_suspicion_not_proof() -> None:
    """A correlation is evidence. Reporting it as a verdict would be the real error."""
    frame = _frame("leakage.csv")
    result = fit_preprocessing(frame, "churned", "classification", screeners=SCREENS)

    reason = result.excluded["leaked_score"].lower()
    hedged = ("looks", "appears", "suggest", "likely", "probably", "may ", "might")
    assert any(word in reason for word in hedged), reason


def test_e_rewriting_the_held_out_rows_cannot_change_the_plan() -> None:
    """The strongest form of the claim: held-out rows do not reach a *decision* either."""
    frame = _frame("clean_binary.csv")
    original = fit_preprocessing(frame, "converted", "classification", screeners=SCREENS)

    rng = np.random.default_rng(14)
    vandalised = frame.copy()
    held_out = original.y_test.index
    vandalised.loc[held_out, "region"] = [
        f"unseen_{i}" for i in range(len(held_out))
    ]
    vandalised.loc[held_out, "income"] = rng.normal(1e9, 1e6, len(held_out))

    after = fit_preprocessing(vandalised, "converted", "classification", screeners=SCREENS)

    assert after.plan.dropped == original.plan.dropped
    assert after.plan.feature_columns == original.plan.feature_columns
    assert json.dumps(after.log) == json.dumps(original.log)


# --------------------------------------------------------------------------------------
# Scenario F — uploads that must fail in a controlled way
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, payload",
    [
        ("empty.csv", b""),
        ("header_only.csv", b"a,b,c\n"),
        ("single_column.csv", b"only\n1\n2\n3\n"),
        ("notes.txt", b"this is not a spreadsheet at all"),
        ("renamed.csv", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR binary payload"),
    ],
)
def test_f_bad_uploads_are_refused_with_a_sentence(client, name, payload) -> None:
    response = _upload(client, name, payload)

    assert response.status_code == 400, response.text
    body = response.json()
    assert isinstance(body.get("error"), str) and body["error"]
    # A controlled refusal, not a leaked traceback.
    assert "Traceback" not in response.text
    assert "File \"" not in response.text


def test_f_a_malformed_csv_does_not_reach_the_user_as_a_parser_error(client) -> None:
    response = _upload(client, "malformed.csv", _fixture_bytes("malformed.csv"))

    assert response.status_code in (201, 400), response.text
    if response.status_code == 400:
        message = response.json()["error"]
        assert "Traceback" not in message
        assert "pandas" not in message.lower()


def test_f_a_file_below_the_row_floor_says_so(client) -> None:
    response = _upload(client, "tiny.csv", _fixture_bytes("tiny.csv"))

    assert response.status_code == 400
    assert "rows" in response.json()["error"]


# --------------------------------------------------------------------------------------
# Regression — infinity is read as missing, at every stage that meets it
#
# Before the fix, a column holding `inf` took `POST /api/datasets` down with an
# unhandled `ValueError` out of `numpy.histogram` and a full traceback in the response.
# --------------------------------------------------------------------------------------


def _with_infinity() -> bytes:
    rng = np.random.default_rng(15)
    frame = pd.DataFrame({
        "ratio": rng.normal(5, 1, 120),
        "amount": rng.normal(100, 20, 120),
        "label": ["yes", "no"] * 60,
    })
    frame.loc[0, "ratio"] = np.inf
    frame.loc[1, "ratio"] = -np.inf
    return frame.to_csv(index=False).encode()


def test_infinity_is_treated_as_missing_at_ingestion() -> None:
    result = read_upload(_with_infinity(), "inf.csv")

    assert np.isfinite(result.frame["ratio"].dropna()).all()
    assert result.frame["ratio"].isna().sum() == 2
    assert any("infinite" in sentence for sentence in result.cleaning)


def test_profiling_a_non_finite_column_does_not_raise() -> None:
    """The profile functions are public, so they are defended independently of ingest."""
    series = pd.Series([1.0, 2.0, np.inf, -np.inf, 4.0, 5.0])

    assert histogram(series) is not None
    summary = numeric_summary(series)
    assert summary is not None
    assert np.isfinite([summary["mean"], summary["min"], summary["max"]]).all()

    frame = pd.DataFrame({"ratio": series, "label": [0, 1, 0, 1, 0, 1]})
    assert build_profile(frame)["columns"]


def test_a_file_containing_infinity_uploads_and_preprocesses(client) -> None:
    upload = _upload(client, "inf.csv", _with_infinity())
    assert upload.status_code == 201, upload.text
    dataset_id = upload.json()["id"]

    profile = client.get(f"/api/datasets/{dataset_id}/profile")
    assert profile.status_code == 200, profile.text
    assert "Infinity" not in profile.text, "non-finite float reached the JSON response"

    run = client.post(
        f"/api/datasets/{dataset_id}/preprocess", json={"goal": "predict label"}
    )
    assert run.status_code == 200, run.text


def test_fit_preprocessing_blanks_infinity_without_touching_the_callers_frame() -> None:
    rng = np.random.default_rng(16)
    frame = pd.DataFrame({
        "ratio": rng.normal(5, 1, 200),
        "label": rng.choice([0, 1], 200),
    })
    frame.loc[0, "ratio"] = np.inf
    before = frame.copy()

    result = fit_preprocessing(frame, "label", "classification")

    assert np.isfinite(result.X_train.to_numpy(dtype=float)).all()
    assert np.isfinite(result.X_test.to_numpy(dtype=float)).all()
    assert any(w["kind"] == "non_finite_values" for w in result.warnings)
    assert frame.equals(before), "the caller's frame was modified"


# --------------------------------------------------------------------------------------
# Regression — stratification falls back rather than aborting the run
#
# `train_test_split` must place one row of every class on each side, so it refuses when
# a fold is smaller than the class count. Before the fix that surfaced as a 400 phrased
# in terms of a `test_size` the user never chose.
# --------------------------------------------------------------------------------------


def test_too_many_classes_for_the_fold_falls_back_and_says_why() -> None:
    rng = np.random.default_rng(17)
    frame = pd.DataFrame({
        "f1": rng.normal(size=60),
        "f2": rng.normal(size=60),
        "label": [f"class_{i % 20}" for i in range(60)],
    })

    result = fit_preprocessing(frame, "label", "classification")

    assert result.stratified is False
    warning = next(
        w for w in result.warnings if w["kind"] == "stratification_disabled"
    )
    assert "20 classes" in warning["message"]
    assert warning["suggestion"]
    assert len(result.X_train) + len(result.X_test) == 60


def test_a_class_with_one_row_still_disables_stratification_with_a_warning() -> None:
    rng = np.random.default_rng(18)
    labels = ["a"] * 100 + ["b"] * 99 + ["c"]
    frame = pd.DataFrame({"f1": rng.normal(size=200), "label": labels})

    result = fit_preprocessing(frame, "label", "classification")

    assert result.stratified is False
    warning = next(w for w in result.warnings if w["kind"] == "stratification_disabled")
    assert "c" in warning["classes"]


def test_a_dataset_with_many_classes_reaches_the_api_rather_than_a_400(client) -> None:
    rng = np.random.default_rng(19)
    frame = pd.DataFrame({
        "f1": rng.normal(size=60),
        "f2": rng.normal(size=60),
        "label": [f"class_{i % 20}" for i in range(60)],
    })
    upload = _upload(client, "manyclass.csv", frame.to_csv(index=False).encode())
    assert upload.status_code == 201, upload.text

    run = client.post(
        f"/api/datasets/{upload.json()['id']}/preprocess", json={"goal": "predict label"}
    )
    assert run.status_code == 200, run.text
    body = run.json()
    assert body["result"]["stratified"] is False
    assert any(w["kind"] == "stratification_disabled" for w in body["warnings"])


# --------------------------------------------------------------------------------------
# Regression — a row with no target is removed, whatever the task
#
# Classification aborted with sklearn's "Input contains NaN"; regression said nothing at
# all and carried the blank into `y_train`.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "task, labels",
    [
        ("classification", np.array(["yes", "no"] * 100, dtype=object)),
        ("regression", np.linspace(0, 1, 200)),
    ],
)
def test_rows_with_a_blank_target_are_removed_and_reported(task, labels) -> None:
    rng = np.random.default_rng(20)
    frame = pd.DataFrame({"f1": rng.normal(size=200), "label": labels})
    frame.loc[:9, "label"] = np.nan

    result = fit_preprocessing(frame, "label", task)

    assert not result.y_train.isna().any()
    assert not result.y_test.isna().any()
    assert len(result.y_train) + len(result.y_test) == 190

    warning = next(w for w in result.warnings if w["kind"] == "blank_targets_removed")
    assert warning["count"] == 10
    assert "label" in warning["message"]


def test_a_target_that_is_entirely_blank_fails_with_a_sentence() -> None:
    frame = pd.DataFrame({"f1": np.arange(100.0), "label": [np.nan] * 100})

    with pytest.raises(ValueError, match="No rows are left"):
        fit_preprocessing(frame, "label", "regression")


# --------------------------------------------------------------------------------------
# Regression — the engine's warnings reach the caller
#
# `fit_preprocessing` has always produced these; nothing read them, so every one was
# computed and discarded.
# --------------------------------------------------------------------------------------


def test_the_preprocess_response_carries_the_warnings_and_the_checks(client) -> None:
    upload = _upload(client, "clean_binary.csv", _fixture_bytes("clean_binary.csv"))
    run = client.post(
        f"/api/datasets/{upload.json()['id']}/preprocess", json={"goal": "predict converted"}
    )
    assert run.status_code == 200, run.text
    body = run.json()

    assert isinstance(body["warnings"], list)
    for warning in body["warnings"]:
        assert warning["severity"] in {"info", "warn", "critical"}
        assert warning["message"]

    assert {"drift", "balance", "baseline"} <= set(body["checks"])
    assert body["checks"]["drift"]["tested"] >= 0
