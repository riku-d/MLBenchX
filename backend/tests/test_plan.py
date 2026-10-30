"""The plan: proposed from the data, reviewed by a person, versioned, never mutated.

The plan is the object the product's central claim rests on. "Every automated
recommendation is visible, explained, optional and reversible" is a statement about
this: each column's decision is shown with the numbers behind it, each can be changed,
and changing one does not disturb the record of what was decided before.

Three properties get the most attention here because each is easy to lose:

* **Exhaustive.** Every column appears, including the ones nothing happens to. A plan
  that lists only the interesting columns leaves a reader unable to tell "nothing needed
  doing" from "we forgot".
* **Immutable once stored.** Editing writes the next version. A run that named version 2
  still means version 2 a month later.
* **Measured on the training fold.** The statistics a user reviews describe the rows the
  pipeline will actually be fitted on — which is only true if the planner and the
  executor prepare and split the data identically.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.plan import ACTIONS_FOR_KIND, GlobalOptions, PreprocessingPlan, prepare, propose
from app.runner import execute
from app.services.ingest import read_upload
from app.services.screening import identifier_screen, target_leakage_screen

FIXTURES = Path(__file__).parent / "fixtures"
TITANIC = FIXTURES / "titanic.csv"
SCREENS = [identifier_screen, target_leakage_screen]


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def frame() -> pd.DataFrame:
    return read_upload(TITANIC.read_bytes(), "titanic.csv").frame


@pytest.fixture
def dataset_id(client) -> str:
    with TITANIC.open("rb") as handle:
        response = client.post(
            "/api/datasets", files={"file": ("titanic.csv", handle, "text/csv")}
        )
    assert response.status_code == 201, response.text
    return response.json()["id"]


# --------------------------------------------------------------------------------------
# Proposing
# --------------------------------------------------------------------------------------


def test_every_column_gets_a_decision_and_a_reason(frame) -> None:
    plan = propose(frame, "Survived", "binary classification", screeners=SCREENS)

    assert {entry.column for entry in plan.columns} == {str(c) for c in frame.columns}
    for entry in plan.columns:
        assert entry.action, f"{entry.column} has no action"
        assert entry.reason.strip(), f"{entry.column} has an action but no reason"


def test_a_reason_quotes_the_numbers_that_drove_it(frame) -> None:
    """A reason without the figures is a restatement of the action, not a justification."""
    plan = propose(frame, "Survived", "binary classification", screeners=SCREENS)

    age = plan.for_column("Age")
    assert age is not None and age.action == "impute_median"
    assert any(char.isdigit() for char in age.reason)
    assert age.stats["n_missing"] > 0
    assert "median" in age.stats


def test_the_target_is_never_a_feature(frame) -> None:
    plan = propose(frame, "Survived", "binary classification", screeners=SCREENS)

    target = plan.for_column("Survived")
    assert target is not None
    assert target.action == "passthrough"
    assert target.alternatives == []
    assert "Survived" not in [entry.column for entry in plan.kept()]


def test_an_identifier_is_dropped_by_the_screen(frame) -> None:
    plan = propose(frame, "Survived", "binary classification", screeners=SCREENS)

    passenger_id = plan.for_column("PassengerId")
    assert passenger_id is not None
    assert passenger_id.action == "drop"


def test_every_column_offers_only_actions_its_kind_allows(frame) -> None:
    plan = propose(frame, "Survived", "binary classification", screeners=SCREENS)

    for entry in plan.columns:
        if entry.column == plan.target:
            continue
        assert set(entry.alternatives) <= set(ACTIONS_FOR_KIND.get(entry.kind, []))


def test_rare_categories_and_indicators_are_recorded_on_the_column(frame) -> None:
    plan = propose(frame, "Survived", "binary classification", screeners=SCREENS)

    age = plan.for_column("Age")
    assert age is not None and age.add_indicator is True
    assert "blank" in age.reason or "missing" in age.reason


def test_the_plan_round_trips_through_json(frame) -> None:
    plan = propose(frame, "Survived", "binary classification", screeners=SCREENS)

    rebuilt = PreprocessingPlan.model_validate_json(plan.model_dump_json())
    assert rebuilt.model_dump() == plan.model_dump()


# --------------------------------------------------------------------------------------
# The planner and the executor must agree about which rows are the training rows
# --------------------------------------------------------------------------------------


def test_preparation_is_shared_so_both_sides_split_the_same_rows() -> None:
    """The defect this closes: the planner measured rows the executor never fitted on.

    `propose` split the frame as given while `execute` deduplicated and dropped blank
    targets first, so the two training folds held different rows — and every statistic
    the user reviewed described data the pipeline never saw.
    """
    rng = np.random.default_rng(3)
    base = pd.DataFrame({
        "amount": rng.normal(100, 10, 300),
        "tier": rng.choice(["bronze", "silver", "gold"], 300),
        "label": rng.choice([0, 1], 300),
    })
    # Give both preparation steps something to do.
    messy = pd.concat([base, base.head(40)], ignore_index=True)
    messy.loc[:9, "label"] = np.nan

    prepared, records = prepare(messy, "label")
    assert {record["kind"] for record in records} == {
        "blank_targets_removed",
        "duplicates_removed",
    }

    plan = propose(messy, "label", "binary classification")
    result = execute(messy, plan, run_id="agreement")

    # The planner measured `rows_measured` rows; the executor fitted on `train_rows`.
    measured = plan.for_column("amount").stats["rows_measured"]
    assert measured == len(result.X_train), (
        "the planner and the executor disagree about the size of the training fold"
    )
    assert len(result.X_train) + len(result.X_test) == len(prepared)


def test_a_fold_too_small_for_its_classes_plans_without_aborting() -> None:
    """The shared stratification rule, reached through the planner."""
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({
        "f1": rng.normal(size=60),
        "label": [f"class_{i % 20}" for i in range(60)],
    })

    plan = propose(frame, "label", "multiclass classification")

    assert any(w["kind"] == "stratification_disabled" for w in plan.warnings)
    result = execute(frame, plan, run_id="manyclass")
    assert result.stratified is False
    assert len(result.X_train) + len(result.X_test) == 60


# --------------------------------------------------------------------------------------
# Editing
# --------------------------------------------------------------------------------------


def test_an_override_is_validated_against_the_columns_kind(frame) -> None:
    plan = propose(frame, "Survived", "binary classification", screeners=SCREENS)

    with pytest.raises(ValueError, match="not a valid action"):
        plan.override("Age", "encode_onehot")

    plan.override("Age", "impute_mean")
    age = plan.for_column("Age")
    assert age.action == "impute_mean"
    assert age.overridden is True
    assert "you chose" in age.reason.lower()


def test_an_override_of_an_unknown_column_is_refused(frame) -> None:
    plan = propose(frame, "Survived", "binary classification", screeners=SCREENS)
    with pytest.raises(KeyError):
        plan.override("NoSuchColumn", "drop")


def test_an_override_changes_what_actually_runs(frame) -> None:
    plan = propose(frame, "Survived", "binary classification", screeners=SCREENS)
    before = execute(frame, plan, run_id="before")

    plan.override("Age", "drop")
    after = execute(frame, plan, run_id="after")

    assert before.output_hash != after.output_hash
    assert not any(name.startswith("Age") for name in after.X_train.columns)


# --------------------------------------------------------------------------------------
# The plan API
# --------------------------------------------------------------------------------------


def test_proposing_creates_version_one(client, dataset_id) -> None:
    response = client.post(
        f"/api/datasets/{dataset_id}/plan", json={"goal": "predict Survived"}
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["version"] == 1
    assert body["origin"] == "proposed"
    assert body["target"] == "Survived"
    assert body["task"] == "binary classification"
    assert body["n_overridden"] == 0
    assert len(body["columns"]) == body["n_columns"]
    # The interface needs the action vocabulary with the plan, not a second copy of it.
    assert body["actions_for_kind"] == ACTIONS_FOR_KIND


def test_editing_writes_the_next_version_and_leaves_the_first_alone(client, dataset_id) -> None:
    first = client.post(
        f"/api/datasets/{dataset_id}/plan", json={"goal": "predict Survived"}
    ).json()

    edited = client.put(
        f"/api/plans/{first['id']}", json={"columns": {"Age": {"action": "impute_mean"}}}
    )
    assert edited.status_code == 201, edited.text
    body = edited.json()

    assert body["version"] == 2
    assert body["origin"] == "edited"
    assert body["parent_id"] == first["id"]
    assert body["n_overridden"] == 1

    unchanged = client.get(f"/api/plans/{first['id']}").json()
    assert unchanged["version"] == 1
    assert unchanged["n_overridden"] == 0
    assert next(c for c in unchanged["columns"] if c["column"] == "Age")["action"] == (
        "impute_median"
    )


def test_an_invalid_edit_is_refused_before_anything_is_stored(client, dataset_id) -> None:
    first = client.post(
        f"/api/datasets/{dataset_id}/plan", json={"goal": "predict Survived"}
    ).json()

    bad = client.put(
        f"/api/plans/{first['id']}", json={"columns": {"Age": {"action": "encode_onehot"}}}
    )
    assert bad.status_code == 400, bad.text
    assert "not a valid action" in bad.json()["error"]

    # No version was written.
    assert client.get(f"/api/datasets/{dataset_id}/plan").json()["version"] == 1


def test_editing_an_unknown_column_is_refused(client, dataset_id) -> None:
    first = client.post(
        f"/api/datasets/{dataset_id}/plan", json={"goal": "predict Survived"}
    ).json()

    bad = client.put(
        f"/api/plans/{first['id']}", json={"columns": {"Nope": {"action": "drop"}}}
    )
    assert bad.status_code == 400
    assert "Nope" in bad.json()["error"]


def test_global_options_can_be_edited_and_are_validated(client, dataset_id) -> None:
    first = client.post(
        f"/api/datasets/{dataset_id}/plan", json={"goal": "predict Survived"}
    ).json()

    ok = client.put(f"/api/plans/{first['id']}", json={"options": {"scaler": "robust"}})
    assert ok.status_code == 201
    assert ok.json()["options"]["scaler"] == "robust"

    bad = client.put(f"/api/plans/{first['id']}", json={"options": {"scaler": "magic"}})
    assert bad.status_code == 400


def test_every_version_stays_retrievable(client, dataset_id) -> None:
    first = client.post(
        f"/api/datasets/{dataset_id}/plan", json={"goal": "predict Survived"}
    ).json()
    second = client.put(
        f"/api/plans/{first['id']}", json={"columns": {"Age": {"action": "impute_mean"}}}
    ).json()

    current = client.get(f"/api/datasets/{dataset_id}/plan").json()
    assert current["version"] == 2
    assert [v["version"] for v in current["versions"]] == [1, 2]

    by_version = client.get(f"/api/datasets/{dataset_id}/plan?version=1").json()
    assert by_version["id"] == first["id"]

    assert client.get(f"/api/datasets/{dataset_id}/plan?version=99").status_code == 404
    assert client.get(f"/api/plans/{second['id']}").status_code == 200


def test_asking_for_a_plan_that_was_never_proposed_says_so(client, dataset_id) -> None:
    response = client.get(f"/api/datasets/{dataset_id}/plan")
    assert response.status_code == 404
    assert "/plan" in response.json()["error"]


def test_planning_an_unknown_dataset_is_a_404(client) -> None:
    assert client.post("/api/datasets/nope/plan").status_code == 404
    assert client.get("/api/plans/nope").status_code == 404


def test_naming_a_column_that_does_not_exist_is_a_sentence(client, dataset_id) -> None:
    response = client.post(
        f"/api/datasets/{dataset_id}/plan", json={"target": "NoSuchColumn"}
    )
    assert response.status_code == 400
    assert "NoSuchColumn" in response.json()["error"]


def test_the_stored_document_is_a_plan_that_can_be_rebuilt(client, dataset_id) -> None:
    """Serialisation is not decoration: the run reads this back to execute it."""
    body = client.post(
        f"/api/datasets/{dataset_id}/plan", json={"goal": "predict Survived"}
    ).json()

    document = {k: v for k, v in body.items() if k in PreprocessingPlan.model_fields}
    rebuilt = PreprocessingPlan.model_validate(document)
    assert rebuilt.target == "Survived"
    assert len(rebuilt.columns) == body["n_columns"]
    assert json.loads(rebuilt.model_dump_json())["target"] == "Survived"
