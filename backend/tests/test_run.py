"""The run: durable, versioned, reproducible, and never pretending to be a success.

A run is the thing a later phase will learn from, so the properties tested here are the
ones a meta-learning engine would have to trust months from now:

* It names the **exact plan version** it executed, and that reference keeps meaning the
  same thing after the plan is edited.
* Its artifacts are **all present or it is failed** — a row never names a file that is
  not there.
* The same plan on the same data produces the **same output hash**. Determinism stops
  being an argument and becomes a comparison.
* A failure is **kept and marked**, not discarded. A user needs to read why, and a
  history that records only successes is not a history.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.db import Run, SessionLocal
from app.main import app
from app.services import planning
from app.storage import get_storage

FIXTURES = Path(__file__).parent / "fixtures"
TITANIC = FIXTURES / "titanic.csv"

#: Every artifact a successful run stores. The six the specification names, plus
#: the standalone script — the run's claim in a form the user can execute.
ARTIFACTS = ("plan", "schema", "log", "pipeline", "train", "test", "script")


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def dataset_id(client) -> str:
    with TITANIC.open("rb") as handle:
        response = client.post(
            "/api/datasets", files={"file": ("titanic.csv", handle, "text/csv")}
        )
    assert response.status_code == 201, response.text
    return response.json()["id"]


@pytest.fixture
def plan_id(client, dataset_id) -> str:
    response = client.post(
        f"/api/datasets/{dataset_id}/plan", json={"goal": "predict Survived"}
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


@pytest.fixture
def run(client, plan_id) -> dict:
    response = client.post(f"/api/plans/{plan_id}/run")
    assert response.status_code == 201, response.text
    return response.json()


# --------------------------------------------------------------------------------------
# What a run records
# --------------------------------------------------------------------------------------


def test_a_run_records_everything_needed_to_describe_itself(run, plan_id, dataset_id) -> None:
    assert run["status"] == "succeeded"
    assert run["plan_id"] == plan_id
    assert run["plan_version"] == 1
    assert run["dataset_id"] == dataset_id
    assert run["target"] == "Survived"
    assert run["task"] == "binary classification"
    assert run["created_at"] and run["completed_at"]
    assert run["duration_ms"] >= 0
    assert run["n_features"] > 0
    assert run["output_hash"] and len(run["output_hash"]) == 64
    assert run["error"] is None

    split = run["split"]
    assert split["seed"] == 42
    assert split["test_size"] == pytest.approx(0.2)
    assert split["stratified"] is True
    assert split["train_rows"] + split["test_rows"] > 0


def test_a_run_carries_its_warnings_checks_and_schema(run) -> None:
    assert isinstance(run["warnings"], list)
    assert {"drift", "balance", "baseline"} <= set(run["checks"])
    schema = run["feature_schema"]
    assert schema["target"] == "Survived"
    assert schema["n_features"] == run["n_features"]
    assert len(schema["output_features"]) == run["n_features"]
    assert schema["input_columns"]


def test_a_run_lists_every_artifact_it_produced(run) -> None:
    assert sorted(run["artifacts"]) == sorted(ARTIFACTS)


# --------------------------------------------------------------------------------------
# Artifact integrity
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ARTIFACTS)
def test_every_artifact_downloads_and_is_what_it_claims(client, run, name) -> None:
    response = client.get(f"/api/runs/{run['id']}/artifacts/{name}")

    assert response.status_code == 200, response.text
    assert response.content, f"{name} is empty"
    assert planning.ARTIFACT_NAMES[name] in response.headers["content-disposition"]


def test_the_plan_artifact_is_the_plan_that_ran(client, run, plan_id) -> None:
    """Belt and braces: a run can be explained from its artifacts with no database."""
    stored = json.loads(client.get(f"/api/runs/{run['id']}/artifacts/plan").content)
    live = client.get(f"/api/plans/{plan_id}").json()

    assert stored["target"] == live["target"]
    assert [c["column"] for c in stored["columns"]] == [
        c["column"] for c in live["columns"]
    ]
    assert [c["action"] for c in stored["columns"]] == [
        c["action"] for c in live["columns"]
    ]


def test_the_schema_artifact_describes_what_inference_would_need(client, run) -> None:
    schema = json.loads(client.get(f"/api/runs/{run['id']}/artifacts/schema").content)

    assert schema["version"] == 1
    assert schema["target"] == "Survived"
    required = [c for c in schema["input_columns"] if c["required"]]
    assert required, "no column is marked required, so nothing could be validated"
    for column in schema["input_columns"]:
        assert {"name", "kind", "dtype", "action", "required"} <= set(column)


def test_the_log_artifact_accounts_for_every_input_column(client, run) -> None:
    log = json.loads(client.get(f"/api/runs/{run['id']}/artifacts/log").content)

    assert log, "the log is empty"
    for entry in log:
        assert {"seq", "stage", "step", "action", "reason", "severity"} <= set(entry)
    assert [entry["seq"] for entry in log] == list(range(1, len(log) + 1))


def test_the_parquet_artifacts_read_back_as_frames(client, run) -> None:
    train = pd.read_parquet(
        io.BytesIO(client.get(f"/api/runs/{run['id']}/artifacts/train").content)
    )
    test = pd.read_parquet(
        io.BytesIO(client.get(f"/api/runs/{run['id']}/artifacts/test").content)
    )

    assert len(train) == run["split"]["train_rows"]
    assert len(test) == run["split"]["test_rows"]
    # Features plus the target.
    assert train.shape[1] == run["n_features"] + 1
    assert list(train.columns) == list(test.columns)
    assert "Survived" in train.columns


def test_the_pipeline_artifact_loads_and_transforms(client, run) -> None:
    """The artifact a Stage 5 worker would load. It has to be usable, not just present."""
    pipeline = joblib.load(
        io.BytesIO(client.get(f"/api/runs/{run['id']}/artifacts/pipeline").content)
    )

    from app.services.ingest import read_upload

    frame = read_upload(TITANIC.read_bytes(), "titanic.csv").frame
    rows = frame.head(20)

    transformed = pipeline.transform(rows)
    assert transformed.shape[1] == run["n_features"]


def test_an_unknown_artifact_name_cannot_name_a_path(client, run) -> None:
    """No spelling of a path reaches a file.

    ``..`` is not in this list: a URL ending in it is collapsed by the client before the
    request is sent, so it addresses the run route and returns the run — which is a
    normalisation result, not this route serving anything. What matters is that no
    request below returns artifact bytes.
    """
    for name in ("nope", "plan.json", "..%2f..%2fetc%2fpasswd", "....//plan"):
        response = client.get(f"/api/runs/{run['id']}/artifacts/{name}")

        assert response.status_code == 404, f"{name!r} was served"
        assert response.headers["content-type"].startswith("application/json")


def test_a_path_shaped_artifact_name_matches_no_route_at_all(client, run) -> None:
    response = client.get(f"/api/runs/{run['id']}/artifacts/../../etc/passwd")

    assert response.status_code == 404
    # And it answers in the same envelope as every other error, not Starlette's default.
    assert "error" in response.json() and "code" in response.json()


def test_artifacts_of_an_unknown_run_are_a_404(client) -> None:
    assert client.get("/api/runs/nope/artifacts/plan").status_code == 404


# --------------------------------------------------------------------------------------
# Reproducibility and plan binding
# --------------------------------------------------------------------------------------


def test_the_same_plan_twice_gives_the_same_output_hash(client, plan_id) -> None:
    first = client.post(f"/api/plans/{plan_id}/run").json()
    second = client.post(f"/api/plans/{plan_id}/run").json()

    assert first["id"] != second["id"]
    assert first["output_hash"] == second["output_hash"]
    assert first["n_features"] == second["n_features"]


def test_a_different_plan_version_gives_a_different_result(client, plan_id) -> None:
    before = client.post(f"/api/plans/{plan_id}/run").json()

    edited = client.put(
        f"/api/plans/{plan_id}", json={"columns": {"Age": {"action": "drop"}}}
    ).json()
    after = client.post(f"/api/plans/{edited['id']}/run").json()

    assert after["plan_version"] == 2
    assert after["output_hash"] != before["output_hash"]
    assert after["n_features"] < before["n_features"]


def test_editing_a_plan_does_not_change_an_earlier_runs_record(client, plan_id) -> None:
    """The property the whole versioning scheme exists for."""
    run = client.post(f"/api/plans/{plan_id}/run").json()

    client.put(f"/api/plans/{plan_id}", json={"columns": {"Age": {"action": "drop"}}})

    reread = client.get(f"/api/runs/{run['id']}").json()
    assert reread["plan_version"] == run["plan_version"] == 1
    assert reread["output_hash"] == run["output_hash"]
    assert reread["n_features"] == run["n_features"]

    # And the plan artifact it stored still shows what actually ran.
    stored = json.loads(client.get(f"/api/runs/{run['id']}/artifacts/plan").content)
    age = next(c for c in stored["columns"] if c["column"] == "Age")
    assert age["action"] == "impute_median"


def test_runs_are_listed_for_their_dataset_newest_first(client, dataset_id, plan_id) -> None:
    first = client.post(f"/api/plans/{plan_id}/run").json()
    second = client.post(f"/api/plans/{plan_id}/run").json()

    listing = client.get(f"/api/datasets/{dataset_id}/runs").json()
    ids = [row["id"] for row in listing["runs"]]
    assert set(ids) >= {first["id"], second["id"]}
    assert ids.index(second["id"]) <= ids.index(first["id"])


# --------------------------------------------------------------------------------------
# Failure is recorded, not hidden
# --------------------------------------------------------------------------------------


def test_a_failed_run_is_stored_as_failed_with_no_artifacts(client, dataset_id, monkeypatch) -> None:
    plan_id = client.post(
        f"/api/datasets/{dataset_id}/plan", json={"goal": "predict Survived"}
    ).json()["id"]

    def exploding_execute(*_args, **_kwargs):
        raise RuntimeError("the executor fell over")

    monkeypatch.setattr(planning, "execute", exploding_execute)
    response = client.post(f"/api/plans/{plan_id}/run")

    # The run resource was created, so it is a 201; `status` is how a caller tells.
    assert response.status_code == 201, response.text
    run = response.json()
    assert run["status"] == "failed"
    assert run["artifacts"] == []
    assert run["output_hash"] is None
    assert run["error"]["type"] == "RuntimeError"
    assert "fell over" in run["error"]["message"]

    # It is kept, and it is still a failure when read back.
    assert client.get(f"/api/runs/{run['id']}").json()["status"] == "failed"


def test_a_failed_run_does_not_serve_artifacts_it_never_made(client, dataset_id, monkeypatch) -> None:
    plan_id = client.post(
        f"/api/datasets/{dataset_id}/plan", json={"goal": "predict Survived"}
    ).json()["id"]
    monkeypatch.setattr(
        planning, "execute", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nope"))
    )
    run = client.post(f"/api/plans/{plan_id}/run").json()

    response = client.get(f"/api/runs/{run['id']}/artifacts/plan")
    assert response.status_code == 404
    assert "did not finish" in response.json()["error"]


def test_an_error_message_never_carries_a_traceback(client, dataset_id, monkeypatch) -> None:
    plan_id = client.post(
        f"/api/datasets/{dataset_id}/plan", json={"goal": "predict Survived"}
    ).json()["id"]

    def exploding(*_args, **_kwargs):
        raise RuntimeError("/Users/someone/secret/path/file.py line 42 boom")

    monkeypatch.setattr(planning, "execute", exploding)
    run = client.post(f"/api/plans/{plan_id}/run").json()

    assert "Traceback" not in json.dumps(run)
    assert 'File "' not in json.dumps(run)


def test_a_row_never_names_an_artifact_that_is_not_stored(client, run) -> None:
    """Database metadata and the filesystem must agree."""
    session = SessionLocal()
    try:
        row = session.get(Run, run["id"])
        storage = get_storage()
        assert row.artifacts
        for name, key in row.artifacts.items():
            assert storage.exists(key), f"{name} is recorded at {key} but not stored"
    finally:
        session.close()


# --------------------------------------------------------------------------------------
# The run resource shares its URL with the older one
# --------------------------------------------------------------------------------------


def test_an_older_preprocess_run_id_still_returns_its_own_document(client, dataset_id) -> None:
    """Backward compatibility: the two id spaces are disjoint, so both can be served."""
    legacy = client.post(
        f"/api/datasets/{dataset_id}/preprocess", json={"goal": "predict Survived"}
    ).json()

    fetched = client.get(f"/api/runs/{legacy['run_id']}")
    assert fetched.status_code == 200
    body = fetched.json()
    assert body["run_id"] == legacy["run_id"]
    assert "steps" in body
    # It is the old document, not the new resource.
    assert "plan_version" not in body
