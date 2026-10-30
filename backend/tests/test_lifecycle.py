"""Two cross-cutting contracts: how an error is shaped, and how a dataset goes away.

**The error envelope.** `docs/PRD.md` §7.3.1 asks for `{code, message, detail}`;
every client and test written so far reads `error`. Both are sent. The tests below pin
that additively — the old field keeps its exact meaning, the three new ones are present
on every failure path, and a malformed request no longer answers in a different shape
from everything else.

**Deletion.** Removing a dataset has to remove everything derived from it and leave
nothing of another dataset touched. The ordering is deliberate and is tested as such:
rows first and committed, then files, so a partial failure strands bytes rather than
leaving rows that point at data which is no longer there.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.db import Compilation, Dataset, Plan, PreprocessingRun, Run, SessionLocal, Workspace
from app.main import app
from app.storage import get_storage

FIXTURES = Path(__file__).parent / "fixtures"
TITANIC = FIXTURES / "titanic.csv"


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def _upload(client, name: str = "titanic.csv") -> str:
    with TITANIC.open("rb") as handle:
        response = client.post("/api/datasets", files={"file": (name, handle, "text/csv")})
    assert response.status_code == 201, response.text
    return response.json()["id"]


# --------------------------------------------------------------------------------------
# The error envelope
# --------------------------------------------------------------------------------------

ENVELOPE = {"error", "code", "message", "detail"}


def test_a_user_error_carries_both_the_old_and_the_specified_fields(client) -> None:
    response = client.post(
        "/api/datasets", files={"file": ("empty.csv", b"", "text/csv")}
    )

    assert response.status_code == 400
    body = response.json()
    assert ENVELOPE <= set(body)
    assert body["code"] == "bad_request"
    # The original contract is unchanged: `error` is still the sentence.
    assert isinstance(body["error"], str) and body["error"]
    assert body["message"] == body["error"]


def test_a_missing_resource_is_not_found_in_the_same_envelope(client) -> None:
    body = client.get("/api/datasets/does-not-exist").json()

    assert ENVELOPE <= set(body)
    assert body["code"] == "not_found"
    assert body["detail"] is None


def test_a_url_no_route_matches_answers_in_the_envelope_too(client) -> None:
    """Starlette's router raises the base class; a handler bound to the subclass misses it."""
    response = client.get("/api/no/such/route")

    assert response.status_code == 404
    body = response.json()
    assert ENVELOPE <= set(body)
    assert body["code"] == "not_found"


def test_a_malformed_request_names_the_fields_that_were_wrong(client) -> None:
    dataset_id = _upload(client)

    response = client.post(
        f"/api/datasets/{dataset_id}/preprocess", json={"test_size": "not a number"}
    )

    assert response.status_code == 422
    body = response.json()
    assert ENVELOPE <= set(body)
    assert body["code"] == "validation_error"
    # The per-field list is the useful part of a validation failure, and it survives.
    fields = body["detail"]["fields"]
    assert fields and any("test_size" in field["field"] for field in fields)
    assert all({"field", "problem"} == set(field) for field in fields)


def test_the_three_error_categories_are_distinguishable(client) -> None:
    """A client has to be able to tell "fix your request" from "this app is broken"."""
    dataset_id = _upload(client)

    malformed = client.post(
        f"/api/datasets/{dataset_id}/preprocess", json={"test_size": 99}
    ).json()
    user_error = client.post(
        "/api/datasets", files={"file": ("x.csv", b"a,b\n", "text/csv")}
    ).json()
    missing = client.get("/api/datasets/nope").json()

    assert malformed["code"] == "validation_error"
    assert user_error["code"] == "bad_request"
    assert missing["code"] == "not_found"
    assert len({malformed["code"], user_error["code"], missing["code"]}) == 3


def test_no_error_body_ever_contains_a_traceback(client) -> None:
    for response in (
        client.post("/api/datasets", files={"file": ("e.csv", b"", "text/csv")}),
        client.get("/api/datasets/nope"),
        client.get("/api/plans/nope"),
        client.get("/api/runs/nope/artifacts/plan"),
    ):
        assert "Traceback" not in response.text
        assert 'File "' not in response.text


# --------------------------------------------------------------------------------------
# Deleting a dataset
# --------------------------------------------------------------------------------------


def test_deleting_a_dataset_removes_everything_derived_from_it(client) -> None:
    dataset_id = _upload(client)
    plan = client.post(
        f"/api/datasets/{dataset_id}/plan", json={"goal": "predict Survived"}
    ).json()
    run = client.post(f"/api/plans/{plan['id']}/run").json()
    client.post(f"/api/datasets/{dataset_id}/preprocess", json={"goal": "predict Survived"})
    workspace = client.post(f"/api/workspaces/from-dataset/{dataset_id}").json()

    response = client.delete(f"/api/datasets/{dataset_id}")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["deleted"] is True
    assert body["artifacts_orphaned"] == []
    assert body["artifacts_removed"] > 0

    assert client.get(f"/api/datasets/{dataset_id}").status_code == 404
    assert client.get(f"/api/plans/{plan['id']}").status_code == 404
    assert client.get(f"/api/runs/{run['id']}").status_code == 404
    assert client.get(f"/api/workspaces/{workspace['id']}").status_code == 404


def test_deletion_leaves_no_rows_behind_in_any_table(client) -> None:
    """The cascades are declared in the schema; this checks they are enforced."""
    dataset_id = _upload(client)
    plan = client.post(
        f"/api/datasets/{dataset_id}/plan", json={"goal": "predict Survived"}
    ).json()
    client.post(f"/api/plans/{plan['id']}/run")
    client.post(f"/api/datasets/{dataset_id}/preprocess", json={"goal": "predict Survived"})
    client.post(f"/api/workspaces/from-dataset/{dataset_id}")

    client.delete(f"/api/datasets/{dataset_id}")

    session = SessionLocal()
    try:
        assert session.get(Dataset, dataset_id) is None
        for model, column in (
            (Plan, Plan.dataset_id),
            (Run, Run.dataset_id),
            (PreprocessingRun, PreprocessingRun.dataset_id),
            (Workspace, Workspace.dataset_id),
        ):
            left = session.query(model).filter(column == dataset_id).count()
            assert left == 0, f"{model.__tablename__} kept {left} row(s)"
    finally:
        session.close()


def test_deletion_removes_the_stored_bytes(client) -> None:
    dataset_id = _upload(client)
    plan = client.post(
        f"/api/datasets/{dataset_id}/plan", json={"goal": "predict Survived"}
    ).json()
    run = client.post(f"/api/plans/{plan['id']}/run").json()

    session = SessionLocal()
    try:
        keys = [session.get(Dataset, dataset_id).storage_key]
        keys += list(session.get(Run, run["id"]).artifacts.values())
    finally:
        session.close()

    storage = get_storage()
    assert all(storage.exists(key) for key in keys)

    client.delete(f"/api/datasets/{dataset_id}")

    assert not any(storage.exists(key) for key in keys)


def test_deleting_one_dataset_does_not_touch_another(client) -> None:
    keep = _upload(client, "keep.csv")
    remove = _upload(client, "remove.csv")
    keep_plan = client.post(
        f"/api/datasets/{keep}/plan", json={"goal": "predict Survived"}
    ).json()
    keep_run = client.post(f"/api/plans/{keep_plan['id']}/run").json()
    client.post(f"/api/datasets/{remove}/plan", json={"goal": "predict Survived"})

    client.delete(f"/api/datasets/{remove}")

    assert client.get(f"/api/datasets/{keep}").status_code == 200
    assert client.get(f"/api/plans/{keep_plan['id']}").status_code == 200
    assert client.get(f"/api/runs/{keep_run['id']}").json()["status"] == "succeeded"

    storage = get_storage()
    session = SessionLocal()
    try:
        for key in session.get(Run, keep_run["id"]).artifacts.values():
            assert storage.exists(key), "another dataset's artifact was removed"
    finally:
        session.close()


def test_deleting_an_unknown_dataset_says_so(client) -> None:
    response = client.delete("/api/datasets/does-not-exist")

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


def test_deleting_twice_is_a_404_the_second_time(client) -> None:
    dataset_id = _upload(client)

    assert client.delete(f"/api/datasets/{dataset_id}").status_code == 200
    assert client.delete(f"/api/datasets/{dataset_id}").status_code == 404


def test_a_dataset_that_was_never_used_deletes_cleanly(client) -> None:
    """No plan, no run, no workspace — only the uploaded bytes."""
    dataset_id = _upload(client)

    body = client.delete(f"/api/datasets/{dataset_id}").json()

    assert body["artifacts_removed"] == 1
    assert body["artifacts_orphaned"] == []


def test_a_storage_failure_still_leaves_the_database_consistent(client, monkeypatch) -> None:
    """The ordering decision, exercised: bytes may be stranded, rows may not survive."""
    dataset_id = _upload(client)
    storage = get_storage()

    def refuse(_key: str) -> None:
        raise OSError("disk is read-only")

    monkeypatch.setattr(storage, "delete", refuse)
    response = client.delete(f"/api/datasets/{dataset_id}")

    assert response.status_code == 200
    body = response.json()
    # Reported rather than hidden: these files are still on disk.
    assert body["artifacts_orphaned"], "a failed removal was reported as a success"
    assert body["artifacts_removed"] == 0
    # The row is gone regardless, which is the side of the failure that is recoverable.
    assert client.get(f"/api/datasets/{dataset_id}").status_code == 404


def test_delete_survives_a_cors_preflight(client) -> None:
    """A browser asks permission before sending a DELETE; a method the policy omits is
    a route the interface cannot use at all."""
    response = client.options(
        "/api/datasets/anything",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "DELETE",
        },
    )

    assert response.status_code == 200, response.text
    assert "DELETE" in response.headers["access-control-allow-methods"]


# --------------------------------------------------------------------------------------
# Which health score is the health score
# --------------------------------------------------------------------------------------
#
# Two numbers out of a hundred are produced from one file, by different arithmetic, and
# they usually disagree. That is allowed — they answer different questions — but only
# while it is impossible to mistake one for the other. These tests hold that line:
# `health.assess` is the score the product means, the ingest-time check is always
# labelled as something else, and neither is quietly substituted for the other.


def test_the_stored_health_score_is_the_six_dimension_one(client) -> None:
    from app.services.health import assess
    from app.services.ingest import read_upload

    dataset_id = _upload(client)
    stored = client.get(f"/api/datasets/{dataset_id}").json()["health_score"]

    ingested = read_upload(TITANIC.read_bytes(), "titanic.csv")
    assert stored == assess(ingested.frame).score
    assert stored != ingested.quality_score, (
        "the fixture no longer separates the two scores, so this test would pass "
        "whichever one were stored"
    )


def test_the_ingest_check_is_always_labelled_as_not_the_health_score(client) -> None:
    dataset_id = _upload(client)

    profile = client.get(f"/api/datasets/{dataset_id}/profile").json()
    analysis = client.post(
        f"/api/datasets/{dataset_id}/preprocess", json={"goal": "predict Survived"}
    ).json()

    for block in (profile["quality"], analysis["quality"]):
        assert "not the workspace health score" in block["label"]
        assert 0 <= block["score"] <= 100


def test_the_workspace_reports_the_six_dimension_score(client) -> None:
    """The number a user acts on comes from one place."""
    from app.services.health import assess
    from app.services.ingest import read_upload

    dataset_id = _upload(client)
    workspace = client.post(f"/api/workspaces/from-dataset/{dataset_id}").json()

    health = client.get(f"/api/workspaces/{workspace['id']}/health").json()
    frame = read_upload(TITANIC.read_bytes(), "titanic.csv").frame

    assert health["score"] == assess(frame).score
    assert {dimension["key"] for dimension in health["dimensions"]} == {
        "completeness",
        "validity",
        "consistency",
        "uniqueness",
        "feature_quality",
        "distribution",
    }


def test_the_health_score_is_a_weighted_mean_of_its_own_dimensions(client) -> None:
    """The arithmetic is handed to the interface so a user can take it apart; if the
    parts do not reconstruct the whole, what is shown is decoration."""
    from app.services.health import DIMENSION_WEIGHTS, assess
    from app.services.ingest import read_upload

    report = assess(read_upload(TITANIC.read_bytes(), "titanic.csv").frame)
    document = dict(report)

    by_key = {dimension["key"]: dimension for dimension in document["dimensions"]}
    assert set(by_key) == set(DIMENSION_WEIGHTS)

    rebuilt = sum(
        by_key[name]["score"] * weight for name, weight in DIMENSION_WEIGHTS.items()
    )
    assert round(rebuilt) == document["score"]
    assert sum(DIMENSION_WEIGHTS.values()) == pytest.approx(1.0)
    # Each dimension states its own weight and contribution, so the arithmetic shown in
    # the interface is the arithmetic that ran.
    for name, weight in DIMENSION_WEIGHTS.items():
        assert by_key[name]["weight"] == pytest.approx(weight)
        assert by_key[name]["contribution"] == pytest.approx(
            by_key[name]["score"] * weight, abs=0.05
        )


def test_health_scoring_is_deterministic(client) -> None:
    from app.services.health import assess
    from app.services.ingest import read_upload

    frame = read_upload(TITANIC.read_bytes(), "titanic.csv").frame
    assert dict(assess(frame)) == dict(assess(frame))
