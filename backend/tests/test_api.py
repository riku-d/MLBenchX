"""The resource-shaped API, end to end, against the Titanic fixture.

Every test drives the app the way the frontend does — through HTTP, with a real
database and real storage, both pointed at a sandbox by ``conftest``. Nothing is mocked,
because the things most likely to break here are the seams: a Pydantic model that does
not match what the route returns, a run that cannot be read back after a restart, a
storage key assembled from a filename.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

FIXTURES = Path(__file__).parent / "fixtures"
TITANIC = FIXTURES / "titanic.csv"


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


# --------------------------------------------------------------------------------------
# Upload
# --------------------------------------------------------------------------------------


def test_uploading_a_dataset_records_what_it_is(client) -> None:
    with TITANIC.open("rb") as handle:
        response = client.post(
            "/api/datasets", files={"file": ("titanic.csv", handle, "text/csv")}
        )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "titanic.csv"
    assert body["n_rows"] == 891
    assert body["n_columns"] == 13
    assert 0 <= body["health_score"] <= 100
    assert body["project_id"] == "default"
    assert body["runs"] == []
    assert "Survived" in body["columns"]


def test_an_uploaded_dataset_can_be_fetched_again(client, dataset_id) -> None:
    body = client.get(f"/api/datasets/{dataset_id}").json()
    assert body["id"] == dataset_id
    assert body["n_rows"] == 891


def test_a_missing_dataset_is_a_404_with_a_sentence(client) -> None:
    response = client.get("/api/datasets/does-not-exist")
    assert response.status_code == 404
    assert "error" in response.json()


@pytest.mark.parametrize(
    "payload,name",
    [
        (b"", "empty.csv"),
        (b"a,b\n1,2\n", "tiny.csv"),
        (b"%PDF-1.4 not a spreadsheet", "notes.pdf"),
    ],
)
def test_a_bad_upload_is_refused_with_an_explanation(client, payload, name) -> None:
    response = client.post("/api/datasets", files={"file": (name, payload, "text/csv")})
    assert response.status_code == 400
    message = response.json()["error"]
    assert message and not message.startswith("Traceback")


def test_an_oversized_upload_is_refused_before_it_is_parsed(client) -> None:
    """The cap is enforced from Content-Length, so the body is never spooled."""
    response = client.post(
        "/api/analyse",
        files={"file": ("huge.csv", b"x" * (51 * 1024 * 1024), "text/csv")},
        data={"goal": "anything"},
    )
    assert response.status_code == 413
    assert "larger than" in response.json()["error"]


# --------------------------------------------------------------------------------------
# Profile
# --------------------------------------------------------------------------------------


def test_the_profile_describes_every_column(client, dataset_id) -> None:
    body = client.get(f"/api/datasets/{dataset_id}/profile").json()

    assert body["dataset_id"] == dataset_id
    assert body["n_rows"] == 891
    assert len(body["columns"]) == 13

    age = next(c for c in body["columns"] if c["column"] == "Age")
    assert age["kind"] == "numeric"
    assert age["n_missing"] == 184
    assert age["missing_rate"] == pytest.approx(184 / 891, abs=1e-3)
    assert age["outliers"] is not None

    name = next(c for c in body["columns"] if c["column"] == "Name")
    assert name["kind"] == "text"
    assert name["outliers"] is None


def test_the_health_score_is_itemised(client, dataset_id) -> None:
    quality = client.get(f"/api/datasets/{dataset_id}/profile").json()["quality"]

    assert 0 <= quality["score"] <= 100
    assert quality["breakdown"], "a score below 100 must say what it lost points for"
    total = sum(item["points_lost"] for item in quality["breakdown"])
    assert round(100 - total) == pytest.approx(quality["score"], abs=1)


def test_no_deduction_exceeds_its_cap(client, dataset_id) -> None:
    for item in client.get(f"/api/datasets/{dataset_id}/profile").json()["quality"]["breakdown"]:
        assert item["points_lost"] <= item["cap"] + 1e-9


# --------------------------------------------------------------------------------------
# Preprocess
# --------------------------------------------------------------------------------------


def test_preprocessing_returns_the_whole_explanation(client, dataset_id) -> None:
    response = client.post(
        f"/api/datasets/{dataset_id}/preprocess",
        json={"target": "Survived", "goal": "who survived"},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["dataset_id"] == dataset_id
    assert body["target"]["column"] == "Survived"
    assert body["target"]["task"] == "binary classification"
    assert body["result"]["train_rows"] + body["result"]["test_rows"] == 891
    assert body["result"]["features"] > 0
    assert body["steps"][0]["stage"] == "split", "the split is shown first"
    assert body["options"]["scaler"] == "standard"


def test_the_split_is_eighty_twenty(client, dataset_id) -> None:
    body = client.post(
        f"/api/datasets/{dataset_id}/preprocess", json={"target": "Survived"}
    ).json()
    assert body["result"]["test_rows"] == pytest.approx(891 * 0.2, abs=1)


def test_identifiers_are_set_aside_with_a_reason(client, dataset_id) -> None:
    body = client.post(
        f"/api/datasets/{dataset_id}/preprocess", json={"target": "Survived"}
    ).json()

    dropped = {item["column"]: item["reason"] for item in body["dropped"]}
    assert "PassengerId" in dropped or "Ticket" in dropped
    for reason in dropped.values():
        assert reason.endswith(".") or reason.endswith("for now.")


def test_a_mostly_blank_column_is_dropped_for_being_blank(client, dataset_id) -> None:
    body = client.post(
        f"/api/datasets/{dataset_id}/preprocess", json={"target": "Survived"}
    ).json()
    dropped = {item["column"]: item["reason"] for item in body["dropped"]}
    # Cabin is 78% missing in the fixture, over the 60% threshold.
    assert "Cabin" in dropped
    assert "blank" in dropped["Cabin"].lower()


@pytest.mark.parametrize("scaler", ["standard", "minmax", "robust"])
def test_every_scaler_is_accepted_and_echoed(client, dataset_id, scaler) -> None:
    body = client.post(
        f"/api/datasets/{dataset_id}/preprocess",
        json={"target": "Survived", "scaler": scaler},
    ).json()
    assert body["options"]["scaler"] == scaler


@pytest.mark.parametrize("encoder", ["frequency", "target"])
def test_every_high_card_encoder_is_accepted(client, dataset_id, encoder) -> None:
    body = client.post(
        f"/api/datasets/{dataset_id}/preprocess",
        json={"target": "Survived", "high_card_encoder": encoder},
    ).json()
    assert body["options"]["high_card_encoder"] == encoder


def test_an_unknown_option_is_a_422_not_a_500(client, dataset_id) -> None:
    response = client.post(
        f"/api/datasets/{dataset_id}/preprocess",
        json={"target": "Survived", "scaler": "quantile"},
    )
    assert response.status_code == 422


def test_an_unknown_target_is_a_400_with_advice(client, dataset_id) -> None:
    response = client.post(
        f"/api/datasets/{dataset_id}/preprocess", json={"target": "Nonexistent"}
    )
    assert response.status_code == 400
    assert "Pick one from the list" in response.json()["error"]


def test_preprocessing_twice_keeps_both_runs(client, dataset_id) -> None:
    first = client.post(
        f"/api/datasets/{dataset_id}/preprocess",
        json={"target": "Survived", "scaler": "standard"},
    ).json()
    second = client.post(
        f"/api/datasets/{dataset_id}/preprocess",
        json={"target": "Survived", "scaler": "robust"},
    ).json()

    assert first["run_id"] != second["run_id"]
    runs = client.get(f"/api/datasets/{dataset_id}").json()["runs"]
    assert len(runs) == 2
    assert {r["scaler"] for r in runs} == {"standard", "robust"}


# --------------------------------------------------------------------------------------
# Log
# --------------------------------------------------------------------------------------


def test_the_log_is_ordered_and_carries_every_spec_field(client, dataset_id) -> None:
    client.post(f"/api/datasets/{dataset_id}/preprocess", json={"target": "Survived"})
    body = client.get(f"/api/datasets/{dataset_id}/preprocess/log").json()

    assert body["dataset_id"] == dataset_id
    assert body["target"] == "Survived"
    assert body["fitted_on"] == "train_fold_only"

    entries = body["entries"]
    assert entries
    assert [e["step"] for e in entries] == list(range(1, len(entries) + 1))
    for entry in entries:
        assert entry["action"]
        assert entry["column"]
        assert entry["reason"]
        assert entry["before"]["n_columns"] >= 1
        assert entry["after"]["n_columns"] >= 1


def test_the_log_can_be_fetched_for_a_specific_run(client, dataset_id) -> None:
    first = client.post(
        f"/api/datasets/{dataset_id}/preprocess",
        json={"target": "Survived", "scaler": "minmax"},
    ).json()
    client.post(
        f"/api/datasets/{dataset_id}/preprocess",
        json={"target": "Survived", "scaler": "robust"},
    )

    body = client.get(
        f"/api/datasets/{dataset_id}/preprocess/log", params={"run_id": first["run_id"]}
    ).json()
    assert body["run_id"] == first["run_id"]
    assert any(e["action"] == "scale_minmax" for e in body["entries"])


def test_asking_for_a_log_before_preprocessing_says_so(client, dataset_id) -> None:
    response = client.get(f"/api/datasets/{dataset_id}/preprocess/log")
    assert response.status_code == 404
    assert "preprocess" in response.json()["error"]


# --------------------------------------------------------------------------------------
# Runs and downloads
# --------------------------------------------------------------------------------------


def test_a_run_can_be_read_back_unchanged(client, dataset_id) -> None:
    created = client.post(
        f"/api/datasets/{dataset_id}/preprocess", json={"target": "Survived"}
    ).json()
    fetched = client.get(f"/api/runs/{created['run_id']}").json()
    assert fetched["run_id"] == created["run_id"]
    assert fetched["steps"] == created["steps"]


def test_the_prepared_data_downloads_as_csv(client, dataset_id) -> None:
    run = client.post(
        f"/api/datasets/{dataset_id}/preprocess", json={"target": "Survived"}
    ).json()
    response = client.get(f"/api/runs/{run['run_id']}/data.csv")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    header = response.text.splitlines()[0]
    assert "Survived" in header
    assert len(response.text.splitlines()) == run["result"]["train_rows"] + 1


def test_the_exported_script_is_valid_python(client, dataset_id) -> None:
    run = client.post(
        f"/api/datasets/{dataset_id}/preprocess", json={"target": "Survived"}
    ).json()
    source = client.get(f"/api/runs/{run['run_id']}/script.py").text

    compile(source, "exported.py", "exec")
    assert "train_test_split" in source
    # The split must appear before the fit in the generated file, not just in ours.
    assert source.index("train_test_split(") < source.index("fit_transform")


def test_a_filename_cannot_inject_a_header(client) -> None:
    hostile = 'evil";\r\nX-Injected: yes.csv'
    with TITANIC.open("rb") as handle:
        dataset = client.post(
            "/api/datasets", files={"file": (hostile, handle, "text/csv")}
        ).json()
    run = client.post(
        f"/api/datasets/{dataset['id']}/preprocess", json={"target": "Survived"}
    ).json()

    response = client.get(f"/api/runs/{run['run_id']}/data.csv")
    disposition = response.headers["content-disposition"]
    assert "X-Injected" not in response.headers
    assert '"' not in disposition.split("filename=")[1].strip('"')
    assert "\r" not in disposition and "\n" not in disposition


# --------------------------------------------------------------------------------------
# The single-call shortcuts the UI uses
# --------------------------------------------------------------------------------------


def test_analyse_does_everything_in_one_call(client) -> None:
    with TITANIC.open("rb") as handle:
        response = client.post(
            "/api/analyse",
            files={"file": ("titanic.csv", handle, "text/csv")},
            data={"goal": "who survived"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["target"]["column"] == "Survived"
    assert body["profile"]["columns"]
    assert body["steps"]
    # The shortcut must leave the same durable trace as the long way round.
    assert client.get(f"/api/datasets/{body['dataset_id']}").status_code == 200
    assert client.get(f"/api/datasets/{body['dataset_id']}/preprocess/log").status_code == 200


def test_the_bundled_example_still_works(client) -> None:
    body = client.post("/api/example", data={"goal": "which students will fail"}).json()
    assert body["steps"][0]["stage"] == "split"
    assert body["result"]["features"] > 0


def test_health_reports_what_is_stored(client, dataset_id) -> None:
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["datasets"] >= 1
