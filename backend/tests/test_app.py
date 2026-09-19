"""Tests for the application layer: reading files, guessing the target, and the API.

The preprocessing engine has its own suite in ``test_glassbox.py``. These cover the
things that sit on top of it — the parts that touch whatever a user actually uploads,
which is where the surprises live.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import SAMPLE_PATH, app
from app.services.ingest import IngestError, read_upload
from app.services.screening import find_identifier_columns
from app.services.target import detect_target

client = TestClient(app)


def messy_csv(rows: int = 200) -> bytes:
    """A file with every format problem the ingest layer claims to handle."""
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({
        "  ref id ": [f"R{i:04d}" for i in range(rows)],
        "site": ["North"] * rows,
        "dept": rng.choice(["CSE", "ECE", "ME"], rows).astype(object),
        "score": np.round(rng.normal(60, 12, rows), 1).astype(object),
        "fee": [f"{v:,}" for v in rng.integers(20000, 90000, rows)],
        "has_backlog": rng.choice(["Yes", "No"], rows),
    })
    frame.loc[[2, 5, 9], "score"] = ["N/A", "-", "unknown"]
    text = frame.to_csv(index=False, sep=";")
    return text.encode("latin-1")


# --------------------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------------------


def test_reads_a_semicolon_latin1_file_with_placeholder_blanks() -> None:
    result = read_upload(messy_csv(), "messy.csv")

    assert result.delimiter == ";"
    assert result.rows == 200
    assert "ref id" in result.frame.columns, "headers should be stripped of spaces"

    # 'N/A', '-' and 'unknown' became real blanks, and the column is now numeric.
    assert pd.api.types.is_numeric_dtype(result.frame["score"])
    assert int(result.frame["score"].isna().sum()) == 3

    # 'fee' held text like "45,231" and is now a number.
    assert pd.api.types.is_numeric_dtype(result.frame["fee"])
    assert "fee" in result.numeric_columns

    joined = " ".join(result.cleaning)
    assert "placeholder" in joined and "text into numbers" in joined


def test_a_column_that_is_only_mostly_numeric_stays_text() -> None:
    """Three stray words in 200 rows means a category, not a broken number column."""
    rows = 200
    values: list[object] = [str(i) for i in range(rows)]
    values[7] = "pending"
    frame = pd.DataFrame({"code": values, "label": ["a", "b"] * (rows // 2)})
    result = read_upload(frame.to_csv(index=False).encode(), "x.csv")
    assert not pd.api.types.is_numeric_dtype(result.frame["code"])


@pytest.mark.parametrize(
    ("payload", "name", "fragment"),
    [
        (b"", "empty.csv", "empty"),
        (b"a,b,target\n1,2,yes\n", "tiny.csv", "at least 50"),
        (b"a,b,c\n", "headers.csv", "no data underneath"),
        (b"col\n1\n2\n3\n", "one.csv", "only has one column"),
    ],
)
def test_bad_files_raise_a_sentence_the_user_can_act_on(payload, name, fragment) -> None:
    with pytest.raises(IngestError) as excinfo:
        read_upload(payload, name)
    message = str(excinfo.value)
    assert fragment in message
    assert message[0].isupper() and message.rstrip().endswith((".", "!"))


def test_an_unsupported_extension_is_refused_helpfully() -> None:
    with pytest.raises(IngestError, match="save a copy as CSV"):
        read_upload(b"\x00\x01binary", "photo.png")


# --------------------------------------------------------------------------------------
# Target detection
# --------------------------------------------------------------------------------------


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "student_id": range(120),
        "attendance": rng.normal(75, 10, 120),
        "department": rng.choice(["CSE", "ECE"], 120),
        "has_backlog": rng.choice(["Yes", "No"], 120),
    })


@pytest.mark.parametrize(
    ("goal", "expected", "tier"),
    [
        ("predict has_backlog", "has_backlog", "exact"),          # names the column
        ("who will have a backlog", "has_backlog", "keyword"),    # keyword association
        ("which students will fail", "has_backlog", "keyword"),   # fail -> backlog
        ("", "has_backlog", "fallback"),                          # most label-shaped
    ],
)
def test_the_goal_resolves_to_a_column(frame, goal, expected, tier) -> None:
    guess = detect_target(frame, goal)
    assert guess.column == expected
    assert guess.confidence == tier
    assert guess.inferred is True
    assert guess.note


def test_an_explicit_choice_wins_and_is_not_marked_inferred(frame) -> None:
    guess = detect_target(frame, "which students will fail", explicit="attendance")
    assert guess.column == "attendance"
    assert guess.inferred is False


def test_an_unknown_explicit_column_says_so(frame) -> None:
    with pytest.raises(ValueError, match="couldn't find a column called 'salary'"):
        detect_target(frame, "anything", explicit="salary")


# --------------------------------------------------------------------------------------
# Identifier screening — the bug that motivated it
# --------------------------------------------------------------------------------------


def test_a_numeric_primary_key_is_caught() -> None:
    """An int64 1..n key profiles as an ordinary number, so the screen has to catch it."""
    n = 300
    frame = pd.DataFrame({"Student_ID": range(1, n + 1), "score": [1.5] * n})
    found = find_identifier_columns(frame, target="score")
    assert "Student_ID" in found
    assert "identifies the row" in found["Student_ID"] or "row number" in found["Student_ID"]


def test_a_continuous_measurement_is_not_mistaken_for_a_key() -> None:
    """Any real float measurement is ~100% distinct; uniqueness alone must not condemn it."""
    rng = np.random.default_rng(1)
    n = 300
    frame = pd.DataFrame({
        "price": rng.normal(500, 90, n),       # every value distinct, but a measurement
        "weight_kg": rng.gamma(4, 2, n),
        "label": rng.choice([0, 1], n),
    })
    assert find_identifier_columns(frame, target="label") == {}


# --------------------------------------------------------------------------------------
# The API
# --------------------------------------------------------------------------------------


def test_the_example_endpoint_returns_a_complete_readable_analysis() -> None:
    response = client.post("/api/example", data={"goal": "which students will fail"})
    assert response.status_code == 200
    body = response.json()

    assert body["target"]["column"] == "has_backlog"
    assert body["target"]["task"] == "binary classification"
    assert body["result"]["train_rows"] > body["result"]["test_rows"] > 0

    steps = body["steps"]
    assert 5 <= len(steps) <= 8, "the mentor has to read these in five minutes"
    assert [s["n"] for s in steps] == list(range(1, len(steps) + 1))

    # The split is its own step and comes first, because that is the order it happened in.
    assert steps[0]["stage"] == "split"

    for step in steps:
        assert step["explanation"].strip()
        assert step["title"].strip()
        # No jargon reaches the user.
        lowered = step["explanation"].lower()
        for word in ["pipeline", "transformer", "imputer", "encoder", "sklearn",
                     "dataframe", "null", "one-hot"]:
            assert word not in lowered, f"jargon {word!r} leaked into: {step['explanation']}"

    # Every fitted step is badged, and each carries a strip of real values.
    for step in steps[1:]:
        assert step["fitted"] is True
        strip = step["strip"]
        assert strip is not None
        assert len(strip["before"]) == len(strip["after"]) == len(strip["changed"]) == 3
        assert any(strip["changed"]), "a step that changed nothing should not be a card"


def test_the_messy_sample_is_cleaned_and_identifiers_are_set_aside() -> None:
    body = client.post("/api/example", data={"goal": "which students will fail"}).json()
    dropped = {item["column"]: item["reason"] for item in body["dropped"]}

    assert "student_ref" in dropped and "identif" in dropped["student_ref"]
    assert "campus" in dropped and "never changes" in dropped["campus"]
    assert "family_income" in dropped and "%" in dropped["family_income"]


def test_downloads_work_and_the_script_is_valid_python() -> None:
    run_id = client.post("/api/example", data={"goal": "fail"}).json()["run_id"]

    data = client.get(f"/api/runs/{run_id}/data.csv")
    assert data.status_code == 200
    frame = pd.read_csv(io.StringIO(data.text))
    assert len(frame) > 0
    assert not frame.drop(columns=["has_backlog"]).isna().any().any()

    script = client.get(f"/api/runs/{run_id}/script.py")
    assert script.status_code == 200
    compile(script.text, "exported", "exec")
    # It must reproduce the *cleaned* frame, so it carries the reader settings too.
    assert "sep=';'" in script.text
    assert script.text.index("train_test_split") < script.text.index("fit_transform")


def test_a_missing_run_does_not_500() -> None:
    response = client.get("/api/runs/does-not-exist/data.csv")
    assert response.status_code == 404
    assert "error" in response.json()


@pytest.mark.parametrize(
    ("payload", "name"),
    [(b"", "empty.csv"), (b"a,b,t\n1,2,x\n", "tiny.csv"), (b"hello world", "notes.txt")],
)
def test_bad_uploads_return_400_with_a_sentence_not_a_traceback(payload, name) -> None:
    response = client.post(
        "/api/analyse",
        files={"file": (name, payload, "text/csv")},
        data={"goal": "predict something"},
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert "Traceback" not in error
    assert len(error.split()) >= 4


def test_choosing_a_different_target_re_runs_against_that_column() -> None:
    body = client.post(
        "/api/example", data={"goal": "anything", "target": "department"}
    ).json()
    assert body["target"]["column"] == "department"
    assert body["target"]["inferred"] is False
    assert "department" not in [d["column"] for d in body["dropped"]]


def test_a_target_with_one_value_is_refused() -> None:
    response = client.post("/api/example", data={"goal": "x", "target": "campus"})
    assert response.status_code == 400
    assert "nothing to predict" in response.json()["error"]


def test_the_bundled_sample_ships_with_the_app() -> None:
    assert SAMPLE_PATH.exists(), "the offline demo depends on this file being present"
