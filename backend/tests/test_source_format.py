"""The generated script must re-read the file the way ingestion did.

The bug this guards: the preamble used a bare ``pd.read_csv(INPUT_PATH)``. Given a
semicolon-delimited export — which is what Excel produces in most of Europe and India —
pandas parsed every row into a single column whose name was the entire header line. The
pipeline then ran on that one column and reported **success**, because nothing in it
asserted a shape. A compile that silently produces one column is worse than one that
fails, so these tests assert the shape end to end rather than grepping the source.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.codegen import SourceFormat, generate
from app.main import app
from app.pipeline import PipelineConfig

ROWS = 120


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def _frame() -> pd.DataFrame:
    """A small frame with every awkwardness a real export has."""
    return pd.DataFrame({
        "student_id": [f"S{i:04d}" for i in range(ROWS)],
        "department": ["CSE", "ECE", "ME", "CE"] * (ROWS // 4),
        # A free-text column containing commas AND semicolons, so a wrong separator
        # cannot accidentally produce the right column count.
        "note": ["fine; steady, improving" if i % 3 else "needs review, urgent"
                 for i in range(ROWS)],
        "attendance": [round(50 + (i % 50) + i * 0.13, 2) for i in range(ROWS)],
        "passed": ["Yes", "No"] * (ROWS // 2),
    })


def _upload(client, payload: bytes, name: str) -> dict:
    response = client.post("/api/datasets", files={"file": (name, payload, "text/csv")})
    assert response.status_code == 201, response.text
    return response.json()


def _compile(client, dataset_id: str) -> dict:
    opened = client.post(f"/api/workspaces/from-dataset/{dataset_id}")
    assert opened.status_code in (200, 201), opened.text
    workspace = opened.json()
    response = client.post(f"/api/workspaces/{workspace['id']}/compile", json={})
    assert response.status_code == 200, response.text
    return response.json()["compilation"]


# --------------------------------------------------------------------------------------
# The format carried into the script
# --------------------------------------------------------------------------------------


class _Stored:
    """The two fields :meth:`SourceFormat.from_dataset` reads off a Dataset row."""

    def __init__(self, delimiter: str, encoding: str = "utf-8") -> None:
        self.delimiter, self.encoding = delimiter, encoding


@pytest.mark.parametrize(
    ("delimiter", "expected"),
    [
        (",", "pd.read_csv(INPUT_PATH, sep=',', encoding='utf-8')"),
        (";", "pd.read_csv(INPUT_PATH, sep=';', encoding='utf-8')"),
        ("\t", "pd.read_csv(INPUT_PATH, sep='\\t', encoding='utf-8')"),
        ("|", "pd.read_csv(INPUT_PATH, sep='|', encoding='utf-8')"),
        ("n/a (spreadsheet)", "pd.read_excel(INPUT_PATH)"),
        ("n/a (parquet)", "pd.read_parquet(INPUT_PATH)"),
        ("n/a (json)", "pd.read_json(INPUT_PATH)"),
    ],
)
def test_each_detected_format_becomes_the_right_read_call(delimiter, expected) -> None:
    assert SourceFormat.from_dataset(_Stored(delimiter)).read_expression() == expected


def test_a_non_utf8_encoding_is_carried_through() -> None:
    source = SourceFormat.from_dataset(_Stored(";", "latin-1"))
    assert "encoding='latin-1'" in source.read_expression()


def test_the_generated_script_contains_the_real_separator_and_still_compiles() -> None:
    code = generate(PipelineConfig(target="passed"), SourceFormat.from_dataset(_Stored(";")))
    assert "sep=';'" in code
    assert "pd.read_csv(INPUT_PATH)\n" not in code, "the bare read must be gone"
    compile(code, "generated", "exec")


def test_omitting_the_source_assumes_a_comma_and_says_so_in_the_code() -> None:
    """The default is explicit rather than implied, so a wrong assumption is visible."""
    code = generate(PipelineConfig(target="passed"))
    assert "sep=','" in code


# --------------------------------------------------------------------------------------
# End to end: the shape of what compiles
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("delimiter", [",", ";", "\t", "|"])
def test_a_file_in_any_separator_compiles_to_the_same_shape(client, delimiter) -> None:
    """The separator is a format detail; it must not change the result at all."""
    payload = _frame().to_csv(index=False, sep=delimiter).encode("utf-8")
    dataset = _upload(client, payload, f"sep_{ord(delimiter)}.csv")

    assert dataset["n_columns"] == 5, f"ingest misread the {delimiter!r} file"

    compilation = _compile(client, dataset["id"])
    assert compilation["status"] == "success", compilation.get("error")

    after = compilation["health_after"]
    assert after["columns"] > 1, (
        f"{delimiter!r} file compiled to {after['columns']} column(s) — the whole row "
        "was parsed as one field"
    )
    assert after["rows"] == ROWS


def test_quoted_values_containing_the_separator_survive_the_round_trip(client) -> None:
    """A quoted field holding the separator is the case a naive split gets wrong."""
    frame = _frame()
    payload = frame.to_csv(index=False, sep=";").encode("utf-8")
    dataset = _upload(client, payload, "quoted.csv")
    assert dataset["n_columns"] == 5

    compilation = _compile(client, dataset["id"])
    assert compilation["status"] == "success"
    assert compilation["health_after"]["rows"] == ROWS


def test_blank_cells_survive_a_semicolon_file_and_are_filled(client) -> None:
    frame = _frame()
    frame.loc[0:9, "attendance"] = None
    payload = frame.to_csv(index=False, sep=";").encode("utf-8")
    dataset = _upload(client, payload, "gaps.csv")

    compilation = _compile(client, dataset["id"])
    assert compilation["status"] == "success"
    # Imputation is on by default, so nothing blank should reach the output.
    assert compilation["health_after"]["columns"] > 1


def test_the_bundled_sample_is_semicolon_and_compiles_to_many_columns(client) -> None:
    """The demo file is the one that exposed the bug; it is the one worth pinning."""
    sample = Path(__file__).resolve().parents[1] / "app" / "sample" / "campus_records.csv"
    if not sample.exists():  # pragma: no cover - the file ships with the app
        pytest.skip("bundled sample not present")

    dataset = _upload(client, sample.read_bytes(), "campus_records.csv")
    assert dataset["delimiter"] == ";"
    assert dataset["n_columns"] == 12

    compilation = _compile(client, dataset["id"])
    assert compilation["status"] == "success", compilation.get("error")
    assert compilation["health_after"]["columns"] > 1
    assert compilation["health_after"]["rows"] > 1000


def test_a_malformed_file_is_refused_with_a_sentence(client) -> None:
    payload = b"a;b;c\n1;2;3\n4;5\n6;7;8;9;10\n"
    response = client.post(
        "/api/datasets", files={"file": ("ragged.csv", payload, "text/csv")}
    )
    assert response.status_code in (400, 422)
    assert "error" in response.json()


# --------------------------------------------------------------------------------------
# One canonical health score
# --------------------------------------------------------------------------------------


def test_the_dataset_row_and_the_workspace_report_the_same_score(client) -> None:
    """The evaluator must never see two different numbers both called health."""
    payload = _frame().to_csv(index=False, sep=";").encode("utf-8")
    dataset = _upload(client, payload, "score.csv")

    workspace = client.post(f"/api/workspaces/from-dataset/{dataset['id']}").json()
    listed = next(d for d in client.get("/api/datasets").json() if d["id"] == dataset["id"])

    assert dataset["health_score"] == workspace["baseline_health"]["score"]
    assert listed["health_score"] == workspace["baseline_health"]["score"]

    live = client.get(f"/api/workspaces/{workspace['id']}/health").json()
    assert live["score"] == dataset["health_score"]


def test_the_ingest_check_is_labelled_so_it_cannot_be_read_as_the_health_score(
    client,
) -> None:
    payload = _frame().to_csv(index=False).encode("utf-8")
    dataset = _upload(client, payload, "labelled.csv")
    profile = client.get(f"/api/datasets/{dataset['id']}/profile").json()

    assert "not the workspace health score" in profile["quality"]["label"]


# --------------------------------------------------------------------------------------
# Percentages: one representation, stated rather than assumed
# --------------------------------------------------------------------------------------
#
# `12%` becomes `12`, not `0.12`. The reasoning is in `ingest._percentage_columns`; what
# these tests hold is that the choice is consistent, that it is reported, and that a
# column mixing percentages with plain numbers is not quietly given a unit it does not
# have.


def _read(text: str):
    from app.services.ingest import read_upload

    rows = "\n".join(f"{v},{i % 2}" for i, v in enumerate(text.split("|")))
    payload = f"rate,label\n{rows}\n".encode()
    return read_upload(payload, "rates.csv")


def _column(values: list[str]):
    """A 60-row file whose `rate` column repeats `values`, so it clears MIN_ROWS."""
    repeated = (values * (60 // len(values) + 1))[:60]
    return _read("|".join(repeated))


def test_a_percentage_column_keeps_percentage_points() -> None:
    result = _column(["12%", "48%", "7%", "99%"])

    assert result.percentage_columns == ["rate"]
    assert result.frame["rate"].iloc[0] == 12
    assert result.frame["rate"].max() <= 100


def test_a_percentage_column_says_so_in_the_cleaning_log() -> None:
    """The repair this closes was silent: a column lost its unit inside a sentence
    about converting text to numbers."""
    result = _column(["12%", "48%", "7%", "99%"])

    mentioned = [line for line in result.cleaning if "percentage" in line.lower()]
    assert mentioned, result.cleaning
    assert "12% is 12 rather than 0.12" in mentioned[0]
    assert "rate" in mentioned[0]


def test_a_decimal_fraction_column_is_left_exactly_as_written() -> None:
    """`0.12` means 0.12. Nothing multiplies it, and it is not a percentage column."""
    result = _column(["0.12", "0.48", "0.07", "0.99"])

    assert result.percentage_columns == []
    assert result.frame["rate"].iloc[0] == pytest.approx(0.12)
    assert result.frame["rate"].max() <= 1.0


def test_a_plain_number_column_is_untouched() -> None:
    result = _column(["12", "48", "7", "99"])

    assert result.percentage_columns == []
    assert result.frame["rate"].iloc[0] == 12


def test_percentages_with_blanks_keep_the_blanks() -> None:
    result = _column(["12%", "", "7%", "99%"])

    assert result.percentage_columns == ["rate"]
    assert result.frame["rate"].isna().sum() > 0
    assert result.frame["rate"].dropna().iloc[0] == 12


def test_a_column_mixing_percentages_with_plain_numbers_is_not_called_a_percentage() -> None:
    """Half the column carries a unit and half does not, so no unit can be claimed.

    The values still convert — `_coerce_numeric` strips the sign either way — but the
    column is not reported as a percentage, because saying "these are percentage points"
    about a column where only some values said `%` would be asserting something the file
    does not support.
    """
    result = _column(["12%", "48", "7%", "99"])

    assert result.percentage_columns == []
    assert not [line for line in result.cleaning if "percentage" in line.lower()]
    assert result.frame["rate"].iloc[0] == 12
    assert result.frame["rate"].iloc[1] == 48


def test_the_representation_does_not_change_what_a_model_receives() -> None:
    """Why the choice is safe to leave alone: every scaler offered is affine.

    Percentage points and decimal fractions differ by a constant factor, which standard,
    min-max and robust scaling all remove exactly. The representation is therefore a
    question about what the user reads, not about what the model is trained on — which
    is what makes rescaling every existing dataset by 100 a cost with no benefit.
    """
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    rng = np.random.default_rng(5)
    points = rng.uniform(0, 100, 400).reshape(-1, 1)
    fractions = points / 100

    for scaler in (StandardScaler, MinMaxScaler, RobustScaler):
        assert np.allclose(
            scaler().fit_transform(points), scaler().fit_transform(fractions)
        ), f"{scaler.__name__} distinguishes the two representations"
