"""One test per preprocessing rule added or changed in the Phase 1 audit.

Each test names the rule it is defending and fails for exactly one reason. Where a test
exists to prove an *absence* — that the test fold never reached a fit, that a flagged
column was not dropped — it is written so that the obvious wrong implementation makes it
fail, rather than so that the right one makes it pass.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.config import (
    HIGH_CORRELATION,
    LEAKAGE_CORRELATION,
    ONEHOT_MAX_CARDINALITY,
)
from app.services.glassbox import (
    DatetimeFeatures,
    build_plan,
    fit_preprocessing,
    profile_frame,
)
from app.services.profile import outlier_summary
from app.services.screening import (
    correlated_pairs,
    identifier_screen,
    target_leakage_screen,
)
from app.services.target import class_balance, describe_task

N = 300


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "amount": np.round(rng.gamma(4, 25, N), 2),
        "tier": rng.choice(["bronze", "silver", "gold"], N),
        "region": rng.choice([f"r{i:02d}" for i in range(30)], N),
        "active": rng.choice([True, False], N),
        "joined": pd.to_datetime("2021-01-01") + pd.to_timedelta(rng.integers(0, 900, N), unit="D"),
        "label": rng.choice([0, 1], N, p=[0.45, 0.55]),
    })


# --------------------------------------------------------------------------------------
# Scaler selection
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scaler,expected",
    [("standard", "StandardScaler"), ("minmax", "MinMaxScaler"), ("robust", "RobustScaler")],
)
def test_each_scaler_is_used_and_named_in_the_log(frame, scaler, expected) -> None:
    result = fit_preprocessing(frame, "label", "classification", scaler=scaler)
    step = next(e for e in result.log if e["stage"] == "scale")
    assert step["transformer"] == expected
    assert type(result.pipeline.named_transformers_["numeric"].named_steps["scale"]).__name__ == expected


def test_minmax_puts_training_values_in_zero_to_one(frame) -> None:
    result = fit_preprocessing(frame, "label", "classification", scaler="minmax")
    values = result.X_train["amount"]
    assert values.min() == pytest.approx(0.0, abs=1e-9)
    assert values.max() == pytest.approx(1.0, abs=1e-9)


def test_robust_centres_on_the_training_median(frame) -> None:
    result = fit_preprocessing(frame, "label", "classification", scaler="robust")
    entry = next(e for e in result.log if e["stage"] == "scale")
    scaler = result.pipeline.named_transformers_["numeric"].named_steps["scale"]
    # The logged centre is read off the fitted object, not recomputed alongside it.
    assert entry["details"]["mean"]["amount"] == pytest.approx(float(scaler.center_[0]))


def test_an_unknown_scaler_is_refused_by_name(frame) -> None:
    with pytest.raises(ValueError, match="scaler must be one of"):
        fit_preprocessing(frame, "label", "classification", scaler="quantile")


# --------------------------------------------------------------------------------------
# High-cardinality encoding
# --------------------------------------------------------------------------------------


def test_frequency_is_the_default_and_never_reads_the_target(frame) -> None:
    result = fit_preprocessing(frame, "label", "classification")
    entry = next(e for e in result.log if e["stage"] == "encode" and e["columns"] == ["region"])
    assert entry["transformer"] == "FrequencyEncoder"
    # Frequencies must sum to 1 across the training fold — a target mean would not.
    table = entry["details"]["frequencies"]["region"]
    assert sum(table.values()) == pytest.approx(1.0)


def test_target_encoding_is_selectable_and_cross_fitted(frame) -> None:
    result = fit_preprocessing(
        frame, "label", "classification", high_card_encoder="target"
    )
    entry = next(e for e in result.log if e["transformer"] == "TargetEncoder")
    encoder = result.pipeline.named_transformers_["high_card"].named_steps["encode"]

    assert entry["details"]["cv"] == encoder.cv
    # The training output is the cross-fitted one, so a row's encoded value is not the
    # mean of the category it belongs to. If it were, this would be exact for every row.
    merged = pd.DataFrame({
        "encoded": result.X_train["region"].to_numpy(),
        "category": frame.loc[result.X_train.index, "region"].to_numpy(),
    })
    full_means = merged.groupby("category")["encoded"].transform("mean")
    assert not np.allclose(merged["encoded"], full_means)


def test_high_card_columns_cost_one_column_each(frame) -> None:
    result = fit_preprocessing(frame, "label", "classification")
    assert result.plan.high_card == ["region"]
    assert sum(1 for c in result.X_train.columns if c.startswith("region")) == 1


def test_the_onehot_threshold_is_the_configured_one() -> None:
    profiles = profile_frame(pd.DataFrame({
        "at_limit": [f"v{i % ONEHOT_MAX_CARDINALITY}" for i in range(N)],
        "over_limit": [f"v{i % (ONEHOT_MAX_CARDINALITY + 1)}" for i in range(N)],
        "y": [0, 1] * (N // 2),
    }))
    plan = build_plan(profiles, target="y", task_type="classification")
    assert plan.low_card == ["at_limit"]
    assert plan.high_card == ["over_limit"]


# --------------------------------------------------------------------------------------
# Datetime features
# --------------------------------------------------------------------------------------


def test_datetime_columns_become_calendar_features_not_drops(frame) -> None:
    result = fit_preprocessing(frame, "label", "classification")
    assert result.plan.datetime == ["joined"]
    assert "joined" not in result.plan.dropped

    produced = [c for c in result.X_train.columns if c.startswith("joined_")]
    for part in ("year", "month", "day", "dayofweek", "hour"):
        assert f"joined_{part}" in produced
    for part in ("month", "day", "dayofweek", "hour"):
        assert f"joined_{part}_sin" in produced and f"joined_{part}_cos" in produced


def test_cyclical_encoding_puts_december_next_to_january() -> None:
    """The whole point of the sine/cosine pair, stated as a distance."""
    stamps = pd.DataFrame({"t": pd.to_datetime(["2024-12-15", "2024-01-15", "2024-06-15"])})
    out = DatetimeFeatures().fit(stamps).transform(stamps)

    def point(row: int) -> np.ndarray:
        return np.array([out.iloc[row]["t_month_sin"], out.iloc[row]["t_month_cos"]])

    dec_to_jan = np.linalg.norm(point(0) - point(1))
    dec_to_jun = np.linalg.norm(point(0) - point(2))
    assert dec_to_jan < dec_to_jun

    # The raw month numbers say the opposite, which is why the pair is needed at all.
    assert abs(12 - 1) > abs(12 - 6)


def test_a_blank_timestamp_produces_no_blank_features() -> None:
    stamps = pd.DataFrame({"t": pd.to_datetime(["2024-03-01", None, "2024-09-09"])})
    out = DatetimeFeatures().fit(stamps).transform(stamps)
    assert not out.isna().any().any()


def test_datetime_features_map_back_to_their_source_column(frame) -> None:
    result = fit_preprocessing(frame, "label", "classification")
    derived = {k: v for k, v in result.feature_map.items() if k.startswith("joined_")}
    assert derived and set(derived.values()) == {"joined"}


# --------------------------------------------------------------------------------------
# Leakage screen — flag, never drop
# --------------------------------------------------------------------------------------


def test_a_copy_of_the_target_is_flagged_and_excluded(frame) -> None:
    """Spec §4c: a leaked column is excluded by default, not merely mentioned.

    The asymmetry is the argument. Excluding a legitimate column costs one click to put
    back; keeping a leaked one costs a model that scores well in testing and fails in
    production, with the warning that would have prevented it sitting unread in a panel.
    """
    leaked = frame.copy()
    rng = np.random.default_rng(0)
    leaked["score"] = leaked["label"] * 100 + rng.normal(0, 0.01, len(leaked))

    result = fit_preprocessing(
        leaked, "label", "classification", screeners=[target_leakage_screen]
    )

    flagged = [f["column"] for f in result.flags]
    assert "score" in flagged

    flag = next(f for f in result.flags if f["column"] == "score")
    assert abs(flag["correlation"]) >= LEAKAGE_CORRELATION
    assert flag["severity"] == "critical"
    assert flag["excluded"] is True
    assert flag["can_reinclude"] is True, "the user must be able to put it back"

    # Excluded, and out of the feature matrix entirely.
    assert "score" in result.excluded
    assert "score" not in result.plan.numeric
    assert "score" not in result.X_train.columns


def test_an_ordinary_feature_is_not_flagged(frame) -> None:
    result = fit_preprocessing(
        frame, "label", "classification", screeners=[target_leakage_screen]
    )
    assert result.flags == []


def test_the_leakage_screen_reads_only_the_training_fold(frame) -> None:
    """Planting a perfect copy in the held-out rows only must not trigger the screen."""
    leaked = frame.copy()
    leaked["sneaky"] = np.random.default_rng(2).normal(size=len(leaked))

    result = fit_preprocessing(
        leaked, "label", "classification", screeners=[target_leakage_screen]
    )
    test_rows = result.X_test.index
    leaked.loc[test_rows, "sneaky"] = leaked.loc[test_rows, "label"].astype(float)

    again = fit_preprocessing(
        leaked, "label", "classification", screeners=[target_leakage_screen]
    )
    assert "sneaky" not in [f["column"] for f in again.flags]


# --------------------------------------------------------------------------------------
# Correlated pairs and outliers
# --------------------------------------------------------------------------------------


def test_redundant_columns_are_reported_as_a_pair() -> None:
    rng = np.random.default_rng(5)
    base = rng.normal(size=N)
    df = pd.DataFrame({
        "metres": base,
        "centimetres": base * 100 + rng.normal(0, 1e-6, N),
        "unrelated": rng.normal(size=N),
    })
    pairs = correlated_pairs(df)
    assert [set(p["columns"]) for p in pairs] == [{"metres", "centimetres"}]
    assert abs(pairs[0]["correlation"]) >= HIGH_CORRELATION


def test_iqr_outliers_are_counted_against_tukeys_fences() -> None:
    values = pd.Series(list(range(1, 101)) + [10_000, -10_000])
    summary = outlier_summary(values)
    assert summary is not None and summary["count"] == 2


def test_a_column_with_no_spread_reports_no_outliers() -> None:
    """A collapsed IQR would otherwise condemn every value that is not the mode."""
    summary = outlier_summary(pd.Series([5] * 90 + [6] * 10))
    assert summary is not None and summary["count"] == 0


def test_text_columns_have_no_outlier_summary() -> None:
    assert outlier_summary(pd.Series(["a", "b", "c"])) is None


# --------------------------------------------------------------------------------------
# Task detection and class balance
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "series,expected",
    [
        (pd.Series([0, 1] * 50), "binary classification"),
        (pd.Series(["yes", "no"] * 50), "binary classification"),
        (pd.Series([f"c{i % 5}" for i in range(100)]), "multiclass classification"),
        # The audit's bug: 30 text categories used to be called regression, which
        # silently disabled stratification.
        (pd.Series([f"c{i % 30}" for i in range(300)]), "multiclass classification"),
        (pd.Series(np.linspace(0.5, 99.5, 200)), "regression"),
        (pd.Series(np.arange(200)), "regression"),
    ],
)
def test_task_detection(series, expected) -> None:
    assert describe_task(series) == expected


def test_a_lopsided_target_is_reported_with_its_ratio() -> None:
    balance = class_balance(pd.Series(["no"] * 95 + ["yes"] * 5))
    assert balance is not None
    assert balance["ratio"] == pytest.approx(19.0)
    assert balance["minority_class"] == "yes"


def test_a_balanced_target_reports_its_ratio_without_warning() -> None:
    """The ratio is always available; only the verdict changes.

    Returning None for a balanced target left the interface with nothing to show, so a
    user could not tell a balanced dataset from one that had not been checked.
    """
    balance = class_balance(pd.Series(["a"] * 50 + ["b"] * 50))
    assert balance is not None
    assert balance["ratio"] == 1.0
    assert balance["imbalanced"] is False
    assert balance["severity"] == "info"


def test_a_regression_target_has_no_class_balance() -> None:
    """A continuous column would otherwise enumerate one 'class' per distinct value."""
    assert class_balance(pd.Series(range(500))) is None


# --------------------------------------------------------------------------------------
# Identifier screen
# --------------------------------------------------------------------------------------


def test_identifiers_are_screened_out_on_the_training_fold(frame) -> None:
    withid = frame.copy()
    withid.insert(0, "record_id", [f"ID{i:05d}" for i in range(len(withid))])

    result = fit_preprocessing(
        withid, "label", "classification", screeners=[identifier_screen]
    )
    assert "record_id" in result.excluded
    assert "record_id" not in result.X_train.columns


def test_a_caller_exclusion_still_wins(frame) -> None:
    result = fit_preprocessing(
        frame, "label", "classification",
        excluded={"amount": "you deselected this column"},
        screeners=[identifier_screen],
    )
    assert result.plan.dropped["amount"] == "you deselected this column"


# --------------------------------------------------------------------------------------
# Log record shape
# --------------------------------------------------------------------------------------


def test_every_log_record_carries_the_spec_fields(frame) -> None:
    result = fit_preprocessing(frame, "label", "classification")
    assert result.log
    for entry in result.log:
        for field in ("step", "column", "action", "params", "reason", "before", "after"):
            assert field in entry, f"{field} missing from {entry['stage']} record"
        assert entry["reason"] and entry["reason"][0].isupper()
        assert entry["before"]["n_columns"] >= 1
        assert entry["after"]["n_columns"] >= 1


def test_impute_records_claim_no_blanks_afterwards(frame) -> None:
    holes = frame.copy()
    holes.loc[holes.index[:40], "amount"] = np.nan
    result = fit_preprocessing(holes, "label", "classification")

    entry = next(e for e in result.log if e["stage"] == "impute" and "amount" in e["columns"])
    assert entry["before"]["missing_values"] > 0
    assert entry["after"]["missing_values"] == 0
    assert not result.X_train.isna().any().any()


def test_the_log_stays_json_serialisable_with_every_option(frame) -> None:
    for scaler in ("standard", "minmax", "robust"):
        for encoder in ("frequency", "target"):
            result = fit_preprocessing(
                frame, "label", "classification",
                scaler=scaler, high_card_encoder=encoder,
                screeners=[identifier_screen, target_leakage_screen],
            )
            json.dumps(result.log)
            json.dumps(result.flags)


# --------------------------------------------------------------------------------------
# Date recognition on the way in
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values,should_convert",
    [
        (["2024-01-05", "2024-02-06", "2024-03-07"], True),
        (["01/05/2024", "02/06/2024", "03/07/2024"], True),
        (["2024-01-05 14:30", "2024-02-06 09:15", "2024-03-07 23:00"], True),
        (["5.1.2024", "6.2.2024", "7.3.2024"], True),
        # The dangerous near-misses. Turning any of these into 1970 timestamps would
        # destroy the column silently, which is worse than missing a date.
        (["2000", "2001", "2002"], False),
        (["1.2.3", "4.5.6", "7.8.9"], False),
        (["12", "7", "99"], False),
        (["1/2", "3/4", "5/6"], False),
        (["2024-01-05", "not a date", "2024-03-07"], False),
    ],
)
def test_only_genuine_dates_are_parsed_as_dates(values, should_convert) -> None:
    from app.services.ingest import _coerce_datetime

    df = pd.DataFrame({"col": values * 30})
    converted = _coerce_datetime(df)

    assert bool(converted) is should_convert
    if should_convert:
        assert pd.api.types.is_datetime64_any_dtype(df["col"])
    else:
        assert df["col"].dtype == object


def test_a_date_column_in_an_uploaded_csv_becomes_calendar_features() -> None:
    """The end-to-end path: a CSV has no dtypes, so this only works if ingest parses."""
    from app.services.ingest import read_upload

    rng = np.random.default_rng(21)
    n = 200
    csv = pd.DataFrame({
        "signed_up": pd.to_datetime("2022-01-01")
        + pd.to_timedelta(rng.integers(0, 700, n), unit="D"),
        "amount": rng.normal(100, 20, n),
        "label": rng.choice([0, 1], n),
    }).to_csv(index=False).encode("utf-8")

    ingested = read_upload(csv, "dates.csv")
    assert pd.api.types.is_datetime64_any_dtype(ingested.frame["signed_up"])
    assert "signed_up" in ingested.datetime_columns

    result = fit_preprocessing(ingested.frame, "label", "classification")
    assert result.plan.datetime == ["signed_up"]
    assert "signed_up_month_sin" in result.X_train.columns


def test_the_exported_script_reproduces_the_apps_split_exactly() -> None:
    """The script's whole claim. An off-by-one row makes it false.

    ``test_size`` must be the value the run used, not one derived back from the row
    counts: 179/891 rounds to 0.2009, and sklearn's ceiling on that is 180 rows, not
    the 179 the app actually held out.
    """
    import re
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    from app.services.glassbox import export_script

    rng = np.random.default_rng(1912)
    n = 891  # a row count where the rounding bites
    df = pd.DataFrame({
        "amount": rng.normal(100, 15, n),
        "tier": rng.choice(["a", "b", "c"], n),
        "label": rng.choice([0, 1], n),
    })

    result = fit_preprocessing(df, "label", "classification", test_size=0.2)

    workspace = Path(tempfile.mkdtemp())
    csv = workspace / "data.csv"
    df.to_csv(csv, index=False)

    source = export_script(result, csv=str(csv), exp_id="t", seed=result.seed)
    script = workspace / "prep.py"
    script.write_text(source)

    run = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr

    shapes = re.search(r"train: \((\d+), (\d+)\)\s+test: \((\d+), (\d+)\)", run.stdout)
    assert shapes, run.stdout
    train_rows, train_cols, test_rows, test_cols = (int(g) for g in shapes.groups())

    assert train_rows == len(result.X_train)
    assert test_rows == len(result.X_test)
    assert train_cols == result.X_train.shape[1]
    assert test_cols == result.X_test.shape[1]


# --------------------------------------------------------------------------------------
# File formats
# --------------------------------------------------------------------------------------


@pytest.fixture
def titanic() -> pd.DataFrame:
    from pathlib import Path

    return pd.read_csv(Path(__file__).parent / "fixtures" / "titanic.csv")


def test_json_records_are_read(titanic) -> None:
    """A list of objects — what most APIs and exports produce."""
    import json as json_module

    from app.services.ingest import read_upload

    raw = json_module.dumps(titanic.to_dict(orient="records"), default=str).encode()
    result = read_upload(raw, "records.json")
    assert result.rows == 891
    assert result.columns == 13


def test_json_columns_are_read(titanic) -> None:
    """What `DataFrame.to_json()` produces by default, so a round trip must survive."""
    from app.services.ingest import read_upload

    result = read_upload(titanic.to_json(orient="columns").encode(), "columns.json")
    assert result.rows == 891


def test_line_delimited_json_is_read(titanic) -> None:
    """What streaming tools emit."""
    from app.services.ingest import read_upload

    raw = titanic.to_json(orient="records", lines=True).encode()
    result = read_upload(raw, "events.jsonl")
    assert result.rows == 891


def test_parquet_is_read(titanic) -> None:
    import io as io_module

    from app.services.ingest import read_upload

    buffer = io_module.BytesIO()
    titanic.to_parquet(buffer, index=False)
    result = read_upload(buffer.getvalue(), "data.parquet")
    assert result.rows == 891
    assert result.columns == 13


@pytest.mark.parametrize(
    "name,payload,fragment",
    [
        ("broken.json", b"{not json", "valid JSON"),
        ("scalar.json", b"42", "single value"),
        ("broken.parquet", b"not a parquet file at all", "Parquet"),
        ("notes.pdf", b"%PDF-1.4", "different extension"),
        # A PDF wearing a .csv name is caught by its contents, not its extension.
        ("disguised.csv", b"%PDF-1.4", "it is a PDF"),
    ],
)
def test_an_unreadable_file_is_refused_with_a_sentence(name, payload, fragment) -> None:
    from app.services.ingest import IngestError, read_upload

    with pytest.raises(IngestError) as caught:
        read_upload(payload, name)
    assert fragment in str(caught.value)


# --------------------------------------------------------------------------------------
# Ingestion rules added in slice 2 (spec §1)
# --------------------------------------------------------------------------------------


def _read(name: str, **kwargs):
    from app.services.ingest import read_upload
    from tests.fixtures import generate as fixtures

    path = fixtures.ensure(name)
    return read_upload(path.read_bytes(), path.name, **kwargs)


@pytest.mark.parametrize(
    "payload,name,fragment",
    [
        (b"\x7fELF\x02\x01", "data.csv", "an executable"),
        (b"%PDF-1.4", "data.csv", "a PDF"),
        (b"\x89PNG\r\n", "data.csv", "an image"),
        (b"SQLite format 3\x00", "data.csv", "a SQLite database"),
        (b"not a zip at all", "book.xlsx", "contents are not one"),
        (b"not parquet", "data.parquet", "contents are not"),
    ],
)
def test_content_is_checked_not_just_the_extension(payload, name, fragment) -> None:
    """Spec §7. An extension is a claim by whoever uploaded the file."""
    from app.services.ingest import IngestError, read_upload

    with pytest.raises(IngestError, match=fragment):
        read_upload(payload, name)


def test_a_headerless_file_is_refused() -> None:
    """Accepting it costs an observation and names every column after a value."""
    from app.services.ingest import IngestError

    with pytest.raises(IngestError, match="no header row"):
        _read("headerless.csv")


def test_a_numeric_column_name_does_not_make_a_file_headerless() -> None:
    """A spreadsheet with a column honestly called 2024 is not headerless.

    The rule requires *every* heading to be numeric, which is what keeps this safe.
    """
    from app.services.ingest import read_upload

    csv = "region,2024,revenue\n" + "\n".join(f"north,{i},{i * 10}" for i in range(80))
    result = read_upload(csv.encode(), "sales.csv")
    assert list(result.frame.columns) == ["region", "2024", "revenue"]


@pytest.mark.parametrize(
    "values,expected",
    [
        (["yes", "no"], True),
        (["Y", "N"], True),
        (["TRUE", "false"], True),
        (["t", "f"], True),
        (["yes", "maybe"], False),      # not every value is a known spelling
        (["yes", "yes"], False),        # constant; calling it boolean hides that
        (["1", "0"], False),            # already numbers, and as likely a code
    ],
)
def test_boolean_coercion(values, expected) -> None:
    from app.services.ingest import read_upload

    n = 80
    csv = "flag,pad\n" + "\n".join(
        f"{values[i % len(values)]},{i}" for i in range(n)
    )
    result = read_upload(csv.encode(), "t.csv")
    is_boolean = pd.api.types.is_bool_dtype(result.frame["flag"])
    assert is_boolean is expected


def test_dates_parse_at_ninety_percent_not_only_at_a_hundred() -> None:
    """Spec §1. Refusing a column because 5% of it is junk leaves all of it as text."""
    from app.services.ingest import read_upload

    good = [f"2024-{(i % 12) + 1:02d}-05" for i in range(95)]
    bad = ["not a date"] * 5
    csv = "when,pad\n" + "\n".join(f"{v},{i}" for i, v in enumerate(good + bad))

    result = read_upload(csv.encode(), "t.csv")

    assert pd.api.types.is_datetime64_any_dtype(result.frame["when"])
    assert "when" in result.datetime_columns
    # The rows that could not be read became blanks, and that is reported rather than
    # quietly absorbed.
    assert result.unparsed_dates["when"] == 5
    assert any("could not be read as a date" in line for line in result.cleaning)


def test_a_column_below_the_date_rate_stays_text() -> None:
    from app.services.ingest import read_upload

    good = [f"2024-{(i % 12) + 1:02d}-05" for i in range(70)]
    bad = ["not a date"] * 30
    csv = "when,pad\n" + "\n".join(f"{v},{i}" for i, v in enumerate(good + bad))

    result = read_upload(csv.encode(), "t.csv")
    assert result.frame["when"].dtype == object


def test_the_original_to_clean_header_mapping_is_kept() -> None:
    """Spec §1. Every message names a column; it must be the name the user typed."""
    result = _read("unicode_duplicate_headers.csv")

    mapping = result.column_mapping
    # A heading written twice maps to both cleaned names.
    assert mapping["région"] == ["région", "région_1"]
    # Whitespace is preserved on the left of the mapping and stripped on the right.
    assert mapping[" naïve  "] == "naïve"


@pytest.mark.parametrize("sheet,rows", [(None, 600), ("first", 600), ("second", 500)])
def test_a_workbook_sheet_can_be_chosen(sheet, rows) -> None:
    result = _read("two_sheets.xlsx", sheet_name=sheet)
    assert result.rows == rows
    assert result.available_sheets == ["first", "second"]


def test_asking_for_a_sheet_that_is_not_there_lists_the_ones_that_are() -> None:
    """Falling back to the first sheet silently would show numbers from the wrong one."""
    from app.services.ingest import IngestError

    with pytest.raises(IngestError, match="first, second"):
        _read("two_sheets.xlsx", sheet_name="Q3 Actuals")


def test_a_near_constant_column_is_treated_as_constant() -> None:
    """Spec §1. 99.4% one value carries almost nothing but costs a whole encoder."""
    from app.services.glassbox import profile_frame

    frame = pd.read_csv(_fixture_path("id_and_constant"))
    kinds = {name: profile.kind for name, profile in profile_frame(frame).items()}
    assert kinds["near_constant"] == "constant"
    assert kinds["country"] == "constant"
    assert kinds["score"] == "numeric"


def _fixture_path(name: str):
    from tests.fixtures import generate as fixtures

    return fixtures.ensure(name)


def test_an_identifier_column_is_caught_at_the_spec_threshold() -> None:
    """Spec §1 puts the bar at 0.95 unique; it had been set at 0.98."""
    from app.services.screening import find_identifier_columns

    frame = pd.read_csv(_fixture_path("id_and_constant"))
    found = find_identifier_columns(frame, target="target")
    assert "record_id" in found
    assert "reference_no" in found
    assert "score" not in found
