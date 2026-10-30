"""The runtime engine and the generated script must agree, action by action.

Two artefacts carry out the same plan: :func:`app.runner.execute` fits a pipeline in
this process, and the script :mod:`app.plan_codegen` emits runs standalone. Keeping both
is the product — the editor is why the code exists — but an audit of this repository
found a decision taken over the whole file in one path and over the training fold in the
other, and it was invisible precisely because the other path was right.

These tests make that class of divergence impossible to ship. For each supported action
they build a plan that uses it, run both paths, and compare the **output hash** — a
digest over the values, the column names and the row order. Two runs match on it or they
do not; there is no "close enough" to argue about.

The comparison is end-to-end on purpose. Asserting that the script *contains* the word
``OneHotEncoder`` would pass just as happily when the surrounding line is wrong.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.codegen import SourceFormat
from app.plan import GlobalOptions, PreprocessingPlan, propose
from app.plan_codegen import generate_from_plan
from app.runner import execute
from app.services.ingest import read_upload

FIXTURES = Path(__file__).parent / "fixtures"


def run_script(plan: PreprocessingPlan, csv_path: Path, workspace: Path) -> dict:
    """Write the generated script, run it in a subprocess, and return its report.

    A real subprocess rather than ``exec``: the claim being tested is that a user can
    run this file themselves, and a script that only works inside the test's own
    namespace would not support that claim.
    """
    source = generate_from_plan(plan, SourceFormat(), run_id="parity")
    compile(source, "pipeline.py", "exec")  # fail loudly on a syntax error

    workspace.mkdir(parents=True, exist_ok=True)
    script = workspace / "pipeline.py"
    script.write_text(source, encoding="utf-8")
    report = workspace / "report.json"

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=600,
        env={
            **os.environ,
            "MLBX_INPUT": str(csv_path),
            "MLBX_OUTPUT": str(workspace / "out.csv"),
            "MLBX_REPORT": str(report),
        },
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    return json.loads(report.read_text())


def assert_parity(plan: PreprocessingPlan, frame: pd.DataFrame, csv_path: Path, workspace: Path):
    """Both paths, compared on the digest of what they produced."""
    runtime = execute(frame, plan, run_id="parity")
    report = run_script(plan, csv_path, workspace)

    assert report["output_hash"] == runtime.output_hash, (
        "the generated script produced different data from the engine"
    )
    assert report["features"] == [str(c) for c in runtime.X_train.columns]
    assert report["train_rows"] == len(runtime.X_train)
    assert report["test_rows"] == len(runtime.X_test)
    assert report["stratified"] is runtime.stratified
    return runtime


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def mixed(tmp_path: Path) -> tuple[pd.DataFrame, Path]:
    """One frame holding a column of every kind the planner routes.

    Written to disk and read back through ingestion, because the script reads a file and
    the engine is handed a frame — and the two have to start from the same place.
    """
    rng = np.random.default_rng(21)
    n = 400
    frame = pd.DataFrame({
        "amount": rng.normal(100, 25, n).round(3),
        "count": rng.integers(0, 50, n),
        "skewed": (rng.pareto(1.5, n) * 100).round(3),
        "pair": rng.choice(["yes", "no"], n),
        "grade": rng.choice(["a", "b", "c", "d"], n),
        "city": rng.choice([f"city_{i}" for i in range(40)], n),
        "signed_on": pd.to_datetime("2024-01-01") + pd.to_timedelta(
            rng.integers(0, 600, n), unit="D"
        ),
        "label": rng.choice([0, 1], n, p=[0.45, 0.55]),
    })
    frame.loc[rng.choice(n, 60, replace=False), "amount"] = np.nan
    frame.loc[rng.choice(n, 40, replace=False), "grade"] = None

    path = tmp_path / "mixed.csv"
    frame.to_csv(path, index=False)
    return read_upload(path.read_bytes(), "mixed.csv").frame, path


# --------------------------------------------------------------------------------------
# The proposal itself, on a frame that exercises every routing branch
# --------------------------------------------------------------------------------------


def test_the_proposed_plan_runs_identically_in_both_paths(mixed, workspace) -> None:
    frame, path = mixed
    plan = propose(frame, "label", "binary classification")

    runtime = assert_parity(plan, frame, path, workspace)
    assert runtime.X_train.shape[1] > 0


def test_every_routed_kind_is_actually_present_in_the_fixture(mixed) -> None:
    """Guards the suite below: parity over a frame with no datetime proves nothing."""
    frame, _ = mixed
    plan = propose(frame, "label", "binary classification")

    kinds = {entry.kind for entry in plan.kept()}
    assert {"numeric", "categorical", "datetime"} <= kinds


# --------------------------------------------------------------------------------------
# One test per action
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "column, action",
    [
        ("amount", "impute_median"),
        ("amount", "impute_mean"),
        ("amount", "impute_constant"),
        ("amount", "keep"),
        ("amount", "drop"),
        ("count", "keep"),
        ("grade", "encode_onehot"),
        ("grade", "encode_ordinal"),
        ("grade", "encode_target"),
        ("grade", "encode_frequency"),
        ("grade", "impute_mode"),
        ("pair", "encode_binary"),
        ("city", "encode_frequency"),
        ("city", "encode_target"),
        ("city", "encode_onehot"),
        ("signed_on", "datetime_expand"),
        ("signed_on", "drop"),
    ],
)
def test_each_action_behaves_the_same_in_both_paths(mixed, workspace, column, action) -> None:
    frame, path = mixed
    plan = propose(frame, "label", "binary classification")
    plan.override(column, action)

    assert_parity(plan, frame, path, workspace)


@pytest.mark.parametrize("scaler", ["standard", "minmax", "robust", "none"])
def test_each_scaler_behaves_the_same_in_both_paths(mixed, workspace, scaler) -> None:
    frame, path = mixed
    plan = propose(
        frame, "label", "binary classification", options=GlobalOptions(scaler=scaler)
    )

    assert_parity(plan, frame, path, workspace)


@pytest.mark.parametrize("outliers", ["report", "clip"])
def test_outlier_handling_behaves_the_same_in_both_paths(mixed, workspace, outliers) -> None:
    frame, path = mixed
    plan = propose(
        frame, "label", "binary classification", options=GlobalOptions(outliers=outliers)
    )

    assert_parity(plan, frame, path, workspace)


@pytest.mark.parametrize("indicators", [True, False])
def test_missing_indicators_behave_the_same_in_both_paths(mixed, workspace, indicators) -> None:
    frame, path = mixed
    plan = propose(
        frame,
        "label",
        "binary classification",
        options=GlobalOptions(missing_indicators=indicators),
    )

    runtime = assert_parity(plan, frame, path, workspace)
    flags = [c for c in runtime.X_train.columns if c.endswith("_was_missing")]
    assert bool(flags) is indicators


def test_rare_category_grouping_behaves_the_same_in_both_paths(mixed, workspace) -> None:
    frame, path = mixed
    plan = propose(
        frame,
        "label",
        "binary classification",
        options=GlobalOptions(rare_category_rate=0.05),
    )

    city = plan.for_column("city")
    assert city.rare_categories, "the fixture no longer has rare categories to group"
    assert_parity(plan, frame, path, workspace)


def test_the_power_transform_behaves_the_same_in_both_paths(mixed, workspace) -> None:
    frame, path = mixed
    plan = propose(
        frame,
        "label",
        "binary classification",
        options=GlobalOptions(power_transform_skewed=True),
    )

    skewed = plan.for_column("skewed")
    assert abs(skewed.stats.get("skew", 0.0)) > 1.0, "the fixture is no longer skewed"
    assert_parity(plan, frame, path, workspace)


# --------------------------------------------------------------------------------------
# Representative datasets, not just a constructed one
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture, target, task",
    [
        ("clean_binary.csv", "converted", "binary classification"),
        ("regression.csv", "price", "regression"),
        ("multiclass.csv", None, "multiclass classification"),
        ("high_missing.csv", None, "binary classification"),
        ("id_and_constant.csv", "target", "binary classification"),
    ],
)
def test_real_fixtures_run_identically_in_both_paths(
    workspace, fixture, target, task
) -> None:
    path = FIXTURES / fixture
    frame = read_upload(path.read_bytes(), fixture).frame
    column = target or str(frame.columns[-1])

    plan = propose(frame, column, task)
    assert_parity(plan, frame, path, workspace)


def test_a_regression_target_splits_without_stratifying_in_both_paths(workspace) -> None:
    path = FIXTURES / "regression.csv"
    frame = read_upload(path.read_bytes(), "regression.csv").frame

    plan = propose(frame, "price", "regression")
    runtime = assert_parity(plan, frame, path, workspace)
    assert runtime.stratified is False


def test_a_fold_too_small_for_its_classes_falls_back_in_both_paths(tmp_path) -> None:
    """The shared stratification rule, reached from both sides."""
    rng = np.random.default_rng(22)
    frame = pd.DataFrame({
        "f1": rng.normal(size=60),
        "f2": rng.normal(size=60),
        "label": [f"class_{i % 20}" for i in range(60)],
    })
    path = tmp_path / "manyclass.csv"
    frame.to_csv(path, index=False)
    loaded = read_upload(path.read_bytes(), "manyclass.csv").frame

    plan = propose(loaded, "label", "multiclass classification")
    runtime = assert_parity(plan, loaded, path, tmp_path / "work")
    assert runtime.stratified is False


# --------------------------------------------------------------------------------------
# The script is a copy of the engine, not a retelling of it
# --------------------------------------------------------------------------------------


def test_the_script_carries_the_engines_own_definitions(mixed) -> None:
    frame, _ = mixed
    plan = propose(frame, "label", "binary classification")
    source = generate_from_plan(plan)

    for name in (
        "class FrequencyEncoder",
        "class DatetimeFeatures",
        "class RareCategoryGrouper",
        "class IQRClipper",
        "class CategoricalCast",
        "class MissingFlags",
        "def choose_stratification",
        "def build_pipeline",
        "def output_hash",
    ):
        assert name in source, f"{name} is missing from the generated script"


def test_the_script_declares_the_thresholds_its_definitions_use(mixed) -> None:
    """A constant the engine closes over must travel with the code that reads it."""
    frame, _ = mixed
    plan = propose(frame, "label", "binary classification")
    source = generate_from_plan(plan)

    assert "OUTLIER_IQR_MULTIPLIER = " in source
    assert "TARGET_ENCODER_CV = " in source


def test_the_script_carries_the_plan_as_editable_data(mixed) -> None:
    frame, _ = mixed
    plan = propose(frame, "label", "binary classification")
    source = generate_from_plan(plan)

    start = source.index('PLAN = json.loads(r"""') + len('PLAN = json.loads(r"""')
    document = json.loads(source[start : source.index('""")', start)])
    assert document["target"] == "label"
    assert [c["column"] for c in document["columns"]] == [
        c.column for c in plan.columns
    ]
    assert [c["action"] for c in document["columns"]] == [
        c.action for c in plan.columns
    ]
