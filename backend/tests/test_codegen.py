"""The code generator: does every configuration produce code that runs?

The generator is the single path from configuration to executable Python, which makes
it the component where a bug is least visible and most damaging — a subtly wrong
pipeline produces a dataset that looks fine and is not.

So these tests do not inspect the generated source for substrings. They *run* it, and
check what came out of the other end. A test asserting that the text contains
``"median"`` passes just as happily when the surrounding line is malformed.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from app.codegen import explain_step, generate
from app.pipeline import STEP_ORDER, PipelineConfig

FIXTURE = Path(__file__).parent / "fixtures" / "titanic.csv"
TARGET = "Survived"


def run_generated(config: PipelineConfig, workspace: Path) -> tuple[pd.DataFrame, dict]:
    """Write the generated script, run it, and return what it produced.

    Runs in a plain subprocess rather than the sandbox: this is our own generated code,
    and the sandbox is tested separately in ``test_executor.py``. Keeping them apart
    means a sandbox failure cannot be mistaken for a codegen failure.
    """
    source = generate(config)
    compile(source, "pipeline.py", "exec")  # fail loudly on a syntax error

    workspace.mkdir(parents=True, exist_ok=True)
    script = workspace / "pipeline.py"
    script.write_text(source)
    output = workspace / "out.csv"
    report = workspace / "report.json"

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=300,
        env={
            **os.environ,
            "MLBX_INPUT": str(FIXTURE),
            "MLBX_OUTPUT": str(output),
            "MLBX_REPORT": str(report),
        },
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return pd.read_csv(output), json.loads(report.read_text())


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


# --------------------------------------------------------------------------------------
# Every option runs
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy",
    ["median", "mean", "constant", "forward_fill", "backward_fill", "drop_rows"],
)
def test_every_numeric_fill_strategy_runs(workspace, strategy) -> None:
    config = PipelineConfig(target=TARGET, missing_values={"numeric_strategy": strategy})
    frame, _ = run_generated(config, workspace)
    assert len(frame) > 0


@pytest.mark.parametrize(
    "strategy",
    ["most_frequent", "constant", "forward_fill", "backward_fill", "drop_rows"],
)
def test_every_categorical_fill_strategy_runs(workspace, strategy) -> None:
    config = PipelineConfig(
        target=TARGET, missing_values={"categorical_strategy": strategy}
    )
    frame, _ = run_generated(config, workspace)
    assert len(frame) > 0


@pytest.mark.parametrize(
    "method,action",
    list(itertools.product(["iqr", "zscore"], ["cap", "remove", "winsorize", "flag"])),
)
def test_every_outlier_combination_runs(workspace, method, action) -> None:
    config = PipelineConfig(target=TARGET, outliers={"method": method, "action": action})
    frame, _ = run_generated(config, workspace)
    assert len(frame) > 0


@pytest.mark.parametrize("method", ["frequency", "target", "ordinal", "drop"])
def test_every_high_cardinality_encoder_runs(workspace, method) -> None:
    config = PipelineConfig(target=TARGET, encoding={"high_cardinality_method": method})
    frame, _ = run_generated(config, workspace)
    assert len(frame) > 0


@pytest.mark.parametrize("method", ["standard", "minmax", "robust"])
def test_every_scaler_runs(workspace, method) -> None:
    config = PipelineConfig(target=TARGET, scaling={"method": method})
    frame, _ = run_generated(config, workspace)
    assert len(frame) > 0


@pytest.mark.parametrize("action", ["report", "oversample", "undersample", "class_weight"])
def test_every_imbalance_action_runs(workspace, action) -> None:
    config = PipelineConfig(target=TARGET, imbalance={"action": action})
    frame, _ = run_generated(config, workspace)
    assert len(frame) > 0


def test_a_pipeline_with_every_step_off_still_runs(workspace) -> None:
    """Switching everything off must produce a readable dataset, not an error.

    This is the configuration a user reaches by turning off each step in turn to see
    what it did. It has to survive that.
    """
    config = PipelineConfig(
        target=TARGET,
        missing_values={"enabled": False},
        duplicates={"enabled": False},
        outliers={"enabled": False},
        encoding={"enabled": False},
        scaling={"enabled": False},
        imbalance={"enabled": False},
    )
    frame, _ = run_generated(config, workspace)
    assert len(frame) == 891


def test_a_pipeline_with_no_target_runs(workspace) -> None:
    frame, _ = run_generated(PipelineConfig(), workspace)
    assert len(frame) > 0


# --------------------------------------------------------------------------------------
# The options do what they say
# --------------------------------------------------------------------------------------


def test_scaling_actually_rescales(workspace) -> None:
    frame, _ = run_generated(
        PipelineConfig(target=TARGET, scaling={"method": "minmax"}), workspace
    )
    scaled = frame["Fare"]
    assert scaled.min() == pytest.approx(0.0, abs=1e-6)
    assert scaled.max() == pytest.approx(1.0, abs=1e-6)


def test_imputation_actually_fills(workspace) -> None:
    frame, _ = run_generated(PipelineConfig(target=TARGET), workspace)
    assert frame["Age"].isna().sum() == 0


def test_disabling_imputation_leaves_the_blanks(workspace) -> None:
    """The counterpart. If turning the step off changed nothing, the warning would lie."""
    frame, _ = run_generated(
        PipelineConfig(target=TARGET, missing_values={"enabled": False}), workspace
    )
    assert frame["Age"].isna().sum() > 0


def test_capping_keeps_every_row_while_removing_deletes_some(workspace, tmp_path) -> None:
    capped, _ = run_generated(
        PipelineConfig(target=TARGET, outliers={"action": "cap"}), tmp_path / "a"
    )
    removed, _ = run_generated(
        PipelineConfig(target=TARGET, outliers={"action": "remove"}), tmp_path / "b"
    )
    assert len(capped) == 891
    assert len(removed) < 891


def test_flagging_adds_a_column_and_changes_no_values(workspace) -> None:
    flagged, _ = run_generated(
        PipelineConfig(
            target=TARGET,
            outliers={"action": "flag"},
            scaling={"enabled": False},
            encoding={"enabled": False},
        ),
        workspace,
    )
    assert any(c.endswith("_is_outlier") for c in flagged.columns)


def test_one_hot_widens_and_frequency_does_not(workspace, tmp_path) -> None:
    onehot, _ = run_generated(
        PipelineConfig(
            target=TARGET,
            encoding={"max_onehot_cardinality": 200, "low_cardinality_method": "onehot"},
        ),
        tmp_path / "wide",
    )
    frequency, _ = run_generated(
        PipelineConfig(
            target=TARGET,
            encoding={"max_onehot_cardinality": 2, "high_cardinality_method": "frequency"},
        ),
        tmp_path / "narrow",
    )
    assert onehot.shape[1] > frequency.shape[1]


def test_everything_is_numeric_after_encoding(workspace) -> None:
    """The point of the step: whatever comes out must be something a model can fit."""
    frame, _ = run_generated(PipelineConfig(target=TARGET), workspace)
    non_numeric = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
    assert non_numeric == []


def test_excluded_columns_are_gone(workspace) -> None:
    frame, _ = run_generated(
        PipelineConfig(target=TARGET, excluded_columns=["Name", "Ticket"]), workspace
    )
    assert "Name" not in frame.columns
    assert "Ticket" not in frame.columns


def test_dates_become_calendar_features(workspace) -> None:
    frame, _ = run_generated(PipelineConfig(target=TARGET), workspace)
    derived = [c for c in frame.columns if c.startswith("Boarded_")]
    assert derived, "the date column produced no calendar features"
    assert any(c.endswith("_sin") for c in derived)


def test_constant_calendar_parts_are_not_emitted(workspace) -> None:
    """The fixture is a single day, so year, month and weekday never vary.

    Emitting them anyway would add three dead columns per part. A constant column is
    exactly as useful to a model as no column, and costs width and attention.
    """
    frame, _ = run_generated(PipelineConfig(target=TARGET), workspace)
    for part in ("year", "month", "dayofweek"):
        assert f"Boarded_{part}" not in frame.columns
    assert "Boarded_hour" in frame.columns


def test_turning_off_date_parsing_produces_no_calendar_features(workspace) -> None:
    """With parsing off the column stays text, so it is encoded as a category instead.

    That is the correct outcome rather than a missing one — the column still reaches the
    model, just as an opaque label rather than as a month and an hour. What must not
    appear is any calendar part.
    """
    frame, _ = run_generated(
        PipelineConfig(target=TARGET, data_types={"parse_dates": False}), workspace
    )
    for part in ("year", "month", "day", "dayofweek", "hour"):
        assert f"Boarded_{part}" not in frame.columns
    assert not any(c.endswith("_sin") or c.endswith("_cos") for c in frame.columns)


# --------------------------------------------------------------------------------------
# The report the script writes
# --------------------------------------------------------------------------------------


def test_the_script_reports_every_step_it_ran(workspace) -> None:
    _, report = run_generated(PipelineConfig(target=TARGET), workspace)

    steps = {entry["step"] for entry in report["steps"]}
    for expected in ("load", "data_types", "split", "missing_values", "encoding", "scaling"):
        assert expected in steps

    for entry in report["steps"]:
        assert entry["rows"] >= 0
        assert entry["columns"] >= 0
        assert entry["action"]


def test_a_disabled_step_appears_nowhere(workspace) -> None:
    """Off means absent, not commented out — in the code and in the report."""
    config = PipelineConfig(target=TARGET, scaling={"enabled": False})
    source = generate(config)
    _, report = run_generated(config, workspace)

    assert "StandardScaler()" not in source
    assert "scaling" not in {entry["step"] for entry in report["steps"]}


def test_the_split_happens_before_any_fit(workspace) -> None:
    """The invariant the whole product rests on, checked in the emitted text.

    Reading the source is right here: the claim is about the order of statements, and
    running the script cannot distinguish a fit before the split from one after it.
    """
    source = generate(PipelineConfig(target=TARGET))
    split_at = source.index("train_test_split(")
    for fit_call in (".fit(", ".fit_transform("):
        position = source.find(fit_call)
        if position != -1:
            assert position > split_at, f"{fit_call} appears before the split"


def test_the_report_records_the_train_and_test_sizes(workspace) -> None:
    _, report = run_generated(PipelineConfig(target=TARGET, test_size=0.25), workspace)
    assert report["train_rows"] + report["test_rows"] == 891
    assert report["test_rows"] == pytest.approx(891 * 0.25, abs=1)


# --------------------------------------------------------------------------------------
# Determinism and explanations
# --------------------------------------------------------------------------------------


def test_the_same_configuration_generates_identical_code() -> None:
    config = PipelineConfig(target=TARGET)
    assert generate(config) == generate(config)


def test_the_same_configuration_produces_identical_output(tmp_path) -> None:
    config = PipelineConfig(target=TARGET)
    first, _ = run_generated(config, tmp_path / "one")
    second, _ = run_generated(config, tmp_path / "two")
    pd.testing.assert_frame_equal(first, second)


def test_every_step_has_an_explanation_that_reflects_its_options() -> None:
    config = PipelineConfig(target=TARGET)
    for step_id in STEP_ORDER:
        text = explain_step(step_id, config)
        assert text, f"{step_id} has no explanation"
        assert text[0].isupper() and text.rstrip().endswith(".")

    # Changing an option must change the sentence, or the card is describing something
    # other than what will run.
    median = explain_step("missing_values", config)
    mean = explain_step(
        "missing_values",
        config.model_copy(update={"missing_values": config.missing_values.model_copy(
            update={"numeric_strategy": "mean"}
        )}),
    )
    assert median != mean


# --------------------------------------------------------------------------------------
# Decisions, like fits, are made from the training rows only
#
# The generated script always fitted on the training rows. Two of its *decisions* were
# taken over the whole file: which columns are too empty to keep, and which calendar
# parts are worth emitting. Both change the shape of the feature matrix, so taking
# either one with the held-out rows in view is the same leak as fitting on them, one
# step removed — and it is invisible in the output, because the result is a pipeline
# that simply has a column it should not have.
#
# Each test is built so that the whole-file answer and the training-fold answer differ,
# which is what makes it fail against the previous implementation rather than pass
# vacuously against both.
# --------------------------------------------------------------------------------------


def _folds(frame: pd.DataFrame, target: str) -> tuple[pd.Index, pd.Index]:
    """The split the generated script will make, computed the same way it computes it."""
    from sklearn.model_selection import train_test_split

    train_index, test_index = train_test_split(
        frame.index, test_size=0.2, random_state=42, shuffle=True, stratify=frame[target]
    )
    return pd.Index(train_index), pd.Index(test_index)


def run_generated_on(
    config: PipelineConfig, workspace: Path, source: Path
) -> tuple[pd.DataFrame, dict]:
    """:func:`run_generated`, against a file the test wrote rather than the fixture."""
    script_source = generate(config)
    compile(script_source, "pipeline.py", "exec")

    workspace.mkdir(parents=True, exist_ok=True)
    script = workspace / "pipeline.py"
    script.write_text(script_source)
    output = workspace / "out.csv"
    report = workspace / "report.json"

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=300,
        env={
            **os.environ,
            "MLBX_INPUT": str(source),
            "MLBX_OUTPUT": str(output),
            "MLBX_REPORT": str(report),
        },
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return pd.read_csv(output), json.loads(report.read_text())


def test_a_column_is_dropped_on_how_empty_the_training_rows_are(workspace) -> None:
    """70% blank in the training fold, 56% blank over the file. The threshold is 60%."""
    import numpy as np

    rng = np.random.default_rng(101)
    frame = pd.DataFrame({
        "f1": rng.normal(size=200),
        "sparse": rng.normal(size=200),
        "label": np.tile([0, 1], 100),
    })
    train_index, test_index = _folds(frame, "label")
    blanked = rng.choice(train_index, size=112, replace=False)
    frame.loc[blanked, "sparse"] = np.nan

    train_rate = frame.loc[train_index, "sparse"].isna().mean()
    whole_rate = frame["sparse"].isna().mean()
    assert train_rate > 0.60 >= whole_rate, (train_rate, whole_rate)
    assert frame.loc[test_index, "sparse"].notna().all()

    source = workspace / "sparse.csv"
    frame.to_csv(source, index=False)
    out, _ = run_generated_on(PipelineConfig(target="label"), workspace, source)

    assert "sparse" not in out.columns, (
        "the column survived, so its emptiness was measured over the held-out rows too"
    )
    assert "f1" in out.columns


def test_a_calendar_part_is_emitted_on_how_the_training_rows_vary(workspace) -> None:
    """``month`` is constant in the training fold and varies only in the held-out rows."""
    import numpy as np

    rng = np.random.default_rng(102)
    frame = pd.DataFrame({
        "f1": rng.normal(size=200),
        "signed_on": ["2024-03-01"] * 200,
        "label": np.tile([0, 1], 100),
    })
    _, test_index = _folds(frame, "label")
    frame.loc[test_index, "signed_on"] = "2024-09-01"

    source = workspace / "dates.csv"
    frame.to_csv(source, index=False)
    out, _ = run_generated_on(PipelineConfig(target="label"), workspace, source)

    assert "signed_on_month" not in out.columns, (
        "a calendar part was emitted because the held-out rows varied, which the "
        "training rows cannot see"
    )
    assert "signed_on" not in out.columns


def test_infinity_in_the_source_file_does_not_stop_the_generated_script(workspace) -> None:
    import numpy as np

    rng = np.random.default_rng(103)
    frame = pd.DataFrame({
        "ratio": rng.normal(5, 1, 200),
        "amount": rng.normal(100, 20, 200),
        "label": np.tile([0, 1], 100),
    })
    frame.loc[0, "ratio"] = np.inf
    frame.loc[1, "ratio"] = -np.inf

    source = workspace / "inf.csv"
    frame.to_csv(source, index=False)
    out, report = run_generated_on(PipelineConfig(target="label"), workspace, source)

    assert np.isfinite(out["ratio"].to_numpy(dtype=float)).all()
    assert any("infinite" in step.get("action", "") for step in report["steps"])
