"""A performance budget for the 10 MB path, scaled to the machine it runs on.

Why this is not a list of wall-clock seconds
--------------------------------------------
A fixed "ingestion must finish in 2.5 seconds" is a test about the hardware, not about
the code. It passes on a developer's laptop, fails on a loaded CI runner, gets a larger
number bolted on, and after two rounds of that it no longer detects anything.

So each budget here is a multiple of a **calibration workload** measured in the same
process, moments before: a fixed pandas job whose cost tracks the operations the
pipeline actually performs — a median fill, a groupby, a sort, a dummy expansion. A
machine three times slower runs the calibration three times slower and gets three times
the budget. What the budget then measures is the code's cost *relative to the machine*,
which is the thing a regression changes and the hardware does not.

The reference measurements
--------------------------
Taken on darwin 25.6.0 (macOS 26.6.2), Apple silicon, Python 3.14.2, pandas 2.3.3,
scikit-learn 1.8.0, on 30 October 2026, against ``tests/fixtures/perf_10mb.csv``
(10.0 MB, 35,910 rows x 50 columns, binary target). Median of three runs:

====================================  ========  ==========
stage                                 measured  budget
====================================  ========  ==========
ingest — read, detect, repair            0.59s     2.5s
profile — every row, every column        0.33s     1.5s
health — six weighted dimensions         0.39s     1.5s
plan — propose, screens included         0.13s     1.0s
run — fit, transform, post-split checks  0.48s     2.0s
**total, ingest to run**                 1.92s     8.0s
====================================  ========  ==========

Calibration on the same machine: 0.029s (median of five, spread under 7%).

The budgets are roughly four times the measured cost. That is wide enough to absorb
normal variance and a cold filesystem cache, and narrow enough that a change making any
stage several times more expensive fails rather than being absorbed.

Limitations, stated rather than discovered later
------------------------------------------------
* **One file shape.** 36k rows x 50 columns. A file with many more columns, or one
  categorical column of very high cardinality, has a different profile, and nothing here
  would notice a regression specific to it.
* **Single process, warm.** No concurrency, and the fixture is read from the OS cache
  after the first test. First-read cost from cold storage is not measured.
* **The calibration is a proxy.** It tracks CPU and memory bandwidth for pandas work. It
  does not track disk speed, so a machine with a slow disk and a fast CPU gets a budget
  that is tighter than it should be for ingestion.
* **Not a service-level objective.** These are regression guards. They say the code has
  not become several times slower; they do not promise a user anything.

Set ``MLBENCHX_SKIP_PERF=1`` to skip the suite where timing is meaningless — a debugger,
a profiler, or a heavily contended shared runner.
"""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from app.plan import propose
from app.runner import execute
from app.services.health import assess
from app.services.ingest import read_upload
from app.services.profile import build_profile

pytestmark = [
    pytest.mark.perf,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("MLBENCHX_SKIP_PERF") == "1",
        reason="MLBENCHX_SKIP_PERF=1",
    ),
]

FIXTURE = Path(__file__).parent / "fixtures" / "perf_10mb.csv"

#: Median seconds for :func:`_calibration` on the reference machine above. The budgets
#: are scaled by how far this machine departs from it.
REFERENCE_CALIBRATION_SECONDS = 0.029

#: Stage -> budget in reference seconds. See the table in the module docstring.
BUDGETS: dict[str, float] = {
    "ingest": 2.5,
    "profile": 1.5,
    "health": 1.5,
    "plan": 1.0,
    "run": 2.0,
    "total": 8.0,
}


def _calibration() -> None:
    """A fixed pandas workload. Its cost is this machine's speed, in one number."""
    rng = np.random.default_rng(0)
    n = 200_000
    frame = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.integers(0, 1000, n),
        "c": rng.choice([f"k{i}" for i in range(50)], n),
    })
    frame.loc[rng.choice(n, n // 10, replace=False), "a"] = np.nan
    frame["a"] = frame["a"].fillna(frame["a"].median())
    frame.groupby("c")["a"].mean()
    frame.sort_values("b")
    pd.get_dummies(frame["c"])


def _median_seconds(work: Callable[[], object], repeats: int = 3) -> float:
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        work()
        timings.append(time.perf_counter() - started)
    return statistics.median(timings)


@pytest.fixture(scope="module")
def machine_factor() -> float:
    """How much slower this machine is than the reference. Never below 1.0.

    Floored deliberately: a faster machine does not get a *tighter* budget. The suite
    exists to catch a regression, and tightening the threshold on fast hardware would
    turn ordinary variance into a failure.
    """
    measured = _median_seconds(_calibration, repeats=5)
    factor = max(1.0, measured / REFERENCE_CALIBRATION_SECONDS)
    print(
        f"\n[perf] calibration {measured * 1000:.0f}ms "
        f"(reference {REFERENCE_CALIBRATION_SECONDS * 1000:.0f}ms) "
        f"-> budgets x{factor:.2f}"
    )
    return factor


@pytest.fixture(scope="module")
def large_frame():
    """The 10 MB fixture, read once for the whole module."""
    if not FIXTURE.exists():
        pytest.skip(
            "tests/fixtures/perf_10mb.csv is generated on demand and gitignored; "
            "run `python tests/fixtures/generate.py --large` to create it"
        )
    ingested = read_upload(FIXTURE.read_bytes(), "perf_10mb.csv")
    return ingested


def _assert_within(stage: str, seconds: float, factor: float) -> None:
    budget = BUDGETS[stage] * factor
    assert seconds <= budget, (
        f"{stage} took {seconds:.2f}s against a budget of {budget:.2f}s "
        f"(={BUDGETS[stage]:.2f}s x{factor:.2f} for this machine). "
        "Either the stage became materially more expensive, or the machine is slower "
        "in a way the calibration does not capture."
    )
    print(f"[perf] {stage:8} {seconds:5.2f}s / {budget:5.2f}s budget")


# --------------------------------------------------------------------------------------
# The budget, stage by stage
# --------------------------------------------------------------------------------------


def test_the_fixture_is_the_shape_the_budgets_were_measured_against(large_frame) -> None:
    """A budget for a file that has changed shape is a budget for something else."""
    assert FIXTURE.stat().st_size > 9 * 1024 * 1024
    assert large_frame.rows > 30_000
    assert large_frame.columns >= 40


def test_ingestion_stays_within_budget(machine_factor) -> None:
    raw = FIXTURE.read_bytes() if FIXTURE.exists() else pytest.skip("fixture missing")
    seconds = _median_seconds(lambda: read_upload(raw, "perf_10mb.csv"))
    _assert_within("ingest", seconds, machine_factor)


def test_profiling_stays_within_budget(large_frame, machine_factor) -> None:
    seconds = _median_seconds(lambda: build_profile(large_frame.frame))
    _assert_within("profile", seconds, machine_factor)


def test_health_scoring_stays_within_budget(large_frame, machine_factor) -> None:
    target = str(large_frame.frame.columns[-1])
    seconds = _median_seconds(lambda: assess(large_frame.frame, target))
    _assert_within("health", seconds, machine_factor)


def test_planning_stays_within_budget(large_frame, machine_factor) -> None:
    target = str(large_frame.frame.columns[-1])
    seconds = _median_seconds(
        lambda: propose(large_frame.frame, target, "binary classification"), repeats=2
    )
    _assert_within("plan", seconds, machine_factor)


def test_executing_a_plan_stays_within_budget(large_frame, machine_factor) -> None:
    target = str(large_frame.frame.columns[-1])
    plan = propose(large_frame.frame, target, "binary classification")
    seconds = _median_seconds(
        lambda: execute(large_frame.frame, plan, run_id="perf"), repeats=2
    )
    _assert_within("run", seconds, machine_factor)


def test_the_whole_path_stays_within_budget(machine_factor) -> None:
    """Ingest to run, once, as a user experiences it."""
    raw = FIXTURE.read_bytes() if FIXTURE.exists() else pytest.skip("fixture missing")

    started = time.perf_counter()
    ingested = read_upload(raw, "perf_10mb.csv")
    target = str(ingested.frame.columns[-1])
    build_profile(ingested.frame)
    assess(ingested.frame, target)
    plan = propose(ingested.frame, target, "binary classification")
    execute(ingested.frame, plan, run_id="perf-total")
    seconds = time.perf_counter() - started

    _assert_within("total", seconds, machine_factor)


# --------------------------------------------------------------------------------------
# Scaling — machine-independent, and the thing a budget alone misses
# --------------------------------------------------------------------------------------


def test_cost_grows_roughly_with_the_data_not_faster(large_frame) -> None:
    """Four times the rows must not cost sixteen times as much.

    A budget catches a constant-factor slowdown. It does not catch an accidental
    quadratic — on a 10 MB file that may still fit inside the budget while making a
    40 MB file impossible. This compares the same work at two sizes, so it needs no
    reference machine and no calibration: the ratio is the measurement.
    """
    frame = large_frame.frame
    quarter = frame.iloc[: len(frame) // 4]

    small = _median_seconds(lambda: build_profile(quarter))
    full = _median_seconds(lambda: build_profile(frame))

    # Linear would be ~4x. Allow 8x for fixed costs and measurement noise; quadratic
    # would be ~16x and fails.
    ratio = full / max(small, 1e-6)
    assert ratio < 8.0, (
        f"profiling 4x the rows cost {ratio:.1f}x the time, which is superlinear — "
        "something in the per-column work is scaling with the square of the rows"
    )
    print(f"[perf] profile scaling: 4x rows -> {ratio:.1f}x time")


def test_memory_stays_proportionate_to_the_file(large_frame) -> None:
    """The frame should not cost many multiples of the bytes it was read from.

    Not a hard ceiling — pandas object columns genuinely cost more than their CSV text —
    but an order-of-magnitude guard against a change that starts copying the frame per
    column.
    """
    on_disk = FIXTURE.stat().st_size
    in_memory = int(large_frame.frame.memory_usage(deep=True).sum())

    assert in_memory < on_disk * 12, (
        f"the frame holds {in_memory / 1e6:.0f} MB from a {on_disk / 1e6:.0f} MB file"
    )
    print(f"[perf] memory: {on_disk / 1e6:.0f} MB on disk -> {in_memory / 1e6:.0f} MB in frame")
