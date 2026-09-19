"""Every threshold the app applies, in one place.

Nothing here is discovered from data or tuned by a search. These are editorial
decisions — the point at which we say a column is too empty to use, too crowded to
one-hot, too unique to be a measurement — and they are collected here so that the plan,
the explanations shown to the user, the exported script and the tests all quote the same
number. A threshold that appears twice will eventually disagree with itself.

Each constant carries the reasoning for its value, because the value on its own is not
reviewable: ``0.6`` means nothing, "more than 60% missing and you are modelling your own
imputation" can be argued with.

Overriding
----------
Every constant can be overridden with an environment variable of the same name prefixed
``MLBENCHX_`` — ``MLBENCHX_MAX_UPLOAD_BYTES=104857600``. Overrides are read once at
import. This exists so an operator can raise a limit without editing code; the defaults
below are what the app is designed around.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "MAX_UPLOAD_BYTES",
    "MIN_ROWS",
    "UPLOAD_CHUNK_BYTES",
    "MISSING_TOKENS",
    "ONEHOT_MAX_CARDINALITY",
    "MAX_MISSING_RATE",
    "TEXT_UNIQUE_RATE",
    "TEXT_MIN_UNIQUE",
    "IDENTIFIER_UNIQUE_RATE",
    "NAMED_IDENTIFIER_UNIQUE_RATE",
    "LEAKAGE_CORRELATION",
    "HIGH_CORRELATION",
    "MIN_ROWS_FOR_CORRELATION",
    "OUTLIER_IQR_MULTIPLIER",
    "IMBALANCE_RATIO",
    "MAX_TARGET_CLASSES",
    "MAX_TARGET_CARDINALITY",
    "TEST_SIZE",
    "SEED",
    "DEFAULT_SCALER",
    "DEFAULT_HIGH_CARD_ENCODER",
    "TARGET_ENCODER_CV",
    "HEALTH_WEIGHTS",
    "DEFAULT_PROJECT_ID",
    "DATABASE_URL",
    "STORAGE_ROOT",
    "MAX_RUNS_IN_MEMORY",
    "EXECUTION_TIMEOUT_SECONDS",
    "EXECUTION_CPU_SECONDS",
    "EXECUTION_MEMORY_BYTES",
    "EXECUTION_MAX_OUTPUT_BYTES",
    "PREVIEW_PAGE_SIZE",
    "MAX_HISTORY_ENTRIES",
    "STRIP_WIDTH",
    "MAX_COLUMNS_IN_EXPLANATION",
]


def _int(name: str, default: int) -> int:
    raw = os.environ.get(f"MLBENCHX_{name}")
    return int(raw) if raw else default


def _float(name: str, default: float) -> float:
    raw = os.environ.get(f"MLBENCHX_{name}")
    return float(raw) if raw else default


def _str(name: str, default: str) -> str:
    return os.environ.get(f"MLBENCHX_{name}") or default


# --------------------------------------------------------------------------------------
# Uploads
# --------------------------------------------------------------------------------------

#: Largest upload accepted, in bytes. Big enough for any spreadsheet a person assembled
#: by hand, small enough that a mis-dragged folder cannot exhaust the machine.
MAX_UPLOAD_BYTES = _int("MAX_UPLOAD_BYTES", 50 * 1024 * 1024)

#: Read size when streaming an upload. Only affects how quickly an oversized file is
#: rejected, not what is accepted.
UPLOAD_CHUNK_BYTES = _int("UPLOAD_CHUNK_BYTES", 1024 * 1024)

#: Below this many rows a train/test split cannot say anything meaningful: a 20% holdout
#: of 40 rows is 8 rows, and any score computed on 8 rows is noise.
MIN_ROWS = _int("MIN_ROWS", 50)

#: Strings people type when they mean "no value". Compared case-insensitively after
#: stripping surrounding whitespace.
MISSING_TOKENS: frozenset[str] = frozenset({
    "", "-", "--", "?", "n/a", "na", "n.a.", "nan", "none", "null", "nil",
    "unknown", "not available", "not applicable", "missing", "#n/a", "#na",
    "#value!", "#div/0!", "tbd", "no data", ".",
})


# --------------------------------------------------------------------------------------
# Column routing
# --------------------------------------------------------------------------------------

#: A categorical column with at most this many distinct values is one-hot encoded.
#: Above it, one column per value costs more width than the signal is worth, so the
#: column is encoded to a single number instead.
ONEHOT_MAX_CARDINALITY = _int("ONEHOT_MAX_CARDINALITY", 10)

#: Columns missing more than this share of their values are dropped. Past roughly this
#: point an imputed column is mostly a record of our own fill value rather than the
#: user's data, and the model learns the imputation.
MAX_MISSING_RATE = _float("MAX_MISSING_RATE", 0.60)

#: A column whose distinct values exceed this share of the rows, *and* which has more
#: than ``TEXT_MIN_UNIQUE`` of them, is free text or an identifier rather than a
#: category. Both conditions are required: 8 distinct values in 10 rows is not free text.
TEXT_UNIQUE_RATE = _float("TEXT_UNIQUE_RATE", 0.5)
TEXT_MIN_UNIQUE = _int("TEXT_MIN_UNIQUE", 50)

#: A column this close to entirely distinct identifies the row, whatever its dtype. Set
#: high on purpose: a genuine measurement like income can be 60% distinct.
IDENTIFIER_UNIQUE_RATE = _float("IDENTIFIER_UNIQUE_RATE", 0.98)

#: A weaker bar applies when the column is *named* like a key — ``student_id``,
#: ``ref_no``. The name is evidence, so less uniqueness is needed to convict.
NAMED_IDENTIFIER_UNIQUE_RATE = _float("NAMED_IDENTIFIER_UNIQUE_RATE", 0.90)


# --------------------------------------------------------------------------------------
# Profiling signals
# --------------------------------------------------------------------------------------

#: Correlation with the target above which a feature is flagged as probable leakage.
#: Deliberately just below 1.0: an exact copy of the target is rare, a rounded or
#: rescaled copy is common, and both should be caught.
LEAKAGE_CORRELATION = _float("LEAKAGE_CORRELATION", 0.99)

#: Correlation between two features above which they are reported as redundant.
HIGH_CORRELATION = _float("HIGH_CORRELATION", 0.90)

#: Correlation is not reported below this many overlapping non-blank rows, where a
#: coefficient is dominated by chance.
MIN_ROWS_FOR_CORRELATION = _int("MIN_ROWS_FOR_CORRELATION", 30)

#: Tukey's multiplier. A value more than this many interquartile ranges outside the
#: quartiles is reported as an outlier. 1.5 is the convention behind every box plot;
#: it is reported, never acted on.
OUTLIER_IQR_MULTIPLIER = _float("OUTLIER_IQR_MULTIPLIER", 1.5)

#: Class imbalance is reported when the largest class is at least this many times the
#: smallest. At 3:1 accuracy starts to flatter a model that always guesses the majority.
IMBALANCE_RATIO = _float("IMBALANCE_RATIO", 3.0)

#: A numeric target with more distinct values than this is a regression rather than a
#: many-class classification.
MAX_TARGET_CLASSES = _int("MAX_TARGET_CLASSES", 20)

#: A non-numeric target with more distinct values than this is a label or free-text
#: field, not an outcome, and is refused with an explanation.
MAX_TARGET_CARDINALITY = _int("MAX_TARGET_CARDINALITY", 50)


# --------------------------------------------------------------------------------------
# Preprocessing defaults
# --------------------------------------------------------------------------------------

#: Share of rows held out. 20% is the default everywhere for good reason: enough to
#: measure with, cheap enough to give up.
TEST_SIZE = _float("TEST_SIZE", 0.2)

#: The only source of randomness in the app. Fixed so the same file always produces the
#: same split, and so the exported script reproduces what the user was shown.
SEED = _int("SEED", 42)

#: ``"standard"`` | ``"minmax"`` | ``"robust"``. Standard is the default because it is
#: the one every downstream model assumes.
DEFAULT_SCALER = _str("DEFAULT_SCALER", "standard")

#: ``"frequency"`` | ``"target"``. Frequency is the default because it never reads the
#: target and so cannot leak under any circumstances; target encoding is offered because
#: the spec asks for it and sklearn's implementation cross-fits internally.
DEFAULT_HIGH_CARD_ENCODER = _str("DEFAULT_HIGH_CARD_ENCODER", "frequency")

#: Inner folds used by ``TargetEncoder`` to cross-fit its training-set output. This is
#: the mechanism that stops a row's own target contributing to its own encoding.
TARGET_ENCODER_CV = _int("TARGET_ENCODER_CV", 5)


# --------------------------------------------------------------------------------------
# Health score
# --------------------------------------------------------------------------------------

#: The health score starts at 100 and loses points for each defect below. Every entry is
#: ``(points per unit of the defect, maximum deduction)``, so no single problem can sink
#: the score on its own and the total is bounded and explainable.
#:
#: The formula, in full:
#:
#:     score = 100
#:       − min(missing_cell_fraction × 100 × 1.5, 35)     blank cells
#:       − min(duplicate_row_fraction × 100 × 1.0, 15)    exact duplicate rows
#:       − min(n_empty_columns × 4, 12)                   columns with nothing in them
#:       − min(n_sparse_columns × 3, 12)                  columns over MAX_MISSING_RATE
#:       − min(n_leakage_flags × 10, 20)                  features that look like the answer
#:       − 8 if rows < SMALL_SAMPLE_ROWS                  too small to trust
#:
#: Weights are ordered by how much each defect actually costs a model. Blank cells top
#: the list because they are the most common and the most quietly damaging; a small
#: sample is a flat penalty because it is a fact about the file rather than a matter of
#: degree. The result is clamped to 0..100.
HEALTH_WEIGHTS: dict[str, tuple[float, float]] = {
    "missing_cells": (1.5, 35.0),
    "duplicate_rows": (1.0, 15.0),
    "empty_columns": (4.0, 12.0),
    "sparse_columns": (3.0, 12.0),
    "leakage_flags": (10.0, 20.0),
}

#: Flat penalty applied to files below ``SMALL_SAMPLE_ROWS`` rows.
SMALL_SAMPLE_ROWS = _int("SMALL_SAMPLE_ROWS", 200)
SMALL_SAMPLE_PENALTY = _float("SMALL_SAMPLE_PENALTY", 8.0)

#: Blank cells below this share are not worth a line in the issues list.
MISSING_MENTION_RATE = _float("MISSING_MENTION_RATE", 0.02)


# --------------------------------------------------------------------------------------
# Running user code
# --------------------------------------------------------------------------------------

#: Wall-clock ceiling for one pipeline run. Generous next to the ~2s a real pipeline
#: takes on the file sizes this accepts, so only a genuinely stuck script hits it.
EXECUTION_TIMEOUT_SECONDS = _int("EXECUTION_TIMEOUT_SECONDS", 120)

#: CPU seconds. Lower than the wall clock on purpose: a busy loop should be caught by
#: the kernel rather than waited out.
EXECUTION_CPU_SECONDS = _int("EXECUTION_CPU_SECONDS", 90)

#: Address space for the child. Enough for pandas plus a one-hot expansion of a
#: reasonable dataset, small enough that a runaway allocation dies instead of the
#: machine swapping.
EXECUTION_MEMORY_BYTES = _int("EXECUTION_MEMORY_BYTES", 2 * 1024 * 1024 * 1024)

#: Largest file the sandboxed script may write, so a loop appending to the output
#: cannot fill the disk.
EXECUTION_MAX_OUTPUT_BYTES = _int("EXECUTION_MAX_OUTPUT_BYTES", 512 * 1024 * 1024)

#: Rows sent to the browser in one dataset preview page.
PREVIEW_PAGE_SIZE = _int("PREVIEW_PAGE_SIZE", 50)

#: Pipeline states kept for the change-history panel.
MAX_HISTORY_ENTRIES = _int("MAX_HISTORY_ENTRIES", 50)


# --------------------------------------------------------------------------------------
# Storage and persistence
# --------------------------------------------------------------------------------------

_BACKEND_ROOT = Path(__file__).resolve().parent.parent

#: Phase 1 has no accounts, but every table carries a project so that adding them later
#: is a migration rather than a redesign. Everything lands in this one.
DEFAULT_PROJECT_ID = _str("DEFAULT_PROJECT_ID", "default")

#: SQLite lives next to the code. One file, no server, trivially inspectable — the right
#: database for an app whose data is a handful of rows per upload.
DATABASE_URL = _str("DATABASE_URL", f"sqlite:///{_BACKEND_ROOT / 'mlbenchx.db'}")

#: Uploaded files and fitted pipelines are written under here by ``LocalStorage``.
STORAGE_ROOT = Path(_str("STORAGE_ROOT", str(_BACKEND_ROOT / "storage")))

#: How many analysed runs to keep hot in memory. The database is the source of truth;
#: this only avoids re-reading and re-fitting for the run the user is looking at.
MAX_RUNS_IN_MEMORY = _int("MAX_RUNS_IN_MEMORY", 20)


# --------------------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------------------

#: Values in a before/after strip. Three is enough to show a change in context and few
#: enough to read at a glance.
STRIP_WIDTH = _int("STRIP_WIDTH", 3)

#: Explanations name at most this many columns inline before switching to a summary.
#: Every value is always present in the machine-readable details block.
MAX_COLUMNS_IN_EXPLANATION = _int("MAX_COLUMNS_IN_EXPLANATION", 6)
