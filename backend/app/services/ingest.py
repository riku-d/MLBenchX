"""Read a user's spreadsheet and make it safe to work with.

Everything here happens before :mod:`app.services.glassbox` sees the data. The job is to
turn whatever the user uploaded — wrong encoding, semicolons, ``N/A`` typed into cells,
numbers stored as text — into a plain ``DataFrame``, and to keep a note of every repair
so the user can be told about it.

Two rules shape this module:

* **Every repair is recorded.** Nothing is silently fixed. Each entry in
  :attr:`IngestResult.cleaning` is a sentence the user reads.
* **Nothing here learns from the data.** These are format-level repairs applied to the
  whole file. Anything that learns a value (a median, a category list) belongs in
  ``glassbox`` and happens after the train/test split.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from charset_normalizer import from_bytes

__all__ = [
    "MIN_ROWS",
    "MISSING_TOKENS",
    "IngestError",
    "IngestResult",
    "read_upload",
    "loader_source",
]

#: Below this, a train/test split cannot say anything meaningful.
MIN_ROWS = 50

#: Strings users type when they mean "no value". Compared case-insensitively after
#: stripping surrounding whitespace.
MISSING_TOKENS: frozenset[str] = frozenset({
    "", "-", "--", "?", "n/a", "na", "n.a.", "nan", "none", "null", "nil",
    "unknown", "not available", "not applicable", "missing", "#n/a", "#na",
    "#value!", "#div/0!", "tbd", "no data", ".",
})

_SPREADSHEET_SUFFIXES = (".xlsx", ".xls", ".xlsm")
_TEXT_SUFFIXES = (".csv", ".tsv", ".txt")

#: Currency symbols, thousands separators and stray percent signs that stop pandas from
#: seeing a number. Stripped only when doing so makes the whole column numeric.
_NUMERIC_NOISE = re.compile(r"[,\s ₹$€£%]")


def _plural(n: int, singular: str, plural: str | None = None) -> str:
    """`1 column` / `4 columns` — "(s)" is not something a user should have to read."""
    return f"{n:,} {singular if n == 1 else (plural or singular + 's')}"


class IngestError(Exception):
    """A problem the user can fix, phrased as a sentence to show them directly."""


@dataclass
class QualityIssue:
    severity: str  # "warning" | "info"
    text: str


@dataclass
class IngestResult:
    """A clean frame plus the story of how it got that way."""

    frame: pd.DataFrame
    name: str
    encoding: str
    delimiter: str
    rows: int
    columns: int
    cleaning: list[str] = field(default_factory=list)
    issues: list[QualityIssue] = field(default_factory=list)
    missing_cells: int = 0
    duplicate_rows: int = 0
    quality_score: int = 100
    #: Columns that were stored as text and converted to numbers. Recorded so the
    #: exported script can repeat the same conversion.
    numeric_columns: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# Format detection
# --------------------------------------------------------------------------------------


def _detect_encoding(raw: bytes) -> str:
    """Best-guess text encoding, defaulting to utf-8 for anything ambiguous."""
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass
    match = from_bytes(raw[:200_000]).best()
    if match is not None and match.encoding:
        return match.encoding
    return "latin-1"  # decodes any byte sequence, so we never hard-fail on encoding


def _detect_delimiter(sample: str) -> str:
    """Sniff the column separator, falling back to whichever candidate is most frequent."""
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        counts = {candidate: first_line.count(candidate) for candidate in [",", ";", "\t", "|"]}
        best = max(counts, key=lambda key: counts[key])
        return best if counts[best] > 0 else ","


# --------------------------------------------------------------------------------------
# Cleaning passes
# --------------------------------------------------------------------------------------


def _blank_placeholders(df: pd.DataFrame) -> int:
    """Turn text that means "no value" into real blanks. Returns how many were found."""
    replaced = 0
    for column in df.columns:
        series = df[column]
        if series.dtype != object:
            continue
        as_text = series.astype("string")
        normalised = as_text.str.strip().str.lower()
        hits = normalised.isin(MISSING_TOKENS) & series.notna()
        count = int(hits.sum())
        if count:
            df.loc[hits, column] = np.nan
            replaced += count
    return replaced


def _coerce_numeric(df: pd.DataFrame) -> list[str]:
    """Convert text columns that are really numbers.

    Only applied when *every* non-blank value converts cleanly. A column where 3 of 500
    values are text is more likely a genuine category than a broken number, and silently
    turning those 3 into blanks would destroy data.
    """
    converted: list[str] = []
    for column in df.columns:
        series = df[column]
        if series.dtype != object:
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue

        stripped = non_null.astype("string").str.strip().str.replace(_NUMERIC_NOISE, "", regex=True)
        # A bare "-" or "" after stripping is not a number.
        if (stripped.str.len() == 0).any():
            continue
        numeric = pd.to_numeric(stripped, errors="coerce")
        if numeric.isna().any():
            continue
        # Guard against turning a 2-value code column into meaningless integers.
        if numeric.nunique() <= 1:
            continue

        rebuilt = pd.to_numeric(
            series.astype("string").str.strip().str.replace(_NUMERIC_NOISE, "", regex=True),
            errors="coerce",
        )
        df[column] = rebuilt
        converted.append(str(column))
    return converted


def _tidy_headers(df: pd.DataFrame) -> int:
    """Strip whitespace from column names and give unnamed columns a stable label."""
    renamed = 0
    new_names: list[str] = []
    for position, name in enumerate(df.columns):
        text = str(name).strip()
        if not text or text.lower().startswith("unnamed:"):
            text = f"column_{position + 1}"
        if text != str(name):
            renamed += 1
        new_names.append(text)

    # Disambiguate duplicates so downstream column lookups stay unambiguous.
    seen: dict[str, int] = {}
    final: list[str] = []
    for text in new_names:
        if text in seen:
            seen[text] += 1
            text = f"{text}_{seen[text]}"
            renamed += 1
        else:
            seen[text] = 0
        final.append(text)

    df.columns = final
    return renamed


# --------------------------------------------------------------------------------------
# Quality
# --------------------------------------------------------------------------------------


def _score_quality(df: pd.DataFrame, duplicates: int) -> tuple[int, list[QualityIssue]]:
    """A 0-100 readable score, with the reasons it is not 100.

    Deliberately simple arithmetic rather than a weighted model: the user is told exactly
    what cost them points, and a score they can reason about is worth more here than a
    precise one they cannot.
    """
    issues: list[QualityIssue] = []
    n_rows, n_columns = df.shape
    total_cells = max(n_rows * n_columns, 1)

    missing_rate = float(df.isna().sum().sum()) / total_cells
    score = 100.0
    score -= min(missing_rate * 100 * 1.5, 35)

    if duplicates:
        share = duplicates / max(n_rows, 1)
        score -= min(share * 100, 15)
        issues.append(QualityIssue(
            "warning",
            f"{_plural(duplicates, 'row is', 'rows are')} an exact duplicate of another row"
            if duplicates == 1
            else f"{duplicates:,} rows are exact duplicates of another row",
        ))

    empty_columns = [c for c in df.columns if df[c].isna().all()]
    if empty_columns:
        score -= min(len(empty_columns) * 4, 12)
        issues.append(QualityIssue(
            "warning",
            f"{_plural(len(empty_columns), 'column is', 'columns are')} completely empty: "
            + ", ".join(empty_columns[:4]),
        ))

    very_sparse = [c for c in df.columns if 0.6 < df[c].isna().mean() < 1.0]
    if very_sparse:
        score -= min(len(very_sparse) * 3, 12)
        issues.append(QualityIssue(
            "warning",
            f"{_plural(len(very_sparse), 'column is', 'columns are')} more than 60% "
            "empty: " + ", ".join(very_sparse[:4]),
        ))

    if missing_rate > 0.02:
        issues.append(QualityIssue(
            "info",
            f"{missing_rate * 100:.1f}% of all cells are blank — these get filled in later",
        ))

    if n_rows < 200:
        score -= 8
        issues.append(QualityIssue(
            "info",
            f"{n_rows} rows is a small sample, so results will be rough",
        ))

    return max(int(round(score)), 0), issues


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def _read_frame(raw: bytes, filename: str) -> tuple[pd.DataFrame, str, str]:
    """Parse the bytes into a frame, returning it with the detected encoding/delimiter."""
    lowered = filename.lower()

    if lowered.endswith(_SPREADSHEET_SUFFIXES):
        try:
            frame = pd.read_excel(io.BytesIO(raw))
        except Exception as exc:  # noqa: BLE001 - surfaced to the user as a sentence
            raise IngestError(
                "We couldn't open that spreadsheet. If it was exported from another "
                "program, try saving it again as CSV."
            ) from exc
        return frame, "n/a (spreadsheet)", "n/a (spreadsheet)"

    if not lowered.endswith(_TEXT_SUFFIXES):
        raise IngestError(
            "We can read CSV, TSV and Excel files. That one has a different extension — "
            "open it and save a copy as CSV."
        )

    encoding = _detect_encoding(raw)
    try:
        text = raw.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        encoding = "latin-1"
        text = raw.decode(encoding)

    if not text.strip():
        raise IngestError("That file is empty. Upload a spreadsheet with some rows in it.")

    delimiter = "\t" if lowered.endswith(".tsv") else _detect_delimiter(text[:64_000])
    try:
        frame = pd.read_csv(io.StringIO(text), sep=delimiter, skip_blank_lines=True)
    except Exception as exc:  # noqa: BLE001
        raise IngestError(
            "We couldn't make sense of the rows in that file. Check that every row has "
            "the same number of columns."
        ) from exc
    return frame, encoding, delimiter


def read_upload(raw: bytes, filename: str) -> IngestResult:
    """Read an uploaded file into a clean frame, recording every repair.

    Args:
        raw: the uploaded bytes.
        filename: used only to pick a parser and to echo back to the user.

    Raises:
        IngestError: with a message written for the user, not for a log file.
    """
    if not raw:
        raise IngestError("That file is empty. Upload a spreadsheet with some rows in it.")

    frame, encoding, delimiter = _read_frame(raw, filename)
    cleaning: list[str] = []

    renamed = _tidy_headers(frame)
    if renamed:
        cleaning.append(
            f"Tidied up {_plural(renamed, 'column heading')} with stray spaces or no name"
        )

    before_rows = len(frame)
    frame = frame.dropna(axis=0, how="all").reset_index(drop=True)
    blank_rows = before_rows - len(frame)
    if blank_rows:
        cleaning.append(f"Skipped {_plural(blank_rows, 'completely blank row')}")

    if frame.empty or frame.shape[1] == 0:
        raise IngestError(
            "That file has headings but no data underneath them. Check the export and "
            "try again."
        )

    if frame.shape[1] < 2:
        raise IngestError(
            "That file only has one column. We need the thing you want to predict plus "
            "at least one column to predict it from — check the separator is a comma or "
            "a semicolon."
        )

    placeholders = _blank_placeholders(frame)
    if placeholders:
        cleaning.append(
            f"Recognised {placeholders:,} placeholder values such as 'N/A' and '-' as "
            "genuinely missing"
        )

    converted = _coerce_numeric(frame)
    if converted:
        shown = ", ".join(converted[:4]) + (" and others" if len(converted) > 4 else "")
        cleaning.append(
            f"Converted {_plural(len(converted), 'column')} stored as text into "
            f"numbers: {shown}"
        )

    if len(frame) < MIN_ROWS:
        raise IngestError(
            f"That file has {_plural(len(frame), 'row')}. We need at least {MIN_ROWS} to set some "
            "aside for honest testing."
        )

    duplicates = int(frame.duplicated().sum())
    missing_cells = int(frame.isna().sum().sum())
    score, issues = _score_quality(frame, duplicates)

    if not cleaning:
        cleaning.append("Your file was already tidy — nothing needed repairing")

    return IngestResult(
        frame=frame,
        name=filename,
        encoding=encoding,
        delimiter=delimiter,
        rows=len(frame),
        columns=frame.shape[1],
        cleaning=cleaning,
        issues=issues,
        missing_cells=missing_cells,
        duplicate_rows=duplicates,
        quality_score=score,
        numeric_columns=converted,
    )


# --------------------------------------------------------------------------------------
# Reproducing this read in exported code
# --------------------------------------------------------------------------------------


def loader_source(result: IngestResult) -> tuple[str, str]:
    """Python source that re-reads the user's file exactly as this module did.

    The exported script has to reproduce the *cleaned* frame, not the raw one — a
    semicolon-delimited export full of ``N/A`` strings would otherwise parse into a
    single column and fail on the first line. Returns ``(block, call)``: a chunk of
    source to paste into the script, and the expression that calls it.
    """
    is_spreadsheet = result.delimiter.startswith("n/a")
    tokens = sorted(MISSING_TOKENS)

    if is_spreadsheet:
        read_line = "    df = pd.read_excel(CSV_PATH)"
    else:
        read_line = (
            f"    df = pd.read_csv(CSV_PATH, sep={result.delimiter!r}, "
            f"encoding={result.encoding!r})"
        )

    block = f'''

# --- Reading the file exactly as the app did -----------------------------------------
# These are format repairs, not anything learned from the data: they apply to every row
# and happen before the split.
MISSING_TOKENS = {tokens!r}
TEXT_NUMBER_COLUMNS = {result.numeric_columns!r}
NUMERIC_NOISE = r"[,\\s ₹$€£%]"


def load_frame() -> pd.DataFrame:
    """Read the spreadsheet and repair it the way the app did."""
{read_line}
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=0, how="all").reset_index(drop=True)

    # Text that means "no value" becomes a real blank.
    for column in df.columns:
        if df[column].dtype == object:
            normalised = df[column].astype("string").str.strip().str.lower()
            df.loc[normalised.isin(MISSING_TOKENS), column] = np.nan

    # Columns stored as text that are really numbers.
    for column in TEXT_NUMBER_COLUMNS:
        if column in df.columns:
            cleaned = (
                df[column].astype("string").str.strip().str.replace(NUMERIC_NOISE, "", regex=True)
            )
            df[column] = pd.to_numeric(cleaned, errors="coerce")

    return df

'''
    return block, "load_frame()"
