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
import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from charset_normalizer import from_bytes

from app.config import (
    HEALTH_WEIGHTS,
    MAX_MISSING_RATE,
    MIN_ROWS,
    MISSING_MENTION_RATE,
    MISSING_TOKENS,
    SMALL_SAMPLE_PENALTY,
    SMALL_SAMPLE_ROWS,
)

__all__ = [
    "IngestError",
    "IngestResult",
    "read_upload",
    "loader_source",
]

_SPREADSHEET_SUFFIXES = (".xlsx", ".xls", ".xlsm")
_TEXT_SUFFIXES = (".csv", ".tsv", ".txt")
_JSON_SUFFIXES = (".json", ".jsonl", ".ndjson")
_PARQUET_SUFFIXES = (".parquet", ".pq")

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
    #: Itemised arithmetic behind ``quality_score`` — one entry per deduction, with the
    #: amount, the weight applied and the points it cost.
    quality_breakdown: list[dict] = field(default_factory=list)
    #: Columns that were stored as text and converted to numbers. Recorded so the
    #: exported script can repeat the same conversion.
    numeric_columns: list[str] = field(default_factory=list)
    #: Columns that were stored as text and recognised as dates, for the same reason.
    datetime_columns: list[str] = field(default_factory=list)


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


#: A value has to look like a date before we try to parse it as one, and the test is a
#: **four-digit year** at one end or the other. Without that anchor pandas reads a
#: version number like ``1.2.3`` as the 3rd of February 2001 and a bare ``2019`` as a
#: timestamp — a column of software versions or scores would become dates, silently.
#: Requiring the year rules those out while still accepting every common written form:
#: ``2024-01-05``, ``05/01/2024``, ``5.1.2024``, each with an optional time.
_DATE_SHAPED = re.compile(
    r"^\s*(?:"
    r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"      # year first: 2024-01-05
    r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{4}"     # year last:  05/01/2024
    r")"
    r"(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?"  # optional time
    r"\s*$"
)


def _coerce_datetime(df: pd.DataFrame) -> list[str]:
    """Convert text columns that are really dates.

    Without this the calendar features never run on an uploaded file: a CSV has no
    dtypes, so ``2012-04-10`` arrives as an ordinary string and profiles as a category.
    A date column would then be one-hot encoded into a column per distinct timestamp,
    which is both useless and enormous.

    The bar is deliberately high — every non-blank value must match a real date shape
    *and* parse — because the failure mode in the other direction is worse. A column of
    version numbers or scores turned into 1970 timestamps is data destroyed silently,
    where a date left as a category is merely a missed opportunity that the dropped- or
    encoded-column list still reports.
    """
    converted: list[str] = []
    for column in df.columns:
        series = df[column]
        if series.dtype != object:
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue

        as_text = non_null.astype("string")
        if not as_text.str.match(_DATE_SHAPED).all():
            continue

        parsed = pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=False)
        # Every value that had something in it must have parsed, or this is not a date
        # column and we would be turning real values into blanks.
        if int(parsed.isna().sum()) != int(series.isna().sum()):
            continue

        df[column] = parsed
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


def _score_quality(
    df: pd.DataFrame, duplicates: int, leakage_flags: int = 0
) -> tuple[int, list[QualityIssue], list[dict]]:
    """A 0-100 health score, the reasons it is not 100, and the arithmetic behind it.

    The formula is the weight table in :data:`app.config.HEALTH_WEIGHTS`, applied as::

        score = 100 - Σ min(amount × weight, cap)

    where *amount* is the size of each defect — the percentage of blank cells, the
    number of empty columns — so a bigger problem costs more, and each cap stops one
    defect from sinking the score alone. The third return value is that sum, itemised:
    every deduction with its amount, its weight and the points it cost. A score the user
    can take apart is worth more than a precise one they have to trust.
    """
    issues: list[QualityIssue] = []
    breakdown: list[dict] = []
    n_rows, n_columns = df.shape
    total_cells = max(n_rows * n_columns, 1)

    def deduct(key: str, amount: float, label: str) -> float:
        """Apply one weighted, capped deduction and record how it was arrived at."""
        weight, cap = HEALTH_WEIGHTS[key]
        points = min(amount * weight, cap)
        if points > 0:
            breakdown.append({
                "factor": key,
                "label": label,
                "amount": round(amount, 4),
                "weight": weight,
                "cap": cap,
                "points_lost": round(points, 2),
            })
        return points

    score = 100.0
    missing_rate = float(df.isna().sum().sum()) / total_cells
    score -= deduct("missing_cells", missing_rate * 100, "blank cells")

    if duplicates:
        share = duplicates / max(n_rows, 1)
        score -= deduct("duplicate_rows", share * 100, "duplicate rows")
        issues.append(QualityIssue(
            "warning",
            f"{_plural(duplicates, 'row is', 'rows are')} an exact duplicate of another row"
            if duplicates == 1
            else f"{duplicates:,} rows are exact duplicates of another row",
        ))

    empty_columns = [c for c in df.columns if df[c].isna().all()]
    if empty_columns:
        score -= deduct("empty_columns", len(empty_columns), "completely empty columns")
        issues.append(QualityIssue(
            "warning",
            f"{_plural(len(empty_columns), 'column is', 'columns are')} completely empty: "
            + ", ".join(empty_columns[:4]),
        ))

    very_sparse = [
        c for c in df.columns if MAX_MISSING_RATE < df[c].isna().mean() < 1.0
    ]
    if very_sparse:
        score -= deduct("sparse_columns", len(very_sparse), "mostly empty columns")
        issues.append(QualityIssue(
            "warning",
            f"{_plural(len(very_sparse), 'column is', 'columns are')} more than "
            f"{MAX_MISSING_RATE * 100:.0f}% empty: " + ", ".join(very_sparse[:4]),
        ))

    if leakage_flags:
        score -= deduct("leakage_flags", leakage_flags, "columns that look like the answer")

    if missing_rate > MISSING_MENTION_RATE:
        issues.append(QualityIssue(
            "info",
            f"{missing_rate * 100:.1f}% of all cells are blank — these get filled in later",
        ))

    if n_rows < SMALL_SAMPLE_ROWS:
        score -= SMALL_SAMPLE_PENALTY
        breakdown.append({
            "factor": "small_sample",
            "label": f"fewer than {SMALL_SAMPLE_ROWS} rows",
            "amount": n_rows,
            "weight": 0.0,
            "cap": SMALL_SAMPLE_PENALTY,
            "points_lost": SMALL_SAMPLE_PENALTY,
        })
        issues.append(QualityIssue(
            "info",
            f"{n_rows} rows is a small sample, so results will be rough",
        ))

    return max(min(int(round(score)), 100), 0), issues, breakdown


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def _read_json(raw: bytes, filename: str) -> pd.DataFrame:
    """Read a JSON dataset, accepting the three shapes people actually export.

    A list of objects is the common one. Line-delimited JSON is what streaming tools
    emit. A dict of columns is what ``DataFrame.to_json()`` produces by default, and a
    user who round-tripped their own data through pandas should not be told their file
    is unreadable.

    Nested objects are left as they are rather than flattened: a column of dicts
    profiles as free text and is reported as such, which is a truthful description of a
    file that is not really tabular.
    """
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        raise IngestError("That file is empty. Upload one with some rows in it.")

    lowered = filename.lower()
    if lowered.endswith((".jsonl", ".ndjson")) or (text[0] == "{" and "\n{" in text):
        try:
            return pd.read_json(io.StringIO(text), lines=True)
        except ValueError as exc:
            raise IngestError(
                "We couldn't read that file as line-delimited JSON. Check that each "
                "line is a complete JSON object."
            ) from exc

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise IngestError(
            f"That file isn't valid JSON — {exc.msg} at line {exc.lineno}."
        ) from exc

    if isinstance(parsed, list):
        frame = pd.json_normalize(parsed, max_level=1)
    elif isinstance(parsed, dict):
        # Either {column: {index: value}} from to_json(), or a wrapper around the rows.
        rows = next(
            (v for v in parsed.values() if isinstance(v, list) and v and isinstance(v[0], dict)),
            None,
        )
        frame = pd.json_normalize(rows, max_level=1) if rows is not None else pd.DataFrame(parsed)
    else:
        raise IngestError(
            "That JSON file holds a single value rather than a table. We need a list of "
            "records, or one object per line."
        )

    if frame.empty:
        raise IngestError("That JSON file has no rows in it.")
    return frame


def _read_parquet(raw: bytes) -> pd.DataFrame:
    """Read a Parquet file. Columnar and typed, so nothing has to be inferred."""
    try:
        return pd.read_parquet(io.BytesIO(raw))
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise IngestError(
            "Reading Parquet needs the pyarrow package, which is not installed here. "
            "Export your data as CSV instead."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as a sentence
        raise IngestError(
            "We couldn't open that Parquet file. It may be truncated, or written with "
            "a feature this reader does not support — try exporting it as CSV."
        ) from exc


def _read_frame(raw: bytes, filename: str) -> tuple[pd.DataFrame, str, str]:
    """Parse the bytes into a frame, returning it with the detected encoding/delimiter."""
    lowered = filename.lower()

    if lowered.endswith(_PARQUET_SUFFIXES):
        return _read_parquet(raw), "n/a (parquet)", "n/a (parquet)"

    if lowered.endswith(_JSON_SUFFIXES):
        return _read_json(raw, filename), "utf-8", "n/a (json)"

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
            "We can read CSV, TSV, Excel, JSON and Parquet files. That one has a "
            "different extension — open it and save a copy as CSV."
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

    dates = _coerce_datetime(frame)
    if dates:
        shown = ", ".join(dates[:4]) + (" and others" if len(dates) > 4 else "")
        cleaning.append(
            f"Recognised {_plural(len(dates), 'column')} as dates rather than text: "
            f"{shown}"
        )

    if len(frame) < MIN_ROWS:
        raise IngestError(
            f"That file has {_plural(len(frame), 'row')}. We need at least {MIN_ROWS} to set some "
            "aside for honest testing."
        )

    duplicates = int(frame.duplicated().sum())
    missing_cells = int(frame.isna().sum().sum())
    score, issues, breakdown = _score_quality(frame, duplicates)

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
        quality_breakdown=breakdown,
        numeric_columns=converted,
        datetime_columns=dates,
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
TEXT_DATE_COLUMNS = {result.datetime_columns!r}
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

    # Columns stored as text that are really dates.
    for column in TEXT_DATE_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce", format="mixed")

    return df

'''
    return block, "load_frame()"
