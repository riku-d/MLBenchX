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
    DATE_PARSE_RATE,
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


#: Leading bytes that identify a file regardless of what it was named. Checking these
#: is the difference between "the extension says CSV" and "this is CSV": an extension
#: is a claim by whoever uploaded the file, and the parser is the wrong place to find
#: out it was false.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"PK\x03\x04", "zip"),        # xlsx, and every other zip container
    (b"PAR1", "parquet"),
    (b"\x7fELF", "executable"),
    (b"MZ", "executable"),
    (b"\xca\xfe\xba\xbe", "executable"),
    (b"\xcf\xfa\xed\xfe", "executable"),
    (b"%PDF", "pdf"),
    (b"\x89PNG", "image"),
    (b"\xff\xd8\xff", "image"),
    (b"GIF8", "image"),
    (b"\x1f\x8b", "gzip"),
    (b"SQLite format 3", "database"),
)

#: Formats we can actually read, and what a file of that kind should look like.
_EXPECTED_MAGIC = {
    "spreadsheet": "zip",
    "parquet": "parquet",
}


def _sniff(raw: bytes) -> str:
    """What this file actually is, from its leading bytes."""
    for signature, kind in _MAGIC:
        if raw.startswith(signature):
            return kind
    # Text is identified by exclusion rather than by a signature: a CSV has none. A NUL
    # byte in the first chunk is the reliable tell that it is not text at all.
    if b"\x00" in raw[:8192]:
        return "binary"
    return "text"


def _check_content(raw: bytes, filename: str) -> None:
    """Refuse a file whose contents contradict its name.

    An extension check alone accepts anything renamed to ``.csv``. This is not much of
    a security boundary on its own — the parser would reject most of it eventually —
    but "that is a PDF, not a spreadsheet" is a far better answer than a pandas
    traceback, and it stops the parser being the first thing to touch hostile input.
    """
    lowered = filename.lower()
    actual = _sniff(raw)

    if lowered.endswith(_SPREADSHEET_SUFFIXES):
        expected = _EXPECTED_MAGIC["spreadsheet"]
        if actual != expected:
            raise IngestError(
                f"That file is named like a spreadsheet but its contents are not one "
                f"({_describe(actual)}). Open it and save a copy as .xlsx or .csv."
            )
        return

    if lowered.endswith(_PARQUET_SUFFIXES):
        if actual != "parquet":
            raise IngestError(
                f"That file is named like Parquet but its contents are not "
                f"({_describe(actual)})."
            )
        return

    if lowered.endswith(_TEXT_SUFFIXES + _JSON_SUFFIXES):
        if actual != "text":
            raise IngestError(
                f"That file is named like a text table but it is {_describe(actual)}. "
                "Upload a CSV, TSV, Excel, JSON or Parquet file."
            )
        return

    # An extension we do not claim to read at all. Say that, rather than commenting on
    # the contents: "we cannot read .png files" is the useful answer, and "that .png is
    # an image" is merely true.
    raise IngestError(
        "We can read CSV, TSV, Excel, JSON and Parquet files. That one has a different "
        "extension — open it and save a copy as CSV."
    )


def _describe(kind: str) -> str:
    return {
        "zip": "a zip archive",
        "parquet": "a Parquet file",
        "executable": "an executable",
        "pdf": "a PDF",
        "image": "an image",
        "gzip": "a gzip archive",
        "database": "a SQLite database",
        "binary": "binary data",
        "text": "text",
    }.get(kind, kind)


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
    #: Columns recognised as true/false answers.
    boolean_columns: list[str] = field(default_factory=list)
    #: ``{original heading: cleaned heading}``. Every message names a column, and a user
    #: who uploaded " Total Revenue " needs to see the name they typed.
    column_mapping: dict[str, str | list[str]] = field(default_factory=dict)
    #: Which sheet was read, and which were available. Empty for non-spreadsheets.
    sheet_name: str | None = None
    available_sheets: list[str] = field(default_factory=list)
    #: Values in a date column that could not be parsed and became blanks.
    unparsed_dates: dict[str, int] = field(default_factory=dict)


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


def _coerce_datetime(df: pd.DataFrame, unparsed: dict[str, int] | None = None) -> list[str]:
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
    unparsed = {} if unparsed is None else unparsed
    for column in df.columns:
        series = df[column]
        if series.dtype != object:
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue

        as_text = non_null.astype("string")
        shaped = as_text.str.match(_DATE_SHAPED)
        if float(shaped.mean()) < DATE_PARSE_RATE:
            continue

        parsed = pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=False)
        # The spec's rule is a *rate*, not unanimity: a column where 95% of values are
        # dates and the rest are "N/A" or a stray note is a date column with some bad
        # rows, and refusing to parse it because of those rows leaves the whole thing as
        # text. The unparsed values become blanks, which the missing-value step then
        # handles and reports — so nothing is lost silently.
        parsed_rate = float(parsed.notna().sum()) / max(int(series.notna().sum()), 1)
        if parsed_rate < DATE_PARSE_RATE:
            continue

        lost = int(series.notna().sum()) - int(parsed.notna().sum())
        df[column] = parsed
        converted.append(str(column))
        if lost:
            unparsed[str(column)] = lost
    return converted


#: Spellings people use for true and false. Compared case-insensitively after stripping.
#: Deliberately does not include ``1``/``0``: those are already numbers, and a column of
#: them is as likely to be a count or a code as a flag.
_TRUE_TOKENS = frozenset({"true", "yes", "y", "t"})
_FALSE_TOKENS = frozenset({"false", "no", "n", "f"})


def _coerce_boolean(df: pd.DataFrame) -> list[str]:
    """Convert text columns that hold a yes/no answer.

    Only when *every* non-blank value is one of the known spellings, and both a true and
    a false appear. Requiring both matters: a column that is entirely ``"Y"`` is a
    constant, and turning it into ``True`` would disguise that as a usable boolean.
    """
    converted: list[str] = []
    for column in df.columns:
        series = df[column]
        if series.dtype != object:
            continue
        values = series.dropna()
        if values.empty:
            continue

        lowered = values.astype("string").str.strip().str.lower()
        if not lowered.isin(_TRUE_TOKENS | _FALSE_TOKENS).all():
            continue
        if not (lowered.isin(_TRUE_TOKENS).any() and lowered.isin(_FALSE_TOKENS).any()):
            continue

        mapped = (
            series.astype("string").str.strip().str.lower().map(
                lambda v: True if v in _TRUE_TOKENS else (False if v in _FALSE_TOKENS else pd.NA)
            )
        )
        df[column] = mapped.astype("boolean")
        converted.append(str(column))
    return converted


def _tidy_headers(df: pd.DataFrame) -> tuple[int, dict[str, str | list[str]]]:
    """Strip whitespace from column names, name the unnamed, and disambiguate repeats.

    Returns the number of names changed and the ``{original: clean}`` mapping.

    The mapping is not bookkeeping. Every message this app shows names a column, and a
    user who uploaded ``" Total Revenue "`` needs to see that name, not ``Total
    Revenue`` — otherwise they go looking in their spreadsheet for a column that is not
    there. It is also the only record of which of two identically-named columns became
    ``region`` and which became ``region_1``, which is unrecoverable once it is lost.
    """
    renamed = 0
    mapping: dict[str, str] = {}
    new_names: list[str] = []

    # Prefer the headings as the file wrote them, when the reader preserved them.
    written = df.attrs.get("raw_headers")
    source = [str(h) for h in written] if written and len(written) == df.shape[1] else [
        str(c) for c in df.columns
    ]

    for position, name in enumerate(source):
        original = str(name)
        text = original.strip()
        if not text or text.lower().startswith("unnamed:"):
            text = f"column_{position + 1}"
        if text != original:
            renamed += 1
        new_names.append(text)

    # Disambiguate duplicates so downstream column lookups stay unambiguous.
    seen: dict[str, int] = {}
    final: list[str] = []
    for original, text in zip(source, new_names):
        if text in seen:
            seen[text] += 1
            text = f"{text}_{seen[text]}"
            renamed += 1
        else:
            seen[text] = 0
        final.append(text)
        # A heading written twice maps to both cleaned names, so the record is a list.
        mapping.setdefault(str(original), [])
        mapping[str(original)].append(text)

    df.columns = final
    # Collapse the common case — one original, one clean name — back to a plain string,
    # so callers are not forced to unwrap a list for every ordinary column.
    collapsed = {k: (v[0] if len(v) == 1 else v) for k, v in mapping.items()}
    return renamed, collapsed


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


def _looks_headerless(columns: list[str]) -> bool:
    """Whether the first row of this file was data rather than names.

    A CSV has no way to say "there is no header", so pandas takes the first row as one
    regardless. Accepting that silently costs the user an observation *and* gives every
    column a name that is really a value — and the damage is invisible, because the file
    loads and looks plausible.

    The test is that *every* header parses as a number. Requiring all of them keeps a
    real file safe: a spreadsheet with a column honestly called ``2024`` alongside
    ``region`` and ``revenue`` is not headerless, and a single numeric name is common.
    A row that is entirely numbers is not a set of names anybody wrote.
    """
    if len(columns) < 2:
        return False
    for name in columns:
        text = str(name).strip()
        # Pandas suffixes repeated headers, so "3" twice arrives as "3" and "3.1".
        text = text.rsplit(".", 1)[0] if text.count(".") == 1 and text.replace(".", "").isdigit() else text
        try:
            float(text)
        except ValueError:
            return False
    return True


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


def list_sheets(raw: bytes) -> list[str]:
    """Sheet names in a workbook, in the order they appear in it."""
    try:
        with pd.ExcelFile(io.BytesIO(raw)) as workbook:
            return [str(name) for name in workbook.sheet_names]
    except Exception as exc:  # noqa: BLE001
        raise IngestError(
            "We couldn't open that spreadsheet. If it was exported from another "
            "program, try saving it again as CSV."
        ) from exc


def _read_excel(raw: bytes, sheet_name: str | None) -> tuple[pd.DataFrame, str, str]:
    """One sheet of a workbook — the named one, or the first.

    Naming a sheet that is not there is refused with the list of ones that are, rather
    than silently falling back to the first. A user who asked for "2024 Actuals" and got
    "Template" would not necessarily notice, and every number they then saw would be
    from the wrong sheet.
    """
    sheets = list_sheets(raw)
    if not sheets:
        raise IngestError("That workbook has no sheets in it.")

    chosen = sheets[0] if sheet_name is None else sheet_name
    if chosen not in sheets:
        available = ", ".join(sheets[:8])
        raise IngestError(
            f"That workbook has no sheet called {chosen!r}. It contains: {available}."
        )

    try:
        frame = pd.read_excel(io.BytesIO(raw), sheet_name=chosen)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as a sentence
        raise IngestError(
            f"We couldn't read the sheet {chosen!r}. Try saving it again as CSV."
        ) from exc
    return frame, "n/a (spreadsheet)", f"sheet: {chosen}"


def _raw_headers(text: str, delimiter: str) -> list[str]:
    """The header line exactly as the file wrote it.

    Needed because pandas resolves duplicate headings while reading — a file with two
    ``région`` columns arrives as ``région`` and ``région.1``, and by then the fact that
    both were written the same is gone. A mapping built from what pandas handed us would
    record ``région.1 -> région.1``, which is true of nothing the user ever saw.
    """
    first = text.splitlines()[0] if text.splitlines() else ""
    try:
        return next(csv.reader([first], delimiter=delimiter))
    except (csv.Error, StopIteration):
        return [cell for cell in first.split(delimiter)]


def _read_frame(
    raw: bytes, filename: str, sheet_name: str | None = None
) -> tuple[pd.DataFrame, str, str]:
    """Parse the bytes into a frame, returning it with the detected encoding/delimiter."""
    lowered = filename.lower()

    if lowered.endswith(_PARQUET_SUFFIXES):
        return _read_parquet(raw), "n/a (parquet)", "n/a (parquet)"

    if lowered.endswith(_JSON_SUFFIXES):
        return _read_json(raw, filename), "utf-8", "n/a (json)"

    if lowered.endswith(_SPREADSHEET_SUFFIXES):
        return _read_excel(raw, sheet_name)

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

    # Stash the headings as written, before pandas' de-duplication is the only record.
    originals = _raw_headers(text, delimiter)
    if len(originals) == frame.shape[1]:
        frame.attrs["raw_headers"] = originals
    return frame, encoding, delimiter


def read_upload(
    raw: bytes, filename: str, sheet_name: str | None = None
) -> IngestResult:
    """Read an uploaded file into a clean frame, recording every repair.

    Args:
        raw: the uploaded bytes.
        filename: picks a parser, and is echoed back to the user.
        sheet_name: which sheet of a workbook to read. ``None`` means the first.

    Raises:
        IngestError: with a message written for the user, not for a log file.
    """
    if not raw:
        raise IngestError("That file is empty. Upload a spreadsheet with some rows in it.")

    # What the file *is*, before the parser is allowed near it.
    _check_content(raw, filename)

    frame, encoding, delimiter = _read_frame(raw, filename, sheet_name)
    cleaning: list[str] = []

    if _looks_headerless(list(frame.columns)):
        raise IngestError(
            "That file has no header row — the first line is already data, so every "
            "column would be named after a value and that first observation would be "
            "lost. Add a row of column names and upload it again."
        )

    renamed, column_mapping = _tidy_headers(frame)
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

    unparsed_dates: dict[str, int] = {}
    dates = _coerce_datetime(frame, unparsed_dates)
    if dates:
        shown = ", ".join(dates[:4]) + (" and others" if len(dates) > 4 else "")
        cleaning.append(
            f"Recognised {_plural(len(dates), 'column')} as dates rather than text: "
            f"{shown}"
        )
    if unparsed_dates:
        total = sum(unparsed_dates.values())
        named = ", ".join(f"{c} ({n})" for c, n in list(unparsed_dates.items())[:3])
        cleaning.append(
            f"Left {_plural(total, 'value')} blank in the date columns because they "
            f"could not be read as a date: {named}"
        )

    booleans = _coerce_boolean(frame)
    if booleans:
        shown = ", ".join(booleans[:4]) + (" and others" if len(booleans) > 4 else "")
        cleaning.append(
            f"Recognised {_plural(len(booleans), 'column')} as true/false answers: {shown}"
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
        boolean_columns=booleans,
        column_mapping=column_mapping,
        sheet_name=sheet_name,
        available_sheets=(
            list_sheets(raw) if filename.lower().endswith(_SPREADSHEET_SUFFIXES) else []
        ),
        unparsed_dates=unparsed_dates,
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
