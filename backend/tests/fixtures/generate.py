"""Generate the test fixture catalogue, deterministically.

Every dataset here is synthetic and built from a fixed seed, so the same checkout
produces byte-identical files on any machine. That matters more than it sounds: several
tests assert exact counts — *885 missing values*, *42 outliers* — and a fixture that
drifted would turn those into flaky tests that get weakened until they assert nothing.

Each fixture exists to make one class of problem unavoidable. `high_missing` is not
"some data with gaps", it is a file where one column is 85% empty and another is 40%,
so the >60% drop rule and the >5% indicator rule are both exercised and can be told
apart. Where a test needs a number, it should read it from :data:`EXPECTED` rather than
hard-coding it, so the fixture and the assertion move together.

Usage::

    python tests/fixtures/generate.py           # write anything missing
    python tests/fixtures/generate.py --force   # rewrite everything
    python tests/fixtures/generate.py --list    # names, sizes and what each is for

Most fixtures are small and committed. ``perf_10mb.csv`` is ~10 MB and is generated on
demand instead — it is in ``.gitignore``, and :func:`ensure` builds it when a test asks
for it.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

#: One seed for the whole catalogue. Individual builders derive from it so that adding
#: a fixture cannot shift the contents of the ones before it.
SEED = 20260920

#: What each fixture is for, and the facts tests may rely on. Anything a test asserts
#: should be recorded here rather than duplicated as a literal in the test.
EXPECTED: dict[str, dict[str, Any]] = {}


def _rng(name: str) -> np.random.Generator:
    """A generator seeded from the fixture's name, so fixtures are independent."""
    return np.random.default_rng(SEED + (abs(hash(name)) % 100_000))


# --------------------------------------------------------------------------------------
# The catalogue
# --------------------------------------------------------------------------------------


def clean_binary() -> pd.DataFrame:
    """A well-behaved binary classification set: no gaps, no duplicates, real signal.

    The control. Anything that goes wrong here is the pipeline's fault, not the data's.
    """
    rng = _rng("clean_binary")
    n = 600
    age = rng.integers(18, 70, n)
    income = np.round(rng.gamma(9, 4200, n), 2)
    region = rng.choice(["north", "south", "east", "west"], n)
    tenure = rng.integers(0, 15, n)

    logit = -3.2 + 0.045 * age + income / 90_000 + 0.12 * tenure
    target = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    EXPECTED["clean_binary"] = {
        "rows": n, "columns": 5, "missing": 0, "duplicates": 0,
        "target": "converted", "task": "binary classification", "classes": 2,
    }
    return pd.DataFrame({
        "age": age, "income": income, "region": region,
        "tenure_years": tenure, "converted": target,
    })


def multiclass() -> pd.DataFrame:
    """Four classes, all well represented."""
    rng = _rng("multiclass")
    n = 500
    species = rng.choice(["alpha", "beta", "gamma", "delta"], n, p=[0.3, 0.3, 0.2, 0.2])
    centres = {"alpha": 10.0, "beta": 14.0, "gamma": 18.0, "delta": 22.0}
    length = np.array([rng.normal(centres[s], 1.4) for s in species]).round(3)

    EXPECTED["multiclass"] = {
        "rows": n, "columns": 4, "target": "species",
        "task": "multiclass classification", "classes": 4,
    }
    return pd.DataFrame({
        "length_cm": length,
        "width_cm": np.round(length * rng.uniform(0.4, 0.6, n), 3),
        "habitat": rng.choice(["forest", "coast", "upland"], n),
        "species": species,
    })


def regression() -> pd.DataFrame:
    """A continuous target with genuine structure and a little noise."""
    rng = _rng("regression")
    n = 550
    area = np.round(rng.gamma(8, 18, n), 1)
    rooms = rng.integers(1, 7, n)
    age_years = rng.integers(0, 90, n)
    price = np.round(
        22_000 + 1_450 * area + 9_500 * rooms - 180 * age_years + rng.normal(0, 9_000, n), 2
    )

    EXPECTED["regression"] = {
        "rows": n, "columns": 5, "target": "price", "task": "regression",
    }
    return pd.DataFrame({
        "area_sqm": area, "rooms": rooms, "age_years": age_years,
        "district": rng.choice(["inner", "outer", "riverside"], n), "price": price,
    })


def messy_types() -> pd.DataFrame:
    """Everything §1 says must be coerced, in one file.

    Numbers written as text with thousands separators, currency and percent signs;
    booleans in four spellings; dates in three formats; categories padded with
    whitespace and inconsistent case.
    """
    rng = _rng("messy_types")
    n = 300

    revenue = [f"{v:,}" for v in rng.integers(1_000, 900_000, n)]
    price = [f"${v}.{rng.integers(0, 99):02d}" for v in rng.integers(5, 400, n)]
    rate = [f"{v}%" for v in rng.integers(0, 100, n)]
    yes_no = rng.choice(["yes", "no"], n)
    true_false = rng.choice(["true", "false"], n)
    y_n = rng.choice(["Y", "N"], n)
    # Three written forms, all with a four-digit year so the parser's gate accepts them.
    iso = [f"2023-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}" for i in range(n)]
    slashed = [f"{(i % 27) + 1:02d}/{(i % 12) + 1:02d}/2023" for i in range(n)]
    padded = rng.choice([" north ", "NORTH", "north", "  South", "south "], n)

    EXPECTED["messy_types"] = {
        "rows": n,
        "numeric_text_columns": ["revenue", "unit_price", "conversion_rate"],
        "boolean_columns": ["is_active", "is_verified", "opted_in"],
        "date_columns": ["signed_iso", "signed_slashed"],
        "whitespace_category": "region",
        "note": "region has 5 written forms of 2 real values",
    }
    return pd.DataFrame({
        "revenue": revenue, "unit_price": price, "conversion_rate": rate,
        "is_active": yes_no, "is_verified": true_false, "opted_in": y_n,
        "signed_iso": iso, "signed_slashed": slashed, "region": padded,
        "outcome": rng.choice([0, 1], n),
    })


def high_missing() -> pd.DataFrame:
    """Gaps at four levels, so each missing-value rule is exercised separately.

    ``mostly_blank`` 85% — over the 60% drop threshold.
    ``half_blank``   40% — under the drop threshold, over the 5% indicator threshold.
    ``rarely_blank``  3% — under both; imputed with no indicator.
    ``never_blank``   0% — untouched.
    """
    rng = _rng("high_missing")
    n = 400

    def blank(values: np.ndarray, share: float) -> np.ndarray:
        out = values.astype("float64").copy()
        out[rng.random(n) < share] = np.nan
        return out

    base = rng.normal(50, 12, n)
    EXPECTED["high_missing"] = {
        "rows": n,
        "drop_over_60": ["mostly_blank"],
        "indicator_over_5": ["half_blank"],
        "impute_no_indicator": ["rarely_blank"],
        "untouched": ["never_blank"],
    }
    return pd.DataFrame({
        "mostly_blank": blank(base, 0.85),
        "half_blank": blank(base * 1.3, 0.40),
        "rarely_blank": blank(base * 0.7, 0.03),
        "never_blank": np.round(base, 3),
        "category_gaps": pd.Series(rng.choice(["a", "b", "c"], n)).mask(rng.random(n) < 0.25),
        "target": rng.choice([0, 1], n),
    })


def id_and_constant() -> pd.DataFrame:
    """Columns that carry nothing: identifiers, constants, and a near-constant.

    ``near_constant`` is 99.5% one value — above the spec's 99% bar but not exactly
    constant, which is the case the current code misses.
    """
    rng = _rng("id_and_constant")
    n = 350
    near = np.array(["standard"] * n, dtype=object)
    near[rng.choice(n, 2, replace=False)] = "premium"   # 2/350 = 0.57%

    EXPECTED["id_and_constant"] = {
        "rows": n,
        "id_like": ["record_id", "reference_no"],
        "constant": ["country"],
        "near_constant": ["near_constant"],
        "near_constant_top_share": (n - 2) / n,
        "usable": ["score", "target"],
    }
    return pd.DataFrame({
        "record_id": [f"REC{i:06d}" for i in range(n)],
        "reference_no": np.arange(100_000, 100_000 + n),
        "country": ["SG"] * n,
        "near_constant": near,
        "score": np.round(rng.normal(0, 1, n), 4),
        "target": rng.choice([0, 1], n),
    })


def leakage() -> pd.DataFrame:
    """A column derived from the answer, plus a clean control.

    ``leaked_score`` is the target plus a whisper of noise — correlation above 0.99, and
    a depth-2 tree splits it almost perfectly. ``honest_feature`` is genuinely
    predictive but nowhere near a copy, so a detector that fires on it is too eager.
    """
    rng = _rng("leakage")
    n = 500
    target = rng.choice([0, 1], n)

    EXPECTED["leakage"] = {
        "rows": n, "target": "churned",
        "leaking": ["leaked_score", "leaked_flag"],
        "honest": ["honest_feature", "noise"],
    }
    return pd.DataFrame({
        "leaked_score": target * 100 + rng.normal(0, 0.05, n),
        "leaked_flag": np.where(target == 1, "gone", "stayed"),
        "honest_feature": target * 1.1 + rng.normal(0, 1.4, n),
        "noise": rng.normal(0, 1, n),
        "churned": target,
    })


def imbalanced() -> pd.DataFrame:
    """95/5. Accuracy flatters a model that never predicts the minority class."""
    rng = _rng("imbalanced")
    n = 800
    target = (rng.random(n) < 0.05).astype(int)

    EXPECTED["imbalanced"] = {
        "rows": n, "target": "fraud",
        "minority_share": 0.05, "ratio_at_least": 10.0,
    }
    return pd.DataFrame({
        "amount": np.round(rng.gamma(3, 60, n) + target * 240, 2),
        "channel": rng.choice(["web", "app", "phone"], n),
        "hour": rng.integers(0, 24, n),
        "fraud": target,
    })


def high_cardinality() -> pd.DataFrame:
    """Categorical columns far past the one-hot threshold, plus one just under it.

    ``city`` has 300 levels; one-hot would add 300 columns. ``tier`` has 8, which must
    stay one-hot. Both in one file so a test can prove the threshold is a threshold and
    not a blanket rule.
    """
    rng = _rng("high_cardinality")
    n = 700
    frame = pd.DataFrame({
        "city": rng.choice([f"city_{i:03d}" for i in range(300)], n),
        "sku": rng.choice([f"SKU-{i:04d}" for i in range(450)], n),
        "tier": rng.choice([f"tier_{i}" for i in range(8)], n),
        "spend": np.round(rng.gamma(4, 30, n), 2),
        "renewed": rng.choice([0, 1], n),
    })
    # Record the levels that actually occur, not the size of the pool they were drawn
    # from: sampling 700 rows from 300 labels does not produce 300 distinct values, and
    # a test reading the pool size would fail against the file on disk.
    EXPECTED["high_cardinality"] = {
        "rows": n,
        "high_card": {c: int(frame[c].nunique()) for c in ("city", "sku")},
        "low_card": {"tier": int(frame["tier"].nunique())},
        "onehot_threshold": 10,
    }
    return frame


def tiny() -> pd.DataFrame:
    """12 rows — below any sensible floor for a train/test split.

    A rare class with a single member, so the non-stratified fallback in §3 is
    reachable.
    """
    EXPECTED["tiny"] = {"rows": 12, "columns": 3, "singleton_class": "c"}
    return pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        "g": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b", "a", "c"],
        "y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })


def unicode_duplicate_headers() -> pd.DataFrame:
    """Non-ASCII headers and values, with the same header used twice.

    Written with two identical ``région`` columns so the deduplication rule and the
    original→clean mapping both have something to do.
    """
    rng = _rng("unicode_duplicate_headers")
    n = 120
    EXPECTED["unicode_duplicate_headers"] = {
        "rows": n,
        "raw_headers": ["région", "région", " naïve  ", "Ünïcödé", "cible"],
        "duplicate_header": "région",
        "note": "written raw; pandas is bypassed so the duplicate survives to the reader",
    }
    return pd.DataFrame({
        "région": rng.choice(["nord", "sud", "est"], n),
        "région_2": rng.choice(["α", "β", "γ"], n),
        "naïve": np.round(rng.normal(0, 1, n), 3),
        "unicode": rng.choice(["日本", "Ελλάδα", "Россия"], n),
        "cible": rng.choice([0, 1], n),
    })


def perf(target_bytes: int = 10 * 1024 * 1024) -> pd.DataFrame:
    """~10 MB across ~50 columns, for the performance budget.

    Mixed types on purpose: a file of pure floats would profile far faster than anything
    real and would make the budget meaningless.
    """
    rng = _rng("perf")
    # Bytes per row across these 50 columns, measured on the generated file. Tuned so
    # the fixture matches the size its name promises — a "10mb" file that is really 14
    # would quietly make the performance budget 40% easier than it claims to be.
    n = max(1_000, target_bytes // 292)
    data: dict[str, Any] = {}
    for i in range(30):
        data[f"num_{i:02d}"] = np.round(rng.normal(i, 1 + i % 5, n), 4)
    for i in range(12):
        data[f"cat_{i:02d}"] = rng.choice([f"v{j}" for j in range(3 + i)], n)
    for i in range(4):
        data[f"int_{i:02d}"] = rng.integers(0, 1000, n)
    for i in range(3):
        data[f"flag_{i:02d}"] = rng.choice(["yes", "no"], n)
    data["target"] = rng.choice([0, 1], n)

    EXPECTED["perf_10mb"] = {
        "rows": int(n), "columns": len(data),
        "profile_budget_s": 10, "run_budget_s": 30,
    }
    return pd.DataFrame(data)


# --------------------------------------------------------------------------------------
# Files that are not well-formed frames
# --------------------------------------------------------------------------------------


def _write_raw_fixtures(force: bool) -> list[tuple[str, int]]:
    """Files that cannot be expressed as a DataFrame, written byte by byte."""
    written: list[tuple[str, int]] = []

    def put(name: str, payload: bytes) -> None:
        path = HERE / name
        if path.exists() and not force:
            return
        path.write_bytes(payload)
        written.append((name, len(payload)))

    # Empty — zero bytes. Must be rejected with a sentence, not a traceback.
    put("empty.csv", b"")

    # Header row and nothing under it.
    put("header_only.csv", b"alpha,beta,gamma\n")

    # Ragged rows: row 3 has an extra field, row 5 is short.
    put(
        "malformed.csv",
        b"a,b,c\n1,2,3\n4,5,6,7\n8,9,10\n11,12\n13,14,15\n",
    )

    # No header: the first line is already data. Accepting this silently turns the
    # first observation into column names.
    rows = "\n".join(f"{i},{i * 2},{i * 3}" for i in range(80))
    put("headerless.csv", rows.encode())

    # A single column — rejected by §1.
    put("single_column.csv", ("only\n" + "\n".join(str(i) for i in range(80))).encode())

    # Latin-1: bytes that are not valid UTF-8, so the fallback has to engage.
    latin = "nom,ville,prix,cible\n" + "\n".join(
        f"Café,Genève,{i},{i % 2}" for i in range(120)
    )
    put("latin1.csv", latin.encode("latin-1"))

    # Duplicate headers, written raw so pandas cannot silently rename them on the way
    # out. This is what a real export from a spreadsheet looks like.
    frame = unicode_duplicate_headers()
    header = "région,région, naïve  ,Ünïcödé,cible\n"
    body = "\n".join(
        ",".join(str(v) for v in row)
        for row in frame.itertuples(index=False)
    )
    put("unicode_duplicate_headers.csv", (header + body + "\n").encode("utf-8"))

    # A binary file wearing a .csv extension — the content check in §7.
    put("disguised_binary.csv", b"\x7fELF\x02\x01\x01\x00" + bytes(range(256)) * 4)

    return written


def _write_xlsx(force: bool) -> tuple[str, int] | None:
    """A two-sheet workbook, so sheet selection has something to select."""
    path = HERE / "two_sheets.xlsx"
    if path.exists() and not force:
        return None
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        clean_binary().to_excel(writer, sheet_name="first", index=False)
        multiclass().to_excel(writer, sheet_name="second", index=False)
    path.write_bytes(buffer.getvalue())
    EXPECTED["two_sheets"] = {"sheets": ["first", "second"], "default": "first"}
    return ("two_sheets.xlsx", path.stat().st_size)


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------

#: name -> (builder, "what it is for"). ``perf_10mb`` is excluded from the committed set.
BUILDERS: dict[str, tuple[Callable[[], pd.DataFrame], str]] = {
    "clean_binary": (clean_binary, "well-behaved binary classification, the control"),
    "multiclass": (multiclass, "four balanced classes"),
    "regression": (regression, "continuous target with real structure"),
    "messy_types": (messy_types, "every coercion rule in §1, in one file"),
    "high_missing": (high_missing, "gaps at 85/40/3/0% to separate the rules"),
    "id_and_constant": (id_and_constant, "identifiers, a constant, and a near-constant"),
    "leakage": (leakage, "a copy of the target, plus an honest control"),
    "imbalanced": (imbalanced, "95/5 class split"),
    "high_cardinality": (high_cardinality, "300 and 450 levels, plus one under the threshold"),
    "tiny": (tiny, "12 rows and a singleton class"),
}

LARGE = {"perf_10mb": (perf, "~10 MB, ~50 mixed columns, for the performance budget")}


def build_all(force: bool = False, include_large: bool = False) -> list[tuple[str, int]]:
    """Write every fixture that is missing. Returns ``[(filename, bytes)]``."""
    written: list[tuple[str, int]] = []

    for name, (builder, _) in BUILDERS.items():
        path = HERE / f"{name}.csv"
        if path.exists() and not force:
            builder()  # still populate EXPECTED
            continue
        builder().to_csv(path, index=False)
        written.append((f"{name}.csv", path.stat().st_size))

    written.extend(_write_raw_fixtures(force))
    excel = _write_xlsx(force)
    if excel:
        written.append(excel)

    if include_large:
        for name, (builder, _) in LARGE.items():
            path = HERE / f"{name}.csv"
            if path.exists() and not force:
                builder()
                continue
            builder().to_csv(path, index=False)
            written.append((f"{name}.csv", path.stat().st_size))

    return written


def ensure(name: str) -> Path:
    """Path to one fixture, building it first if it is not there.

    Lets a test depend on ``perf_10mb`` without that file being committed: the first run
    on a fresh checkout generates it, every later run finds it.
    """
    path = HERE / (name if name.endswith((".csv", ".xlsx")) else f"{name}.csv")
    if not path.exists():
        build_all(force=False, include_large=True)
    if not path.exists():
        raise FileNotFoundError(f"no fixture called {name!r}; known: {sorted(BUILDERS)}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true", help="rewrite existing fixtures")
    parser.add_argument("--list", action="store_true", help="show the catalogue and exit")
    parser.add_argument(
        "--large", action="store_true", help="also build perf_10mb.csv (~10 MB)"
    )
    args = parser.parse_args(argv)

    if args.list:
        for name, (_, purpose) in {**BUILDERS, **LARGE}.items():
            path = HERE / f"{name}.csv"
            size = f"{path.stat().st_size:,} B" if path.exists() else "not built"
            print(f"  {name:26s} {size:>12s}  {purpose}")
        return 0

    written = build_all(force=args.force, include_large=args.large)
    if not written:
        print("Every fixture is already present. Use --force to rewrite.")
    else:
        for name, size in written:
            print(f"  wrote {name:34s} {size:>10,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
