"""Execute a :class:`~app.plan.PreprocessingPlan` and record everything it did.

The planner decides; this runs those decisions and nothing else. The separation is what
makes the plan meaningful: if the executor second-guessed the plan — filling a column
the plan said to drop, choosing a different encoder because the data looked odd — then
what the user reviewed and approved would not be what happened, and the review would be
theatre.

So the rule here is strict: **every column's fate comes from its** ``ColumnPlan``. The
executor's only freedom is *how* to carry out an action, never *whether* to.

Ordering follows §4, and the order is load-bearing:

1. Stateless repairs and the split (both already done by the time a plan exists, and
   reproduced here from the plan's seed so the folds are identical).
2. Rare-category collapsing, imputation and encoding — all fitted on the training fold.
3. Scaling, last, because it needs everything already numeric.

The log
-------
Every action appends a record in §5's shape. Two fields there are easy to get wrong and
worth stating: ``before`` and ``after`` describe the *data*, not the configuration, so
they are measured rather than asserted; and ``overridden`` is carried from the plan, so
a reader can see which decisions were the app's and which were the user's.

:func:`verify_log_invariant` then checks §5's guarantee — that every input column has a
stated fate — and raises if it does not. It is a real check rather than a comment
because the invariant is the one thing a glass-box tool cannot be wrong about.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)
from sklearn.utils.validation import check_is_fitted

from app.config import OUTLIER_IQR_MULTIPLIER, TARGET_ENCODER_CV
from app.plan import ColumnPlan, PreprocessingPlan, prepare
from app.services.glassbox import (
    DatetimeFeatures,
    FrequencyEncoder,
    choose_stratification,
)

__all__ = ["RunResult", "execute", "verify_log_invariant", "output_hash", "build_schema"]


# --------------------------------------------------------------------------------------
# Transformers the plan needs and sklearn does not provide
# --------------------------------------------------------------------------------------


class RareCategoryGrouper(TransformerMixin, BaseEstimator):
    """Replace values below a frequency threshold with ``"Other"``.

    Fitted on the training fold, so the set of rare values is decided without reference
    to the held-out rows — which matters more here than it first appears. A category
    that is rare in training but common in the test fold would otherwise be encoded as
    itself in one and as ``Other`` in the other.

    Values never seen during ``fit`` also become ``Other``, which is the honest handling:
    the model has no basis for treating an unseen category as anything but unfamiliar.
    """

    def __init__(self, rate: float = 0.01, keep: dict[str, list[str]] | None = None):
        self.rate = rate
        self.keep = keep

    def fit(self, X: Any, y: Any = None) -> "RareCategoryGrouper":
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(list(frame.columns), dtype=object)
        self.n_features_in_ = frame.shape[1]
        self.keep_ = {}
        for column in frame.columns:
            if self.keep and str(column) in self.keep:
                self.keep_[str(column)] = set(self.keep[str(column)])
                continue
            shares = frame[column].value_counts(normalize=True, dropna=True)
            self.keep_[str(column)] = {str(v) for v, s in shares.items() if s >= self.rate}
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        check_is_fitted(self, "keep_")
        frame = (X if isinstance(X, pd.DataFrame) else pd.DataFrame(
            X, columns=list(self.feature_names_in_)
        )).copy()
        for column in frame.columns:
            allowed = self.keep_.get(str(column), set())
            values = frame[column].astype(object)
            frame[column] = values.where(
                values.isna() | values.astype(str).isin(allowed), "Other"
            )
        return frame

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "keep_")
        names = list(self.feature_names_in_) if input_features is None else list(input_features)
        return np.asarray([str(n) for n in names], dtype=object)


class IQRClipper(TransformerMixin, BaseEstimator):
    """Bound each column to Tukey's fences, learned from the training fold.

    Only used when the plan asks for it. The default is to report outliers and change
    nothing, because an unusually large real value and a mistyped one are
    indistinguishable from here.
    """

    def __init__(self, multiplier: float = OUTLIER_IQR_MULTIPLIER):
        self.multiplier = multiplier

    def fit(self, X: Any, y: Any = None) -> "IQRClipper":
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(list(frame.columns), dtype=object)
        self.n_features_in_ = frame.shape[1]
        self.bounds_: dict[str, tuple[float, float]] = {}
        for column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if values.empty:
                continue
            q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
            iqr = q3 - q1
            if iqr <= 0:
                continue
            self.bounds_[str(column)] = (
                q1 - self.multiplier * iqr, q3 + self.multiplier * iqr
            )
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        check_is_fitted(self, "bounds_")
        frame = (X if isinstance(X, pd.DataFrame) else pd.DataFrame(
            X, columns=list(self.feature_names_in_)
        )).copy()
        for column, (low, high) in self.bounds_.items():
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").clip(low, high)
        return frame

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "bounds_")
        names = list(self.feature_names_in_) if input_features is None else list(input_features)
        return np.asarray([str(n) for n in names], dtype=object)


class CategoricalCast(TransformerMixin, BaseEstimator):
    """Present categoricals as plain objects.

    ``SimpleImputer`` refuses a ``bool`` column outright and pandas' nullable dtypes
    reach sklearn as masked arrays it does not expect. Without this a file with a real
    True/False column fails the whole run on a dtype error that says nothing about the
    user's data.
    """

    def fit(self, X: Any, y: Any = None) -> "CategoricalCast":
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(list(frame.columns), dtype=object)
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        frame = (X if isinstance(X, pd.DataFrame) else pd.DataFrame(
            X, columns=list(self.feature_names_in_)
        )).copy()
        for column in frame.columns:
            frame[column] = frame[column].astype(object).where(frame[column].notna(), np.nan)
        return frame

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        names = list(self.feature_names_in_) if input_features is None else list(input_features)
        return np.asarray([str(n) for n in names], dtype=object)


# --------------------------------------------------------------------------------------
# The result
# --------------------------------------------------------------------------------------


@dataclass
class RunResult:
    """Everything one execution produced."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    pipeline: ColumnTransformer
    plan: PreprocessingPlan
    log: list[dict[str, Any]]
    schema: dict[str, Any]
    warnings: list[dict[str, Any]] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0
    #: Whether the split kept each class's share on both sides. Recorded here rather
    #: than recovered by reading the log back: the log is prose plus a params bag, and
    #: a caller picking a boolean out of it by matching on a step name is one rename
    #: away from silently reporting the wrong thing.
    stratified: bool = False

    @property
    def output_hash(self) -> str:
        return output_hash(self.X_train, self.X_test)


def output_hash(X_train: pd.DataFrame, X_test: pd.DataFrame) -> str:
    """A stable digest of the prepared data.

    §6 asks for determinism, and "the frames compare equal" is a weaker claim than it
    sounds: it depends on what is compared and how. A hash over the serialised values,
    the column names and the row order is a single number two runs either match on or
    do not, which is what makes the guarantee checkable rather than argued.
    """
    digest = hashlib.sha256()
    for frame in (X_train, X_test):
        digest.update(",".join(str(c) for c in frame.columns).encode())
        digest.update(frame.to_csv(index=True, float_format="%.10g").encode())
    return digest.hexdigest()


# --------------------------------------------------------------------------------------
# Building the pipeline from a plan
# --------------------------------------------------------------------------------------

_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}

_IMPUTE_STRATEGY = {
    "impute_median": "median",
    "impute_mean": "mean",
    "impute_mode": "most_frequent",
    "impute_constant": "constant",
}


def _numeric_block(entries: list[ColumnPlan], plan: PreprocessingPlan) -> Pipeline | None:
    """One pipeline for all numeric columns sharing an imputation strategy."""
    steps: list[tuple[str, Any]] = []
    strategies = {e.action for e in entries if e.action in _IMPUTE_STRATEGY}

    if strategies:
        strategy = _IMPUTE_STRATEGY[next(iter(strategies))]
        kwargs: dict[str, Any] = {"strategy": strategy, "keep_empty_features": True}
        if strategy == "constant":
            kwargs["fill_value"] = float(
                next(iter(e.params.get("fill_value", 0.0) for e in entries), 0.0)
            )
        steps.append(("impute", SimpleImputer(**kwargs)))

    if plan.options.outliers == "clip":
        steps.append(("clip", IQRClipper()))

    if plan.options.power_transform_skewed and any(
        abs(e.stats.get("skew", 0.0)) > 1.0 for e in entries
    ):
        # Yeo-Johnson rather than a log: it accepts zero and negative values, which a
        # log does not, and a transform that fails on half the real columns is not a
        # transform anyone can switch on safely.
        steps.append(("power", PowerTransformer(method="yeo-johnson", standardize=False)))

    if plan.options.scaler != "none":
        steps.append(("scale", _SCALERS[plan.options.scaler]()))

    return Pipeline(steps) if steps else None


def _categorical_block(entry: ColumnPlan, plan: PreprocessingPlan) -> Pipeline:
    """One pipeline per categorical column — they rarely share an encoder."""
    steps: list[tuple[str, Any]] = [("cast", CategoricalCast())]

    if entry.rare_categories:
        steps.append((
            "rare",
            RareCategoryGrouper(
                rate=plan.options.rare_category_rate,
                keep={entry.column: entry.params.get("keep", [])} if entry.params.get("keep") else None,
            ),
        ))

    steps.append((
        "impute",
        SimpleImputer(strategy="most_frequent", keep_empty_features=True),
    ))

    if entry.action == "encode_onehot":
        steps.append(("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
    elif entry.action in ("encode_binary", "encode_ordinal"):
        steps.append((
            "encode",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        ))
    elif entry.action == "encode_target":
        steps.append((
            "encode",
            TargetEncoder(cv=TARGET_ENCODER_CV, random_state=plan.options.seed),
        ))
    elif entry.action == "encode_frequency":
        steps.append(("encode", FrequencyEncoder()))

    return Pipeline(steps)


def _block_name(column: str, position: int) -> str:
    """A transformer name sklearn accepts, derived from a column name it may not."""
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(column)).strip("_")
    safe = safe.replace("__", "_")[:40] or "col"
    return f"{position}_{safe}"


def build_pipeline(plan: PreprocessingPlan) -> ColumnTransformer:
    """Turn a plan into an unfitted ``ColumnTransformer``.

    Numeric columns are grouped — they share an imputer and a scaler — while each
    categorical column gets its own block, because the plan may well have chosen a
    different encoder for each.
    """
    blocks: list[tuple[str, Any, list[str]]] = []

    numeric = [e for e in plan.kept() if e.kind == "numeric"]
    if numeric:
        block = _numeric_block(numeric, plan)
        if block is not None:
            blocks.append(("numeric", block, [e.column for e in numeric]))
        else:
            blocks.append(("numeric", "passthrough", [e.column for e in numeric]))

    for position, entry in enumerate(plan.kept()):
        # sklearn rejects "__" in a transformer name (it is its parameter separator) and
        # requires the names to be unique, so the block is identified by position with
        # the column name sanitised onto it — readable in a repr, and safe for any
        # column name a user's file might contain.
        label = _block_name(entry.column, position)
        if entry.kind in ("categorical", "boolean"):
            blocks.append((f"cat_{label}", _categorical_block(entry, plan), [entry.column]))
        elif entry.kind == "datetime":
            blocks.append((
                f"date_{label}",
                Pipeline([("derive", DatetimeFeatures())]),
                [entry.column],
            ))

    # Missing indicators are their own block so they survive imputation: computing them
    # after the gaps are filled would record nothing.
    indicated = [e.column for e in plan.kept() if e.add_indicator]
    if indicated:
        blocks.append((
            "missing_flags",
            Pipeline([("flag", MissingFlags())]),
            indicated,
        ))

    if not blocks:
        raise ValueError(
            "This plan keeps no columns, so there is nothing to build. Re-include at "
            "least one column, or relax the thresholds that dropped them."
        )

    return ColumnTransformer(
        transformers=blocks, remainder="drop", verbose_feature_names_out=False
    ).set_output(transform="pandas")


class MissingFlags(TransformerMixin, BaseEstimator):
    """A 0/1 column per input, recording which rows were blank.

    Kept separate from imputation rather than using ``SimpleImputer(add_indicator=True)``
    so the flag's column name is ours and predictable — ``age_was_missing`` rather than
    ``missingindicator_age`` — and so the plan can add a flag to a column it is not
    imputing.
    """

    def fit(self, X: Any, y: Any = None) -> "MissingFlags":
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(list(frame.columns), dtype=object)
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(
            X, columns=list(self.feature_names_in_)
        )
        return pd.DataFrame(
            {f"{c}_was_missing": frame[c].isna().astype(float) for c in frame.columns},
            index=frame.index,
        )

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        names = list(self.feature_names_in_) if input_features is None else list(input_features)
        return np.asarray([f"{n}_was_missing" for n in names], dtype=object)


# --------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------


def _record(
    log: list[dict[str, Any]],
    run_id: str,
    stage: str,
    step: str,
    action: str,
    reason: str,
    before: dict[str, Any],
    after: dict[str, Any],
    column: str | None = None,
    params: dict[str, Any] | None = None,
    severity: str = "info",
    overridden: bool = False,
) -> None:
    """Append one §5 record. ``seq`` is assigned here so it is always contiguous."""
    log.append({
        "run_id": run_id,
        "seq": len(log) + 1,
        "stage": stage,
        "step": step,
        "column": column,
        "action": action,
        "params": params or {},
        "reason": reason,
        "before": before,
        "after": after,
        "severity": severity,
        "overridden": overridden,
    })


def _shape(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "missing": int(frame.isna().sum().sum()),
    }


def build_schema(plan: PreprocessingPlan, raw: pd.DataFrame, out: pd.DataFrame) -> dict[str, Any]:
    """The contract a later phase needs to transform a new row.

    This is the artifact §6 singles out for Phase 2 inference, and the thing it has to
    answer is narrow: *given a raw row, is it shaped the way the fitted pipeline
    expects?* So it records the input columns and their types, the categories each
    categorical column was fitted on, the target and the task — and the output feature
    names, so a caller can check the transform produced what the model was trained on.
    """
    return {
        "version": 1,
        "target": plan.target,
        "task": plan.task,
        "input_columns": [
            {
                "name": entry.column,
                "original_name": entry.original_name,
                "kind": entry.kind,
                "dtype": entry.dtype,
                "action": entry.action,
                "required": entry.action not in ("drop", "passthrough"),
                "categories": (
                    sorted({str(v) for v in raw[entry.column].dropna().unique()})
                    if entry.kind in ("categorical", "boolean")
                    and entry.action not in ("drop", "passthrough")
                    and raw[entry.column].nunique(dropna=True) <= 500
                    else None
                ),
            }
            for entry in plan.columns
        ],
        "output_features": [str(c) for c in out.columns],
        "n_features": int(out.shape[1]),
        "options": plan.options.model_dump(),
    }


def verify_log_invariant(plan: PreprocessingPlan, log: list[dict[str, Any]]) -> None:
    """§5's guarantee: every input column has a stated fate.

    Raised rather than logged. A glass-box tool that quietly loses track of a column has
    failed at the one thing it claims to do, and a run that cannot account for its own
    inputs should not be presented as a result.
    """
    accounted = {record["column"] for record in log if record["column"]}
    missing = [entry.column for entry in plan.columns if entry.column not in accounted]
    if missing:
        raise RuntimeError(
            "The log does not account for every input column, which breaks the "
            f"glass-box guarantee. Unaccounted: {', '.join(sorted(missing))}. "
            "This is a bug in the executor, not in the data."
        )


def execute(df: pd.DataFrame, plan: PreprocessingPlan, run_id: str) -> RunResult:
    """Run a plan against a dataset, and record every decision as it happens.

    The split is recreated from the plan's seed and test size, so the folds are the ones
    the planner measured on. Nothing here re-decides anything: each column's fate comes
    from its :class:`~app.plan.ColumnPlan`.
    """
    import time

    from app.services.checks import run_post_split_checks, sanity_baseline

    started = time.perf_counter()
    log: list[dict[str, Any]] = []
    warnings = list(plan.warnings)

    before_all = _shape(df)

    # ---- stage 1: stateless, pre-split ------------------------------------------
    #
    # The identical call the planner made (:func:`app.plan.prepare`), so the folds below
    # contain exactly the rows the plan's statistics were measured on. Doing the same
    # work a second time here, in slightly different words, is how the two drift.
    df, prepared_records = prepare(df, plan.target)
    for record in prepared_records:
        after_prepare = _shape(df)
        _record(
            log, run_id, "prepare", record["kind"], "drop_rows",
            record["message"] + " " + record["suggestion"],
            before_all, after_prepare,
            column=plan.target if record["kind"] == "blank_targets_removed" else "",
            severity=record["severity"],
        )

    if df.empty:
        raise ValueError(
            f"No rows are left once rows with no value in {plan.target!r} are set "
            "aside, so there is nothing to run."
        )

    # ---- stage 2: the split ------------------------------------------------------
    #
    # Stratification is decided by the one shared rule, not a local copy of it.
    classification = "class" in plan.task
    y = df[plan.target]
    stratify, stratify_warning = choose_stratification(y, classification, plan.options.test_size)
    if stratify_warning is not None and not any(
        w.get("kind") == "stratification_disabled" for w in warnings
    ):
        warnings.append(stratify_warning)

    train_index, test_index = train_test_split(
        df.index, test_size=plan.options.test_size, random_state=plan.options.seed,
        shuffle=True, stratify=stratify,
    )
    train_index, test_index = pd.Index(train_index), pd.Index(test_index)

    _record(
        log, run_id, "split", "train_test_split", "split",
        f"Set aside {len(test_index):,} row(s) — {plan.options.test_size:.0%} — before "
        "anything was measured or fitted. Every value learned below comes from the "
        f"other {len(train_index):,} rows alone."
        + ("" if stratify is not None else " The split could not be stratified."),
        _shape(df),
        {"train_rows": int(len(train_index)), "test_rows": int(len(test_index))},
        params={
            "test_size": plan.options.test_size,
            "seed": plan.options.seed,
            "stratified": stratify is not None,
        },
        severity="info" if stratify is not None else "warn",
    )

    # ---- stage 3: per-column actions, fitted on train ----------------------------
    features = df.drop(columns=[plan.target])
    for entry in plan.columns:
        if entry.column == plan.target:
            _record(
                log, run_id, "columns", "target", "passthrough", entry.reason,
                {"role": "target"}, {"role": "target", "task": plan.task},
                column=entry.column, overridden=entry.overridden,
            )
            continue

        if entry.action == "drop":
            _record(
                log, run_id, "columns", "drop", "drop", entry.reason,
                {"present": True, **entry.stats}, {"present": False},
                column=entry.column,
                severity=(entry.warning or {}).get("severity", "info"),
                overridden=entry.overridden,
            )
            continue

        _record(
            log, run_id, "columns", entry.action, entry.action, entry.reason,
            {"kind": entry.kind, "dtype": entry.dtype, **entry.stats},
            {
                "action": entry.action,
                "indicator_added": entry.add_indicator,
                "rare_grouped": len(entry.rare_categories),
            },
            column=entry.column,
            params=entry.params,
            severity=(entry.warning or {}).get("severity", "info"),
            overridden=entry.overridden,
        )

    kept = [e.column for e in plan.kept()]
    X = features[[c for c in kept if c in features.columns]]

    pipeline = build_pipeline(plan)
    X_train_raw, X_test_raw = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]

    before_fit = _shape(X_train_raw)
    X_train = pipeline.fit_transform(X_train_raw, y_train)
    X_test = pipeline.transform(X_test_raw)

    _record(
        log, run_id, "fit", "fit_transform", "fit_pipeline",
        f"Fitted every transformer on the {len(train_index):,} training rows and "
        f"applied the frozen result to the {len(test_index):,} held-out rows. "
        f"{before_fit['columns']} input column(s) became {X_train.shape[1]}.",
        before_fit, _shape(X_train),
        params={"scaler": plan.options.scaler, "outliers": plan.options.outliers},
    )

    # ---- stage 4: post-split checks ----------------------------------------------
    checks = run_post_split_checks(X_train, X_test, y_train, y_test)
    checks["baseline"] = sanity_baseline(X_train, y_train, plan.task, plan.options.seed)
    warnings.extend(checks["warnings"])

    drift = checks["drift"]
    _record(
        log, run_id, "checks", "train_test_drift", "ks_test",
        f"Compared each feature's distribution across the two folds: "
        f"{drift['flagged']} of {drift['tested']} differ at p<{drift['threshold']}. "
        + drift["note"],
        {"tested": drift["tested"]}, {"flagged": drift["flagged"]},
        params={"threshold": drift["threshold"]},
        severity="warn" if drift["flagged"] else "info",
    )

    if checks["baseline"]:
        baseline = checks["baseline"]
        _record(
            log, run_id, "checks", "sanity_baseline", "cross_validate",
            baseline["verdict"] + " " + baseline["note"],
            {"model": "dummy", "score": baseline["baseline"]},
            {"model": baseline["model"], "score": baseline["model_score"]},
            params={"metric": baseline["metric"], "folds": baseline["folds"]},
            severity=baseline["severity"],
        )

    verify_log_invariant(plan, log)
    schema = build_schema(plan, df, X_train)

    return RunResult(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        pipeline=pipeline, plan=plan, log=log, schema=schema,
        warnings=warnings, checks=checks, stratified=stratify is not None,
        duration_ms=round((time.perf_counter() - started) * 1000),
    )
