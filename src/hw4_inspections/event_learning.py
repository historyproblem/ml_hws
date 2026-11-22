"""Site-cross-fitted first-routine risk heads from strictly historical landmarks."""
from __future__ import annotations

import hashlib
from functools import lru_cache
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from .features import INSPECTION_TYPES, MISSING, RESULTS, STATIC_CATEGORIES, _codes

WINDOW = pd.Timedelta(days=365)
WEEK = pd.Timedelta(days=7)
FOLDS = 5
HEADS = ("event_logit", "event_cb")
CATEGORICAL = (*STATIC_CATEGORIES, "latest_type", "latest_result", "routine_result", "previous_routine_result")
CODES = [f"{prefix}_{i:02d}" for prefix in ("last_routine_code", "routine_code_fraction") for i in range(1, 45)]
HISTORY = ["forecast_period", "site_id", "inspection_week", "inspection_type", "result",
           *STATIC_CATEGORIES, "violation_codes"]
CB_PARAMS = dict(iterations=250, depth=3, learning_rate=.04, l2_leaf_reg=12., random_seed=42,
                 loss_function="Logloss", bootstrap_type="MVS", subsample=.8,
                 allow_writing_files=False, verbose=False, task_type="CPU")
LOGIT_PARAMS = dict(C=.1, solver="lbfgs", max_iter=1000, tol=1e-6, fit_intercept=True, class_weight=None)


@lru_cache(maxsize=100000)
def site_fold(site: str) -> int:
    return int.from_bytes(hashlib.sha256(site.encode("utf-8")).digest()[:8], "big") % FOLDS


def previous_cutoff(start):
    """Monday of the week containing the quarter start, minus one week."""
    start = pd.Timestamp(start)
    return start - pd.Timedelta(days=start.dayofweek + 7)


def snapshot(history, period, cutoff):
    period, cutoff = pd.Timestamp(period), pd.Timestamp(cutoff)
    if cutoff != previous_cutoff(period) or not period.is_quarter_start:
        raise ValueError("Unsupported forecast landmark/cutoff calendar")
    field = history.loc[history.forecast_period.eq(period.strftime("%Y-%m-%d")), HISTORY].copy()
    field["week"] = pd.to_datetime(field.inspection_week, errors="raise")
    field = field.loc[field.week.lt(cutoff)].drop(columns=["forecast_period", "inspection_week"])
    if field.empty or field[["site_id", "week"]].isna().any().any() or not field.week.dt.dayofweek.eq(0).all():
        raise ValueError("No valid weekly history in requested snapshot")
    for name in ["inspection_type", "result", *STATIC_CATEGORIES]:
        field[name] = field[name].fillna(MISSING).astype(str)
    field["tokens"] = field.violation_codes.fillna("").astype(str).map(_codes)
    field = field.drop(columns="violation_codes").drop_duplicates()
    field["load"] = field.tokens.map(len)
    field["fold"] = field.site_id.map(site_fold)
    field["row_fail"] = field.result.eq("fail").astype(float)
    field["row_conditional"] = field.result.eq("conditional").astype(float)
    # Multiple distinct rows in an older source week cannot establish their
    # within-week order. Preserve them until endpoint selection, then exclude
    # ambiguous sites from last-visit descriptors if that week is in a prefix.
    return field.sort_values(["week", "site_id"], kind="stable").reset_index(drop=True)


def first_endpoints(hist, start, end):
    r = hist.loc[hist.inspection_type.eq("routine") & hist.week.lt(end)
                 & (hist.week + WEEK).gt(start)]
    earliest = r.groupby("site_id", sort=False).week.transform("min")
    first = r.loc[r.week.eq(earliest)]
    unique = ~first.site_id.duplicated(keep=False)
    return first.loc[unique & first.week.ge(start) & (first.week + WEEK).le(end)
                     & first.result.isin(RESULTS), ["site_id", "week", "row_fail", "fold"]].reset_index(drop=True)


def visit_state(hist, cutoff, sites):
    """Feature values available before this landmark, with no endpoint fields."""
    index = pd.Index(sites, name="site_id")
    if not index.is_unique:
        raise ValueError("Repeated site within a query landmark")
    part = hist.loc[hist.week.lt(cutoff) & hist.week.ge(cutoff - WINDOW)].copy()
    # A conflicting past site/week is unavailable to the learned event head.
    # Do not impose an artificial chronology within it.
    part = part.loc[~part.duplicated(["site_id", "week"], keep=False)]
    part["age"] = (cutoff - part.week).dt.days
    routine = part.loc[part.inspection_type.eq("routine")]
    out = pd.DataFrame(index=index)
    g = part.groupby("site_id", sort=False)
    rg = routine.groupby("site_id", sort=False)
    last = g.tail(1).set_index("site_id")
    prev_routine = routine.loc[~routine.index.isin(rg.tail(1).index)].groupby("site_id", sort=False).tail(1).set_index("site_id")
    last_routine = rg.tail(1).set_index("site_id")

    for name in STATIC_CATEGORIES:
        out[name] = last[name]
    for name, values in [("latest_type", last.inspection_type), ("latest_result", last.result),
                         ("routine_result", last_routine.result), ("previous_routine_result", prev_routine.result)]:
        out[name] = values
    out[list(CATEGORICAL)] = out[list(CATEGORICAL)].fillna(MISSING).astype(str)

    for label, group in [("visit", g), ("routine", rg)]:
        out[f"{label}_n"] = group.size().reindex(index, fill_value=0)
        out[f"{label}_fail_mean"] = group.row_fail.mean()
        out[f"{label}_conditional_mean"] = group.row_conditional.mean()
        out[f"{label}_load_mean"] = group.load.mean()
        out[f"{label}_load_max"] = group.load.max()
    for name, source in [("latest", last), ("routine", last_routine), ("previous_routine", prev_routine)]:
        out[f"{name}_age"] = source.age
        out[f"{name}_load"] = source.load
    out["routine_interval"] = out.previous_routine_age - out.routine_age
    out["routine_load_change"] = out.routine_load - out.previous_routine_load
    for kind in INSPECTION_TYPES:
        group = part.loc[part.inspection_type.eq(kind)].groupby("site_id", sort=False)
        out[f"n_{kind}"] = group.size().reindex(index, fill_value=0)
    since = part.loc[part.week.gt(part.site_id.map(last_routine.week))]
    followups = since.groupby("site_id", sort=False)
    out["after_routine_n"] = followups.size().reindex(index, fill_value=0)
    out["after_routine_load_mean"] = followups.load.mean()
    out["after_routine_fail_mean"] = followups.row_fail.mean()
    matrix = np.zeros((len(routine), 44), dtype=float)
    for i, codes in enumerate(routine.tokens):
        if codes:
            matrix[i, np.asarray(codes) - 1] = 1.
    code_frame = pd.DataFrame(matrix, index=routine.index)
    code_frame["site_id"] = routine.site_id
    code_group = code_frame.groupby("site_id", sort=False)
    last_codes = code_group.tail(1).set_index("site_id").reindex(index, fill_value=0)
    fraction = code_group.mean().reindex(index, fill_value=0)
    code_values = np.column_stack([last_codes.to_numpy(), fraction.to_numpy()])
    out = pd.concat([out, pd.DataFrame(code_values, index=index, columns=CODES)], axis=1)
    return out.reset_index(drop=True)


def landmark_examples(hist, period, cutoff, query_sites):
    """Build distinct historical site-quarter landmarks only from this snapshot."""
    period, cutoff = pd.Timestamp(period), pd.Timestamp(cutoff)
    if hist.week.max() >= cutoff:
        raise ValueError("Unclosed source event")
    records, matrices = [], []
    # Build the candidates once. Eligibility for training an individual head is
    # checked again using the coverage of its 4 admissible site partitions.
    for quarter in pd.period_range(hist.week.min().to_period("Q"), period.to_period("Q") - 1, freq="Q"):
        start, end = quarter.start_time, (quarter + 1).start_time
        landmark = previous_cutoff(start)
        if end > cutoff or landmark - WINDOW < hist.week.min() or end > hist.week.max() + WEEK:
            continue
        endpoints = first_endpoints(hist, start, end)
        if endpoints.empty:
            continue
        inputs = visit_state(hist, landmark, endpoints.site_id)
        record = endpoints.rename(columns={"week":"label_week", "row_fail":"y"}).assign(
            landmark=landmark, quarter_start=start, quarter_end=end)
        records.append(record)
        matrices.append(inputs)
    if not records:
        raise ValueError("No fully covered retrospective training quarters")
    events = pd.concat(records, ignore_index=True)
    x = pd.concat(matrices, ignore_index=True)
    if (events.duplicated(["site_id", "quarter_start"]).any() or not events.label_week.gt(events.landmark).all()
            or not (events.label_week + WEEK).le(cutoff).all()):
        raise AssertionError("Endpoint is ambiguous or outside the permitted interval")
    query = visit_state(hist, cutoff, query_sites)
    if list(x) != list(query):
        raise AssertionError("Event and forecast schemas differ")
    return x, events, query


def training_mask(hist, events, heldout):
    other_weeks = hist.loc[hist.fold.ne(heldout), "week"]
    if other_weeks.empty:
        raise ValueError("No remaining sites after focal site-fold exclusion")
    return (events.fold.ne(heldout) & (events.landmark - WINDOW).ge(other_weeks.min())
            & events.quarter_end.le(other_weeks.max() + WEEK)).to_numpy()


def fit_heads(x, y, threads, smoke=False):
    from catboost import CatBoostClassifier

    if not 1 <= threads <= 2 or set(np.unique(y)) != {0., 1.}:
        raise ValueError("Each head needs two outcome classes and one or two CPU threads")
    continuous = [c for c in x if c not in (*CATEGORICAL, *CODES)]
    numeric = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), StandardScaler())
    design = ColumnTransformer([("continuous", numeric, continuous), ("code", "passthrough", CODES),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(CATEGORICAL))])
    logit = make_pipeline(design, LogisticRegression(**LOGIT_PARAMS))
    with threadpool_limits(limits=threads), warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        logit.fit(x, y)
    params = {**CB_PARAMS, "thread_count": threads}
    if smoke:
        params["iterations"] = 4
    boost = CatBoostClassifier(**params).fit(x, y, cat_features=list(CATEGORICAL))
    return {HEADS[0]: logit, HEADS[1]: boost, "columns": list(x), "threads": threads}


def predict_heads(bundle, x):
    if list(x) != bundle["columns"]:
        raise ValueError("Mismatching event head schema")
    with threadpool_limits(limits=bundle["threads"]):
        scores = np.column_stack([bundle[HEADS[0]].predict_proba(x)[:, 1],
            bundle[HEADS[1]].predict_proba(x, thread_count=bundle["threads"])[:, 1]]).astype(np.float64)
    if scores.shape != (len(x),2) or not np.isfinite(scores).all() or ((scores<0) | (scores>1)).any():
        raise ValueError("Invalid first-level risk scores")
    return scores
