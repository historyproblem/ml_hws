"""Expand event supervision by earliest admissible issuance per historical site-quarter."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import event_learning as event

LOAD_HEAD = "event_log_recorded_load"
ALL_HEADS = [*event.HEADS, LOAD_HEAD]
LOAD_PARAMS = {k:v for k,v in event.CB_PARAMS.items() if k != "loss_function"}
LOAD_PARAMS.update(loss_function="RMSE")


@dataclass
class Release:
    period: pd.Timestamp
    cutoff: pd.Timestamp
    features: pd.DataFrame
    outcomes: pd.DataFrame
    coverage: dict


def make_release(hist, period, cutoff, query_sites):
    """Each example's predictor+target come from a single historical snapshot."""
    period, cutoff = pd.Timestamp(period), pd.Timestamp(cutoff)
    x, y, query = event.landmark_examples(hist, period, cutoff, query_sites)
    # The already identified earliest, unambiguous endpoint uniquely selects its
    # recorded load. No endpoint fields are joined to either feature matrix.
    keys = pd.MultiIndex.from_frame(y[["site_id","label_week"]])
    observations = hist.loc[hist.inspection_type.eq("routine"),["site_id","week","load"]].rename(columns={"week":"label_week"})
    observations = observations.set_index(["site_id","label_week"])
    matching = observations.loc[observations.index.isin(keys)]
    if not matching.index.is_unique or not keys.is_unique or len(matching) != len(y):
        raise ValueError("Endpoint load cannot be identified without ambiguity")
    y = y.assign(recorded_load=matching.reindex(keys).load.to_numpy(), source_period=period, source_cutoff=cutoff)
    if not y.recorded_load.between(0,44).all() or not (y.label_week + event.WEEK).le(cutoff).all():
        raise ValueError("Invalid endpoint load or availability")
    coverage = {}
    for heldout in range(event.FOLDS):
        dates = hist.loc[hist.fold.ne(heldout),"week"]
        if dates.empty:
            raise ValueError("No non-focal coverage")
        coverage[heldout] = (dates.min(), dates.max()+event.WEEK)
    return Release(period,cutoff,x,y,coverage),query


def asof_library(releases, period, cutoff, heldout):
    """Earliest released, sufficiently covered example per site-quarter."""
    period,cutoff = pd.Timestamp(period),pd.Timestamp(cutoff)
    available=sorted((s for s in releases if s.period<=period and s.cutoff<=cutoff),key=lambda s:s.period)
    if not available or len({s.period for s in available}) != len(available):
        raise ValueError("Missing or repeated snapshot issuance")
    frame, labels = [], []
    for source in available:
        minimum, exclusive_end = source.coverage[heldout]
        y = source.outcomes
        selected = (y.fold.ne(heldout) & (y.landmark-event.WINDOW).ge(minimum) & y.quarter_end.le(exclusive_end)
                    & (y.label_week+event.WEEK).le(source.cutoff) & (y.label_week+event.WEEK).le(cutoff))
        if selected.any():
            frame.append(source.features.loc[selected])
            labels.append(y.loc[selected])
    if not labels:
        raise ValueError("No mature, non-focal event labels")
    x=pd.concat(frame,ignore_index=True)
    y=pd.concat(labels,ignore_index=True)
    keep=~y.duplicated(["site_id","quarter_start"],keep="first")
    kept=y.loc[keep].reset_index(drop=True)
    # Differences in later representations are a diagnostic only. The earliest
    # admissible record, including its features and labels, remains canonical.
    groups=y.groupby(["site_id","quarter_start"],sort=False)
    differing=(groups[["label_week","y","recorded_load"]].nunique()>1).any(axis=1)
    audit={"candidate_records":len(y),"unique_site_quarters":len(kept),
        "discarded_later_copies":int((~keep).sum()),"keys_with_conflicting_later_outcomes":int(differing.sum()),
        "unique_sites":kept.site_id.nunique(),"covered_quarters":kept.quarter_start.nunique(),
        "fold":heldout,"forecast_period":period.strftime("%Y-%m-%d"),"history_before":cutoff.strftime("%Y-%m-%d"),
        "first_source":kept.source_period.min().strftime("%Y-%m-%d"),
        "latest_retained_source":kept.source_period.max().strftime("%Y-%m-%d"),
        "latest_endpoint_week":kept.label_week.max().strftime("%Y-%m-%d"),
        "unique_rows_by_history_quarter":{k.strftime("%Y-%m-%d"):int(v) for k,v in kept.quarter_start.value_counts().sort_index().items()},
        "unique_rows_by_released_source":{k.strftime("%Y-%m-%d"):int(v) for k,v in kept.source_period.value_counts().sort_index().items()}}
    if kept.fold.eq(heldout).any() or not (kept.label_week+event.WEEK).le(kept.source_cutoff).all():
        raise AssertionError("Focal-site or future label in canonical library")
    return x.loc[keep].reset_index(drop=True), kept, audit


def fit_library(x, outcomes, threads, smoke=False):
    from catboost import CatBoostRegressor
    bundle=event.fit_heads(x,outcomes.y,threads,smoke=smoke)
    params={**LOAD_PARAMS,"thread_count":threads}
    if smoke:
        params["iterations"]=4
    values=outcomes.recorded_load.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values<0).any() or (values>44).any():
        raise ValueError("Invalid observed distinct code count")
    bundle[LOAD_HEAD]=CatBoostRegressor(**params).fit(x,np.log1p(values),cat_features=list(event.CATEGORICAL))
    return bundle


def predict_library(bundle, x):
    risk=event.predict_heads(bundle,x)
    logload=bundle[LOAD_HEAD].predict(x,thread_count=bundle["threads"])
    # The auxiliary target is log1p(recorded code count), naturally [0,log45].
    if not np.isfinite(logload).all():
        raise ValueError("Nonfinite future recorded-load prediction")
    return np.column_stack([risk,np.clip(logload,0.,np.log1p(44))]).astype(np.float64)
