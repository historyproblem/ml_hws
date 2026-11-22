"""Past, other-site routine transitions as a small HW4 feature block.

Each cutoff uses its exact forecast snapshot. Both routine visits of every
adjacent pair must already be observed. All pairs of the focal site are removed
from counts, global priors and conditional code adjustments. Official training
targets are never read; codes remain opaque tokens without sanitary meanings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import MISSING, RESULTS, _codes


SMOOTHING_PAIRS = 20.0
FEATURE_COLUMNS = (
    "transition_next_fail_risk_burden",
    "transition_next_fail_state",
    "transition_state_log1p_pairs",
    "transition_code_fail_effect_mean",
    "transition_code_fail_effect_max",
    "transition_code_min_pairs",
    "transition_next_load_state",
)
_ROW_COLUMNS = ["forecast_period", "site_id", "history_before"]
_HISTORY_COLUMNS = ["forecast_period", "site_id", "inspection_week",
                    "inspection_type", "result", "violation_codes", "risk_level"]


def _burden_bucket(counts):
    """Five fixed buckets: 0, 1-2, 3-5, 6-8 and 9+ distinct recorded codes."""
    return np.searchsorted([0, 2, 5, 8], np.asarray(counts), side="left")


def _routine_and_pairs(snapshot, cutoff):
    routine = snapshot.loc[snapshot._date.lt(cutoff)].copy()
    # Equivalent duplicate records do not manufacture additional visits. A
    # conflicting same-site/week routine cannot be ordered and fails explicitly.
    canonical = ["site_id", "_date", "_risk", "_result", "_codes"]
    routine = routine.drop_duplicates(canonical)
    if routine.duplicated(["site_id", "_date"]).any():
        raise ValueError("Conflicting routine visits share a site and inspection week")
    routine = routine.sort_values("_date", kind="stable")
    group = routine.groupby("site_id", sort=False)
    routine["_next_result"] = group["_result"].shift(-1)
    routine["_next_load"] = group["_load"].shift(-1)
    # Form adjacency BEFORE excluding unknown outcomes: a known visit must not
    # jump over an unknown routine and become an invented completed pair.
    pairs = routine.loc[routine._result.isin(RESULTS) & routine._next_result.isin(RESULTS)].copy()
    pairs["_next_fail"] = pairs._next_result.eq("fail").astype(float)
    return routine, pairs


def _focal_excluded_features(pairs, states, focal_sites):
    """Vectorized site subtraction, including every conditional code baseline."""
    count = len(states)
    result = np.full((count, len(FEATURE_COLUMNS)), np.nan, dtype=float)
    result[:, 2] = 0.0
    result[:, 5] = 0.0
    if pairs.empty:
        return result
    sites = pd.Index(pd.unique(focal_sites))
    query_site = sites.get_indexer(focal_sites)
    pair_site = sites.get_indexer(pairs.site_id)
    own = pair_site >= 0
    own_ids = pair_site[own]

    # A stratum is risk x prior burden; detailed states add prior outcome.
    stratum_id, strata = pd.factorize(pd.MultiIndex.from_frame(pairs[["_risk", "_bucket"]]), sort=True)
    detail_id, details = pd.factorize(pd.MultiIndex.from_frame(pairs[["_risk", "_bucket", "_result"]]), sort=True)
    n_sites, n_strata, n_details = len(sites), len(strata), len(details)
    fail = pairs._next_fail.to_numpy(float)
    load = pairs._next_load.to_numpy(float)
    n_s = np.bincount(stratum_id, minlength=n_strata).astype(float)
    f_s = np.bincount(stratum_id, weights=fail, minlength=n_strata)
    l_s = np.bincount(stratum_id, weights=load, minlength=n_strata)
    n_d = np.bincount(detail_id, minlength=n_details).astype(float)
    f_d = np.bincount(detail_id, weights=fail, minlength=n_details)
    l_d = np.bincount(detail_id, weights=load, minlength=n_details)
    own_n_s = np.zeros((n_sites, n_strata))
    own_f_s = np.zeros_like(own_n_s)
    own_l_s = np.zeros_like(own_n_s)
    own_n_d = np.zeros((n_sites, n_details))
    own_f_d = np.zeros_like(own_n_d)
    own_l_d = np.zeros_like(own_n_d)
    for destination, group_id, weights in [
        (own_n_s, stratum_id, np.ones(len(pairs))),
        (own_f_s, stratum_id, fail), (own_l_s, stratum_id, load),
        (own_n_d, detail_id, np.ones(len(pairs))),
        (own_f_d, detail_id, fail), (own_l_d, detail_id, load),
    ]:
        np.add.at(destination, (own_ids, group_id[own]), weights[own])

    other_total = len(pairs) - own_n_s.sum(axis=1)
    global_fail = np.divide(fail.sum() - own_f_s.sum(axis=1), other_total,
                            out=np.full(n_sites, np.nan), where=other_total > 0)
    global_load = np.divide(load.sum() - own_l_s.sum(axis=1), other_total,
                            out=np.full(n_sites, np.nan), where=other_total > 0)
    other_n_s = n_s[None, :] - own_n_s
    # Critically, these parents exclude the focal site for EVERY stratum, not
    # only its current stratum. Code residuals below use all of these parents.
    parent_fail = (f_s[None, :] - own_f_s + SMOOTHING_PAIRS * global_fail[:, None]) / (other_n_s + SMOOTHING_PAIRS)
    parent_load = (l_s[None, :] - own_l_s + SMOOTHING_PAIRS * global_load[:, None]) / (other_n_s + SMOOTHING_PAIRS)
    query_s = strata.get_indexer(pd.MultiIndex.from_frame(states[["_risk", "_bucket"]]))
    query_d = details.get_indexer(pd.MultiIndex.from_frame(states[["_risk", "_bucket", "_result"]]))
    risk_fail = global_fail[query_site].copy()
    risk_load = global_load[query_site].copy()
    found = query_s >= 0
    risk_fail[found] = parent_fail[query_site[found], query_s[found]]
    risk_load[found] = parent_load[query_site[found], query_s[found]]
    support, state_fail, state_load = np.zeros(count), np.zeros(count), np.zeros(count)
    found = query_d >= 0
    support[found] = n_d[query_d[found]] - own_n_d[query_site[found], query_d[found]]
    state_fail[found] = f_d[query_d[found]] - own_f_d[query_site[found], query_d[found]]
    state_load[found] = l_d[query_d[found]] - own_l_d[query_site[found], query_d[found]]
    result[:, 0] = risk_fail
    result[:, 1] = (state_fail + SMOOTHING_PAIRS * risk_fail) / (support + SMOOTHING_PAIRS)
    result[:, 2] = np.log1p(support)
    result[:, 6] = (state_load + SMOOTHING_PAIRS * risk_load) / (support + SMOOTHING_PAIRS)

    codes = np.zeros((len(pairs), 44), dtype=float)
    for i, tokens in enumerate(pairs._codes):
        if tokens:
            codes[i, np.asarray(tokens) - 1] = 1.0
    n_sc = np.zeros((n_strata, 44))
    np.add.at(n_sc, stratum_id, codes)
    own_n_sc = np.zeros((n_sites, n_strata, 44))
    np.add.at(own_n_sc, (own_ids, stratum_id[own]), codes[own])
    own_f_code = np.zeros((n_sites, 44))
    np.add.at(own_f_code, own_ids, codes[own] * fail[own, None])
    other_n_sc = n_sc[None, :, :] - own_n_sc
    code_support = other_n_sc.sum(axis=1)
    expected_fail = (other_n_sc * parent_fail[:, :, None]).sum(axis=1)
    code_fail = (codes * fail[:, None]).sum(axis=0)[None, :] - own_f_code
    # e_c = sum_b(f_bc - n_bc*p_b)/(sum_b n_bc + 20), with all counts
    # AND all p_b computed from other sites for the particular query site.
    effects = (code_fail - expected_fail) / (code_support + SMOOTHING_PAIRS)
    for i, (site, tokens) in enumerate(zip(query_site, states._codes)):
        if tokens:
            columns = np.asarray(tokens) - 1
            result[i, 3] = effects[site, columns].mean()
            result[i, 4] = effects[site, columns].max()
            result[i, 5] = code_support[site, columns].min()
    return result


def build_transition_features(rows: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Seven numeric transition estimates, in the exact input row/index order.

    The query state is its latest recorded routine (risk, code-count bucket,
    outcome and opaque codes). Unknown latest outcomes are ineligible, without
    falling back to an older visit. Missing/ineligible states receive NaN
    estimates and zero support. No other-site pairs yields the same missing
    estimates. A blank code list yields missing code effects and zero support.
    Risk is taken from the corresponding routine record, never a future row.
    """
    for frame, required, name in [(rows, _ROW_COLUMNS, "rows"),
                                   (history, _HISTORY_COLUMNS, "history")]:
        missing = set(required) - set(frame)
        if missing:
            raise ValueError(f"Missing {name} columns: {sorted(missing)}")
    r = rows[_ROW_COLUMNS].reset_index(drop=True).copy()
    r["_row"] = np.arange(len(r))
    for name in ["forecast_period", "history_before"]:
        r[name] = pd.to_datetime(r[name], errors="raise")
    if r[_ROW_COLUMNS].isna().any().any():
        raise ValueError("Query snapshot, site and cutoff must be nonmissing")
    out = pd.DataFrame(np.nan, index=pd.RangeIndex(len(r)), columns=FEATURE_COLUMNS)
    out[FEATURE_COLUMNS[2]] = 0.0
    out[FEATURE_COLUMNS[5]] = 0.0
    h = history[_HISTORY_COLUMNS].copy()
    h["forecast_period"] = pd.to_datetime(h["forecast_period"], errors="raise")
    h = h.loc[h.forecast_period.isin(r.forecast_period) & h.inspection_type.eq("routine")].copy()
    h["_date"] = pd.to_datetime(h.inspection_week, errors="raise")
    if h[["site_id", "_date"]].isna().any().any():
        raise ValueError("Routine site and inspection week must be nonmissing")
    h["_risk"] = h.risk_level.fillna(MISSING).astype(str)
    h["_result"] = h.result.fillna("").astype(str)
    h["_codes"] = h.violation_codes.fillna("").astype(str).map(_codes)
    h["_load"] = h._codes.map(len).astype(float)
    h["_bucket"] = _burden_bucket(h._load)
    for (period, cutoff), query in r.groupby(["forecast_period", "history_before"], sort=False):
        routine, pairs = _routine_and_pairs(h.loc[h.forecast_period.eq(period)], cutoff)
        if routine.empty:
            continue
        latest = routine.groupby("site_id", sort=False).tail(1).set_index("site_id")
        states = latest.reindex(query.site_id).reset_index(drop=True)
        eligible = states._result.isin(RESULTS).to_numpy()
        if not eligible.any():
            continue
        positions = query._row.to_numpy()[eligible]
        values = _focal_excluded_features(
            pairs, states.loc[eligible].reset_index(drop=True), query.site_id.to_numpy()[eligible])
        out.loc[positions, :] = values
    out.index = rows.index
    if np.isinf(out.to_numpy()).any():
        raise AssertionError("Transition features must be finite or missing")
    return out
