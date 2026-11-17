"""Optional, label-free HW4 hypotheses, separate from the measured baseline.

No feature pools forecast snapshots. Context excludes the focal site's visits,
including from the global smoothing prior. Dynamics uses the focal history only.
Both builders return numeric columns in the exact input row/index order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import MISSING, _codes

CONTEXT_GROUPS = {
    "area": ["area_id"],
    "facility": ["facility_type"],
    "risk": ["risk_level"],
    "area_facility": ["area_id", "facility_type"],
    "area_risk": ["area_id", "risk_level"],
    "facility_risk": ["facility_type", "risk_level"],
}
CATEGORIES = ["area_id", "facility_type", "risk_level"]
HISTORY_COLUMNS = ["forecast_period", "site_id", "inspection_week",
                   "inspection_type", "result", "violation_codes"]
SMOOTHING_VISITS = 20.0


def _prepare(rows, history, *, context=False):
    required = ["forecast_period", "site_id", "history_before"]
    hcols = HISTORY_COLUMNS + (CATEGORIES if context else [])
    if context:
        required += CATEGORIES
    for frame, columns, name in [(rows, required, "rows"), (history, hcols, "history")]:
        missing = set(columns) - set(frame)
        if missing:
            raise ValueError(f"Missing {name} columns: {sorted(missing)}")
    r = rows[required].reset_index(drop=True).copy()
    r["_row"] = np.arange(len(r))
    r["forecast_period"] = pd.to_datetime(r["forecast_period"], errors="raise")
    r["history_before"] = pd.to_datetime(r["history_before"], errors="raise")
    h = history[hcols].copy()
    h["forecast_period"] = pd.to_datetime(h["forecast_period"], errors="raise")
    h["_date"] = pd.to_datetime(h["inspection_week"], errors="raise")
    for name in ["inspection_type", "result", "violation_codes"]:
        h[name] = h[name].fillna("").astype(str)
    h["_blank_codes"] = h["violation_codes"].str.strip().eq("").astype(float)
    h["_codes"] = h["violation_codes"].map(_codes)
    h["_load"] = h["_codes"].map(len).astype(float)
    h["_fail"] = (h["result"] == "fail").astype(float)
    h["_conditional"] = (h["result"] == "conditional").astype(float)
    if context:
        for name in CATEGORIES:
            r[name] = r[name].fillna(MISSING).astype(str)
            h[name] = h[name].fillna(MISSING).astype(str)
    return r, h


def _summary(part, keys):
    return part.groupby(keys, sort=False, dropna=False).agg(
        n=("_fail", "size"), fail=("_fail", "sum"), load=("_load", "sum"))


def _lookup(table, keys):
    if len(keys.columns) == 1:
        index = pd.Index(keys.iloc[:, 0], name=keys.columns[0])
    else:
        index = pd.MultiIndex.from_frame(keys)
    return table.reindex(index).fillna(0).reset_index(drop=True)


def build_context_features(rows: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Return 72 numeric context columns, excluding the focal site's records.

    Each query uses its exact forecast snapshot and strict per-row cutoff.
    Within all available and recent365d windows, six contextual groups receive
    counts and failure/code-load means, for all visits and routine-only visits.
    Means use a fixed 20-visit prior from OTHER sites in the same snapshot/window
    and visit subset. Empty contexts fall back to that background; no background
    yields NaN, rather than an assumed zero risk.
    """
    r, h = _prepare(rows, history, context=True)
    names = [f"ctx_{group}_{window}_{subset}_{metric}"
             for window in ["all", "365d"] for group in CONTEXT_GROUPS
             for subset in ["visits", "routine"]
             for metric in ["n", "fail_smooth", "load_smooth"]]
    out = pd.DataFrame(np.nan, index=pd.RangeIndex(len(r)), columns=names)
    for (period, cutoff), query in r.groupby(["forecast_period", "history_before"], sort=False):
        snapshot = h.loc[(h["forecast_period"] == period) & (h["_date"] < cutoff)]
        query = query.reset_index(drop=True)
        positions = query["_row"].to_numpy()
        for window in ["all", "365d"]:
            past = snapshot if window == "all" else snapshot.loc[(cutoff - snapshot["_date"]).dt.days <= 365]
            for subset in ["visits", "routine"]:
                part = past if subset == "visits" else past.loc[past["inspection_type"] == "routine"]
                global_totals = part[["_fail", "_load"]].sum().to_numpy()
                global_count = len(part)
                own_global = _lookup(_summary(part, ["site_id"]), query[["site_id"]])
                other_n = global_count - own_global["n"].to_numpy()
                other_fail = global_totals[0] - own_global["fail"].to_numpy()
                other_load = global_totals[1] - own_global["load"].to_numpy()
                prior_fail = np.divide(other_fail, other_n, out=np.full(len(query), np.nan), where=other_n > 0)
                prior_load = np.divide(other_load, other_n, out=np.full(len(query), np.nan), where=other_n > 0)
                for group, keys in CONTEXT_GROUPS.items():
                    total = _lookup(_summary(part, keys), query[keys])
                    own = _lookup(_summary(part, ["site_id", *keys]), query[["site_id", *keys]])
                    others = total - own
                    prefix = f"ctx_{group}_{window}_{subset}"
                    count = others["n"].to_numpy()
                    out.loc[positions, f"{prefix}_n"] = count
                    out.loc[positions, f"{prefix}_fail_smooth"] = (
                        others["fail"].to_numpy() + SMOOTHING_VISITS * prior_fail) / (count + SMOOTHING_VISITS)
                    out.loc[positions, f"{prefix}_load_smooth"] = (
                        others["load"].to_numpy() + SMOOTHING_VISITS * prior_load) / (count + SMOOTHING_VISITS)
    out.index = rows.index
    return out


def build_dynamics_features(rows: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Return 92 numeric within-site dynamics columns, with no target access.

    Codes are opaque tokens. Their repeats/resolution refer to code presence,
    not a claim about severity or actual remediation. Half-life means elapsed
    days since the historical inspection relative to that query's cutoff.
    """
    r, history = _prepare(rows, history)
    h = r.merge(history, on=["forecast_period", "site_id"], how="inner", validate="many_to_many")
    h = h.loc[h["_date"] < h["history_before"]].copy()
    h["_age"] = (h["history_before"] - h["_date"]).dt.days.astype(float)
    h = h.sort_values(["_row", "_date", "inspection_type", "result", "violation_codes"], kind="mergesort")
    idx = pd.RangeIndex(len(r))
    columns = {}

    def put(name, values, fill=None):
        values = pd.to_numeric(values, errors="raise").astype(float).reindex(idx)
        if fill is not None:
            values = values.fillna(fill)
        columns[name] = values

    routine = h.loc[h["inspection_type"] == "routine"]
    last_routine = routine.groupby("_row", sort=False).tail(1).set_index("_row")
    put("dyn_last_routine_codes_blank", last_routine["_blank_codes"])
    put("dyn_last_routine_known_code_count", last_routine["_load"])
    put("dyn_routine_known_code_count_mean", routine.groupby("_row", sort=False)["_load"].mean())
    for code in range(1, 45):
        put(f"dyn_last_routine_code{code:02d}", last_routine["_codes"].map(lambda codes: float(code in codes)), 0)

    for label, part in [("all", h), ("routine", routine)]:
        for half_life in [180, 365]:
            part = part.copy()
            part["_weight"] = np.exp2(-part["_age"] / half_life)
            group = part.groupby("_row", sort=False)
            weight = group["_weight"].sum()
            put(f"dyn_{label}_ew{half_life}_mass", weight, 0)
            for variable, name in [("_fail", "fail_rate"), ("_conditional", "conditional_rate"),
                                   ("_load", "load_mean"), ("_blank_codes", "codes_blank_rate")]:
                weighted = (part[variable] * part["_weight"]).groupby(part["_row"], sort=False).sum()
                put(f"dyn_{label}_ew{half_life}_{name}", weighted / weight)
        group = part.groupby("_row", sort=False)
        pairs = group.tail(2).copy()
        has_pair = group.size() >= 2
        put(f"dyn_{label}_has_pair", has_pair.astype(float), 0)
        pairs = pairs.loc[pairs["_row"].isin(has_pair.loc[has_pair].index)]
        previous = pairs.groupby("_row", sort=False).head(1).set_index("_row")
        last = pairs.groupby("_row", sort=False).tail(1).set_index("_row")
        pair_index = last.index
        new, repeated, resolved, jaccard = [], [], [], []
        for older, newer in zip(previous["_codes"], last["_codes"]):
            a, b = set(older), set(newer)
            new.append(len(b - a))
            repeated.append(len(a & b))
            resolved.append(len(a - b))
            jaccard.append(len(a & b) / len(a | b) if a | b else 1.0)
        for name, values in [("new_codes", new), ("repeated_codes", repeated),
                             ("resolved_codes", resolved), ("codes_jaccard", jaccard)]:
            put(f"dyn_{label}_pair_{name}", pd.Series(values, index=pair_index, dtype=float))
        put(f"dyn_{label}_pair_load_change", last["_load"] - previous["_load"])
        for name, condition in [
                ("fail_to_pass", (previous["result"] == "fail") & (last["result"] == "pass")),
                ("nonfail_to_fail", (previous["result"] != "fail") & (last["result"] == "fail")),
                ("fail_to_fail", (previous["result"] == "fail") & (last["result"] == "fail")),
                ("conditional_to_pass", (previous["result"] == "conditional") & (last["result"] == "pass"))]:
            put(f"dyn_{label}_pair_{name}", condition.astype(float))

    last_any = h.groupby("_row", sort=False).tail(1).set_index("_row")
    put("dyn_last_any_codes_blank", last_any["_blank_codes"])
    put("dyn_days_last_routine_to_last_any", (last_any["_date"] - last_routine["_date"]).dt.days)
    # A routine must exist to define the post-routine segment; otherwise NaN.
    after = h.merge(last_routine[["_date"]].rename(columns={"_date": "_routine_date"}),
                    left_on="_row", right_index=True, how="inner").reset_index(drop=True)
    after = after.loc[after["_date"] > after["_routine_date"]]
    after_group = after.groupby("_row", sort=False)
    for name, values in [("visits", after_group.size()), ("fail_count", after_group["_fail"].sum()),
                         ("load_mean", after_group["_load"].mean())]:
        # Counts are zero when a routine exists and there are no following visits.
        values = values.reindex(last_routine.index)
        if name != "load_mean":
            values = values.fillna(0)
        put(f"dyn_after_routine_{name}", values)

    out = pd.DataFrame(columns, index=idx)
    out.index = rows.index
    if len(out.columns) > 100 or not out.columns.is_unique:
        raise AssertionError("Dynamics feature budget or uniqueness violated")
    return out
