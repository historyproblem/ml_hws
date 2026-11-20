"""Latest routine state of other sites, rather than visit-weighted context."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .extensions import CATEGORIES, _lookup, _prepare, _summary
from .features import RESULTS


STATE_GROUPS = {
    "area": ["area_id"],
    "risk": ["risk_level"],
    "area_risk": ["area_id", "risk_level"],
    "area_facility": ["area_id", "facility_type"],
}
SMOOTHING_SITES = 20.0


def _latest_routine(part):
    """Keep the latest actual routine, then exclude unknown latest outcomes."""
    if part.site_id.isna().any():
        raise ValueError("Routine states require nonmissing site_id")
    # Weekly source keys are unique in the official data. Exact copied records
    # are harmless; conflicting records for one site/week cannot define a state.
    relevant = ["site_id", "_date", "result", "violation_codes", *CATEGORIES]
    part = part.drop_duplicates(relevant)
    if part.duplicated(["site_id", "_date"]).any():
        raise ValueError("Conflicting routine states for one site and week")
    latest = part.sort_values(["_date", "site_id"], kind="stable").drop_duplicates("site_id", keep="last")
    return latest.loc[latest.result.isin(RESULTS)]


def build_site_state_features(rows: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Return 24 label-free latest-state fields in input row/index order.

    Each window uses the exact forecast snapshot and the row's strict cutoff,
    then takes one latest routine per site. If its outcome is unknown the site
    is excluded, with no fallback to an older known state. The focal site is
    excluded from both group statistics and the global 20-site smoothing prior.
    Empty group support is zero and its rates are missing, including when a
    global background exists. Codes remain opaque tokens; load is their count.
    """
    query_rows, prepared = _prepare(rows, history, context=True)
    names = [f"state_{group}_{window}_{metric}"
             for window in ["all", "365d"] for group in STATE_GROUPS
             for metric in ["n_sites", "fail_smooth", "load_smooth"]]
    output = pd.DataFrame(np.nan, index=pd.RangeIndex(len(rows)), columns=names)
    for (period, cutoff), query in query_rows.groupby(["forecast_period", "history_before"], sort=False):
        snapshot = prepared.loc[prepared.forecast_period.eq(period) & prepared._date.lt(cutoff)
                                & prepared.inspection_type.eq("routine")]
        query = query.reset_index(drop=True)
        positions = query._row.to_numpy()
        for window in ["all", "365d"]:
            part = snapshot if window == "all" else snapshot.loc[(cutoff - snapshot._date).dt.days.le(365)]
            states = _latest_routine(part)
            global_count = len(states)
            global_totals = states[["_fail", "_load"]].sum().to_numpy()
            own = _lookup(_summary(states, ["site_id"]), query[["site_id"]])
            other_n = global_count - own.n.to_numpy()
            prior_fail = np.divide(global_totals[0] - own.fail.to_numpy(), other_n,
                                   out=np.full(len(query), np.nan), where=other_n > 0)
            prior_load = np.divide(global_totals[1] - own.load.to_numpy(), other_n,
                                   out=np.full(len(query), np.nan), where=other_n > 0)
            for group, keys in STATE_GROUPS.items():
                total = _lookup(_summary(states, keys), query[keys])
                focal = _lookup(_summary(states, ["site_id", *keys]), query[["site_id", *keys]])
                others = total - focal
                count = others.n.to_numpy()
                prefix = f"state_{group}_{window}"
                output.loc[positions, f"{prefix}_n_sites"] = count
                for metric, prior, column in [("fail_smooth", prior_fail, "fail"),
                                               ("load_smooth", prior_load, "load")]:
                    value = (others[column].to_numpy() + SMOOTHING_SITES * prior) / (count + SMOOTHING_SITES)
                    value[count == 0] = np.nan
                    output.loc[positions, f"{prefix}_{metric}"] = value
    output.index = rows.index
    return output
