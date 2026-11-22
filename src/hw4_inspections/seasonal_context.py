"""Conservative first-routine seasonal context from other sites only.

One contribution is the first possible routine inspection of a site in a past
calendar quarter with the same season as the forecast. A weekly cell belongs
unambiguously to a quarter only when all seven days lie in it. If the earliest
cell overlapping a quarter straddles its boundary, has conflicting records, or
has an unknown outcome, exclude that site-quarter; never substitute a later
known record. A later boundary cell cannot invalidate a definite earlier first.

A historical quarter must end no later than the row cutoff, precede the
forecast, and have its start-containing Monday at/after the earliest available
OTHER-site week in that exact snapshot. This last rule conservatively excludes
left-truncated quarters and ensures focal-history changes cannot alter its own
coverage eligibility. Coverage is an observed-snapshot envelope, not evidence
that an absent inspection or unrecorded violation did not occur.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import MISSING, _codes

GROUPS = {
    "area": ["area_id"],
    "risk": ["risk_level"],
    "area_risk": ["area_id", "risk_level"],
    "facility_risk": ["facility_type", "risk_level"],
}
METRICS = ("n_siteyears", "fail_smooth", "recorded_load_smooth")
SMOOTHING_SITEYEARS = 20.0
CATEGORIES = ("area_id", "risk_level", "facility_type")
KNOWN_OUTCOMES = frozenset({"pass", "conditional", "fail"})
COLUMNS = tuple(f"seas_{group}_{metric}" for group in GROUPS for metric in METRICS)


def _prepare(rows, history):
    row_columns = ["forecast_period", "site_id", "history_before", *CATEGORIES]
    history_columns = ["forecast_period", "site_id", "inspection_week", "inspection_type",
                       "result", "violation_codes", *CATEGORIES]
    for frame, columns, name in [(rows, row_columns, "rows"), (history, history_columns, "history")]:
        if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
            raise ValueError(f"{name} must be a DataFrame with unique columns")
        missing = set(columns) - set(frame)
        if missing:
            raise ValueError(f"Missing {name} columns: {sorted(missing)}")
        if frame.site_id.isna().any():
            raise ValueError(f"{name} site_id must be nonmissing")
    r = rows[row_columns].reset_index(drop=True).copy()
    r["_row"] = np.arange(len(r))
    h = history[history_columns].copy()
    for frame in [r, h]:
        frame["forecast_period"] = pd.to_datetime(frame.forecast_period, errors="raise")
        if frame.forecast_period.isna().any() or frame.forecast_period.dt.tz is not None:
            raise ValueError("Forecast periods must be valid timezone-free dates")
        if not frame.forecast_period.dt.is_quarter_start.all():
            raise ValueError("Forecast periods must be calendar quarter starts")
        if not frame.forecast_period.eq(frame.forecast_period.dt.normalize()).all():
            raise ValueError("Forecast periods must be exact dates")
        for name in CATEGORIES:
            frame[name] = frame[name].fillna(MISSING).astype(str)
    r["history_before"] = pd.to_datetime(r.history_before, errors="raise")
    h["_week"] = pd.to_datetime(h.inspection_week, errors="raise")
    if (r.history_before.isna().any() or r.history_before.dt.tz is not None
            or h._week.isna().any() or h._week.dt.tz is not None):
        raise ValueError("Cutoffs and inspection weeks must be valid timezone-free dates")
    if (r.history_before > r.forecast_period).any():
        raise ValueError("History cutoffs cannot follow the forecast quarter start")
    for name in ["inspection_type", "result", "violation_codes"]:
        h[name] = h[name].fillna("").astype(str)
    return r, h


def _summary(frame, keys):
    return frame.groupby(keys, dropna=False, sort=False).agg(
        n=("_fail", "size"), fail=("_fail", "sum"), load=("_load", "sum"))


def _lookup(table, keys):
    index = (pd.Index(keys.iloc[:, 0], name=keys.columns[0]) if len(keys.columns) == 1
             else pd.MultiIndex.from_frame(keys))
    return table.reindex(index).fillna(0).reset_index(drop=True).to_numpy(dtype=float)


def _first_routines(snapshot, start, end):
    routine = snapshot.loc[snapshot.inspection_type.eq("routine")].copy()
    routine = routine.loc[(routine._week < end) & (routine._week + pd.Timedelta(days=6) >= start)]
    routine["_load"] = routine.violation_codes.map(lambda value: float(len(_codes(value))))
    # Exact duplicates and alternative code spellings with the same recorded
    # count give the same contribution. Conflicting first-cell values do not.
    identity = ["site_id", "_week", "result", "_load", *CATEGORIES]
    routine = routine.drop_duplicates(identity)
    earliest = routine.groupby("site_id", sort=False)._week.transform("min")
    first = routine.loc[routine._week.eq(earliest)].copy()
    unique_first = first.groupby("site_id", sort=False).site_id.transform("size").eq(1)
    unambiguous = first._week.ge(start) & (first._week + pd.Timedelta(days=6) < end)
    first = first.loc[unique_first & unambiguous & first.result.isin(KNOWN_OUTCOMES)].copy()
    first["_fail"] = first.result.eq("fail").astype(float)
    return first


def build_seasonal_context_features(rows: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Return twelve numeric columns, in the exact input row/index order.

    Four groups receive other-site/year counts, smoothed first-routine failure
    rates, and smoothed recorded code counts. One site may contribute once in
    each eligible prior year. Natural blank codes contribute zero *recorded*
    codes; this does not assert no actual violations. Empty groups use a fixed
    20-site/year prior from all eligible OTHER sites. With no other-site prior,
    the count is zero and both means are NaN. No target column is read.
    """
    r, h = _prepare(rows, history)
    out = pd.DataFrame({name: np.zeros(len(r), dtype=float) if name.endswith("_n_siteyears")
                        else np.full(len(r), np.nan) for name in COLUMNS})
    for (forecast, cutoff), query in r.groupby(["forecast_period", "history_before"], sort=False):
        query = query.reset_index(drop=True)
        positions = query._row.to_numpy()
        snapshot = h.loc[h.forecast_period.eq(forecast) & h._week.lt(cutoff)].copy()
        if snapshot.empty:
            continue
        if (not snapshot._week.dt.dayofweek.eq(0).all()
                or not snapshot._week.eq(snapshot._week.dt.normalize()).all()):
            raise ValueError("Inspection weeks must be exact Monday dates")
        # Only the first two distinct sites are needed to remove each query's
        # own earliest week from the coverage envelope.
        site_min = snapshot.groupby("site_id", sort=False)._week.min().sort_values(kind="stable")
        earliest_week, earliest_site = site_min.iloc[0], site_min.index[0]
        coverage = pd.Series(earliest_week, index=query.index, dtype="datetime64[ns]")
        own_earliest = query.site_id.eq(earliest_site)
        coverage.loc[own_earliest] = site_min.iloc[1] if len(site_min) > 1 else pd.NaT
        global_totals = np.zeros((len(query), 3), dtype=float)
        grouped_totals = {name: np.zeros((len(query), 3), dtype=float) for name in GROUPS}
        periods = pd.period_range(earliest_week.to_period("Q"), forecast.to_period("Q") - 1, freq="Q")
        for quarter in periods:
            start, end = quarter.start_time, (quarter + 1).start_time
            if quarter.quarter != forecast.quarter or end > cutoff or start >= forecast:
                continue
            monday = start - pd.Timedelta(days=start.dayofweek)
            eligible = coverage.notna().to_numpy() & coverage.le(monday).to_numpy()
            if not eligible.any():
                continue
            first = _first_routines(snapshot, start, end)
            if first.empty:
                continue
            totals = np.array([len(first), first._fail.sum(), first._load.sum()], dtype=float)
            own = _lookup(_summary(first, ["site_id"]), query[["site_id"]])
            global_totals += (totals - own) * eligible[:, None]
            for name, keys in GROUPS.items():
                totals = _lookup(_summary(first, keys), query[keys])
                own = _lookup(_summary(first, ["site_id", *keys]), query[["site_id", *keys]])
                grouped_totals[name] += (totals - own) * eligible[:, None]
        prior_fail = np.divide(global_totals[:, 1], global_totals[:, 0],
                               out=np.full(len(query), np.nan), where=global_totals[:, 0] > 0)
        prior_load = np.divide(global_totals[:, 2], global_totals[:, 0],
                               out=np.full(len(query), np.nan), where=global_totals[:, 0] > 0)
        for name, totals in grouped_totals.items():
            count = totals[:, 0]
            out.loc[positions, f"seas_{name}_n_siteyears"] = count
            out.loc[positions, f"seas_{name}_fail_smooth"] = (
                totals[:, 1] + SMOOTHING_SITEYEARS * prior_fail) / (count + SMOOTHING_SITEYEARS)
            out.loc[positions, f"seas_{name}_recorded_load_smooth"] = (
                totals[:, 2] + SMOOTHING_SITEYEARS * prior_load) / (count + SMOOTHING_SITEYEARS)
    out.index = rows.index
    return out
