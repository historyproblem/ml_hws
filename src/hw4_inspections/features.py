"""Snapshot-specific, label-free features for future inspection outcomes.

Every visit is matched on (forecast_period, site_id), then checked against the
individual row's strict history_before cutoff. Repeated snapshots are never
pooled. Codes are opaque tokens; their sanitary meaning is not assumed.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

STATIC_CATEGORIES = ("facility_type", "risk_level", "area_id")
RESULTS = ("pass", "conditional", "fail")
INSPECTION_TYPES = (
    "routine", "complaint", "licensing", "other", "routine_reinspection",
    "complaint_reinspection", "licensing_reinspection",
)
MISSING = "<none>"


@lru_cache(maxsize=100_000)
def _codes(value: str) -> tuple[int, ...]:
    """One binary occurrence per code per visit, including repeated tokens."""
    return tuple(sorted({int(s) for s in value.split() if s.isdigit() and 1 <= int(s) <= 44}))


def build_features(rows: pd.DataFrame, history: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return features in input row/index order and names of categorical columns.

    ``rows.target`` is ignored. No target-derived statistics, identifiers, or
    future snapshot records enter the output. Missing history yields zero counts
    and missing rates/ages. Dates must be valid; malformed dates fail explicitly.
    """
    required = {"forecast_period", "site_id", "history_before", *STATIC_CATEGORIES}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Missing row columns: {sorted(missing)}")
    hrequired = {"forecast_period", "site_id", "inspection_week", "inspection_type", "result", "violation_codes"}
    missing = hrequired - set(history.columns)
    if missing:
        raise ValueError(f"Missing history columns: {sorted(missing)}")

    n = len(rows)
    frame = pd.DataFrame(index=pd.RangeIndex(n))
    cats: list[str] = []
    # Position, rather than input index, supports arbitrary/nonunique indices.
    for name in STATIC_CATEGORIES:
        frame[name] = rows[name].fillna(MISSING).astype(str).to_numpy()
        cats.append(name)
    period = pd.to_datetime(rows["forecast_period"], errors="raise").reset_index(drop=True)
    cutoff = pd.to_datetime(rows["history_before"], errors="raise").reset_index(drop=True)
    frame["forecast_year"] = period.dt.year.to_numpy()
    frame["forecast_quarter"] = period.dt.quarter.to_numpy()
    frame["cutoff_gap_days"] = (period - cutoff).dt.days.to_numpy()

    keys = rows[["forecast_period", "site_id", "history_before"]].reset_index(drop=True).copy()
    keys["_row"] = np.arange(n)
    h = keys.merge(history[list(hrequired)], on=["forecast_period", "site_id"], how="inner", validate="many_to_many")
    h["_date"] = pd.to_datetime(h["inspection_week"], errors="raise")
    h["_cutoff"] = pd.to_datetime(h["history_before"], errors="raise")
    h = h.loc[h["_date"] < h["_cutoff"]].copy()
    h["_age"] = (h["_cutoff"] - h["_date"]).dt.days.astype(float)
    for name in ("inspection_type", "result", "violation_codes"):
        h[name] = h[name].fillna("").astype(str)
    # Stable tie handling also makes the function invariant to input row order.
    h = h.sort_values(["_row", "_date", "inspection_type", "result", "violation_codes"], kind="mergesort")
    h["_n_codes"] = h["violation_codes"].map(lambda x: len(_codes(x))).astype(float)
    for result in RESULTS:
        h[f"_{result}"] = (h["result"] == result).astype(float)
    h["_reinspection"] = h["inspection_type"].str.endswith("_reinspection").astype(float)
    columns: dict[str, pd.Series | np.ndarray] = {}

    def put(name: str, series: pd.Series, fill: float | str | None = None) -> None:
        series = series.reindex(frame.index)
        if fill is not None:
            series = series.fillna(fill)
        columns[name] = series

    def summarize(part: pd.DataFrame, prefix: str) -> None:
        group = part.groupby("_row", sort=False)
        put(f"{prefix}_n_visits", group.size(), 0.0)
        for result in RESULTS:
            put(f"{prefix}_{result}_count", group[f"_{result}"].sum(), 0.0)
            put(f"{prefix}_{result}_rate", group[f"_{result}"].mean())
        put(f"{prefix}_reinspection_count", group["_reinspection"].sum(), 0.0)
        put(f"{prefix}_violation_mean", group["_n_codes"].mean())
        put(f"{prefix}_violation_max", group["_n_codes"].max())
        put(f"{prefix}_violation_total", group["_n_codes"].sum(), 0.0)

    summarize(h, "history")
    for days in (90, 180, 365):
        summarize(h.loc[h["_age"] <= days], f"recent{days}")

    group = h.groupby("_row", sort=False)
    put("oldest_visit_age_days", group["_age"].max())
    put("history_span_days", group["_age"].max() - group["_age"].min())
    h["_gap_days"] = group["_date"].diff().dt.days
    group = h.groupby("_row", sort=False)
    put("visit_gap_mean_days", group["_gap_days"].mean())
    put("visit_gap_std_days", group["_gap_days"].std())
    put("visit_gap_min_days", group["_gap_days"].min())
    for result in RESULTS:
        part = h.loc[h["result"] == result].groupby("_row", sort=False)
        put(f"last_{result}_age_days", part["_age"].min())

    for inspection_type in INSPECTION_TYPES:
        part = h.loc[h["inspection_type"] == inspection_type]
        tgroup = part.groupby("_row", sort=False)
        put(f"type_{inspection_type}_count", tgroup.size(), 0.0)
        put(f"type_{inspection_type}_fail_rate", tgroup["_fail"].mean())
        put(f"type_{inspection_type}_last_age_days", tgroup["_age"].min())

    for lag in (1, 2, 3):
        last = group.tail(lag).groupby("_row", sort=False).head(1)
        # A shorter history must not repeat its first record into all lags.
        enough = group.size().loc[lambda s: s >= lag].index
        last = last.loc[last["_row"].isin(enough)].set_index("_row")
        for name in ("result", "inspection_type"):
            col = f"lag{lag}_{name}"
            put(col, last[name], MISSING)
            cats.append(col)
        put(f"lag{lag}_age_days", last["_age"])
        put(f"lag{lag}_violation_count", last["_n_codes"])
    last = group.tail(1).set_index("_row")
    put("last_gap_days", last["_gap_days"])
    put("violation_change_last_vs_mean", last["_n_codes"] - group["_n_codes"].mean())
    last3 = group.tail(3).groupby("_row", sort=False)
    put("last3_fail_rate", last3["_fail"].mean())
    put("last3_violation_mean", last3["_n_codes"].mean())
    routine = h.loc[h["inspection_type"] == "routine"].groupby("_row", sort=False).tail(1).set_index("_row")
    put("last_routine_result", routine["result"], MISSING)
    cats.append("last_routine_result")
    put("last_routine_age_days", routine["_age"])
    put("last_routine_violation_count", routine["_n_codes"])

    # Codes remain opaque features. Snapshot membership and cutoff are already
    # applied before any code aggregation, so recent test visits cannot leak.
    matrix = np.zeros((len(h), 44), dtype=np.int8)
    for i, value in enumerate(h["violation_codes"]):
        codes = _codes(value)
        if codes:
            matrix[i, np.asarray(codes) - 1] = 1
    code_names = [f"code{code:02d}" for code in range(1, 45)]
    codes_frame = pd.DataFrame(matrix, columns=code_names, index=h.index)
    codes_frame["_row"] = h["_row"]
    totals = codes_frame.groupby("_row", sort=False)[code_names].sum()
    recent = codes_frame.loc[h["_age"] <= 180].groupby("_row", sort=False)[code_names].sum()
    last_codes = codes_frame.loc[group.tail(1).index].set_index("_row")
    for name in code_names:
        put(f"{name}_count", totals[name], 0.0)
        put(f"{name}_ever", (totals[name] > 0).astype(float), 0.0)
        put(f"{name}_recent180_count", recent[name], 0.0)
        put(f"{name}_last", last_codes[name], 0.0)

    frame = pd.concat([frame, pd.DataFrame(columns, index=frame.index)], axis=1)
    frame.index = rows.index
    if not frame.columns.is_unique:
        raise AssertionError("Duplicate feature names")
    if len(frame.columns) > 300:
        raise AssertionError("Initial feature budget exceeded")
    return frame, cats
