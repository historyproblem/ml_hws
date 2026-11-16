"""Causal market means for an ordinary regression with date-centred responses.

The market is the population active in the 180 days before an anchor.  Its
response is mean(log1p(next-30-day spend)), including the non-buyers.  Historical
responses may only enter a forecast after their complete 30-day window closes.
"""
import numpy as np
import pandas as pd


def _day(value):
    return np.datetime64(pd.Timestamp(value).date(), "D").astype(np.int64)


def cohort_mean(history, anchor, *, backward=False):
    """A market statistic, with exactly the same active-180 eligibility rule."""
    t = _day(anchor)
    result = []
    for dates, _, cumulative, *_ in history.customers.values():
        lo, hi = np.searchsorted(dates, [t - 180, t], side="left")
        if hi == lo:
            continue
        left, right = (t - 30, t) if backward else (t, t + 30)
        a, b = np.searchsorted(dates, [left, right], side="left")
        result.append(np.log1p(max(cumulative[b] - cumulative[a], 0)))
    return float(np.mean(result)) if result else 0.0


def market_forecasts(objects, history, history_start):
    """One known market forecast per row, without reading object target columns.

    If a fully covered prior-year cohort exists, use its mean plus half the
    average year-on-year change in the last three completed comparable windows.
    Otherwise use the last three completed market means, or backward activity.
    The coefficient 0.5 is fixed in advance; no leaderboard fitting is involved.
    """
    start = pd.Timestamp(history_start).normalize()
    cutoffs = sorted(pd.to_datetime(objects.as_of_date).unique())
    first = (start + pd.Timedelta(days=180)).ceil("D")
    anchors = pd.date_range(first.replace(day=1), max(cutoffs), freq="MS")
    anchors = anchors[anchors >= first]
    means = {t: cohort_mean(history, t) for t in anchors
             if t + pd.Timedelta(days=30) <= max(cutoffs)}
    records = []
    for raw in cutoffs:
        t = pd.Timestamp(raw)
        # This restriction is essential even though the table is precomputed.
        closed = {a: v for a, v in means.items() if a + pd.Timedelta(days=30) <= t}
        recent = sorted(closed)[-3:]
        previous = t - pd.DateOffset(years=1)
        pairs = [closed[a] - closed[a - pd.DateOffset(years=1)] for a in recent
                 if a - pd.DateOffset(years=1) in closed]
        trend = float(np.mean(pairs)) if pairs else 0.0
        year = closed.get(previous)
        recent_mean = float(np.mean([closed[a] for a in recent])) if recent else cohort_mean(history, t, backward=True)
        forecast = year + 0.5 * trend if year is not None else recent_mean
        records.append({"as_of_date": t, "market_forecast": forecast,
                        "previous_year_mean": year, "recent_mean": recent_mean,
                        "trend": trend, "trend_pairs": len(pairs),
                        "max_used_target_end": max((a + pd.Timedelta(days=30) for a in closed), default=pd.NaT)})
    table = pd.DataFrame(records).set_index("as_of_date")
    assert (table.max_used_target_end.dropna() <= table.max_used_target_end.dropna().index).all()
    values = table.market_forecast.reindex(pd.to_datetime(objects.as_of_date)).to_numpy(float)
    assert np.isfinite(values).all()
    return values, table


def centre_features_by_date(x, dates):
    """Only unlabelled features from the current forecast population are used."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    for date in pd.unique(dates):
        ix = np.flatnonzero(np.asarray(dates) == date)
        block = x[ix]
        known = np.isfinite(block)
        mean = np.divide(np.where(known, block, 0).sum(0), known.sum(0),
                         out=np.zeros(x.shape[1]), where=known.sum(0) > 0)
        out[ix] = np.where(known, block - mean, 0)
    assert np.isfinite(out).all()
    return out


def centre_targets_by_date(z, dates):
    """Call on the purged TRAIN rows only; these are responses, not features."""
    result = np.asarray(z, dtype=float).copy()
    for date in pd.unique(dates):
        ix = np.flatnonzero(np.asarray(dates) == date)
        result[ix] -= result[ix].mean()
    return result
