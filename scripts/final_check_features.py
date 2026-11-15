"""Extra feature blocks for explicit final checks. Original suite stays unchanged."""
import numpy as np
import pandas as pd

SEASON_COLUMNS = [f"forecast_month_share_{m:02d}" for m in range(1, 13)] + [
    "yearago_available", "yearago_log_spend30", "yearago_log_days30",
    "yearago_baseline_available", "yearago_log_mean30_from90", "yearago_season_log_ratio",
    "recent_vs_yearago_baseline_log_ratio", "prior_november_available",
    "prior_november_log_spend", "prior_november_log_days", "prior_november_baseline_available",
    "prior_november_season_log_ratio",
]
TARGET_COLUMNS = [f"target_log_lag_{i:02d}" for i in range(1, 7)] + [
    "target_lags_available", "target_history_mean_log", "target_history_std_log",
    "target_history_zero_fraction", "target_history_positive_mean_log",
    "target_prior_available", "target_history_mean_log_shrunk",
    "target_history_positive_rate_shrunk", "target_history_positive_log_shrunk",
]
BLOCK_COLUMNS = {"none": [], "season": SEASON_COLUMNS, "target_history": TARGET_COLUMNS}


def _ordinal(values):
    return np.asarray(values, dtype="datetime64[D]").astype(np.int64)


def causal_target_prior(official, dates):
    """Only labels whose 30-day windows have already finished enter the prior."""
    known = official.loc[official.spend_30d.notna()].sort_values("target_end")
    ends = _ordinal(known.target_end)
    z = np.log1p(known.spend_30d.to_numpy(float))
    positive = known.spend_30d.to_numpy(float) > 0
    sz = np.r_[0.0, np.cumsum(z)]
    np_ = np.r_[0, np.cumsum(positive)]
    sp = np.r_[0.0, np.cumsum(z * positive)]
    count = np.searchsorted(ends, _ordinal(dates), side="right")
    mean = np.divide(sz[count], count, out=np.full(len(count), np.nan), where=count > 0)
    rate = np.divide(np_[count], count, out=np.full(len(count), np.nan), where=count > 0)
    positive_mean = np.divide(sp[count], np_[count], out=np.full(len(count), np.nan), where=np_[count] > 0)
    return mean, rate, positive_mean, count > 0


def build_extra_features(objects, days, official, *, history_start, prior_strength=3.0):
    if prior_strength <= 0:
        raise ValueError("Prior strength must be positive")
    objects = objects.reset_index(drop=True)
    start = int(_ordinal([history_start])[0])
    n = len(objects)
    season = np.full((n, len(SEASON_COLUMNS)), np.nan, dtype=np.float32)
    targets = np.full((n, len(TARGET_COLUMNS)), np.nan, dtype=np.float32)
    si = {name: i for i, name in enumerate(SEASON_COLUMNS)}
    ti = {name: i for i, name in enumerate(TARGET_COLUMNS)}
    unique_dates = pd.DatetimeIndex(sorted(objects.as_of_date.unique()))
    prior = causal_target_prior(official, unique_dates)
    date_meta = {}
    for i, date in enumerate(unique_dates):
        t = int(_ordinal([date])[0])
        yearago = int(_ordinal([date - pd.DateOffset(years=1)])[0])
        november = pd.Timestamp(year=date.year - 1, month=11, day=1)
        nov = int(_ordinal([november])[0])
        nov_end = int(_ordinal([november + pd.DateOffset(months=1)])[0])
        shares = np.bincount(pd.date_range(date, periods=30).month, minlength=13)[1:] / 30.0
        date_meta[t] = (yearago, nov, nov_end, shares,
                        prior[0][i], prior[1][i], prior[2][i], bool(prior[3][i]))
    histories = {}
    for cid, group in days.groupby("customer_id"):
        group = group.sort_values("date")
        d = _ordinal(group.date)
        histories[str(cid)] = (d, np.r_[0.0, np.cumsum(group.amount.to_numpy(float))])

    for cid, group in objects.groupby("customer_id", sort=False):
        rows = group.index.to_numpy()
        t = _ordinal(group.as_of_date)
        dates, cumulative = histories.get(str(cid), (np.array([], dtype=int), np.array([0.0])))

        def window(left, right):
            lo = np.searchsorted(dates, left, side="left")
            hi = np.searchsorted(dates, right, side="left")
            available = (left >= start) & (right <= t)
            amount = cumulative[hi] - cumulative[lo]
            return np.where(available, amount, np.nan), np.where(available, hi - lo, np.nan), available

        meta = [date_meta[int(v)] for v in t]
        ya = np.array([v[0] for v in meta])
        nov = np.array([v[1] for v in meta])
        nov_end = np.array([v[2] for v in meta])
        season[rows, :12] = np.stack([v[3] for v in meta])
        ysum, yndays, yavail = window(ya, ya + 30)
        ybase, _, ybavail = window(ya - 90, ya)
        recent, _, _ = window(t - 90, t)
        nsum, ndays, navail = window(nov, nov_end)
        nbase, _, nbavail = window(nov - 90, nov)
        values = {
            "yearago_available": yavail, "yearago_log_spend30": np.log1p(ysum),
            "yearago_log_days30": np.log1p(yndays),
            "yearago_baseline_available": ybavail,
            "yearago_log_mean30_from90": np.log1p(ybase / 3),
            "yearago_season_log_ratio": np.log1p(ysum) - np.log1p(ybase / 3),
            "recent_vs_yearago_baseline_log_ratio": np.log1p(recent / 3) - np.log1p(ybase / 3),
            "prior_november_available": navail, "prior_november_log_spend": np.log1p(nsum),
            "prior_november_log_days": np.log1p(ndays),
            "prior_november_baseline_available": nbavail,
            "prior_november_season_log_ratio": np.log1p(nsum) - np.log1p(nbase / 3),
        }
        for name, value in values.items():
            season[rows, si[name]] = value
        lags = []
        for k in range(1, 7):
            amount, _, _ = window(t - 30 * k, t - 30 * (k - 1))
            lag = np.log1p(amount)
            lags.append(lag)
            targets[rows, ti[f"target_log_lag_{k:02d}"]] = lag
        lag = np.stack(lags, axis=1)
        available = np.isfinite(lag)
        count = available.sum(axis=1)
        total = np.nansum(lag, axis=1)
        positive_count = (lag > 0).sum(axis=1)
        mean = np.divide(total, count, out=np.full(len(t), np.nan), where=count > 0)
        variance = np.nansum((lag - mean[:, None]) ** 2, axis=1) / np.maximum(count, 1)
        pm = np.array([v[4] for v in meta])
        pp = np.array([v[5] for v in meta])
        ppos = np.array([v[6] for v in meta])
        pavail = np.array([v[7] for v in meta])
        values = {
            "target_lags_available": count,
            "target_history_mean_log": mean,
            "target_history_std_log": np.where(count > 0, np.sqrt(variance), np.nan),
            "target_history_zero_fraction": np.where(count > 0, 1 - positive_count / np.maximum(count, 1), np.nan),
            "target_history_positive_mean_log": np.divide(total, positive_count, out=np.full(len(t), np.nan), where=positive_count > 0),
            "target_prior_available": pavail,
            "target_history_mean_log_shrunk": (total + prior_strength * pm) / (count + prior_strength),
            "target_history_positive_rate_shrunk": (positive_count + prior_strength * pp) / (count + prior_strength),
            "target_history_positive_log_shrunk": (total + prior_strength * ppos) / (positive_count + prior_strength),
        }
        for name, value in values.items():
            targets[rows, ti[name]] = value
    if np.isinf(season).any() or np.isinf(targets).any():
        raise ValueError("Infinite additional feature value")
    return {"season": season, "target_history": targets}
