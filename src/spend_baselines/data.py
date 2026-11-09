"""Cleaning and features with a separate history cutoff for every example."""
from pathlib import Path
import numpy as np
import pandas as pd

RFM_COLUMNS = [
    "recency", "tenure", "n_30", "n_60", "n_90", "n_180",
    "sum_30", "sum_60", "sum_90", "sum_180", "last_amount",
    "mean_amount_180", "median_amount_180", "month_sin", "month_cos",
]
EXTRA_COLUMNS = [
    "median_gap", "gap_cv", "weekly_share", "fortnight_share",
    "monthly_share", "recency_over_gap", "weekday_concentration",
    "clock_concentration", "median_hour", "amount_cv", "amount_iqr_over_median",
]
LOG_COLUMNS = set(RFM_COLUMNS + ["median_gap", "recency_over_gap"]) - {"month_sin", "month_cos"}


def canonical_ids(values):
    return values.astype("string").str.replace(r"\.0$", "", regex=True)


def read_objects(path):
    frame = pd.read_csv(path, dtype={"Id": "string", "customer_id": "string"},
                        parse_dates=["as_of_date", "target_end"])
    frame["customer_id"] = canonical_ids(frame.customer_id)
    if frame[["Id", "customer_id", "as_of_date", "target_end"]].isna().any().any():
        raise ValueError(f"Missing prediction key/date in {path}")
    if not frame.Id.is_unique:
        raise ValueError(f"Duplicate prediction Id in {path}")
    if not frame.as_of_date.eq(frame.as_of_date.dt.normalize()).all():
        raise ValueError("Customer-day features require midnight forecast dates")
    if not frame.target_end.sub(frame.as_of_date).eq(pd.Timedelta(days=30)).all():
        raise ValueError("Every target window must contain exactly 30 days")
    if "spend_30d" in frame and (not np.isfinite(frame.spend_30d).all() or frame.spend_30d.lt(0).any()):
        raise ValueError("Training targets must be finite and nonnegative")
    return frame


def clean_transactions(raw):
    """Positive product purchases, matching the competition's target definition."""
    required = ["event_ts", "order_id", "customer_id", "item_id", "qty", "unit_price"]
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing transaction columns: {sorted(missing)}")
    frame = raw.drop_duplicates().copy()
    frame["customer_id"] = canonical_ids(frame.customer_id)
    frame["order_id"] = frame.order_id.astype("string")
    frame["item_id"] = frame.item_id.astype("string")
    frame["event_ts"] = pd.to_datetime(frame.event_ts, errors="coerce")
    for c in ["qty", "unit_price"]:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    valid = frame[required].notna().all(axis=1)
    valid &= np.isfinite(frame.qty) & np.isfinite(frame.unit_price)
    valid &= frame.qty.gt(0) & frame.unit_price.gt(0)
    valid &= ~frame.order_id.str.startswith("R", na=True)
    service = frame.item_id.str.fullmatch(r"SVC(?:0[0-9]|1[0-3])", na=True)
    service |= frame.item_id.str.startswith("GFT", na=True)
    frame = frame.loc[valid & ~service].copy()
    frame["amount"] = frame.qty * frame.unit_price
    frame["date"] = frame.event_ts.dt.normalize()
    days = frame.groupby(["customer_id", "date"], sort=True).agg(
        amount=("amount", "sum"), event_ts=("event_ts", "min"),
        n_orders=("order_id", "nunique"), n_products=("item_id", "nunique"),
    ).reset_index()
    days["hour"] = days.event_ts.dt.hour + days.event_ts.dt.minute / 60
    days["dow"] = days.date.dt.dayofweek
    if not np.isclose(days.amount.sum(), frame.amount.sum()):
        raise ValueError("Customer-day aggregation changed total purchase amount")
    stats = {"raw_rows": len(raw), "exact_duplicates": int(raw.duplicated().sum()),
             "clean_rows": len(frame), "customers": int(days.customer_id.nunique()),
             "purchase_days": len(days)}
    return days, stats


class PurchaseHistory:
    def __init__(self, days):
        self.customers = {}
        for cid, group in days.groupby("customer_id"):
            group = group.sort_values("date")
            dates = group.date.to_numpy().astype("datetime64[D]").astype(np.int64)
            a = group.amount.to_numpy(dtype=float)
            self.customers[str(cid)] = (dates, a, np.r_[0.0, np.cumsum(a)],
                                       group.hour.to_numpy(), group.dow.to_numpy(dtype=int))

    def build_features(self, objects):
        rows = []
        empty = (np.array([], dtype=int), np.array([]), np.array([0.0]), np.array([]), np.array([], dtype=int))
        for row in objects.itertuples():
            dates, amounts, cumulative, hours, dows = self.customers.get(str(row.customer_id), empty)
            t = int(np.datetime64(row.as_of_date, "D").astype(np.int64))
            right = int(np.searchsorted(dates, t, side="left"))
            f = {"recency": float(t - dates[right - 1]) if right else 181.0,
                 "tenure": float(t - dates[0]) if right else 0.0}
            for window in [30, 60, 90, 180]:
                left = int(np.searchsorted(dates, t - window, side="left"))
                f[f"n_{window}"] = right - left
                f[f"sum_{window}"] = float(cumulative[right] - cumulative[left])
            left = int(np.searchsorted(dates, t - 180, side="left"))
            a, d, h, dw = amounts[left:right], dates[left:right], hours[left:right], dows[left:right]
            gap = np.diff(d)
            f.update({"last_amount": float(amounts[right - 1]) if right else 0.0,
                      "mean_amount_180": float(a.mean()) if len(a) else 0.0,
                      "median_amount_180": float(np.median(a)) if len(a) else 0.0,
                      "month_sin": float(np.sin(2 * np.pi * row.as_of_date.month / 12)),
                      "month_cos": float(np.cos(2 * np.pi * row.as_of_date.month / 12))})
            f["median_gap"] = float(np.median(gap)) if len(gap) else np.nan
            f["gap_cv"] = float(gap.std() / gap.mean()) if len(gap) >= 3 else np.nan
            f["weekly_share"] = float(((gap >= 6) & (gap <= 8)).mean()) if len(gap) >= 3 else np.nan
            f["fortnight_share"] = float(((gap >= 13) & (gap <= 15)).mean()) if len(gap) >= 3 else np.nan
            f["monthly_share"] = float(((gap >= 26) & (gap <= 32)).mean()) if len(gap) >= 3 else np.nan
            f["recency_over_gap"] = f["recency"] / f["median_gap"] if len(gap) else np.nan
            f["weekday_concentration"] = float(np.bincount(dw, minlength=7).max() / len(dw)) if len(dw) >= 4 else np.nan
            f["clock_concentration"] = float(np.bincount((h // 3).astype(int), minlength=8).max() / len(h)) if len(h) >= 4 else np.nan
            f["median_hour"] = float(np.median(h)) if len(h) >= 4 else np.nan
            f["amount_cv"] = float(a.std() / a.mean()) if len(a) >= 4 else np.nan
            f["amount_iqr_over_median"] = float((np.quantile(a, .75) - np.quantile(a, .25)) / np.median(a)) if len(a) >= 4 else np.nan
            lag_sums = []
            for k in range(6):
                lo = int(np.searchsorted(dates, t - (k + 1) * 30, side="left"))
                hi = int(np.searchsorted(dates, t - k * 30, side="left"))
                lag_sums.append(float(cumulative[hi] - cumulative[lo]))
            f["logmean_prediction"] = float(np.log1p(lag_sums).mean())
            rows.append(f)
        return pd.DataFrame(rows, index=objects.index)

    def reconstruct_targets(self, objects):
        out = np.zeros(len(objects))
        for i, row in enumerate(objects.itertuples()):
            h = self.customers.get(str(row.customer_id))
            if h is None:
                continue
            dates, _, cumulative, _, _ = h
            start = int(np.datetime64(row.as_of_date, "D").astype(np.int64))
            end = int(np.datetime64(row.target_end, "D").astype(np.int64))
            lo, hi = np.searchsorted(dates, [start, end], side="left")
            out[i] = np.round(cumulative[hi] - cumulative[lo], 2)
        return out


def feature_matrix(features, group="rfm"):
    if group not in {"rfm", "extended"}:
        raise ValueError(f"Unknown feature group: {group}")
    columns = RFM_COLUMNS + (EXTRA_COLUMNS if group == "extended" else [])
    x = features[columns].to_numpy(dtype=float).copy()
    for i, column in enumerate(columns):
        if column in LOG_COLUMNS:
            x[:, i] = np.log1p(x[:, i])
    if np.isinf(x).any():
        raise ValueError("Infinite feature value")
    return x


def load_dataset(data_dir):
    data_dir = Path(data_dir)
    raw = pd.read_csv(data_dir / "transactions.csv", dtype={
        "order_id": "string", "customer_id": "string", "item_id": "string",
        "item_name": "string", "region": "string"})
    days, stats = clean_transactions(raw)
    train = read_objects(data_dir / "train.csv")
    test = read_objects(data_dir / "test.csv")
    if set(train.Id) & set(test.Id):
        raise ValueError("Train/test Id overlap")
    history = PurchaseHistory(days)
    rebuilt = history.reconstruct_targets(train)
    error = np.abs(rebuilt - train.spend_30d.to_numpy())
    stats["train_targets_checked"] = len(train)
    stats["max_target_error"] = float(error.max())
    if error.max() > .011:
        raise ValueError(f"Target reconstruction failed: max error {error.max():.4f}")
    return train, test, history, stats
