"""Transaction-only features. Every window is left-closed and right-open."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from spend_baselines.data import canonical_ids
from spend_baselines.suite_data import (
    BASE_COLUMNS, CachedData, atomic_json, digest, prepare_cache,
)

REGULARITY = ["gap_mean", "gap_median", "gap_cv", "gap_last", "gap_count",
              "recency_gap_mean", "recency_gap_last"]
ITEMS = [f"{name}_{w}" for w in (30, 90, 180) for name in
         ("item_unique", "item_repeat", "item_hhi", "item_entropy", "item_new")]
ITEMS += ["diversity_change_30_90", "diversity_change_90_180"]
SEGMENT = ["order_max", "order_median", "order_lines", "large_order_share"]
SEASON = ["year_amount", "year_orders", "year_days", "year_coverage",
          "year_ratio_60", "year_ratio_90", "season_share", "global_season"]
CALENDAR = ["year_calendar_amount", "year_calendar_orders", "year_calendar_days"]
SEASON += CALENDAR
HISTORY = ["label_count", "label_mean", "label_zero", "label_ewm"]
BLOCKS = {"regularity": REGULARITY, "items": ITEMS, "season": SEASON,
          "segment": SEGMENT, "history": HISTORY}
COLUMNS = REGULARITY + ITEMS + SEGMENT + SEASON


def calendar_month_features(objects, lines, start):
    """Exact prior-year calendar month, distinct from the aligned 30d window."""
    monthly = lines.assign(month=lines.event_ts.dt.to_period("M"),
                           day=lines.event_ts.dt.normalize()).groupby(["customer_id", "month"]).agg(
                               amount=("amount", "sum"), orders=("order_id", "nunique"),
                               days=("day", "nunique"))
    previous = objects.as_of_date.dt.to_period("M") - 12
    keys = pd.MultiIndex.from_arrays([objects.customer_id, previous])
    values = np.log1p(monthly.reindex(keys, fill_value=0).to_numpy(float))
    covered = ((previous.dt.start_time >= start.normalize()) &
               ((previous + 1).dt.start_time <= objects.as_of_date)).to_numpy()
    values[~covered] = np.nan
    return values.astype(np.float32)


def positive_lines(data_dir):
    raw = pd.read_csv(Path(data_dir) / "transactions.csv", dtype={
        c: "string" for c in ("customer_id", "order_id", "item_id")}).drop_duplicates()
    raw["customer_id"] = canonical_ids(raw.customer_id)
    raw["event_ts"] = pd.to_datetime(raw.event_ts, errors="coerce")
    valid = raw[["customer_id", "order_id", "item_id", "event_ts", "qty",
                 "unit_price"]].notna().all(axis=1)
    valid &= np.isfinite(raw.qty) & np.isfinite(raw.unit_price)
    valid &= raw.qty.gt(0) & raw.unit_price.gt(0)
    valid &= ~raw.order_id.str.startswith("R", na=True)
    valid &= ~raw.item_id.str.fullmatch(r"SVC(?:0[0-9]|1[0-3])", na=True)
    valid &= ~raw.item_id.str.startswith("GFT", na=True)
    lines = raw.loc[valid].copy()
    lines["amount"] = lines.qty * lines.unit_price
    return lines.sort_values("event_ts"), raw.event_ts.min()


def global_season(lines, dates, start, prior_days=60):
    """Prior-year target-month rate, shrunk towards the prior-year daily rate.

    Both intervals end before the row's cutoff; missing prior-year coverage is
    neutral, with an explicit coverage indicator in the customer feature block.
    """
    daily = lines.groupby(lines.event_ts.dt.normalize()).amount.sum()
    result = {}
    for t in pd.to_datetime(dates).unique():
        t = pd.Timestamp(t)
        left = t.replace(day=1) - pd.DateOffset(years=1)
        right = left + pd.DateOffset(months=1)
        if left < start.normalize() or right > t:
            result[t] = 0.0
            continue
        anchor = max(start.normalize(), right - pd.Timedelta(days=365))
        base = daily.loc[(daily.index >= anchor) & (daily.index < right)].sum()
        base /= max((right - anchor).days, 1)
        amount = daily.loc[(daily.index >= left) & (daily.index < right)].sum()
        rate = (amount + prior_days * base) / ((right - left).days + prior_days)
        result[t] = float(np.log(max(rate, 1e-12) / max(base, 1e-12)))
    return result


def transaction_features(objects, lines, start, prior_days=60):
    out = np.full((len(objects), len(COLUMNS)), np.nan, dtype=np.float32)
    next_event = np.full(len(objects), np.inf)
    global_values = global_season(lines, objects.as_of_date, start, prior_days)
    groups = {str(k): g for k, g in lines.groupby("customer_id", sort=False)}
    epoch = pd.Timestamp("1970-01-01")
    for number, (cid, rows) in enumerate(objects.groupby("customer_id", sort=False)):
        g = groups.get(str(cid))
        if g is None:
            continue
        ts = g.event_ts.to_numpy(dtype="datetime64[ns]")
        item = g.item_id.to_numpy(str)
        amount = g.amount.to_numpy(float)
        order = g.order_id.to_numpy(str)
        first_item = g.groupby("item_id").event_ts.min().to_dict()
        # Summaries are reused only when the exact past interval is identical.
        summaries = {}
        for idx, row in rows.iterrows():
            t = row.as_of_date
            hi = np.searchsorted(ts, t.to_datetime64())
            if hi < len(ts):
                next_event[idx] = (pd.Timestamp(ts[hi]) - epoch).total_seconds() / 86400
            lo = np.searchsorted(ts, (t - pd.Timedelta(days=180)).to_datetime64())
            key = (int(lo), int(hi))
            if key not in summaries:
                past = g.iloc[lo:hi]
                days = past.event_ts.dt.normalize().drop_duplicates().sort_values()
                gaps = days.diff().dt.total_seconds().dropna().to_numpy() / 86400
                f = {"gap_count": np.log1p(len(gaps))}
                for name, value in (("mean", np.mean(gaps) if len(gaps) else np.nan),
                                    ("median", np.median(gaps) if len(gaps) else np.nan),
                                    ("last", gaps[-1] if len(gaps) else np.nan),
                                    ("cv", gaps.std() / gaps.mean() if len(gaps) > 1 else np.nan)):
                    f[f"gap_{name}"] = np.log1p(value)
                orders = past.groupby("order_id").agg(amount=("amount", "sum"),
                                                       lines=("amount", "size"))
                f.update(order_max=np.log1p(orders.amount.max()),
                         order_median=np.log1p(orders.amount.median()),
                         order_lines=np.log1p(orders.lines.mean()),
                         # A fixed monetary threshold, never a future quantile.
                         large_order_share=float(orders.amount.ge(1000).mean()))
                summaries[key] = f
            f = summaries[key].copy()
            last_day = pd.Timestamp(ts[hi - 1]).normalize() if hi else t
            recency = (t - last_day).total_seconds() / 86400
            for name in ("mean", "last"):
                gap = np.expm1(f[f"gap_{name}"])
                f[f"recency_gap_{name}"] = np.log1p(recency / gap) if gap > 0 else np.nan
            for w in (30, 90, 180):
                left = t - pd.Timedelta(days=w)
                a = np.searchsorted(ts, left.to_datetime64())
                cache_key = ("items", int(a), int(hi), w, left)
                if cache_key not in summaries:
                    its, inv, counts = np.unique(item[a:hi], return_inverse=True,
                                                 return_counts=True)
                    shares = np.bincount(inv, weights=amount[a:hi], minlength=len(its))
                    shares = shares / max(shares.sum(), 1e-12)
                    summaries[cache_key] = {
                        f"item_unique_{w}": np.log1p(len(its)),
                        f"item_repeat_{w}": float(np.maximum(counts - 1, 0).sum() / max(counts.sum(), 1)),
                        f"item_hhi_{w}": float(np.sum(shares ** 2)),
                        f"item_entropy_{w}": float(-np.sum(shares * np.log(np.maximum(shares, 1e-30)))),
                        f"item_new_{w}": float(np.mean([first_item[x] >= left for x in its])) if len(its) else 0.0,
                    }
                f.update(summaries[cache_key])
            f["diversity_change_30_90"] = f["item_unique_30"] - f["item_unique_90"]
            f["diversity_change_90_180"] = f["item_unique_90"] - f["item_unique_180"]
            previous = t - pd.DateOffset(years=1)
            right = previous + pd.Timedelta(days=30)
            covered = previous >= start.normalize() and right <= t
            year = g.iloc[np.searchsorted(ts, previous.to_datetime64()):
                          np.searchsorted(ts, right.to_datetime64())]
            f.update(year_coverage=float(covered),
                     year_amount=np.log1p(year.amount.sum()) if covered else np.nan,
                     year_orders=np.log1p(year.order_id.nunique()) if covered else np.nan,
                     year_days=np.log1p(year.event_ts.dt.normalize().nunique()) if covered else np.nan,
                     global_season=global_values[t])
            for w in (60, 90):
                old_left = previous - pd.Timedelta(days=w)
                old_lo, old_hi = np.searchsorted(ts, [old_left.to_datetime64(), previous.to_datetime64()])
                recent_lo = np.searchsorted(ts, (t - pd.Timedelta(days=w)).to_datetime64())
                f[f"year_ratio_{w}"] = (np.log1p(amount[recent_lo:hi].sum()) -
                                         np.log1p(amount[old_lo:old_hi].sum())) if old_left >= start.normalize() else np.nan
            yearly_lo = np.searchsorted(ts, (right - pd.Timedelta(days=365)).to_datetime64())
            yearly_hi = np.searchsorted(ts, right.to_datetime64())
            f["season_share"] = float(year.amount.sum() / max(amount[yearly_lo:yearly_hi].sum(), 1)) if covered else np.nan
            out[idx, : -len(CALENDAR)] = [f[c] for c in COLUMNS if c not in CALENDAR]
        if number % 500 == 0:
            print(f"Behaviour features: {number}/{len(groups)} customers", flush=True)
    out[:, -len(CALENDAR):] = calendar_month_features(objects, lines, start)
    if np.isinf(out).any():
        raise ValueError("Infinite behaviour feature")
    return out, next_event


def load_base(data_dir, directory, external=None, smoke_customers=None):
    inputs = {n: digest(Path(data_dir) / n) for n in
              ("transactions.csv", "train.csv", "test.csv", "sample_submission.csv")}
    if external:
        external = Path(external).resolve()
        ready = json.loads((external / "ready.json").read_text())
        ident = ready["identity"]
        if ident.get("input_sha256") != inputs or ready["columns"] != BASE_COLUMNS:
            raise ValueError("Base cache inputs/columns differ")
        for name in ("data.py", "suite_data.py"):
            source = Path(__file__).parents[1] / "spend_baselines" / name
            if ident["source_sha256"].get(name) != digest(source):
                raise ValueError(f"Base cache source differs: {name}")
        if "D2" not in ident["config"]["datasets"] or ident["config"].get("smoke_customers") != smoke_customers:
            raise ValueError("Base cache population differs")
        for n, sha in ready["sha256"].items():
            if digest(external / n) != sha:
                raise ValueError(f"Corrupt base cache: {n}")
        return CachedData(external)
    cfg = {"datasets": ["D2"]}
    if smoke_customers:
        cfg["smoke_customers"] = smoke_customers
    return prepare_cache(data_dir, directory, cfg, {"input_sha256": inputs, "config": cfg})


class BehaviorData:
    def __init__(self, base, extra, next_event):
        self.base, self.objects, self.extra, self.next_event = base, base.objects, extra, next_event
        self.history = None

    def matrix(self, indices, blocks=()):
        parts = [self.base.matrix("F2", indices)[0]]
        for block in blocks:
            if block == "history":
                if self.history is None:
                    raise ValueError("Causal label history not prepared")
                parts.append(self.history[indices])
            else:
                parts.append(self.extra[np.ix_(indices, [COLUMNS.index(c) for c in BLOCKS[block]])])
        return np.column_stack(parts)


def prepare(data_dir, out_dir, config, external=None):
    try:
        base = load_base(data_dir, Path(out_dir) / "base_cache", external,
                         config.get("smoke_customers"))
    except (ValueError, KeyError) as error:
        if not external:
            raise
        print(f"Cannot reuse old base cache ({error}); rebuilding in the new output directory", flush=True)
        base = load_base(data_dir, Path(out_dir) / "base_cache", None,
                         config.get("smoke_customers"))
    directory = Path(out_dir) / "behavior_cache"
    directory.mkdir(parents=True, exist_ok=True)
    ready = directory / "ready.json"
    if ready.exists():
        state = json.loads(ready.read_text())
        for name, sha in state["sha256"].items():
            if digest(directory / name) != sha:
                raise ValueError(f"Corrupt behaviour cache: {name}")
    else:
        lines, start = positive_lines(data_dir)
        features, events = transaction_features(base.objects, lines, start,
                                                config["season_prior_days"])
        np.save(directory / "features.npy", features)
        np.save(directory / "next_event.npy", events)
        atomic_json(ready, {"columns": COLUMNS, "base_objects_sha256": digest(base.directory / "objects.csv.gz"),
                            "sha256": {n: digest(directory / n) for n in ("features.npy", "next_event.npy")}})
    state = json.loads(ready.read_text())
    if state["columns"] != COLUMNS:
        raise ValueError("Behaviour feature schema changed; use a fresh output directory")
    if state["base_objects_sha256"] != digest(base.directory / "objects.csv.gz"):
        raise ValueError("Behaviour cache population differs")
    return BehaviorData(base, np.load(directory / "features.npy", mmap_mode="r"),
                        np.load(directory / "next_event.npy"))
