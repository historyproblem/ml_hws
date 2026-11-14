"""Causal snapshot/feature cache for the manually specified experiment suite."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    EXTRA_COLUMNS,
    RFM_COLUMNS,
    canonical_ids,
    clean_transactions,
    load_dataset,
)

ACTIVITY = [f"{kind}_{w}" for w in (7, 14) for kind in ("n", "sum")]
ACTIVITY += [
    f"{kind}_bin_{lo}_{hi}"
    for lo, hi in ((0, 30), (30, 60), (60, 90), (90, 180))
    for kind in ("n", "sum")
]
ACTIVITY += ["frequency_change", "spend_change"]
BASKET = [
    f"{kind}_{w}"
    for w in (30, 180)
    for kind in (
        "orders",
        "products",
        "units",
        "returns",
        "return_amount",
        "amount_per_order",
    )
]
BASKET += [f"region_{i}" for i in range(8)]
CLV_COLUMNS = [
    f"{model}_{window}_{key}"
    for model in ("bg", "pareto")
    for window in ("180", "all")
    for key in ("alive", "count", "check")
]
FEATURES = {
    "F0": list(RFM_COLUMNS),
    "F1": RFM_COLUMNS + ACTIVITY,
    "F2": RFM_COLUMNS + ACTIVITY + BASKET,
    "F3": RFM_COLUMNS + ACTIVITY + BASKET + EXTRA_COLUMNS,
    "F4": RFM_COLUMNS + ACTIVITY + BASKET + CLV_COLUMNS,
}
BASE_COLUMNS = list(dict.fromkeys(FEATURES["F3"]))
CHANNELS = [
    "amount",
    "purchase_days",
    "orders",
    "products",
    "units",
    "return_amount",
    "coverage",
]
SEQUENCES = {
    "S0": (None, [0, 1, 6]),
    "S1": (None, list(range(7))),
    "S0_F0": ("F0", [0, 1, 6]),
    "S1_F2": ("F2", list(range(7))),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def range_sum(dates, values, left, right):
    cumulative = np.r_[0.0, np.cumsum(values)]
    return (
        cumulative[np.searchsorted(dates, right)]
        - cumulative[np.searchsorted(dates, left)]
    )


def distinct_ranges(items, left, right):
    """Answer repeated distinct-item queries once per unique event-index interval."""
    key = left.astype(np.int64) * (len(items) + 1) + right
    unique, inverse = np.unique(key, return_inverse=True)
    counts = np.zeros(len(unique), dtype=float)
    for i, code in enumerate(unique):
        lo, hi = divmod(int(code), len(items) + 1)
        if hi > lo:
            counts[i] = len(np.unique(items[lo:hi]))
    return counts[inverse].reshape(left.shape)


def row_weights(objects, indices, dataset, bootstrap=None):
    rows = objects.iloc[indices]
    w = np.ones(len(rows), dtype=float)
    if dataset in {"D2", "D3"}:
        # Normalise AFTER the temporal purge; future snapshots cannot affect weights.
        counts = rows.groupby(["customer_id", "month"])["Id"].transform("size")
        w /= counts.to_numpy()
    if bootstrap is not None:
        w *= rows.customer_id.map(bootstrap).fillna(0).to_numpy()
    if w.sum() <= 0:
        raise ValueError("No positive training weights")
    # Keep total weight one per customer-month in D2/D3. Rescaling to mean
    # one would weaken Ridge/logistic regularisation as snapshots multiply.
    return w


def split_indices(objects, dataset, cutoff):
    available = objects.target_end.le(pd.Timestamp(cutoff))
    if dataset == "D0":
        available &= objects.official
    elif dataset in {"D1", "D2"}:
        available &= objects.as_of_date.dt.day.isin([1, 8, 15, 22])
    elif dataset != "D3":
        raise ValueError(dataset)
    return np.flatnonzero(available.to_numpy())


def prepare_cache(data_dir, cache_dir, config, identity):
    cache_dir = Path(cache_dir)
    ready = cache_dir / "ready.json"
    if ready.exists():
        old = json.loads(ready.read_text())
        if old["identity"] != identity:
            raise ValueError("Cache definition changed; use a new output directory")
        for name, expected in old["sha256"].items():
            if digest(cache_dir / name) != expected:
                raise ValueError(f"Corrupt cache: {name}")
        return CachedData(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    train, test, history, checks = load_dataset(data_dir)
    raw = pd.read_csv(
        Path(data_dir) / "transactions.csv",
        dtype={c: "string" for c in ("order_id", "customer_id", "item_id", "region")},
    ).drop_duplicates()
    raw["customer_id"] = canonical_ids(raw.customer_id)
    raw["event_ts"] = pd.to_datetime(raw.event_ts, errors="coerce")
    raw["date"] = raw.event_ts.dt.normalize()
    valid = (
        raw[["customer_id", "date", "qty", "unit_price", "item_id", "order_id"]]
        .notna()
        .all(axis=1)
    )
    valid &= np.isfinite(raw.qty) & np.isfinite(raw.unit_price) & raw.unit_price.gt(0)
    valid &= ~raw.item_id.str.fullmatch(r"SVC(?:0[0-9]|1[0-3])", na=True)
    valid &= ~raw.item_id.str.startswith("GFT", na=True)
    returns = raw.loc[
        valid & (raw.qty.lt(0) | raw.order_id.str.startswith("R", na=False))
    ].copy()
    returns["return_amount"] = (returns.qty * returns.unit_price).abs()
    positive = raw.loc[
        valid & raw.qty.gt(0) & ~raw.order_id.str.startswith("R", na=True)
    ].copy()
    positive["amount"] = positive.qty * positive.unit_price
    days, _ = clean_transactions(raw.drop(columns=["date"]))
    start, end = train.as_of_date.min(), test.as_of_date.min() - pd.Timedelta(days=30)
    calendar = pd.date_range(start, end, freq="D")
    if "D3" not in config["datasets"]:
        calendar = (
            calendar[calendar.day.isin([1, 8, 15, 22])]
            if any(d in config["datasets"] for d in ("D1", "D2"))
            else calendar[calendar.day == 1]
        )
    customer_ids = sorted(set(train.customer_id) | set(test.customer_id))
    if config.get("smoke_customers"):
        # Stable subset, independent of targets.
        customer_ids = sorted(
            customer_ids, key=lambda c: hashlib.sha256(c.encode()).hexdigest()
        )[: config["smoke_customers"]]
    frames = []
    calendar_days = calendar.to_numpy().astype("datetime64[D]").astype(np.int64)
    for cid in customer_ids:
        dates = history.customers[cid][0]
        r = np.searchsorted(dates, calendar_days)
        l = np.searchsorted(dates, calendar_days - 180)
        chosen = calendar[r > l]
        frames.append(pd.DataFrame({"customer_id": cid, "as_of_date": chosen}))
    objects = pd.concat(frames, ignore_index=True)
    official = train[["customer_id", "as_of_date", "Id", "spend_30d"]]
    objects = objects.merge(
        official, on=["customer_id", "as_of_date"], how="left", validate="one_to_one"
    )
    objects["official"] = objects.Id.notna()
    expected = train[train.customer_id.isin(customer_ids)]
    if objects.official.sum() != len(expected):
        raise ValueError("Expanded population failed to preserve all official rows")
    objects["Id"] = objects.Id.fillna(
        "snapshot:"
        + objects.customer_id
        + ":"
        + objects.as_of_date.dt.strftime("%Y-%m-%d")
    )
    objects["target_end"] = objects.as_of_date + pd.Timedelta(days=30)
    objects["month"] = objects.as_of_date.dt.strftime("%Y-%m")
    objects["is_test"] = False
    testing = test[test.customer_id.isin(customer_ids)].copy()
    testing["official"], testing["is_test"] = False, True
    testing["month"] = testing.as_of_date.dt.strftime("%Y-%m")
    objects = (
        pd.concat([objects, testing], ignore_index=True)
        .sort_values(["as_of_date", "customer_id"])
        .reset_index(drop=True)
    )
    if not objects.Id.is_unique:
        raise ValueError("Snapshot IDs collide")
    n = len(objects)
    features = np.lib.format.open_memmap(
        cache_dir / "features.npy",
        mode="w+",
        dtype="float32",
        shape=(n, len(BASE_COLUMNS)),
    )
    sequence = np.lib.format.open_memmap(
        cache_dir / "sequence.npy",
        mode="w+",
        dtype="float32",
        shape=(n, 26, len(CHANNELS)),
    )
    features[:] = np.nan
    sequence[:] = 0
    target = np.full(n, np.nan)
    purchase_groups = {
        str(k): g.sort_values("date") for k, g in positive.groupby("customer_id")
    }
    return_groups = {
        str(k): g.sort_values("date") for k, g in returns.groupby("customer_id")
    }
    day_groups = {str(k): g.sort_values("date") for k, g in days.groupby("customer_id")}
    coverage_start = int(
        raw.date.min().to_datetime64().astype("datetime64[D]").astype(np.int64)
    )
    for number, (cid, group) in enumerate(objects.groupby("customer_id", sort=False)):
        indices = group.index.to_numpy()
        t = group.as_of_date.to_numpy().astype("datetime64[D]").astype(np.int64)
        gd, gp = day_groups[cid], purchase_groups[cid]
        d = gd.date.to_numpy().astype("datetime64[D]").astype(np.int64)
        a = gd.amount.to_numpy()
        e = gp.date.to_numpy().astype("datetime64[D]").astype(np.int64)
        r = np.searchsorted(d, t)
        f = {
            "recency": (t - d[np.maximum(r - 1, 0)]).astype(float),
            "tenure": (t - d[0]).astype(float),
        }
        last = a[np.maximum(r - 1, 0)]
        for w in (7, 14, 30, 60, 90, 180):
            f[f"n_{w}"] = r - np.searchsorted(d, t - w)
            f[f"sum_{w}"] = range_sum(d, a, t - w, t)
        f["last_amount"] = last
        f["mean_amount_180"] = f["sum_180"] / np.maximum(f["n_180"], 1)
        f["month_sin"] = np.sin(2 * np.pi * group.as_of_date.dt.month.to_numpy() / 12)
        f["month_cos"] = np.cos(2 * np.pi * group.as_of_date.dt.month.to_numpy() / 12)
        for lo, hi in ((0, 30), (30, 60), (60, 90), (90, 180)):
            f[f"n_bin_{lo}_{hi}"] = np.searchsorted(d, t - lo) - np.searchsorted(
                d, t - hi
            )
            f[f"sum_bin_{lo}_{hi}"] = range_sum(d, a, t - hi, t - lo)
        f["frequency_change"] = np.log1p(f["n_bin_0_30"]) - np.log1p(f["n_bin_30_60"])
        f["spend_change"] = np.log1p(f["sum_bin_0_30"]) - np.log1p(f["sum_bin_30_60"])
        order_ids = gp.order_id.to_numpy(dtype=str)
        ret = return_groups.get(cid)
        rd = (
            np.array([], dtype=np.int64)
            if ret is None
            else ret.date.to_numpy().astype("datetime64[D]").astype(np.int64)
        )
        ra = np.array([]) if ret is None else ret.return_amount.to_numpy()
        items = gp.item_id.to_numpy(dtype=str)
        for w in (30, 180):
            left, right = np.searchsorted(e, t - w), np.searchsorted(e, t)
            f[f"orders_{w}"] = distinct_ranges(order_ids, left, right)
            f[f"products_{w}"] = distinct_ranges(items, left, right)
            f[f"units_{w}"] = range_sum(e, gp.qty.to_numpy(), t - w, t)
            f[f"returns_{w}"] = np.searchsorted(rd, t) - np.searchsorted(rd, t - w)
            f[f"return_amount_{w}"] = range_sum(rd, ra, t - w, t)
            f[f"amount_per_order_{w}"] = f[f"sum_{w}"] / np.maximum(f[f"orders_{w}"], 1)
        latest = np.searchsorted(e, t) - 1
        region = (
            gp.region.fillna("<missing>").to_numpy(dtype=str)
            if "region" in gp
            else np.full(len(gp), "<missing>")
        )
        bins = np.array(
            [int(hashlib.sha256(v.encode()).hexdigest()[:8], 16) % 8 for v in region]
        )
        for k in range(8):
            f[f"region_{k}"] = (bins[np.maximum(latest, 0)] == k).astype(float)
        # Cache non-vectorisable summaries by the actual past slice, not by date.
        summaries = {}
        for j, (lo, hi) in enumerate(zip(np.searchsorted(d, t - 180), r)):
            key = (int(lo), int(hi))
            if key not in summaries:
                aa, dd = a[lo:hi], d[lo:hi]
                gaps = np.diff(dd)
                dow = gd.dow.to_numpy()[lo:hi]
                hours = gd.hour.to_numpy()[lo:hi]
                summaries[key] = {
                    "median_amount_180": np.median(aa),
                    "median_gap": np.median(gaps) if len(gaps) else np.nan,
                    "gap_cv": gaps.std() / gaps.mean() if len(gaps) >= 3 else np.nan,
                    "weekly_share": ((gaps >= 6) & (gaps <= 8)).mean()
                    if len(gaps) >= 3
                    else np.nan,
                    "fortnight_share": ((gaps >= 13) & (gaps <= 15)).mean()
                    if len(gaps) >= 3
                    else np.nan,
                    "monthly_share": ((gaps >= 26) & (gaps <= 32)).mean()
                    if len(gaps) >= 3
                    else np.nan,
                    "weekday_concentration": np.bincount(dow, minlength=7).max()
                    / len(dow)
                    if len(dow) >= 4
                    else np.nan,
                    "clock_concentration": np.bincount(
                        (hours // 3).astype(int), minlength=8
                    ).max()
                    / len(hours)
                    if len(hours) >= 4
                    else np.nan,
                    "median_hour": np.median(hours) if len(hours) >= 4 else np.nan,
                    "amount_cv": aa.std() / aa.mean() if len(aa) >= 4 else np.nan,
                    "amount_iqr_over_median": (
                        np.quantile(aa, 0.75) - np.quantile(aa, 0.25)
                    )
                    / np.median(aa)
                    if len(aa) >= 4
                    else np.nan,
                }
            for name, value in summaries[key].items():
                if name not in f:
                    f[name] = np.full(len(t), np.nan)
                f[name][j] = value
        f["recency_over_gap"] = f["recency"] / f["median_gap"]
        for col, name in enumerate(BASE_COLUMNS):
            values = np.asarray(f[name], dtype=float)
            if name not in {
                "month_sin",
                "month_cos",
                "frequency_change",
                "spend_change",
            } and not name.startswith("region_"):
                values = np.log1p(values)
            features[indices, col] = values
        # Oldest -> newest weekly bins, all right edges at or before as_of_date.
        right = t[:, None] - np.arange(25, -1, -1)[None, :] * 7
        left = right - 7
        sequence[indices, :, 0] = np.log1p(range_sum(d, a, left, right))
        sequence[indices, :, 1] = np.log1p(
            np.searchsorted(d, right) - np.searchsorted(d, left)
        )
        sequence[indices, :, 2] = np.log1p(
            distinct_ranges(
                order_ids, np.searchsorted(e, left), np.searchsorted(e, right)
            )
        )
        sequence[indices, :, 3] = np.log1p(
            distinct_ranges(items, np.searchsorted(e, left), np.searchsorted(e, right))
        )
        sequence[indices, :, 4] = np.log1p(range_sum(e, gp.qty.to_numpy(), left, right))
        sequence[indices, :, 5] = np.log1p(range_sum(rd, ra, left, right))
        sequence[indices, :, 6] = np.clip(
            (right - np.maximum(left, coverage_start)) / 7, 0, 1
        )
        target[indices] = np.round(range_sum(d, a, t, t + 30), 2)
        if number % 500 == 0:
            print(f"Features: {number}/{len(customer_ids)} customers", flush=True)
    known = objects.official.to_numpy()
    if (
        np.max(np.abs(target[known] - objects.loc[known, "spend_30d"].to_numpy()))
        > 0.011
    ):
        raise ValueError("Fast feature builder failed official target reconstruction")
    objects["spend_30d"] = target
    objects.loc[objects.is_test, "spend_30d"] = np.nan
    # Independent comparison with the original feature implementation.
    sample = objects.loc[known].iloc[
        np.linspace(0, known.sum() - 1, min(128, known.sum()), dtype=int)
    ]
    reference = history.build_features(sample)
    for column in RFM_COLUMNS:
        v = reference[column].to_numpy()
        if column not in {"month_sin", "month_cos"}:
            v = np.log1p(v)
        np.testing.assert_allclose(
            features[sample.index, BASE_COLUMNS.index(column)], v, rtol=2e-6, atol=2e-6
        )
    features.flush()
    sequence.flush()
    objects.to_csv(cache_dir / "objects.csv.gz", index=False)
    files = ["objects.csv.gz", "features.npy", "sequence.npy"]
    atomic_json(
        ready,
        {
            "identity": identity,
            "checks": checks,
            "rows": n,
            "columns": BASE_COLUMNS,
            "channels": CHANNELS,
            "sha256": {name: digest(cache_dir / name) for name in files},
        },
    )
    return CachedData(cache_dir)


class CachedData:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.objects = pd.read_csv(
            self.directory / "objects.csv.gz",
            dtype={"Id": "string", "customer_id": "string"},
            parse_dates=["as_of_date", "target_end"],
        )
        self.features = np.load(self.directory / "features.npy", mmap_mode="r")
        self.sequence = np.load(self.directory / "sequence.npy", mmap_mode="r")

    def matrix(self, representation, indices):
        if representation in SEQUENCES:
            tab, channels = SEQUENCES[representation]
            seq = np.asarray(self.sequence[indices][:, :, channels]).reshape(
                len(indices), -1
            )
            x = (
                np.empty((len(indices), 0), dtype=np.float32)
                if tab is None
                else self.matrix(tab, indices)[0]
            )
            return np.c_[x, seq], (26, len(channels))
        columns = FEATURES[representation]
        base = [BASE_COLUMNS.index(c) for c in columns if c in BASE_COLUMNS]
        x = np.asarray(self.features[np.ix_(indices, base)])
        if representation == "F4":
            if not (self.directory / "clv.npy").exists():
                raise ValueError("F4 requires the causal CLV feature stage")
            extra = np.load(self.directory / "clv.npy", mmap_mode="r")
            x = np.c_[x, extra[indices]]
        return x, None
