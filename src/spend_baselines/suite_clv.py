"""BG/NBD, Pareto/NBD and Gamma-Gamma, with causal per-date estimation.

Likelihoods follow Fader/Hardie notes 004, 009 and 025. Pareto's dropout
integral uses log-time Gauss-Legendre quadrature and is checked at double order.
The first observed ORDER anchors exposure; it is not claimed to be acquisition.
"""

import json
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, hyp2f1, logsumexp

from .data import canonical_ids
from .suite_data import CLV_COLUMNS, atomic_json, digest


@lru_cache(maxsize=8)
def quadrature(order):
    return np.polynomial.legendre.leggauss(order)


def likelihood(parameters, x, tx, age, kind, order=64):
    r, alpha, a, b = np.asarray(parameters)
    constant = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    if kind == "bg":
        alive = (
            constant + betaln(a, b + x) - betaln(a, b) - (r + x) * np.log(alpha + age)
        )
        dead = (
            constant
            + betaln(a + 1, b + np.maximum(x - 1, 0))
            - betaln(a, b)
            - (r + x) * np.log(alpha + tx)
        )
        dead = np.where(x > 0, dead, -np.inf)
    elif kind == "pareto":
        alive = (
            constant
            + a * np.log(b)
            - (r + x) * np.log(alpha + age)
            - a * np.log(b + age)
        )
        nodes, weights = quadrature(order)
        # u=log(1+t/c), rather than t: resolves sharp posterior mass near t=0.
        scale = min(alpha, b)
        lo, hi = np.log1p(tx / scale), np.log1p(age / scale)
        width = (hi - lo) / 2
        u = lo[:, None] + width[:, None] * (nodes[None, :] + 1)
        t = scale * np.expm1(u)
        with np.errstate(divide="ignore"):
            integration = (
                np.log(weights)[None, :]
                + np.log(width)[:, None]
                + np.log(scale)
                + u
                - (r + x[:, None]) * np.log(alpha + t)
                - (a + 1) * np.log(b + t)
            )
        dead = constant + a * np.log(b) + np.log(a) + logsumexp(integration, axis=1)
    else:
        raise ValueError(kind)
    return np.logaddexp(alive, dead), alive


def gg_likelihood(parameters, n, mean):
    p, q, gamma = parameters
    return (
        gammaln(p * n + q)
        - gammaln(p * n)
        - gammaln(q)
        + q * np.log(gamma)
        + (p * n - 1) * np.log(mean)
        + (p * n) * np.log(n)
        - (p * n + q) * np.log(gamma + n * mean)
    )


def fit_positive(objective, start, maxiter, penalty, lower=None):
    lower = np.zeros(len(start)) if lower is None else np.asarray(lower)

    def loss(log_parameters):
        values = objective(lower + np.exp(log_parameters))
        return -float(np.mean(values)) + penalty * float(np.sum(log_parameters**2))

    result = minimize(
        loss,
        np.log(np.asarray(start) - lower),
        method="L-BFGS-B",
        bounds=[(-7, 10)] * len(start),
        options={"maxiter": maxiter, "ftol": 1e-10},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"CLV optimizer did not converge: {result.message}")
    return lower + np.exp(result.x), {
        "objective": float(result.fun),
        "iterations": int(result.nit),
    }


class CustomerModel:
    def __init__(self, kind, settings):
        self.kind, self.settings = kind, settings

    def fit(self, summary):
        self.summary = summary.copy()
        x, tx, age = [summary[c].to_numpy(float) for c in ("x", "tx", "age")]
        if len(summary) < 4:
            raise ValueError("CLV needs at least four observed customer histories")
        self.parameters, self.fit_info = fit_positive(
            lambda p: likelihood(
                p, x, tx, age, self.kind, self.settings["quadrature_order"]
            )[0],
            [
                1,
                max(float(age.mean()), 1),
                1,
                3 if self.kind == "bg" else max(float(age.mean()), 1),
            ],
            self.settings["maxiter"],
            self.settings["penalty"],
        )
        if self.kind == "pareto":
            a = likelihood(
                self.parameters,
                x,
                tx,
                age,
                self.kind,
                self.settings["quadrature_order"],
            )[0]
            b = likelihood(
                self.parameters,
                x,
                tx,
                age,
                self.kind,
                2 * self.settings["quadrature_order"],
            )[0]
            self.fit_info["quadrature_max_log_error"] = float(np.max(np.abs(a - b)))
            if (
                self.fit_info["quadrature_max_log_error"]
                > self.settings["quadrature_tolerance"]
            ):
                raise ArithmeticError("Pareto likelihood quadrature has not converged")
        self.money, self.money_info = fit_positive(
            lambda p: gg_likelihood(
                p, summary.n.to_numpy(float), summary["mean"].to_numpy(float)
            ),
            [2, 3, max(float(summary["mean"].median()), 0.1)],
            self.settings["maxiter"],
            self.settings["penalty"],
            lower=[0, 1, 0],
        )
        # q>1 is necessary for a finite population prior monetary mean.
        if self.money[1] <= 1:
            raise ArithmeticError("Gamma-Gamma q<=1: infinite population monetary mean")
        return self

    def values(self, summary, horizon=30):
        x, tx, age = [summary[c].to_numpy(float) for c in ("x", "tx", "age")]
        ll, la = likelihood(
            self.parameters, x, tx, age, self.kind, self.settings["quadrature_order"]
        )
        alive = np.exp(la - ll)
        r, alpha, a, b = self.parameters
        nodes, weights = quadrature(64)
        t = (nodes + 1) * horizon / 2
        if self.kind == "pareto":
            exposure = (
                np.sum(
                    ((b + age[:, None]) / (b + age[:, None] + t)) ** a * weights, axis=1
                )
                * horizon
                / 2
            )
            count = alive * (r + x) / (alpha + age) * exposure
        else:
            # Integrate E[lambda exp(-lambda*p*t)]. The beta expectation is
            # a hypergeometric function, avoiding overflow of Jacobi weights
            # for highly concentrated dropout priors.
            rate = (r + x) / (alpha + age)
            integrand = hyp2f1(
                r + x[:, None] + 1,
                a,
                a + b + x[:, None],
                -t[None, :] / (alpha + age[:, None]),
            )
            count = alive * rate * (integrand @ weights) * horizon / 2
        if not np.isfinite(count).all() or (count < 0).any():
            raise FloatingPointError("Invalid posterior transaction expectation")
        p, q, gamma = self.money
        check = (
            p
            * (gamma + summary.n.to_numpy() * summary["mean"].to_numpy())
            / (q + p * summary.n.to_numpy() - 1)
        )
        return np.c_[alive, count, check]

    def predict(self, summary, seed=42, draws=4096, horizon=30):
        rng = np.random.default_rng(seed)
        values = self.values(summary, horizon)
        z = np.empty(len(summary))
        money_mean = np.empty(len(summary))
        errors = np.empty(len(summary))
        r, alpha, a, b = self.parameters
        p, q, gamma = self.money
        for i, row in enumerate(summary.itertuples()):
            lam = rng.gamma(r + row.x, 1 / (alpha + row.age), draws)
            if self.kind == "bg":
                probability = np.clip(rng.beta(a, b + row.x, draws), 1e-12, 1)
                count = np.minimum(
                    rng.poisson(lam * horizon), rng.geometric(probability)
                )
            else:
                death_rate = rng.gamma(a, 1 / (b + row.age), draws)
                lifetime = rng.exponential(1 / death_rate)
                count = rng.poisson(lam * np.minimum(lifetime, horizon))
            count[rng.random(draws) > values[i, 0]] = 0
            rate = rng.gamma(q + p * row.n, 1 / (gamma + row.n * row.mean), draws)
            spend = np.zeros(draws)
            keep = count > 0
            spend[keep] = rng.gamma(p * count[keep], 1 / rate[keep])
            log = np.log1p(spend)
            z[i], errors[i] = log.mean(), log.std(ddof=1) / np.sqrt(draws)
            # Analytic mean is a diagnostic; RMSLE uses E[log1p(spend)].
            money_mean[i] = values[i, 1] * values[i, 2]
        return z, money_mean, errors

    def state(self):
        return {
            "kind": self.kind,
            "parameters": self.parameters.tolist(),
            "money": self.money.tolist(),
            "fit": self.fit_info,
            "money_fit": self.money_info,
            "exposure_origin": "first observed order within selected history window; acquisition unobserved",
            "monetary_observations": "all positive orders in selected window, including first",
        }


class OrderHistory:
    def __init__(self, data_dir):
        raw = pd.read_csv(
            data_dir / "transactions.csv",
            dtype={c: "string" for c in ("customer_id", "order_id", "item_id")},
        ).drop_duplicates()
        raw["customer_id"] = canonical_ids(raw.customer_id)
        raw["event_ts"] = pd.to_datetime(raw.event_ts, errors="coerce")
        ok = (
            raw[["customer_id", "order_id", "item_id", "event_ts", "qty", "unit_price"]]
            .notna()
            .all(axis=1)
        )
        ok &= (
            np.isfinite(raw.qty)
            & np.isfinite(raw.unit_price)
            & raw.qty.gt(0)
            & raw.unit_price.gt(0)
        )
        ok &= ~raw.order_id.str.startswith("R", na=True) & ~raw.item_id.str.startswith(
            "GFT", na=True
        )
        ok &= ~raw.item_id.str.fullmatch(r"SVC(?:0[0-9]|1[0-3])", na=True)
        raw = raw.loc[ok].copy()
        raw["amount"] = raw.qty * raw.unit_price
        raw["date"] = raw.event_ts.dt.normalize()
        # Split before aggregation so a future line sharing an order_id cannot
        # retroactively change the amount of a historical order.
        orders = (
            raw.groupby(["customer_id", "order_id", "date"])
            .agg(time=("event_ts", "min"), amount=("amount", "sum"))
            .reset_index()
        )
        self.groups = {
            cid: g.sort_values("time") for cid, g in orders.groupby("customer_id")
        }

    def summary(self, date, window, customers=None):
        date = pd.Timestamp(date)
        result = []
        allowed = None if customers is None else set(customers)
        for cid, g in self.groups.items():
            if allowed is not None and cid not in allowed:
                continue
            h = g[g.time.lt(date)]
            if window == "180":
                h = h[h.time.ge(date - pd.Timedelta(days=180))]
            if h.empty:
                continue
            h = (
                h.groupby("order_id", sort=False)
                .agg(time=("time", "min"), amount=("amount", "sum"))
                .sort_values("time")
            )
            times = h.time
            result.append(
                {
                    "customer_id": cid,
                    "x": len(h) - 1,
                    "tx": (times.iloc[-1] - times.iloc[0]).total_seconds() / 86400,
                    "age": (date - times.iloc[0]).total_seconds() / 86400,
                    "n": len(h),
                    "mean": float(h.amount.mean()),
                }
            )
        return pd.DataFrame(result).set_index("customer_id", drop=False)


def prepare_clv(cache, data_dir, settings, identity):
    """F4 parameters for each snapshot date use transactions strictly before it."""
    dest = cache.directory
    marker = dest / "clv_ready.json"
    if marker.exists():
        info = json.loads(marker.read_text())
        if info["identity"] != identity or digest(dest / "clv.npy") != info["sha256"]:
            raise ValueError("CLV cache mismatch")
        return
    history = OrderHistory(data_dir)
    extra = np.lib.format.open_memmap(
        dest / "clv.npy",
        mode="w+",
        dtype="float32",
        shape=(len(cache.objects), len(CLV_COLUMNS)),
    )
    for date, group in cache.objects.groupby("as_of_date"):
        shard = dest / "clv_dates" / str(date.date())
        shard.mkdir(parents=True, exist_ok=True)
        if (shard / "ready.json").exists():
            saved = json.loads((shard / "ready.json").read_text())
            if (
                saved["identity"] != identity
                or digest(shard / "features.npy") != saved["sha256"]
            ):
                raise ValueError("Corrupt causal CLV date shard")
            extra[group.index] = np.load(shard / "features.npy")
            continue
        local = np.empty((len(group), len(CLV_COLUMNS)), dtype=np.float32)
        states = {}
        for wi, window in enumerate(("180", "all")):
            summary = history.summary(date, window, cache.objects.customer_id.unique())
            for mi, kind in enumerate(("bg", "pareto")):
                model = CustomerModel(kind, settings).fit(summary)
                values = model.values(summary.loc[group.customer_id])
                offset = (mi * 2 + wi) * 3
                local[:, offset : offset + 3] = np.log1p(values)
                states[f"{kind}_{window}"] = model.state()
        if not np.isfinite(local).all():
            raise FloatingPointError("Non-finite CLV features")
        np.save(shard / "features.npy", local)
        atomic_json(
            shard / "ready.json",
            {
                "identity": identity,
                "sha256": digest(shard / "features.npy"),
                "states": states,
            },
        )
        extra[group.index] = local
        print(f"Causal F4: {date.date()}", flush=True)
    extra.flush()
    atomic_json(marker, {"identity": identity, "sha256": digest(dest / "clv.npy")})
