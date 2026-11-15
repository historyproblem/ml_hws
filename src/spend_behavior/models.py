"""Penalised statistical models; predictions are always in log1p units."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

from spend_baselines.suite_data import row_weights, split_indices
from spend_baselines.suite_models import Preprocessor, Tabular

from .data import COLUMNS, HISTORY


def ridge(x, y, w, alpha=10, task="direct"):
    return Tabular("linear", task, {"ridge_alpha": alpha, "logistic_C": 1.0}).fit(x, y, w)


def inner_date(objects, cutoff):
    available = objects.official & objects.target_end.le(pd.Timestamp(cutoff))
    date = objects.loc[available, "as_of_date"].max()
    if pd.isna(date):
        raise ValueError("No fully observed inner validation date")
    return date


def validation_rows(objects, cutoff):
    return np.flatnonzero((objects.official & objects.as_of_date.eq(pd.Timestamp(cutoff))).to_numpy())


def causal_histories(objects, residuals, lambdas, prior=5, half_life=90):
    """Only official monthly labels with target_end <= row time.

    Residuals must be chronological out-of-time predictions. Missing OOF values
    contribute neither to their numerator nor their effective observation count.
    """
    history = np.full((len(objects), len(HISTORY)), np.nan, dtype=np.float32)
    effects = {lam: np.zeros(len(objects)) for lam in lambdas}
    labels = objects.loc[objects.official].copy().sort_values("target_end")
    labels["z"] = np.log1p(labels.spend_30d)
    labels["r"] = residuals[labels.index]
    availability = labels.target_end.to_numpy()
    global_sum = np.r_[0.0, labels.z.cumsum()]
    global_zero = np.r_[0.0, labels.z.eq(0).cumsum()]
    groups = dict(tuple(labels.groupby("customer_id", sort=False)))
    for cid, rows in objects.groupby("customer_id", sort=False):
        lab = groups.get(cid)
        for idx, row in rows.iterrows():
            n_global = np.searchsorted(availability, row.as_of_date.to_datetime64(), side="right")
            mean = global_sum[n_global] / max(n_global, 1)
            zero = global_zero[n_global] / max(n_global, 1)
            past = lab.loc[lab.target_end.le(row.as_of_date)] if lab is not None else labels.iloc[:0]
            n = len(past)
            age = (row.as_of_date - past.target_end).dt.total_seconds().to_numpy() / 86400
            weight = np.exp2(-age / half_life)
            history[idx] = [np.log1p(n), (past.z.sum() + prior * mean) / (n + prior),
                            (past.z.eq(0).sum() + prior * zero) / (n + prior),
                            (np.sum(weight * past.z) + prior * mean) / (weight.sum() + prior)]
            valid = past.r.dropna()
            for lam in lambdas:
                effects[lam][idx] = valid.sum() / (len(valid) + lam)
    return history, effects


def chronological_residuals(data, alpha=10, min_dates=3, progress=None):
    objects = data.objects
    residuals = np.full(len(objects), np.nan)
    metadata = []
    for date in sorted(objects.loc[objects.official, "as_of_date"].unique()):
        train = split_indices(objects, "D2", date)
        if objects.iloc[train].as_of_date.nunique() < min_dates:
            continue
        valid = validation_rows(objects, date)
        model = ridge(data.matrix(train), objects.iloc[train].spend_30d.to_numpy(),
                      row_weights(objects, train, "D2"), alpha)
        residuals[valid] = np.log1p(objects.iloc[valid].spend_30d) - model.predict(data.matrix(valid))
        metadata.append({"date": str(pd.Timestamp(date).date()), "n_train": len(train),
                         "max_target_end": str(objects.iloc[train].target_end.max()), "n_oof": len(valid)})
        if progress:
            progress(f"OOF Ridge {pd.Timestamp(date).date()}: {len(valid)} customers")
    return residuals, metadata


def survival_observations(objects, next_event, cutoff):
    # Monthly origins limit duplicated risk sets; incomplete current windows are
    # retained as censored observations, never fabricated negative 30d labels.
    rows = np.flatnonzero((objects.official & objects.as_of_date.lt(pd.Timestamp(cutoff))).to_numpy())
    origin = objects.iloc[rows].as_of_date.to_numpy().astype("datetime64[ns]").astype(np.int64) / 86400e9
    stop = min(pd.Timestamp(cutoff).value / 86400e9,
               objects.loc[objects.is_test, "as_of_date"].min().value / 86400e9)
    censor = np.minimum(30.0, stop - origin)
    delta = next_event[rows] - origin
    event = delta < censor  # event_ts == cutoff is unavailable
    duration = np.maximum(np.minimum(delta, censor), 1e-6)
    return rows, duration, event


class DiscreteHazard:
    def __init__(self, c=1.0, edges=(0, 7, 14, 21, 30)):
        self.c, self.edges = c, np.asarray(edges)

    def design(self, a, bins):
        return np.c_[a, np.eye(len(self.edges) - 1)[bins]]

    def fit(self, x, duration, event, weights):
        self.preprocessor = Preprocessor().fit(x, weights)
        a = self.preprocessor.transform(x)
        # Censoring partway through a bin conveys no binary outcome for that bin.
        r, b = np.where((duration[:, None] >= self.edges[1:]) |
                        (event[:, None] & (duration[:, None] >= self.edges[:-1])))
        target = event[r] & (duration[r] < self.edges[b + 1])
        self.constant = float(target[0]) if len(np.unique(target)) == 1 else None
        self.n_periods = len(r)
        self.n_censored = int((~event).sum())
        if self.constant is None:
            self.model = LogisticRegression(C=self.c, max_iter=1000).fit(
                self.design(a[r], b), target, sample_weight=weights[r])
            if self.model.n_iter_.max() >= 1000:
                raise RuntimeError("Hazard logistic regression did not converge")
        return self

    def predict(self, x):
        a = self.preprocessor.transform(x)
        survival = np.ones(len(x))
        for b in range(len(self.edges) - 1):
            h = (np.full(len(x), self.constant) if self.constant is not None else
                 self.model.predict_proba(self.design(a, np.full(len(x), b)))[:, 1])
            survival *= 1 - h
        return np.clip(1 - survival, 0, 1)


def weibull_objective(theta, a, duration, event, weights, penalty):
    """Weibull AFT right-censored likelihood and analytic gradient."""
    beta, log_shape = theta[:-1], theta[-1]
    shape = np.exp(log_shape)
    v = np.log(duration) - a @ beta
    power = shape * v
    cumulative = np.exp(np.clip(power, -60, 60))
    log_pdf = log_shape - np.log(duration) + power - cumulative
    loss = np.sum(weights * np.where(event, -log_pdf, cumulative)) / weights.sum()
    loss += penalty * np.sum(beta[1:] ** 2) / 2
    common = weights * (cumulative - event) / weights.sum()
    grad_beta = -shape * (a.T @ common)
    grad_beta[1:] += penalty * beta[1:]
    grad_shape = np.sum(common * power) - np.sum(weights * event) / weights.sum()
    return loss, np.r_[grad_beta, grad_shape]


class WeibullAFT:
    def __init__(self, penalty=0.1):
        self.penalty = penalty

    def fit(self, x, duration, event, weights):
        self.preprocessor = Preprocessor().fit(x, weights)
        a = np.c_[np.ones(len(x)), self.preprocessor.transform(x)].astype(float)
        initial = np.zeros(a.shape[1] + 1)
        initial[0] = np.log(30)
        fit = minimize(weibull_objective, initial, args=(a, duration, event, weights, self.penalty),
                       jac=True, method="L-BFGS-B", bounds=[(-12, 12)] * a.shape[1] + [(-2, 2)],
                       options={"maxiter": 300, "ftol": 1e-9})
        if not fit.success:
            raise RuntimeError(f"Weibull fit failed: {fit.message}")
        self.theta, self.n_iter = fit.x, fit.nit
        return self

    def predict(self, x):
        a = np.c_[np.ones(len(x)), self.preprocessor.transform(x)]
        power = np.exp(self.theta[-1]) * (np.log(30) - a @ self.theta[:-1])
        return -np.expm1(-np.exp(np.clip(power, -60, 60)))


class SegmentRidge:
    def __init__(self, method, k, mode, min_customers=100, shrink=200, alpha=10):
        self.method, self.k, self.mode = method, k, mode
        self.min_customers, self.shrink, self.alpha = min_customers, shrink, alpha

    def probabilities(self, x):
        a = self.preprocessor.transform(x)
        if self.method == "gmm":
            return self.cluster.predict_proba(a)
        if self.method == "rules":
            # Input columns 0..3: log n_days, log mean check, log max check, log CV.
            large = x[:, 1] >= self.large_threshold
            regular = (x[:, 0] >= np.log1p(6)) & (np.nan_to_num(x[:, 3], nan=10) <= np.log1p(1))
            labels = np.where(large & regular, 2, np.where(large, 1, 0))
            return np.eye(3)[labels]
        distances = self.cluster.transform(a) ** 2
        return softmax(-distances / self.temperature, axis=1)

    def fit(self, x, sx, y, w, customers):
        # One most recent available snapshot per customer, without target-based
        # filtering. Prevent heavy customers/synthetic rows dominating clusters.
        last = ~pd.Series(customers).duplicated(keep="last").to_numpy()
        self.preprocessor = Preprocessor().fit(sx[last], np.ones(last.sum()))
        a = self.preprocessor.transform(sx[last])
        if self.method == "kmeans":
            self.cluster = KMeans(self.k, n_init=10, random_state=42).fit(a)
            self.temperature = max(float(np.median(self.cluster.transform(a).min(1) ** 2)), 1)
        elif self.method == "gmm":
            self.cluster = GaussianMixture(self.k, covariance_type="diag", reg_covar=0.05,
                                          n_init=2, random_state=42).fit(a)
            if not self.cluster.converged_:
                raise RuntimeError("GMM did not converge")
        elif self.method == "rules":
            self.large_threshold = float(np.nanquantile(sx[last, 1], 0.85))
        else:
            raise ValueError(self.method)
        p = self.probabilities(sx)
        hard = np.eye(self.k)[p.argmax(1)]
        gates = p if self.mode == "soft" else hard
        self.global_model = ridge(np.c_[x, p] if self.mode == "feature" else x, y, w, self.alpha)
        self.experts, self.strength, self.summary = [], [], []
        for k in range(self.k):
            wk = w * gates[:, k]
            unique = len(np.unique(np.asarray(customers)[hard[:, k] > 0]))
            mass = float(wk.sum())
            enough = unique >= self.min_customers and mass >= self.min_customers
            fit = ridge(x, y, wk, self.alpha) if enough and self.mode != "feature" else None
            self.experts.append(fit)
            self.strength.append(mass / (mass + self.shrink) if enough else 0)
            self.summary.append({"segment": k, "customers": unique, "weight": mass,
                                 "fallback": not enough,
                                 "mean_log_frequency": float(np.average(sx[:, 0], weights=wk)) if mass else None,
                                 "mean_log_order_amount": float(np.average(sx[:, 1], weights=wk)) if mass else None})
        return self

    def predict(self, x, sx):
        p = self.probabilities(sx)
        if self.mode == "feature":
            return self.global_model.predict(np.c_[x, p])
        if self.mode == "hard":
            p = np.eye(self.k)[p.argmax(1)]
        baseline = self.global_model.predict(x)
        result = baseline.copy()
        for k, expert in enumerate(self.experts):
            if expert is not None:
                result += p[:, k] * self.strength[k] * (expert.predict(x) - baseline)
        return np.maximum(result, 0)


def segment_matrix(data, indices):
    from spend_baselines.suite_data import BASE_COLUMNS
    # Existing returns/cancellations remain available as basket aggregates.
    base_names = ["n_180", "amount_per_order_180", "orders_180", "products_180",
                  "units_180", "returns_180", "return_amount_180"]
    base = data.base.features[np.ix_(indices, [BASE_COLUMNS.index(c) for c in base_names])]
    extra_names = ["order_max", "gap_cv", "order_lines", "large_order_share", "item_repeat_180"]
    extra = data.extra[np.ix_(indices, [COLUMNS.index(c) for c in extra_names])]
    # First four have a fixed interpretation for rule-based segmentation.
    return np.c_[base[:, :2], extra[:, :2], base[:, 2:], extra[:, 2:]]
