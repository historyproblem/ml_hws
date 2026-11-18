"""Strict forward-OOF comparisons and fixed blends; no model or weight fitting.

``analyze_predictions`` accepts long predictions with identical validation IDs
for every model and a ``cb_reference`` model. It returns ``predictions`` (input
models plus fixed CB/candidate 50:50 probability and within-quarter rank blends),
``quarter_scores``, ``summary``, ``diagnostics`` DataFrames, and a JSON-compatible
``comparison`` protocol dict. Only the three established development quarters
are accepted. Optional ``history_violation_mean`` is retained for diagnostics.
Bootstrap intervals describe exploratory, already-used development data.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

DEVELOPMENT_QUARTERS = ("2023-07-01", "2023-10-01", "2024-01-01")
REFERENCE = "cb_reference"
REQUIRED = ("row_id", "site_id", "forecast_period", "target", "model",
            "prediction", "seen_site")


def prepare_weighted_ap(y, prediction):
    """Pre-sort binary targets once; returned scorer handles tied scores/weights.

    A resample containing no positive weight has AP zero, as in sklearn. Empty
    inputs are rejected; weights must be finite, nonnegative and row aligned.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(prediction, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("AP needs nonempty aligned one-dimensional inputs")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError("AP needs binary targets and finite predictions")
    order = np.argsort(-p, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(p[order]) != 0), len(order) - 1]
    positive = y[order]

    def score(weight):
        weight = np.asarray(weight, dtype=float)
        if weight.shape != y.shape or not np.isfinite(weight).all() or (weight < 0).any():
            raise ValueError("AP weights must be aligned, finite and nonnegative")
        w = weight[order]
        tp = np.cumsum(positive * w)[ends]
        if tp[-1] == 0:
            return 0.0
        total = np.cumsum(w)[ends]
        precision = np.divide(tp, total, out=np.zeros_like(tp), where=total > 0)
        return float(np.sum(np.diff(np.r_[0, tp]) * precision) / tp[-1])

    return score


def _metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    result = {"n": len(y), "positive": int(y.sum()),
              "prevalence": float(y.mean()) if len(y) else np.nan,
              "ap": float(average_precision_score(y, p)) if y.sum() else 0.0,
              "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
              "logloss": float(log_loss(y, p, labels=[0, 1])) if len(y) else np.nan}
    order = np.argsort(-p, kind="stable")
    for percent in (5, 10, 20):
        k = max(1, math.ceil(len(y) * percent / 100)) if len(y) else 0
        found = int(y[order[:k]].sum())
        result[f"precision_at_{percent}pct"] = found / k if k else np.nan
        result[f"recall_at_{percent}pct"] = found / int(y.sum()) if y.sum() else np.nan
    return result


def _aligned_inputs(predictions):
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a DataFrame")
    missing = set(REQUIRED) - set(predictions)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    columns = [*REQUIRED]
    if "history_violation_mean" in predictions:
        columns.append("history_violation_mean")
    data = predictions[columns].copy()
    if data.empty or data[list(REQUIRED)].isna().any().any():
        raise ValueError("Prediction inputs must be nonempty and required values nonmissing")
    if not data.model.map(lambda v: isinstance(v, str) and bool(v.strip())).all():
        raise ValueError("Model names must be nonempty strings")
    if data.duplicated(["model", "row_id"]).any():
        raise ValueError("Duplicate model/row_id predictions")
    dates = pd.to_datetime(data.forecast_period, errors="raise")
    data["forecast_period"] = dates.dt.strftime("%Y-%m-%d")
    if not data.forecast_period.isin(DEVELOPMENT_QUARTERS).all():
        raise ValueError("Only the established development quarters may be scored")
    # Reject intraday timestamps that would otherwise be silently rounded away.
    if not dates.eq(dates.dt.normalize()).all():
        raise ValueError("forecast_period must be an exact quarter-start date")
    data["prediction"] = pd.to_numeric(data.prediction, errors="raise").astype(float)
    if not np.isfinite(data.prediction).all() or not data.prediction.between(0, 1).all():
        raise ValueError("Predictions must be finite probabilities in [0, 1]")
    if not data.target.isin([0, 1]).all():
        raise ValueError("Targets must be binary")
    if not data.seen_site.isin([False, True]).all():
        raise ValueError("seen_site must contain booleans or binary values")
    data["target"] = data.target.astype(int)
    data["seen_site"] = data.seen_site.astype(bool)
    if "history_violation_mean" in data:
        data["history_violation_mean"] = pd.to_numeric(data.history_violation_mean, errors="raise")
        if np.isinf(data.history_violation_mean).any():
            raise ValueError("History burden must be finite or missing")
    names = sorted(data.model.unique())
    if REFERENCE not in names:
        raise ValueError(f"Required reference model missing: {REFERENCE}")
    models = [REFERENCE, *[name for name in names if name != REFERENCE]]
    reference = data.loc[data.model.eq(REFERENCE)].sort_values(
        ["forecast_period", "row_id"], kind="stable").set_index("row_id")
    aligned = []
    metadata = [c for c in columns if c not in ["row_id", "model", "prediction"]]
    for model in models:
        part = data.loc[data.model.eq(model)].set_index("row_id")
        if len(part) != len(reference) or set(part.index) != set(reference.index):
            raise ValueError(f"Prediction ID mismatch for {model}")
        part = part.loc[reference.index]
        # pandas equals treats matching NaNs in optional metadata as equal.
        if not part[metadata].equals(reference[metadata]):
            raise ValueError(f"Site/quarter/target/metadata mismatch for {model}")
        aligned.append(part.reset_index())
    return pd.concat(aligned, ignore_index=True), reference.reset_index(), models


def analyze_predictions(predictions: pd.DataFrame, bootstrap=1000, seed=20261118):
    """Compare aligned OOF models and fixed blends without selecting a winner.

    ``bootstrap=0`` skips intervals for smoke checks. Otherwise sample unique
    sites with replacement and share each site's multiplicity across every
    quarter/model. Mean AP assigns equal weight to each supplied dev quarter.
    No closed-quarter rows, learned blend weights, or stacking are accepted.
    """
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, (int, np.integer)) or bootstrap < 0:
        raise ValueError("bootstrap must be a nonnegative integer")
    data, reference, original_models = _aligned_inputs(predictions)
    quarters = [q for q in DEVELOPMENT_QUARTERS if q in set(reference.forecast_period)]
    original = {name: data.loc[data.model.eq(name), "prediction"].to_numpy()
                for name in original_models}
    names = list(original_models)
    values = dict(original)
    appended = [data]
    for model in original_models[1:]:
        probability_name, rank_name = f"{model}_blend50", f"{model}_rankblend50"
        if probability_name in values or rank_name in values:
            raise ValueError("Model name collides with a generated fixed blend")
        probability = .5 * (original[REFERENCE] + original[model])
        ranks = []
        for name in [REFERENCE, model]:
            ranks.append(pd.Series(original[name]).groupby(reference.forecast_period).rank(
                method="average", pct=True).to_numpy())
        rank = .5 * (ranks[0] + ranks[1])
        for name, p in [(probability_name, probability), (rank_name, rank)]:
            values[name] = p
            names.append(name)
            part = reference.copy()
            part["model"] = name
            part["prediction"] = p
            appended.append(part)
    output = pd.concat(appended, ignore_index=True)
    quarter_rows, diagnostics, prepared = [], [], {}
    masks = {q: reference.forecast_period.eq(q).to_numpy() for q in quarters}
    y = reference.target.to_numpy()
    for quarter, mask in masks.items():
        for name in names:
            p = values[name][mask]
            scores = _metrics(y[mask], p)
            quarter_rows.append({"model": name, "forecast_period": quarter, **scores})
            prepared[quarter, name] = prepare_weighted_ap(y[mask], p)
            a, b = pd.Series(values[REFERENCE][mask]), pd.Series(p)
            diagnostics.append({"kind": "correlation", "model": name,
                                "forecast_period": quarter, "slice": "all", "n": int(mask.sum()),
                                "pearson_to_reference": float(a.corr(b)),
                                "spearman_to_reference": float(a.rank().corr(b.rank()))})
            slices = {"warm": reference.seen_site.to_numpy(),
                      "cold": ~reference.seen_site.to_numpy()}
            if "history_violation_mean" in reference:
                burden = reference.history_violation_mean
                slices.update({"burden_le2": burden.le(2).to_numpy(),
                               "burden_gt2": burden.gt(2).to_numpy(),
                               "burden_missing": burden.isna().to_numpy()})
            for label, selected in slices.items():
                subset = mask & selected
                diagnostics.append({"kind": "slice", "model": name, "forecast_period": quarter,
                                    "slice": label, **_metrics(y[subset], values[name][subset])})
    quarter_scores = pd.DataFrame(quarter_rows)
    means = quarter_scores.groupby("model").ap.mean().to_dict()
    site_codes, unique_sites = pd.factorize(reference.site_id, sort=True)
    rng = np.random.default_rng(seed)
    draws = np.empty((bootstrap, len(names)), dtype=float)
    probabilities = np.full(len(unique_sites), 1 / len(unique_sites))
    for repeat in range(bootstrap):
        # One draw supplies every model and quarter, preserving repeated sites.
        weights = rng.multinomial(len(unique_sites), probabilities)[site_codes]
        for j, name in enumerate(names):
            draws[repeat, j] = np.mean([prepared[q, name](weights[masks[q]]) for q in quarters])
    summary_rows = []
    for j, name in enumerate(names):
        delta_draws = draws[:, j] - draws[:, 0]
        low, high = np.quantile(delta_draws, [.025, .975]) if bootstrap else (np.nan, np.nan)
        summary_rows.append({"model": name, "mean_ap": means[name],
                             "delta_vs_reference": means[name] - means[REFERENCE],
                             "ci95_low": float(low), "ci95_high": float(high), "folds": len(quarters)})
    summary = pd.DataFrame(summary_rows).sort_values(
        ["mean_ap", "model"], ascending=[False, True], kind="stable").reset_index(drop=True)
    comparison = {"reference_model": REFERENCE, "quarters": quarters,
                  "original_models": original_models, "all_models": names,
                  "validation_rows": len(reference), "unique_sites": len(unique_sites),
                  "mean_ap": "equal-weight mean of within-quarter sklearn Average Precision",
                  "blends": "fixed 50/50 probabilities and within-quarter average percentile ranks",
                  "rank_ties": "average rank; percentile denominator is quarter row count",
                  "top_fraction_ties": "stable canonical order by forecast_period,row_id",
                  "bootstrap": {"repeats": int(bootstrap), "seed": int(seed), "unit": "site_id",
                                "weights": "same site multiplicity in every quarter and model",
                                "confidence": "nominal 95% development intervals; no selection correction",
                                "no_positive_resample_ap": 0.0},
                  "new_fits": 0, "weight_search": False, "stacking": False,
                  "lockbox_scored": False, "uploads": 0}
    return {"predictions": output, "quarter_scores": quarter_scores, "summary": summary,
            "diagnostics": pd.DataFrame(diagnostics), "comparison": comparison}
