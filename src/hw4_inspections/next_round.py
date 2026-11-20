"""Predeclared old-development comparison gate for the next HW4 candidates.

The caller constructs every candidate, including any fixed blend. This module
does not generate blends, search weights, fit models, or accept later quarters.
``evaluate_candidates`` uses positional row alignment and returns quarter_scores
and summary DataFrames, an eligibility mapping, and a JSON-compatible protocol.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .classical_analysis import prepare_weighted_ap

QUARTERS = ("2023-07-01", "2023-10-01", "2024-01-01")
REQUIRED = ("row_id", "site_id", "forecast_period", "target")


def _scores(values, expected_n=None):
    if np.iscomplexobj(values):
        raise ValueError("Scores must be real")
    try:
        # Cast BEFORE ranking; ranking float32 directly can round percentiles.
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Scores must be numeric") from error
    if result.ndim != 1 or not len(result):
        raise ValueError("Scores must be a nonempty one-dimensional array")
    if expected_n is not None and len(result) != expected_n:
        raise ValueError("Scores must align with every supplied row")
    if not np.isfinite(result).all():
        raise ValueError("Scores must be finite")
    return result


def rank_scores(values):
    """Return float64 average percentile ranks for any finite real raw scores."""
    scores = _scores(values)
    return pd.Series(scores).rank(method="average", pct=True).to_numpy(dtype=np.float64)


def _rows(rows):
    if not isinstance(rows, pd.DataFrame) or not rows.columns.is_unique:
        raise ValueError("rows must be a DataFrame with unique columns")
    missing = set(REQUIRED) - set(rows)
    if missing:
        raise ValueError(f"Missing row metadata: {sorted(missing)}")
    metadata = rows[list(REQUIRED)].reset_index(drop=True).copy()
    if metadata.empty or metadata.isna().any().any():
        raise ValueError("Row metadata must be nonempty and nonmissing")
    if not metadata.row_id.is_unique:
        raise ValueError("row_id must be unique")
    if not metadata.target.isin([0, 1]).all():
        raise ValueError("Targets must be binary")
    dates = pd.to_datetime(metadata.forecast_period, errors="raise")
    if not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Forecast periods must be exact quarter-start dates")
    metadata["forecast_period"] = dates.dt.strftime("%Y-%m-%d")
    if set(metadata.forecast_period) != set(QUARTERS):
        raise ValueError("Exactly the three old-development quarters are required; later quarters are forbidden")
    metadata["target"] = metadata.target.astype(int)
    return metadata


def evaluate_candidates(rows, predictionsdict, reference="context_3seed",
                        bootstrap=2000, seed=20261121):
    """Evaluate each supplied raw-score vector against the fixed reference.

    Eligibility is independent for each candidate: positive equal-quarter mean
    AP difference and nonnegative AP difference in at least two of three
    quarters. Latest-Q1 and worst-quarter differences are diagnostics; confidence
    intervals do not add an eligibility gate. Reference eligibility is False.
    Prediction vectors must already be in exactly the order of ``rows``.
    ``bootstrap=0`` omits candidate intervals for small interface checks.
    """
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, Integral) or bootstrap < 0:
        raise ValueError("bootstrap must be a nonnegative integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    metadata = _rows(rows)
    if not isinstance(predictionsdict, Mapping) or not predictionsdict:
        raise ValueError("predictionsdict must be a nonempty mapping")
    if not all(isinstance(name, str) and bool(name.strip()) for name in predictionsdict):
        raise ValueError("Candidate names must be nonempty strings")
    if reference not in predictionsdict:
        raise ValueError("Reference prediction vector is required")
    names = [reference, *sorted(name for name in predictionsdict if name != reference)]
    values = {name: _scores(predictionsdict[name], len(metadata)) for name in names}
    masks = {q: metadata.forecast_period.eq(q).to_numpy() for q in QUARTERS}
    y = metadata.target.to_numpy()
    scores, prepared = {}, {}
    for quarter, mask in masks.items():
        for name in names:
            p = values[name][mask]
            scores[quarter, name] = float(average_precision_score(y[mask], p)) if y[mask].sum() else 0.0
            prepared[quarter, name] = prepare_weighted_ap(y[mask], p)
    quarter_rows = []
    for quarter, mask in masks.items():
        for name in names:
            quarter_rows.append({"model": name, "forecast_period": quarter,
                                 "ap": scores[quarter, name],
                                 "delta_vs_reference": scores[quarter, name] - scores[quarter, reference],
                                 "n": int(mask.sum()), "positive": int(y[mask].sum())})
    quarter_scores = pd.DataFrame(quarter_rows)
    means = {name: float(np.mean([scores[q, name] for q in QUARTERS])) for name in names}
    codes, sites = pd.factorize(metadata.site_id, sort=True)
    rng = np.random.default_rng(seed)
    draws = np.empty((bootstrap, len(names)), dtype=np.float64)
    probabilities = np.full(len(sites), 1 / len(sites))
    for repeat in range(bootstrap):
        # One site multiplicity is shared by all quarters and all candidates.
        weights = rng.multinomial(len(sites), probabilities)[codes]
        for j, name in enumerate(names):
            draws[repeat, j] = np.mean([prepared[q, name](weights[masks[q]]) for q in QUARTERS])
    eligibility, summary_rows = {}, []
    for j, name in enumerate(names):
        differences = [scores[q, name] - scores[q, reference] for q in QUARTERS]
        mean_difference = means[name] - means[reference]
        nonnegative = sum(delta >= 0 for delta in differences)
        eligible = bool(name != reference and mean_difference > 0 and nonnegative >= 2)
        eligibility[name] = eligible
        low, high = np.quantile(draws[:, j] - draws[:, 0], [.025, .975]) if bootstrap else (np.nan, np.nan)
        summary_rows.append({"model": name, "mean_ap": means[name],
                             "delta_vs_reference": mean_difference,
                             "ci95_low": float(low), "ci95_high": float(high),
                             "nonnegative_quarters": int(nonnegative),
                             "latest_q1_delta": differences[-1],
                             "worst_quarter_delta": min(differences), "eligible": eligible})
    summary = pd.DataFrame(summary_rows).sort_values(
        ["mean_ap", "model"], ascending=[False, True], kind="stable").reset_index(drop=True)
    protocol = {"reference": reference, "quarters": list(QUARTERS), "models": names,
                "rows": len(metadata), "unique_sites": len(sites),
                "alignment": "Each vector is positional and aligned with the supplied unique row IDs",
                "objective": "Equal-weight mean of the three within-quarter sklearn Average Precision scores",
                "eligibility_rule": "Candidate mean AP delta > 0 and nonnegative delta in at least 2 of 3 quarters",
                "eligibility_is_independent_per_candidate": True,
                "reference_eligible": False, "hard_ci_gate": False,
                "latest_and_worst_quarter_deltas": "Reported diagnostics, not additional selection thresholds",
                "bootstrap": {"repeats": int(bootstrap), "seed": int(seed), "unit": "site_id",
                              "common_site_weights_across_models_and_quarters": True,
                              "confidence": "Nominal 95% exploratory old-development intervals; no selection correction",
                              "zero_positive_resample_ap": 0.0},
                "new_fits": 0, "generated_blends": 0, "weight_search": False,
                "later_quarters_scored": False, "kaggle_uploads": 0}
    return {"quarter_scores": quarter_scores, "summary": summary,
            "eligibility": eligibility, "protocol": protocol}
