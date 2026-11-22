"""One descriptive paired diagnostic of two previously consumed quarters.

No model fitting, weight search, eligibility rule or winner selection lives here.
Call only after the finalist recipes and predictions have been frozen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .classical_analysis import prepare_weighted_ap

QUARTERS = ("2024-04-01", "2024-07-01")
META = ("row_id", "site_id", "forecast_period", "target")


def align_recent(train, predictions, models):
    """Validate and order the exact official recent rows without computing AP."""
    if (not isinstance(predictions, pd.DataFrame) or not predictions.columns.is_unique
            or not models or len(models) != len(set(models))
            or any(not isinstance(n, str) or not n.strip() or n in META for n in models)):
        raise ValueError("Unique named finalist columns required")
    if set(predictions) != set(META) | set(models):
        raise ValueError("CSV must contain exactly metadata and the frozen finalist columns")
    if any(c not in train for c in META) or train.row_id.duplicated().any():
        raise ValueError("Invalid official training identities")
    expected = pd.concat([train.loc[train.forecast_period.eq(q), list(META)] for q in QUARTERS], ignore_index=True)
    if (expected.empty or set(expected.forecast_period) != set(QUARTERS)
            or expected[list(META)].isna().any().any() or not expected.target.isin([0, 1]).all()
            or predictions.row_id.isna().any() or predictions.row_id.duplicated().any()
            or set(predictions.row_id) != set(expected.row_id) or len(predictions) != len(expected)):
        raise ValueError("Prediction CSV must contain every unique official Q2/Q3 row exactly once")
    aligned = predictions.set_index("row_id").loc[expected.row_id].reset_index()
    pd.testing.assert_frame_equal(expected, aligned[list(META)], check_dtype=False, check_exact=True)
    for name in models:
        values = pd.to_numeric(aligned[name], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Finalist probabilities must be finite and in [0,1]")
        aligned[name] = values
    return aligned[[*META, *models]].reset_index(drop=True)


def describe_recent(aligned, models, reference, *, bootstrap=2000, seed=20261122):
    """Exact quarter AP, equal-quarter means and common-site paired intervals."""
    if reference not in models or len(models) != len(set(models)):
        raise ValueError("A unique frozen reference is required")
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or bootstrap < 0:
        raise ValueError("Nonnegative integer bootstrap count required")
    if set(aligned.forecast_period) != set(QUARTERS):
        raise ValueError("Exactly the two recent quarters are allowed")
    names = [reference, *[name for name in models if name != reference]]
    y = aligned.target.to_numpy(dtype=int)
    masks = {q: aligned.forecast_period.eq(q).to_numpy() for q in QUARTERS}
    values = {name: aligned[name].to_numpy(dtype=float) for name in names}
    scores, prepared, rows = {}, {}, []
    for quarter, mask in masks.items():
        for name in names:
            scores[quarter, name] = float(average_precision_score(y[mask], values[name][mask])) if y[mask].sum() else 0.
            prepared[quarter, name] = prepare_weighted_ap(y[mask], values[name][mask])
        for name in names:
            rows.append(dict(model=name, forecast_period=quarter, ap=scores[quarter, name],
                             delta_vs_reference=scores[quarter, name] - scores[quarter, reference],
                             n=int(mask.sum()), positive=int(y[mask].sum())))
    means = {name: float(np.mean([scores[q, name] for q in QUARTERS])) for name in names}
    codes, sites = pd.factorize(aligned.site_id, sort=True)
    probabilities = np.full(len(sites), 1 / len(sites))
    rng = np.random.default_rng(seed)
    draws = np.empty((bootstrap, len(names)), dtype=float)
    for i in range(bootstrap):
        weights = rng.multinomial(len(sites), probabilities)[codes]
        for j, name in enumerate(names):
            draws[i, j] = np.mean([prepared[q, name](weights[masks[q]]) for q in QUARTERS])
    summary = []
    for j, name in enumerate(names):
        low, high = np.quantile(draws[:, j] - draws[:, 0], [.025, .975]) if bootstrap else (None, None)
        differences = [scores[q, name] - scores[q, reference] for q in QUARTERS]
        summary.append(dict(model=name, mean_ap=means[name], delta_vs_reference=means[name] - means[reference],
                            ci95_low=float(low) if low is not None else None,
                            ci95_high=float(high) if high is not None else None,
                            nonnegative_quarters=sum(v >= 0 for v in differences), worst_quarter_delta=min(differences)))
    return dict(
        quarter_metrics=pd.DataFrame(rows),
        summary=pd.DataFrame(summary).sort_values(["mean_ap", "model"], ascending=[False, True], kind="stable").reset_index(drop=True),
        protocol=dict(reference=reference, models=names, quarters=list(QUARTERS), rows=len(aligned),
                      unique_sites=len(sites), equal_quarter_weight=True, fresh_holdout=False,
                      method="Previously consumed quarters, one descriptive frozen-finalist temporal diagnostic",
                      bootstrap=dict(repeats=bootstrap, seed=seed, unit="site_id", common_weights_across_models_and_quarters=True,
                                     confidence="Nominal paired95%; no correction for earlier adaptive model research"),
                      new_fits=0, model_selection_performed=False, parameter_search=False, weight_search=False,
                      kaggle_uploads=0),
    )
