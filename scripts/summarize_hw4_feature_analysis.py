"""Summarize HW4 feature refits with matched seeds and site-cluster uncertainty."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/hw4-features-20261116"


def ranked_ap_parts(y, p):
    order = np.argsort(-p, kind="mergesort")
    scores = p[order]
    ends = np.flatnonzero(np.r_[scores[:-1] != scores[1:], True])
    return order, ends, y[order]


def weighted_ap(parts, w):
    order, ends, y = parts
    weight = w[order]
    tp = np.cumsum(weight * y)[ends]
    n = np.cumsum(weight)[ends]
    if not len(tp) or tp[-1] <= 0:
        return np.nan
    precision = np.divide(tp, n, out=np.zeros_like(tp, dtype=float), where=n > 0)
    return float(np.sum(np.diff(np.r_[0, tp]) * precision) / tp[-1])


def main():
    dirs = ["ablation", "seed_confirmation", "extensions_screen", "context_confirmation"]
    pred = pd.concat([pd.read_csv(OUT / name / "predictions.csv") for name in dirs], ignore_index=True)
    assert not pred.duplicated(["row_id", "case", "seed"]).any()
    assert set(pred.forecast_period) == {"2023-07-01", "2023-10-01", "2024-01-01"}
    fold = pred.groupby(["case", "seed", "forecast_period"]).apply(
        lambda p: average_precision_score(p.target, p.prediction), include_groups=False).rename("ap").reset_index()
    baseline = fold.loc[fold.case == "baseline", ["seed", "forecast_period", "ap"]].rename(columns={"ap": "baseline_ap"})
    fold = fold.merge(baseline, on=["seed", "forecast_period"], validate="many_to_one")
    fold["matched_seed_delta"] = fold.ap - fold.baseline_ap
    fold.to_csv(OUT / "matched_seed_fold_metrics.csv", index=False)
    means = fold.groupby(["case", "seed"], as_index=False).agg(mean_ap=("ap", "mean"), matched_delta=("matched_seed_delta", "mean"))
    means.to_csv(OUT / "matched_seed_summary.csv", index=False)
    confirmed = means.groupby("case", as_index=False).agg(mean_ap_across_seeds=("mean_ap", "mean"),
        mean_matched_delta=("matched_delta", "mean"), seeds=("seed", "nunique"))
    confirmed.to_csv(OUT / "case_summary.csv", index=False)
    # Inference is conditional on these THREE time folds, not nine independent tests.
    chosen = ["baseline", "static_plus_burden", "compact_no_codes_types", "plus_context"]
    wide = pred.loc[pred.case.isin(chosen)].pivot(index=["row_id", "site_id", "forecast_period", "target"], columns=["case", "seed"], values="prediction").reset_index()
    y = wide[("target", "")].to_numpy()
    ids, sites = pd.factorize(wide[("site_id", "")])
    quarters = sorted(wide[("forecast_period", "")].unique())
    chunks = [np.flatnonzero((wide[("forecast_period", "")] == q).to_numpy()) for q in quarters]
    prepared = {(case, seed, q): ranked_ap_parts(y[ix], wide[(case, seed)].to_numpy()[ix])
                for case in chosen for seed in [42, 17, 2026] for q, ix in zip(quarters, chunks)}
    # Check fast AP against sklearn, including repeated-score/zero-weight ties.
    for key, parts in prepared.items():
        case, seed, q = key
        ix = chunks[quarters.index(q)]
        np.testing.assert_allclose(weighted_ap(parts, np.ones(len(ix))),
            average_precision_score(y[ix], wide[(case, seed)].to_numpy()[ix]), atol=1e-14)
    rng = np.random.default_rng(20261116)
    distributions = {case: [] for case in chosen if case != "baseline"}
    for _ in range(2000):
        counts = np.bincount(rng.integers(len(sites), size=len(sites)), minlength=len(sites))[ids]
        scores = {case: np.mean([weighted_ap(prepared[(case, seed, q)], counts[ix])
                    for seed in [42, 17, 2026] for q, ix in zip(quarters, chunks)]) for case in chosen}
        for case in distributions:
            distributions[case].append(scores[case] - scores["baseline"])
    result = {"bootstrap": "paired site-cluster 2000; same site weights across quarters/seeds; average of seed-specific quarter AP, NOT AP of seed ensemble",
              "uncertainty_limits": "Exploratory feature selection on same development quarters; nominal intervals, not corrected for candidate selection; only three evaluated time periods.",
              "closed_quarters_scored": False,
              "comparisons": {case: {"delta": float(means.loc[means.case == case, "matched_delta"].mean()),
                  "ci95": np.quantile(values, [.025, .975]).tolist()} for case, values in distributions.items()}}
    (OUT / "comparison_uncertainty.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(confirmed.sort_values("mean_ap_across_seeds", ascending=False).to_string(index=False))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
