"""Bounded, no-fit interpretation of saved plus_context seed42 dev models."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hw4_inspections.features import build_features
from hw4_inspections.extensions import build_context_features

QUARTERS = ["2023-07-01", "2023-10-01", "2024-01-01"]


def main():
    out = ROOT / "analysis/hw4-features-20261116/context_interpret"
    out.mkdir(parents=True, exist_ok=True)
    screen = ROOT / "analysis/hw4-features-20261116/extensions_screen"
    rows = pd.read_csv(ROOT / "data/hw4/raw/train.csv", usecols=lambda c: c != "target")
    rows = rows.loc[rows.forecast_period.isin(QUARTERS)].reset_index(drop=True)
    history = pd.read_csv(ROOT / "data/hw4/raw/inspection_history.csv")
    history = history.loc[history.forecast_period.isin(QUARTERS)]
    base, cats = build_features(rows, history)
    context = build_context_features(rows, history)
    features = pd.concat([base, context], axis=1)
    for col in cats:
        features[col] = features[col].fillna("__MISSING__").astype(str)
    reference = pd.read_csv(screen / "predictions.csv", float_precision="round_trip")
    reference = reference.loc[(reference["case"] == "plus_context") & (reference.seed == 42)].set_index("row_id").loc[rows.row_id]
    original = pd.read_csv(ROOT / "outputs/hw4/catboost-initial/development_predictions.csv", float_precision="round_trip")
    original = original.loc[original.model == "history_catboost"].set_index("row_id").loc[rows.row_id]
    np.testing.assert_array_equal(reference.target, original.target)
    y = reference.target.to_numpy()
    groups = {"context_count": [c for c in context if c.endswith("_n")],
              "context_fail_smooth": [c for c in context if c.endswith("_fail_smooth")],
              "context_load_smooth": [c for c in context if c.endswith("_load_smooth")],
              "all_context": list(context)}
    shap_records, perm_records, catches, transitions, burden_records, checks = [], [], [], [], [], []
    for quarter in QUARTERS:
        ix = np.flatnonzero((rows.forecast_period == quarter).to_numpy())
        X = features.iloc[ix]
        target = y[ix]
        pbase = original.prediction.to_numpy()[ix]
        model = CatBoostClassifier()
        model_path = screen / f"plus_context_seed42_{quarter}.cbm"
        model.load_model(str(model_path))
        assert model.feature_names_ == list(X)
        p = model.predict_proba(X, thread_count=4)[:, 1]
        np.testing.assert_array_equal(p, reference.prediction.to_numpy()[ix])
        ap = float(average_precision_score(target, p))
        shap = model.get_feature_importance(Pool(X, cat_features=cats), type="ShapValues", thread_count=4)
        raw = model.predict(X, prediction_type="RawFormulaVal", thread_count=4)
        error = float(np.abs(shap.sum(axis=1) - raw).max())
        assert error < 1e-10
        for col in context:
            jj = X.columns.get_loc(col)
            values = X[col].to_numpy()
            phi = shap[:, jj]
            rho = np.nan
            if np.nanstd(values) > 0 and np.ptp(phi) > 0:
                rho = float(pd.Series(values).corr(pd.Series(phi), method="spearman"))
            shap_records.append({"quarter": quarter, "feature": col,
                "mean_abs_shap": float(np.abs(phi).mean()), "mean_shap": float(phi.mean()),
                "spearman_value_shap": rho, "used": bool(np.any(phi != 0))})
        for group, columns in groups.items():
            for repeat in range(5):
                perm = np.random.default_rng(20261116 + repeat).permutation(len(X))
                changed = X.copy()
                for col in columns:
                    changed[col] = X[col].iloc[perm].to_numpy()
                pp = model.predict_proba(changed, thread_count=4)[:, 1]
                pap = float(average_precision_score(target, pp))
                perm_records.append({"quarter": quarter, "group": group, "repeat": repeat,
                    "ap": ap, "permuted_ap": pap, "ap_drop": ap - pap})
        def top20(values):
            mask = np.zeros(len(values), dtype=bool)
            mask[np.argsort(-values, kind="stable")[:int(np.ceil(.2 * len(values)))]] = True
            return mask
        btop, ctop = top20(pbase), top20(p)
        positive = target == 1
        catches.append({"quarter": quarter, "n": len(X), "positives": int(positive.sum()),
            "baseline_top20_positives": int((btop & positive).sum()),
            "context_top20_positives": int((ctop & positive).sum()),
            "new_positive_catches": int((~btop & ctop & positive).sum()),
            "lost_positive_catches": int((btop & ~ctop & positive).sum()),
            "baseline_ap": float(average_precision_score(target, pbase)), "context_ap": ap})
        burden = X.history_violation_mean
        buckets = pd.cut(burden, bins=[-np.inf, 2, 4, 8, np.inf], labels=["<=2", "2-4", "4-8", ">8"])
        for bucket in buckets.cat.categories:
            mask = (buckets == bucket).to_numpy()
            burden_records.append({"quarter": quarter, "burden_bin": str(bucket), "n": int(mask.sum()),
                "positives": int((mask & positive).sum()),
                "baseline_top20_positives": int((mask & positive & btop).sum()),
                "context_top20_positives": int((mask & positive & ctop).sum()),
                "new_positive_catches": int((mask & positive & ctop & ~btop).sum()),
                "lost_positive_catches": int((mask & positive & ~ctop & btop).sum())})
        for label, mask in {"new_caught_positive": positive & ctop & ~btop,
                            "lost_positive": positive & btop & ~ctop,
                            "retained_caught_positive": positive & btop & ctop,
                            "still_missed_positive": positive & ~btop & ~ctop}.items():
            part = X.loc[mask]
            transitions.append({"quarter": quarter, "cohort": label, "n": len(part),
                "median_history_burden": float(part.history_violation_mean.median()),
                "median_last_burden": float(part.lag1_violation_count.median()),
                "last_routine_pass_fraction": float((part.last_routine_result == "pass").mean()),
                "mean_baseline_prediction": float(pbase[mask].mean()),
                "mean_context_prediction": float(p[mask].mean())})
        checks.append({"quarter": quarter, "reference_prediction_exact": True,
            "shap_additivity_max_error": error, "used_context_features": int(sum(r["used"] for r in shap_records if r["quarter"] == quarter)),
            "context_sum_mean_abs_shap": float(np.abs(shap[:, [X.columns.get_loc(c) for c in context]]).mean(axis=0).sum()),
            "total_sum_mean_abs_shap": float(np.abs(shap[:, :-1]).mean(axis=0).sum()),
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()})
        print(json.dumps(catches[-1]), flush=True)
    frames = {"context_shap_by_quarter": pd.DataFrame(shap_records),
        "context_permutation_repeats": pd.DataFrame(perm_records), "top20_catches": pd.DataFrame(catches),
        "positive_transition_profiles": pd.DataFrame(transitions), "top20_burden_strata": pd.DataFrame(burden_records)}
    for name, frame in frames.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    summary = frames["context_shap_by_quarter"].groupby("feature", as_index=False).agg(
        mean_abs_shap=("mean_abs_shap", "mean"), used_quarters=("used", "sum"),
        mean_spearman_value_shap=("spearman_value_shap", "mean"))
    summary.sort_values("mean_abs_shap", ascending=False).to_csv(out / "context_shap_summary.csv", index=False)
    psummary = frames["context_permutation_repeats"].groupby(["quarter", "group"], as_index=False).agg(
        mean_ap_drop=("ap_drop", "mean"), repeat_std=("ap_drop", "std"))
    psummary.to_csv(out / "context_permutation_by_quarter.csv", index=False)
    psummary.groupby("group", as_index=False).mean(numeric_only=True).to_csv(out / "context_permutation_summary.csv", index=False)
    corr = context.corr(method="spearman")
    pairs = [{"feature1": a, "feature2": b, "spearman": float(corr.loc[a, b])}
             for i, a in enumerate(corr) for b in corr.columns[i + 1:] if abs(corr.loc[a, b]) >= .95]
    pd.DataFrame(pairs, columns=["feature1", "feature2", "spearman"]).to_csv(out / "high_correlation_context_pairs.csv", index=False)
    result = {"quarters": QUARTERS, "fit_calls": 0, "closed_holdout_labels_loaded": False,
        "checks": checks, "groups": groups,
        "high_context_correlation_pairs_abs_ge_095": len(pairs),
        "caveats": ["SHAP explains saved models, not causal effects or removal/refit gains.",
            "Joint group permutation preserves within-group structure but breaks correlations between context count/fail/load and focal-site features; OOD combinations can inflate or reverse importance.",
            "Repeat standard deviation is Monte Carlo variability, not future-quarter uncertainty.",
            "Only existing development quarters; no test/closed-quarter scoring."],
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), ROOT / "src/hw4_inspections/features.py", ROOT / "src/hw4_inspections/extensions.py"]}}
    (out / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(summary.sort_values("mean_abs_shap", ascending=False).head(12).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
