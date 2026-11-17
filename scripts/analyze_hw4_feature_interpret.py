"""Explain the three saved forward-validation HW4 CatBoost models, without fitting.

Only development labels 2023Q3, 2023Q4, 2024Q1 are read from saved predictions.
2024Q2/Q3 labels are not loaded or scored. Group permutations are a model
reliance diagnostic, not estimates of gain from removing/refitting features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hw4_inspections.features import build_features

QUARTERS = ["2023-07-01", "2023-10-01", "2024-01-01"]


def feature_group(name: str) -> str:
    if name.startswith("code"):
        return "violation_codes"
    if name == "area_id":
        return "geography"
    if name == "risk_level":
        return "risk_level"
    if name == "facility_type":
        return "facility_type"
    if name.startswith("forecast_") or name == "cutoff_gap_days":
        return "calendar"
    if "age_days" in name or "gap_" in name or "span_days" in name:
        return "recency_and_gaps"
    if "violation" in name:
        return "violation_burden"
    if name.endswith("_result") or any(s in name for s in ["_pass_", "_conditional_", "_fail_"]):
        return "past_outcomes"
    if name.startswith("type_") or "inspection_type" in name or "reinspection" in name:
        return "inspection_types"
    return "visit_frequency"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/hw4/raw")
    parser.add_argument("--models", type=Path, default=ROOT / "outputs/hw4/catboost-initial")
    parser.add_argument("--out", type=Path, default=ROOT / "analysis/hw4-features-20261116/interpret")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.models / "manifest.json").read_text())
    source = ROOT / "src/hw4_inspections/features.py"
    assert sha256(source) == manifest["source_sha256"]["src/hw4_inspections/features.py"]
    # Deliberately never load the target column from the raw training CSV.
    rows = pd.read_csv(args.raw / "train.csv", usecols=lambda c: c != "target")
    rows = rows.loc[rows.forecast_period.isin(QUARTERS)].reset_index(drop=True)
    history = pd.read_csv(args.raw / "inspection_history.csv")
    history = history.loc[history.forecast_period.isin(QUARTERS)]
    X, cats = build_features(rows, history)
    X = X[manifest["features"]]
    for col in cats:
        X[col] = X[col].fillna("__MISSING__").astype(str)
    numeric = X.columns.difference(cats)
    X[numeric] = X[numeric].replace([np.inf, -np.inf], np.nan)
    reference = pd.read_csv(args.models / "development_predictions.csv", float_precision="round_trip")
    reference = reference.loc[reference.model == "history_catboost"].set_index("row_id")
    assert set(reference.forecast_period) == set(QUARTERS)
    ref = reference.loc[rows.row_id]
    y = ref.target.to_numpy()
    groups = {group: [c for c in X if feature_group(c) == group]
              for group in sorted({feature_group(c) for c in X})}
    shap_rows, level_rows, group_rows, permutation_rows, error_rows, checks = [], [], [], [], [], []
    important_profile_features = ["history_n_visits", "history_violation_mean", "lag1_violation_count",
        "last_routine_violation_count", "last_routine_age_days", "last_fail_age_days", "history_fail_rate"]

    for quarter in QUARTERS:
        ix = np.flatnonzero((rows.forecast_period == quarter).to_numpy())
        features, target = X.iloc[ix].copy(), y[ix]
        model = CatBoostClassifier()
        model_path = args.models / f"history_catboost_{quarter}.cbm"
        model.load_model(str(model_path))
        probability = model.predict_proba(features, thread_count=4)[:, 1]
        expected = ref.prediction.to_numpy()[ix]
        np.testing.assert_array_equal(probability, expected)
        base_ap = float(average_precision_score(target, probability))
        shap = model.get_feature_importance(Pool(features, cat_features=cats),
                                            type="ShapValues", thread_count=4)
        raw = model.predict(features, prediction_type="RawFormulaVal", thread_count=4)
        additivity_error = float(np.max(np.abs(shap.sum(axis=1) - raw)))
        assert additivity_error < 1e-10
        phi = shap[:, :-1]
        checks.append({"quarter": quarter, "n": len(ix), "ap": base_ap,
                       "reference_predictions_exact": True, "shap_max_additivity_error": additivity_error,
                       "model_sha256": sha256(model_path)})
        for j, col in enumerate(features):
            values = features[col]
            corr = np.nan
            if col not in cats and values.nunique(dropna=True) > 1 and np.ptp(phi[:, j]) > 0:
                corr = float(pd.Series(values.to_numpy()).corr(pd.Series(phi[:, j]), method="spearman"))
            shap_rows.append({"quarter": quarter, "feature": col, "group": feature_group(col),
                "mean_abs_shap": float(np.mean(np.abs(phi[:, j]))), "mean_shap": float(phi[:, j].mean()),
                "p95_abs_shap": float(np.quantile(np.abs(phi[:, j]), .95)),
                "fraction_positive_shap": float((phi[:, j] > 0).mean()),
                "numeric_spearman_value_shap": corr})
            if col in cats:
                level = pd.DataFrame({"level": values.to_numpy(), "shap": phi[:, j], "target": target})
                for value, part in level.groupby("level", sort=True):
                    level_rows.append({"quarter": quarter, "feature": col, "level": value,
                        "n": len(part), "mean_shap": float(part.shap.mean()),
                        "positive_rate": float(part.target.mean())})
        for group, columns in groups.items():
            jj = [features.columns.get_loc(c) for c in columns]
            group_phi = phi[:, jj].sum(axis=1)
            group_rows.append({"quarter": quarter, "group": group, "n_features": len(columns),
                "mean_abs_net_shap": float(np.abs(group_phi).mean()),
                "sum_feature_mean_abs_shap": float(np.abs(phi[:, jj]).mean(axis=0).sum()),
                "mean_net_shap": float(group_phi.mean())})
            for repeat in range(args.repeats):
                perm = np.random.default_rng(20261116 + repeat).permutation(len(features))
                changed = features.copy()
                # Same row permutation preserves all within-group relationships.
                for col in columns:
                    changed[col] = features[col].iloc[perm].to_numpy()
                p = model.predict_proba(changed, thread_count=4)[:, 1]
                ap = float(average_precision_score(target, p))
                permutation_rows.append({"quarter": quarter, "group": group, "repeat": repeat,
                    "n_features": len(columns), "original_ap": base_ap, "permuted_ap": ap,
                    "ap_drop": base_ap - ap})
        rank = np.empty(len(features), dtype=int)
        rank[np.argsort(-probability, kind="stable")] = np.arange(len(features))
        top20 = rank < int(np.ceil(.2 * len(features)))
        bottom20 = rank >= int(np.floor(.8 * len(features)))
        cohorts = {"all_positive": target == 1, "all_negative": target == 0,
            "top20_true_positive": top20 & (target == 1), "top20_false_positive": top20 & (target == 0),
            "below_top20_positive": ~top20 & (target == 1), "bottom20_positive": bottom20 & (target == 1)}
        for cohort, mask in cohorts.items():
            part = features.loc[mask]
            record = {"quarter": quarter, "cohort": cohort, "n": int(mask.sum()),
                      "mean_prediction": float(probability[mask].mean())}
            for col in important_profile_features:
                record[col + "_median"] = float(part[col].median()) if part[col].notna().any() else np.nan
            for col in ["risk_level", "facility_type", "lag1_result", "last_routine_result"]:
                for value in features[col].unique():
                    record[f"{col}={value}_fraction"] = float((part[col] == value).mean())
            error_rows.append(record)
        print(json.dumps(checks[-1]), flush=True)

    frames = {"shap_features_by_quarter": pd.DataFrame(shap_rows),
              "shap_groups_by_quarter": pd.DataFrame(group_rows),
              "categorical_level_shap": pd.DataFrame(level_rows),
              "group_permutation_repeats": pd.DataFrame(permutation_rows),
              "error_profiles": pd.DataFrame(error_rows)}
    for filename, frame in frames.items():
        frame.to_csv(args.out / (filename + ".csv"), index=False)
    mean_shap = frames["shap_features_by_quarter"].groupby(["feature", "group"], as_index=False).agg(
        mean_abs_shap=("mean_abs_shap", "mean"), mean_shap=("mean_shap", "mean"),
        numeric_spearman_value_shap=("numeric_spearman_value_shap", "mean"))
    mean_shap.sort_values("mean_abs_shap", ascending=False).to_csv(args.out / "shap_feature_summary.csv", index=False)
    perms = frames["group_permutation_repeats"].groupby(["quarter", "group"], as_index=False).agg(
        ap_drop_mean=("ap_drop", "mean"), ap_drop_repeat_std=("ap_drop", "std"))
    perms.to_csv(args.out / "group_permutation_by_quarter.csv", index=False)
    perm_summary = perms.groupby("group", as_index=False).agg(
        mean_quarter_ap_drop=("ap_drop_mean", "mean"), min_quarter_ap_drop=("ap_drop_mean", "min"),
        max_quarter_ap_drop=("ap_drop_mean", "max"))
    perm_summary.sort_values("mean_quarter_ap_drop", ascending=False).to_csv(args.out / "group_permutation_summary.csv", index=False)
    order = perm_summary.sort_values("mean_quarter_ap_drop").group.tolist()
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for offset, (quarter, color) in enumerate(zip(QUARTERS, ["#4477AA", "#EE6677", "#228833"])):
        pp = perms.loc[perms.quarter == quarter].set_index("group").reindex(order)
        ax.scatter(pp.ap_drop_mean, np.arange(len(order)) + (offset - 1) * .16,
                   color=color, s=28, label=quarter)
    ax.axvline(0, color="#888888", linewidth=1)
    ax.set_yticks(np.arange(len(order)), order)
    ax.set_xlabel("Average Precision drop after joint group permutation")
    ax.set_title("Held-out model reliance, five permutations per quarter")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(args.out / "group_permutation.png", dpi=160)
    plt.close(fig)
    top = mean_shap.sort_values("mean_abs_shap", ascending=False).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top.feature, top.mean_abs_shap, color="#4477AA")
    ax.set_xlabel("Mean absolute SHAP (raw log-odds), equal-quarter average")
    ax.set_title("Leading features in saved forward-validation models")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(args.out / "shap_feature_summary.png", dpi=160)
    plt.close(fig)
    result = {"quarters": QUARTERS, "checks": checks, "groups": groups,
        "n_features": len(X.columns), "shap_units": "raw log-odds contribution",
        "permutation_repeats": args.repeats, "fit_calls": 0, "closed_holdout_labels_loaded": False,
        "caveats": ["SHAP explains learned predictions, not causal effects or feature-removal gains.",
            "Correlated features share/substitute attribution; joint group permutation preserves within-group correlation but breaks cross-group relationships.",
            "Permutation repeat standard deviations are Monte Carlo variability, not confidence intervals for future AP.",
            "Only three existing development quarters; no final-model or in-sample importance is used."],
        "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__)),
                          "src/hw4_inspections/features.py": sha256(source)}}
    (args.out / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(perm_summary.sort_values("mean_quarter_ap_drop", ascending=False).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
