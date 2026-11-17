"""Controlled HW4 feature-group refits, same frozen development folds/params.

No closed-quarter scoring or automatic Kaggle submissions. Run from repo root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hw4_inspections.features import build_features

QUARTERS = ["2023-07-01", "2023-10-01", "2024-01-01"]
PARAMS = dict(iterations=500, depth=5, learning_rate=0.04, l2_leaf_reg=10,
              loss_function="Logloss", thread_count=4, verbose=False, allow_writing_files=False)


def feature_groups(columns):
    groups = {k: [] for k in ["static", "calendar", "codes", "burden", "inspection_types", "outcomes", "timing"]}
    for c in columns:
        if c in {"facility_type", "risk_level", "area_id"}:
            name = "static"
        elif c in {"forecast_year", "forecast_quarter", "cutoff_gap_days"}:
            name = "calendar"
        elif c.startswith("code"):
            name = "codes"
        elif "violation" in c:
            name = "burden"
        elif c.startswith("type_") or "inspection_type" in c or "reinspection_count" in c:
            name = "inspection_types"
        elif c.endswith("_result") or any(f"_{r}_" in c and "age_days" not in c for r in ["pass", "fail", "conditional"]):
            name = "outcomes"
        else:
            name = "timing"
        groups[name].append(c)
    assert sum(map(len, groups.values())) == len(columns)
    return groups


def configurations(columns, groups):
    cases = {"baseline": list(columns)}
    for group in ["codes", "burden", "inspection_types", "outcomes", "timing", "calendar"]:
        cases["drop_" + group] = [c for c in columns if c not in groups[group]]
    cases["drop_area"] = [c for c in columns if c != "area_id"]
    cases["static_plus_burden"] = [c for c in columns if c in groups["static"] + groups["calendar"] + groups["burden"]]
    cases["compact_no_codes_types"] = [c for c in columns if c not in groups["codes"] + groups["inspection_types"]]
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "analysis/hw4-features-20261116/ablation")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--extensions", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raw = ROOT / "data/hw4/raw"
    # Drop closed-period rows before accessing any labels.
    train = pd.read_csv(raw / "train.csv")
    train = train.loc[train.forecast_period <= QUARTERS[-1]].reset_index(drop=True)
    history = pd.read_csv(raw / "inspection_history.csv")
    rows = train.drop(columns="target")
    X, cats = build_features(rows, history)
    for c in cats:
        X[c] = X[c].fillna("__MISSING__").astype(str)
    groups = feature_groups(X.columns)
    cases = configurations(list(X), groups)
    if args.extensions:
        from hw4_inspections.extensions import build_context_features, build_dynamics_features
        context = build_context_features(rows, history)
        dynamics = build_dynamics_features(rows, history)
        original = list(X)
        for extra in [context, dynamics]:
            assert extra.index.equals(X.index)
            assert not set(extra) & set(X)
            X = pd.concat([X, extra], axis=1)
        cases.update({"plus_context": original + list(context), "plus_dynamics": original + list(dynamics),
                      "plus_context_dynamics": list(X)})
    if args.cases:
        cases = {k: cases[k] for k in args.cases}
    metadata = {"params": PARAMS, "seeds": args.seeds, "quarters": QUARTERS,
                "lockbox_scored": False, "groups": groups, "cases": cases,
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [Path(__file__), ROOT / "src/hw4_inspections/features.py"]},
                "selection_note": "Exploratory group screen; seed confirmations are stability checks on the same dev quarters, not independent holdout."}
    (args.out / "protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")
    baseline = pd.read_csv(ROOT / "outputs/hw4/catboost-initial/development_predictions.csv")
    baseline = baseline.loc[baseline.model == "history_catboost"].set_index("row_id")
    metrics, predictions = [], []
    for case, columns in cases.items():
        for seed in args.seeds:
            for quarter in QUARTERS:
                fit = train.forecast_period < quarter
                valid = train.forecast_period == quarter
                target = train.loc[valid, "target"]
                t0 = time.monotonic()
                modelpath = args.out / f"{case}_seed{seed}_{quarter}.cbm"
                if case == "baseline" and seed == 42:
                    p = baseline.loc[train.loc[valid, "row_id"], "prediction"].to_numpy()
                else:
                    model = CatBoostClassifier(**PARAMS, random_seed=seed)
                    model.fit(X.loc[fit, columns], train.loc[fit, "target"],
                              cat_features=[c for c in cats if c in columns])
                    p = model.predict_proba(X.loc[valid, columns])[:, 1]
                    model.save_model(str(modelpath))
                ap = float(average_precision_score(target, p))
                b = baseline.loc[train.loc[valid, "row_id"], "prediction"]
                baseap = float(average_precision_score(target, b))
                row = dict(case=case, seed=seed, quarter=quarter, n_features=len(columns),
                           ap=ap, delta_from_original_seed42=ap-baseap,
                           logloss=float(log_loss(target, p)), seconds=time.monotonic()-t0)
                metrics.append(row)
                pred = train.loc[valid, ["row_id", "site_id", "forecast_period", "target"]].copy()
                pred["case"], pred["seed"], pred["prediction"] = case, seed, p
                predictions.append(pred)
                pd.DataFrame(metrics).to_csv(args.out / "fold_metrics.csv", index=False)
                print(json.dumps(row), flush=True)
    pd.concat(predictions, ignore_index=True).to_csv(args.out / "predictions.csv", index=False)
    frame = pd.DataFrame(metrics)
    summary = frame.groupby(["case", "seed"], as_index=False).agg(
        mean_ap=("ap", "mean"), mean_delta_from_original_seed42=("delta_from_original_seed42", "mean"),
        n_features=("n_features", "first"))
    summary.to_csv(args.out / "summary.csv", index=False)
    print(summary.sort_values("mean_ap", ascending=False).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
