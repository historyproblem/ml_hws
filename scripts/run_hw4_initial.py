"""Fixed first HW4 experiment; the 2024-Q2/Q3 lockbox is not scored.

Run from the project root with .venv/bin/python scripts/run_hw4_initial.py.
This script never contacts Kaggle or consumes a submission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import importlib.metadata
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hw4_inspections.features import build_features

DEVELOPMENT_QUARTERS = ["2023-07-01", "2023-10-01", "2024-01-01"]
LOCKBOX_QUARTERS = ["2024-04-01", "2024-07-01"]
PARAMS = dict(iterations=500, depth=5, learning_rate=0.04, l2_leaf_reg=10,
              loss_function="Logloss", random_seed=42, thread_count=4,
              verbose=False, allow_writing_files=False)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y, p):
    return {"ap": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)),
            "logloss": float(log_loss(y, p)), "n": len(y),
            "positive": int(np.sum(y))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/hw4/raw")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/hw4/initial")
    parser.add_argument("--backend", choices=["catboost", "histgradient"], default="catboost")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.raw / "train.csv")
    test = pd.read_csv(args.raw / "test_student.csv")
    history = pd.read_csv(args.raw / "inspection_history.csv")
    # The feature builder never receives a target column.
    rows = pd.concat([train.drop(columns="target"), test], ignore_index=True)
    started = time.monotonic()
    X, categorical = build_features(rows, history)
    assert len(X) == len(rows) and X.index.equals(rows.index)
    assert not set(X) & {"row_id", "site_id", "target", "history_before", "forecast_period"}
    assert X.columns.is_unique
    for c in categorical:
        X[c] = X[c].fillna("__MISSING__").astype(str)
    numeric = X.columns.difference(categorical)
    X[numeric] = X[numeric].replace([np.inf, -np.inf], np.nan)
    static = ["facility_type", "risk_level", "area_id", "forecast_quarter", "forecast_year", "cutoff_gap_days"]
    missing = set(static) - set(X)
    if missing:
        raise ValueError(f"Static feature naming mismatch: {missing}")
    feature_sets = {f"static_{args.backend}": static, f"history_{args.backend}": list(X)}
    if args.backend == "catboost":
        from catboost import CatBoostClassifier
        params = PARAMS
    else:
        params = dict(max_iter=300, max_depth=5, max_leaf_nodes=25,
                      learning_rate=0.05, l2_regularization=10,
                      min_samples_leaf=30, early_stopping=False, random_state=42)

    def fit_model(features, labels, cats):
        if args.backend == "catboost":
            model = CatBoostClassifier(**params)
            model.fit(features, labels, cat_features=cats)
        else:
            # Vocabularies are fitted only on the fold training rows.
            transformer = ColumnTransformer([
                ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), cats),
                ("numeric", "passthrough", [c for c in features if c not in cats])])
            categorical_mask = [True] * len(cats) + [False] * (features.shape[1] - len(cats))
            model = make_pipeline(transformer, HistGradientBoostingClassifier(
                **params, categorical_features=categorical_mask))
            with threadpool_limits(limits=4):
                model.fit(features, labels)
        return model

    def predict(model, features):
        with threadpool_limits(limits=4):
            return model.predict_proba(features)[:, 1]

    def save_model(model, filename):
        if args.backend == "catboost":
            model.save_model(str(filename.with_suffix(".cbm")))
        else:
            joblib.dump(model, filename.with_suffix(".joblib"))
    records, predictions = [], []
    y = train.target.to_numpy()
    train_period = train.forecast_period
    Xtrain = X.iloc[:len(train)]
    for model_name, columns in feature_sets.items():
        cats = [c for c in categorical if c in columns]
        for quarter in DEVELOPMENT_QUARTERS:
            fit = (train_period < quarter).to_numpy()
            valid = (train_period == quarter).to_numpy()
            assert train_period[fit].max() < quarter
            t0 = time.monotonic()
            model = fit_model(Xtrain.loc[fit, columns], y[fit], cats)
            p = predict(model, Xtrain.loc[valid, columns])
            record = {"model": model_name, "quarter": quarter,
                      "train_n": int(fit.sum()), **metrics(y[valid], p),
                      "seconds": time.monotonic() - t0}
            records.append(record)
            fold = train.loc[valid, ["row_id", "site_id", "forecast_period", "target"]].copy()
            fold["prediction"] = p
            fold["model"] = model_name
            fold["seen_site"] = fold.site_id.isin(train.loc[fit, "site_id"])
            predictions.append(fold)
            save_model(model, args.out / f"{model_name}_{quarter}")
            print(json.dumps(record), flush=True)
            pd.DataFrame(records).to_csv(args.out / "cv_metrics.csv", index=False)
    cv = pd.DataFrame(records)
    means = cv.groupby("model").ap.mean().to_dict()
    # Two pre-specified candidates, equal-quarter mean AP. No LB feedback.
    selected = max(means, key=means.get)
    columns = feature_sets[selected]
    cats = [c for c in categorical if c in columns]
    final_model = fit_model(Xtrain[columns], y, cats)
    save_model(final_model, args.out / "model")
    p = predict(final_model, X.iloc[len(train):][columns])
    submission = pd.DataFrame({"row_id": test.row_id, "target": p})
    sample = pd.read_csv(args.raw / "sample_submission.csv")
    assert submission.row_id.equals(sample.row_id)
    assert submission.row_id.is_unique and len(submission) == 2246
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    submission.to_csv(args.out / "submission.csv", index=False)
    if args.backend == "catboost":
        loaded = CatBoostClassifier()
        loaded.load_model(str(args.out / "model.cbm"))
    else:
        loaded = joblib.load(args.out / "model.joblib")
    p2 = predict(loaded, X.iloc[len(train):][columns])
    np.testing.assert_array_equal(p, p2)
    pd.concat(predictions, ignore_index=True).to_csv(args.out / "development_predictions.csv", index=False)
    if args.backend == "catboost":
        pd.DataFrame({"feature": columns, "importance": final_model.feature_importances_}).sort_values(
            "importance", ascending=False).to_csv(args.out / "feature_importance.csv", index=False)
    result = {"selected_model": selected, "mean_quarter_ap": means,
              "development_quarters": DEVELOPMENT_QUARTERS,
              "lockbox_quarters_unscored": LOCKBOX_QUARTERS,
              "params": params, "backend": args.backend, "features": columns, "categorical": cats,
              "n_features": len(columns), "training_rows": len(train), "test_rows": len(test),
              "runtime_seconds": time.monotonic() - started,
              "versions": {"python": platform.python_version(),
                           "catboost": importlib.metadata.version("catboost") if args.backend == "catboost" else None,
                           "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
              "input_sha256": {p.name: sha256(p) for p in sorted(args.raw.glob("*.csv"))},
              "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in [
                  Path(__file__), ROOT / "src/hw4_inspections/features.py"]},
              "submission_sha256": sha256(args.out / "submission.csv"),
              "loaded_model_predictions_exact": True, "kaggle_uploads": 0}
    (args.out / "manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"selected": selected, "mean_quarter_ap": means,
                      "submission": str(args.out / "submission.csv")}), flush=True)


if __name__ == "__main__":
    main()
