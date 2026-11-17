"""Refit the measured context-only extension; no upload, no lockbox scoring."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hw4_inspections.features import build_features
from hw4_inspections.extensions import build_context_features


def main():
    out = ROOT / "outputs/hw4/context-features-20261116"
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / "data/hw4/raw/train.csv")
    test = pd.read_csv(ROOT / "data/hw4/raw/test_student.csv")
    history = pd.read_csv(ROOT / "data/hw4/raw/inspection_history.csv")
    rows = pd.concat([train.drop(columns="target"), test], ignore_index=True)
    base, cats = build_features(rows, history)
    context = build_context_features(rows, history)
    X = pd.concat([base, context], axis=1)
    for c in cats:
        X[c] = X[c].fillna("__MISSING__").astype(str)
    protocol = json.loads((ROOT / "analysis/hw4-features-20261116/extensions_screen/protocol.json").read_text())
    assert list(X) == protocol["cases"]["plus_context"]
    stored = pd.read_csv(ROOT / "analysis/hw4-features-20261116/extensions_screen/predictions.csv")
    stored = stored.loc[stored.case == "plus_context"].set_index("row_id")
    errors = {}
    for quarter in ["2023-07-01", "2023-10-01", "2024-01-01"]:
        mask = train.forecast_period == quarter
        model = CatBoostClassifier()
        model.load_model(str(ROOT / f"analysis/hw4-features-20261116/extensions_screen/plus_context_seed42_{quarter}.cbm"))
        p = model.predict_proba(X.iloc[:len(train)].loc[mask])[:, 1]
        expected = stored.loc[train.loc[mask, "row_id"], "prediction"].to_numpy()
        np.testing.assert_allclose(p, expected, atol=2e-15, rtol=0)
        errors[quarter] = float(np.max(np.abs(p - expected)))
    params = json.loads((ROOT / "outputs/hw4/catboost-initial/manifest.json").read_text())["params"]
    model = CatBoostClassifier(**params)
    model.fit(X.iloc[:len(train)], train.target, cat_features=cats)
    model.save_model(str(out / "model.cbm"))
    p = model.predict_proba(X.iloc[len(train):])[:, 1]
    loaded = CatBoostClassifier()
    loaded.load_model(str(out / "model.cbm"))
    np.testing.assert_array_equal(p, loaded.predict_proba(X.iloc[len(train):])[:, 1])
    sample = pd.read_csv(ROOT / "data/hw4/raw/sample_submission.csv")
    submission = pd.DataFrame({"row_id": test.row_id, "target": p})
    assert submission.row_id.equals(sample.row_id)
    assert len(submission) == 2246 and submission.row_id.is_unique
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    submission.to_csv(out / "submission_context.csv", index=False)
    pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False).to_csv(out / "feature_importance.csv", index=False)
    source_paths = [Path(__file__), ROOT / "src/hw4_inspections/features.py", ROOT / "src/hw4_inspections/extensions.py"]
    manifest = {"candidate": "baseline274 +context72", "features": list(X), "categorical": cats,
                "params": params, "seed": 42, "n_features": X.shape[1], "train_rows": len(train),
                "test_rows": len(test), "versions": {"catboost": catboost.__version__, "numpy": np.__version__, "pandas": pd.__version__},
                "development_reproduction_max_abs_error": errors, "saved_model_predictions_exact": True,
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "data/hw4/raw").glob("*.csv")},
                "submission_sha256": hashlib.sha256((out / "submission_context.csv").read_bytes()).hexdigest(),
                "closed_quarters_scored": False, "closed_quarter_labels_in_final_refit": True,
                "agent_uploads": 0, "new_candidate_public_score": None}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"submission": str(out / "submission_context.csv"), "sha256": manifest["submission_sha256"],
                      "dev_forward_errors": errors, "saved_model_exact": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
