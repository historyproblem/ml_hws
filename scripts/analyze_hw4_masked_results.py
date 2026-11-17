"""Audit returned HW4 masked-run CSVs; score fixed blends without new fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
QUARTERS = ["2023-07-01", "2023-10-01", "2024-01-01"]
MODELS = ["group_supervised", "group_masked", "catboost_ssl64"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_ap(y, p):
    order = np.argsort(-p, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(p[order]) != 0), len(order) - 1]

    def score(weight):
        w = weight[order]
        tp = np.cumsum(y[order] * w)[ends]
        n = np.cumsum(w)[ends]
        assert tp[-1] > 0
        precision = np.divide(tp, n, out=np.zeros_like(tp), where=n > 0)
        return float(np.sum(np.diff(np.r_[0, tp]) * precision) / tp[-1])

    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incoming", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    incoming = args.out / "incoming"
    incoming.mkdir(exist_ok=True)
    filenames = ["protocol.json", "predictions.csv", "fold_metrics.csv", "summary.csv"]
    for name in filenames:
        target = incoming / name
        if target.exists():
            assert sha(target) == sha(args.incoming / name), "Use a new audit folder for changed inputs"
        else:
            shutil.copy2(args.incoming / name, target)
    protocol = json.loads((incoming / "protocol.json").read_text())
    assert protocol["status"] == "complete" and not protocol["config"]["smoke"]
    assert protocol["config"]["device"] == "cuda"
    assert protocol["config"]["quarters"] == QUARTERS
    expected_config = {"variants": ["supervised", "masked"], "pretrain_epochs": 20,
                       "max_epochs": 40, "patience": 8, "batch_size": 256, "width": 96,
                       "layers": 3, "mask_probability": .35, "seed": 42, "with_catboost": True}
    assert all(protocol["config"][key] == value for key, value in expected_config.items())
    assert not protocol["lockbox_scored"]
    assert not protocol["pretraining_uses_targets"]
    assert not protocol["pretraining_uses_validation_or_test_inputs"]
    signed = {key: protocol[key] for key in ["config", "sources", "inputs", "versions", "hardware"]}
    signature = hashlib.sha256(json.dumps(signed, sort_keys=True).encode()).hexdigest()
    assert signature == protocol["signature"]
    expected_sources = {"scripts/run_hw4_masked.py", *[f"src/hw4_inspections/{name}.py" for name in
                        ["features", "extensions", "neural", "masked", "masked_training"]]}
    assert set(protocol["sources"]) == expected_sources
    for name, digest in protocol["sources"].items():
        source = args.source_root / name
        assert sha(source) == digest, f"Source mismatch: {name}"
        snapshot = args.out / "verified_sources" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, snapshot)
    assert set(protocol["inputs"]) == {"train.csv", "test_student.csv", "inspection_history.csv", "sample_submission.csv"}
    for name, digest in protocol["inputs"].items():
        assert sha(args.raw / name) == digest, f"Input mismatch: {name}"

    train = pd.read_csv(args.raw / "train.csv")
    official = train.loc[train.forecast_period.isin(QUARTERS)].set_index("row_id")
    pred = pd.read_csv(incoming / "predictions.csv", float_precision="round_trip")
    metrics = pd.read_csv(incoming / "fold_metrics.csv", float_precision="round_trip")
    summary = pd.read_csv(incoming / "summary.csv", float_precision="round_trip").set_index("model")
    assert set(summary.index) == set(MODELS) and summary.index.is_unique
    assert set(pred.model) == set(MODELS) and len(pred) == len(official) * len(MODELS)
    assert not pred.duplicated(["model", "row_id"]).any()
    assert len(metrics) == len(QUARTERS) * len(MODELS)
    assert not metrics.duplicated(["model", "quarter"]).any()
    assert metrics.loaded_forward_exact.all()
    assert metrics.loc[metrics.model.eq("catboost_ssl64"), "embedding_uses_targets"].astype(str).str.lower().eq("false").all()
    assert np.isfinite(pred.prediction).all() and pred.prediction.between(0, 1).all()
    combined = official[["site_id", "forecast_period", "target"]].copy()
    recomputed = []
    for model in MODELS:
        part = pred.loc[pred.model.eq(model)].set_index("row_id")
        assert set(part.index) == set(official.index)
        part = part.loc[official.index]
        for name in ["site_id", "forecast_period", "target"]:
            np.testing.assert_array_equal(part[name], official[name])
        combined[model] = part.prediction
        for quarter in QUARTERS:
            fold = part.loc[part.forecast_period.eq(quarter)]
            ap = average_precision_score(fold.target, fold.prediction)
            recorded = metrics.loc[metrics.model.eq(model) & metrics.quarter.eq(quarter)].iloc[0]
            assert abs(ap - recorded.ap) < 1e-14
            assert recorded.valid_rows == len(fold)
            assert recorded.train_rows == int(train.forecast_period.lt(quarter).sum())
            assert recorded.seed == protocol["config"]["seed"]
            assert 1 <= recorded.selected_epochs <= protocol["config"]["max_epochs"]
            recomputed.append({"model": model, "quarter": quarter, "ap": ap})
        mean = np.mean([row["ap"] for row in recomputed if row["model"] == model])
        assert abs(mean - summary.loc[model, "mean_ap"]) < 1e-14
        assert summary.loc[model, "folds"] == 3
    baseline = pd.read_csv(args.baseline, float_precision="round_trip")
    baseline = baseline.loc[baseline["case"].eq("plus_context") & baseline.seed.eq(42)].set_index("row_id")
    assert set(baseline.index) == set(official.index)
    baseline = baseline.loc[official.index]
    for name in ["site_id", "forecast_period", "target"]:
        np.testing.assert_array_equal(baseline[name], official[name])
    combined["catboost"] = baseline.prediction
    for model in MODELS:
        combined[model + "_blend50"] = .5 * combined[model] + .5 * combined.catboost
    models = ["catboost", *MODELS, *[m + "_blend50" for m in MODELS]]
    scores, prepared = [], {}
    for quarter in QUARTERS:
        idx = np.flatnonzero(combined.forecast_period.eq(quarter).to_numpy())
        y = combined.target.to_numpy(dtype=float)[idx]
        for model in models:
            p = combined[model].to_numpy()[idx]
            prepared[quarter, model] = prepare_ap(y, p)
            ap = average_precision_score(y, p)
            assert abs(ap - prepared[quarter, model](np.ones(len(y)))) < 1e-14
            scores.append({"model": model, "quarter": quarter, "ap": ap})
    scores = pd.DataFrame(scores)
    means = scores.groupby("model").ap.mean().to_dict()
    sites, unique = pd.factorize(combined.site_id)
    rng = np.random.default_rng(20261117)
    bootstrap = np.empty((args.bootstrap, len(models)))
    for repeat in range(args.bootstrap):
        weights = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)))[sites]
        for j, model in enumerate(models):
            values = []
            for quarter in QUARTERS:
                mask = combined.forecast_period.eq(quarter).to_numpy()
                values.append(prepared[quarter, model](weights[mask]))
                if repeat < 3:
                    expected = average_precision_score(combined.loc[mask, "target"], combined.loc[mask, model], sample_weight=weights[mask])
                    assert abs(expected - values[-1]) < 1e-14
            bootstrap[repeat, j] = np.mean(values)
    rows = []
    for j, model in enumerate(models):
        delta = bootstrap[:, j] - bootstrap[:, 0]
        rows.append({"model": model, "mean_ap": means[model], "delta_vs_catboost": means[model] - means["catboost"],
                     "ci95_low": float(np.quantile(delta, .025)), "ci95_high": float(np.quantile(delta, .975))})
    comparison = pd.DataFrame(rows).sort_values("mean_ap", ascending=False)
    difference = bootstrap[:, models.index("group_masked")] - bootstrap[:, models.index("group_supervised")]
    audit = {"incoming_sha256": {name: sha(incoming / name) for name in filenames}, "baseline_sha256": sha(args.baseline),
             "protocol_signature_verified": True, "all_6_sources_match": True, "all_4_inputs_match": True,
             "prediction_rows": len(pred), "unique_validation_rows": len(official), "positive_rows": int(official.target.sum()),
             "all_labels_sites_quarters_match_official": True, "all_9_ap_and_3_summary_means_reproduced": True,
             "reported_checkpoint_forward_checks": bool(metrics.loaded_forward_exact.all()),
             "checkpoint_weights_independently_replayed": False,
             "bootstrap": {"repeats": args.bootstrap, "unit": "site_id", "seed": 20261117,
                           "quarter_weights": "shared across models and quarters", "confidence": "nominal development CI, no selection correction"},
             "masked_minus_supervised": {"delta": means["group_masked"] - means["group_supervised"],
                                          "ci95": np.quantile(difference, [.025, .975]).tolist()},
             "blend_note": "Exploratory fixed 50/50 probability averages; no weights searched; same previously used development folds",
             "new_fits": 0, "kaggle_uploads": 0, "lockbox_scored": False}
    scores.to_csv(args.out / "quarter_scores.csv", index=False)
    comparison.to_csv(args.out / "comparison.csv", index=False)
    combined.reset_index().to_csv(args.out / "comparison_predictions.csv", index=False)
    (args.out / "verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(comparison.to_string(index=False), flush=True)
    print(json.dumps(audit["masked_minus_supervised"]), flush=True)
    print("Provenance, predictions and reported metrics verified; new fits: 0", flush=True)


if __name__ == "__main__":
    main()
