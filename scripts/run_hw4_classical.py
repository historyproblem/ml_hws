"""Fixed classical HW4 screen on forward folds, with resumable verified jobs.

No test refit, closed-quarter evaluation or Kaggle uploads. CPU by design.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import sys
import tarfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hw4_inspections.classical_analysis import analyze_predictions
from hw4_inspections.classical_features import (build_relative_features, recency_weights,
                                               select_feature_columns)
from hw4_inspections.classical_models import SPECS, choose_iterations, fit_model, predict_model
from hw4_inspections.extensions import build_context_features
from hw4_inspections.features import build_features

QUARTERS = ["2023-07-01", "2023-10-01", "2024-01-01"]
RAW_FILES = ["train.csv", "test_student.csv", "inspection_history.csv", "sample_submission.csv"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare_development(raw, smoke=False):
    train = pd.read_csv(raw / "train.csv")
    # Discard lockbox rows before feature construction, selection or scoring.
    train = train.loc[train.forecast_period <= QUARTERS[-1]].reset_index(drop=True)
    if train.row_id.duplicated().any():
        raise ValueError("Duplicate train row_id")
    if smoke:
        train = train.groupby("forecast_period", group_keys=False).head(48).reset_index(drop=True)
    history = pd.read_csv(raw / "inspection_history.csv")
    rows = train.drop(columns="target")
    frame, cats = build_features(rows, history)
    frame = pd.concat([frame, build_context_features(rows, history)], axis=1)
    # Relative features are optional and never enter the frozen reference.
    relative = build_relative_features(frame)
    return train, frame, relative, cats


def aligned_reference(path, rows, seed):
    """Reuse old full-precision OOF only after exact official identity alignment."""
    old = pd.read_csv(path, float_precision="round_trip")
    if "case" in old:
        old = old.loc[old.case == "plus_context"]
    elif "model" in old:
        old = old.loc[old.model == "cb_reference"]
    if "seed" in old:
        old = old.loc[old.seed == seed]
    elif seed != 42:
        return None
    if not len(old):
        return None
    if old.row_id.duplicated().any():
        raise ValueError("Duplicate row_id in reference")
    needed = rows.loc[rows.forecast_period.isin(QUARTERS)]
    old = old.set_index("row_id").reindex(needed.row_id)
    for column in ["site_id", "forecast_period", "target"]:
        if column not in old or not np.array_equal(old[column].to_numpy(), needed[column].to_numpy()):
            raise ValueError(f"Reference does not align on {column}")
    p = old.prediction.to_numpy(dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid reference predictions")
    return pd.Series(p, index=needed.row_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "analysis/hw4-classical-01")
    parser.add_argument("--models", nargs="+", choices=list(SPECS), default=list(SPECS))
    parser.add_argument("--quarters", nargs="+", choices=QUARTERS, default=QUARTERS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--reference-oof", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.bootstrap < 1:
        raise ValueError("threads and bootstrap must be positive")
    if len(set(args.models)) != len(args.models) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate models/seeds")
    if len(set(args.quarters)) != len(args.quarters):
        raise ValueError("Duplicate quarters")
    if "cb_reference" not in args.models:
        raise ValueError("Include cb_reference for matched comparison")
    started = time.monotonic()

    def log(event, **details):
        print(json.dumps({"event": event, "elapsed_seconds": round(time.monotonic() - started, 3),
                          **details}), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    versions = {name: importlib.metadata.version(name) for name in
                ["numpy", "pandas", "scikit-learn", "catboost", "lightgbm", "joblib", "scipy", "threadpoolctl"]}
    sources = [Path(__file__), *(ROOT / "src/hw4_inspections" / name for name in
                ["features.py", "extensions.py", "classical_features.py", "classical_models.py", "classical_analysis.py"])]
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items() if key != "resume"}
    provenance = {"config": config, "specs": {k: SPECS[k] for k in args.models}, "versions": versions,
                  "python": platform.python_version(), "machine": platform.machine(), "backend": "CPU",
                  "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
                  "input_sha256": {name: sha(args.raw / name) for name in RAW_FILES},
                  "reference_sha256": sha(args.reference_oof) if args.reference_oof else None}
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    protocol_path = args.out / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        if not args.resume or previous["signature"] != signature:
            raise ValueError("Use new --out, or --resume with unchanged code/data/config/environment")
    protocol = {**provenance, "signature": signature, "status": "running", "lockbox_scored": False,
                "test_refits": 0, "kaggle_uploads": 0,
                "selection": "Exploratory screen on old dev; fixed blends, no stacking or weight search"}
    write_json(protocol_path, protocol)
    log("cpu_ready", threads=args.threads, models=args.models, seeds=args.seeds,
        note="GPU idle is expected in this classical screen")
    cache = args.out / "features.joblib"
    cache_meta = args.out / "features.json"
    if args.resume and cache.exists() and cache_meta.exists():
        check = json.loads(cache_meta.read_text())
        if check != {"signature": signature, "sha256": sha(cache)}:
            raise ValueError("Feature cache changed; use a new output directory")
        train, frame, relative, cats = joblib.load(cache)
        log("feature_cache_reused", rows=len(train), columns=frame.shape[1])
    else:
        log("features_start")
        train, frame, relative, cats = prepare_development(args.raw, args.smoke)
        joblib.dump((train, frame, relative, cats), cache)
        write_json(cache_meta, {"signature": signature, "sha256": sha(cache)})
        log("features_complete", rows=len(train), columns=frame.shape[1])
    references = {s: aligned_reference(args.reference_oof, train, s)
                  if args.reference_oof else None for s in args.seeds}
    records, predictions, reused = [], [], 0
    for name in args.models:
        spec = copy.deepcopy(SPECS[name])
        if args.smoke:
            spec["max_iterations"] = 5
        feature_set = spec["feature_set"]
        columns = select_feature_columns(frame, "compact_context" if feature_set == "compact_context" else "context")
        x = frame[columns].copy()
        if feature_set == "context_relative":
            x = pd.concat([x, relative], axis=1)
        model_cats = [c for c in cats if c in x]
        if spec.get("add_site", False):
            x["site_id"] = train.site_id.fillna("<none>").astype(str)
            model_cats.append("site_id")
        for seed in args.seeds:
            for quarter in args.quarters:
                fit = train.forecast_period < quarter
                valid = train.forecast_period == quarter
                if not fit.any() or not valid.any():
                    raise ValueError("Empty temporal fold")
                job = args.out / f"{name}_s{seed}_{quarter}"
                job.mkdir(exist_ok=True)
                marker = job / "complete.json"
                if args.resume and marker.exists():
                    record = json.loads(marker.read_text())
                    if record["signature"] != signature or any(sha(job / p) != h for p, h in record["artifacts"].items()):
                        raise ValueError(f"Completed job changed: {job}")
                    fold = pd.read_csv(job / "predictions.csv", float_precision="round_trip")
                    reused += 1
                    log("job_reused", model=name, seed=seed, quarter=quarter)
                else:
                    log("fit_start", model=name, seed=seed, quarter=quarter, train_n=int(fit.sum()),
                        valid_n=int(valid.sum()), features=x.shape[1])
                    t0 = time.monotonic()
                    fold = train.loc[valid, ["row_id", "site_id", "forecast_period", "target"]].copy()
                    fold["seen_site"] = fold.site_id.isin(train.loc[fit, "site_id"])
                    fold["history_violation_mean"] = frame.loc[valid, "history_violation_mean"].to_numpy()
                    reference = references[seed] if name == "cb_reference" else None
                    inner_quarter = sorted(train.loc[fit, "forecast_period"].unique())[-1]
                    iterations, curve = spec["max_iterations"], []
                    if reference is not None:
                        p = reference.reindex(fold.row_id).to_numpy()
                        forward_exact = None  # Already saved OOF; do not claim a new checkpoint test.
                        source = "reused_reference_oof"
                    else:
                        inner_fit = train.forecast_period < inner_quarter
                        inner_valid = train.forecast_period == inner_quarter
                        weights = recency_weights(train.loc[inner_fit, "forecast_period"], inner_quarter) if spec.get("recency") else None
                        with threadpool_limits(limits=args.threads):
                            iterations, curve = choose_iterations(spec, x.loc[inner_fit], train.loc[inner_fit, "target"],
                                x.loc[inner_valid], train.loc[inner_valid, "target"], model_cats,
                                seed, args.threads, sample_weight=weights)
                            weights = recency_weights(train.loc[fit, "forecast_period"], quarter) if spec.get("recency") else None
                            bundle = fit_model(spec, x.loc[fit], train.loc[fit, "target"], model_cats,
                                               seed, args.threads, iterations=iterations, sample_weight=weights)
                            p = predict_model(bundle, x.loc[valid])
                            joblib.dump(bundle, job / "model.joblib")
                            restored = predict_model(joblib.load(job / "model.joblib"), x.loc[valid])
                        forward_exact = bool(np.array_equal(p, restored))
                        if not forward_exact:
                            raise AssertionError("Saved model does not reproduce prediction exactly")
                        source = "new_fit"
                    fold["prediction"] = p
                    fold["model"] = name
                    fold["seed"] = seed
                    fold.to_csv(job / "predictions.csv", index=False)
                    write_json(job / "inner_curve.json", curve)
                    record = {"signature": signature, "model": name, "seed": seed, "quarter": quarter,
                              "train_n": int(fit.sum()), "valid_n": int(valid.sum()), "features": list(x),
                              "iterations": int(iterations), "inner_quarter": inner_quarter,
                              "ap": float(average_precision_score(fold.target, p)),
                              "seconds": time.monotonic() - t0, "source": source,
                              "loaded_forward_exact": forward_exact,
                              "artifacts": {filename: sha(job / filename) for filename in
                                            ["predictions.csv", "inner_curve.json", "model.joblib"]
                                            if (job / filename).is_file()}}
                    write_json(marker, record)
                    log("fit_complete", model=name, seed=seed, quarter=quarter, ap=record["ap"],
                        iterations=int(iterations), seconds=round(record["seconds"], 2), loaded_forward_exact=forward_exact)
                records.append(record)
                predictions.append(fold)
    raw_predictions = pd.concat(predictions, ignore_index=True)
    raw_predictions.to_csv(args.out / "predictions_by_seed.csv", index=False)
    seed_metrics = pd.DataFrame([{k: r[k] for k in ["model", "seed", "quarter", "ap", "iterations", "seconds", "source", "loaded_forward_exact"]} for r in records])
    seed_metrics.to_csv(args.out / "fold_metrics.csv", index=False)
    keys = ["model", "row_id", "site_id", "forecast_period", "target", "seen_site", "history_violation_mean"]
    averaged = raw_predictions.groupby(keys, dropna=False, sort=False, as_index=False).prediction.mean()
    log("analysis_start", bootstrap=args.bootstrap)
    result = analyze_predictions(averaged, bootstrap=args.bootstrap)
    for key in ["predictions", "quarter_scores", "summary", "diagnostics"]:
        result[key].to_csv(args.out / f"{key}.csv", index=False)
    write_json(args.out / "comparison.json", result["comparison"])
    protocol.update(status="complete", completed_jobs=len(records), reused_jobs=reused,
                    elapsed_seconds=time.monotonic() - started)
    write_json(protocol_path, protocol)
    report_files = ["protocol.json", "comparison.json", "fold_metrics.csv", "predictions_by_seed.csv",
                    "predictions.csv", "quarter_scores.csv", "summary.csv", "diagnostics.csv"]
    with tarfile.open(args.out / "return_results.tar.gz", "w:gz") as archive:
        for filename in report_files:
            archive.add(args.out / filename, arcname=filename)
    print(result["summary"].to_string(index=False), flush=True)
    log("complete", jobs=len(records), reused=reused, results=str(args.out / "return_results.tar.gz"),
        lockbox_scored=False, test_refits=0, kaggle_uploads=0)


if __name__ == "__main__":
    main()
