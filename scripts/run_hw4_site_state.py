"""One fixed latest-site-state feature experiment on the three old dev folds."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hw4_inspections.classical_models import SPECS, fit_model, predict_model
from hw4_inspections.delivery import validate_submission
from hw4_inspections.extensions import build_context_features
from hw4_inspections.features import build_features
from hw4_inspections.next_round import QUARTERS, evaluate_candidates, rank_scores
from hw4_inspections.site_state import STATE_GROUPS, SMOOTHING_SITES, build_site_state_features

RAW_FILES = ("train.csv", "test_student.csv", "inspection_history.csv", "sample_submission.csv")
SEEDS = (42, 17, 2026)
CANDIDATES = ("site_state", "context_site_state_75_25")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def probabilities(values, expected_n):
    values = np.asarray(values, dtype=np.float64)
    if (values.shape != (expected_n,) or not np.isfinite(values).all()
            or ((values < 0) | (values > 1)).any()):
        raise ValueError("Predictions must be finite row-aligned probabilities")
    return values


def roundtrip_csv(frame, path):
    frame.to_csv(path, index=False)
    loaded = pd.read_csv(path, float_precision="round_trip")
    if list(frame) != list(loaded) or len(frame) != len(loaded):
        raise AssertionError("CSV schema/row-count changed")
    for name in frame:
        if pd.api.types.is_float_dtype(frame[name]):
            if not np.array_equal(frame[name].to_numpy(dtype=np.float64), loaded[name].to_numpy(dtype=np.float64)):
                raise AssertionError(f"CSV float64 roundtrip changed {name}")
    return loaded


def features(rows, history):
    base, cats = build_features(rows, history)
    frame = pd.concat([base, build_context_features(rows, history),
                       build_site_state_features(rows, history)], axis=1)
    if len(frame.columns) != 370 or not frame.columns.is_unique:
        raise ValueError("The frozen 346+24 feature schema changed")
    if {"target", "site_id", "row_id", "forecast_period", "history_before"} & set(frame):
        raise ValueError("Raw targets or identifiers reached model features")
    return frame, cats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "analysis/hw4-site-state-20261121")
    parser.add_argument("--baseline-cv", type=Path, required=True)
    parser.add_argument("--context-test", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.bootstrap < 0 or (not args.smoke and args.bootstrap < 1):
        raise ValueError("Positive threads/bootstrap required; bootstrap=0 is smoke-only")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def log(event, **details):
        print(json.dumps({"event": event, "elapsed_seconds": round(time.monotonic() - started, 3),
                          **details}), flush=True)

    source_files = [Path(__file__), *(ROOT / "src/hw4_inspections" / name for name in
                    ["features.py", "extensions.py", "site_state.py", "classical_models.py",
                     "next_round.py", "classical_analysis.py", "delivery.py"])]
    spec = copy.deepcopy(SPECS["cb_reference"])
    spec["max_iterations"] = 5 if args.smoke else 500
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items() if key != "resume"}
    provenance = {
        "config": config, "input_sha256": {name: sha(args.raw / name) for name in RAW_FILES},
        "baseline_sha256": sha(args.baseline_cv), "test_anchor_sha256": sha(args.context_test),
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in source_files},
        "versions": {name: importlib.metadata.version(name) for name in
                     ["catboost", "numpy", "pandas", "scikit-learn", "joblib", "threadpoolctl"]},
        "python": platform.python_version(), "machine": platform.machine(), "backend": "CPU",
        "recipe": {"features": "full context346 + latest-site-state24 = 370", "state_groups": STATE_GROUPS,
                   "state_windows": ["all", "365d"], "smoothing_sites": SMOOTHING_SITES,
                   "state_selection": "Latest ALL routine per site/window; unknown latest outcome excludes site without older-known fallback",
                   "empty_state_group": "n_sites=0, fail/load rates NaN even if global prior exists",
                   "own_exclusion": "Focal site excluded from group AND global smoothing prior",
                   "model": spec, "seeds": list(SEEDS), "aggregation": "Float64 mean probabilities across all three seeds BEFORE ranking",
                   "blend": "0.75 context_3seed percentile rank + 0.25 site_state percentile rank within each complete quarter",
                   "gate": "Mean AP delta >0 and nonnegative delta in >=2/3 old quarters, independently per export",
                   "old_development_quarters": list(QUARTERS), "new_closed_quarter_metrics": False,
                   "final_refits": "Exactly three full-train fits only if at least one predeclared export is eligible",
                   "weight_search": False, "tree_tuning": False},
    }
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    protocol_path = args.out / "protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text())
        if not args.resume or old.get("signature") != signature:
            raise ValueError("Use a new --out or --resume with unchanged sources/data/config/environment")
        for name, digest in old.get("output_artifacts", {}).items():
            path = args.out / name
            if not path.is_file() or sha(path) != digest:
                raise ValueError(f"Changed or absent completed output: {name}")
    protocol = {**provenance, "signature": signature, "status": "running", "kaggle_uploads": 0,
                "new_closed_quarter_metrics": False}
    write_json(protocol_path, protocol)

    train = pd.read_csv(args.raw / "train.csv")
    test = pd.read_csv(args.raw / "test_student.csv")
    sample = pd.read_csv(args.raw / "sample_submission.csv")
    if args.smoke:
        train = train.groupby("forecast_period", group_keys=False).head(32).reset_index(drop=True)
    if train.row_id.duplicated().any() or train.target.isna().any() or not train.target.isin([0, 1]).all():
        raise ValueError("Official training IDs/labels are invalid")
    history = pd.read_csv(args.raw / "inspection_history.csv")
    # Closed-quarter rows are not even featurized for development decisions.
    dev = train.loc[train.forecast_period.le(QUARTERS[-1])].reset_index(drop=True)
    log("development_features_start", rows=len(dev))
    x, cats = features(dev.drop(columns="target"), history)
    meta = dev.loc[dev.forecast_period.isin(QUARTERS),
                   ["row_id", "site_id", "forecast_period", "target"]].reset_index(drop=True)
    baseline = pd.read_csv(args.baseline_cv, float_precision="round_trip")
    required = {"row_id", "site_id", "forecast_period", "target", "prediction"}
    if not required.issubset(baseline) or baseline.row_id.isna().any() or baseline.row_id.duplicated().any():
        raise ValueError("Baseline must have unique complete official metadata and predictions")
    if "model" in baseline and not baseline.model.eq("context_3seed").all():
        raise ValueError("Baseline must be the frozen context_3seed predictor")
    if not args.smoke and set(baseline.row_id) != set(meta.row_id):
        raise ValueError("Baseline must contain exactly the old-development IDs")
    aligned = baseline.set_index("row_id").reindex(meta.row_id)
    for name in ["site_id", "forecast_period", "target"]:
        if not np.array_equal(aligned[name].to_numpy(), meta[name].to_numpy()):
            raise ValueError(f"Baseline official metadata mismatch: {name}")
    base_probability = probabilities(aligned.prediction, len(meta))
    records, per_seed = [], []
    reused_dev, new_dev, reused_final, new_final = 0, 0, 0, 0

    def completed(job, expected, count):
        record = json.loads((job / "complete.json").read_text())
        if record.get("signature") != signature or any(record.get(k) != v for k, v in expected.items()):
            raise ValueError("Completed job signature/identity differs")
        for name, digest in record["artifacts"].items():
            if not (job / name).is_file() or sha(job / name) != digest:
                raise ValueError("Completed job artifact changed or disappeared")
        return record, probabilities(np.load(job / "prediction.npy", allow_pickle=False), count)

    for seed in SEEDS:
        for quarter in QUARTERS:
            fitting = np.flatnonzero(dev.forecast_period.lt(quarter))
            validation = np.flatnonzero(dev.forecast_period.eq(quarter))
            if not len(fitting) or not len(validation):
                raise ValueError("Each old-development quarter requires earlier training and validation rows")
            job = args.out / f"state_s{seed}_{quarter}"
            job.mkdir(exist_ok=True)
            marker = job / "complete.json"
            expected = {"stage": "old_development", "seed": seed, "quarter": quarter,
                        "row_ids_sha256": hashlib.sha256(json.dumps(dev.iloc[validation].row_id.tolist()).encode()).hexdigest()}
            if args.resume and marker.exists():
                record, probability = completed(job, expected, len(validation))
                reused_dev += 1
                log("job_reused", seed=seed, quarter=quarter)
            else:
                log("fit_start", seed=seed, quarter=quarter, rows=len(fitting))
                with threadpool_limits(limits=args.threads):
                    model = fit_model(spec, x.iloc[fitting], dev.iloc[fitting].target, cats, seed, args.threads)
                    probability = probabilities(predict_model(model, x.iloc[validation]), len(validation))
                    joblib.dump(model, job / "model.joblib")
                    check = predict_model(joblib.load(job / "model.joblib"), x.iloc[validation])
                if not np.array_equal(probability, check):
                    raise AssertionError("State model reload failed to reproduce development probabilities")
                np.save(job / "prediction.npy", probability, allow_pickle=False)
                record = {"signature": signature, **expected, "train_rows": len(fitting),
                          "valid_rows": len(validation), "iterations": spec["max_iterations"],
                          "ap": float(average_precision_score(dev.iloc[validation].target, probability)),
                          "loaded_forward_exact": True,
                          "artifacts": {name: sha(job / name) for name in ["model.joblib", "prediction.npy"]}}
                write_json(marker, record)
                new_dev += 1
                log("fit_complete", seed=seed, quarter=quarter, ap=record["ap"])
            records.append(record)
            per_seed.append(dev.iloc[validation][["row_id", "site_id", "forecast_period", "target"]]
                            .assign(model="site_state", seed=seed, prediction=probability))
    long = pd.concat(per_seed, ignore_index=True)
    roundtrip_csv(long, args.out / "predictions_by_seed.csv")
    means = []
    for seed in SEEDS:
        part = long.loc[long.seed.eq(seed)].set_index("row_id").reindex(meta.row_id)
        means.append(probabilities(part.prediction, len(meta)))
    state_probability = np.mean(np.stack(means), axis=0, dtype=np.float64)
    blend = np.empty(len(meta), dtype=np.float64)
    for quarter in QUARTERS:
        mask = meta.forecast_period.eq(quarter).to_numpy()
        blend[mask] = .75 * rank_scores(base_probability[mask]) + .25 * rank_scores(state_probability[mask])
    candidates = {"context_3seed": base_probability, "site_state": state_probability,
                  "context_site_state_75_25": blend}
    full_oof = pd.concat([meta.assign(model=name, prediction=values) for name, values in candidates.items()], ignore_index=True)
    roundtrip_csv(full_oof, args.out / "predictions.csv")
    report = evaluate_candidates(meta, candidates, bootstrap=args.bootstrap)
    for name in ["summary", "quarter_scores"]:
        report[name].to_csv(args.out / f"{name}.csv", index=False)
    permission = any(report["eligibility"][name] for name in CANDIDATES)
    write_json(args.out / "eligibility.json", {"gate": report["protocol"], "eligible": report["eligibility"],
                                              "three_full_refits_permitted": permission})
    print(report["summary"].to_string(index=False), flush=True)
    exports = {}
    if permission:
        # Final fits may use all official train labels. No Q2/Q3 metric is made.
        rows = pd.concat([train.drop(columns="target"), test], ignore_index=True)
        log("full_features_start", training_rows=len(train), test_rows=len(test))
        final_x, final_cats = features(rows, history)
        if list(final_x) != list(x) or final_cats != cats:
            raise ValueError("Full and development feature schemas differ")
        anchor = pd.read_csv(args.context_test, float_precision="round_trip")
        if (not {"row_id", "target"}.issubset(anchor) or anchor.row_id.isna().any()
                or anchor.row_id.duplicated().any() or set(anchor.row_id) != set(test.row_id)):
            raise ValueError("The fixed context test anchor has invalid IDs")
        anchor_probability = probabilities(anchor.set_index("row_id").loc[test.row_id].target, len(test))
        final_probabilities = []
        for seed in SEEDS:
            job = args.out / f"full_state_s{seed}"
            job.mkdir(exist_ok=True)
            marker = job / "complete.json"
            expected = {"stage": "full_train", "seed": seed,
                        "row_ids_sha256": hashlib.sha256(json.dumps(test.row_id.tolist()).encode()).hexdigest()}
            if args.resume and marker.exists():
                record, probability = completed(job, expected, len(test))
                reused_final += 1
                log("full_job_reused", seed=seed)
            else:
                log("full_refit_start", seed=seed, rows=len(train))
                with threadpool_limits(limits=args.threads):
                    model = fit_model(spec, final_x.iloc[:len(train)], train.target, cats, seed, args.threads)
                    probability = probabilities(predict_model(model, final_x.iloc[len(train):]), len(test))
                    joblib.dump(model, job / "model.joblib")
                    check = predict_model(joblib.load(job / "model.joblib"), final_x.iloc[len(train):])
                if not np.array_equal(probability, check):
                    raise AssertionError("Full state-model reload failed to reproduce test probabilities")
                np.save(job / "prediction.npy", probability, allow_pickle=False)
                record = {"signature": signature, **expected, "train_rows": len(train), "test_rows": len(test),
                          "iterations": spec["max_iterations"], "loaded_forward_exact": True,
                          "artifacts": {name: sha(job / name) for name in ["model.joblib", "prediction.npy"]}}
                write_json(marker, record)
                new_final += 1
            final_probabilities.append(probability)
        state_test = np.mean(np.stack(final_probabilities), axis=0, dtype=np.float64)
        test_candidates = {"site_state": state_test,
                           "context_site_state_75_25": .75 * rank_scores(anchor_probability) + .25 * rank_scores(state_test)}
        raw_test = test[["row_id"]].reset_index(drop=True).assign(
            context_3seed=anchor_probability, site_state=state_test,
            context_site_state_75_25=test_candidates["context_site_state_75_25"])
        for seed, probability in zip(SEEDS, final_probabilities):
            raw_test[f"site_state_s{seed}"] = probability
        roundtrip_csv(raw_test, args.out / "test_predictions.csv")
        for name, values in test_candidates.items():
            eligible = report["eligibility"][name]
            directory = args.out / ("recommended" if eligible else "diagnostic")
            directory.mkdir(exist_ok=True)
            path = directory / f"submission_{name}.csv"
            submission = validate_submission(test, sample, values, expected_rows=None if args.smoke else 2246)
            roundtrip_csv(submission, path)
            exports[name] = {"file": str(path), "sha256": sha(path), "eligible": eligible,
                             "rows": len(submission), "sample_order_exact": True}
    report_files = ["predictions_by_seed.csv", "predictions.csv", "summary.csv", "quarter_scores.csv", "eligibility.json"]
    if permission:
        report_files += ["test_predictions.csv", *(str(Path(v["file"]).relative_to(args.out)) for v in exports.values())]
    protocol.update(status="complete", development_jobs=len(records), new_development_fits=new_dev,
                    reused_development_jobs=reused_dev, new_final_fits=new_final,
                    reused_final_jobs=reused_final, full_refit_gate_passed=permission, exports=exports,
                    loaded_forwards_exact=all(record["loaded_forward_exact"] for record in records),
                    elapsed_seconds=time.monotonic() - started,
                    output_artifacts={name: sha(args.out / name) for name in report_files})
    write_json(protocol_path, protocol)
    log("complete", eligible=report["eligibility"], new_development_fits=new_dev,
        new_final_fits=new_final, exports=exports, kaggle_uploads=0)


if __name__ == "__main__":
    main()
