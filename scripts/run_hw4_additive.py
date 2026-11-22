"""One fixed additive logit on old HW4 dev, with one gated final CSV at most."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_hw4_final_research as primary
from hw4_inspections.additive_logit import (
    CATEGORICAL_COLUMNS, CLASSIFIER_PARAMS, KNOT_QUANTILES, OMITTED_COLUMNS,
    SPLINE_COLUMNS, fit_additive, predict_additive,
)

NAME = "additive_logit"
sha, read_json, write_json = primary.sha, primary.read_json, primary.write_json
read_csv, write_csv = primary.read_csv, primary.write_csv


def provenance(args):
    pool = {NAME: {"family": NAME, "view": "base", "inner": False}}
    plan = primary.provenance(args, pool)
    module = ROOT / "src" / "hw4_inspections"
    plan["source_sha256"].update({
        "additive_logit.py": sha(module / "additive_logit.py"),
        "classical_features.py": sha(module / "classical_features.py"),
        Path(__file__).name: sha(Path(__file__)),
    })
    plan.update(
        family=NAME, classifier=dict(CLASSIFIER_PARAMS), seeds=[],
        tree_selection="None. One fixed C=.1 and optimizer; no hyperparameter, seed or epoch search.",
        selection="One best eligible raw or fixed75%context04+25%additive probability variant by old-dev mean AP; lexical model-name tie. At most one full fit and CSV.",
        recipe={
            "view": "base346 -> exact compact96", "omit": list(OMITTED_COLUMNS),
            "categorical_ohe": list(CATEGORICAL_COLUMNS),
            "spline_fields": list(SPLINE_COLUMNS), "degree": 2,
            "fit_quantiles": list(KNOT_QUANTILES), "tail": "constant", "include_bias": False,
            "small_knot_fallback": "<3 unique knots: standardized raw linear; constant/allmissing zero",
            "linear": "fitmedian+observed-fit-minmax clip; log1p only non-spline suffixes _n/_violation_total/_violation_count",
            "scaling": "fit StandardScaler continuous/spline bases; raw missing/OHE indicators binary unscaled",
            "target_encoding": False, "interactions": False, "monotonicity": False,
            "regularization": "L2 all slopes, unpenalized lbfgs intercept, no class/sample weights",
        },
        extra_source="run_hw4_final_research.py cache/raw/anchor validation and assess; all primary helper sources hashed",
    )
    return plan, pool


def read_job(path, identity, expected_n):
    marker = read_json(path / "complete.json")
    required = {"model.joblib", "prediction.npy", "fit-info.json"}
    if marker["identity"] != identity or marker["reload_exact"] is not True:
        raise ValueError("Completed additive job identity/reload differs")
    if set(marker["sha256"]) != required or any(sha(path / n) != marker["sha256"][n] for n in required):
        raise ValueError("Completed additive job artifact changed")
    values = np.load(path / "prediction.npy", allow_pickle=False)
    if values.dtype != np.float64:
        raise ValueError("Additive predictions must be float64")
    return primary.probabilities(values, expected_n)


def run_job(args, signature, quarter, train, frame):
    full = quarter == "full"
    fit_idx = np.arange(len(train)) if full else np.flatnonzero(train.forecast_period.lt(quarter))
    pred_idx = np.arange(len(train), len(frame)) if full else np.flatnonzero(train.forecast_period.eq(quarter))
    if not len(fit_idx) or not len(pred_idx):
        raise ValueError("Empty additive training or prediction partition")
    identity = dict(signature=signature, family=NAME, period=quarter,
                    train_rows=len(fit_idx), predict_rows=len(pred_idx), classifier=dict(CLASSIFIER_PARAMS))
    path = args.out / "jobs" / quarter
    if path.exists():
        if not args.resume:
            raise ValueError("Existing additive job requires --resume")
        values = read_job(path, identity, len(pred_idx))
        print(json.dumps({"event": "job_reused", "quarter": quarter}), flush=True)
        return values
    print(json.dumps({"event": "fit_start", **identity}), flush=True)
    started = time.monotonic()
    bundle = fit_additive(frame.iloc[fit_idx], train.target.iloc[fit_idx])
    values = primary.probabilities(predict_additive(bundle, frame.iloc[pred_idx]), len(pred_idx))
    path.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=path.parent, prefix=".pending-additive-") as temp:
        job = Path(temp) / "job"
        job.mkdir()
        joblib.dump(bundle, job / "model.joblib", compress=1)
        replay = predict_additive(joblib.load(job / "model.joblib"), frame.iloc[pred_idx])
        if not np.array_equal(values, replay):
            raise AssertionError("Additive job reload did not reproduce forecasts exactly")
        np.save(job / "prediction.npy", values, allow_pickle=False)
        write_json(job / "fit-info.json", dict(
            recipe=bundle["recipe"], iterations=bundle["iterations"], converged=bundle["converged"],
            feature_names=bundle["feature_names"], seconds=time.monotonic() - started,
            new_q2_q3_metrics=False, inner_search=False,
        ))
        digests = {n: sha(job / n) for n in ["model.joblib", "prediction.npy", "fit-info.json"]}
        write_json(job / "complete.json", dict(identity=identity, reload_exact=True, sha256=digests))
        os.rename(job, path)
    print(json.dumps({"event": "fit_complete", "quarter": quarter,
                      "iterations": bundle["iterations"], "design_columns": len(bundle["feature_names"]),
                      "seconds": time.monotonic() - started, "reload_exact": True}), flush=True)
    return values


def run(args):
    args.threads = 4  # Fixed by the additive module; not a new tuning dimension.
    if args.bootstrap < 0 or not args.smoke and not args.bootstrap:
        raise ValueError("Positive bootstrap required; zero is smoke-only")
    plan, pool = provenance(args)
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    plan["signature"] = signature
    freeze = args.out / "freeze.json"
    if freeze.exists():
        if not args.resume or read_json(freeze) != plan:
            raise ValueError("Changed additive source/data/cache/anchors/packages/settings; resume refused")
    else:
        if any(p.name not in {".lock", "run.log"} for p in args.out.iterdir()):
            raise ValueError("New additive run requires an empty output directory")
        write_json(freeze, plan)
    manifest_path = args.out / "forecast-manifest.json"
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if previous["signature"] != signature or any(sha(args.out / n) != h for n, h in previous["output_sha256"].items()):
            raise ValueError("Completed additive output changed")
    train, test, sample, views, _ = primary.validate_cache(joblib.load(args.feature_cache), args)
    frame = views["base"]
    meta = pd.concat([train.loc[train.forecast_period.eq(q), list(primary.REQUIRED)]
                      for q in primary.QUARTERS], ignore_index=True)
    anchor, anchor_test = primary.align_anchors(meta, test, args)
    values = np.concatenate([run_job(args, signature, q, train, frame) for q in primary.QUARTERS])
    selected = primary.assess("screen", args.out, meta, anchor, {NAME: values}, pool, args.bootstrap)
    if len(selected) > 1:
        raise AssertionError("Only one additive family export is allowed")
    files = []
    if selected:
        choice = selected[0]
        forecast = run_job(args, signature, "full", train, frame)
        if choice["variant"] == "blend75":
            forecast = .75 * anchor_test + .25 * forecast
        submission = primary.validate_submission(test, sample, forecast, len(test) if args.smoke else 2246)
        path = args.out / f"submission_additive_{choice['variant']}.csv"
        write_csv(path, submission)
        pd.testing.assert_frame_equal(submission, read_csv(path), check_exact=True)
        files = [{**choice, "csv": path.name, "sha256": sha(path), "rows": len(submission)}]
    outputs = [p for p in args.out.iterdir() if p.suffix in {".csv", ".json"} and p != manifest_path]
    manifest = dict(signature=signature, complete=True, family=NAME, smoke=args.smoke,
                    old_dev_only=True, new_q2_q3_metrics=False, uploads=0,
                    fixed_old_fold_fits=3, qualified_families=len(files),
                    conditional_full_fits=int(bool(files)), primary=files[0]["csv"] if files else None,
                    files=files, output_sha256={p.name: sha(p) for p in outputs})
    write_json(manifest_path, manifest)
    print(json.dumps({"event": "complete", "files": files, "new_q2_q3_metrics": False}, ensure_ascii=False), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ["raw", "out", "feature-cache", "baseline-cv", "context-test"]:
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
