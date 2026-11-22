"""Bounded causal HW4 CPU screen, confirmation and gated final forecasts."""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hw4_inspections.deep_boosting import FORBIDDEN_COLUMNS, SPECS, choose_tree_count, fit_booster, predict_booster
from hw4_inspections.delivery import validate_submission
from hw4_inspections.next_round import QUARTERS, REQUIRED, evaluate_candidates
from hw4_inspections.seasonal_context import COLUMNS as SEASONAL_COLUMNS
from hw4_inspections.transition_features import FEATURE_COLUMNS as TRANSITION_COLUMNS

RAW_FILES = ("train.csv", "test_student.csv", "inspection_history.csv", "sample_submission.csv")
BUILDERS = ("features.py", "extensions.py", "transition_features.py", "seasonal_context.py")
MODULES = (*BUILDERS, "deep_boosting.py", "delivery.py", "next_round.py", "classical_analysis.py")
SEEDS = (42, 17, 2026)
ANCHOR = "context_3seed"
OOF_SHA = "0096456a838264be615680be4eead0ed8b449a4182d9f20acd519937e7cc1f7c"
TEST_SHA = "636f8cbe87367b6df0bae6d9a1d79a57de863bc78a21bc2296f5c2ff85970634"
WORST_FLOOR = -.005


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def write_csv(path, data):
    temp = path.with_suffix(".csv.tmp")
    data.to_csv(temp, index=False)
    temp.replace(path)


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def probabilities(values, size):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (size,) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid probabilities or row count")
    return values


def cases(smoke):
    pool = {}
    for name, spec in SPECS.items():
        pool[name] = {"family": spec["family"], "spec": deepcopy(spec), "view": "base", "inner": True}
    for view in ["transition", "seasonal"]:
        pool[view] = {"family": view, "view": view, "inner": False, "spec": {
            "family": "catboost", "max_iterations": 500,
            "params": {"learning_rate": .04, "depth": 5, "l2_leaf_reg": 10.,
                       "boosting_type": "Plain", "bootstrap_type": "MVS", "subsample": .8}}}
    if smoke:
        for case in pool.values():
            case["spec"]["max_iterations"] = 3
    return pool


def package_evidence():
    versions = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "catboost", "xgboost", "joblib"):
        dist = metadata.distribution(name)
        record = dist.read_text("RECORD")
        versions[name] = {"version": dist.version, "distribution_record_sha256":
                          hashlib.sha256(record.encode()).hexdigest() if record is not None else None}
    return versions


def normalize_digests(digests):
    normalized = {Path(key).name: value for key, value in digests.items()}
    if len(digests) != len(normalized):
        raise ValueError("Ambiguous basename in cache metadata")
    return normalized


def provenance(args, pool):
    module_dir = ROOT / "src" / "hw4_inspections"
    source = {n: sha(module_dir / n) for n in MODULES}
    source[Path(__file__).name] = sha(Path(__file__))
    raw = {n: sha(args.raw / n) for n in RAW_FILES}
    meta_path = args.feature_cache.with_name("features-meta.json")
    cache_meta = read_json(meta_path)
    cache_hash = sha(args.feature_cache)
    if cache_meta["cache_sha256"] != cache_hash:
        raise ValueError("Feature cache hash mismatch")
    if normalize_digests(cache_meta["raw_sha256"]) != raw:
        raise ValueError("Raw inputs differ from cache inputs")
    if normalize_digests(cache_meta["builder_sha256"]) != {n: source[n] for n in BUILDERS}:
        raise ValueError("Feature builder source differs from trusted cache")
    anchors = {"old_oof": sha(args.baseline_cv), "context04_test": sha(args.context_test)}
    if not args.smoke and anchors != {"old_oof": OOF_SHA, "context04_test": TEST_SHA}:
        raise ValueError("Production anchors must match the verified context04 predictions")
    return {"format": 1, "smoke": args.smoke, "raw_sha256": raw, "source_sha256": source,
            "cache_sha256": cache_hash, "cache_metadata_sha256": sha(meta_path), "anchors_sha256": anchors,
            "packages": package_evidence(), "python": platform.python_version(), "system": platform.platform(),
            "threads": args.threads, "bootstrap": args.bootstrap, "bootstrap_seed": 20261122,
            "pool": pool, "seeds": list(SEEDS), "old_folds": list(QUARTERS),
            "tree_selection": "42 only: strict previous inner quarter, minimum logloss among multiples of 50 and cap, earliest tie. Feature models fixed cap. Seeds 17/2026 reuse count; final median.",
            "scores": "Within-quarter sklearn AP, equal-quarter mean. One raw and one probability mixture 0.75*context04+0.25*candidate, raw within-family seed means.",
            "admission": {"min_mean_delta_exclusive": 0, "minimum_nonnegative_quarters": 2,
                          "minimum_worst_quarter_delta": WORST_FLOOR},
            "selection": "One best eligible case per family by mean AP across both exports, lexical model name tie. Re-evaluate the same two exports per confirmed case after 3 seeds. One CSV per eligible family.",
            "interpretation": "Reused old development data, nominal intervals with no correction for selection.",
            "new_q2_q3_metrics": False, "uploads": 0}


def validate_cache(cache, args):
    train, test, sample, views, cats = [cache[k] for k in ["train", "test", "sample", "views", "cats"]]
    for frame, filename in [(train, "train.csv"), (test, "test_student.csv"), (sample, "sample_submission.csv")]:
        if not frame.index.equals(pd.RangeIndex(len(frame))) or not frame.columns.is_unique:
            raise ValueError(f"Noncanonical frame index or columns in {filename}")
        pd.testing.assert_frame_equal(frame, read_csv(args.raw / filename), check_dtype=False)
        if not frame.row_id.is_unique or frame.row_id.isna().any():
            raise ValueError(f"Invalid official row IDs in {filename}")
    if set(train.row_id) & set(test.row_id) or not train.target.isin([0, 1]).all() or "target" in test:
        raise ValueError("Invalid test/train separation or labels")
    if args.smoke and train.groupby("forecast_period").size().max() > 32:
        raise ValueError("Smoke requires a synthetic cache with at most 32 rows per quarter")
    if not args.smoke and (len(train) != 21017 or len(test) != 2246 or len(cats) != 10):
        raise ValueError("Real HW4 rows or category count changed")
    expected_test = len(test) if args.smoke else 2246
    validate_submission(test, sample, np.zeros(len(test)), expected_test)
    if set(views) != {"base", "transition", "seasonal"}:
        raise ValueError("Unexpected feature cache views")
    base = views["base"]
    if not args.smoke and len(base.columns) != 346:
        raise ValueError("Expected 346 context columns")
    if len(set(cats)) != len(cats) or not set(cats).issubset(base):
        raise ValueError("Invalid cached categorical columns")
    for view, additions in [("base", ()), ("transition", TRANSITION_COLUMNS), ("seasonal", SEASONAL_COLUMNS)]:
        frame = views[view]
        if (not frame.index.equals(pd.RangeIndex(len(train) + len(test))) or not frame.columns.is_unique
                or FORBIDDEN_COLUMNS.intersection(frame.columns) or list(frame) != [*base.columns, *additions]):
            raise ValueError(f"Invalid cached feature schema/index for {view}")
        if view != "base":
            for column in base:
                if not base[column].equals(frame[column]):
                    raise ValueError(f"Base input differs in {view}/{column}")
    return train, test, sample, views, list(cats)


def align_anchors(meta, test, args):
    baseline = read_csv(args.baseline_cv)
    if ("model" in baseline and not baseline.model.eq(ANCHOR).all()) or set(baseline.row_id) != set(meta.row_id):
        raise ValueError("Wrong old OOF anchor IDs/model")
    if not baseline.row_id.is_unique:
        raise ValueError("Duplicate anchor OOF IDs")
    aligned = baseline.set_index("row_id").loc[meta.row_id].reset_index()
    pd.testing.assert_frame_equal(meta[list(REQUIRED)], aligned[list(REQUIRED)], check_dtype=False)
    anchor = read_csv(args.context_test)
    if list(anchor) != ["row_id", "target"]:
        raise ValueError("Invalid context04 CSV columns")
    # Supplied values initially follow anchor row order. This returns test order.
    anchored = validate_submission(anchor, test, anchor.target, len(test))
    return probabilities(aligned.prediction, len(meta)), probabilities(anchored.target, len(test))


def assess(stage, out, meta, anchor, values, pool, bootstrap):
    predictions = {ANCHOR: anchor}
    identity = {}
    for case, array in values.items():
        for variant, scores in [("raw", array), ("blend75", .75 * anchor + .25 * array)]:
            name = f"{case}__{variant}"
            predictions[name] = scores
            identity[name] = (case, pool[case]["family"], variant)
    result = evaluate_candidates(meta, predictions, reference=ANCHOR, bootstrap=bootstrap, seed=20261122)
    summary = result["summary"].copy()
    summary["eligible"] &= summary.worst_quarter_delta.ge(WORST_FLOOR)
    for i, field in enumerate(["case", "family", "variant"]):
        summary[field] = [identity.get(name, (None, None, None))[i] for name in summary.model]
    summary = summary.sort_values(["mean_ap", "model"], ascending=[False, True], kind="stable")
    best = summary.loc[summary.eligible].drop_duplicates("family")
    choices = best[["case", "family", "variant", "model", "mean_ap", "delta_vs_reference"]].to_dict("records")
    write_csv(out / f"{stage}-quarter-metrics.csv", result["quarter_scores"])
    write_csv(out / f"{stage}-summary.csv", summary)
    write_csv(out / f"{stage}-oof.csv", pd.concat([meta, pd.DataFrame(predictions)], axis=1))
    write_json(out / f"{stage}-selection.json", choices)
    return choices


def read_job(path, identity, n):
    rec = read_json(path / "complete.json")
    if rec["identity"] != identity or not rec["reload_exact"]:
        raise ValueError(f"Job identity or exact reload mismatch at {path}")
    expected = {"model.joblib", "prediction.npy", "curve.json"}
    if set(rec["sha256"]) != expected or any(sha(path / name) != rec["sha256"][name] for name in expected):
        raise ValueError(f"Changed job artifact at {path}")
    value = np.load(path / "prediction.npy", allow_pickle=False)
    if value.dtype != np.float64:
        raise ValueError(f"Non-float64 cached predictions at {path}")
    return probabilities(value, n), rec["iterations"]


def run_job(out, signature, name, case, seed, quarter, iterations, train, frame, cats, threads, resume):
    is_full = quarter == "full"
    periods = train.forecast_period
    train_idx = np.arange(len(train)) if is_full else np.flatnonzero(periods.lt(quarter))
    valid_idx = np.arange(len(train), len(frame)) if is_full else np.flatnonzero(periods.eq(quarter))
    if not len(train_idx) or not len(valid_idx):
        raise ValueError("Empty train or prediction partition")
    path = out / "jobs" / f"{name}__{quarter}__s{seed}"
    identity = {"signature": signature, "case": name, "seed": seed, "period": quarter,
                "fixed_count": iterations, "train_rows": len(train_idx), "predict_rows": len(valid_idx)}
    if path.exists():
        if not resume:
            raise ValueError(f"Existing job without --resume: {path}")
        return read_job(path, identity, len(valid_idx))
    print(json.dumps({"event": "fit_start", **identity}), flush=True)
    curve = []
    inner_q = None
    spec = case["spec"]
    if iterations is None:
        if is_full or seed != SEEDS[0]:
            raise ValueError("Only screen jobs may select tree counts")
        if case["inner"]:
            inner_q = (pd.Period(quarter, freq="Q") - 1).start_time.strftime("%Y-%m-%d")
            inner_fit = np.flatnonzero(periods.lt(inner_q))
            inner_valid = np.flatnonzero(periods.eq(inner_q))
            iterations, curve = choose_tree_count(spec, frame.iloc[inner_fit], train.target.iloc[inner_fit],
                frame.iloc[inner_valid], train.target.iloc[inner_valid], cats, seed=seed, threads=threads)
        else:
            iterations = spec["max_iterations"]
    bundle = fit_booster(spec, frame.iloc[train_idx], train.target.iloc[train_idx], cats,
                         seed=seed, threads=threads, iterations=iterations)
    pred = probabilities(predict_booster(bundle, frame.iloc[valid_idx]), len(valid_idx))
    path.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=path.parent, prefix=".pending-") as temp:
        job = Path(temp) / "job"
        job.mkdir()
        joblib.dump(bundle, job / "model.joblib", compress=1)
        replay = predict_booster(joblib.load(job / "model.joblib"), frame.iloc[valid_idx])
        if not np.array_equal(pred, replay):
            raise ValueError("Model forward differs after save/reload")
        np.save(job / "prediction.npy", pred, allow_pickle=False)
        write_json(job / "curve.json", {"inner_quarter": inner_q, "points": curve, "iterations": iterations})
        digests = {p.name: sha(p) for p in job.iterdir()}
        write_json(job / "complete.json", {"identity": identity, "iterations": iterations, "reload_exact": True,
                                          "sha256": digests, "inner_quarter": inner_q})
        os.rename(job, path)
    print(json.dumps({"event": "job_complete", "case": name, "quarter": quarter,
                      "seed": seed, "trees": iterations}), flush=True)
    return pred, iterations


def run(args):
    if args.threads < 1 or args.bootstrap < 0 or (not args.smoke and not args.bootstrap):
        raise ValueError("Positive threads and bootstrap required; zero bootstrap is smoke-only")
    pool = cases(args.smoke)
    plan = provenance(args, pool)
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    plan["signature"] = signature
    frozen = args.out / "freeze.json"
    if frozen.exists():
        if not args.resume or read_json(frozen) != plan:
            raise ValueError("Changed source, cache, data, library, anchor or settings; resume refused")
    else:
        if any(p.name != ".lock" for p in args.out.iterdir()):
            raise ValueError("Use an empty output directory for a new run")
        write_json(frozen, plan)
    manifest_path = args.out / "forecast-manifest.json"
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old["signature"] != signature or any(sha(args.out / file) != digest for file,digest in old["output_sha256"].items()):
            raise ValueError("Changed completed run outputs")
    train, test, sample, views, cats = validate_cache(joblib.load(args.feature_cache), args)
    meta = pd.concat([train.loc[train.forecast_period.eq(q), list(REQUIRED)] for q in QUARTERS], ignore_index=True)
    anchor, anchor_test = align_anchors(meta, test, args)
    predictions, counts = {}, {}
    for name, case in pool.items():
        counts[name] = {}
        fold_p = []
        for q in QUARTERS:
            arr, counts[name][q] = run_job(args.out, signature, name, case, 42, q, None, train,
                                           views[case["view"]], cats, args.threads, args.resume)
            fold_p.append(arr)
        predictions[name] = np.concatenate(fold_p)
    selected = assess("screen", args.out, meta, anchor, predictions, pool, args.bootstrap)
    confirmed = {}
    for choice in selected:
        name = choice["case"]
        case = pool[name]
        seed_p = [predictions[name]]
        for seed in SEEDS[1:]:
            fold_p = [run_job(args.out, signature, name, case, seed, q, counts[name][q], train,
                              views[case["view"]], cats, args.threads, args.resume)[0] for q in QUARTERS]
            seed_p.append(np.concatenate(fold_p))
        confirmed[name] = np.mean(seed_p, axis=0, dtype=np.float64)
    qualified = assess("confirmed", args.out, meta, anchor, confirmed, pool, args.bootstrap)
    files = []
    for number, choice in enumerate(qualified, start=1):
        name = choice["case"]
        case = pool[name]
        count = int(np.median(list(counts[name].values())))
        fits = [run_job(args.out, signature, name, case, seed, "full", count, train,
                        views[case["view"]], cats, args.threads, args.resume)[0] for seed in SEEDS]
        value = np.mean(fits, axis=0, dtype=np.float64)
        if choice["variant"] == "blend75":
            value = .75 * anchor_test + .25 * value
        sub = validate_submission(test, sample, value, len(test) if args.smoke else 2246)
        path = args.out / f"submission_{number:02d}_{name}_{choice['variant']}.csv"
        write_csv(path, sub)
        pd.testing.assert_frame_equal(sub, read_csv(path), check_exact=True)
        files.append({**choice, "csv": path.name, "sha256": sha(path), "rows": len(sub), "full_trees": count})
    outputs = [p for p in args.out.iterdir() if p.suffix in {".csv", ".json"} and p != manifest_path]
    manifest = {"signature": signature, "complete": True, "old_dev_only": True, "new_q2_q3_metrics": False,
                "uploads": 0, "smoke": args.smoke, "screen_cases": len(pool), "confirmed_cases": len(confirmed),
                "qualified_families": len(files), "primary": files[0]["csv"] if files else None, "files": files,
                "output_sha256": {p.name: sha(p) for p in outputs}}
    write_json(manifest_path, manifest)
    print(json.dumps({"event":"complete", "files": files}, ensure_ascii=False), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("raw", "out", "feature-cache", "baseline-cv", "context-test"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
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
