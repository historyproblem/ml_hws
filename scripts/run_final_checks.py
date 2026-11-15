#!/usr/bin/env python3
"""Locked final refit first, then a bounded, explicit set of optional checks."""
import argparse
from copy import deepcopy
import fcntl
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.final_check_features import BLOCK_COLUMNS, build_extra_features
from scripts.final_check_io import FrozenReference, export_locked_submissions
from scripts.final_check_models import FinalJoint
from spend_baselines.data import clean_transactions
from spend_baselines.suite import identity, job_key, metric, read_config
from spend_baselines.suite_data import (
    BASE_COLUMNS, FEATURES, CachedData, atomic_json, digest, prepare_cache, row_weights, split_indices,
)
from spend_baselines.suite_models import Tabular


def read_final_config(path):
    c = json.loads(Path(path).read_text())
    if c["version"] != 1 or (c["dataset"], c["base_features"]) != ("D2", "F2"):
        raise ValueError("Final checks must preserve D2/F2 as their reference")
    if (c["locked_primary"], c["locked_reserve"]) != ("mlp:hurdle_joint:compact_l2", "wide:direct:compact_l2"):
        raise ValueError("Do not replace the locked reference candidates")
    if len(c["seeds"]) != len(set(c["seeds"])) or not c["seeds"]:
        raise ValueError("Invalid repeated-run seed list")
    names = [v["name"] for v in c["cases"]]
    if len(names) != len(set(names)) or any(not n.replace("_", "").isalnum() for n in names):
        raise ValueError("Invalid case names")
    if set(c["run_order"]) != set(c["validation_dates"]) or len(c["run_order"]) != len(c["validation_dates"]):
        raise ValueError("Run order must cover each date once")
    for case in c["cases"]:
        if case["model"] not in {"joint", "ridge"} or case["feature_block"] not in BLOCK_COLUMNS:
            raise ValueError("Unknown final-check case")
        if set(case.get("settings", {})) - {"weight_decay", "dropout"}:
            raise ValueError("Only explicit L2/dropout changes belong to this final plan")
    return c


def plan_jobs(config, smoke=False):
    jobs = []
    for date in config["run_order"]:
        for case in config["cases"]:
            seeds = [42] if case["model"] == "ridge" or smoke else config["seeds"]
            jobs += [{"case": case["name"], "date": date, "seed": seed, "test": False} for seed in seeds]
    return jobs


def extension_hashes():
    return {name: digest(ROOT / "scripts" / name) for name in [
        "run_final_checks.py", "final_check_features.py", "final_check_models.py", "final_check_io.py"]}


def load_cache(args, base, definition):
    directory = args.cache_dir or (args.reference_out / "cache")
    marker = directory / "ready.json"
    if not marker.exists():
        return prepare_cache(args.data_dir, args.out_dir / "core_cache", base, definition)
    ready = json.loads(marker.read_text())
    for key in ["source_sha256", "input_sha256"]:
        if ready["identity"][key] != definition[key]:
            raise ValueError("Reference cache code/inputs differ; choose a current --cache-dir or new reference run")
    if ready["columns"] != BASE_COLUMNS:
        raise ValueError("Reference feature order changed")
    for name, sha in ready["sha256"].items():
        if digest(directory / name) != sha:
            raise ValueError(f"Corrupt reference cache: {name}")
    return CachedData(directory)


def prepare_extra(args, config, cache, signature):
    directory = args.out_dir / "extra_features"
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "ready.json"
    if marker.exists():
        old = json.loads(marker.read_text())
        if old["signature"] != signature:
            raise ValueError("Extra feature definition changed; use a new output")
        for name, sha in old["sha256"].items():
            if digest(directory / name) != sha:
                raise ValueError("Corrupt extra feature cache")
    else:
        raw = pd.read_csv(args.data_dir / "transactions.csv", dtype={
            c: "string" for c in ["customer_id", "order_id", "item_id", "region"]})
        days, _ = clean_transactions(raw)
        start = pd.to_datetime(raw.event_ts).min().normalize()
        official = cache.objects.loc[cache.objects.official]
        print("Preparing causal annual/November and completed-target features", flush=True)
        arrays = build_extra_features(cache.objects, days, official, history_start=start,
                                      prior_strength=config["target_history_prior_strength"])
        for name, a in arrays.items():
            np.save(directory / f"{name}.npy", a)
        atomic_json(marker, {"signature": signature,
                            "sha256": {f"{name}.npy": digest(directory / f"{name}.npy") for name in arrays}})
    arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r") for name in ["season", "target_history"]}
    schema = {"reference": FEATURES["F2"],
              "cases": {v["name"]: FEATURES["F2"] + BLOCK_COLUMNS[v["feature_block"]] for v in config["cases"]},
              "rule": "All transaction windows end by each row's as_of_date; target prior uses target_end<=as_of_date"}
    if any("fbid" in name.lower() for cols in schema["cases"].values() for name in cols):
        raise ValueError("Forbidden factor")
    atomic_json(args.out_dir / "feature_schema.json", schema)
    return arrays


def matrix(cache, arrays, block, indices):
    x = cache.matrix("F2", indices)[0]
    return x if block == "none" else np.c_[x, np.asarray(arrays[block][indices])]


def model_settings(base, config, case, smoke):
    profile = next(p for p in base["profiles"] if p["name"] == "compact_l2")
    settings = {**base["neural"], **{k: v for k, v in profile.items() if k != "name"}}
    settings.update(case.get("settings", {}))
    settings["loss_weights"] = case.get("loss_weights", config["loss_weights"])
    if smoke:
        settings.update(max_epochs=2, patience=1, batch_size=64, hidden_sizes=[8, 4])
    return settings


def smoke_indices(objects, indices):
    import hashlib
    customers = sorted(objects.customer_id.unique(), key=lambda c: hashlib.sha256(str(c).encode()).hexdigest())[:64]
    return indices[objects.iloc[indices].customer_id.isin(customers).to_numpy()]


def read_completed(directory, job):
    marker = directory / "result.json"
    if not marker.exists():
        return None
    r = json.loads(marker.read_text())
    if r["job"] != job:
        raise ValueError("Completed final-check job differs")
    for name, sha in r["sha256"].items():
        if Path(name).name != name or digest(directory / name) != sha:
            raise ValueError("Corrupt completed final-check artifact")
    return r


def execute(args, config, base, cache, arrays, job):
    case = next(c for c in config["cases"] if c["name"] == job["case"])
    directory = args.out_dir / "checks" / "jobs" / job_key(job)
    old = read_completed(directory, job) if args.resume else None
    if old is not None:
        print(f"skip completed {job}", flush=True)
        return old
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    objects = cache.objects
    outer = split_indices(objects, "D2", job["date"])
    valid = np.flatnonzero((objects.as_of_date.eq(pd.Timestamp(job["date"])) &
                           (objects.is_test if job["test"] else objects.official)).to_numpy())
    if args.smoke:
        outer, valid = smoke_indices(objects, outer), smoke_indices(objects, valid)
    if not len(outer) or not len(valid):
        raise ValueError("Empty final-check temporal split")
    x = matrix(cache, arrays, case["feature_block"], outer)
    vx = matrix(cache, arrays, case["feature_block"], valid)
    y = objects.iloc[outer].spend_30d.to_numpy(float)
    weights = row_weights(objects, outer, "D2")
    metadata = {"n_train": len(outer), "input_columns": x.shape[1], "feature_block": case["feature_block"]}
    files = []
    print(f"start {job}, train={len(outer)}, features={x.shape[1]}", flush=True)
    if case["model"] == "ridge":
        model = Tabular("linear", "direct", base["tabular"], job["seed"], args.threads).fit(x, y, weights)
        import joblib
        joblib.dump(model, directory / "model.joblib", compress=3)
        files.append("model.joblib")
    else:
        settings = model_settings(base, config, case, args.smoke)
        eligible = objects.official & objects.target_end.le(pd.Timestamp(job["date"]))
        inner_date = objects.loc[eligible, "as_of_date"].max()
        inner = split_indices(objects, "D2", inner_date)
        iv = np.flatnonzero((objects.official & objects.as_of_date.eq(inner_date)).to_numpy())
        if args.smoke:
            inner, iv = smoke_indices(objects, inner), smoke_indices(objects, iv)
        selector = FinalJoint("mlp", "hurdle_joint", settings, job["seed"], args.device, args.threads)
        selector.fit(matrix(cache, arrays, case["feature_block"], inner),
                     objects.iloc[inner].spend_30d.to_numpy(float), row_weights(objects, inner, "D2"),
                     validation=(matrix(cache, arrays, case["feature_block"], iv),
                                 objects.iloc[iv].spend_30d.to_numpy(float)))
        chosen = selector.best_epoch
        pd.DataFrame(selector.history).to_csv(directory / "selection.csv", index=False)
        del selector
        metadata.update(inner_date=str(inner_date.date()), selected_epoch=chosen, settings=settings)
        model = FinalJoint("mlp", "hurdle_joint", settings, job["seed"], args.device, args.threads)
        model.fit(x, y, weights, epochs=chosen)
        pd.DataFrame(model.history).to_csv(directory / "refit.csv", index=False)
        model.save(directory / "model.pt")
        files += ["model.pt", "selection.csv", "refit.csv"]
    z = model.predict(vx)
    probability = model.predict_probability(vx) if case["model"] == "joint" else None
    np.savez_compressed(directory / "predictions.npz", Id=objects.iloc[valid].Id.to_numpy(str), z=z)
    files.append("predictions.npz")
    scores = {"n": len(valid)} if job["test"] else metric(objects.iloc[valid].spend_30d.to_numpy(float), z, probability)
    result = {"job": job, **metadata, **scores, "seconds": time.monotonic() - started,
              "sha256": {name: digest(directory / name) for name in files}}
    atomic_json(directory / "result.json", result)
    print(f"done {job}: {scores.get('RMSLE', 'test predictions')}", flush=True)
    return result


def aligned_prediction(args, result, ids):
    path = args.out_dir / "checks" / "jobs" / job_key(result["job"]) / "predictions.npz"
    with np.load(path, allow_pickle=False) as p:
        saved_ids, z = p["Id"].astype(str), p["z"].astype(float)
    order = pd.Index(saved_ids).get_indexer(ids)
    if (order < 0).any() or len(saved_ids) != len(ids):
        raise ValueError("Final-check prediction population mismatch")
    return z[order]


def paired_delta(frame, repeats=1000):
    codes, customers = pd.factorize(frame.customer_id)
    counts = np.bincount(codes)
    target = np.log1p(frame.spend_30d.to_numpy(float))
    a = np.bincount(codes, weights=(frame.z.to_numpy()-target)**2)
    b = np.bincount(codes, weights=(frame.reference_z.to_numpy()-target)**2)
    actual = float(np.sqrt(a.sum()/counts.sum())-np.sqrt(b.sum()/counts.sum()))
    rng = np.random.default_rng(20261115)
    values = []
    for _ in range(repeats):
        ix = rng.integers(0, len(customers), len(customers))
        n = counts[ix].sum()
        values.append(np.sqrt(a[ix].sum()/n)-np.sqrt(b[ix].sum()/n))
    lo, hi = np.quantile(values, [.025, .975])
    return actual, float(lo), float(hi)


def summarize(args, config, cache, reference, jobs):
    directory = args.out_dir / "reports"
    directory.mkdir(parents=True, exist_ok=True)
    reference_predictions = {}
    locked_rows = []
    for date in config["validation_dates"]:
        for key, name in [("primary", config["locked_primary"]), ("wide", config["locked_reserve"]),
                          ("ridge", "linear:direct:fixed")]:
            reference_predictions[(date, key)] = reference.mean_prediction(name, date, cache.objects)
        ids, zmain = reference_predictions[(date, "primary")]
        wide_ids, zwide = reference_predictions[(date, "wide")]
        if not np.array_equal(ids, wide_ids):
            raise ValueError("Locked OOT candidate order mismatch")
        y = cache.objects.set_index("Id").loc[ids, "spend_30d"].to_numpy(float)
        for weight in config["blend_joint_weights"]:
            locked_rows.append({"date": date, "joint_weight": weight,
                                **metric(y, weight*zmain+(1-weight)*zwide)})
    pd.DataFrame(locked_rows).to_csv(directory / "locked_blend_validation.csv", index=False)
    records, ensembles, points = [], [], {}
    for case in config["cases"]:
        expected = [42] if case["model"] == "ridge" or args.smoke else config["seeds"]
        for date in config["validation_dates"]:
            results = []
            for seed in expected:
                job = {"case": case["name"], "date": date, "seed": seed, "test": False}
                r = read_completed(args.out_dir / "checks" / "jobs" / job_key(job), job)
                if r is not None:
                    results.append(r)
                    records.append({**job, **{k: v for k, v in r.items() if k not in {"job", "sha256", "settings"}}})
            if not results:
                continue
            valid = np.flatnonzero((cache.objects.official & cache.objects.as_of_date.eq(pd.Timestamp(date))).to_numpy())
            if args.smoke:
                valid = smoke_indices(cache.objects, valid)
            frame = cache.objects.iloc[valid][["Id", "customer_id", "spend_30d"]].copy()
            ids = frame.Id.to_numpy(str)
            frame["z"] = np.mean([aligned_prediction(args, r, ids) for r in results], axis=0)
            ref_ids, ref_z = reference_predictions[(date, "primary")]
            order = pd.Index(ref_ids).get_indexer(ids)
            if (order < 0).any():
                raise ValueError("Reference lacks evaluation objects")
            frame["reference_z"] = ref_z[order]
            own_ids, own_z = reference_predictions[(date, "ridge" if case["model"] == "ridge" else "primary")]
            own = own_z[pd.Index(own_ids).get_indexer(ids)]
            score = metric(frame.spend_30d.to_numpy(float), frame.z.to_numpy())
            row = {"case": case["name"], "date": date, "seeds_done": len(results), "seeds_expected": len(expected),
                   "complete": len(results) == len(expected), "smoke": args.smoke, **score,
                   "delta_vs_primary": score["RMSLE"]-metric(frame.spend_30d.to_numpy(float), frame.reference_z.to_numpy())["RMSLE"],
                   "delta_vs_same_family_control": score["RMSLE"]-metric(frame.spend_30d.to_numpy(float), own)["RMSLE"]}
            ensembles.append(row)
            if row["complete"]:
                points[(case["name"], date)] = frame
    pd.DataFrame(records).to_csv(directory / "scores.csv", index=False)
    pd.DataFrame(ensembles).to_csv(directory / "candidate_scores.csv", index=False)
    comparisons = []
    late = ["2018-09-01", "2018-10-01"]
    for case in config["cases"]:
        frames = [points[(case["name"], d)] for d in late if (case["name"], d) in points]
        if len(frames) != 2:
            continue
        f = pd.concat(frames, ignore_index=True)
        delta, lo, hi = paired_delta(f)
        rows = [r for r in ensembles if r["case"] == case["name"] and r["date"] in late]
        complete_dates = [d for d in config["validation_dates"] if (case["name"], d) in points]
        full = pd.concat([points[(case["name"], d)] for d in complete_dates], ignore_index=True)
        both = all(r["delta_vs_primary"] < 0 for r in rows)
        comparisons.append({"case": case["name"], "late2_RMSLE": metric(f.spend_30d.to_numpy(float), f.z.to_numpy())["RMSLE"],
                            "delta_late2": delta, "bootstrap95_lo": lo, "bootstrap95_hi": hi,
                            "both_late_dates_better": both, "covered_dates": len(complete_dates),
                            "pooled5_RMSLE": metric(full.spend_30d.to_numpy(float), full.z.to_numpy())["RMSLE"] if len(complete_dates)==5 else None,
                            "preliminary": abs(delta) <= .003 or hi >= 0 or not both,
                            "automatic_replacement": False, "smoke": args.smoke})
    pd.DataFrame(comparisons).to_csv(directory / "late2_comparison.csv", index=False)
    complete_jobs = len(records)
    atomic_json(args.out_dir / "status.json", {"completed_validation_jobs": complete_jobs,
                "planned_validation_jobs": len(jobs), "pending_validation_jobs": len(jobs)-complete_jobs,
                "full_plan_complete": complete_jobs == len(jobs), "smoke": args.smoke,
                "primary_locked": config["locked_primary"], "automatic_model_selection": False,
                "kaggle_uploaded": False})
    print(f"Validation coverage: {complete_jobs}/{len(jobs)} jobs; primary stays locked", flush=True)
    if comparisons:
        print(pd.DataFrame(comparisons).to_string(index=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "finish", "run", "analyze", "refit"])
    parser.add_argument("--config", type=Path, default=Path("configs/final_checks.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--reference-out", type=Path)
    parser.add_argument("--reference-archive", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/final-checks-20261115"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--dates", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--max-minutes", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--case", help="Explicit candidate for final refit")
    args = parser.parse_args(argv)
    config = read_final_config(args.config)
    base = read_config(config["reference_config"])
    args.reference_out = args.reference_out or Path(config["reference_out"])
    jobs = plan_jobs(config, args.smoke)
    names = {c["name"] for c in config["cases"]}
    if args.threads < 1 or (args.cases and set(args.cases) - names) or (args.dates and set(args.dates) - set(config["validation_dates"])):
        parser.error("Invalid threads/cases/dates")
    if args.max_minutes is not None and args.max_minutes < 0:
        parser.error("Time budget cannot be negative")
    if args.seeds and set(args.seeds) - set(config["seeds"]):
        parser.error("Select seeds from the explicit configuration")
    if args.smoke and args.command in {"finish", "refit"}:
        parser.error("Smoke must not generate final submissions")
    print(json.dumps({"optional_jobs": len(jobs), "neural_fit_calls": 2*sum(j["case"] not in {c['name'] for c in config['cases'] if c['model']=='ridge'} for j in jobs),
                      "primary": config["locked_primary"], "no_automatic_model_selection": True}), flush=True)
    if args.command == "plan":
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.out_dir / "final_checks.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.device == "cuda" and args.command != "analyze":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
    definition = identity(base, args.data_dir)
    if args.command == "finish":
        if args.reference_archive:
            parser.error("finish needs the original live run, not a prediction-only archive")
        candidates = [config["locked_primary"]] + ([config["locked_reserve"]] if config["refit_reserve"] else [])
        for i, name in enumerate(candidates):
            subprocess.run([sys.executable, "-u", "-m", "spend_baselines.suite", "refit", "--config", config["reference_config"],
                            "--data-dir", str(args.data_dir), "--out-dir", str(args.reference_out), "--device", args.device,
                            "--threads", str(args.threads), "--resume", "--datasets", "D2", "--representations", "F2", "--candidate", name], check=True)
            export_config = {**config, "refit_reserve": i > 0}
            primary = export_locked_submissions(args.reference_out, export_config, args.data_dir, args.out_dir / "submissions")
            print(f"PRIMARY SUBMISSION READY: {primary}", flush=True)
    cache = load_cache(args, base, definition)
    reference = FrozenReference(args.reference_out, definition, args.reference_archive)
    signature = {"reference_definition": definition, "config": config, "extensions_sha256": extension_hashes(),
                 "device": args.device, "threads": args.threads, "python": platform.python_version(), "smoke": args.smoke,
                 "libraries": {n: version(n) for n in ["numpy", "pandas", "scikit-learn", "scipy", "torch"]}}
    marker = args.out_dir / "checks" / "run_manifest.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        if not args.resume or json.loads(marker.read_text()) != signature:
            raise ValueError("Final-check definition changed or --resume missing; choose a new --out-dir")
    else:
        atomic_json(marker, signature)
    arrays = prepare_extra(args, config, cache, signature)
    if args.command == "refit":
        if args.case not in names:
            parser.error("refit needs one explicit --case")
        results = []
        case = next(c for c in config["cases"] if c["name"] == args.case)
        date = str(cache.objects.loc[cache.objects.is_test, "as_of_date"].min().date())
        for seed in ([42] if case["model"] == "ridge" else config["seeds"]):
            results.append(execute(args, config, base, cache, arrays, {"case": args.case, "date": date, "seed": seed, "test": True}))
        testing = cache.objects.loc[cache.objects.is_test]
        z = np.mean([aligned_prediction(args, r, testing.Id.to_numpy(str)) for r in results], axis=0)
        from spend_baselines.cli import write_submission
        sample = pd.read_csv(args.data_dir / "sample_submission.csv", dtype={"Id": "string"})
        dest = args.out_dir / "submissions" / f"{args.case}.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_submission(testing, z, dest, sample.Id)
        print(f"Manual final candidate ready: {dest}", flush=True)
        return 0
    failures = []
    if args.command != "analyze":
        begin = time.monotonic()
        budget = args.max_minutes if args.max_minutes is not None else config["optional_checks_time_budget_minutes"]
        for job in jobs:
            if (args.cases and job["case"] not in args.cases) or (args.dates and job["date"] not in args.dates) or (args.seeds and job["seed"] not in args.seeds):
                continue
            if time.monotonic() - begin > budget * 60:
                print("Optional time budget reached; primary submission is retained", flush=True)
                break
            try:
                execute(args, config, base, cache, arrays, job)
            except Exception:
                error = {"job": job, "traceback": traceback.format_exc()}
                failures.append(error)
                directory = args.out_dir / "checks" / "jobs" / job_key(job)
                directory.mkdir(parents=True, exist_ok=True)
                atomic_json(directory / "error.json", error)
                print(error["traceback"], flush=True)
    summarize(args, config, cache, reference, jobs)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
