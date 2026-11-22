"""Bounded INNER-AP stopping diagnostic using already saved primary curves.

Only cb_plain_d6 or xgb_d3_reg can be investigated, with a four-count inner
grid. No new inner fit, full prefix sweep, parameter change or later-quarter
metric is permitted. This is adaptive research on reused development data.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_hw4_final_research as primary

GRID = (100, 200, 400, 800)
EXPECTED_COUNTS = {"cb_plain_d6": (200, 100, 200), "xgb_d3_reg": (100, 200, 200)}
ORIGINAL_COUNTS = {"cb_plain_d6": (100, 100, 150), "xgb_d3_reg": (50, 50, 100)}


def derive_counts(case, curves):
    """Independently select only the four predeclared stored INNER AP points."""
    if case not in EXPECTED_COUNTS or set(curves) != set(primary.QUARTERS):
        raise ValueError("Exactly one permitted case and the three old inner curves are required")
    counts, evidence = {}, {}
    for q, expected, original in zip(primary.QUARTERS, EXPECTED_COUNTS[case], ORIGINAL_COUNTS[case]):
        curve = curves[q]
        inner_q = (pd.Period(q, freq="Q") - 1).start_time.strftime("%Y-%m-%d")
        if curve.get("inner_quarter") != inner_q or curve.get("iterations") != original:
            raise ValueError("Stored primary inner-quarter/count identity differs")
        selected = [row for row in curve["points"] if row["iterations"] in GRID]
        if len(selected) != len(GRID) or {row["iterations"] for row in selected} != set(GRID):
            raise ValueError("The stored curve must contain each coarse-grid count exactly once")
        if any(not np.isfinite(row["ap"]) or not 0 <= row["ap"] <= 1 for row in selected):
            raise ValueError("Stored inner AP must be finite and in [0,1]")
        best = max(selected, key=lambda row: (row["ap"], -row["iterations"]))
        if best["iterations"] != expected:
            raise ValueError(f"Coarse inner AP count changed for {case}/{q}: expected {expected}")
        counts[q] = int(best["iterations"])
        evidence[q] = {"inner_quarter": inner_q, "original_logloss_count": original,
                       "coarse_inner_ap_points": [{"iterations": int(row["iterations"]), "ap": float(row["ap"])}
                                                   for row in sorted(selected, key=lambda row: row["iterations"])],
                       "chosen_ap_count": counts[q], "tie_rule": "earliest count"}
    return counts, evidence


def validate_primary(args):
    """Verify source/cache/input/anchor/package fingerprints before any fit."""
    source_path = args.primary_run / "freeze.json"
    frozen = primary.read_json(source_path)
    recorded_signature = frozen.get("signature")
    original = {k: v for k, v in frozen.items() if k != "signature"}
    actual_signature = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
    if actual_signature != recorded_signature:
        raise ValueError("Primary freeze signature mismatch")
    expected = primary.provenance(args, frozen["pool"])
    if expected != original:
        raise ValueError("Primary sources, cache, data, anchor, packages or settings differ")
    current_case = primary.cases(False)[args.case]
    if frozen["pool"].get(args.case) != current_case:
        raise ValueError("The primary case is not the exact frozen bounded specification")
    if frozen["new_q2_q3_metrics"] is not False or frozen["old_folds"] != list(primary.QUARTERS):
        raise ValueError("Primary run must use old development only")
    return frozen, current_case


def run(args):
    if args.case not in EXPECTED_COUNTS:
        raise ValueError("Unsupported followup case")
    if args.threads < 1 or args.bootstrap < 0 or (not args.smoke and not args.bootstrap):
        raise ValueError("Positive limits required; bootstrap0 is synthetic-smoke only")
    frozen, original_case = validate_primary(args)
    cache = joblib.load(args.feature_cache)
    train, test, sample, views, cats = primary.validate_cache(cache, args)
    frame = views["base"]
    meta = pd.concat([train.loc[train.forecast_period.eq(q), list(primary.REQUIRED)]
                      for q in primary.QUARTERS], ignore_index=True)
    anchor, anchor_test = primary.align_anchors(meta, test, args)
    curves, source_jobs = {}, {}
    for q, original_count in zip(primary.QUARTERS, ORIGINAL_COUNTS[args.case]):
        train_rows = int(train.forecast_period.lt(q).sum())
        valid_rows = int(train.forecast_period.eq(q).sum())
        path = args.primary_run / "jobs" / f"{args.case}__{q}__s42"
        identity = {"signature": frozen["signature"], "case": args.case, "seed": 42,
                    "period": q, "fixed_count": None, "train_rows": train_rows, "predict_rows": valid_rows}
        probability, count = primary.read_job(path, identity, valid_rows)
        if count != original_count:
            raise ValueError("Original job count differs from the predeclared evidence")
        curve = primary.read_json(path / "curve.json")
        curves[q] = curve
        source_jobs[q] = {"path": str(path.resolve()), "identity": identity, "iterations": count,
                          "sha256": {name: primary.sha(path / name) for name in
                                     ["complete.json", "curve.json", "model.joblib", "prediction.npy"]}}
    counts, derivation = derive_counts(args.case, curves)
    name = f"{args.case}_inner_ap"
    case = deepcopy(original_case)
    case["inner"] = False  # All new jobs receive a fixed count; no inner fit call.
    plan = {"format": 1, "case": args.case, "new_case_name": name,
            "source_primary_signature": frozen["signature"],
            "source_primary_freeze_sha256": primary.sha(args.primary_run / "freeze.json"),
            "primary_provenance": {k: v for k, v in frozen.items() if k != "signature"},
            "source_primary_jobs": source_jobs, "coarse_inner_grid": list(GRID),
            "count_derivation": derivation, "counts": counts, "model": case,
            "followup_source_sha256": primary.sha(Path(__file__)),
            "output_directory": str(args.out.resolve()), "expected_final_median": 200,
            "screen_new_outer_fits": 2 if args.case == "cb_plain_d6" else 3,
            "screen_source_reuse": "D6 middle quarter100trees only" if args.case == "cb_plain_d6" else "none",
            "confirmation": "Only after screen gate: seeds17/2026 on the SAME counts, six outer fits; raw probability mean over42/17/2026 before blends",
            "final": "Only after three-seed gate: three full fits at median200; export one best eligible raw/blend75 variant",
            "gate": {"mean_delta_exclusive": 0, "nonnegative_quarters_at_least": 2,
                     "worst_quarter_delta_at_least": primary.WORST_FLOOR},
            "exports": "raw and fixed 0.75*context_probability+0.25*candidate_probability; no rank or weight search",
            "new_inner_max_fits": 0, "new_full_prefix_sweeps": 0,
            "new_q2_q3_metrics": False, "uploads": 0, "smoke": args.smoke,
            "interpretation": "Adaptive hypothesis after the primary screen on reused old dev; not an unbiased holdout or independent validation."}
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    plan["signature"] = signature
    plan_path = args.out / "freeze.json"
    if plan_path.exists():
        if not args.resume or primary.read_json(plan_path) != plan:
            raise ValueError("Changed followup source, count evidence or fingerprints; resume refused")
    else:
        if any(path.name != ".lock" for path in args.out.iterdir()):
            raise ValueError("A new followup requires an empty output directory")
        primary.write_json(plan_path, plan)  # Frozen BEFORE any new outer fit.
    manifest_path = args.out / "forecast-manifest.json"
    if manifest_path.exists():
        previous = primary.read_json(manifest_path)
        if (not args.resume or previous["signature"] != signature
                or any(primary.sha(args.out / p) != h for p, h in previous["output_sha256"].items())):
            raise ValueError("Changed completed followup outputs")
    pool = {name: case}
    new_jobs, reused_jobs = 0, 0

    def fixed_job(seed, q, count):
        nonlocal new_jobs, reused_jobs
        destination = args.out / "jobs" / f"{name}__{q}__s{seed}"
        existed = destination.exists()
        probability, actual_count = primary.run_job(args.out, signature, name, case, seed, q, count,
            train, frame, cats, args.threads, args.resume)
        if actual_count != count:
            raise ValueError("A fixed followup tree count changed")
        if existed:
            reused_jobs += 1
        else:
            new_jobs += 1
        return probability

    middle_replay = None
    fold_predictions = []
    for q in primary.QUARTERS:
        if args.case == "cb_plain_d6" and q == primary.QUARTERS[1]:
            if counts[q] != source_jobs[q]["iterations"]:
                raise ValueError("The reused middle count must be unchanged")
            path = Path(source_jobs[q]["path"])
            record = primary.read_json(path / "complete.json")
            probability, _ = primary.read_job(path, source_jobs[q]["identity"],
                                               int(train.forecast_period.eq(q).sum()))
            bundle = joblib.load(path / "model.joblib")
            if (bundle["spec"] != case["spec"] or bundle["features"] != list(frame)
                    or bundle["cats"] != cats or bundle["seed"] != 42 or bundle["iterations"] != 100):
                raise ValueError("The primary middle model specification or schema differs")
            # Replay against official-ID-validated cache rows proves alignment,
            # rather than trusting only an array length or a historical flag.
            replay = primary.predict_booster(bundle, frame.iloc[np.flatnonzero(train.forecast_period.eq(q))])
            if not np.array_equal(probability, replay):
                raise ValueError("Reused primary middle forecast fails exact official-row replay")
            middle_replay = {"quarter": q, "source": source_jobs[q], "official_row_ids_sha256":
                             hashlib.sha256(json.dumps(train.loc[train.forecast_period.eq(q), "row_id"].tolist()).encode()).hexdigest(),
                             "replayed_forward_exact": True, "saved_reload_flag": record["reload_exact"]}
            primary.write_json(args.out / "reused-middle.json", middle_replay)
        else:
            probability = fixed_job(42, q, counts[q])
        fold_predictions.append(probability)
    screen_probability = np.concatenate(fold_predictions)
    screened = primary.assess("screen", args.out, meta, anchor, {name: screen_probability}, pool, args.bootstrap)
    confirmed, seed_probability = {}, [screen_probability]
    if screened:
        for seed in primary.SEEDS[1:]:
            fold = [fixed_job(seed, q, counts[q]) for q in primary.QUARTERS]
            seed_probability.append(np.concatenate(fold))
        confirmed[name] = np.mean(seed_probability, axis=0, dtype=np.float64)
    qualified = primary.assess("confirmed", args.out, meta, anchor, confirmed, pool, args.bootstrap)
    if len(qualified) > 1:
        raise AssertionError("Only one best variant may be exported")
    files = []
    if qualified:
        choice = qualified[0]
        count = int(np.median(list(counts.values())))
        if count != 200:
            raise ValueError("The final median must remain200")
        full = [fixed_job(seed, "full", count) for seed in primary.SEEDS]
        probability = np.mean(full, axis=0, dtype=np.float64)
        if choice["variant"] == "blend75":
            probability = .75 * anchor_test + .25 * probability
        submission = primary.validate_submission(test, sample, probability, len(test) if args.smoke else 2246)
        path = args.out / f"submission_{args.case}_inner_ap_{choice['variant']}.csv"
        primary.write_csv(path, submission)
        pd.testing.assert_frame_equal(submission, primary.read_csv(path), check_exact=True)
        files.append({**choice, "csv": path.name, "sha256": primary.sha(path),
                      "rows": len(submission), "full_trees": count})
    outputs = [path for path in args.out.iterdir() if path.suffix in {".csv", ".json"} and path != manifest_path]
    manifest = {"signature": signature, "complete": True, "case": args.case,
                "counts": counts, "new_inner_fits": 0, "new_full_prefix_sweeps": 0,
                "new_jobs_this_invocation": new_jobs, "reused_followup_jobs_this_invocation": reused_jobs,
                "primary_middle_reused_exact": middle_replay is not None,
                "screen_eligible": bool(screened), "confirmed_eligible": bool(qualified),
                "files": files, "primary": files[0]["csv"] if files else None,
                "smoke": args.smoke, "new_q2_q3_metrics": False, "uploads": 0,
                "interpretation": plan["interpretation"],
                "output_sha256": {path.name: primary.sha(path) for path in outputs}}
    primary.write_json(manifest_path, manifest)
    print(json.dumps({"event": "followup_complete", "case": args.case, "counts": counts,
                      "new_jobs": new_jobs, "files": files}, ensure_ascii=False), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ["raw", "feature-cache", "baseline-cv", "context-test", "primary-run", "out"]:
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--case", choices=sorted(EXPECTED_COUNTS), default="cb_plain_d6")
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
