"""Frozen earlier-quarter additive selection, matched control and gated finals."""
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_hw4_final_research as primary
from hw4_inspections.additive_search import (
    CLASSIFIER, RECIPES, choose_final, choose_inner,
    fit_recipe, predict_recipe, recipe_by_id,
)
from hw4_inspections.next_round import evaluate_candidates

CORE = "core_C0.1"
DEFAULT_PLAN = ROOT / "docs/hw4-additive-search-plan-20261122.json"
sha, read_json, write_json = primary.sha, primary.read_json, primary.write_json
read_csv, write_csv = primary.read_csv, primary.write_csv


def old_core_evidence(directory, expected):
    if directory is None:
        return None
    frozen = read_json(directory / "freeze.json")
    if frozen["raw_sha256"] != expected["raw_sha256"] or frozen["cache_sha256"] != expected["cache_sha256"]:
        raise ValueError("Old fixed-core inputs/cache differ")
    for name in ["additive_logit.py", "classical_features.py"]:
        if frozen["source_sha256"][name] != expected["source_sha256"][name]:
            raise ValueError("Original fixed-core source differs")
    jobs = {}
    for q in [*primary.QUARTERS, "full"]:
        path = directory / "jobs" / q
        marker = read_json(path / "complete.json")
        if marker["identity"]["signature"] != frozen["signature"] or marker["reload_exact"] is not True:
            raise ValueError("Old fixed-core job identity/reload invalid")
        if any(sha(path / n) != h for n, h in marker["sha256"].items()):
            raise ValueError("Old fixed-core artifact changed")
        jobs[q] = {"marker_sha256": sha(path / "complete.json"), **marker["sha256"]}
    return dict(directory=str(directory.resolve()), freeze_sha256=sha(directory / "freeze.json"), jobs=jobs)


def make_freeze(args):
    plan = read_json(args.plan)
    if plan["recipes"] != list(RECIPES) or plan["recipe_count"] != 24:
        raise ValueError("Pre-results plan differs from the frozen 24 recipes")
    if plan["optimizer"]["max_iter"] != CLASSIFIER["max_iter"] or plan["optimizer"]["threads"] != 2:
        raise ValueError("Pre-results optimizer changed")
    frozen = primary.provenance(args, {r["id"]: r for r in RECIPES})
    module = ROOT / "src/hw4_inspections"
    frozen["source_sha256"].update({
        n: sha(module / n) for n in ["additive_logit.py", "classical_features.py", "additive_search.py"]})
    frozen["source_sha256"][Path(__file__).name] = sha(Path(__file__))
    frozen.update(plan_sha256=sha(args.plan), pre_results_plan=plan,
                  seeds=[], tree_selection="No trees/epochs. Fixed24recipes select only on previous inner AP.",
                  selection="Outer algorithm/control only; final majority/tie by mean inner AP; no recipe selection on outer AP")
    frozen["old_core"] = old_core_evidence(args.fixed_core, frozen)
    if not args.smoke and frozen["old_core"] is None:
        raise ValueError("Real search requires verified prior fixed-core jobs")
    return frozen


def load_job(path, identity):
    record = read_json(path / "complete.json")
    if record["identity"] != identity:
        raise ValueError("Changed additive-search job identity")
    if any(sha(path / n) != h for n, h in record["sha256"].items()):
        raise ValueError("Changed additive-search artifact")
    if record["status"] == "invalid":
        if set(record["sha256"]) != {"invalid-info.json"}:
            raise ValueError("Invalid numerical job artifact schema")
        return record, None
    if (record["status"] != "complete" or record["reload_exact"] is not True
            or set(record["sha256"]) != {"model.joblib", "prediction.npy", "fit-info.json"}):
        raise ValueError("Completed job artifact/reload schema")
    value = np.load(path / "prediction.npy", allow_pickle=False)
    if value.dtype != np.float64:
        raise ValueError("Nonfloat64 prediction artifact")
    return record, primary.probabilities(value, identity["predict_rows"])


def job(args, signature, recipe_id, quarter, role, train, frame):
    full = quarter == "full"
    fit = np.arange(len(train)) if full else np.flatnonzero(train.forecast_period.lt(quarter))
    pred = np.arange(len(train), len(frame)) if full else np.flatnonzero(train.forecast_period.eq(quarter))
    identity = dict(signature=signature, recipe=recipe_id, period=quarter,
                    train_rows=len(fit), predict_rows=len(pred))
    if not len(fit) or not len(pred):
        raise ValueError("Empty training/prediction fold; no quarter can be skipped")
    path = args.out / "jobs" / f"{quarter}__{recipe_id}"
    if path.exists():
        # First-role outer jobs may become next quarter's inner jobs in the
        # SAME invocation. Cross-invocation --resume is enforced by run/freeze.
        record, value = load_job(path, identity)
        print(json.dumps({"event": "job_reused", "period": quarter, "recipe": recipe_id, "role": role}), flush=True)
        return record, value
    started = time.monotonic()
    print(json.dumps({"event": "fit_start", **identity, "role": role}), flush=True)
    bundle = None
    numerical = None
    try:
        reference = "2024-10-01" if full else quarter
        bundle = fit_recipe(frame.iloc[fit], train.target.iloc[fit], recipe_by_id(recipe_id),
                            periods=train.forecast_period.iloc[fit], reference=reference)
        value = primary.probabilities(predict_recipe(bundle, frame.iloc[pred]), len(pred))
    except (ConvergenceWarning, FloatingPointError, np.linalg.LinAlgError) as error:
        numerical = dict(type=type(error).__name__, message=str(error))
    path.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=path.parent, prefix=".pending-additive-search-") as tmp:
        tmp = Path(tmp) / "job"
        tmp.mkdir()
        record = dict(identity=identity, recipe=recipe_id, period=quarter, first_role=role,
                      status="invalid" if numerical else "complete", ap=None,
                      seconds=time.monotonic() - started)
        if numerical:
            write_json(tmp / "invalid-info.json", numerical)
            record.update(numerical_failure=numerical, reload_exact=False)
        else:
            joblib.dump(bundle, tmp / "model.joblib", compress=1)
            restored = predict_recipe(joblib.load(tmp / "model.joblib"), frame.iloc[pred])
            if not np.array_equal(value, restored):
                raise AssertionError("Model reload forward differs")
            np.save(tmp / "prediction.npy", value, allow_pickle=False)
            write_json(tmp / "fit-info.json", dict(
                recipe=bundle["recipe"], classifier=bundle["classifier"],
                features=bundle["feature_names"], iterations=bundle["iterations"],
                sample_weights=bundle["sample_weights"], threads=2))
            record.update(reload_exact=True, iterations=bundle["iterations"], features=len(bundle["feature_names"]))
            if not full:
                record["ap"] = float(average_precision_score(train.target.iloc[pred], value))
        record["sha256"] = {p.name: sha(p) for p in tmp.iterdir()}
        write_json(tmp / "complete.json", record)
        os.rename(tmp, path)
    print(json.dumps({"event": "job_complete", "period": quarter, "recipe": recipe_id,
                      "role": role, "status": record["status"], "ap": record["ap"],
                      "seconds": record["seconds"]}), flush=True)
    return record, None if numerical else value


def assess(args, meta, anchor, algorithm, core):
    predictions = {primary.ANCHOR: anchor, "algorithm_raw": algorithm,
                   "algorithm_blend75": .75 * anchor + .25 * algorithm,
                   "fixed_core_raw": core, "fixed_core_blend75": .75 * anchor + .25 * core}
    result = evaluate_candidates(meta, predictions, reference=primary.ANCHOR,
                                 bootstrap=args.bootstrap, seed=20261122)
    summary = result["summary"].copy()
    gates, paired = {}, []
    for variant in ["raw", "blend75"]:
        name, fixed = f"algorithm_{variant}", f"fixed_core_{variant}"
        comparison = evaluate_candidates(meta, {fixed: predictions[fixed], name: predictions[name]},
                                          reference=fixed, bootstrap=args.bootstrap, seed=20261122)
        paired.append(comparison["summary"].assign(variant=variant))
        row = summary.loc[summary.model.eq(name)].iloc[0]
        matched = comparison["summary"].loc[lambda f: f.model.eq(name)].iloc[0]
        passes = bool(row.eligible and row.worst_quarter_delta >= -.005
                      and matched.delta_vs_reference > 0 and matched.nonnegative_quarters >= 2)
        gates[variant] = dict(eligible=passes, mean_ap=float(row.mean_ap),
            delta_context=float(row.delta_vs_reference), nonnegative_context=int(row.nonnegative_quarters),
            worst_context=float(row.worst_quarter_delta), delta_fixed_core=float(matched.delta_vs_reference),
            nonnegative_fixed_core=int(matched.nonnegative_quarters))
    write_csv(args.out / "outer-quarter-metrics.csv", result["quarter_scores"])
    write_csv(args.out / "outer-summary.csv", summary)
    write_csv(args.out / "algorithm-vs-fixed-core.csv", pd.concat(paired, ignore_index=True))
    write_csv(args.out / "outer-oof.csv", pd.concat([meta, pd.DataFrame(predictions)], axis=1))
    write_json(args.out / "gates.json", gates)
    return gates


def old_replay(args, core_predictions, meta):
    if args.fixed_core is None:
        return {"status": "smoke_no_previous_jobs"}
    records = []
    for quarter in primary.QUARTERS:
        old = np.load(args.fixed_core / "jobs" / quarter / "prediction.npy", allow_pickle=False)
        current = core_predictions[meta.forecast_period.eq(quarter).to_numpy()]
        if len(old) != len(current):
            raise ValueError("Old core row count differs")
        target = meta.loc[meta.forecast_period.eq(quarter), "target"]
        records.append(dict(quarter=quarter, exact=bool(np.array_equal(old, current)),
            max_abs_probability_difference=float(np.max(np.abs(old-current))),
            ap_old=float(average_precision_score(target, old)),
            ap_current=float(average_precision_score(target, current))))
    return dict(status="checked", current_control="New2thread jobs, sometimes shared with inner grid; NOT reused old forecasts",
                max_iter_note="Old1500/new5000 only a cap; old core converged before1500. Other core transform/C/tol unchanged.",
                quarters=records)


def run(args):
    args.threads = 2
    if args.bootstrap < 0 or not args.smoke and not args.bootstrap:
        raise ValueError("Positive bootstrap required; zero only smoke")
    frozen = make_freeze(args)
    signature = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    frozen["signature"] = signature
    freeze = args.out / "freeze.json"
    if freeze.exists():
        if not args.resume or read_json(freeze) != frozen:
            raise ValueError("Changed plan/source/raw/cache/anchors/packages; resume refused")
    else:
        if any(p.name not in {".lock", "plan.json", "run.log"} for p in args.out.iterdir()):
            raise ValueError("New run requires an empty directory apart from its frozen plan/log")
        if (args.out / "plan.json").exists() and sha(args.out / "plan.json") != sha(args.plan):
            raise ValueError("Pre-results plan copy changed")
        write_json(freeze, frozen)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if previous["signature"] != signature or any(sha(args.out / n) != h for n,h in previous["output_sha256"].items()):
            raise ValueError("Completed search output changed")
    train, test, sample, views, _ = primary.validate_cache(joblib.load(args.feature_cache), args)
    frame = views["base"]
    meta = pd.concat([train.loc[train.forecast_period.eq(q), list(primary.REQUIRED)]
                      for q in primary.QUARTERS], ignore_index=True)
    anchor, anchor_test = primary.align_anchors(meta, test, args)
    inner_records, winners, algorithm, core = [], [], [], []
    decisions = args.out / "decisions"
    decisions.mkdir(exist_ok=True)
    for outer in primary.QUARTERS:
        inner = (pd.Period(outer, freq="Q") - 1).start_time.strftime("%Y-%m-%d")
        records = [job(args, signature, r["id"], inner, "inner", train, frame)[0] for r in RECIPES]
        selected = choose_inner(records)
        choice = dict(outer=outer, inner=inner, selected_recipe=selected,
                      inner_complete_sha256={r["recipe"]:sha(args.out / "jobs" / f"{inner}__{r['recipe']}" / "complete.json") for r in records})
        decision_path = decisions / f"{outer}.json"
        if decision_path.exists() and read_json(decision_path) != choice:
            raise ValueError("Earlier-quarter decision changed")
        write_json(decision_path, choice)
        winners.append(selected)
        inner_records.extend(records)
        selected_record, selected_p = job(args, signature, selected, outer, "algorithm_outer", train, frame)
        core_record, core_p = job(args, signature, CORE, outer, "fixed_core_outer", train, frame)
        if selected_record["status"] != "complete" or core_record["status"] != "complete":
            raise RuntimeError("Selected outer or stable control is numerically invalid; no fold dropped")
        algorithm.append(selected_p)
        core.append(core_p)
        print(json.dumps({"event":"outer_complete", **choice, "algorithm_ap":selected_record["ap"],
                          "core_ap":core_record["ap"]}), flush=True)
    algorithm, core = np.concatenate(algorithm), np.concatenate(core)
    write_csv(args.out / "inner-records.csv", pd.DataFrame([
        {"recipe":r["recipe"], "quarter":r["period"], "status":r["status"], "ap":r["ap"],
         "iterations":r.get("iterations"), "numerical_failure":json.dumps(r.get("numerical_failure"), sort_keys=True)} for r in inner_records]))
    write_json(args.out / "fixed-core-replay.json", old_replay(args, core, meta))
    gates = assess(args, meta, anchor, algorithm, core)
    final_recipe = choose_final(winners, inner_records)
    write_json(args.out / "final-recipe.json", dict(inner_winners=winners, selected_recipe=final_recipe,
                                                    selection_uses_outer_scores=False))
    files, full_source = [], None
    if any(g["eligible"] for g in gates.values()):
        if final_recipe == CORE and args.fixed_core is not None:
            value = primary.probabilities(np.load(args.fixed_core / "jobs/full/prediction.npy", allow_pickle=False), len(test))
            full_source = "Reused verified original core/C=.1 full forecast; no new full fit, duplicate prior recipe"
        else:
            record, value = job(args, signature, final_recipe, "full", "gated_full", train, frame)
            if record["status"] != "complete":
                raise RuntimeError("Selected gated full fit failed numerically; no replacement selected on test")
            full_source = "One new gated full fit"
        for variant, gate in gates.items():
            if not gate["eligible"]:
                continue
            p = value if variant == "raw" else .75*anchor_test+.25*value
            submission = primary.validate_submission(test, sample, p, len(test) if args.smoke else 2246)
            path = args.out / f"submission_additive_search_{variant}.csv"
            write_csv(path, submission)
            pd.testing.assert_frame_equal(submission, read_csv(path), check_exact=True)
            files.append(dict(variant=variant, csv=path.name, sha256=sha(path), rows=len(submission),
                              recipe=final_recipe, duplicate_prior_recipe=final_recipe==CORE and args.fixed_core is not None))
    outputs = [p for p in args.out.iterdir() if p.suffix in {".csv", ".json"} and p != manifest_path]
    manifest = dict(signature=signature, complete=True, inner_candidate_jobs=72,
        inner_winners=winners, final_recipe=final_recipe, gates=gates, files=files, full_source=full_source,
        new_q2_q3_metrics=False, uploads=0, outer_selection_of24recipes=False,
        output_sha256={p.name:sha(p) for p in outputs})
    write_json(manifest_path, manifest)
    print(json.dumps({"event":"complete", "final_recipe":final_recipe,"gates":gates,"files":files}, ensure_ascii=False), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["raw", "out", "feature-cache", "baseline-cv", "context-test"]:
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--fixed-core", type=Path)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args=parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args)


if __name__=="__main__":
    main()
