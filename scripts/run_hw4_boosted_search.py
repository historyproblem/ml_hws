"""Nested old-quarter boosting algorithm comparison; no new Q2/Q3 metrics.

All24 recipe paths are tuned on previous INNER quarters before any new outer
result is read. Outer predictions evaluate three family tuners and a pooled
inner selector. DART paths save actual historical checkpoints, not retrospective
prefixes of a later reweighted model. Outputs are gated against exact context04.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_hw4_final_research as primary
from hw4_inspections.boosted_search import RECIPES, fit_fixed, fit_path, predict

GRID = (100, 250, 500, 1000, 1600)
SCOPES = {"lgb_gbdt_tuned": "lgb_gbdt", "lgb_dart_tuned": "lgb_dart",
          "catboost_tuned": "catboost", "pooled_tuned": None}
SEEDS = (42, 17, 2026)
WORST_FLOOR = -.005


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def selected_point(record):
    rows = [row for row in record["curve"] if row["count"] == record["best_count"]]
    if len(rows) != 1:
        raise ValueError("Selected count must identify exactly one stored inner point")
    return rows[0]


def select_inner(records, recipes):
    """Each outer choice sees its own earlier inner quarter only.

    Full recipes aggregate the three old INNER best scores; no closed/latest
    outer labels or 2024Q2/Q3 validation enter this final-recipe choice.
    """
    outer, final = {}, {}
    for algorithm, family in SCOPES.items():
        allowed = [name for name, recipe in recipes.items() if family is None or recipe["family"] == family]
        if not allowed:
            raise ValueError(f"Empty family scope for {algorithm}")
        outer[algorithm] = {}
        for q in primary.QUARTERS:
            def score(name):
                row = selected_point(records[name][q])
                return (-row["ap"], row["logloss"], row["count"], name)
            name = min(allowed, key=score)
            outer[algorithm][q] = {"recipe": name, "count": records[name][q]["best_count"],
                                   "inner_quarter": records[name][q]["inner_quarter"],
                                   "inner_ap": selected_point(records[name][q])["ap"]}
        def aggregate_score(name):
            points = [selected_point(records[name][q]) for q in primary.QUARTERS]
            return (-float(np.mean([r["ap"] for r in points])),
                    float(np.mean([r["logloss"] for r in points])), name)
        name = min(allowed, key=aggregate_score)
        counts = [records[name][q]["best_count"] for q in primary.QUARTERS]
        final[algorithm] = {"recipe": name, "count": int(np.median(counts)),
                            "old_inner_counts": counts,
                            "mean_old_inner_best_ap": float(np.mean([
                                selected_point(records[name][q])["ap"] for q in primary.QUARTERS]))}
    return {"outer": outer, "final": final}


def assess(stage, directory, meta, anchor, values, bootstrap):
    vectors = {primary.ANCHOR: anchor}
    names = {}
    for algorithm, probability in values.items():
        for variant, p in [("raw", probability), ("blend75", .75 * anchor + .25 * probability)]:
            label = f"{algorithm}__{variant}"
            vectors[label] = p
            names[label] = (algorithm, variant)
    result = primary.evaluate_candidates(meta, vectors, reference=primary.ANCHOR,
                                         bootstrap=bootstrap, seed=20261122)
    summary = result["summary"].copy()
    summary["eligible"] &= summary.worst_quarter_delta.ge(WORST_FLOOR)
    summary["algorithm"] = [names.get(name, (None, None))[0] for name in summary.model]
    summary["variant"] = [names.get(name, (None, None))[1] for name in summary.model]
    summary = summary.sort_values(["mean_ap", "model"], ascending=[False, True], kind="stable")
    primary.write_csv(directory / f"{stage}-summary.csv", summary)
    primary.write_csv(directory / f"{stage}-quarter-metrics.csv", result["quarter_scores"])
    primary.write_csv(directory / f"{stage}-oof.csv", pd.concat([meta, pd.DataFrame(vectors)], axis=1))
    fields = ["model", "algorithm", "variant", "mean_ap", "delta_vs_reference",
              "nonnegative_quarters", "worst_quarter_delta"]
    choices = summary.loc[summary.eligible].drop_duplicates("algorithm")[fields].to_dict("records")
    return choices


def choose_confirmation(choices, vectors):
    """At most two distinct algorithm predictors, never two identical OOF sets."""
    selected, rejected = [], []
    for choice in choices:
        algorithm = choice["algorithm"]
        if any(np.array_equal(vectors[algorithm], vectors[old["algorithm"]]) for old in selected):
            rejected.append({"algorithm": algorithm, "reason": "Exact duplicate of a higher-priority algorithm OOF"})
        elif len(selected) < 2:
            selected.append(choice)
        else:
            rejected.append({"algorithm": algorithm, "reason": "Predeclared maximum of two family/pooled algorithms"})
    return selected, rejected


def run(args):
    if args.threads not in (1, 2) or args.bootstrap < 0 or (not args.smoke and not args.bootstrap):
        raise ValueError("Use one or two CPU threads; bootstrap0 is synthetic-smoke only")
    started = time.monotonic()

    def log(event, **detail):
        print(json.dumps({"event": event, "elapsed_seconds": round(time.monotonic() - started, 3),
                          **detail}), flush=True)

    recipes = deepcopy(RECIPES)
    grid = list(GRID)
    if args.smoke:
        for recipe in recipes.values():
            recipe["cap"] = 5
        grid = [2, 5]
    if len(recipes) != 24 or sorted(recipe["family"] for recipe in recipes.values()).count("lgb_dart") != 4:
        raise ValueError("The frozen24 recipe pool changed")
    # Reuse trusted cache/raw/anchor validation; no source or data reconstruction.
    plan = primary.provenance(args, recipes)
    module_dir = ROOT / "src" / "hw4_inspections"
    plan["source_sha256"].update({"boosted_search.py": primary.sha(module_dir / "boosted_search.py"),
                                   Path(__file__).name: primary.sha(Path(__file__))})
    dist = metadata.distribution("lightgbm")
    plan["packages"]["lightgbm"] = {"version": dist.version,
        "distribution_record_sha256": hashlib.sha256((dist.read_text("RECORD") or "").encode()).hexdigest()}
    plan.update(experiment="Nested24-recipe boosting search", recipes=recipes,
                inner_grid=grid, scopes=SCOPES, threads=args.threads, seeds=list(SEEDS),
                selection="For each old outer: choose config+count ONLY on the strict previous inner quarter by sklearn AP; ties lower logloss, earlier count, lexical ID. No outer score chooses among24 recipes.",
                tree_selection="One inner max-fit per recipe/fold; sklearnAP on the frozen coarse grid; earliest within-recipe AP tie. Genuine DART checkpoints. Outer/final fits have fixed selected counts.",
                final_recipe_selection="Within each algorithm scope: largest equal-weight mean of three OLD INNER best APs, then lower mean inner logloss and lexical recipe ID. Full count is median of that recipe's three inner-selected counts.",
                DART_selection="After-iteration immutable model snapshots at actual grid states; never truncate a later normalized DART booster retrospectively.",
                confirmation="Adaptive reused-old-outer choice of at most two non-identical family/pooled algorithm predictors; seeds17/2026 keep the per-fold42-selected recipe/count. No repeated inner tuning.",
                admission={"mean_delta_exclusive": 0, "minimum_nonnegative_quarters": 2,
                           "minimum_worst_quarter_delta": WORST_FLOOR},
                scores="Within-quarter sklearn AP, equal-quarter mean; raw probabilities and fixed .75 context04 +.25 candidate probabilities. Mean all three seed probabilities before blend.",
                maximum_inner_fits=72, maximum_screen_outer_fits=9,
                maximum_confirmation_outer_fits=12, maximum_full_fits=6,
                interpretation="Old development heavily reused; outer scores evaluate a frozen inner-selection algorithm. Selection of up to two algorithms for confirmation remains adaptive; intervals are nominal, not fresh holdout.",
                new_q2_q3_metrics=False, uploads=0)
    plan = canonical(plan)
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    plan["signature"] = signature
    frozen_path = args.out / "freeze.json"
    if frozen_path.exists():
        if not args.resume or primary.read_json(frozen_path) != plan:
            raise ValueError("Source, recipe, cache, anchor or environment changed; resume refused")
    else:
        if any(p.name != ".lock" for p in args.out.iterdir()):
            raise ValueError("Use an empty output directory for a new experiment")
        primary.write_json(frozen_path, plan)
    manifest_path = args.out / "forecast-manifest.json"
    if manifest_path.exists():
        previous = primary.read_json(manifest_path)
        if (previous["signature"] != signature or any(primary.sha(args.out / p) != h
                for p, h in previous["output_sha256"].items())):
            raise ValueError("Completed output artifact changed")
    train, test, sample, views, cats = primary.validate_cache(joblib.load(args.feature_cache), args)
    frame = views["base"]
    if not args.smoke and len([name for name in frame if not name.startswith("code")]) != 170:
        raise ValueError("The frozen no-code view must contain exactly170 features")
    meta = pd.concat([train.loc[train.forecast_period.eq(q), list(primary.REQUIRED)]
                      for q in primary.QUARTERS], ignore_index=True)
    anchor, anchor_test = primary.align_anchors(meta, test, args)
    recipes = canonical(recipes)
    records = {name: {} for name in recipes}
    new_inner = reused_inner = new_outer_full = reused_outer_full = 0
    args.out.joinpath("inner").mkdir(exist_ok=True)
    args.out.joinpath("models").mkdir(exist_ok=True)

    def load_job(path, identity, files):
        saved = primary.read_json(path / "complete.json")
        if saved["identity"] != identity or not saved["reload_exact"] or set(saved["sha256"]) != set(files):
            raise ValueError(f"Cached job identity or schema changed: {path}")
        if any(primary.sha(path / f) != h for f, h in saved["sha256"].items()):
            raise ValueError(f"Cached model artifact changed: {path}")
        return saved

    # All72 inner paths complete BEFORE the first new outer prediction.
    for name, recipe in recipes.items():
        for q in primary.QUARTERS:
            inner_q = (pd.Period(q, freq="Q") - 1).start_time.strftime("%Y-%m-%d")
            fitting = np.flatnonzero(train.forecast_period.lt(inner_q))
            validation = np.flatnonzero(train.forecast_period.eq(inner_q))
            identity = {"signature": signature, "stage": "inner", "recipe": name,
                        "outer_quarter": q, "inner_quarter": inner_q, "seed": 42,
                        "fit_rows": len(fitting), "validation_rows": len(validation)}
            path = args.out / "inner" / f"{name}__{q}"
            if path.exists():
                if not args.resume:
                    raise ValueError("Existing inner job requires --resume")
                saved = load_job(path, identity, ["model.joblib", "curve.json"])
                curve = primary.read_json(path / "curve.json")
                if saved["best_count"] != curve["best_count"]:
                    raise ValueError("Cached inner selection changed")
                reused_inner += 1
                log("inner_reused", recipe=name, outer=q, count=saved["best_count"])
            else:
                log("inner_start", recipe=name, family=recipe["family"], view=recipe["view"], outer=q,
                    fit_rows=len(fitting), validation_rows=len(validation))
                with threadpool_limits(limits=args.threads):
                    result = fit_path(recipe, frame.iloc[fitting], train.target.iloc[fitting],
                                      frame.iloc[validation], train.target.iloc[validation], cats,
                                      seed=42, threads=args.threads, cap=recipe["cap"], grid=grid)
                curve = {"inner_quarter": inner_q, "best_count": int(result["best_count"]),
                         "curve": result["curve"], "seconds": float(result["seconds"])}
                with TemporaryDirectory(dir=path.parent, prefix=".pending-") as temporary:
                    job = Path(temporary) / "job"
                    job.mkdir()
                    joblib.dump(result["bundle"], job / "model.joblib", compress=1)
                    with threadpool_limits(limits=args.threads):
                        expected = predict(result["bundle"], frame.iloc[validation])
                        reloaded = predict(joblib.load(job / "model.joblib"), frame.iloc[validation])
                    if not np.array_equal(expected, reloaded):
                        raise ValueError("Inner actual-checkpoint reload is not exact")
                    primary.write_json(job / "curve.json", curve)
                    primary.write_json(job / "complete.json", {"identity": identity, "reload_exact": True,
                        "best_count": curve["best_count"], "sha256": {f: primary.sha(job / f)
                            for f in ["model.joblib", "curve.json"]}})
                    os.rename(job, path)
                new_inner += 1
                log("inner_complete", recipe=name, outer=q, count=curve["best_count"],
                    ap=selected_point(curve)["ap"], seconds=curve["seconds"],
                    inner_paths_completed=new_inner + reused_inner, inner_paths_total=72)
            records[name][q] = curve
    selection = select_inner(records, recipes)
    primary.write_json(args.out / "inner-selection.json", selection)
    log("inner_selection_frozen", selections=selection)

    created_fixed = set()

    def fixed_probability(recipe_id, q, count, seed):
        nonlocal new_outer_full, reused_outer_full
        fitting = np.arange(len(train)) if q == "full" else np.flatnonzero(train.forecast_period.lt(q))
        validation = np.arange(len(train), len(frame)) if q == "full" else np.flatnonzero(train.forecast_period.eq(q))
        identity = {"signature": signature, "stage": "full" if q == "full" else "outer",
                    "recipe": recipe_id, "period": q, "count": count, "seed": seed,
                    "fit_rows": len(fitting), "prediction_rows": len(validation)}
        path = args.out / "models" / f"{recipe_id}__{q}__n{count}__s{seed}"
        if path.exists():
            if not args.resume and path not in created_fixed:
                raise ValueError("Existing fixed model requires --resume")
            load_job(path, identity, ["model.joblib", "prediction.npy"])
            value = primary.probabilities(np.load(path / "prediction.npy", allow_pickle=False), len(validation))
            reused_outer_full += 1
            log("model_reused", recipe=recipe_id, period=q, count=count, seed=seed)
            return value
        log("fixed_fit_start", recipe=recipe_id, period=q, count=count, seed=seed, rows=len(fitting))
        with threadpool_limits(limits=args.threads):
            bundle = fit_fixed(recipes[recipe_id], frame.iloc[fitting], train.target.iloc[fitting], cats,
                               count=count, seed=seed, threads=args.threads)
            value = primary.probabilities(predict(bundle, frame.iloc[validation]), len(validation))
        with TemporaryDirectory(dir=path.parent, prefix=".pending-") as temporary:
            job = Path(temporary) / "job"
            job.mkdir()
            joblib.dump(bundle, job / "model.joblib", compress=1)
            with threadpool_limits(limits=args.threads):
                replay = predict(joblib.load(job / "model.joblib"), frame.iloc[validation])
            if not np.array_equal(value, replay):
                raise ValueError("Fixed-model serialization reload differs")
            np.save(job / "prediction.npy", value, allow_pickle=False)
            primary.write_json(job / "complete.json", {"identity": identity, "reload_exact": True,
                "sha256": {f: primary.sha(job / f) for f in ["model.joblib", "prediction.npy"]}})
            os.rename(job, path)
        new_outer_full += 1
        created_fixed.add(path)
        log("fixed_fit_complete", recipe=recipe_id, period=q, count=count, seed=seed)
        return value

    screen = {}
    for algorithm, choices in selection["outer"].items():
        screen[algorithm] = np.concatenate([fixed_probability(choices[q]["recipe"], q, choices[q]["count"], 42)
                                            for q in primary.QUARTERS])
    choices = assess("screen", args.out, meta, anchor, screen, args.bootstrap)
    chosen, rejected = choose_confirmation(choices, screen)
    primary.write_json(args.out / "confirmation-selection.json", {"selected": chosen, "rejected": rejected,
                        "interpretation": "Adaptive selection among four frozen nested algorithms on reused old outer quarters"})
    confirmed = {}
    for choice in chosen:
        algorithm = choice["algorithm"]
        seeds = [screen[algorithm]]
        for seed in SEEDS[1:]:
            seeds.append(np.concatenate([fixed_probability(selection["outer"][algorithm][q]["recipe"], q,
                selection["outer"][algorithm][q]["count"], seed) for q in primary.QUARTERS]))
        confirmed[algorithm] = np.mean(seeds, axis=0, dtype=np.float64)
    final_choices = assess("confirmed", args.out, meta, anchor, confirmed, args.bootstrap)
    files, seen_sha = [], {}
    for choice in final_choices:
        algorithm, variant = choice["algorithm"], choice["variant"]
        final = selection["final"][algorithm]
        probabilities = [fixed_probability(final["recipe"], "full", final["count"], seed) for seed in SEEDS]
        value = np.mean(probabilities, axis=0, dtype=np.float64)
        if variant == "blend75":
            value = .75 * anchor_test + .25 * value
        submission = primary.validate_submission(test, sample, value, len(test) if args.smoke else 2246)
        filename = f"submission_{algorithm}_{variant}.csv"
        primary.write_csv(args.out / filename, submission)
        pd.testing.assert_frame_equal(submission, primary.read_csv(args.out / filename), check_exact=True)
        digest = primary.sha(args.out / filename)
        duplicate_of = seen_sha.get(digest)
        seen_sha[digest] = filename
        files.append({"algorithm": algorithm, "variant": variant, "csv": filename, "sha256": digest,
                      "rows": len(submission), "final_inner_selected_recipe": final,
                      "duplicate_csv_of": duplicate_of, "old_mean_ap": choice["mean_ap"],
                      "old_delta_vs_reference": choice["delta_vs_reference"]})
    reports = [p for p in args.out.iterdir() if p.suffix in {".csv", ".json"} and p != manifest_path]
    inner_total = sum(1 for _ in (args.out / "inner").glob("*/complete.json"))
    fixed_total = sum(1 for _ in (args.out / "models").glob("*/complete.json"))
    manifest = {"signature": signature, "complete": True, "smoke": args.smoke,
                "inner_max_fits_total": inner_total, "outer_full_fits_total": fixed_total,
                "actual_model_fits_total": inner_total + fixed_total,
                "new_inner_fits_this_invocation": new_inner, "reused_inner_jobs": reused_inner,
                "new_outer_full_fits_this_invocation": new_outer_full,
                "reused_outer_full_requests": reused_outer_full,
                "screen_algorithm_count": len(screen), "confirmed_algorithms": list(confirmed),
                "final_files": files, "new_q2_q3_metrics": False, "uploads": 0,
                "elapsed_seconds": time.monotonic() - started,
                "interpretation": plan["interpretation"],
                "output_sha256": {p.name: primary.sha(p) for p in reports}}
    primary.write_json(manifest_path, manifest)
    log("complete", files=files, model_fits_total=manifest["actual_model_fits_total"])
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ["raw", "out", "feature-cache", "baseline-cv", "context-test"]:
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
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
