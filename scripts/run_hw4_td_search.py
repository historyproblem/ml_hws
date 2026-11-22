"""Two author-derived CatBoost presets as one nested old-quarter tuner.

Original CB_TD core parameters and a matched L2=1 control are compared only on
the previous INNER quarter. Raw and one fixed75%previous07+25%TD probability
mixture are gated against07; context04 remains a diagnostic. No Q2/Q3 tuning.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
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
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_hw4_final_research as primary
from hw4_inspections.boosted_search import fit_fixed, fit_path, predict
from hw4_inspections.td_presets import TD_RECIPES, choose_td

GRID = (100, 250, 500, 1000)
SEEDS = (42, 17, 2026)
REFERENCE = "previous07"
OOF07_SHA = "58755d1e12123d1dd18553ac45b5075b3b503cf3e1e5b5d4dd97ce526fc5c9a0"
TEST07_SHA = "ef638b33e25e391f943c8a3758a0ff27618b23467b01ae0a5d227aa5d9b4d0f0"


def assess(stage, directory, meta, anchor, context, value, bootstrap):
    values = {REFERENCE: anchor, "context04": context}
    if value is not None:
        values.update(td_raw=value, td_blend75=.75 * anchor + .25 * value)
    result = primary.evaluate_candidates(meta, values, reference=REFERENCE, bootstrap=bootstrap, seed=20261122)
    summary = result["summary"].copy()
    summary["eligible"] &= summary.worst_quarter_delta.ge(-.005)
    summary = summary.sort_values(["mean_ap", "model"], ascending=[False, True], kind="stable")
    primary.write_csv(directory / f"{stage}-summary.csv", summary)
    primary.write_csv(directory / f"{stage}-quarter-metrics.csv", result["quarter_scores"])
    primary.write_csv(directory / f"{stage}-oof.csv", pd.concat([meta, pd.DataFrame(values)], axis=1))
    eligible = summary.loc[summary.eligible & summary.model.isin(["td_raw", "td_blend75"])]
    fields = ["model", "mean_ap", "delta_vs_reference", "nonnegative_quarters", "worst_quarter_delta"]
    choice = eligible[fields].iloc[0].to_dict() if len(eligible) else None
    primary.write_json(directory / f"{stage}-selection.json", choice)
    return choice


def run(args):
    if args.threads not in (1, 2) or args.bootstrap < 0 or (not args.smoke and not args.bootstrap):
        raise ValueError("Use one/twoCPU threads; bootstrap0 is synthetic-smoke only")
    started = time.monotonic()

    def log(event, **details):
        print(json.dumps(dict(event=event, elapsed_seconds=round(time.monotonic() - started, 3), **details)), flush=True)

    recipes = deepcopy(TD_RECIPES)
    grid = list(GRID)
    if args.smoke:
        for recipe in recipes.values():
            recipe["cap"] = 5
        grid = [2, 5]
    plan = primary.provenance(args, recipes)
    directory = ROOT / "src/hw4_inspections"
    plan["source_sha256"].update({n: primary.sha(directory / n) for n in ["td_presets.py", "boosted_search.py"]})
    plan["source_sha256"][Path(__file__).name] = primary.sha(Path(__file__))
    extras = {"previous07_oof": primary.sha(args.additive_oof), "previous07_test": primary.sha(args.additive_test)}
    if not args.smoke and extras != {"previous07_oof": OOF07_SHA, "previous07_test": TEST07_SHA}:
        raise ValueError("Previous07 anchors differ from the verified fixed recipe")
    plan.update(
        experiment="CB_TD author-derived original and matchedL2 control nested tuner",
        author_source="https://github.com/dholzmueller/pytabkit/blob/main/pytabkit/models/sklearn/default_params.py#L87-L92",
        author_core="CB_TD_CLASS:1000trees,.08lr,d7,L2=1e-5,Plain,Bernoulli.9,random_strength.8,one_hot15,leaf_estimation_iterations1,max_bin254",
        adaptation="Our causal346 view and temporal folds; authors early_stopping300 replaced by the frozen coarse INNER sklearnAP grid. This is not a reproduction of their benchmark.",
        recipes=recipes, grid=grid, additional_anchors=extras, admission_reference=REFERENCE,
        selection="Two configs→onealgorithm: each outer recipe/count chosen ONLY on previous inner AP; lower selected logloss/earlierN/lexicaltie. All6innerpaths finish before outer predictions.",
        tree_selection="One max1000 innerfit per preset/fold; coarse100/250/500/1000AP only; earliest same-recipe tie.",
        final="Bestmean3OLD INNER bestAP recipe; median3selectedcounts for that SAME recipe. Never tune on2024Q2/Q3.",
        scores="raw probabilities and ONE fixed .75*previous07+.25*TD probability mixture;04 is diagnostic only",
        admission="mean AP delta to07>0; nonnegative in>=2/3oldquarters; worst>=-.005",
        confirmation="Only after screen admission:17/2026keep42-selectedperfoldrecipe/count; equal probabilitymean all3seeds; repeat same per-export gate",
        maximum_inner_fits=6, maximum_outer42_fits=3, maximum_confirmation_fits=6, maximum_full_fits=3,
        interpretation="Adaptive continuation after prior searches on reused olddev. Not independent confirmation; nominal site intervals only.",
        new_q2_q3_metrics=False, uploads=0)
    plan = json.loads(json.dumps(plan, sort_keys=True, allow_nan=False))
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    plan["signature"] = signature
    freeze = args.out / "freeze.json"
    if freeze.exists():
        if not args.resume or primary.read_json(freeze) != plan:
            raise ValueError("Source, data, anchors, presets or settings changed; resume refused")
    else:
        if any(p.name != ".lock" for p in args.out.iterdir()):
            raise ValueError("NewTD run requires an empty output folder")
        primary.write_json(freeze, plan)
    manifest_path = args.out / "forecast-manifest.json"
    if manifest_path.exists():
        old = primary.read_json(manifest_path)
        if old["signature"] != signature or any(primary.sha(args.out / p) != h for p, h in old["output_sha256"].items()):
            raise ValueError("Completed TD output changed")
    train, test, sample, views, cats = primary.validate_cache(joblib.load(args.feature_cache), args)
    frame = views["base"]
    meta = pd.concat([train.loc[train.forecast_period.eq(q), list(primary.REQUIRED)] for q in primary.QUARTERS], ignore_index=True)
    context, _ = primary.align_anchors(meta, test, args)
    previous = primary.read_csv(args.additive_oof)
    if not previous.row_id.is_unique or set(previous.row_id) != set(meta.row_id):
        raise ValueError("Previous07 OOF IDs differ")
    previous = previous.set_index("row_id").loc[meta.row_id].reset_index()
    pd.testing.assert_frame_equal(meta, previous[list(primary.REQUIRED)], check_dtype=False)
    anchor = primary.probabilities(previous.additive_logit__blend75, len(meta))
    test_anchor = primary.read_csv(args.additive_test)
    anchored_test = primary.validate_submission(test_anchor, test, test_anchor.target, len(test))
    anchor_test = primary.probabilities(anchored_test.target, len(test))
    args.out.joinpath("inner").mkdir(exist_ok=True)
    args.out.joinpath("models").mkdir(exist_ok=True)
    records = {n: {} for n in recipes}
    new_inner = reused_inner = new_models = reused_models = 0

    def validate_job(path, identity, expected):
        rec = primary.read_json(path / "complete.json")
        if (rec["identity"] != identity or not rec["reload_exact"] or set(rec["sha256"]) != set(expected)
                or any(primary.sha(path / n) != h for n, h in rec["sha256"].items())):
            raise ValueError("Changed TD job identity/checkpoint")
        return rec

    for name, recipe in recipes.items():
        for q in primary.QUARTERS:
            inner_q = (pd.Period(q, freq="Q") - 1).start_time.strftime("%Y-%m-%d")
            fitting = np.flatnonzero(train.forecast_period.lt(inner_q))
            validation = np.flatnonzero(train.forecast_period.eq(inner_q))
            identity = dict(signature=signature, stage="inner", recipe=name, outer=q, inner=inner_q,
                            seed=42, fit_rows=len(fitting), validation_rows=len(validation))
            path = args.out / "inner" / f"{name}__{q}"
            if path.exists():
                if not args.resume:
                    raise ValueError("Existing TD job requires --resume")
                validate_job(path, identity, ["model.joblib", "curve.json"])
                record = primary.read_json(path / "curve.json")
                reused_inner += 1
                log("inner_reused", recipe=name, outer=q)
            else:
                log("inner_start", recipe=name, outer=q)
                with threadpool_limits(limits=args.threads):
                    result = fit_path(recipe, frame.iloc[fitting], train.target.iloc[fitting],
                                      frame.iloc[validation], train.target.iloc[validation], cats,
                                      seed=42, threads=args.threads, cap=recipe["cap"], grid=grid)
                record = dict(inner_quarter=inner_q, best_count=int(result["best_count"]), curve=result["curve"], seconds=result["seconds"])
                with TemporaryDirectory(dir=path.parent, prefix=".pending-") as temporary:
                    job = Path(temporary) / "job"
                    job.mkdir()
                    joblib.dump(result["bundle"], job / "model.joblib", compress=1)
                    if not np.array_equal(predict(result["bundle"], frame.iloc[validation]),
                                          predict(joblib.load(job / "model.joblib"), frame.iloc[validation])):
                        raise ValueError("TD inner checkpoint reload mismatch")
                    primary.write_json(job / "curve.json", record)
                    primary.write_json(job / "complete.json", dict(identity=identity, reload_exact=True,
                        sha256={n: primary.sha(job / n) for n in ["model.joblib", "curve.json"]}))
                    os.rename(job, path)
                new_inner += 1
                log("inner_complete", recipe=name, outer=q, count=record["best_count"], seconds=record["seconds"])
            records[name][q] = record
    choices = choose_td(records, recipes)
    primary.write_json(args.out / "inner-selection.json", choices)
    log("inner_selection_frozen", choices=choices)
    created = set()

    def fixed_probability(name, q, count, seed):
        nonlocal new_models, reused_models
        fitting = np.arange(len(train)) if q == "full" else np.flatnonzero(train.forecast_period.lt(q))
        validation = np.arange(len(train), len(frame)) if q == "full" else np.flatnonzero(train.forecast_period.eq(q))
        identity = dict(signature=signature, stage="full" if q == "full" else "outer", recipe=name,
                        period=q, count=count, seed=seed, fit_rows=len(fitting), prediction_rows=len(validation))
        path = args.out / "models" / f"{name}__{q}__n{count}__s{seed}"
        if path.exists():
            if not args.resume and path not in created:
                raise ValueError("Existing TD model requires --resume")
            validate_job(path, identity, ["model.joblib", "prediction.npy"])
            reused_models += 1
            return primary.probabilities(np.load(path / "prediction.npy", allow_pickle=False), len(validation))
        log("fixed_start", recipe=name, period=q, count=count, seed=seed)
        with threadpool_limits(limits=args.threads):
            bundle = fit_fixed(recipes[name], frame.iloc[fitting], train.target.iloc[fitting], cats,
                               count=count, seed=seed, threads=args.threads)
            probability = primary.probabilities(predict(bundle, frame.iloc[validation]), len(validation))
        with TemporaryDirectory(dir=path.parent, prefix=".pending-") as temporary:
            job = Path(temporary) / "job"
            job.mkdir()
            joblib.dump(bundle, job / "model.joblib", compress=1)
            if not np.array_equal(probability, predict(joblib.load(job / "model.joblib"), frame.iloc[validation])):
                raise ValueError("TD fixed-model reload mismatch")
            np.save(job / "prediction.npy", probability, allow_pickle=False)
            primary.write_json(job / "complete.json", dict(identity=identity, reload_exact=True,
                sha256={n: primary.sha(job / n) for n in ["model.joblib", "prediction.npy"]}))
            os.rename(job, path)
        new_models += 1
        created.add(path)
        log("fixed_complete", recipe=name, period=q, count=count, seed=seed)
        return probability

    def old_forecasts(seed):
        return np.concatenate([fixed_probability(choices["outer"][q]["recipe"], q,
                              choices["outer"][q]["count"], seed) for q in primary.QUARTERS])

    screen = old_forecasts(42)
    selected = assess("screen", args.out, meta, anchor, context, screen, args.bootstrap)
    confirmed_choice = None
    confirmed = None
    if selected:
        values = [screen, old_forecasts(17), old_forecasts(2026)]
        confirmed = np.mean(values, axis=0, dtype=np.float64)
    confirmed_choice = assess("confirmed", args.out, meta, anchor, context, confirmed, args.bootstrap)
    files = []
    if confirmed_choice:
        final = choices["final"]
        values = [fixed_probability(final["recipe"], "full", final["count"], seed) for seed in SEEDS]
        probability = np.mean(values, axis=0, dtype=np.float64)
        variant = "blend75" if confirmed_choice["model"] == "td_blend75" else "raw"
        if variant == "blend75":
            probability = .75 * anchor_test + .25 * probability
        submission = primary.validate_submission(test, sample, probability, len(test) if args.smoke else 2246)
        path = args.out / f"submission_td_tuned_{variant}.csv"
        primary.write_csv(path, submission)
        pd.testing.assert_frame_equal(submission, primary.read_csv(path), check_exact=True)
        files = [dict(csv=path.name, sha256=primary.sha(path), variant=variant, rows=len(submission),
                      final_recipe=final, old_mean_ap=confirmed_choice["mean_ap"],
                      old_delta_vs_previous07=confirmed_choice["delta_vs_reference"])]
    reports = [p for p in args.out.iterdir() if p.suffix in {".json", ".csv"} and p != manifest_path]
    inner_total = len(list((args.out / "inner").glob("*/complete.json")))
    models_total = len(list((args.out / "models").glob("*/complete.json")))
    manifest = dict(complete=True, signature=signature, smoke=args.smoke, inner_fits_total=inner_total,
                    outer_full_fits_total=models_total, actual_model_fits_total=inner_total + models_total,
                    new_inner_fits_this_invocation=new_inner, reused_inner_jobs=reused_inner,
                    new_outer_full_fits_this_invocation=new_models, reused_model_requests=reused_models,
                    screen_eligible=selected is not None, confirmed_eligible=confirmed_choice is not None,
                    final_recipe=choices["final"], files=files, new_q2_q3_metrics=False, uploads=0,
                    elapsed_seconds=time.monotonic() - started, interpretation=plan["interpretation"],
                    output_sha256={p.name: primary.sha(p) for p in reports})
    primary.write_json(manifest_path, manifest)
    log("td_complete", files=files, model_fits_total=inner_total + models_total)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ["raw", "out", "feature-cache", "baseline-cv", "context-test", "additive-oof", "additive-test"]:
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
