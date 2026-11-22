"""Four preregistered greedy/canonical EBM recipes, one nested algorithm.

Matched A/B change only the greedy scheduler. C/D use bounded upstream-style
bagging and random validation entirely within the already-past fit pool.
All twelve old inner fits finish before any outer forecast. External quarter
labels never enter a fit. Admission uses previous07 with a strong-GAM guard.
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
from sklearn.metrics import average_precision_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_hw4_final_research as primary
from hw4_inspections.ebm_search import load_runtime, predict_ebm
from hw4_inspections.ebm_canonical import CANONICAL_RECIPES as EBM_RECIPES, fit_canonical as fit_ebm

SEEDS = (42, 17, 2026)
REFERENCE = "previous07"
OOF07_SHA = "58755d1e12123d1dd18553ac45b5075b3b503cf3e1e5b5d4dd97ce526fc5c9a0"
TEST07_SHA = "ef638b33e25e391f943c8a3758a0ff27618b23467b01ae0a5d227aa5d9b4d0f0"
DOMINATION_GAP = .002


def select_inner(records, recipes):
    """Choose configs only on earlier inner scores; full uses old INNER mean."""
    if set(records) != set(recipes):
        raise ValueError("All four frozen recipes must finish all three inner folds")
    outer = {}
    for q in primary.QUARTERS:
        def key(name):
            r = records[name][q]
            return (-r["ap"], r["logloss"], recipes[name]["params"]["max_rounds"], name)
        name = min(recipes, key=key)
        outer[q] = dict(recipe=name, inner_quarter=records[name][q]["inner_quarter"],
                        inner_ap=records[name][q]["ap"], max_rounds_cap=recipes[name]["params"]["max_rounds"])
    def mean_key(name):
        return (-float(np.mean([records[name][q]["ap"] for q in primary.QUARTERS])),
                float(np.mean([records[name][q]["logloss"] for q in primary.QUARTERS])), name)
    name = min(recipes, key=mean_key)
    return dict(outer=outer, final=dict(recipe=name, max_rounds_cap=recipes[name]["params"]["max_rounds"],
                mean_old_inner_ap=float(np.mean([records[name][q]["ap"] for q in primary.QUARTERS]))))


def domination(meta, raw, anchor, gam):
    """Predeclared guard: both variants lose every quarter and mean by>=.002."""
    if raw is None:
        return dict(evaluated=False, full_stop=False)
    details = {}
    for name, value in [("raw", raw), ("blend75", .75 * anchor + .25 * raw)]:
        differences = []
        for q in primary.QUARTERS:
            mask = meta.forecast_period.eq(q).to_numpy()
            differences.append(float(average_precision_score(meta.target[mask], value[mask])
                                     - average_precision_score(meta.target[mask], gam[mask])))
        details[name] = dict(quarter_deltas=differences, mean_delta=float(np.mean(differences)),
                             all_three_negative=all(v < 0 for v in differences),
                             clearly_dominated=bool(all(v < 0 for v in differences)
                                                     and np.mean(differences) <= -DOMINATION_GAP))
    return dict(evaluated=True, threshold=DOMINATION_GAP, variants=details,
                full_stop=all(r["clearly_dominated"] for r in details.values()))


def assess(stage, out, meta, anchor, context, gam, raw, bootstrap):
    vectors = {REFERENCE: anchor, "context04": context, "new_gam_raw": gam}
    if raw is not None:
        vectors.update(ebm_raw=raw, ebm_blend75=.75 * anchor + .25 * raw)
    r = primary.evaluate_candidates(meta, vectors, reference=REFERENCE, bootstrap=bootstrap, seed=20261122)
    summary = r["summary"].copy()
    summary["eligible"] &= summary.worst_quarter_delta.ge(-.005) & summary.model.isin(["ebm_raw", "ebm_blend75"])
    summary = summary.sort_values(["mean_ap", "model"], ascending=[False, True], kind="stable")
    primary.write_csv(out / f"{stage}-summary.csv", summary)
    primary.write_csv(out / f"{stage}-quarter-metrics.csv", r["quarter_scores"])
    primary.write_csv(out / f"{stage}-oof.csv", pd.concat([meta, pd.DataFrame(vectors)], axis=1))
    fields = ["model", "mean_ap", "delta_vs_reference", "nonnegative_quarters", "worst_quarter_delta"]
    eligible = summary.loc[summary.eligible, fields]
    chosen = eligible.iloc[0].to_dict() if len(eligible) else None
    primary.write_json(out / f"{stage}-selection.json", chosen)
    primary.write_json(out / f"{stage}-GAM-domination.json", domination(meta, raw, anchor, gam))
    return chosen


def runtime_evidence(target):
    load_runtime(target)
    dist = metadata.distribution("interpret-core")
    files = {}
    for relative in dist.files or []:
        name = str(relative)
        if name.endswith((".dylib", ".so", ".dll")) or name.endswith("glassbox/_ebm/_ebm.py"):
            files[name] = primary.sha(dist.locate_file(relative))
    return dict(version=dist.version, target=str(Path(target).resolve()),
                distribution_record_sha256=hashlib.sha256((dist.read_text("RECORD") or "").encode()).hexdigest(),
                native_and_ebm_source_sha256=files)


def run(args):
    if args.threads not in (1, 2) or args.bootstrap < 0 or (not args.smoke and not args.bootstrap):
        raise ValueError("Use one/twoCPU threads; bootstrap0 is synthetic-smoke only")
    started = time.monotonic()
    def log(event, **details):
        print(json.dumps(dict(event=event, elapsed_seconds=round(time.monotonic() - started, 3), **details)), flush=True)
    runtime = runtime_evidence(args.runtime)
    recipes = deepcopy(EBM_RECIPES)
    if args.smoke:
        for recipe in recipes.values():
            recipe["params"]["max_rounds"] = 2 if recipe["params"]["max_rounds"] == 300 else 3
    if len(recipes) != 4:
        raise ValueError("The frozen EBM continuation has exactly four recipes")
    plan = primary.provenance(args, recipes)
    plan["source_sha256"].update({"ebm_search.py": primary.sha(ROOT / "src/hw4_inspections/ebm_search.py"),
                                  "ebm_canonical.py": primary.sha(ROOT / "src/hw4_inspections/ebm_canonical.py"),
                                  Path(__file__).name: primary.sha(Path(__file__))})
    extras = dict(previous07_oof=primary.sha(args.additive_oof), previous07_test=primary.sha(args.additive_test),
                  new_gam_oof=primary.sha(args.gam_oof))
    if not args.smoke and (extras["previous07_oof"] != OOF07_SHA or extras["previous07_test"] != TEST07_SHA):
        raise ValueError("Verified previous07 anchors changed")
    plan.update(
        experiment="Four greedy/canonical EBM recipes→one nested old-quarter algorithm", recipes=recipes,
        official_runtime=runtime, additional_anchors=extras, new_gam_column="algorithm_raw",
        native_behavior_note="All four use upstream greedy_ratio10/cyclic_progressFalse; training-gain greedy scheduling follows75 smoothing cycles. No forced cyclic progress. best_iteration_[stage,bag] records attempted update steps, not a retained-round prefix.",
        internal_validation_policy="A/B:validation0,onebag,noES,300fixedcycles. C/D:random15%validation per8bags ONLY from caller-suppliedpastFITrows,cap2000/ES50/tol1e-5. Nativebins see the full pastFITpool per upstream. This internal split is random, not temporal; external chronologically held-out labels/rows never enter fit.",
        full_training_scope="If admitted, full fitting uses all available train labels including2024Q2/Q3 as PASTFITdata. C/D internal mixed-past validation is part the fixed algorithm, never a separate recent-quarter diagnostic or tuning result.",
        primary_sources=["https://pypi.org/project/interpret-core/0.7.8/",
            "https://github.com/interpretml/interpret/blob/v0.7.8/python/interpret-core/interpret/glassbox/_ebm/_ebm.py"],
        selection="All12inner fits before any outer. Each old outer chooses among4fixedalgorithms on strict previous inner sklearnAP; ties lowerlogloss/cap/lexID. C/D native internal ES is separate from this external quarter score.",
        tree_selection="No prefixes or warm starts. A/Bfixed300cycles;C/Dfixed2000cap with upstream internal50roundES/tol1e-5 and8bags. Actualperbag/stageupdatecounts saved. No externalvalidationrowsconcatenated.",
        final="Largest equal-weight mean of three OLD INNER APs then lower meanlogloss/lexrecipe. Recipe fixes compact94,greedy,terms,cap and internalESpolicy; no2024Q2/Q3 external tuning.",
        scores="Raw and fixed .75*previous07+.25*EBM probabilities.07primary,04/newGAMdiagnostics. Mean probabilities of all42/17/2026seeds.",
        admission="meanΔ07>0,>=2/3nonnegativequarters,worst>=-.005. Screen pass→six17/2026fixed-recipeconfirmfits→samegate.",
        full_guard="Afterconfirmation,if BOTHraw ANDfixed07blend lose ALL3quarters to supplied newGAM and EACH mean gap>=.002,skipfull. This guard was declared before anyouterresults, not model tuning.",
        model_fit_budget=dict(inner=12, outer42=3, confirmation=6, full=3),
        interpretation="Adaptive source-grounded continuation on heavily reused old development; nominal site intervals and no independent holdout claim.",
        new_q2_q3_metrics=False, new_q2_q3_external_validation_metrics=False, uploads=0)
    plan = json.loads(json.dumps(plan, sort_keys=True, allow_nan=False))
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest();plan["signature"] = signature
    freeze = args.out / "freeze.json"
    if freeze.exists():
        if not args.resume or primary.read_json(freeze) != plan:
            raise ValueError("EBM source/runtime/data/anchors/recipes changed; resume refused")
    else:
        if any(p.name != ".lock" for p in args.out.iterdir()):
            raise ValueError("New EBM run requires an empty output folder")
        primary.write_json(freeze, plan)
    manifest_path = args.out / "forecast-manifest.json"
    if manifest_path.exists():
        old = primary.read_json(manifest_path)
        if old["signature"] != signature or any(primary.sha(args.out / p) != h for p,h in old["output_sha256"].items()):
            raise ValueError("Changed completed EBM output")
    train,test,sample,views,cats = primary.validate_cache(joblib.load(args.feature_cache),args);frame=views["base"]
    meta=pd.concat([train.loc[train.forecast_period.eq(q),list(primary.REQUIRED)] for q in primary.QUARTERS],ignore_index=True)
    context,_=primary.align_anchors(meta,test,args)
    def aligned(path):
        data=primary.read_csv(path)
        if not data.row_id.is_unique or set(data.row_id)!=set(meta.row_id):raise ValueError("EBM benchmark IDs differ")
        data=data.set_index("row_id").loc[meta.row_id].reset_index()
        pd.testing.assert_frame_equal(meta,data[list(primary.REQUIRED)],check_dtype=False)
        return data
    previous=aligned(args.additive_oof);anchor=primary.probabilities(previous.additive_logit__blend75,len(meta))
    gamframe=aligned(args.gam_oof);gam=primary.probabilities(gamframe.algorithm_raw,len(meta))
    test_anchor=primary.read_csv(args.additive_test)
    anchor_test=primary.validate_submission(test_anchor,test,test_anchor.target,len(test)).target.to_numpy(dtype=np.float64)
    args.out.joinpath("inner").mkdir(exist_ok=True);args.out.joinpath("models").mkdir(exist_ok=True)
    records={name:{} for name in recipes};new_inner=reused_inner=new_models=reused_models=0
    def validate_job(path,identity,files):
        r=primary.read_json(path/"complete.json")
        if (r["identity"]!=identity or not r["reload_exact"] or set(r["sha256"])!=set(files)
                or any(primary.sha(path/n)!=h for n,h in r["sha256"].items())):raise ValueError("Changed EBM job")
        return r
    for name,recipe in recipes.items():
        for q in primary.QUARTERS:
            inner_q=(pd.Period(q,freq="Q")-1).start_time.strftime("%Y-%m-%d")
            fitting=np.flatnonzero(train.forecast_period.lt(inner_q));validation=np.flatnonzero(train.forecast_period.eq(inner_q))
            identity=dict(signature=signature,stage="inner",recipe=name,outer=q,inner=inner_q,seed=42,fit_rows=len(fitting),validation_rows=len(validation))
            path=args.out/"inner"/f"{name}__{q}"
            if path.exists():
                if not args.resume:raise ValueError("Existing EBM path requires --resume")
                validate_job(path,identity,["model.joblib","scores.json"]);record=primary.read_json(path/"scores.json");reused_inner+=1
                log("inner_reused",recipe=name,outer=q)
            else:
                log("inner_start",recipe=name,outer=q,rows=len(fitting));t0=time.monotonic()
                bundle=fit_ebm(recipe,frame.iloc[fitting],train.target.iloc[fitting],cats,runtime=args.runtime,seed=42,threads=args.threads)
                value=primary.probabilities(predict_ebm(bundle,frame.iloc[validation]),len(validation))
                record=dict(inner_quarter=inner_q,ap=float(average_precision_score(train.target.iloc[validation],value)),
                    logloss=float(log_loss(train.target.iloc[validation],value,labels=[0,1])),seconds=time.monotonic()-t0)
                with TemporaryDirectory(dir=path.parent,prefix=".pending-") as temporary:
                    job=Path(temporary)/"job";job.mkdir();joblib.dump(bundle,job/"model.joblib",compress=1)
                    if not np.array_equal(value,predict_ebm(joblib.load(job/"model.joblib"),frame.iloc[validation])):raise ValueError("EBMinnerreloadmismatch")
                    primary.write_json(job/"scores.json",record)
                    primary.write_json(job/"complete.json",dict(identity=identity,reload_exact=True,sha256={n:primary.sha(job/n) for n in ["model.joblib","scores.json"]}))
                    os.rename(job,path)
                new_inner+=1;log("inner_complete",recipe=name,outer=q,ap=record["ap"],seconds=record["seconds"],inner_done=new_inner+reused_inner)
            records[name][q]=record
    selection=select_inner(records,recipes);primary.write_json(args.out/"inner-selection.json",selection);log("inner_selection_frozen",selection=selection)
    created=set()
    def fixed_probability(name,q,seed):
        nonlocal new_models,reused_models
        fitting=np.arange(len(train)) if q=="full" else np.flatnonzero(train.forecast_period.lt(q))
        validation=np.arange(len(train),len(frame)) if q=="full" else np.flatnonzero(train.forecast_period.eq(q))
        identity=dict(signature=signature,stage="full" if q=="full" else "outer",recipe=name,period=q,seed=seed,fit_rows=len(fitting),prediction_rows=len(validation))
        path=args.out/"models"/f"{name}__{q}__s{seed}"
        if path.exists():
            if not args.resume and path not in created:raise ValueError("Existing EBMmodelrequiresresume")
            validate_job(path,identity,["model.joblib","prediction.npy"]);reused_models+=1
            return primary.probabilities(np.load(path/"prediction.npy",allow_pickle=False),len(validation))
        log("fixed_start",recipe=name,period=q,seed=seed)
        bundle=fit_ebm(recipes[name],frame.iloc[fitting],train.target.iloc[fitting],cats,runtime=args.runtime,seed=seed,threads=args.threads)
        value=primary.probabilities(predict_ebm(bundle,frame.iloc[validation]),len(validation))
        with TemporaryDirectory(dir=path.parent,prefix=".pending-") as temporary:
            job=Path(temporary)/"job";job.mkdir();joblib.dump(bundle,job/"model.joblib",compress=1)
            if not np.array_equal(value,predict_ebm(joblib.load(job/"model.joblib"),frame.iloc[validation])):raise ValueError("EBMfixedreloadmismatch")
            np.save(job/"prediction.npy",value,allow_pickle=False)
            primary.write_json(job/"complete.json",dict(identity=identity,reload_exact=True,sha256={n:primary.sha(job/n) for n in ["model.joblib","prediction.npy"]}))
            os.rename(job,path)
        new_models+=1;created.add(path);log("fixed_complete",recipe=name,period=q,seed=seed);return value
    def old_forecast(seed):return np.concatenate([fixed_probability(selection["outer"][q]["recipe"],q,seed) for q in primary.QUARTERS])
    screen=old_forecast(42);selected=assess("screen",args.out,meta,anchor,context,gam,screen,args.bootstrap)
    confirmed=None
    if selected:confirmed=np.mean([screen,old_forecast(17),old_forecast(2026)],axis=0,dtype=np.float64)
    choice=assess("confirmed",args.out,meta,anchor,context,gam,confirmed,args.bootstrap)
    guard=domination(meta,confirmed,anchor,gam);files=[]
    if choice and not guard["full_stop"]:
        final=selection["final"];value=np.mean([fixed_probability(final["recipe"],"full",seed) for seed in SEEDS],axis=0,dtype=np.float64)
        variant="blend75" if choice["model"]=="ebm_blend75" else "raw"
        if variant=="blend75":value=.75*anchor_test+.25*value
        sub=primary.validate_submission(test,sample,value,len(test) if args.smoke else 2246)
        path=args.out/f"submission_ebm_canonical_{variant}.csv";primary.write_csv(path,sub)
        pd.testing.assert_frame_equal(sub,primary.read_csv(path),check_exact=True)
        files=[dict(csv=path.name,sha256=primary.sha(path),variant=variant,final_recipe=final,rows=len(sub),old_mean_ap=choice["mean_ap"],old_delta_previous07=choice["delta_vs_reference"])]
    reports=[p for p in args.out.iterdir() if p.suffix in {".json",".csv"} and p!=manifest_path]
    inner_total=len(list((args.out/"inner").glob("*/complete.json")));model_total=len(list((args.out/"models").glob("*/complete.json")))
    result=dict(complete=True,signature=signature,smoke=args.smoke,inner_fits_total=inner_total,outer_full_fits_total=model_total,
        actual_model_fits_total=inner_total+model_total,new_inner_fits_this_invocation=new_inner,reused_inner_jobs=reused_inner,
        new_outer_full_fits_this_invocation=new_models,reused_model_requests=reused_models,screen_eligible=selected is not None,
        confirmed_eligible=choice is not None,full_stopped_by_GAM_domination=guard["full_stop"],files=files,final_recipe=selection["final"],
        new_q2_q3_metrics=False,new_q2_q3_external_validation_metrics=False,uploads=0,elapsed_seconds=time.monotonic()-started,interpretation=plan["interpretation"],output_sha256={p.name:primary.sha(p) for p in reports})
    primary.write_json(manifest_path,result);log("ebm_complete",files=files,model_fits=inner_total+model_total);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ["raw","out","feature-cache","baseline-cv","context-test","additive-oof","additive-test","gam-oof","runtime"]:p.add_argument("--"+flag,type=Path,required=True)
    p.add_argument("--threads",type=int,default=2);p.add_argument("--bootstrap",type=int,default=2000)
    p.add_argument("--resume",action="store_true");p.add_argument("--smoke",action="store_true");a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/".lock").open("a") as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(a)


if __name__=="__main__":main()
