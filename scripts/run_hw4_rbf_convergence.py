"""Exact eight RBF-SVC recipes with a repaired numerical convergence cap.

Fit-only log/median/RobustScaler/missing masks/OHE define the geometry.
Internal Platt calibration sees only the caller-supplied past fit rows.
All24 inner fits precede any outer forecast; finite max_iter is enforced.
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
from hw4_inspections.rbf_convergence import RBF_RECIPES, fit_rbf, predict_rbf, InvalidConvergenceError

SEEDS = (42, 17, 2026)
REFERENCE = "previous07"
OOF07_SHA = "58755d1e12123d1dd18553ac45b5075b3b503cf3e1e5b5d4dd97ce526fc5c9a0"
TEST07_SHA = "ef638b33e25e391f943c8a3758a0ff27618b23467b01ae0a5d227aa5d9b4d0f0"
DOMINATION_GAP = .002


def select_inner(records, recipes):
    """Numerical admissibility preserves every quarter; AP never drops cells."""
    if set(records) != set(recipes) or any(set(table) != set(primary.QUARTERS) for table in records.values()):
        raise ValueError("All eight recipes need a valid or explicit budget-invalid record for all three INNER quarters")
    valid = lambda name, q: records[name][q].get("valid", True)
    outer = {}
    for q in primary.QUARTERS:
        allowed = [name for name in recipes if valid(name, q)]
        if not allowed:
            raise ValueError(f"All recipes budget-invalid on INNER for {q}; no quarter may be discarded")
        def key(name):
            score = records[name][q]
            return (-score["ap"], score["logloss"], recipes[name]["C"], recipes[name]["gamma_multiplier"], name)
        name = min(allowed, key=key)
        outer[q] = dict(recipe=name, inner_quarter=records[name][q]["inner_quarter"],
                        inner_ap=records[name][q]["ap"])
    allowed = [name for name in recipes if all(valid(name, q) for q in primary.QUARTERS)]
    if not allowed:
        raise ValueError("No recipe is numerically valid on all three INNER folds; stop before OUTER/full selection")
    def mean_key(name):
        return (-float(np.mean([records[name][q]["ap"] for q in primary.QUARTERS])),
                float(np.mean([records[name][q]["logloss"] for q in primary.QUARTERS])),
                recipes[name]["C"], recipes[name]["gamma_multiplier"], name)
    name = min(allowed, key=mean_key)
    return dict(outer=outer, final=dict(recipe=name,
                mean_old_inner_ap=float(np.mean([records[name][q]["ap"] for q in primary.QUARTERS]))),
                fully_valid_full_candidates=allowed)


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
        vectors.update(svm_raw=raw, svm_blend75=.75 * anchor + .25 * raw)
    r = primary.evaluate_candidates(meta, vectors, reference=REFERENCE, bootstrap=bootstrap, seed=20261122)
    summary = r["summary"].copy()
    summary["eligible"] &= summary.worst_quarter_delta.ge(-.005) & summary.model.isin(["svm_raw", "svm_blend75"])
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


def runtime_evidence():
    dist = metadata.distribution("scikit-learn")
    native = {}
    for relative in dist.files or []:
        name = str(relative)
        if ("svm/_libsvm" in name and name.endswith((".so", ".dylib", ".dll"))) or name.endswith(("svm/_base.py", "svm/_classes.py")):
            native[name] = primary.sha(dist.locate_file(relative))
    if not native:
        raise ValueError("Installed native LIBSVM/source evidence is missing")
    return dict(version=dist.version, native_and_source_sha256=native,
                calibration="Built-in five-fold Platt calibration only on supplied past FIT rows; no external eval rows")


def run(args):
    if args.threads not in (1, 2) or args.bootstrap < 0 or (not args.smoke and not args.bootstrap):
        raise ValueError("Use one/twoCPU threads; bootstrap0 is synthetic-smoke only")
    started = time.monotonic()
    def log(event, **details):
        print(json.dumps(dict(event=event, elapsed_seconds=round(time.monotonic() - started, 3), **details)), flush=True)
    runtime = runtime_evidence()
    recipes = deepcopy(RBF_RECIPES)
    if len(recipes) != 8:
        raise ValueError("The frozen RBF pool has exactly eight recipes")
    plan = primary.provenance(args, recipes)
    plan["source_sha256"].update({"ebm_search.py": primary.sha(ROOT / "src/hw4_inspections/ebm_search.py"),
                                  "rbf_search.py": primary.sha(ROOT / "src/hw4_inspections/rbf_search.py"),
                                  "rbf_convergence.py": primary.sha(ROOT / "src/hw4_inspections/rbf_convergence.py"),
                                  Path(__file__).name: primary.sha(Path(__file__))})
    extras = dict(previous07_oof=primary.sha(args.additive_oof), previous07_test=primary.sha(args.additive_test),
                  new_gam_oof=primary.sha(args.gam_oof))
    if not args.smoke and (extras["previous07_oof"] != OOF07_SHA or extras["previous07_test"] != TEST07_SHA):
        raise ValueError("Verified previous07 anchors changed")
    invalid_receipt=args.out.parent/"rbf-invalid-before-outer.json"
    receipt_sha=None if args.smoke else primary.sha(invalid_receipt)
    if not args.smoke:
        receipt=primary.read_json(invalid_receipt)
        if (receipt_sha!="5b48c3b2fde87c5492335a3caa1566bacae2178803f1f1d6565a6804eef2efea"
                or receipt["outer_models"]!=0 or receipt["outer_metrics"]!=0
                or receipt["completed_inner_models"]!=8 or receipt["failed_inner_fit_attempts"]!=1):
            raise ValueError("Initial numerical-invalid receipt changed")
    extras["initial_invalid_receipt"]=receipt_sha
    plan.update(
        experiment="Exact eight RBF-SVC recipes with a pre-outer numerical cap amendment", recipes=recipes,
        pre_outer_amendment=dict(reason="Initial100000cap invalidated ninth INNER before any OUTER existed. Saved C5 final-margin iterations78259/98566 approached cap. Increase only numerical maximum to1000000; convergence tolerance/model/grid/transforms/calibration unchanged. No partial models or failed folds dropped. Refit all24 INNER from scratch under new source/signature.",
            initial_failed_run="analysis/hw4-six-hour-20261122/rbf/", initial_invalid_report_sha256="5b48c3b2fde87c5492335a3caa1566bacae2178803f1f1d6565a6804eef2efea", completed_original_models=8, failed_original_fit_attempts=1, original_outer_metrics=0),
        official_runtime=runtime, additional_anchors=extras, new_gam_column="algorithm_raw",
        preprocessing="Fit-only count log1p (signed only own change), median (all-missing0), RobustScaler numeric, raw missing flags for all numeric, dense OHE nominal quarter/categories with unknown ignore. No clipping. own22 versus compact94; year/gap excluded.",
        gamma="gamma_multiplier in {.25,1} times 1/(D*Var(Z_FIT)) after complete FIT-only transform, equivalent baseline to sklearn scale. Constant C .5 or5, no sample normalization.",
        native_behavior_note="kernel RBF, probability=True, tol1e-3, max_iter1000000, shrinking=True, cache512MB, class_weight=None. Any convergence warning or nonzero fit_status makes the run invalid; no partial model accepted.",
        internal_validation_policy="Native five-fold Platt calibration ONLY inside caller-supplied past FIT rows. This is random calibration, not temporal validation; external inner/outer rows and labels are never supplied. Product is class1 probability only; decision scores are never an alternative selected after scoring.",
        full_training_scope="If admitted, full fit includes all available train labels (Q2/Q3 are past FIT data). Internal calibration is part the fixed algorithm, not a separate recent-period diagnostic.",
        primary_sources=["https://scikit-learn.org/1.7/modules/generated/sklearn.svm.SVC.html",
            "https://scikit-learn.org/1.7/modules/svm.html", "https://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html",
            "https://github.com/scikit-learn/scikit-learn/blob/1.7.2/sklearn/svm/src/libsvm/svm.cpp"],
        native_status_scope="Upstream1.7.2 propagates the same solver status through five calibration subfits and final margin; any timeout invalidates the fit. Saved n_iter_ is final-margin only, not calibration total.",
        selection="All24 INNER fits before any OUTER. Each old outer selects recipe/view/C/gamma multiplier solely by strict previous INNER AP. Ties: lower logloss, lower C, lower multiplier, lexical ID.",
        tree_selection="No trees/iteration tuning. Fixed native max_iter1000000 convergence cap; failure is invalid, never a prefix candidate.",
        final="Largest equal-weight mean of three OLD INNER APs, then lower mean logloss/C/multiplier/ID. Entire recipe fixed, no new Q2/Q3 external tuning.",
        scores="Raw and fixed .75*previous07+.25*RBF class1 probabilities. Previous07 primary; context04/newGAM diagnostics. Equal probability mean of all42/17/2026 seeds, which vary calibration folds rather than margin-fit initialization.",
        admission="mean delta07>0, at least2/3 nonnegative quarters, worst>=-.005. Pass42 screen then six fixed-recipe17/2026 confirmation fits and same gate.",
        full_guard="After confirmation, if BOTH raw AND fixed07 blend lose ALL3 quarters to newGAM and EACH mean gap>=.002, skip full. Declared before any outer results.",
        numerical_admissibility="At cap1000000 each failed native solve gets an explicit budget-invalid record, no scores or partial model. Each INNER chooses only its numerically valid recipes, preserving all quarters; full aggregate considers only recipes valid on all3. All invalid on any INNER or no fully valid recipe stops before OUTER. Any later OUTER/full failure is recorded and stops, never replaced by another scored recipe. No further cap increase.",
        model_fit_budget=dict(inner=24, outer42=3, confirmation=6, full=3),
        native_calibration_folds_per_estimator=5,
        accounting="Fit counts refer to estimator.fit calls, as for bagged RBF. Each probability=True SVC additionally invokes built-in five-fold calibration; these native subfits are not hidden extra search recipes.",
        interpretation="Adaptive source-grounded final family on heavily reused old development; nominal site intervals, no independent holdout claim.",
        new_q2_q3_metrics=False, new_q2_q3_external_validation_metrics=False, uploads=0)
    plan = json.loads(json.dumps(plan, sort_keys=True, allow_nan=False))
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest();plan["signature"] = signature
    freeze = args.out / "freeze.json"
    if freeze.exists():
        if not args.resume or primary.read_json(freeze) != plan:
            raise ValueError("RBF source/runtime/data/anchors/recipes changed; resume refused")
    else:
        if any(p.name != ".lock" for p in args.out.iterdir()):
            raise ValueError("New RBF run requires an empty output folder")
        primary.write_json(freeze, plan)
    manifest_path = args.out / "forecast-manifest.json"
    if manifest_path.exists():
        old = primary.read_json(manifest_path)
        if old["signature"] != signature or any(primary.sha(args.out / p) != h for p,h in old["output_sha256"].items()):
            raise ValueError("Changed completed RBF output")
    train,test,sample,views,cats = primary.validate_cache(joblib.load(args.feature_cache),args);frame=views["base"]
    meta=pd.concat([train.loc[train.forecast_period.eq(q),list(primary.REQUIRED)] for q in primary.QUARTERS],ignore_index=True)
    context,_=primary.align_anchors(meta,test,args)
    def aligned(path):
        data=primary.read_csv(path)
        if not data.row_id.is_unique or set(data.row_id)!=set(meta.row_id):raise ValueError("RBF benchmark IDs differ")
        data=data.set_index("row_id").loc[meta.row_id].reset_index()
        pd.testing.assert_frame_equal(meta,data[list(primary.REQUIRED)],check_dtype=False)
        return data
    previous=aligned(args.additive_oof);anchor=primary.probabilities(previous.additive_logit__blend75,len(meta))
    gamframe=aligned(args.gam_oof);gam=primary.probabilities(gamframe.algorithm_raw,len(meta))
    test_anchor=primary.read_csv(args.additive_test)
    anchor_test=primary.validate_submission(test_anchor,test,test_anchor.target,len(test)).target.to_numpy(dtype=np.float64)
    args.out.joinpath("inner").mkdir(exist_ok=True);args.out.joinpath("models").mkdir(exist_ok=True)
    records={name:{} for name in recipes};new_inner=reused_inner=new_models=reused_models=0
    def validate_job(path,identity,files,*,invalid=False):
        r=primary.read_json(path/"complete.json")
        if (r["identity"]!=identity or r["reload_exact"] is not (not invalid)
                or r.get("status","valid") != ("budget_invalid" if invalid else "valid")
                or set(r["sha256"])!=set(files)
                or any(primary.sha(path/n)!=h for n,h in r["sha256"].items())):
            raise ValueError("Changed RBF job")
        return r
    def persist_invalid(path,identity,error,*,record=None):
        with TemporaryDirectory(dir=path.parent,prefix=".pending-") as temporary:
            job=Path(temporary)/"job";job.mkdir()
            name="scores.json" if record is not None else "error.json"
            primary.write_json(job/name,record if record is not None else dict(status="budget_invalid",reason=str(error),max_iter=1000000))
            primary.write_json(job/"complete.json",dict(identity=identity,status="budget_invalid",reload_exact=False,
                sha256={name:primary.sha(job/name)}))
            os.rename(job,path)
    for name,recipe in recipes.items():
        for q in primary.QUARTERS:
            inner_q=(pd.Period(q,freq="Q")-1).start_time.strftime("%Y-%m-%d")
            fitting=np.flatnonzero(train.forecast_period.lt(inner_q));validation=np.flatnonzero(train.forecast_period.eq(inner_q))
            identity=dict(signature=signature,stage="inner",recipe=name,outer=q,inner=inner_q,seed=42,fit_rows=len(fitting),validation_rows=len(validation))
            path=args.out/"inner"/f"{name}__{q}"
            if path.exists():
                if not args.resume:raise ValueError("Existing RBF path requires --resume")
                record=primary.read_json(path/"scores.json");invalid=not record.get("valid",True)
                validate_job(path,identity,["scores.json"] if invalid else ["model.joblib","scores.json"],invalid=invalid)
                reused_inner+=1;log("inner_reused",recipe=name,outer=q,valid=not invalid)
            else:
                log("inner_start",recipe=name,outer=q,rows=len(fitting));t0=time.monotonic()
                try:
                    bundle=fit_rbf(recipe,frame.iloc[fitting],train.target.iloc[fitting],cats,seed=42,threads=args.threads)
                except InvalidConvergenceError as error:
                    record=dict(valid=False,status="budget_invalid",inner_quarter=inner_q,ap=None,logloss=None,
                        reason=str(error),max_iter=1000000,seconds=time.monotonic()-t0)
                    persist_invalid(path,identity,error,record=record)
                else:
                    value=primary.probabilities(predict_rbf(bundle,frame.iloc[validation]),len(validation))
                    record=dict(valid=True,inner_quarter=inner_q,ap=float(average_precision_score(train.target.iloc[validation],value)),
                        logloss=float(log_loss(train.target.iloc[validation],value,labels=[0,1])),seconds=time.monotonic()-t0)
                    with TemporaryDirectory(dir=path.parent,prefix=".pending-") as temporary:
                        job=Path(temporary)/"job";job.mkdir();joblib.dump(bundle,job/"model.joblib",compress=1)
                        if not np.array_equal(value,predict_rbf(joblib.load(job/"model.joblib"),frame.iloc[validation])):raise ValueError("RBF inner reload mismatch")
                        primary.write_json(job/"scores.json",record)
                        primary.write_json(job/"complete.json",dict(identity=identity,status="valid",reload_exact=True,sha256={n:primary.sha(job/n) for n in ["model.joblib","scores.json"]}))
                        os.rename(job,path)
                new_inner+=1;log("inner_complete",recipe=name,outer=q,ap=record["ap"],valid=record["valid"],seconds=record["seconds"],inner_done=new_inner+reused_inner)
            records[name][q]=record
    try:
        selection=select_inner(records,recipes)
    except ValueError as error:
        primary.write_json(args.out/"numerical-selection-invalid.json",dict(reason=str(error),records=records,outer_models=0))
        raise
    primary.write_json(args.out/"inner-selection.json",selection);log("inner_selection_frozen",selection=selection)
    created=set()
    def fixed_probability(name,q,seed):
        nonlocal new_models,reused_models
        fitting=np.arange(len(train)) if q=="full" else np.flatnonzero(train.forecast_period.lt(q))
        validation=np.arange(len(train),len(frame)) if q=="full" else np.flatnonzero(train.forecast_period.eq(q))
        identity=dict(signature=signature,stage="full" if q=="full" else "outer",recipe=name,period=q,seed=seed,fit_rows=len(fitting),prediction_rows=len(validation))
        path=args.out/"models"/f"{name}__{q}__s{seed}"
        if path.exists():
            if not args.resume and path not in created:raise ValueError("Existing RBFmodelrequiresresume")
            if primary.read_json(path/"complete.json").get("status")=="budget_invalid":
                validate_job(path,identity,["error.json"],invalid=True)
                raise InvalidConvergenceError("Previously recorded OUTER/full numerical failure; no replacement recipe")
            validate_job(path,identity,["model.joblib","prediction.npy"]);reused_models+=1
            return primary.probabilities(np.load(path/"prediction.npy",allow_pickle=False),len(validation))
        log("fixed_start",recipe=name,period=q,seed=seed)
        try:
            bundle=fit_rbf(recipes[name],frame.iloc[fitting],train.target.iloc[fitting],cats,seed=seed,threads=args.threads)
        except InvalidConvergenceError as error:
            persist_invalid(path,identity,error)
            raise
        value=primary.probabilities(predict_rbf(bundle,frame.iloc[validation]),len(validation))
        with TemporaryDirectory(dir=path.parent,prefix=".pending-") as temporary:
            job=Path(temporary)/"job";job.mkdir();joblib.dump(bundle,job/"model.joblib",compress=1)
            if not np.array_equal(value,predict_rbf(joblib.load(job/"model.joblib"),frame.iloc[validation])):raise ValueError("RBFfixedreloadmismatch")
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
        variant="blend75" if choice["model"]=="svm_blend75" else "raw"
        if variant=="blend75":value=.75*anchor_test+.25*value
        sub=primary.validate_submission(test,sample,value,len(test) if args.smoke else 2246)
        path=args.out/f"submission_rbf_converged_{variant}.csv";primary.write_csv(path,sub)
        pd.testing.assert_frame_equal(sub,primary.read_csv(path),check_exact=True)
        files=[dict(csv=path.name,sha256=primary.sha(path),variant=variant,final_recipe=final,rows=len(sub),old_mean_ap=choice["mean_ap"],old_delta_previous07=choice["delta_vs_reference"])]
    reports=[p for p in args.out.iterdir() if p.suffix in {".json",".csv"} and p!=manifest_path]
    inner_total=len(list((args.out/"inner").glob("*/complete.json")));model_total=len(list((args.out/"models").glob("*/complete.json")))
    result=dict(complete=True,signature=signature,smoke=args.smoke,inner_fits_total=inner_total,outer_full_fits_total=model_total,
        actual_model_fits_total=inner_total+model_total,actual_estimator_fit_attempts=inner_total+model_total,
        budget_invalid_inner_jobs=sum(not record.get("valid",True) for table in records.values() for record in table.values()),
        completed_valid_estimator_fits=sum(record.get("valid",True) for table in records.values() for record in table.values())+model_total,
        new_inner_fits_this_invocation=new_inner,reused_inner_jobs=reused_inner,
        new_outer_full_fits_this_invocation=new_models,reused_model_requests=reused_models,screen_eligible=selected is not None,
        confirmed_eligible=choice is not None,full_stopped_by_GAM_domination=guard["full_stop"],files=files,final_recipe=selection["final"],
        new_q2_q3_metrics=False,new_q2_q3_external_validation_metrics=False,uploads=0,elapsed_seconds=time.monotonic()-started,interpretation=plan["interpretation"],output_sha256={p.name:primary.sha(p) for p in reports})
    primary.write_json(manifest_path,result);log("svm_complete",files=files,model_fits=inner_total+model_total);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ["raw","out","feature-cache","baseline-cv","context-test","additive-oof","additive-test","gam-oof"]:p.add_argument("--"+flag,type=Path,required=True)
    p.add_argument("--threads",type=int,default=2);p.add_argument("--bootstrap",type=int,default=2000)
    p.add_argument("--resume",action="store_true");p.add_argument("--smoke",action="store_true");a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/".lock").open("a") as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(a)


if __name__=="__main__":main()
