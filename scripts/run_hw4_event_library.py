"""Bounded longitudinal library and privileged future recorded-burden ablations."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import joblib
import numpy as np
import pandas as pd

import run_hw4_final_research as primary
import run_hw4_event_learning as window
from hw4_inspections import event_learning as event, event_library as library

RISK = "cb_library_two_risks"
LOAD = "cb_library_two_risks_and_logload"
MEAN = "library_risk_mean"


def v1_proof(args):
    directory=args.window_run
    freeze=primary.read_json(directory/"freeze.json")
    original={k:v for k,v in freeze.items() if k!="signature"}
    manifest=primary.read_json(directory/"forecast-manifest.json")
    if window.signature(original)!=freeze["signature"] or manifest["signature"]!=freeze["signature"] or not manifest["complete"]:
        raise ValueError("Invalid original event-window provenance")
    if any(primary.sha(directory/name)!=v for name,v in manifest["sha256"].items()):
        raise ValueError("Changed original event-window artifacts")
    for name in ["event_learning.py", "run_hw4_event_learning.py", "run_hw4_final_research.py"]:
        path=primary.ROOT/("scripts" if name.startswith("run_") else "src/hw4_inspections")/name
        if primary.sha(path)!=freeze["source_sha256"][name]:
            raise ValueError("Original event-window source has changed")
    return freeze,{"freeze_sha256":primary.sha(directory/"freeze.json"),
                   "manifest_sha256":primary.sha(directory/"forecast-manifest.json"),
                   "oof_sha256":primary.sha(directory/"screen-oof.csv")}


def plan_run(args):
    first,proof=v1_proof(args)
    plan=primary.provenance(args,{})
    for key in ["raw_sha256","anchors_sha256","cache_sha256","cache_metadata_sha256","smoke","threads","bootstrap","packages"]:
        if plan[key]!=first[key]:
            raise ValueError("Inputs, versions or old-fold settings differ from the matched event-window run")
    for key in ["selection","scores","tree_selection","pool","seeds"]:
        plan.pop(key)
    plan["source_sha256"].update({
        "event_learning.py":primary.sha(primary.ROOT/"src/hw4_inspections/event_learning.py"),
        "event_library.py":primary.sha(primary.ROOT/"src/hw4_inspections/event_library.py"),
        "run_hw4_event_learning.py":primary.sha(Path(window.__file__)),Path(__file__).name:primary.sha(Path(__file__))})
    plan.update({"scope":"Separate adaptive expansion; candidate library and continuous target proposed prior to reading original outer metrics, exact implementation specified after that run.",
        "previous_window_proof":proof,"event_recipe":first["event_recipe"],
        "library":"At each query collect only prior and current issuances with source forecast<=query forecast and source cutoff<=query cutoff. Retain earliest admissible release per (site,historical quarter), with full 365-day antecedent coverage from admissible sites, completed entire quarter, closed first observed endpoint before both cutoffs. Raw histories never combined.",
        "out_of_site":"Same five fixed SHA256 buckets; all historical contributions by every query bucket site excluded in all earlier issuances as well as current.",
        "heads":["Unchanged original L2 logit and CatBoost binary heads",dict(library.LOAD_PARAMS)],
        "auxiliary_target":"log1p(number of distinct 1..44 recorded codes at the eligible historical endpoint). Clip estimates to [0,log(45)]. Missing/blank field has zero recorded codes. Endpoint fields never join X.",
        "matched_second_stage":{"control":window.BASE,"original_window_2":window.AUG,"library_2":RISK,"library_3":LOAD,
                                "parameters":window.CB,"matched_seed":42,"confirm_seeds":[17,2026]},
        "comparisons":"Raw and fixed 75% context04 probability +25% candidate, separately for direct mean binary heads, base346+library2 and base346+library2+load. Individual heads and standalone count are not eligible. Confirm each qualifying CB case on the 3 fixed seeds. At most one final CSV, ranked by eligible meanAP then lexical name.",
        "registered_diagnostics":"Matched42 base vs both library variants; original-window348 vs library348; library348 vs library349; original vs expanded direct event binary mean.",
        "final":"Full three CB fits only for winning augmented recipe. Auxiliary burden is a feature, not a predicted class probability.",
        "new_q2_q3_metrics":False,"uploads":0,"source_cohort_availability":"Snapshot issuance is represented by its official period and as-of cutoff; actual system publication timestamps are unavailable."})
    sig=window.signature(plan)
    plan["signature"]=sig
    path=args.out/"freeze.json"
    if path.exists():
        if not args.resume or primary.read_json(path)!=plan:
            raise ValueError("Changed event-library freeze")
    else:
        if any(f.name!=".lock" for f in args.out.iterdir()):
            raise ValueError("New library run requires empty output directory")
        primary.write_json(path,plan)
    return sig


def score_bucket(args,path,identity,x,y,query,site_ids):
    if path.exists():
        if not args.resume:
            raise ValueError("Existing library heads require --resume")
        record=primary.read_json(path/"complete.json")
        if (record["identity"]!=identity or not record["reloaded_exact"] or
                set(record["sha256"])!={"heads.joblib","values.npy","site-order.json"} or
                any(primary.sha(path/n)!=v for n,v in record["sha256"].items()) or
                primary.read_json(path/"site-order.json")!=site_ids):
            raise ValueError("Library model or cached risk features changed")
        scores=np.load(path/"values.npy",allow_pickle=False)
    else:
        bundle=library.fit_library(x,y,args.threads,args.smoke)
        scores=library.predict_library(bundle,query)
        with TemporaryDirectory(dir=path.parent,prefix=".pending-") as tmp:
            d=Path(tmp)/"bucket";d.mkdir()
            joblib.dump(bundle,d/"heads.joblib",compress=1)
            if not np.array_equal(scores,library.predict_library(joblib.load(d/"heads.joblib"),query)):
                raise AssertionError("Library heads do not reproduce saved values")
            np.save(d/"values.npy",scores,allow_pickle=False)
            primary.write_json(d/"site-order.json",site_ids)
            hashes={p.name:primary.sha(p) for p in d.iterdir()}
            primary.write_json(d/"complete.json",{"identity":identity,"reloaded_exact":True,"sha256":hashes})
            os.rename(d,path)
    if scores.dtype!=np.float64 or scores.shape!=(len(query),3) or not np.isfinite(scores).all():
        raise ValueError("Invalid library scores")
    if (scores<0).any() or (scores[:,:2]>1).any() or (scores[:,2]>np.log1p(44)).any():
        raise ValueError("Risk/load scores outside their admissible ranges")
    return scores


class SequentialLibrary:
    def __init__(self,args,sig,history):
        self.args,self.sig,self.history=args,sig,history
        self.releases=[]

    def issue(self,rows,score=True):
        rows=rows[["row_id","site_id","forecast_period","history_before"]].reset_index(drop=True)
        if rows.forecast_period.nunique()!=1 or rows.history_before.nunique()!=1 or not rows.site_id.is_unique:
            raise ValueError("Mixed landmark or nonunique query sites")
        period,cutoff=rows.loc[0,["forecast_period","history_before"]]
        if self.releases and (self.releases[-1].period>=pd.Timestamp(period) or self.releases[-1].cutoff>=pd.Timestamp(cutoff)):
            raise ValueError("Cannot issue the archive out of order")
        hist=event.snapshot(self.history,period,cutoff)
        release,query=library.make_release(hist,period,cutoff,rows.site_id)
        self.releases.append(release)
        if not score:
            return None
        then=time.monotonic()
        qfold=rows.site_id.map(event.site_fold).to_numpy()
        result=np.full((len(rows),3),np.nan)
        stats=[]
        destination=self.args.out/"heads"/period
        destination.mkdir(parents=True,exist_ok=True)
        for heldout in range(event.FOLDS):
            pos=np.flatnonzero(qfold==heldout)
            if not len(pos):
                continue
            x,y,audit=library.asof_library(self.releases,period,cutoff,heldout)
            if set(y.site_id).intersection(rows.site_id.iloc[pos]):
                raise ValueError("Direct focal site in pooled history")
            audit["query_rows"]=len(pos)
            audit["query_ids_sha256"]=window.signature(rows.row_id.iloc[pos].tolist())
            identity={"freeze":self.sig,**audit}
            result[pos]=score_bucket(self.args,destination/f"bucket-{heldout}",identity,x,y,query.iloc[pos],rows.site_id.iloc[pos].tolist())
            stats.append(audit)
        output=pd.DataFrame(result,columns=library.ALL_HEADS)
        output.insert(0,"row_id",rows.row_id)
        primary.write_csv(destination/"scores.csv",output)
        primary.write_json(destination/"library-audit.json",{"period":period,"cutoff":cutoff,"buckets":stats})
        print(json.dumps({"event":"library_heads_ready","period":period,"rows":len(rows),"other_site_training_sizes":
            [s["unique_site_quarters"] for s in stats],"historical_quarters":stats[0]["covered_quarters"],
            "seconds":round(time.monotonic()-then,2)}),flush=True)
        return output

    def score_ordered(self,rows):
        parts=[self.issue(group) for _,group in rows.groupby(["forecast_period","history_before"],sort=True)]
        frame=pd.concat(parts,ignore_index=True)
        if not frame.row_id.is_unique or set(frame.row_id)!=set(rows.row_id):
            raise ValueError("Library scores lost official ID alignment")
        return frame.set_index("row_id").loc[rows.row_id,library.ALL_HEADS].reset_index(drop=True)


def read_matched(args,meta,anchor):
    old=primary.read_csv(args.window_run/"screen-oof.csv").set_index("row_id").loc[meta.row_id].reset_index()
    pd.testing.assert_frame_equal(meta,old[list(primary.REQUIRED)],check_dtype=False)
    if not np.array_equal(old[primary.ANCHOR],anchor):
        raise ValueError("Original matched scores and incumbent differ in row-aligned inputs")
    return old


def run(args):
    if not 1<=args.threads<=2 or args.bootstrap<(0 if args.smoke else 1):
        raise ValueError("One/two threads and positive bootstrap required")
    sig=plan_run(args)
    path=args.out/"forecast-manifest.json"
    if path.exists():
        prev=primary.read_json(path)
        if prev["signature"]!=sig or any(primary.sha(args.out/n)!=v for n,v in prev["sha256"].items()):
            raise ValueError("Changed completed library result")
    train,test,sample,views,cats=primary.validate_cache(joblib.load(args.feature_cache),args)
    base=views["base"]
    old_i=np.flatnonzero(train.forecast_period.le(primary.QUARTERS[-1]))
    old=train.iloc[old_i].reset_index(drop=True)
    archive=SequentialLibrary(args,sig,pd.read_csv(args.raw/"inspection_history.csv",usecols=event.HISTORY))
    heads=archive.score_ordered(old)
    old_base=base.iloc[old_i].reset_index(drop=True)
    feature_sets={RISK:list(event.HEADS),LOAD:library.ALL_HEADS}
    frames={n:pd.concat([old_base,heads[cols]],axis=1) for n,cols in feature_sets.items()}
    order=np.concatenate([np.flatnonzero(old.forecast_period.eq(q)) for q in primary.QUARTERS])
    meta=old.iloc[order][list(primary.REQUIRED)].reset_index(drop=True)
    anchor,anchor_test=primary.align_anchors(meta,test,args)
    source=read_matched(args,meta,anchor)
    count=3 if args.smoke else 500
    def fixed(name, frame, rows, q, seed):
        case={"family":name,"spec":{**window.CB,"max_iterations":count},"inner":False,"view":name}
        values,actual=primary.run_job(args.out,sig,name,case,seed,q,count,rows,frame,cats,args.threads,args.resume)
        if actual!=count:
            raise ValueError("Unexpected library CB capacity")
        return values
    def folds(name,seed):
        return np.concatenate([fixed(name,frames[name],old,q,seed) for q in primary.QUARTERS])
    headmean=heads.iloc[order][list(event.HEADS)].mean(axis=1).to_numpy()
    stage={primary.ANCHOR:anchor,MEAN:headmean,MEAN+"__blend75":.75*anchor+.25*headmean}
    singles={}
    eligible={MEAN,MEAN+"__blend75"}
    for name in frames:
        singles[name]=folds(name,42)
        stage.update({name:singles[name],name+"__blend75":.75*anchor+.25*singles[name]})
        eligible.update([name,name+"__blend75"])
    stage.update({n:heads.iloc[order][n].to_numpy() for n in event.HEADS})
    passed=window.evaluate(args,"screen",meta,stage,eligible)
    for prefix,refs,allvals in [
        ("matched-base42",window.BASE,{window.BASE:source[window.BASE].to_numpy(),**singles}),
        ("window-vs-library42",window.AUG,{window.AUG:source[window.AUG].to_numpy(),RISK:singles[RISK]}),
        ("marginal-recorded-load42",RISK,singles),
        ("direct-window-vs-library",window.DIRECT,{window.DIRECT:source[window.DIRECT].to_numpy(),MEAN:headmean})]:
        window.evaluate(args,prefix,meta,allvals,set(),reference=refs)
    confirmed={k:stage[k] for k in [primary.ANCHOR,MEAN,MEAN+"__blend75"]}
    for n in frames:
        if {n,n+"__blend75"}.intersection(passed):
            avg=np.mean([singles[n],*[folds(n,s) for s in primary.SEEDS[1:]]],axis=0,dtype=np.float64)
            confirmed.update({n:avg,n+"__blend75":.75*anchor+.25*avg})
    chosen_list=window.evaluate(args,"confirmed",meta,confirmed,eligible)
    winner=chosen_list[0] if chosen_list else None
    files=[]
    if winner:
        name=winner.removesuffix("__blend75")
        late=train.loc[train.forecast_period.gt(primary.QUARTERS[-1])]
        if name in frames:
            late_heads=archive.score_ordered(late)
        else:
            for _,group in late.groupby(["forecast_period","history_before"],sort=True):
                archive.issue(group,score=False)
        test_heads=archive.score_ordered(test)
        if name in frames:
            vals=pd.concat([heads.set_index(old.row_id),late_heads.set_index(late.row_id),test_heads.set_index(test.row_id)])
            vals=vals.loc[pd.concat([train.row_id,test.row_id]),feature_sets[name]].reset_index(drop=True)
            xfull=pd.concat([base,vals],axis=1)
            forecast=np.mean([fixed(name,xfull,train,"full",s) for s in primary.SEEDS],axis=0,dtype=np.float64)
        else:
            forecast=test_heads[list(event.HEADS)].mean(axis=1).to_numpy()
        if winner.endswith("__blend75"):
            forecast=.75*anchor_test+.25*forecast
        submission=primary.validate_submission(test,sample,forecast,len(test) if args.smoke else 2246)
        csv=args.out/f"submission_{winner}.csv"
        primary.write_csv(csv,submission)
        pd.testing.assert_frame_equal(submission,primary.read_csv(csv),check_exact=True)
        files.append({"model":winner,"file":csv.name,"sha256":primary.sha(csv),"rows":len(submission)})
    root_files=[f for f in args.out.iterdir() if f.suffix in {".json",".csv"} and f!=path]
    result={"complete":True,"signature":sig,"smoke":args.smoke,"screen_eligible":passed,
            "confirmed_eligible":chosen_list,"chosen":winner,"files":files,"new_q2_q3_metrics":False,"uploads":0,
            "issued_snapshots":len(archive.releases),"first_level_three_head_bundles":len(list((args.out/"heads").glob("*/*/complete.json"))),
            "second_stage_jobs":len(list((args.out/"jobs").glob("*/complete.json"))),
            "sha256":{p.name:primary.sha(p) for p in root_files}}
    primary.write_json(path,result)
    print(json.dumps({"event":"library_complete","chosen":winner,"files":files}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["out","raw","feature-cache","baseline-cv","context-test","window-run"]:
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--bootstrap",type=int,default=2000)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args)


if __name__=="__main__":
    main()
