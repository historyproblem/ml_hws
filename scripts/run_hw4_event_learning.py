"""Train chronological, other-site event risk heads and evaluate their downstream value."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import joblib
import numpy as np
import pandas as pd

import run_hw4_final_research as primary
from hw4_inspections import event_learning as events

BASE = "matched_context_seed42"
AUG = "context_plus_event_risks"
DIRECT = "event_head_mean"
CB = {"family":"catboost", "params":{"depth":5, "learning_rate":.04, "l2_leaf_reg":10.,
      "boosting_type":"Plain", "bootstrap_type":"MVS", "subsample":.8, "rsm":1., "border_count":254},
      "max_iterations":500}


def signature(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def freeze(args):
    plan = primary.provenance(args, {})
    for key in ["tree_selection","selection","scores","pool","seeds"]:
        plan.pop(key)
    plan["source_sha256"].update({"event_learning.py":primary.sha(primary.ROOT/"src/hw4_inspections/event_learning.py"),
                                 Path(__file__).name:primary.sha(Path(__file__))})
    plan.update({
        "event_recipe":{"landmark":"Monday in calendar quarter start's week minus 7 days",
          "inference":"Quarter-specific model; historical observation pool exclusively from the same forecast snapshot with week before row cutoff.",
          "labels":"Fail of first unambiguous routine in a completed historical quarter. Both its entire calendar quarter and preceding 365-day event feature window must fit other-site coverage. Exclude earliest conflicting/unknown or boundary-week record; never select a later substitute.",
          "site_protection":"SHA256(UTF8 site_id), first eight bytes unsigned big-endian, modulo 5. Every assessment site's fold omitted from event model training including its own earlier/later events. Fold-specific coverage envelope.",
          "feature_availability":"Events only in [landmark-365d,landmark). Static attributes only from a past observed visit. No label fields or future gap from endpoint enter model features. Missing past is explicit.",
          "heads":[dict(events.LOGIT_PARAMS),dict(events.CB_PARAMS)], "max_cpu_threads":2,
          "event_model_versions":list(events.HEADS), "official_targets_used_in_head":False,
          "smoke_event_cb_iterations":4 if args.smoke else None},
        "second_stage":{"control":"Current base346 CB at seed42 with same settings, 2 threads and rows.",
          "augmented":"Append exactly two OOB event probabilities to base346. Fixed incumbent fit settings, no C or tree selection.",
          "screen":"1 seed42 augmented and 1 matched seed42 control per old quarter. Also each head, mean heads and frozen probability blend.",
          "confirmation":"Only if raw augmented or fixed blend is eligible vs incumbent, fit 17 and 2026 as well, average 42/17/2026. Direct heads are deterministic/fixed; no additional seed voting.",
          "eligible_variants":["event_head_mean","event_head_mean__blend75",AUG,AUG+"__blend75"],
          "reference":"Exact provided incumbent mean of seeds42/17/2026; matched42 as separate diagnostic.",
          "blend":"75% context04 probability + 25% new candidate probability.",
          "output":"At most one candidate across eligible direct and eligible confirmed augmented exports. Highest old mean AP; lexical name tie. Full augmented fits only if its export is chosen.",
          "cb_params":CB, "seeds":list(primary.SEEDS), "smoke_second_stage_iterations":3 if args.smoke else None},
        "independence":"Exploratory comparisons on already consulted old quarters; nominal paired site bootstrap. No new Q2/Q3 metric or external inputs."})
    sig = signature(plan)
    plan["signature"] = sig
    path = args.out/"freeze.json"
    if path.exists():
        if not args.resume or primary.read_json(path) != plan:
            raise ValueError("Cannot resume with changed source, recipe, inputs, or software")
    else:
        if any(p.name != ".lock" for p in args.out.iterdir()):
            raise ValueError("New run requires empty output directory")
        primary.write_json(path,plan)
    return sig


def single_bucket(path, identity, x, y, query, mask, positions, site_ids, args):
    if path.exists():
        if not args.resume:
            raise ValueError("Existing event heads require --resume")
        rec = primary.read_json(path/"complete.json")
        if (rec["identity"] != identity or not rec["loaded_forward_exact"] or
            set(rec["sha256"]) != {"models.joblib","probabilities.npy","query-ids.json"} or
            any(primary.sha(path/n)!=v for n,v in rec["sha256"].items())):
            raise ValueError("Event bucket identity or artifact mismatch")
        if primary.read_json(path/"query-ids.json") != site_ids:
            raise ValueError("Event bucket changed query order")
        values=np.load(path/"probabilities.npy",allow_pickle=False)
        if values.dtype != np.float64 or values.shape != (len(positions),2) or not np.isfinite(values).all():
            raise ValueError("Changed first-stage risk array")
        return values
    bundle=events.fit_heads(x.loc[mask],y.loc[mask,"y"],args.threads,args.smoke)
    values=events.predict_heads(bundle,query.iloc[positions])
    with TemporaryDirectory(dir=path.parent,prefix=".partial-") as work:
        stage=Path(work)/"bucket"
        stage.mkdir()
        joblib.dump(bundle,stage/"models.joblib",compress=1)
        replay=events.predict_heads(joblib.load(stage/"models.joblib"),query.iloc[positions])
        if not np.array_equal(values,replay):
            raise AssertionError("Event heads failed saved forward replay")
        np.save(stage/"probabilities.npy",values,allow_pickle=False)
        primary.write_json(stage/"query-ids.json",site_ids)
        primary.write_json(stage/"complete.json",{"identity":identity,"loaded_forward_exact":True,
              "sha256":{n:primary.sha(stage/n) for n in ["models.joblib","probabilities.npy","query-ids.json"]}})
        os.rename(stage,path)
    return values


def score_snapshot(args, sig, history, rows):
    # Select only identifiers/dates from official query tables, even if called on train.
    rows=rows[["row_id","site_id","forecast_period","history_before"]].reset_index(drop=True)
    if rows.forecast_period.nunique()!=1 or rows.history_before.nunique()!=1 or not rows.site_id.is_unique:
        raise ValueError("Each score block must have one landmark and unique query sites")
    period, cutoff=rows.loc[0,["forecast_period","history_before"]]
    hist=events.snapshot(history,period,cutoff)
    x,y,query=events.landmark_examples(hist,period,cutoff,rows.site_id)
    buckets=rows.site_id.map(events.site_fold).to_numpy()
    scores=np.full((len(rows),2), np.nan)
    directory=args.out/"heads"/f"{period}__{cutoff}"
    directory.mkdir(parents=True,exist_ok=True)
    begin=time.monotonic()
    details=[]
    for bucket in range(events.FOLDS):
        idx=np.flatnonzero(buckets==bucket)
        if not len(idx):
            continue
        mask=events.training_mask(hist,y,bucket)
        train_sites=y.loc[mask,"site_id"]
        if not mask.any() or set(train_sites).intersection(rows.site_id.iloc[idx]):
            raise ValueError("Event training overlaps focal sites, or lacks history")
        latest=y.loc[mask,"label_week"].max()
        if latest + events.WEEK > pd.Timestamp(cutoff):
            raise ValueError("First-level outcome not closed before forecast cutoff")
        identity={"signature":sig,"forecast_period":period,"history_before":cutoff,"bucket":bucket,
                  "past_event_rows":int(mask.sum()),"unique_training_sites":train_sites.nunique(),
                  "old_endpoint_weeks_before":latest.strftime("%Y-%m-%d"),"query_ids_digest":signature(rows.row_id.iloc[idx].tolist()),
                  "query_rows":len(idx),"sources":sorted(y.loc[mask,"quarter_start"].dt.strftime("%Y-%m-%d").unique().tolist())}
        scores[idx]=single_bucket(directory/f"bucket-{bucket}",identity,x,y,query,mask,idx,rows.site_id.iloc[idx].tolist(),args)
        details.append(identity)
    if not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():
        raise AssertionError("Missing event forecasts")
    audit={"period":period,"cutoff":cutoff,"rows":len(rows),"event_example_rows":len(y),
        "unique_event_sites":y.site_id.nunique(),"event_features":len(x.columns),"feature_names":list(x.columns),
        "query_without_past_visits_365d":int(query.visit_n.eq(0).sum()),
        "query_without_past_routine_365d":int(query.routine_n.eq(0).sum()),
        "own_site_target_used":False,"first_level_heldout_official_targets_read":False,
        "covered_past_quarters":sorted(y.quarter_start.dt.strftime("%Y-%m-%d").unique().tolist()),"buckets":details}
    primary.write_json(directory/"audit.json",audit)
    output=pd.DataFrame(scores,columns=events.HEADS)
    output.insert(0,"row_id",rows.row_id)
    primary.write_csv(directory/"event-scores.csv",output)
    print(json.dumps({"event":"heads_done","period":period,"historic_examples":len(y),"query_n":len(rows),
                      "seconds":round(time.monotonic()-begin,2)}),flush=True)
    return output


def score_rows(args,sig,history,rows):
    by_snapshot=[score_snapshot(args,sig,history,group) for _,group in rows.groupby(["forecast_period","history_before"],sort=True)]
    output=pd.concat(by_snapshot,ignore_index=True)
    if not rows.row_id.is_unique or not output.row_id.is_unique or set(output.row_id)!=set(rows.row_id):
        raise ValueError("Head outputs not one-to-one with official IDs")
    return output.set_index("row_id").loc[rows.row_id,list(events.HEADS)].reset_index(drop=True)


def evaluate(args, stage, meta, forecasts, eligible_names, reference=primary.ANCHOR):
    result=primary.evaluate_candidates(meta,forecasts,reference=reference,bootstrap=args.bootstrap,seed=20261122)
    summary=result["summary"].copy()
    summary["eligible"] &= summary.model.isin(eligible_names) & summary.worst_quarter_delta.ge(primary.WORST_FLOOR)
    primary.write_csv(args.out/f"{stage}-quarter-metrics.csv",result["quarter_scores"])
    primary.write_csv(args.out/f"{stage}-summary.csv",summary)
    primary.write_csv(args.out/f"{stage}-oof.csv",pd.concat([meta,pd.DataFrame(forecasts)],axis=1))
    return summary.loc[summary.eligible,"model"].tolist()


def run(args):
    if not 1<=args.threads<=2 or args.bootstrap<(0 if args.smoke else 1):
        raise ValueError("One/two CPU threads and positive bootstrap required")
    sig=freeze(args)
    manifest_path=args.out/"forecast-manifest.json"
    if manifest_path.exists():
        closed=primary.read_json(manifest_path)
        if closed["signature"]!=sig or any(primary.sha(args.out/n)!=v for n,v in closed["sha256"].items()):
            raise ValueError("Changed completed event run")
    train,test,sample,views,cats=primary.validate_cache(joblib.load(args.feature_cache),args)
    base=views["base"]
    history=pd.read_csv(args.raw/"inspection_history.csv",usecols=events.HISTORY)
    old_idx=np.flatnonzero(train.forecast_period.le(primary.QUARTERS[-1]))
    old=train.iloc[old_idx].reset_index(drop=True)
    old_scores=score_rows(args,sig,history,old)
    old_base=base.iloc[old_idx].reset_index(drop=True)
    old_aug=pd.concat([old_base,old_scores],axis=1)
    old_held=[np.flatnonzero(old.forecast_period.eq(q)) for q in primary.QUARTERS]
    order=np.concatenate(old_held)
    meta=old.iloc[order][list(primary.REQUIRED)].reset_index(drop=True)
    anchor, anchor_test=primary.align_anchors(meta,test,args)
    count=3 if args.smoke else 500
    spec={**CB,"max_iterations":count}
    def fixed(name, frame, target_rows, quarter, seed):
        case={"family":name,"view":name,"inner":False,"spec":spec}
        pred,actual=primary.run_job(args.out,sig,name,case,seed,quarter,count,target_rows,frame,cats,args.threads,args.resume)
        if actual!=count:
            raise ValueError("Unplanned second-level capacity")
        return pred
    def folds(name,frame,seed):
        return np.concatenate([fixed(name,frame,old,q,seed) for q in primary.QUARTERS])
    matched=folds(BASE,old_base,42)
    augmented=folds(AUG,old_aug,42)
    scored=old_scores.iloc[order].reset_index(drop=True)
    headmean=scored.mean(axis=1).to_numpy()
    screen={primary.ANCHOR:anchor,BASE:matched,**{k:scored[k].to_numpy() for k in events.HEADS},
            DIRECT:headmean,DIRECT+"__blend75":.75*anchor+.25*headmean,
            AUG:augmented,AUG+"__blend75":.75*anchor+.25*augmented}
    eligible={DIRECT,DIRECT+"__blend75",AUG,AUG+"__blend75"}
    screen_pass=evaluate(args,"screen",meta,screen,eligible)
    evaluate(args,"matched42",meta,{BASE:matched,AUG:augmented},set(),reference=BASE)
    confirmed={k:screen[k] for k in [primary.ANCHOR,DIRECT,DIRECT+"__blend75"]}
    if {AUG,AUG+"__blend75"}.intersection(screen_pass):
        preds=[augmented]+[folds(AUG,old_aug,seed) for seed in primary.SEEDS[1:]]
        average=np.mean(preds,axis=0,dtype=np.float64)
        confirmed.update({AUG:average,AUG+"__blend75":.75*anchor+.25*average})
    choices=evaluate(args,"confirmed",meta,confirmed,eligible)
    chosen=choices[0] if choices else None
    files=[]
    if chosen:
        late_rows=train.loc[train.forecast_period.gt(primary.QUARTERS[-1])]
        test_scores=score_rows(args,sig,history,test)
        if chosen.startswith(AUG):
            late_scores=score_rows(args,sig,history,late_rows)
            all_scores=pd.concat([old_scores.set_index(old.row_id),late_scores.set_index(late_rows.row_id),
                                  test_scores.set_index(test.row_id)]).loc[pd.concat([train.row_id,test.row_id])]
            all_aug=pd.concat([base,all_scores.reset_index(drop=True)],axis=1)
            prediction=np.mean([fixed(AUG,all_aug,train,"full",seed) for seed in primary.SEEDS],axis=0,dtype=np.float64)
        else:
            prediction=test_scores.mean(axis=1).to_numpy()
        if chosen.endswith("__blend75"):
            prediction=.75*anchor_test+.25*prediction
        sub=primary.validate_submission(test,sample,prediction,len(test) if args.smoke else 2246)
        file=args.out/f"submission_{chosen}.csv"
        primary.write_csv(file,sub)
        pd.testing.assert_frame_equal(sub,primary.read_csv(file),check_exact=True)
        files.append({"name":chosen,"file":file.name,"sha256":primary.sha(file),"rows":len(sub)})
    artifacts=[p for p in args.out.iterdir() if p.suffix in {".csv",".json"} and p!=manifest_path]
    summary={"signature":sig,"complete":True,"smoke":args.smoke,"screen_eligible":screen_pass,
             "confirmed_eligible":choices,"chosen":chosen,"files":files,"new_q2_q3_metrics":False,"uploads":0,
             "first_level_snapshots":len(list((args.out/"heads").glob("*/audit.json"))),
             "first_level_two_head_bundles":len(list((args.out/"heads").glob("*/*/complete.json"))),
             "second_level_jobs":len(list((args.out/"jobs").glob("*/complete.json"))),
             "sha256":{p.name:primary.sha(p) for p in artifacts}}
    primary.write_json(manifest_path,summary)
    print(json.dumps({"event":"complete","selected":chosen,"files":files}),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("raw","out","feature-cache","baseline-cv","context-test"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--bootstrap",type=int,default=2000)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args)


if __name__=="__main__":
    main()
