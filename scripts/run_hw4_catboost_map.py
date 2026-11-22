"""One post-Public CatBoost native MAP experiment, causal OLD3 selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_hw4_final_research as trusted
from hw4_inspections.classical_features import select_feature_columns
from hw4_inspections.deep_boosting import FORBIDDEN_COLUMNS
from hw4_inspections.next_round import rank_scores

GRID = [100, 300, 500, 800]
PARAMETERS = dict(loss_function="YetiRank:mode=MAP", depth=5, learning_rate=.04,
    l2_leaf_reg=10., boosting_type="Plain", bootstrap_type="MVS", subsample=.8,
    rsm=1., border_count=254, one_hot_max_size=2, allow_writing_files=False,
    thread_count=2, random_seed=42, verbose=False)


def clean(frame, cats):
    x = frame.copy()
    for c in cats: x[c] = x[c].fillna("<none>").astype(str)
    numeric = [c for c in x if c not in cats]
    x[numeric] = x[numeric].astype(float).replace([np.inf, -np.inf], np.nan)
    return x


def fit_predict(frame, cats, train, fit, pred, count):
    order = np.argsort(train.iloc[fit].forecast_period.to_numpy(), kind="stable")
    fit = np.asarray(fit)[order]
    group = train.iloc[fit].forecast_period.astype(str).to_numpy()
    assert np.all(group[1:] >= group[:-1]) and not FORBIDDEN_COLUMNS.intersection(frame)
    fitting = Pool(frame.iloc[fit], label=train.iloc[fit].target, cat_features=cats, group_id=group)
    query = Pool(frame.iloc[pred], cat_features=cats)
    start = time.monotonic()
    with threadpool_limits(limits=2):
        model = CatBoostRanker(iterations=count, **PARAMETERS); model.fit(fitting)
        values = {n:np.asarray(model.predict(query, ntree_end=n), dtype=np.float64)
                  for n in GRID if n <= count}
        values[count] = np.asarray(model.predict(query), dtype=np.float64)
    assert all(np.isfinite(p).all() for p in values.values())
    return model, values, query, time.monotonic()-start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["raw", "feature-cache", "baseline-cv", "context-test", "out", "preflight"]:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--per-job-seconds",type=int,default=30)
    parser.add_argument("--total-seconds",type=int,default=900)
    args = parser.parse_args(); args.threads = 2; args.bootstrap = 0; args.smoke = False
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()): raise ValueError("New experiment requires an empty output directory")
    evidence = trusted.read_json(args.preflight)
    if (evidence.get("status") != "PASS" or evidence.get("bootstrap") != "MVS"
        or not evidence.get("reload_exact") or evidence.get("real_HW4_fits") != 0):
        raise ValueError("Require successful nativeMAP MVS synthetic preflight before real fitting")
    pool = {"full346":{"columns":346}, "compact94":{"columns":94}}
    plan = trusted.provenance(args, pool)
    plan.update(experiment="Post-Public native CatBoost YetiRank:mode=MAP", seeds=[42],
        source_sha256_new=trusted.sha(Path(__file__)), preflight_sha256=trusted.sha(args.preflight),
        extra_source_sha256={"classical_features.py":trusted.sha(ROOT/"src/hw4_inspections/classical_features.py")},
        parameters=PARAMETERS, grid=GRID, limits={"inner_fits":6,"outer_fits":3,"full_fits":1},
        tree_selection="Previous inner quarter sklearnAP across100/300/500/800, earliest tie; recipe also previous innerAP only",
        selection="All6innerfits before3outer. Fullview meaninnerAP, lexicaltie; fullcount median innerbest counts.",
        scores="EqualmeanOLD3 sklearnAP. Rawranking and fixedwithinquarterfractionalrank50context/50MAP only. No probability averaging of uncalibrated rawmargin.",
        ranking_queries="Quarter; sorted contiguous groups; nativecategoricalstr; numericNaN preserved",
        full_gate="meanDelta>0, >=2nonnegativequarters,worstDelta>=-.005 AND2023Q4delta>=0",
        admission={"min_mean_delta_exclusive":0,"minimum_nonnegative_quarters":2,"minimum_worst_quarter_delta":-.005,"required_nonnegative_quarter":"2023-10-01"},
        interpretation="Exploratory after Public11/12; reused OLDdevelopment, no independent confirmation claim",
        new_q2_q3_metrics=False, uploads=0,
        source="https://catboost.ai/docs/en/concepts/loss-functions-ranking",
        time_gate={"per_fit_external_seconds":args.per_job_seconds,"total_external_seconds":args.total_seconds},
        allocation_amendment=("90sec perjob/15min total authorized before any real AP or completedHW4model; sameparams/views/grid, original30sec cancelledattempt preserved" if args.per_job_seconds==90 else None))
    trusted.write_json(args.out/"freeze.json", plan)
    cache = joblib.load(args.feature_cache)
    train, test, sample, views, cats = trusted.validate_cache(cache, args)
    full = clean(views["base"], cats)
    compact_cols = [c for c in select_feature_columns(full,"compact_context") if c not in ["forecast_year","cutoff_gap_days"]]
    compact_cats = ["facility_type","risk_level","area_id","forecast_quarter"]
    compact = clean(full[compact_cols], compact_cats)
    assert len(compact.columns)==94
    frames={"full346":(full,list(cats)),"compact94":(compact,compact_cats)}
    meta=pd.concat([train.loc[train.forecast_period.eq(q),list(trusted.REQUIRED)] for q in trusted.QUARTERS],ignore_index=True)
    anchor, anchor_test = trusted.align_anchors(meta,test,args)
    started=time.monotonic()
    def log(event,**d): print(json.dumps({"event":event,"elapsed_seconds":time.monotonic()-started,**d}),flush=True)
    inner={name:{} for name in pool}; actual_fits=0; jobs=[]
    for name in pool:
        frame, cc=frames[name]
        for q in trusted.QUARTERS:
            iq=(pd.Period(q,freq="Q")-1).start_time.strftime("%Y-%m-%d")
            fit=np.flatnonzero(train.forecast_period.lt(iq));pred=np.flatnonzero(train.forecast_period.eq(iq))
            log("inner_start",view=name,outer=q,inner=iq,train_rows=len(fit),predict_rows=len(pred))
            model,values,query,seconds=fit_predict(frame,cc,train,fit,pred,GRID[-1]);actual_fits+=1
            scores={n:float(average_precision_score(train.iloc[pred].target,p)) for n,p in values.items()}
            count=min(scores,key=lambda n:(-scores[n],n))
            inner[name][q]={"inner_quarter":iq,"count":count,"ap":scores[count],"curve":[{"count":n,"ap":scores[n]} for n in sorted(scores)],"seconds":seconds}
            jobs.append({"phase":"inner","view":name,"outer":q,"fit_before":iq,"train_rows":len(fit),"predict_rows":len(pred),"trees":800,"seconds":seconds})
            log("inner_complete",view=name,outer=q,count=count,ap=scores[count],seconds=seconds)
    trusted.write_json(args.out/"inner-curves.json",inner)
    choices={q:min(pool,key=lambda n:(-inner[n][q]["ap"],n)) for q in trusted.QUARTERS}
    choices["full"]=min(pool,key=lambda n:(-np.mean([inner[n][q]["ap"] for q in trusted.QUARTERS]),n))
    choices["full_count"]=int(np.median([inner[choices["full"]][q]["count"] for q in trusted.QUARTERS]))
    trusted.write_json(args.out/"inner-only-selection.json",choices)
    models=args.out/"models";models.mkdir();chunks=[]
    for q in trusted.QUARTERS:
        name=choices[q];count=inner[name][q]["count"];frame,cc=frames[name]
        fit=np.flatnonzero(train.forecast_period.lt(q));pred=np.flatnonzero(train.forecast_period.eq(q))
        log("outer_start",quarter=q,view=name,count=count)
        model,values,query,seconds=fit_predict(frame,cc,train,fit,pred,count);actual_fits+=1
        path=models/f"{q}.cbm";model.save_model(path);reloaded=CatBoostRanker();reloaded.load_model(path)
        assert np.array_equal(values[count],np.asarray(reloaded.predict(query),dtype=np.float64))
        chunks.append(values[count]);jobs.append({"phase":"outer","quarter":q,"view":name,"fit_before":q,"train_rows":len(fit),"predict_rows":len(pred),"trees":count,"reload_exact":True,"model_sha256":trusted.sha(path),"seconds":seconds})
        log("outer_complete",quarter=q,seconds=seconds,reload_exact=True)
    raw=np.concatenate(chunks);mix=np.empty(len(meta))
    for q in trusted.QUARTERS:
        mask=meta.forecast_period.eq(q).to_numpy();mix[mask]=.5*rank_scores(anchor[mask])+.5*rank_scores(raw[mask])
    vectors={"context_3seed":anchor,"catboost_MAP":raw,"catboost_MAP_context_rank50":mix}
    metrics=[]
    for q in trusted.QUARTERS:
        mask=meta.forecast_period.eq(q).to_numpy()
        for name,p in vectors.items():metrics.append({"quarter":q,"model":name,"ap":float(average_precision_score(meta.target[mask],p[mask]))})
    metrics=pd.DataFrame(metrics);metrics.to_csv(args.out/"quarter-metrics.csv",index=False);table=metrics.pivot(index="quarter",columns="model",values="ap")
    summary=[]
    for name in table:
        d=table[name]-table.context_3seed
        summary.append({"model":name,"mean_ap":float(table[name].mean()),"delta_vs_context":float(d.mean()),"nonnegative_quarters":int((d>=0).sum()),"worst_delta":float(d.min()),"q4_delta":float(d.loc["2023-10-01"]),"eligible":bool(name!="context_3seed" and d.mean()>0 and (d>=0).sum()>=2 and d.min()>=-.005 and d.loc["2023-10-01"]>=0)})
    summary=pd.DataFrame(summary).sort_values("mean_ap",ascending=False);summary.to_csv(args.out/"summary.csv",index=False)
    pd.concat([meta,pd.DataFrame(vectors)],axis=1).to_csv(args.out/"old-oof.csv",index=False)
    exports=[]
    if summary.eligible.any():
        selected=summary.loc[summary.eligible].iloc[0].model;name=choices["full"];count=choices["full_count"];frame,cc=frames[name]
        log("full_start",view=name,count=count)
        model,values,query,seconds=fit_predict(frame,cc,train,np.arange(len(train)),np.arange(len(train),len(frame)),count);actual_fits+=1
        path=models/"full.cbm";model.save_model(path);reloaded=CatBoostRanker();reloaded.load_model(path);assert np.array_equal(values[count],reloaded.predict(query))
        pred=rank_scores(values[count])
        if selected.endswith("rank50"):pred=.5*rank_scores(anchor_test)+.5*pred
        output=trusted.validate_submission(test,sample,pred,len(test));path=args.out/f"submission_postpublic_{selected}.csv";output.to_csv(path,index=False)
        exports.append({"file":path.name,"model":selected,"sha256":trusted.sha(path),"rows":len(output)})
        jobs.append({"phase":"full","view":name,"train_rows":len(train),"predict_rows":len(test),"trees":count,"seconds":seconds,"reload_exact":True})
        log("full_complete",view=name,count=count,seconds=seconds)
    trusted.write_json(args.out/"jobs.json",jobs)
    trusted.write_json(args.out/"complete.json",{"status":"complete","new_fits":actual_fits,"inner_fits":6,"outer_fits":3,"full_fits":actual_fits-9,"seeds":[42],"exports":exports,"recent_metrics":0,"kaggle_uploads":0,"seconds":time.monotonic()-started,"source_sha256":trusted.sha(Path(__file__))})
    print(summary.to_string(index=False),flush=True);log("complete",new_fits=actual_fits,exports=len(exports))


if __name__=="__main__":main()
