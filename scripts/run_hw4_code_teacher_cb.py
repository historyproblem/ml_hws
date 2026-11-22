"""Post-Public fixed code teachers + CatBoost: cached LR versus nonlinear trees.

Teachers describe completed historical routines only. A query site's entire
hash bucket is excluded from its teacher; forecasts remain strictly temporal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_hw4_final_research as common
from hw4_inspections import coded_visit_embedding as coded
from hw4_inspections.deep_boosting import fit_booster, predict_booster

FIELDS = ["last_routine_teacher_logit", "last3_routine_teacher_logit_mean", "all_routine_teacher_logit_mean"]
SPEC = {"family":"catboost", "max_iterations":500, "params":{
    "learning_rate":.04,"depth":5,"l2_leaf_reg":10.,"boosting_type":"Plain",
    "bootstrap_type":"MVS","subsample":.8,"rsm":1.,"border_count":254}}
TEACHER = dict(n_estimators=200, max_depth=3, learning_rate=.05,
    min_child_weight=10., reg_lambda=10., subsample=.8, colsample_bytree=1.,
    max_bin=64, n_jobs=2, tree_method="hist", device="cpu",
    objective="binary:logistic", verbosity=0)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def tree_snapshot(rows, history, out):
    import xgboost as xgb
    q, cutoff = rows.iloc[0][["forecast_period", "history_before"]]
    h = coded.routine_snapshot(history, q, cutoff)
    x = coded.code_matrix(h.tokens)
    known = h.result.isin(coded.RESULTS).to_numpy()
    y = h.result.eq("fail").to_numpy(dtype=int)
    buckets = h.bucket.to_numpy()
    score = np.full(len(h), np.nan)
    models, audits = {}, []
    for k in range(coded.FOLDS):
        fit = known & (buckets != k); pred = buckets == k
        if not pred.any():
            continue
        if set(h.loc[fit,"site_id"]) & set(h.loc[pred,"site_id"]):
            raise AssertionError("Teacher contains query sites")
        model = xgb.XGBClassifier(**TEACHER, random_state=42+k)
        with threadpool_limits(limits=2):
            model.fit(x[fit], y[fit])
            values = model.predict(x[pred], output_margin=True).astype(float)
        score[pred] = values
        path = out / f"fold{k}.ubj";model.save_model(path)
        restored = xgb.XGBClassifier();restored.load_model(path)
        if not np.array_equal(values, restored.predict(x[pred], output_margin=True).astype(float)):
            raise AssertionError("Teacher native model reload changed scores")
        models[k] = model
        audits.append({"bucket":k,"fit_visits":int(fit.sum()),"predicted_visits":int(pred.sum()),
            "training_max_week":h.loc[fit,"week"].max().strftime("%Y-%m-%d"),"cutoff":cutoff,
            "excluded_query_sites":True,"model_sha256":common.sha(path),"reload_exact":True})
    if not np.isfinite(score).all():
        raise ValueError("Teacher does not cover all historical routines")
    values = coded.aggregate(rows.reset_index(drop=True), h, score).to_numpy()
    np.save(out/"scores.npy",values,allow_pickle=False)
    common.write_json(out/"audit.json",{"period":q,"cutoff":cutoff,"query_rows":len(rows),
        "historical_visits":len(h),"heads":audits,"teacher_fits":len(models),
        "unknown_outcomes_excluded_from_fit":True,"official_future_targets_used":False})
    return values, len(models)


def encode(rows, history, directory, lr_root, mode):
    assembled, fits = [], 0
    source_signature = common.read_json(lr_root/"freeze.json")["signature"]
    for (q, cutoff), group in rows.groupby(["forecast_period","history_before"],sort=True):
        if not group.site_id.is_unique:
            raise ValueError("One official query per site and snapshot required")
        target = directory / f"{q}__{cutoff}";target.mkdir(parents=True)
        if mode == "lr":
            origin = lr_root/"heads"/f"{q}__{cutoff}"
            mark = common.read_json(origin/"complete.json")
            expected={"signature":source_signature,"quarter":q,"cutoff":cutoff,"n":len(group),
                      "ids":digest(group.row_id.tolist())}
            if mark["identity"] != expected or not mark["reloaded_exact"]:
                raise ValueError("Cached LR code-score query identity differs")
            if any(common.sha(origin/n)!=h for n,h in mark["sha256"].items()):
                raise ValueError("Cached LR teacher artifacts changed")
            values = np.load(origin/"scores.npy",allow_pickle=False)
            common.write_json(target/"reuse.json",{"source":str(origin),"complete_sha256":common.sha(origin/"complete.json"),"new_fits":0})
        else:
            values, count = tree_snapshot(group,history,target);fits += count
        if values.shape!=(len(group),3) or np.isinf(values).any():
            raise ValueError("Bad auxiliary feature dimensions")
        assembled.append(pd.DataFrame(values,index=group.row_id,columns=FIELDS))
    result=pd.concat(assembled)
    if not result.index.is_unique or set(result.index)!=set(rows.row_id):
        raise ValueError("Auxiliary row identities changed")
    return result.loc[rows.row_id].reset_index(drop=True),fits


def forecast(mode, seed, q, frame, train, cats, directory):
    full=q=="full"
    fit=np.arange(len(train)) if full else np.flatnonzero(train.forecast_period.lt(q))
    val=np.arange(len(train),len(frame)) if full else np.flatnonzero(train.forecast_period.eq(q))
    model=fit_booster(SPEC,frame.iloc[fit],train.target.iloc[fit],cats,seed=seed,threads=2,iterations=500)
    p=predict_booster(model,frame.iloc[val]);file=directory/f"{mode}__{q}__s{seed}.joblib"
    joblib.dump(model,file,compress=1)
    if not np.array_equal(p,predict_booster(joblib.load(file),frame.iloc[val])):
        raise AssertionError("Augmented CatBoost reload changed prediction")
    np.save(file.with_suffix(".npy"),p,allow_pickle=False)
    return p


def assess(meta, anchor, icl, predictions):
    rows=[]
    vectors={"context_3seed":anchor,"tabicl94":icl}
    for mode,p in predictions.items():
        vectors[mode]=p;vectors[mode+"_icl50"]=(p+icl)/2
    for q in common.QUARTERS:
        mask=meta.forecast_period.eq(q).to_numpy()
        for name,p in vectors.items():
            rows.append({"model":name,"quarter":q,"ap":float(average_precision_score(meta.target[mask],p[mask]))})
    metrics=pd.DataFrame(rows);table=metrics.pivot(index="quarter",columns="model",values="ap")
    summary=[]
    for name in table:
        reference="tabicl94" if name.endswith("_icl50") else "context_3seed"
        delta=table[name]-table[reference]
        eligible=name not in ["context_3seed","tabicl94"] and bool(delta.mean()>0 and (delta>=0).sum()>=2 and delta.min()>=-.005)
        summary.append({"model":name,"mean_ap":float(table[name].mean()),"reference":reference,
            "mean_delta":float(delta.mean()),"nonnegative_quarters":int((delta>=0).sum()),
            "worst_delta":float(delta.min()),"eligible":eligible})
    return metrics,pd.DataFrame(summary).sort_values("mean_ap",ascending=False),vectors


def main():
    parser=argparse.ArgumentParser()
    for name in ["raw","feature-cache","baseline-cv","context-test","lr-embeddings","icl-old","icl-test","out"]:
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();args.smoke=False;args.threads=2;args.bootstrap=0
    args.out.mkdir(parents=True,exist_ok=True)
    if any(args.out.iterdir()):raise ValueError("New experiment requires empty output directory")
    start=time.monotonic()
    train,test,sample,views,cats=common.validate_cache(joblib.load(args.feature_cache),args)
    pool={"lr_code_teacher_cb":SPEC,"tree_code_teacher_cb":SPEC}
    plan=common.provenance(args,pool)
    plan.update(seeds=[42,17,2026],tree_selection="Fixed500 referenceCatBoost, no iteration or recipe search",
        scores="OLD3 AP: rawnewCB vs context04; fixed50%newCB+50%ICL94 probabilitymean vsICL94",
        admission="positive mean,>=2/3nonnegative,worst>=-.005; ifscreenpassesconfirm2seeds, choosebestconfirmedrawormix",
        source_new_sha256=common.sha(Path(__file__)),teacher=TEACHER,
        teacher_inputs="44codepresence+log1p(count)/log45; known historical routine result fail1/passconditional0; hashsiteOOB5",
        cached_lr_freeze_sha256=common.sha(args.lr_embeddings/"freeze.json"),
        icl_old_sha256=common.sha(args.icl_old),icl_test_sha256=common.sha(args.icl_test),
        stages="Screen2views*3OLDfit, confirmatmost1winner*2seeds*3OLD, fullmean3onlyafterconfirmation; teacherheads5persnapshot",
        interpretation="Post-Public exploratory, OLDdevelopmentheavilyreused; noindependentvalidationclaim",
        new_q2_q3_metrics=False,uploads=0)
    common.write_json(args.out/"freeze.json",plan)
    history=common.read_csv(args.raw/"inspection_history.csv")
    early_idx=np.flatnonzero(train.forecast_period.le(common.QUARTERS[-1]))
    early=train.iloc[early_idx].copy()
    base=views["base"]
    earlybase=base.iloc[early_idx].reset_index(drop=True)
    models=args.out/"forecast-models";models.mkdir()
    meta=pd.concat([early.loc[early.forecast_period.eq(q),list(common.REQUIRED)]for q in common.QUARTERS],ignore_index=True)
    anchor,_=common.align_anchors(meta,test,args)
    icl_old=common.read_csv(args.icl_old).set_index("row_id").loc[meta.row_id].reset_index()
    pd.testing.assert_frame_equal(meta,icl_old[list(common.REQUIRED)],check_dtype=False)
    icl_column="tabicl_v2_compact94__raw"
    icl=common.probabilities(icl_old[icl_column],len(meta))
    encoded,predictions={},{}
    teacher_fits=forecast_fits=0
    for mode in ["lr","tree"]:
        print(json.dumps({"event":"encode_start","mode":mode}),flush=True)
        values,n=encode(early,history,args.out/(mode+"-heads"),args.lr_embeddings,mode);teacher_fits+=n
        encoded[mode]=values
        frame=pd.concat([earlybase,values],axis=1)
        chunks=[]
        for q in common.QUARTERS:
            print(json.dumps({"event":"forecast_start","mode":mode,"quarter":q,"seed":42}),flush=True)
            chunks.append(forecast(mode,42,q,frame,early,cats,models));forecast_fits+=1
        predictions[mode]=np.concatenate(chunks)
    metrics,summary,vectors=assess(meta,anchor,icl,predictions)
    metrics.to_csv(args.out/"screen-quarter-metrics.csv",index=False);summary.to_csv(args.out/"screen-summary.csv",index=False)
    pd.concat([meta,pd.DataFrame(vectors)],axis=1).to_csv(args.out/"screen-old-oof.csv",index=False)
    eligible=summary.loc[summary.eligible]
    winner=eligible.iloc[0].model if len(eligible) else None
    exports=[]
    if winner:
        mode=winner.split("_")[0]
        frame=pd.concat([earlybase,encoded[mode]],axis=1)
        extra_seeds=[]
        for seed in [17,2026]:
            extra_seeds.append(np.concatenate([forecast(mode,seed,q,frame,early,cats,models)for q in common.QUARTERS]));forecast_fits+=3
        confirmed=np.mean([predictions[mode],*extra_seeds],axis=0)
        qm,sm,v=assess(meta,anchor,icl,{mode:confirmed})
        qm.to_csv(args.out/"confirmed-quarter-metrics.csv",index=False);sm.to_csv(args.out/"confirmed-summary.csv",index=False)
        pd.concat([meta,pd.DataFrame(v)],axis=1).to_csv(args.out/"confirmed-old-oof.csv",index=False)
        # Keep the exact screen-selected formula; confirmation does not choose a new weight.
        admissible=bool(sm.set_index("model").loc[winner,"eligible"])
        if admissible:
            later=train.loc[train.forecast_period.gt(common.QUARTERS[-1])]
            future=pd.concat([later,test],ignore_index=True)
            later_values,n=encode(future,history,args.out/(mode+"-late-heads"),args.lr_embeddings,mode);teacher_fits+=n
            early_scores=encoded[mode].set_axis(early.row_id)
            late_scores=later_values.set_axis(future.row_id)
            values=pd.concat([early_scores,late_scores]).loc[pd.concat([train.row_id,test.row_id])].reset_index(drop=True)
            fullframe=pd.concat([base,values],axis=1)
            p=np.mean([forecast(mode,seed,"full",fullframe,train,cats,models)for seed in [42,17,2026]],axis=0);forecast_fits+=3
            if winner.endswith("_icl50"):
                it=common.read_csv(args.icl_test).set_index("row_id").loc[test.row_id,"target"].to_numpy()
                p=(p+it)/2
            submission=common.validate_submission(test,sample,p,len(test))
            path=args.out/f"submission_postpublic_{winner}.csv";submission.to_csv(path,index=False)
            exports.append({"file":path.name,"sha256":common.sha(path),"rows":len(submission)})
    common.write_json(args.out/"complete.json",{"new_teacher_fits":teacher_fits,"new_forecast_fits":forecast_fits,
        "screen_winner":winner,"exports":exports,"new_recent_metrics":0,"uploads":0,
        "elapsed_seconds":time.monotonic()-start,"source_sha256":common.sha(Path(__file__))})
    print(summary.to_string(index=False),flush=True)


if __name__=="__main__":main()
