"""One fixed out-of-site historical code-score augmentation of the exact G0 logistic model."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

import run_hw4_final_research as common
from hw4_inspections import coded_visit_embedding as coded
from hw4_inspections.additive_search import predict_recipe

NAME="gam_g0_plus_code_score"
BLEND="gam_g0_50_plus_code_score_50"
RECENT=("2024-04-01","2024-07-01")
KEYS=["row_id","site_id","forecast_period","target"]
RAW_SHA={"train.csv":"30b6acd8c9f2eec4259a775b098017934ba5497cb37689eb00b639433472f0d9",
    "test_student.csv":"1014585cdbdb45e6b1ce7bcb5723700aa025826975a6c5c6910ed5ce39f6f643",
    "inspection_history.csv":"9637ec0279d9cf521795268a0f44099d7d51a795b5155a0d78405a770966afc7",
    "sample_submission.csv":"2d9b5084b510302cd027d46e0c5f9b8e986d48f417d9a582a8b5ec2b3c953b42"}
OOF_SHA="e55e6c6f7e77ffd8b82262640ae808bcd7a59f7f5550ff3a0282cddbe193a366"
FULL_SHA="58d41da9004178840ff9ee164e5b414d22cfb4eff86d561923c1564498b9c810"
sha,read,save=common.sha,common.read_json,common.write_json


def signature(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()


def directory(args,q):
    return (args.g0_final/"jobs"/"full__core_interactions_C0.01" if q=="full" else args.g0_teacher/"teacher"/q)


def make_plan(args):
    meta=args.feature_cache.with_name("features-meta.json");cache=read(meta)
    rawhash={n:sha(args.raw/n) for n in RAW_SHA}
    if rawhash!=RAW_SHA or cache["raw_sha256"]!=rawhash or sha(args.feature_cache)!=cache["cache_sha256"]:
        raise ValueError("Wrong cache or official input hashes")
    module=common.ROOT/"src/hw4_inspections"
    if cache["builder_sha256"]!={n:sha(module/n) for n in common.BUILDERS}: raise ValueError("Builder hashes changed")
    oldfile=args.g0_teacher/"outer-oof.csv";fullfile=args.g0_final/"submission_additive_search_raw.csv"
    if sha(oldfile)!=OOF_SHA or sha(fullfile)!=FULL_SHA: raise ValueError("Require the verified G0 OOF/test exports")
    marks={}
    for q in (*common.QUARTERS,*RECENT,"full"):
        d=directory(args,q);m=read(d/"complete.json")
        if (m["identity"]["recipe"]!=coded.G0 or m["identity"]["train_rows"]<=0
            or any(sha(d/n)!=h for n,h in m["sha256"].items())):
            raise ValueError(f"Invalid saved G0 checkpoint: {q}")
        if q!="full" and (m["identity"]["quarter"]!=q or m["train_max_period"]>=q or not m["reload_raw_exact"]):
            raise ValueError("Lookahead or incorrect G0 quarter")
        marks[q]={"complete_sha256":sha(d/"complete.json"),"artifacts":m["sha256"],"identity":m["identity"]}
    files=["coded_visit_embedding.py","event_learning.py","features.py","additive_search.py","additive_logit.py",
           "classical_features.py","next_round.py","classical_analysis.py","delivery.py"]
    return {"schema":1,"model":NAME,"smoke":False,"source_sha256":{**{f:sha(module/f) for f in files},
              "run_hw4_final_research.py":sha(Path(common.__file__)),Path(__file__).name:sha(Path(__file__))},
            "raw_sha256":rawhash,"feature_cache_sha256":cache["cache_sha256"],"cache_metadata_sha256":sha(meta),
            "g0_teacher_model_markers":marks,"g0_oof_sha256":OOF_SHA,"g0_test_sha256":FULL_SHA,
            "packages":{n:importlib.metadata.version(n) for n in ("numpy","pandas","scikit-learn","scipy","joblib")},
            "machine":platform.machine(),"python":platform.python_version(),"threads":2,"bootstrap":args.bootstrap,
            "teacher":{"objective":"fail of same already historical routine, pass/conditional=0; unknown outcomes excluded only at fit",
              "input":"44 distinct code presence indicators and log1p(count)/log(45); blank means zero recorded codes",
              "parameters":coded.TEACHER,"folds":5,"site_split":"SHA256(site_id UTF8), first 8 bytes big endian mod 5; exclude all visits in query site's bucket.",
              "snapshot_rule":"Exact period and week < row cutoff, all history. Duplicate equivalent records coalesced; conflicting site/week fails explicitly."},
            "features":list(coded.FEATURES),"aggregation":"Decision logits of historical routine visits; latest/last3 mean/all mean. Unknown historical outcomes may be scored from codes. Empty history -> NaN.",
            "auxiliary":"2024Q1 snapshot only: site-fold held-out AP predicting already observed historical routine failure, comparing code+count with one fixed count-only model. Never affects screening or stop.",
            "second_stage":"Reuse exact saved unweighted G0 core_interactions/C.01 transformer for the same training prefix. Append 3 training-median-imputed/StandardScaler logits + 3 missing flags. Joint fit with identical L2 parameters and intercept; no search.",
            "replay":"Saved G0 probabilities replay within 2e-14 if BLAS rounding differs; G0 design prefix must be bitwise exact. New models reload exactly.",
            "admission":"Two fixed variants: raw joint model and probability average 50:50 with G0. Mean AP delta vs G0>0, at least 2/3 nonnegative, worst delta >= -.005. Prefer highest eligible mean AP, lexical name tie.",
            "old_folds":list(common.QUARTERS),"recent_prediction_only_quarters":list(RECENT),
            "full":"Fit full and produce two unscored recent augmented forecasts only after at least one admissible older variant. At most one submission.",
            "new_q2_q3_metrics":False,"uploads":0}


def embed_snapshot(args,sig,history,group):
    rows=group[["row_id","site_id","forecast_period","history_before"]].reset_index(drop=True)
    q,cut=rows.loc[0,["forecast_period","history_before"]]
    job=args.out/"heads"/f"{q}__{cut}"
    identity={"signature":sig,"quarter":q,"cutoff":cut,"n":len(rows),"ids":signature(rows.row_id.tolist())}
    if job.exists():
        m=read(job/"complete.json")
        if (not args.resume or m["identity"]!=identity or not m["reloaded_exact"] or
            any(sha(job/n)!=h for n,h in m["sha256"].items())):
            raise ValueError("Changed snapshot embedding checkpoint")
        values=np.load(job/"scores.npy",allow_pickle=False)
    else:
        frame,models,stats=coded.embed(rows,history,auxiliary=q=="2024-01-01")
        values=frame.to_numpy(dtype=np.float64);job.parent.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=job.parent,prefix=".pending-") as tmp:
            d=Path(tmp)/"job";d.mkdir()
            joblib.dump(models,d/"teachers.joblib",compress=1)
            replay=coded.apply_heads(rows,history,joblib.load(d/"teachers.joblib")).to_numpy()
            if not np.array_equal(values,replay,equal_nan=True): raise AssertionError("Historical teacher replay changed")
            np.save(d/"scores.npy",values,allow_pickle=False);save(d/"audit.json",stats)
            save(d/"complete.json",{"identity":identity,"reloaded_exact":True,"sha256":{p.name:sha(p) for p in d.iterdir()}})
            os.rename(d,job)
        print(json.dumps({"event":"snapshot_encoded","period":q,"visits":stats["unique_routine_visits"],
              "query_rows":len(rows),"with_routine":stats["query_with_any_routine"]}),flush=True)
    if values.shape!=(len(rows),3) or np.isinf(values).any(): raise ValueError("Invalid historical score matrix")
    return pd.DataFrame(values,index=rows.row_id,columns=coded.FEATURES)


def score_rows(args,sig,history,rows):
    frames=[embed_snapshot(args,sig,history,g) for _,g in rows.groupby(["forecast_period","history_before"],sort=True)]
    joined=pd.concat(frames)
    if not joined.index.is_unique or set(joined.index)!=set(rows.row_id): raise ValueError("Embedding IDs changed")
    return joined.loc[rows.row_id].reset_index(drop=True)


def fit_job(args,sig,q,train,test,frame,extra,anchor):
    full=q=="full"
    fit=np.arange(len(train)) if full else np.flatnonzero(train.forecast_period.lt(q))
    val=np.arange(len(train),len(frame)) if full else np.flatnonzero(train.forecast_period.eq(q))
    ids=test.row_id if full else train.iloc[val].row_id
    d=directory(args,q);old=joblib.load(d/"model.joblib");coded.check_base(old)
    parent=read(d/"complete.json")
    if parent["identity"]["train_rows"]!=len(fit) or parent["identity"]["predict_rows"]!=len(val):
        raise ValueError("G0 training or query count differs")
    if not full and not np.array_equal(ids.to_numpy(dtype=str),np.load(d/"row-ids.npy",allow_pickle=False)):
        raise ValueError("G0 saved row order mismatch")
    g0=predict_recipe(old,frame.iloc[val])
    expected=expit(np.load(d/"raw.npy",allow_pickle=False)) if not full else np.load(d/"prediction.npy",allow_pickle=False)
    if np.max(np.abs(g0-expected))>2e-14: raise ValueError("G0 checkpoint failed official row replay")
    if anchor is not None and np.max(np.abs(anchor-g0))>2e-14: raise ValueError("G0 anchor failed official row replay")
    ref=expected if anchor is None else anchor
    job=args.out/"jobs"/q
    identity={"signature":sig,"quarter":q,"train_rows":len(fit),"predict_rows":len(val),
              "train_ids":signature(train.iloc[fit].row_id.tolist()),"predict_ids":signature(ids.tolist())}
    if job.exists():
        m=read(job/"complete.json")
        if (not args.resume or m["identity"]!=identity or not m["loaded_forward_exact"] or
            any(sha(job/n)!=h for n,h in m["sha256"].items())): raise ValueError("Changed augmented model")
        p=np.load(job/"prediction.npy",allow_pickle=False)
    else:
        bundle=coded.fit_augmented(old,frame.iloc[fit],extra.iloc[fit],train.target.iloc[fit])
        p=coded.predict(bundle,frame.iloc[val],extra.iloc[val])
        job.parent.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=job.parent,prefix=".pending-") as tmp:
            out=Path(tmp)/"job";out.mkdir();joblib.dump(bundle,out/"model.joblib",compress=1)
            replay=coded.predict(joblib.load(out/"model.joblib"),frame.iloc[val],extra.iloc[val])
            if not np.array_equal(p,replay): raise AssertionError("Saved augmented model forward changed")
            np.save(out/"prediction.npy",p,allow_pickle=False)
            save(out/"fit.json",{"base_prefix_exact":bundle["base_prefix_exact"],"base_columns":bundle["base_columns"],
                "augmented_columns":len(bundle["feature_names"]),"iterations":bundle["n_iter"],"extra_medians":bundle["extra"].medians_.tolist(),
                "g0_max_abs_replay_difference":float(np.max(np.abs(ref-g0))),"evaluation_metrics_computed_in_job":False})
            save(out/"complete.json",{"identity":identity,"loaded_forward_exact":True,"sha256":{f.name:sha(f) for f in out.iterdir()}})
            os.rename(out,job)
    return common.probabilities(p,len(val)),ref


def run(args):
    if args.bootstrap<1: raise ValueError("Use a positive bootstrap count")
    plan=make_plan(args);sig=signature(plan);plan["signature"]=sig
    f=args.out/"freeze.json"
    if f.exists():
        if not args.resume or read(f)!=plan: raise ValueError("Changed frozen code-score hypothesis")
    else:
        if any(p.name not in {".lock","run.log"} for p in args.out.iterdir()): raise ValueError("Require fresh output")
        save(f,plan)
    manifest=args.out/"manifest.json"
    if manifest.exists():
        old=read(manifest)
        if old["signature"]!=sig or any(sha(args.out/n)!=v for n,v in old["output_sha256"].items()):
            raise ValueError("Changed completed outputs")
    train,test,sample,views,_=common.validate_cache(joblib.load(args.feature_cache),args);base=views["base"]
    history=pd.read_csv(args.raw/"inspection_history.csv",usecols=coded.HIST_FIELDS)
    oldidx=np.flatnonzero(train.forecast_period.le(common.QUARTERS[-1]))
    early=train.iloc[oldidx].reset_index(drop=True);earlybase=base.iloc[oldidx].reset_index(drop=True)
    extras=score_rows(args,sig,history,early)
    source=common.read_csv(args.g0_teacher/"outer-oof.csv").set_index("row_id")
    meta=pd.concat([early.loc[early.forecast_period.eq(q),KEYS] for q in common.QUARTERS],ignore_index=True)
    source=source.loc[meta.row_id].reset_index()
    pd.testing.assert_frame_equal(meta,source[KEYS],check_dtype=False)
    preds=[]
    for q in common.QUARTERS:
        anchor=source.loc[source.forecast_period.eq(q),"fixed_teacher"].to_numpy()
        p,_=fit_job(args,sig,q,early,test,earlybase,extras,anchor)
        preds.append(p)
    aug=np.concatenate(preds);g0=source.fixed_teacher.to_numpy()
    forecasts={"G0":g0,NAME:aug,BLEND:(g0+aug)/2}
    validation=common.evaluate_candidates(meta,forecasts,reference="G0",bootstrap=args.bootstrap,seed=20261122)
    summary=validation["summary"].copy();summary["eligible"]&=summary.worst_quarter_delta.ge(-.005)
    common.write_csv(args.out/"old-summary.csv",summary)
    common.write_csv(args.out/"old-quarter-metrics.csv",validation["quarter_scores"])
    common.write_csv(args.out/"old-oof.csv",pd.concat([meta,pd.DataFrame(forecasts)],axis=1))
    passed=summary.loc[summary.eligible].sort_values(["mean_ap","model"],ascending=[False,True]).model.tolist()
    winner=passed[0] if passed else None
    files=[]
    if winner:
        late=train.loc[train.forecast_period.gt(common.QUARTERS[-1])]
        later=score_rows(args,sig,history,late);test_ex=score_rows(args,sig,history,test)
        all_scores=pd.concat([extras.set_index(early.row_id),later.set_index(late.row_id),test_ex.set_index(test.row_id)])
        all_scores=all_scores.loc[pd.concat([train.row_id,test.row_id])].reset_index(drop=True)
        full_base=common.read_csv(args.g0_final/"submission_additive_search_raw.csv").set_index("row_id").loc[test.row_id,"target"].to_numpy()
        full,g0=fit_job(args,sig,"full",train,test,base,all_scores,full_base)
        test_values={NAME:full,BLEND:(full+g0)/2}
        common.write_csv(args.out/"test-predictions.csv",test[["row_id","forecast_period"]].assign(**{"G0":g0,**test_values}))
        submission=common.validate_submission(test,sample,test_values[winner],2246)
        path=args.out/f"submission_{winner}.csv";common.write_csv(path,submission)
        pd.testing.assert_frame_equal(submission,common.read_csv(path),check_exact=True)
        files.append({"csv":path.name,"sha256":sha(path),"model":winner,"rows":len(submission)})
        sealed=[]
        for q in RECENT:
            p,ref=fit_job(args,sig,q,train,test,base,all_scores,None)
            block=train.loc[train.forecast_period.eq(q),KEYS].copy()
            block["G0"]=ref;block[NAME]=p;block[BLEND]=(p+ref)/2
            sealed.append(block)
        common.write_csv(args.out/"recent-predictions.csv",pd.concat(sealed,ignore_index=True))
    outpaths=[p for p in args.out.iterdir() if p.suffix in {".csv",".json"} and p!=manifest]
    report={"complete":True,"signature":sig,"eligible_old":passed,"chosen":winner,"files":files,
        "head_snapshots":len(list((args.out/"heads").glob("*/audit.json"))),"second_stage_fits":len(list((args.out/"jobs").glob("*/fit.json"))),
        "new_q2_q3_metrics":False,"uploads":0,"output_sha256":{p.name:sha(p) for p in outpaths}}
    save(manifest,report);print(json.dumps({"event":"complete","chosen":winner,"files":files}),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ["raw","feature-cache","out","g0-teacher","g0-final"]:p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--bootstrap",type=int,default=2000);p.add_argument("--resume",action="store_true")
    args=p.parse_args();args.smoke=False;args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/".lock").open("a") as lock,threadpool_limits(limits=2):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(args)
if __name__=="__main__":main()
