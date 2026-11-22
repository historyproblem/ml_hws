"""Two additional native32 seeds, preserving all earlier published predictors."""
from __future__ import annotations

import argparse
import fcntl
import gc
from importlib import metadata
import json
import os
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

import run_hw4_tabicl_ensemble32 as prior32
import run_hw4_tabicl_v100 as original
import run_hw4_tabpfn_v100 as common

SEEDS = (17, 2026)
PERIODS = (*common.OLD, "full")
COLS = ("icl32_s42", "icl32_s17", "icl32_s2026", "icl32_probmean3")
FILES = prior32.FILES
BASELINE = {
    "protocol.json": "cc972d3ef53b8361e08080208c0519d8a45887d4007893e8cd8d4ecfb0e85b72",
    "completion.json": "0137561d44778484629443711661f4088167220ed53f39d841e13ef9f8483010",
    "modelinfo.json": "2bfdd928d9a542421e712e0103e4158afd6f327a95e18d5ebe61d39cdf715284",
    "old_oof.csv": "a72192be075ac49eecb6f87d2098f803d2027174ce67b59753ce569a50c010cf",
    "test_predictions.csv": "4d7e6fdf0955d85dff708464fd7d3896c27b88de9f1090681b0ef66a6b4682e9",
    "fold_metrics.csv": "361015f2448ea2dce1f2f9f691c6619a409429b985dd2c13b30ce0af5c89ab6e",
}
PARENTS = {
    "duo_old_oof": "ae56491b074ff15fe7f6570d58757608c8da28876a74f600cdf314917bebe5a6",
    "context04_test": "636f8cbe87367b6df0bae6d9a1d79a57de863bc78a21bc2296f5c2ff85970634",
    "native4_test": "7c8fe950aa0470e4833b6ffefdccdf5918a9f0cd9595417373d030c6059059fe",
    "winner15_test": "3a64705549d64f3cc729ee32af6f522b8d8f61ce47bc9fa5effe9a2284851bef",
}
FORMS = {
    "seedbag": "I=mean_probability(native32_s42,native32_s17,native32_s2026); final=expit(.5*logit(clip(C04))+.5*logit(clip(I)))",
    "retain4_seedbag": "I=.5*native4_s42+.5*mean_probability(native32_s42,native32_s17,native32_s2026); final=expit(.5*logit(clip(C04))+.5*logit(clip(I)))",
}
GATE = {"reference": "winner15_equal50_logit_C04_native4", "mean_delta_exclusive": 0.,
    "nonnegative_quarters_min": 2, "worst_delta_min": -.002,
    "required_nonnegative_quarter": "2023-10-01"}
SOURCES = tuple(dict.fromkeys(["scripts/run_hw4_tabicl_seedbag.py",
    "scripts/run_hw4_tabicl_seedbag.sh", "scripts/analyze_hw4_tabicl_seedbag.py",
    "docs/hw4-tabicl-seedbag-20261123.md", *prior32.SOURCES]))


def settings(seed):
    return {**prior32.SETTINGS, "random_state": int(seed)}


def baseline(directory):
    for name, expected in BASELINE.items():
        if common.sha(directory / name) != expected:
            raise ValueError("Different accepted native32 seed42 baseline: " + name)
    p, c = [common.load(directory / n) for n in ("protocol.json", "completion.json")]
    if (common.digest({k:v for k,v in p.items() if k != "signature"}) != p["signature"]
        or c["signature"] != p["signature"] or p["settings"] != prior32.SETTINGS
        or p["input_sha256"] != common.RAW or p["checkpoint_sha256"] != original.WEIGHT):
        raise ValueError("Wrong native32 seed42 baseline protocol")
    for name, expected in p["source_sha256"].items():
        if common.sha(common.ROOT / name) != expected: raise ValueError("Changed baseline source: " + name)
    return p, common.load(directory / "modelinfo.json")


def estimator(args, seed):
    from tabicl import TabICLClassifier
    return TabICLClassifier(**settings(seed), model_path=args.checkpoint,
                           device=args.device, n_jobs=args.threads)


def actual_members(model):
    return sum(len(v) for v in model.ensemble_generator_.class_shuffles_.values())


def job(args, signature, seed, period, train, test, frame, cats):
    fit = original.prefix(train, period)
    query = np.arange(len(train), len(frame)) if period == "full" else np.flatnonzero(train.forecast_period.eq(period))
    ids = test.row_id if period == "full" else train.iloc[query].row_id
    identity = {"signature": signature, "seed": seed, "period": period,
        "case": "tabicl_v2_compact94", "settings": settings(seed),
        "train_rows": len(fit), "predict_rows": len(query), "features": list(frame),
        "categorical_indices": cats, "training_ids_sha256": common.digest(train.iloc[fit].row_id.tolist()),
        "prediction_ids_sha256": common.digest(ids.tolist()),
        "training_quarter_counts": {q:int(n) for q,n in train.iloc[fit].forecast_period.value_counts().sort_index().items()}}
    path = args.out / "jobs" / f"s{seed}__{period}"
    if path.exists():
        record = common.load(path / "complete.json")
        if not args.resume or record["identity"] != identity or common.sha(path / "prediction.npy") != record["prediction_sha256"]:
            raise ValueError("Changed seed-specific context checkpoint")
        values = np.load(path / "prediction.npy", allow_pickle=False)
        if (values.dtype != np.float64 or values.shape != (len(query),)
            or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any()
            or record["actual_native_components"] != 32 or record["actual_random_state"] != seed):
            raise ValueError("Invalid persisted native seed forecast")
        print(json.dumps({"event":"reuse", "seed":seed, "period":period}), flush=True)
        return values, record
    print(json.dumps({"event":"start", "seed":seed, "period":period,
        "train_rows":len(fit), "query_rows":len(query)}), flush=True)
    torch.manual_seed(seed)
    if args.device == "cuda": torch.cuda.reset_peak_memory_stats()
    start = time.monotonic(); model = estimator(args, seed)
    model.fit(frame.iloc[fit].copy(), train.iloc[fit].target.to_numpy())
    count = actual_members(model)
    if (list(model.classes_) != [0,1] or count != 32 or model.n_estimators != 32
        or model.random_state != seed): raise ValueError("Actual native seed/member settings differ")
    values, chunk, oom = common.predict_with_memory_policy(model, frame.iloc[query], args.device)
    repeat = period == common.OLD[0]
    if repeat and not np.array_equal(values, common.advance(model, frame.iloc[query], chunk)):
        raise ValueError("First complete OLD forward changed for this seed")
    record = {"identity":identity, "seconds":time.monotonic()-start,
        "checkpoint_sha256":original.WEIGHT, "actual_native_components":count,
        "actual_random_state":int(model.random_state), "actual_device":str(next(model.model_.parameters()).device),
        "query_chunk":chunk, "prior_cuda_oom_chunks":oom,
        "repeated_full_forward":repeat, "repeated_forward_exact":True if repeat else None,
        "gpu_peak_allocated_bytes":int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else None,
        "gpu_peak_reserved_bytes":int(torch.cuda.max_memory_reserved()) if args.device == "cuda" else None}
    path.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=path.parent, prefix=".pending-") as tmp:
        pending=Path(tmp)/"job";pending.mkdir();np.save(pending/"prediction.npy",values,allow_pickle=False)
        if not np.array_equal(np.load(pending/"prediction.npy",allow_pickle=False), values): raise ValueError("Saved native prediction changed")
        record["prediction_sha256"]=common.sha(pending/"prediction.npy")
        common.write_json(pending/"complete.json",record);os.rename(pending,path)
    del model;gc.collect()
    if args.device == "cuda": torch.cuda.empty_cache()
    print(json.dumps({"event":"finished", "seed":seed,"period":period,"seconds":record["seconds"]}),flush=True)
    return values,record


def smoke(args):
    rng=np.random.default_rng(91);x=pd.DataFrame(rng.normal(size=(80,7)),columns=[f"n{i}" for i in range(7)])
    x["category"]=["a","b"]*38+["unseen"]*4;x.loc[::9,"n1"]=np.nan
    y=(x.n0.to_numpy()[:64]>0).astype(int);results=[];vectors=[]
    for seed in SEEDS:
        torch.manual_seed(seed);model=estimator(args,seed);model.fit(x.iloc[:64],y)
        count=actual_members(model);values=common.advance(model,x.iloc[64:],16)
        if count!=32 or model.random_state!=seed or not np.array_equal(values,common.advance(model,x.iloc[64:],16)):
            raise ValueError("Native seed smoke failed")
        # The same safe persistence and float64 digest contract as real jobs.
        path=args.out/f"smoke-s{seed}.npy";np.save(path,values,allow_pickle=False)
        if not np.array_equal(values,np.load(path,allow_pickle=False)):raise ValueError("Smoke reload mismatch")
        results.append({"seed":seed,"actual_native_components":count,"repeated_exact":True,"saved_reload_exact":True,
            "training_ids_sha256":common.digest(list(range(64))),"query_ids_sha256":common.digest(list(range(64,80))),
            "prediction_npy_sha256":common.sha(path)});vectors.append(values)
        del model;gc.collect()
    if np.array_equal(*vectors): raise ValueError("Distinct seeds unexpectedly produced identical smoke vectors")
    result={"status":"PASS","jobs":results,"real_hw4_contexts":0,"distinct_seed_predictions":True}
    common.write_json(args.out/"smoke.json",result);print(json.dumps(result),flush=True)


def run(args):
    original.verify_env(args);torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    if args.device == "cuda":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8",":16:8"):raise ValueError("Set CUBLAS_WORKSPACE_CONFIG")
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    attempts=common.offline_runtime(args.out/".offline-state")
    if args.smoke:
        smoke(args)
        if attempts:raise ValueError("Smoke attempted network")
        return
    if args.device!="cuda":raise ValueError("Real seedbag run is registered for CUDA only")
    baseplan,baseinfo=baseline(args.baseline32_dir)
    if {n:common.sha(args.raw/n) for n in common.RAW}!=common.RAW:raise ValueError("Official inputs changed")
    plan={"format":"hw4-tabicl-seedbag32-1","seeds":list(SEEDS),"periods":list(PERIODS),"old_quarters":list(common.OLD),
        "settings_by_seed":{str(seed):settings(seed) for seed in SEEDS},"checkpoint_sha256":original.WEIGHT,
        "input_sha256":common.RAW,"source_sha256":{n:common.sha(common.ROOT/n) for n in SOURCES},
        "baseline42_artifact_sha256":BASELINE,"baseline42_signature":baseplan["signature"],"fixed_parent_sha256":PARENTS,
        "predeclared_forms":FORMS,"gate":GATE,"clip":1e-6,"recent_periods":[],
        "interpretation":"Post-Public exploratory extension of winner15; heavily reusedOLD, not independent confirmation",
        "context":"All official earlier rows sorted by (forecast_period,row_id); full21017 for test; identicalcompact94",
        "pooling":"Equal probabilitymean of native32 seeds42(existing),17,2026; no best-seedselection; optionalretain4 formfixedbeforequality",
        "new_jobs":8,"reused_baseline42_jobs":4,"new_optimizer_fit":0,"new_2024q2_q3_metrics":False,"kaggle_uploads":0,
        "versions":{n:metadata.version(n) for n in ["tabicl","scikit-learn","torch","numpy","pandas","scipy"]},
        "device":args.device,"threads":args.threads,"gpu":common.gpu_info(args.device),"python":platform.python_version()}
    plan["signature"]=common.digest(plan);path=args.out/"protocol.json"
    if path.exists():
        if not args.resume or common.load(path)!=plan:raise ValueError("Changed seedbag plan")
    else:
        if any(p.name not in {".lock","run.log",".offline-state"} for p in args.out.iterdir()):raise ValueError("New run requires empty output")
        common.write_json(path,plan)
    train,test,sample=[pd.read_csv(args.raw/n) for n in ["train.csv","test_student.csv","sample_submission.csv"]]
    if (len(train),len(test))!=(21017,2246) or "target" in test or any(not v.row_id.is_unique or v.row_id.isna().any() for v in [train,test,sample]):raise ValueError("Official shape/identity mismatch")
    rows=pd.concat([train.drop(columns="target"),test],ignore_index=True);history=pd.read_csv(args.raw/"inspection_history.csv")
    own,cats=common.build_features(rows,history);features=pd.concat([own,common.build_context_features(rows,history)],axis=1)
    frame,cat_ids=common.views(features,cats)[common.CASES[0]]
    keys=["row_id","site_id","forecast_period","target"]
    old=pd.concat([train.loc[train.forecast_period.eq(q),keys] for q in common.OLD],ignore_index=True);forecast=test[["row_id","forecast_period"]].copy()
    bo=pd.read_csv(args.baseline32_dir/"old_oof.csv",float_precision="round_trip");bt=pd.read_csv(args.baseline32_dir/"test_predictions.csv",float_precision="round_trip")
    pd.testing.assert_frame_equal(old,bo[keys],check_dtype=False,check_exact=True);pd.testing.assert_frame_equal(forecast,bt[list(forecast)],check_dtype=False,check_exact=True)
    old[COLS[0]]=bo[prior32.CASE];forecast[COLS[0]]=bt[prior32.CASE]
    jobs=[];metrics=[]
    for seed in SEEDS:
        pieces=[]
        for period in PERIODS:
            values,record=job(args,plan["signature"],seed,period,train,test,frame,cat_ids);jobs.append(record)
            if period=="full":forecast[f"icl32_s{seed}"]=values
            else:pieces.append(values)
        old[f"icl32_s{seed}"]=np.concatenate(pieces)
    for target in [old,forecast]:target[COLS[-1]]=np.mean(target[list(COLS[:3])].to_numpy(),axis=1)
    for q in common.OLD:
        selected=old.loc[old.forecast_period.eq(q)]
        for column in COLS[1:]:metrics.append({"forecast_period":q,"model":column,"n":len(selected),"positive":int(selected.target.sum()),"ap":float(average_precision_score(selected.target,selected[column]))})
    if attempts:raise ValueError("Network attempted during offlineinference")
    for name,data in [("old_oof.csv",old),("test_predictions.csv",forecast),("fold_metrics.csv",pd.DataFrame(metrics))]:common.write_csv(args.out/name,data)
    common.write_json(args.out/"modelinfo.json",{"jobs":jobs,"baseline42_jobs":baseinfo["jobs"],"runtime_network_attempts":0,"checkpoint_sha256":original.WEIGHT})
    common.write_json(args.out/"completion.json",{"status":"complete","signature":plan["signature"],"new_jobs":8,"reused42_jobs":4,
        "old_rows":len(old),"test_rows":len(forecast),"new_2024q2_q3_metrics":False,"kaggle_uploads":0,
        "artifacts":{n:common.sha(args.out/n) for n in FILES if n!="completion.json"}})
    archive=prior32.pack(args.out);print(json.dumps({"event":"complete","archive":str(archive),"sha256":common.sha(archive),"new_jobs":8}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["raw","out","checkpoint","baseline32-dir"]:parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--device",choices=["cuda","cpu"],default="cuda");parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--resume",action="store_true");parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args();args.checkpoint=args.checkpoint.resolve();args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/".lock").open("a") as lock,threadpool_limits(limits=args.threads):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(args)


if __name__=="__main__":main()
