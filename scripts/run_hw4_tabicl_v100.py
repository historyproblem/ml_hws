"""TabICLv2 full-context offline diagnostics with score-isolated recent quarters."""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
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

import run_hw4_tabpfn_v100 as common

CASES = ("tabicl_v2_compact94","tabicl_v2_full346")
WEIGHT = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
WHEEL = "c80b8ec2719be26bc0bf1dd758a2361f58019d8f7877038b785529a425789ce4"
SOURCES = ["scripts/run_hw4_tabicl_v100.py","scripts/run_hw4_tabicl_v100.sh","configs/hw4-tabicl-v2-overlay.txt",
           "scripts/run_hw4_tabpfn_v100.py",*[n for n in common.SOURCES if n.startswith("src/")]]
SETTINGS = {"n_estimators":4,"batch_size":1,"kv_cache":False,"random_state":42,"use_amp":False,
            "use_fa3":False,"offload_mode":"auto","allow_auto_download":False,
            "checkpoint_version":"tabicl-classifier-v2-20260212.ckpt"}
sha, write_json, load, digest = common.sha, common.write_json, common.load, common.digest


def model(args):
    from tabicl import TabICLClassifier
    return TabICLClassifier(**SETTINGS,model_path=args.checkpoint,device=args.device,n_jobs=args.threads)


def prefix(train,period):
    # Use the common stable time/ID ordering with a cap equal to the entire
    # table. TabICLv2 therefore retains every earlier row in each context.
    return common.recent_indices(train,period,len(train))


def verify_env(args):
    if not 1<=args.threads<=2 or (args.device=="cpu" and not args.smoke):
        raise ValueError("Production requires CUDA; CPU is synthetic QA only")
    if args.device=="cuda" and not torch.cuda.is_available(): raise ValueError("No CUDA; no CPU fallback")
    if sha(args.checkpoint)!=WEIGHT: raise ValueError("Wrong TabICLv2 checkpoint checksum")
    if metadata.version("tabicl")!="2.2.0" or metadata.version("scikit-learn")!="1.7.2":
        raise ValueError("Use the independent TabICL 2.2.0 / sklearn 1.7.2 environment")
    if not args.smoke:
        for line in (common.ROOT/"configs/hw4-tabicl-v2-overlay.txt").read_text().splitlines():
            if line and not line.startswith("#"):
                name,version=line.split("==")
                if metadata.version(name)!=version: raise ValueError(f"Isolated version mismatch for {name}")


def probe(args):
    rng=np.random.default_rng(29)
    x=pd.DataFrame(rng.normal(size=(80,7)),columns=[f"v{i}" for i in range(7)])
    x["kind"]=["a","b"]*38+["new"]*4
    x.loc[::7,"v1"]=np.nan
    estimator=model(args)
    estimator.fit(x.iloc[:64],(x.v0.iloc[:64].to_numpy()>0).astype(int))
    p=common.advance(estimator,x.iloc[64:],16)
    if not np.array_equal(p,common.advance(estimator,x.iloc[64:],16)):
        raise ValueError("Native TabICL synthetic repeated forward changed")
    del estimator
    if args.device=="cuda": torch.cuda.empty_cache()
    return {"train_rows":64,"test_rows":16,"columns":8,"repeated_exact":True,
            "sha256":hashlib.sha256(p.tobytes()).hexdigest()}


def job(args,sig,case,period,train,test,frame,cats):
    train_i=prefix(train,period)
    full=period=="full"
    query_i=np.arange(len(train),len(frame)) if full else np.flatnonzero(train.forecast_period.eq(period))
    ids=test.row_id if full else train.iloc[query_i].row_id
    y=train.iloc[train_i].target.to_numpy()
    if set(np.unique(y))!={0,1} or not len(query_i): raise ValueError("Invalid prefix or query")
    identity={"signature":sig,"case":case,"period":period,"train_rows":len(train_i),"predict_rows":len(query_i),
              "training_ids_sha256":digest(train.iloc[train_i].row_id.tolist()),"prediction_ids_sha256":digest(ids.tolist()),
              "features":list(frame),"categorical_indices_for_dtype_confirmation":cats,
              "training_quarter_counts":{k:int(v) for k,v in train.iloc[train_i].forecast_period.value_counts().sort_index().items()}}
    path=args.out/"jobs"/f"{case}__{period}"
    if path.exists():
        rec=load(path/"complete.json")
        if not args.resume or rec["identity"]!=identity or rec["prediction_sha256"]!=sha(path/"prediction.npy"):
            raise ValueError("Changed TabICL intermediate result")
        p=np.load(path/"prediction.npy",allow_pickle=False)
        if p.dtype!=np.float64 or p.shape!=(len(query_i),) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
            raise ValueError("Invalid saved TabICL probabilities")
        print(json.dumps({"event":"reuse","case":case,"period":period}),flush=True)
        return p,rec
    print(json.dumps({"event":"start","case":case,"period":period,"context":len(train_i),"query":len(query_i),
                      "features":len(frame.columns)}),flush=True)
    t=time.monotonic()
    if args.device=="cuda": torch.cuda.reset_peak_memory_stats()
    classifier=model(args)
    classifier.fit(frame.iloc[train_i].copy(),y)
    if list(classifier.classes_)!=[0,1]: raise ValueError("Unexpected binary class order")
    query=frame.iloc[query_i]
    p,chunk,oom=common.predict_with_memory_policy(classifier,query,args.device)
    repeat=case==CASES[0] and period==common.OLD[0]
    if repeat and not np.array_equal(p,common.advance(classifier,query,chunk)):
        raise ValueError("First complete historical validation prediction did not repeat exactly")
    rec={"identity":identity,"seconds":time.monotonic()-t,"n_estimators":classifier.n_estimators,
         "query_chunk":chunk,"prior_cuda_oom_chunks":oom,"repeated_full_forward":repeat,
         "repeated_forward_exact":True if repeat else None,"checkpoint_sha256":WEIGHT,
         "gpu_peak_allocated_bytes":int(torch.cuda.max_memory_allocated()) if args.device=="cuda" else None,
         "gpu_peak_reserved_bytes":int(torch.cuda.max_memory_reserved()) if args.device=="cuda" else None}
    path.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=path.parent,prefix=".pending-") as tmp:
        d=Path(tmp)/"result";d.mkdir()
        np.save(d/"prediction.npy",p,allow_pickle=False)
        if not np.array_equal(np.load(d/"prediction.npy",allow_pickle=False),p): raise ValueError("Prediction persistence changed")
        rec["prediction_sha256"]=sha(d/"prediction.npy")
        write_json(d/"complete.json",rec)
        os.rename(d,path)
    del classifier
    gc.collect()
    if args.device=="cuda": torch.cuda.empty_cache()
    print(json.dumps({"event":"finished","case":case,"period":period,"seconds":round(rec["seconds"],2),
                      "peak_cuda_bytes":rec["gpu_peak_allocated_bytes"],"query_chunk":chunk}),flush=True)
    return p,rec


def run(args):
    verify_env(args)
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    if args.device=="cuda":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8",":16:8"): raise ValueError("Set CUBLAS_WORKSPACE_CONFIG")
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    inputs={n:sha(args.raw/n) for n in common.RAW}
    if not args.smoke and inputs!=common.RAW: raise ValueError("Original raw HW4 input hashes mismatch")
    plan={"format":"hw4-tabicl-v2-offline-1","smoke":args.smoke,"cases":list(CASES),"seeds":[42],
          "old_quarters":list(common.OLD),"recent_prediction_only_quarters":list(common.RECENT),
          "input_sha256":inputs,"source_sha256":{n:sha(common.ROOT/n) for n in SOURCES},"settings":SETTINGS,
          "versions":{n:metadata.version(n) for n in ["tabicl","torch","scikit-learn","numpy","pandas","scipy","joblib","psutil","einops","huggingface-hub"]},
          "python":platform.python_version(),"machine":platform.machine(),"gpu":common.gpu_info(args.device),
          "device":args.device,"threads":args.threads,"checkpoint_sha256":WEIGHT,"tabicl_wheel_sha256":WHEEL,
          "checkpoint_url":"https://huggingface.co/jingang/TabICL/blob/main/tabicl-classifier-v2-20260212.ckpt",
          "license":"BSD-3-Clause","external_pretrained_weights":True,"context":"Every earlier official row, ordered by (forecast_period,row_id). All 21017 official rows for test. No context truncation.",
          "preprocessing":"Native categorical string columns, NaN preserved, estimator default ensemble normalization learned inside each fit.",
          "memory":"Float32 AMP/FA3 off, one ensemble member per batch, auto offload. Full query first; only on explicit CUDA OOM try query chunks 512/128/32 retaining all context and four members.",
          "recent_policy":"User-approved prediction-only diagnostics prior to real GPU run; no scoring or model selection on recent quarters here.",
          "variants_deferred":["raw",".75*context04+.25*TabICL",".75*earlier07+.25*TabICL"],"new_2024q2_q3_metrics":False,"kaggle_uploads":0}
    plan["signature"]=digest(plan);proto=args.out/"protocol.json"
    if proto.exists():
        if not args.resume or load(proto)!=plan: raise ValueError("Changed frozen TabICL provenance")
    else:
        if any(f.name not in {".lock","run.log"}for f in args.out.iterdir()):raise ValueError("New run needs empty output")
        write_json(proto,plan)
    completed=args.out/"completion.json"
    if completed.exists():
        old=load(completed)
        expected_members=set(common.RETURN_FILES)-{"completion.json"}
        if (old["signature"]!=plan["signature"] or set(old["artifacts"])!=expected_members or
            old.get("expected_jobs")!=12 or old.get("jobs")!=12 or old.get("status")!="complete" or
            any(sha(args.out/n)!=h for n,h in old["artifacts"].items())):
            raise ValueError("Changed completed outputs")
        records=load(args.out/"modelinfo.json")["jobs"]
        expected={(name,q) for name in CASES for q in (*common.OLD,"full",*common.RECENT)}
        if len(records)!=12 or {(r["identity"]["case"],r["identity"]["period"]) for r in records}!=expected:
            raise ValueError("Incomplete TabICL job registry")
        for record in records:
            identity=record["identity"]
            job_dir=args.out/"jobs"/f"{identity['case']}__{identity['period']}"
            if (identity["signature"]!=plan["signature"] or load(job_dir/"complete.json")!=record
                    or sha(job_dir/"prediction.npy")!=record["prediction_sha256"]):
                raise ValueError("Changed completed TabICL job")
        archive=common.pack(args.out)
        print(json.dumps({"event":"reuse_complete","reused_jobs":12,"archive":str(archive),"sha256":sha(archive)}),flush=True)
        return old
    attempted=common.offline_runtime(args.out/".offline-state")
    hardware=probe(args)
    train=pd.read_csv(args.raw/"train.csv");test=pd.read_csv(args.raw/"test_student.csv");sample=pd.read_csv(args.raw/"sample_submission.csv")
    if ("target" in test or not train.target.isin([0,1]).all() or any(not v.row_id.is_unique or v.row_id.isna().any()
        for v in [train,test,sample]) or set(train.row_id)&set(test.row_id) or set(test.row_id)!=set(sample.row_id)):
        raise ValueError("Official identity or target mismatch")
    if args.smoke:
        if train.groupby("forecast_period").size().max()>32 or len(test)>32: raise ValueError("Tiny synthetic smoke only")
    elif (len(train),len(test))!=(21017,2246): raise ValueError("Unexpected official raw shapes")
    rows=pd.concat([train.drop(columns="target"),test],ignore_index=True);h=pd.read_csv(args.raw/"inspection_history.csv")
    cause,cats=common.build_features(rows,h);x=pd.concat([cause,common.build_context_features(rows,h)],axis=1)
    ids=["row_id","site_id","forecast_period","target"]
    old=pd.concat([train.loc[train.forecast_period.eq(q),ids] for q in common.OLD],ignore_index=True)
    recent=pd.concat([train.loc[train.forecast_period.eq(q),ids] for q in common.RECENT],ignore_index=True)
    forecast=test[["row_id","forecast_period"]].copy()
    reports,metrics=[],[]
    views=common.views(x,cats)
    for name,old_name in zip(CASES,common.CASES):
        frame,cat_ids=views[old_name]
        oldp,recentp=[],[]
        for q in (*common.OLD,"full",*common.RECENT):
            p,rec=job(args,plan["signature"],name,q,train,test,frame,cat_ids);reports.append(rec)
            if q=="full": forecast[name]=p
            elif q in common.RECENT: recentp.append(p)
            else:
                oldp.append(p); y=train.loc[train.forecast_period.eq(q),"target"]
                metrics.append({"case":name,"seed":42,"forecast_period":q,"n":len(y),"positive":int(y.sum()),
                                "ap":float(average_precision_score(y,p)),"train_rows":rec["identity"]["train_rows"]})
        old[name]=np.concatenate(oldp);recent[name]=np.concatenate(recentp)
    if attempted: raise ValueError("Network operation attempted during local-only inference")
    for filename,frame in [("old_oof.csv",old),("test_predictions.csv",forecast),("recent_predictions.csv",recent),("fold_metrics.csv",pd.DataFrame(metrics))]:
        common.write_csv(args.out/filename,frame)
    write_json(args.out/"modelinfo.json",{"probe":hardware,"runtime_network_attempts":0,"jobs":reports,
        "checkpoint_sha256":WEIGHT,"saved_model_weights":False,"native_training_configuration":SETTINGS})
    write_json(completed,{"status":"complete","signature":plan["signature"],"jobs":len(reports),"expected_jobs":12,
        "old_rows":len(old),"test_rows":len(test),"recent_prediction_rows":len(recent),"new_2024q2_q3_metrics":False,"kaggle_uploads":0,
        "artifacts":{n:sha(args.out/n) for n in common.RETURN_FILES if n!="completion.json"}})
    result=common.pack(args.out)
    print(json.dumps({"event":"complete","archive":str(result),"sha256":sha(result),"jobs":len(reports)}),flush=True)
    return load(completed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("raw","out","checkpoint"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--resume",action="store_true");p.add_argument("--smoke",action="store_true")
    args=p.parse_args();args.checkpoint=args.checkpoint.resolve();args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/".lock").open("a") as lock,threadpool_limits(limits=args.threads):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args)

if __name__=="__main__":main()
