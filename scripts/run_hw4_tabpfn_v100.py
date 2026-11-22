"""Standalone offline V2 diagnostic producer. Built with PriorLabs-TabPFN."""
from __future__ import annotations

import argparse
import fcntl
import gc
import gzip
import hashlib
import io
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
import tarfile
from tempfile import TemporaryDirectory
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from hw4_inspections.classical_features import select_feature_columns
from hw4_inspections.extensions import build_context_features
from hw4_inspections.features import build_features

CASES = ("pfn_v2_compact94","pfn_v2_full346")
OLD = ("2023-07-01","2023-10-01","2024-01-01")
RECENT = ("2024-04-01","2024-07-01")
RAW = {"train.csv":"30b6acd8c9f2eec4259a775b098017934ba5497cb37689eb00b639433472f0d9",
       "test_student.csv":"1014585cdbdb45e6b1ce7bcb5723700aa025826975a6c5c6910ed5ce39f6f643",
       "inspection_history.csv":"9637ec0279d9cf521795268a0f44099d7d51a795b5155a0d78405a770966afc7",
       "sample_submission.csv":"2d9b5084b510302cd027d46e0c5f9b8e986d48f417d9a582a8b5ec2b3c953b42"}
CHECKPOINT = "cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49"
WHEEL = "9aa3a0df8ecebcae5f426b1fec9bb2e2a690880bc8ad079191e8d959bf0f157b"
RETURN_FILES = ("protocol.json","completion.json","old_oof.csv","test_predictions.csv","fold_metrics.csv","modelinfo.json","recent_predictions.csv")
SOURCES = ("scripts/run_hw4_tabpfn_v100.py","scripts/run_hw4_tabpfn_v100.sh","configs/hw4-tabpfn-v2-overlay.txt",
           "src/hw4_inspections/classical_features.py","src/hw4_inspections/features.py","src/hw4_inspections/extensions.py")


def sha(path):
    with Path(path).open("rb") as reader:
        return hashlib.file_digest(reader,"sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def write_json(path,value):
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n")
    os.replace(temp,path)


def load(path):
    return json.loads(path.read_text())


def write_csv(path,frame):
    tmp=path.with_suffix(".csv.tmp")
    frame.to_csv(tmp,index=False)
    pd.testing.assert_frame_equal(frame.reset_index(drop=True),pd.read_csv(tmp,float_precision="round_trip"),
                                  check_exact=True,check_dtype=False)
    os.replace(tmp,path)


def verify_runtime(checkpoint,device,smoke):
    if sha(checkpoint)!=CHECKPOINT:
        raise ValueError("Require the original public V2 checkpoint with the registered SHA-256")
    for n,v in [("tabpfn","2.2.1"),("tabpfn-common-utils","0.1.9"),("scikit-learn","1.6.1")]:
        if metadata.version(n)!=v:
            raise ValueError(f"Require {n}=={v}; use the isolated launcher")
    if device=="cpu" and not smoke:
        raise ValueError("Production run requires CUDA; CPU is for small synthetic QA")
    if device=="cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA not available; no CPU fallback")
    if not smoke:
        # Check exact setup lock before spending GPU time.
        for line in (ROOT/"configs/hw4-tabpfn-v2-overlay.txt").read_text().splitlines():
            if line and not line.startswith("#"):
                name,version=line.split("==")
                if metadata.version(name)!=version:
                    raise ValueError(f"Isolated package drift: {name}, expected {version}")


def recent_indices(train,period,cap):
    if period!="full" and period not in (*OLD,*RECENT):
        raise ValueError("Only the fixed old/recent quarters and full are permitted")
    if not train.row_id.is_unique or train.row_id.isna().any() or cap<=0:
        raise ValueError("Invalid training IDs or cap")
    prior=train if period=="full" else train.loc[train.forecast_period.lt(period)]
    if prior.empty:
        raise ValueError("Empty past training block")
    chosen=prior[["forecast_period","row_id"]].sort_values(["forecast_period","row_id"],kind="stable").tail(cap)
    return chosen.index.to_numpy(dtype=int)


def views(frame,cats):
    compact=[c for c in select_feature_columns(frame,"compact_context") if c not in ("forecast_year","cutoff_gap_days")]
    out={}
    for case,columns,categories,size in [(CASES[0],compact,["facility_type","risk_level","area_id","forecast_quarter"],94),
                                           (CASES[1],list(frame),list(cats),346)]:
        if len(columns)!=size or not set(categories)<=set(columns):
            raise ValueError("Unexpected causal feature-view schema")
        selected=frame[columns].copy()
        for col in categories:
            selected[col]=selected[col].fillna("<none>").astype(str)
        numeric=[col for col in columns if col not in categories]
        selected[numeric]=selected[numeric].astype(float).replace([np.inf,-np.inf],np.nan)
        out[case]=(selected,[columns.index(c) for c in categories])
    return out


def offline_runtime(state):
    os.environ.update(TABPFN_DISABLE_TELEMETRY="1",HF_HUB_DISABLE_TELEMETRY="1",HF_HUB_OFFLINE="1",
                      TABPFN_STATE_DIR=str(state),TABPFN_MODEL_CACHE_DIR=str(state/"model-cache"))
    attempted=[]
    def block(event,arguments):
        if event in {"socket.connect","socket.connect_ex","socket.getaddrinfo"}:
            attempted.append(event)
            raise RuntimeError(f"Network operation during offline prediction: {event}")
    sys.addaudithook(block)
    return attempted


def model_for(checkpoint,cats,device):
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier(n_estimators=4,categorical_features_indices=cats,model_path=checkpoint,
        device=device,memory_saving_mode=True,fit_mode="fit_preprocessors",inference_precision=torch.float32,
        random_state=42,n_jobs=1,ignore_pretraining_limits=False,balance_probabilities=False)


def gpu_info(device):
    if device=="cpu":
        return None
    p=torch.cuda.get_device_properties(0)
    return {"name":p.name,"capability":[p.major,p.minor],"total_vram_bytes":int(p.total_memory),
            "cuda_build":torch.version.cuda,"cudnn":torch.backends.cudnn.version()}


def advance(clf,xx,chunk):
    vals=[]
    with torch.inference_mode():
        for first in range(0,len(xx),chunk):
            p=np.asarray(clf.predict_proba(xx.iloc[first:first+chunk].copy())[:,1],dtype=np.float64)
            vals.append(p)
    scores=np.concatenate(vals)
    if scores.shape!=(len(xx),) or not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():
        raise ValueError("Invalid TabPFN class-1 probabilities")
    return scores


def predict_with_memory_policy(clf,xx,device):
    attempts=[]
    chunks=[len(xx),*[v for v in (512,128,32) if v<len(xx)]]
    for chunk in chunks:
        try:
            prob=advance(clf,xx,chunk)
            return prob,chunk,attempts
        except torch.cuda.OutOfMemoryError:
            if device!="cuda":
                raise
            attempts.append(chunk)
        gc.collect()
        torch.cuda.empty_cache()
    raise RuntimeError(f"V2 inference exhausted CUDA memory at query sizes {attempts}; 10k context was held fixed")


def preflight(checkpoint,device):
    # Also probes the actual attention implementation and fp32 on this device.
    rng=np.random.default_rng(144)
    x=pd.DataFrame(rng.normal(size=(40,4)),columns=["a","b","c","d"])
    x.loc[::7,"b"]=np.nan
    x["kind"]=["a","b"]*19+["new","new"]
    y=np.arange(32)%2
    model=model_for(checkpoint,[4],device)
    model.fit(x.iloc[:32].copy(),y)
    a=advance(model,x.iloc[32:],8)
    b=advance(model,x.iloc[32:],8)
    if not np.array_equal(a,b):
        raise ValueError("Synthetic same-device repeated V2 forward changed")
    del model
    if device=="cuda": torch.cuda.empty_cache()
    return {"train_rows":32,"predict_rows":8,"feature_count":5,"device":device,"precision":"float32",
            "repeat_forward_exact":True,"output_sha256":hashlib.sha256(a.tobytes()).hexdigest()}


def run_job(args,sig,case,period,train,test,frame,cats):
    full=period=="full"
    train_i=recent_indices(train,period,64 if args.smoke else 10000)
    query_i=np.arange(len(train),len(frame)) if full else np.flatnonzero(train.forecast_period.eq(period))
    query_ids=test.row_id if full else train.iloc[query_i].row_id
    y=train.iloc[train_i].target.to_numpy()
    if set(np.unique(y))!={0,1} or not len(query_i):
        raise ValueError("Model context needs both classes and nonempty query batch")
    identity={"signature":sig,"case":case,"period":period,"seed":42,"n_estimators":4,"train_rows":len(train_i),
              "available_train_rows":len(train) if full else int(train.forecast_period.lt(period).sum()),
              "training_ids_sha256":digest(train.iloc[train_i].row_id.tolist()),"prediction_ids_sha256":digest(query_ids.tolist()),
              "predict_rows":len(query_i),"feature_columns":list(frame),"categorical_indices":cats,
              "training_quarter_counts":{str(k):int(v) for k,v in train.iloc[train_i].forecast_period.value_counts().sort_index().items()}}
    path=args.out/"jobs"/f"{case}__{period}"
    if path.exists():
        record=load(path/"complete.json")
        if not args.resume or record["identity"]!=identity or record["prediction_sha256"]!=sha(path/"prediction.npy"):
            raise ValueError("Changed or unapproved previous diagnostic job")
        value=np.load(path/"prediction.npy",allow_pickle=False)
        if (value.dtype!=np.float64 or value.shape!=(len(query_i),) or not np.isfinite(value).all()
                or ((value<0)|(value>1)).any()):
            raise ValueError("Corrupt diagnostics")
        print(json.dumps({"event":"reuse","case":case,"period":period}),flush=True)
        return value,record
    print(json.dumps({"event":"predict_start","case":case,"period":period,"context":len(train_i),
                      "queries":len(query_i),"features":len(frame.columns)}),flush=True)
    started=time.monotonic()
    if args.device=="cuda": torch.cuda.reset_peak_memory_stats()
    model=model_for(args.checkpoint,cats,args.device)
    model.fit(frame.iloc[train_i].copy(),y)
    if list(model.classes_)!=[0,1]:
        raise ValueError("Unexpected class order")
    future=frame.iloc[query_i]
    values,chunk,oom_attempts=predict_with_memory_policy(model,future,args.device)
    repeat=case==CASES[0] and period==OLD[0]
    if repeat:
        # Matching input block sizes avoid conflating roundoff from batch shapes
        # with reproducibility. All other jobs retain a checked raw array.
        replay=advance(model,future,chunk)
        if not np.array_equal(replay,values):
            raise ValueError("First complete old-fold forward did not reproduce exactly")
    record={"identity":identity,"seconds":time.monotonic()-started,"successful_query_chunk":chunk,
            "prior_cuda_oom_chunks":oom_attempts,"repeated_full_forward":repeat,
            "repeated_forward_exact":True if repeat else None,"fit_mode":"fit_preprocessors",
            "gpu_peak_allocated_bytes":int(torch.cuda.max_memory_allocated()) if args.device=="cuda" else None,
            "gpu_peak_reserved_bytes":int(torch.cuda.max_memory_reserved()) if args.device=="cuda" else None,
            "actual_ensemble_estimators":len(model.executor_.ensemble_configs),
            "stored_fitted_weights":False,"external_base_checkpoint_sha256":CHECKPOINT}
    path.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=path.parent,prefix=".pending-") as tmp:
        d=Path(tmp)/"job";d.mkdir()
        np.save(d/"prediction.npy",values,allow_pickle=False)
        if not np.array_equal(np.load(d/"prediction.npy",allow_pickle=False),values):
            raise ValueError("Prediction serialization changed")
        record["prediction_sha256"]=sha(d/"prediction.npy")
        write_json(d/"complete.json",record)
        os.rename(d,path)
    del model
    gc.collect()
    if args.device=="cuda": torch.cuda.empty_cache()
    print(json.dumps({"event":"predict_complete","case":case,"period":period,"seconds":round(record["seconds"],2),
                      "gpu_peak_allocated_bytes":record["gpu_peak_allocated_bytes"],"chunk":chunk}),flush=True)
    return values,record


def pack(out):
    archive=out/"return_results.tar.gz"
    temp=out/"return_results.tar.gz.tmp"
    with temp.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb",fileobj=raw,mtime=0) as stream:
            with tarfile.open(fileobj=stream,mode="w",format=tarfile.USTAR_FORMAT) as tar:
                for name in RETURN_FILES:
                    data=(out/name).read_bytes()
                    entry=tarfile.TarInfo(name)
                    entry.size=len(data)
                    entry.mode=0o644
                    entry.mtime=0
                    tar.addfile(entry,io.BytesIO(data))
    with tarfile.open(temp,"r:gz") as tar:
        members=tar.getmembers()
        if [m.name for m in members]!=list(RETURN_FILES) or any(not m.isreg() or m.linkname or m.size>16<<20 for m in members):
            raise AssertionError("Unexpected return artifact")
        for member in members:
            if hashlib.sha256(tar.extractfile(member).read()).hexdigest()!=sha(out/member.name):
                raise AssertionError("Archived member differs")
    os.replace(temp,archive)
    return archive


def run(args):
    if not 1<=args.threads<=2:
        raise ValueError("Use one or two CPU preprocessing threads")
    verify_runtime(args.checkpoint,args.device,args.smoke)
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    if args.device=="cuda":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8",":16:8"):
            raise ValueError("Set CUBLAS_WORKSPACE_CONFIG before CUDA initialization")
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    hashes={n:sha(args.raw/n) for n in RAW}
    if not args.smoke and hashes!=RAW:
        raise ValueError("Source data differ from the four verified official files")
    plan={"format":"hw4-tabpfn-v2-offline-1","smoke":args.smoke,"cases":list(CASES),"seeds":[42],
          "old_quarters":list(OLD),"recent_prediction_only_quarters":list(RECENT),"input_sha256":hashes,"source_sha256":{n:sha(ROOT/n) for n in SOURCES},
          "versions":{n:metadata.version(n) for n in ["tabpfn","tabpfn-common-utils","torch","scikit-learn","numpy",
                       "pandas","scipy","joblib","einops","pydantic","huggingface-hub"]},
          "python":platform.python_version(),"machine":platform.machine(),"device":args.device,"gpu":gpu_info(args.device),
          "threads":args.threads,"tabpfn_wheel_sha256":WHEEL,"checkpoint_sha256":CHECKPOINT,
          "checkpoint_url":"https://huggingface.co/Prior-Labs/TabPFN-v2-clf/blob/main/tabpfn-v2-classifier-v2_default.ckpt",
          "license":"https://huggingface.co/Prior-Labs/TabPFN-v2-clf/blob/main/LICENSE.txt",
          "attribution":"Built with PriorLabs-TabPFN","external_pretrained_weights":True,
          "context":"Available official past rows ordered (forecast_period,row_id) ascending; take tail min(10000,n). Full forecast uses same cap across all official train rows. No selection by labels.",
          "context_cap":64 if args.smoke else 10000,"ensemble_estimators":4,"random_state":42,"memory_saving_mode":True,
          "fit_mode":"fit_preprocessors","inference_precision":"torch.float32","n_jobs":1,
          "classification":"native class-1 probabilities, sklearn training vocabulary learned within each context",
          "query_batch":"Full horizon first; explicit CUDA OOM only retry 512,128,32 query rows with frozen original context.",
          "replay":"Synthetic device probe twice; first full actual compact old-development fold twice, other jobs checked by array roundtrip and hashes.",
          "variants_deferred":["raw",".75*context04+.25*TabPFN",".75*earlier07+.25*TabPFN"],
          "recent_policy":"User approved one later diagnostic after finalist freeze. Before any real GPU job, two recent quarters added as prediction-only artifacts. This producer computes no score for these quarters, and never tunes using them.",
          "kaggle_uploads":0,"new_2024q2_q3_metrics":False,"full_policy":"All two diagnostic test forecasts produced to permit exact local gate. No public score or Q2/Q3 metric drives selection."}
    plan["signature"]=digest(plan)
    protocol=args.out/"protocol.json"
    if protocol.exists():
        if not args.resume or load(protocol)!=plan: raise ValueError("Changed source, data, environment or configuration")
    else:
        if any(f.name not in {".lock","run.log"} for f in args.out.iterdir()): raise ValueError("New producer requires empty output")
        write_json(protocol,plan)
    completion=args.out/"completion.json"
    if completion.exists():
        prior=load(completion)
        if prior["signature"]!=plan["signature"] or any(sha(args.out/n)!=h for n,h in prior["artifacts"].items()):
            raise ValueError("Changed completed return bundle")
    attempted=offline_runtime(args.out/".pfn-state")
    probe=preflight(args.checkpoint,args.device)
    print(json.dumps({"event":"device_probe_ok","device":args.device,"gpu":plan["gpu"],"repeat_exact":True}),flush=True)
    train=pd.read_csv(args.raw/"train.csv")
    test=pd.read_csv(args.raw/"test_student.csv")
    sample=pd.read_csv(args.raw/"sample_submission.csv")
    if (not train.target.isin([0,1]).all() or "target" in test or any(not v.row_id.is_unique or v.row_id.isna().any()
            for v in [train,test,sample]) or set(train.row_id)&set(test.row_id) or set(sample.row_id)!=set(test.row_id)):
        raise ValueError("Invalid official training, test or sample IDs")
    if args.smoke:
        if train.groupby("forecast_period").size().max()>32 or len(test)>32:
            raise ValueError("Synthetic smoke allows at most 32 rows per quarter/test set")
    elif len(train)!=21017 or len(test)!=2246: raise ValueError("Unexpected real sizes")
    rows=pd.concat([train.drop(columns="target"),test],ignore_index=True)
    hist=pd.read_csv(args.raw/"inspection_history.csv")
    x,cats=build_features(rows,hist)
    frame=pd.concat([x,build_context_features(rows,hist)],axis=1)
    old=pd.concat([train.loc[train.forecast_period.eq(q),["row_id","site_id","forecast_period","target"]] for q in OLD],ignore_index=True)
    recent=pd.concat([train.loc[train.forecast_period.eq(q),["row_id","site_id","forecast_period","target"]] for q in RECENT],ignore_index=True)
    forecast=test[["row_id","forecast_period"]].copy()
    records,metrics=[],[]
    for case,(inputs,categories) in views(frame,cats).items():
        oof=[]
        recent_values=[]
        for q in [*OLD,"full",*RECENT]:
            values,record=run_job(args,plan["signature"],case,q,train,test,inputs,categories)
            records.append(record)
            if q=="full":
                forecast[case]=values
            elif q in RECENT:
                recent_values.append(values)
            else:
                y=train.loc[train.forecast_period.eq(q),"target"]
                oof.append(values)
                metrics.append({"case":case,"seed":42,"forecast_period":q,"n":len(y),"positive":int(y.sum()),
                                "ap":float(average_precision_score(y,values)),"train_rows":record["identity"]["train_rows"]})
        old[case]=np.concatenate(oof)
        recent[case]=np.concatenate(recent_values)
    if attempted:
        raise RuntimeError("Offline job encountered blocked network attempts")
    write_csv(args.out/"old_oof.csv",old)
    write_csv(args.out/"test_predictions.csv",forecast)
    write_csv(args.out/"recent_predictions.csv",recent)
    write_csv(args.out/"fold_metrics.csv",pd.DataFrame(metrics))
    write_json(args.out/"modelinfo.json",{"probe":probe,"runtime_network_attempts":len(attempted),"jobs":records,
        "checkpoint_sha256":CHECKPOINT,"attribution":plan["attribution"],"numerical_state_saved":"predictions only, no fitted models or raw data in archive"})
    write_json(completion,{"status":"complete","signature":plan["signature"],"old_rows":len(old),"test_rows":len(forecast),
        "recent_prediction_rows":len(recent),"jobs":len(records),"expected_jobs":12,"artifacts":{n:sha(args.out/n) for n in RETURN_FILES if n!="completion.json"},
        "new_2024q2_q3_metrics":False,"kaggle_uploads":0,"exports":"diagnostic, requires matched local admission"})
    archive=pack(args.out)
    print(json.dumps({"event":"complete","archive":str(archive),"sha256":sha(archive),"jobs":len(records)}),flush=True)
    return load(completion)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ["raw","out","checkpoint"]: parser.add_argument("--"+n,type=Path,required=True)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    args.checkpoint=args.checkpoint.resolve()
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/".lock").open("a") as lock, threadpool_limits(limits=args.threads):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            run(args)
        except Exception as exc:
            print(json.dumps({"event":"failed","error_type":type(exc).__name__,"error":str(exc)}),flush=True)
            raise


if __name__=="__main__":
    main()
