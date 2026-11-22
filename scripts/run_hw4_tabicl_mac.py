"""One registered own-history TabICLv2 ablation on native Mac CPU. No scores before six predictions."""
from __future__ import annotations

import argparse
import fcntl
import gc
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import psutil

LIMIT_BYTES = 12 * 1024**3
MIN_FREE = 2 * 1024**3


def guard(output):
    """Watch the inference subprocess without allocating model tensors in the monitor."""
    command=[sys.executable,"-u",str(Path(__file__).resolve()),*sys.argv[1:],"--worker"]
    proc=subprocess.Popen(command)
    watched=psutil.Process(proc.pid);peak=0;min_free=float("inf");stopped=False
    try:
        while proc.poll() is None:
            try: rss=watched.memory_info().rss
            except psutil.NoSuchProcess: break
            available=psutil.virtual_memory().available
            peak=max(peak,rss);min_free=min(min_free,available)
            if rss>LIMIT_BYTES or available<MIN_FREE:
                stopped=True;proc.terminate();break
            time.sleep(.3)
        try: status=proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill();status=proc.wait()
    finally:
        if proc.poll() is None: proc.terminate()
    receipt=output.parent/(output.name+"-memory-guard.json")
    record={"exit":status,"guard_stopped":stopped,"peak_sampled_worker_rss":peak,
            "min_sampled_host_available":min_free if math.isfinite(min_free) else None}
    temp=receipt.with_suffix(".json.tmp")
    temp.write_text(json.dumps(record,indent=2)+"\n");os.replace(temp,receipt)
    if stopped: raise RuntimeError("Registered safe memory boundary reached")
    return status



# Only the worker imports PyTorch/OpenMP. The watchdog remains inexpensive and
# does not initialize a second set of native thread pools in the same run.
if __name__=="__main__" and not any(flag in sys.argv for flag in ("--worker","--help","-h")):
    early=argparse.ArgumentParser(add_help=False)
    early.add_argument("--out",type=Path,required=True)
    out=early.parse_known_args()[0].out;out.mkdir(parents=True,exist_ok=True)
    raise SystemExit(guard(out))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

import run_hw4_tabicl_v100 as gpu
import run_hw4_tabpfn_v100 as common
from hw4_inspections.classical_features import COMPACT_BASE_COLUMNS
from hw4_inspections.features import build_features
from hw4_inspections.next_round import evaluate_candidates

CASE = "tabicl_mac_own22"
CATS = ("facility_type", "risk_level", "area_id", "forecast_quarter")
FIELDS = tuple(n for n in COMPACT_BASE_COLUMNS if n not in ("forecast_year", "cutoff_gap_days"))
CONFIG = {key: {"safety_factor": .01, "min_batch_size": 1} for key in ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG")}
PERIODS = (*common.OLD, "full", *common.RECENT)
sha, read, save = common.sha, common.load, common.write_json


def view(frame):
    if len(FIELDS) != 22 or len(set(FIELDS)) != 22 or not set(FIELDS).issubset(frame):
        raise ValueError("Expected the exact 22 prespecified columns")
    x = frame[list(FIELDS)].copy()
    for name in CATS:
        x[name] = x[name].fillna("<none>").astype(str)
    numeric = [n for n in FIELDS if n not in CATS]
    x[numeric] = x[numeric].astype(float).replace([np.inf, -np.inf], np.nan)
    return x


def predeclare(args):
    evidence = read(args.resource_profile)
    if (evidence.get("result") != "PASS" or evidence.get("device") != "cpu" or evidence.get("safety") != .01
            or evidence.get("features") != 22 or evidence.get("real_target_metrics") != 0 or
            evidence.get("actual_device") != "cpu" or not evidence.get("finite")):
        raise ValueError("Must provide successful native CPU resource-only evidence for this view")
    cap = int(evidence["actual_train_rows"])
    if not args.smoke and (cap not in (10000,21017) or evidence["actual_query_rows"] != 2246):
        raise ValueError("Only the preregistered full-context or 10k resource fallback is admissible")
    if not args.smoke and platform.system() != "Darwin":
        raise ValueError("Real experiment registered for the Mac")
    if sha(args.checkpoint) != gpu.WEIGHT or metadata.version("tabicl") != "2.2.0" or metadata.version("scikit-learn") != "1.7.2":
        raise ValueError("Unexpected weights or inference library versions")
    if args.threads != 2 or args.bootstrap < (0 if args.smoke else 1):
        raise ValueError("Use two threads and nonzero bootstrap on official data")
    inputs = {n:sha(args.raw/n) for n in common.RAW}
    if not args.smoke and inputs != common.RAW:
        raise ValueError("Official raw inputs changed")
    if not args.smoke and (
            sha(args.anchor_oof)!="0096456a838264be615680be4eead0ed8b449a4182d9f20acd519937e7cc1f7c" or
            sha(args.additive_oof)!="58755d1e12123d1dd18553ac45b5075b3b503cf3e1e5b5d4dd97ce526fc5c9a0"):
        raise ValueError("Supplied historical anchors differ from their verified artifacts")
    sources = ("scripts/run_hw4_tabicl_mac.py", "scripts/run_hw4_tabicl_v100.py", "scripts/run_hw4_tabpfn_v100.py",
               "configs/hw4-tabicl-v2-overlay.txt", "src/hw4_inspections/classical_features.py", "src/hw4_inspections/features.py",
               "src/hw4_inspections/next_round.py", "src/hw4_inspections/classical_analysis.py")
    compared={}
    if not args.smoke:
        for name in ("mps-full","mps-10000","cpu-4000","cpu-10000","cpu-full"):
            probe_path=args.resource_profile.with_name(name+".json")
            if probe_path.exists():
                data=read(probe_path)
                compared[name]={"sha256":sha(probe_path),"device":data.get("device"),"context":data.get("context"),
                    "query_rows":data.get("query_count"),"result":data.get("result"),"seconds":data.get("seconds_one_est"),
                    "rss_bytes":data.get("post_memory",{}).get("process_peak_rss_bytes"),"error":data.get("error")}
    return {"format":"hw4-tabicl-mac-own22-1","resource_comparisons":compared,"cases":[CASE],"seeds":[42],"smoke":args.smoke,"device":"cpu",
        "input_sha256":inputs,"source_sha256":{n:sha(common.ROOT/n) for n in sources},
        "checkpoint_sha256":gpu.WEIGHT,"checkpoint_url":"https://huggingface.co/jingang/TabICL/blob/main/tabicl-classifier-v2-20260212.ckpt",
        "external_pretrained_weights":True,"resource_probe_sha256":sha(args.resource_profile),"resource_evidence":evidence,
        "threads":args.threads,"context_cap":cap,"candidate":"Exactly one new model view, four internal members and fixed seed42.",
        "resource_selection":"Full standard MPS failed at ICL SDPA. Native CPU full prefix preferred when safe under measured resource limits; capped most recent10k only if full CPU infeasible. Selection performed before computing any real AP.",
        "context":"Previous official quarters sorted by (forecast_period,row_id); full test uses available official train rows. Same chosen cap applies on all six jobs.",
        "settings":gpu.SETTINGS,"inference_config":CONFIG,"categories":list(CATS),"fields":list(FIELDS),
        "old_quarters":list(common.OLD),"recent_prediction_only_quarters":list(common.RECENT),"forecast_order":list(PERIODS),
        "scoring":"No AP until all six predictions and checkpoint validations exist; thereafter old folds only. Recent predictions reserved for one joint diagnostic after portfolio freeze.",
        "eligible_forms":["raw",".75*context04+.25*own22",".75*previous07+.25*own22"],"weight_search":False,
        "reference":"Verified old OOF context04 and already frozen75%context04/25%additive (07). Nominal site-cluster bootstrap, no selection correction.",
        "anchor_oof_sha256":sha(args.anchor_oof),"additive_oof_sha256":sha(args.additive_oof),"bootstrap":args.bootstrap,
        "safety_limits":{"max_worker_rss_bytes":LIMIT_BYTES,"min_host_available_bytes":MIN_FREE},
        "machine":platform.machine(),"system":platform.platform(),"python":platform.python_version(),
        "versions":{n:metadata.version(n) for n in ("tabicl","torch","scikit-learn","numpy","pandas","scipy","psutil","joblib","einops")},
        "new_2024q2_q3_metrics":False,"kaggle_uploads":0}


def forecast_job(args,plan,period,train,test,x):
    fit_idx = common.recent_indices(train,period,plan["context_cap"])
    if period == "full":
        val_idx=np.arange(len(train),len(x));ids=test.row_id
    else:
        val_idx=np.flatnonzero(train.forecast_period.eq(period));ids=train.iloc[val_idx].row_id
    identity={"signature":plan["signature"],"case":CASE,"period":period,"train_rows":len(fit_idx),"predict_rows":len(val_idx),
        "training_ids_sha256":common.digest(train.iloc[fit_idx].row_id.tolist()),"prediction_ids_sha256":common.digest(ids.tolist()),
        "training_quarter_counts":{k:int(v) for k,v in train.iloc[fit_idx].forecast_period.value_counts().sort_index().items()}}
    job=args.out/"jobs"/period
    if job.exists():
        record=read(job/"complete.json")
        if not args.resume or record["identity"]!=identity or sha(job/"prediction.npy")!=record["prediction_sha256"]:
            raise ValueError("Changed or unexpected saved own22 job")
        scores=np.load(job/"prediction.npy",allow_pickle=False)
        if scores.dtype!=np.float64 or scores.shape!=(len(val_idx),) or not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():
            raise ValueError("Invalid saved probabilities")
        print(json.dumps({"event":"reused","period":period}),flush=True)
        return scores,record
    from tabicl import TabICLClassifier
    started=time.monotonic()
    print(json.dumps({"event":"native_start","period":period,"context":len(fit_idx),"query":len(val_idx)}),flush=True)
    model=TabICLClassifier(**gpu.SETTINGS,device="cpu",model_path=args.checkpoint,n_jobs=2,inference_config=CONFIG)
    model.fit(x.iloc[fit_idx].copy(),train.target.iloc[fit_idx].to_numpy())
    if list(model.classes_)!=[0,1] or next(model.model_.parameters()).device.type!="cpu":
        raise ValueError("Class or execution-device mismatch")
    future=x.iloc[val_idx]
    values=common.advance(model,future,len(future))
    repeat=period==common.OLD[0]
    if repeat and not np.array_equal(common.advance(model,future,len(future)),values):
        raise ValueError("First complete old-fold replay changed")
    record={"identity":identity,"seconds":time.monotonic()-started,"actual_device":"cpu","ensemble_components":model.n_estimators,
            "repeat_full_forward":repeat,"repeat_full_exact":True if repeat else None,
            "worker_peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"stored_model":False}
    job.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=job.parent,prefix=".partial-") as tmp:
        pending=Path(tmp)/"job";pending.mkdir()
        np.save(pending/"prediction.npy",values,allow_pickle=False)
        if not np.array_equal(np.load(pending/"prediction.npy",allow_pickle=False),values):
            raise ValueError("Saved prediction differs")
        record["prediction_sha256"]=sha(pending/"prediction.npy")
        save(pending/"complete.json",record);os.rename(pending,job)
    print(json.dumps({"event":"native_complete","period":period,"seconds":round(record["seconds"],2),
                      "peak_rss_bytes":record["worker_peak_rss_bytes"]}),flush=True)
    del model
    gc.collect()
    return values,record


def local_scores(args,old):
    anchor=pd.read_csv(args.anchor_oof,float_precision="round_trip")
    prior=pd.read_csv(args.additive_oof,float_precision="round_trip")
    keys=["row_id","site_id","forecast_period","target"]
    for frame in (anchor,prior):
        if (not frame.row_id.is_unique or set(old.row_id)!=set(frame.row_id) or
                not old[keys].reset_index(drop=True).equals(frame.set_index("row_id").loc[old.row_id].reset_index()[keys])):
            raise ValueError("Fixed previous forecasts not strictly aligned to official old rows")
    base=anchor.set_index("row_id").loc[old.row_id,"prediction"].to_numpy()
    older=prior.set_index("row_id").loc[old.row_id]
    if not np.array_equal(base,older.context_3seed):
        raise ValueError("Different context04 anchor in saved additive run")
    seven=older.additive_logit__blend75.to_numpy()
    if not np.array_equal(seven,.75*base+.25*older.additive_logit__raw.to_numpy()):
        raise ValueError("Previous07 is not the exact frozen probability mixture")
    own=old[CASE].to_numpy()
    family={CASE:own,CASE+"__context04_75":.75*base+.25*own,CASE+"__previous07_75":.75*seven+.25*own}
    for ref,vector in [("context_3seed",base),("previous_07",seven)]:
        result=evaluate_candidates(old[keys],{ref:vector,**family},ref,bootstrap=args.bootstrap,seed=20261122)
        summary=result["summary"].copy()
        summary["eligible"] &= summary.worst_quarter_delta.ge(-.005)
        common.write_csv(args.out/f"local-{ref}-summary.csv",summary)
        common.write_csv(args.out/f"local-{ref}-quarters.csv",result["quarter_scores"])


def run(args):
    plan=predeclare(args);plan["signature"]=common.digest(plan)
    if (args.out/"protocol.json").exists():
        if not args.resume or read(args.out/"protocol.json")!=plan:
            raise ValueError("Changed Mac model, source, resource profile or settings")
    else:
        if any(p.name not in (".lock","run.log") for p in args.out.iterdir()):
            raise ValueError("New Mac run requires empty output directory")
        save(args.out/"protocol.json",plan)
    completion=args.out/"completion.json"
    if completion.exists():
        meta=read(completion)
        if (meta["signature"]!=plan["signature"] or meta["jobs"]!=6 or
                any(sha(args.out/n)!=v for n,v in {**meta["artifacts"],**meta["local_artifacts"]}.items())):
            raise ValueError("Changed finished run")
        for record in read(args.out/"modelinfo.json")["jobs"]:
            job=args.out/"jobs"/record["identity"]["period"]
            if read(job/"complete.json")!=record or sha(job/"prediction.npy")!=record["prediction_sha256"]:
                raise ValueError("Changed finished job")
        result=common.pack(args.out)
        print(json.dumps({"event":"reused_complete","jobs":6,"archive_sha256":sha(result)}),flush=True)
        return meta
    attempted=common.offline_runtime(args.out/".offline-state")
    torch.set_num_threads(args.threads);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    train=pd.read_csv(args.raw/"train.csv");test=pd.read_csv(args.raw/"test_student.csv");sample=pd.read_csv(args.raw/"sample_submission.csv")
    if (set(train.row_id)&set(test.row_id) or set(test.row_id)!=set(sample.row_id) or not train.target.isin([0,1]).all() or
        "target" in test or any(f.row_id.isna().any() or not f.row_id.is_unique for f in [train,test,sample])):
        raise ValueError("Official row identity or labels invalid")
    if args.smoke:
        if train.groupby("forecast_period").size().max()>32 or len(test)>32: raise ValueError("Synthetic-only smoke")
    elif (len(train),len(test))!=(21017,2246): raise ValueError("Unexpected real rows")
    rows=pd.concat([train.drop(columns="target"),test],ignore_index=True)
    base,_=build_features(rows,pd.read_csv(args.raw/"inspection_history.csv"))
    x=view(base);del base
    values,records={},[]
    for q in PERIODS:
        values[q],record=forecast_job(args,plan,q,train,test,x);records.append(record)
    if attempted: raise ValueError("Network attempt during offline forecasts")
    if len(records)!=6: raise AssertionError("No scoring until every context is complete")
    keys=["row_id","site_id","forecast_period","target"]
    old=pd.concat([train.loc[train.forecast_period.eq(q),keys] for q in common.OLD],ignore_index=True)
    old[CASE]=np.concatenate([values[q]for q in common.OLD])
    recent=pd.concat([train.loc[train.forecast_period.eq(q),keys] for q in common.RECENT],ignore_index=True)
    recent[CASE]=np.concatenate([values[q]for q in common.RECENT])
    forecast=test[["row_id","forecast_period"]].assign(**{CASE:values["full"]})
    metrics=[]
    for q in common.OLD:
        y=train.loc[train.forecast_period.eq(q),"target"]
        metrics.append({"case":CASE,"seed":42,"forecast_period":q,"n":len(y),"positive":int(y.sum()),
            "ap":float(average_precision_score(y,values[q])),"train_rows":next(r["identity"]["train_rows"] for r in records if r["identity"]["period"]==q)})
    for name,frame in [("old_oof.csv",old),("recent_predictions.csv",recent),("test_predictions.csv",forecast),("fold_metrics.csv",pd.DataFrame(metrics))]:
        common.write_csv(args.out/name,frame)
    local_scores(args,old)
    save(args.out/"modelinfo.json",{"jobs":records,"checkpoint_sha256":gpu.WEIGHT,"scored_after_all_six_jobs":True,
                                   "runtime_network_attempts":0,"recent_metrics_computed":0})
    tracked=[n for n in common.RETURN_FILES if n!="completion.json"]
    local_artifacts={p.name:sha(p) for p in sorted(args.out.glob("local-*.csv"))}
    save(completion,{"status":"complete","signature":plan["signature"],"jobs":6,"expected_jobs":6,"old_rows":len(old),
         "test_rows":len(test),"recent_prediction_rows":len(recent),"artifacts":{n:sha(args.out/n) for n in tracked},
         "local_artifacts":local_artifacts,"case":CASE,"new_2024q2_q3_metrics":False,"kaggle_uploads":0})
    archive=common.pack(args.out)
    print(json.dumps({"event":"complete","jobs":6,"archive_sha256":sha(archive)}),flush=True)
    return read(completion)




def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ["out","raw","checkpoint","resource-profile","anchor-oof","additive-oof"]:p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--bootstrap",type=int,default=2000);p.add_argument("--threads",type=int,default=2)
    p.add_argument("--smoke",action="store_true");p.add_argument("--resume",action="store_true");p.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);args.checkpoint=args.checkpoint.resolve()
    with (args.out/".lock").open("a") as lock,threadpool_limits(limits=2):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args)

if __name__=="__main__":main()
