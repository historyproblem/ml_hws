"""Safe native32 seedbag consumer; fixed winner15 comparisons on OLD3 only."""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
import run_hw4_tabicl_seedbag as producer
import run_hw4_tabpfn_v100 as common
from safe_archives import read_known_tar


def read(value):
    return pd.read_csv(BytesIO(value) if isinstance(value,bytes) else value,float_precision="round_trip")


def scores(values, n):
    values=np.asarray(values,dtype=np.float64)
    if values.shape!=(n,) or not np.isfinite(values).all() or ((values<0)|(values>1)).any():raise ValueError("Invalid probabilityvector")
    return values


def odds50(c, i):
    return expit(.5*logit(np.clip(c,1e-6,1-1e-6))+.5*logit(np.clip(i,1e-6,1-1e-6)))


def align_submission(path, ids):
    frame=read(path)
    if list(frame)!=["row_id","target"] or not frame.row_id.is_unique or set(frame.row_id)!=set(ids):raise ValueError("Different submission IDs/schema")
    return scores(frame.set_index("row_id").loc[ids,"target"].to_numpy(),len(ids))


def verify(args):
    files=read_known_tar(args.archive,producer.FILES)
    p,c,info=[json.loads(files[n]) for n in ["protocol.json","completion.json","modelinfo.json"]]
    if (common.digest({k:v for k,v in p.items() if k!="signature"})!=p["signature"] or c["signature"]!=p["signature"]
        or p["format"]!="hw4-tabicl-seedbag32-1" or p["seeds"]!=list(producer.SEEDS) or p["periods"]!=list(producer.PERIODS)
        or p["old_quarters"]!=list(common.OLD) or p["recent_periods"]!=[] or p["device"]!="cuda" or p["threads"]!=2
        or p["settings_by_seed"]!={str(s):producer.settings(s) for s in producer.SEEDS}
        or p["predeclared_forms"]!=producer.FORMS or p["gate"]!=producer.GATE or p["clip"]!=1e-6
        or p["fixed_parent_sha256"]!=producer.PARENTS or p["baseline42_artifact_sha256"]!=producer.BASELINE
        or p["checkpoint_sha256"]!=producer.original.WEIGHT or p["input_sha256"]!=common.RAW
        or p["versions"]["tabicl"]!="2.2.0" or p["versions"]["scikit-learn"]!="1.7.2"
        or p["new_2024q2_q3_metrics"] or p["kaggle_uploads"] or p["new_optimizer_fit"]!=0
        or p["new_jobs"]!=8 or p["reused_baseline42_jobs"]!=4
        or c["status"]!="complete" or c["new_jobs"]!=8 or c["reused42_jobs"]!=4
        or c["new_2024q2_q3_metrics"] or c["kaggle_uploads"] or info["runtime_network_attempts"]!=0
        or info["checkpoint_sha256"]!=producer.original.WEIGHT):raise ValueError("Different predeclared seedbagprotocol")
    if set(p["source_sha256"])!=set(producer.SOURCES):raise ValueError("Different executed source set")
    for name,h in p["source_sha256"].items():
        if common.sha(ROOT/name)!=h:raise ValueError("Changed executed source: "+name)
    for name,h in common.RAW.items():
        if common.sha(args.raw/name)!=h:raise ValueError("Different official rawfile")
    if set(c["artifacts"])!=set(producer.FILES)-{"completion.json"}:raise ValueError("Different archiveartifactset")
    for name,h in c["artifacts"].items():
        if hashlib.sha256(files[name]).hexdigest()!=h:raise ValueError("Changed archiveartifact")
    bp,bi=producer.baseline(args.baseline32_dir)
    if p["baseline42_signature"]!=bp["signature"] or info["baseline42_jobs"]!=bi["jobs"]:raise ValueError("Wrong reused seed42 registry")
    for name,arg in [("duo_old_oof",args.duo_oof),("context04_test",args.context_test),("native4_test",args.native4_test),("winner15_test",args.winner_test)]:
        if common.sha(arg)!=producer.PARENTS[name]:raise ValueError("Changed fixedwinner15parent: "+name)
    train,test,sample=[read(args.raw/n) for n in ["train.csv","test_student.csv","sample_submission.csv"]]
    keys=["row_id","site_id","forecast_period","target"]
    meta=pd.concat([train.loc[train.forecast_period.eq(q),keys] for q in common.OLD],ignore_index=True)
    old,forecast,metrics=[read(files[n]) for n in ["old_oof.csv","test_predictions.csv","fold_metrics.csv"]]
    for frame,want in [(old,meta),(forecast,test[["row_id","forecast_period"]])]:
        if list(frame)!=[*want,*producer.COLS] or not frame.row_id.is_unique or frame.row_id.isna().any():raise ValueError("Wrong forecastschema/IDs")
        pd.testing.assert_frame_equal(frame[list(want)],want,check_dtype=False,check_exact=True)
        for col in producer.COLS:scores(frame[col],len(frame))
        if not np.array_equal(frame[producer.COLS[-1]],np.mean(frame[list(producer.COLS[:3])].to_numpy(),axis=1)):raise ValueError("Not exact3seedprobabilitymean")
    if c["old_rows"]!=len(old) or c["test_rows"]!=len(forecast) or len(old)!=5815 or len(forecast)!=2246:raise ValueError("Wrong totals")
    for frame,name in [(old,"old_oof.csv"),(forecast,"test_predictions.csv")]:
        base=read(args.baseline32_dir/name)
        if not np.array_equal(frame["icl32_s42"],base[producer.prior32.CASE]):raise ValueError("Reused seed42 vector changed")
    oldjobs={j["identity"]["period"]:j for j in bi["jobs"]}
    expected={(s,q) for s in producer.SEEDS for q in producer.PERIODS}
    if len(info["jobs"])!=8 or {(j["identity"]["seed"],j["identity"]["period"]) for j in info["jobs"]}!=expected:raise ValueError("Duplicate/missingnewcontexts")
    for job in info["jobs"]:
        identity=job["identity"];s,q=identity["seed"],identity["period"]
        chosen=train if q=="full" else train.loc[train.forecast_period.lt(q)]
        chosen=chosen.sort_values(["forecast_period","row_id"],kind="stable")
        predicted=forecast if q=="full" else old.loc[old.forecast_period.eq(q)]
        base=oldjobs[q]["identity"]
        if (identity["signature"]!=p["signature"] or identity["settings"]!=producer.settings(s) or identity["case"]!=producer.prior32.CASE
            or identity["train_rows"]!=len(chosen) or identity["predict_rows"]!=len(predicted)
            or identity["training_ids_sha256"]!=common.digest(chosen.row_id.tolist()) or identity["prediction_ids_sha256"]!=common.digest(predicted.row_id.tolist())
            or identity["training_quarter_counts"]!={n:int(k) for n,k in chosen.forecast_period.value_counts().sort_index().items()}
            or identity["features"]!=base["features"] or identity["categorical_indices"]!=base["categorical_indices_for_dtype_confirmation"]
            or job["checkpoint_sha256"]!=producer.original.WEIGHT or job["actual_native_components"]!=32 or job["actual_random_state"]!=s):raise ValueError("Different actualseed/context/memberconfiguration")
        values=predicted[f"icl32_s{s}"].to_numpy(dtype=np.float64);buf=BytesIO();np.save(buf,values,allow_pickle=False)
        if hashlib.sha256(buf.getvalue()).hexdigest()!=job["prediction_sha256"]:raise ValueError("Changed persistedseedvector")
        if q==common.OLD[0]:
            if not job["repeated_full_forward"] or job["repeated_forward_exact"] is not True:raise ValueError("Missing perseed firstOLDexactrepeat")
        elif job["repeated_full_forward"] or job["repeated_forward_exact"] is not None:raise ValueError("Unexpected replaymetadata")
        if (not isinstance(job["query_chunk"],int) or not 1<=job["query_chunk"]<=len(predicted)
            or not np.isfinite(job["seconds"]) or job["seconds"]<0
            or not 0<job["gpu_peak_allocated_bytes"]<=job["gpu_peak_reserved_bytes"]<=p["gpu"]["total_vram_bytes"]):raise ValueError("Invalid runtimemetadata")
    expected_metrics={(q,col) for q in common.OLD for col in producer.COLS[1:]}
    if len(metrics)!=9 or set(zip(metrics.forecast_period,metrics.model))!=expected_metrics:raise ValueError("Wrong OLDmetricscope")
    for row in metrics.itertuples():
        g=old.loc[old.forecast_period.eq(row.forecast_period)]
        if (row.n!=len(g) or row.positive!=int(g.target.sum()) or not np.isfinite(row.ap)
            or abs(row.ap-average_precision_score(g.target,g[row.model]))>1e-12):raise ValueError("WrongreportedOLDAP/counts")
    duo=read(args.duo_oof)
    pd.testing.assert_frame_equal(duo[keys],meta,check_dtype=False,check_exact=True)
    old_reference=odds50(duo.context04.to_numpy(),duo.icl94_raw.to_numpy())
    if not np.allclose(old_reference,duo.context50_icl50_odds.to_numpy(),rtol=0,atol=2e-16):raise ValueError("Winner15OLDformula differs")
    ct=align_submission(args.context_test,test.row_id);it=align_submission(args.native4_test,test.row_id);wt=align_submission(args.winner_test,test.row_id)
    if not np.allclose(odds50(ct,it),wt,rtol=0,atol=2e-16):raise ValueError("Winner15testformula differs")
    old_variants={"seedbag":odds50(duo.context04.to_numpy(),old.icl32_probmean3.to_numpy()),
        "retain4_seedbag":odds50(duo.context04.to_numpy(),.5*duo.icl94_raw.to_numpy()+.5*old.icl32_probmean3.to_numpy())}
    test_variants={"seedbag":odds50(ct,forecast.icl32_probmean3.to_numpy()),
        "retain4_seedbag":odds50(ct,.5*it+.5*forecast.icl32_probmean3.to_numpy())}
    return files,p,meta,old,forecast,test,sample,old_reference,old_variants,test_variants


def run(args):
    files,p,meta,old,forecast,test,sample,reference,variants,test_variants=verify(args)
    if args.out.exists() and any(args.out.iterdir()):raise ValueError("Use a new emptyconsumeroutput")
    rows=[]
    for q in common.OLD:
        mask=meta.forecast_period.eq(q).to_numpy();target=meta.target.to_numpy()[mask]
        ref=float(average_precision_score(target,reference[mask]))
        rows.append({"forecast_period":q,"model":"winner15","ap":ref,"delta_vs_winner15":0.})
        for name,values in variants.items():
            ap=float(average_precision_score(target,values[mask]));rows.append({"forecast_period":q,"model":name,"ap":ap,"delta_vs_winner15":ap-ref})
    quarter=pd.DataFrame(rows);summary=[]
    for name in producer.FORMS:
        subset=quarter.loc[quarter.model.eq(name)].set_index("forecast_period");d=subset.delta_vs_winner15
        eligible=bool(d.mean()>0 and (d>=0).sum()>=2 and d.min()>=-.002 and d.loc["2023-10-01"]>=0)
        summary.append({"model":name,"mean_ap":float(subset.ap.mean()),"mean_delta_vs_winner15":float(d.mean()),
            "nonnegative_quarters":int((d>=0).sum()),"worst_delta":float(d.min()),"q4_delta":float(d.loc["2023-10-01"]),"eligible":eligible})
    args.out.mkdir(parents=True,exist_ok=True)
    for name,value in files.items():(args.out/name).write_bytes(value)
    common.write_csv(args.out/"final-blend-old-quarter-metrics.csv",quarter)
    common.write_csv(args.out/"final-blend-summary.csv",pd.DataFrame(summary))
    outputs=[]
    for item in summary:
        if not item["eligible"]:continue
        name=item["model"];number=17 if name=="seedbag" else 18
        values=scores(test_variants[name],len(test));frame=pd.DataFrame({"row_id":test.row_id,"target":values}).set_index("row_id").loc[sample.row_id].reset_index()
        path=args.out/f"submission_{number}_context50_{name}_odds.csv";common.write_csv(path,frame)
        outputs.append({"model":name,"file":path.name,"sha256":common.sha(path),"rows":len(frame),"public_ap":None})
    receipt={"status":"PASS","archive_sha256":common.sha(args.archive),"signature":p["signature"],"new_contexts":8,"reused42_contexts":4,
        "new_recent_AP":0,"new_optimizer_fit":0,"weight_search":False,"kaggle_uploads":0,"final_forms":producer.FORMS,"gate":producer.GATE,"summary":summary,"exports":outputs}
    common.write_json(args.out/"verification.json",receipt);print(json.dumps(receipt),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["archive","raw","out","baseline32-dir","duo-oof","context-test","native4-test","winner-test"]:parser.add_argument("--"+name,type=Path,required=True)
    run(parser.parse_args())
