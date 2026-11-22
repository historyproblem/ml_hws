"""Temporal-cutoff, group-OOB, boundary, resume and full packaging checks."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import run_hw4_event_learning as runner
from hw4_inspections import event_learning as e
from test_hw4_final_research import fixture, expect_error


def historical(period,sites):
    start=pd.Timestamp(period)
    cutoff=e.previous_cutoff(start)
    events=[]
    def add(site,week,kind,result,codes):
        events.append({"forecast_period":period,"site_id":site,"inspection_week":week.strftime("%Y-%m-%d"),
                       "inspection_type":kind,"result":result,"violation_codes":codes,
                       "area_id":"a","facility_type":"restaurant","risk_level":"medium"})
    low=cutoff-pd.Timedelta(days=854)
    # Coverage sentinel is a true historic licensing row, not a future row.
    for i,site in enumerate(sites):
        add(site,low,"licensing","pass","")
        add(site,cutoff-pd.Timedelta(days=7),"complaint","pass","1" if i%2 else "")
        for q in pd.period_range(low.to_period("Q"),start.to_period("Q")-1,freq="Q"):
            week=e.previous_cutoff(q.start_time)+pd.Timedelta(days=49+i%3*7)
            if low<week<cutoff:
                high=i>=len(sites)//2
                add(site,week,"routine","fail" if high else "pass","1 2 3 4 5" if high else "1")
    return pd.DataFrame(events,columns=e.HISTORY)


def test_event_landmark_cutoff_endpoint_attributes_and_unknown_first():
    quarter="2024-01-01"
    sites=[f"site{i}" for i in range(20)]
    h=historical(quarter,sites)
    cutoff=e.previous_cutoff(quarter)
    hist=e.snapshot(h,quarter,cutoff)
    x,y,query=e.landmark_examples(hist,quarter,cutoff,sites)
    assert len(x)==len(y)>40 and len(query)==20 and list(x)==list(query)
    assert not set(x).intersection({"y","target","site_id","label_week","row_id","event_gap_to_target"})
    row=y.index[0]
    source=y.loc[row]
    look=h.site_id.eq(source.site_id)&h.inspection_week.eq(source.label_week.strftime("%Y-%m-%d"))
    assert look.sum()==1
    alternative=h.copy()
    alternative.loc[look,["risk_level","area_id","violation_codes","result"]]=["future-risk","new-area","42 43 44",
                                                                          "pass" if source.y else "fail"]
    x2,y2,_=e.landmark_examples(e.snapshot(alternative,quarter,cutoff),quarter,cutoff,sites)
    pd.testing.assert_series_equal(x.iloc[row],x2.iloc[row])
    assert y2.loc[row,"y"]!=source.y
    f=h.iloc[:1].copy()
    f["inspection_week"]=cutoff.strftime("%Y-%m-%d")
    f["result"]="fail"
    assert e.snapshot(pd.concat([h,f]),quarter,cutoff).equals(hist)
    alien=f.assign(forecast_period="1999-01-01",inspection_week="2023-01-02")
    assert e.snapshot(pd.concat([h,alien]),quarter,cutoff).equals(hist)
    # An ambiguous or unknown first routine invalidates the site-quarter.
    start=pd.Timestamp("2023-01-01");end=pd.Timestamp("2023-04-01")
    cohort=pd.DataFrame({"site_id":["w","w","u","u","c","c","c","ok"],
                         "week":pd.to_datetime(["2022-12-26","2023-01-16","2023-01-02","2023-01-16",
                                                "2023-01-02","2023-01-02","2023-01-16","2023-01-16"]),
                         "inspection_type":"routine","result":["pass","fail","unknown","pass","pass","fail","pass","fail"],
                         "row_fail":[0.,1.,0.,0.,0.,1.,0.,1.],"fold":1})
    assert e.first_endpoints(cohort,start,end).site_id.tolist()==["ok"]


def test_event_group_exclusion_and_exact_prediction_invariance():
    period="2024-01-01"
    sites=[f"site{i}" for i in range(30)]
    q=e.previous_cutoff(period)
    h=historical(period,sites)
    hist=e.snapshot(h,period,q)
    x,y,v=e.landmark_examples(hist,period,q,sites)
    bucket=e.site_fold(sites[0])
    mask=e.training_mask(hist,y,bucket)
    assert not set(y.loc[mask,"site_id"])&{s for s in sites if e.site_fold(s)==bucket}
    bad=next(s for s in sites[1:] if e.site_fold(s)==bucket)
    changed=h.copy()
    changed.loc[changed.site_id.eq(bad),["violation_codes","result","area_id"]]=["44","fail","other"]
    h2=e.snapshot(changed,period,q);x2,y2,v2=e.landmark_examples(h2,period,q,sites)
    mask2=e.training_mask(h2,y2,bucket)
    pd.testing.assert_frame_equal(x.loc[mask],x2.loc[mask2])
    pd.testing.assert_frame_equal(y.loc[mask],y2.loc[mask2])
    pd.testing.assert_frame_equal(v.iloc[:1],v2.iloc[:1])
    a=e.fit_heads(x.loc[mask],y.loc[mask,"y"],1,True)
    b=e.fit_heads(x2.loc[mask2],y2.loc[mask2,"y"],1,True)
    np.testing.assert_array_equal(e.predict_heads(a,v.iloc[:1]),e.predict_heads(b,v2.iloc[:1]))
    expect_error(lambda:e.snapshot(h,period,q+pd.Timedelta(days=7)))


def test_event_upper_stage_complete_artifacts_both_pass_and_fail():
    # Only the final admission gate is patched, so this smoke covers the
    # expensive augmented full-execution branch regardless of tiny fake scores.
    with TemporaryDirectory() as raw_temp, redirect_stdout(StringIO()):
        root=Path(raw_temp)
        args=fixture(root)
        raw=pd.read_csv(args.raw/"train.csv")
        test=pd.read_csv(args.raw/"test_student.csv")
        sites=raw.site_id.unique().tolist()
        qlist=[*sorted(raw.forecast_period.unique()),*sorted(test.forecast_period.unique())]
        for filename in ["train.csv","test_student.csv"]:
            frame=pd.read_csv(args.raw/filename)
            frame["history_before"]=[e.previous_cutoff(q).strftime("%Y-%m-%d") for q in frame.forecast_period]
            frame.to_csv(args.raw/filename,index=False)
        import joblib
        cache=joblib.load(args.feature_cache)
        cache["train"]=pd.read_csv(args.raw/"train.csv")
        cache["test"]=pd.read_csv(args.raw/"test_student.csv")
        joblib.dump(cache,args.feature_cache)
        h=pd.concat([historical(q,sites) for q in qlist],ignore_index=True)
        h.to_csv(args.raw/"inspection_history.csv",index=False)
        cache_meta=args.feature_cache.with_name("features-meta.json")
        proof=runner.primary.read_json(cache_meta)
        proof["raw_sha256"]={k:runner.primary.sha(args.raw/k) for k in runner.primary.RAW_FILES}
        proof["cache_sha256"]=runner.primary.sha(args.feature_cache)
        runner.primary.write_json(cache_meta,proof)
        real=runner.evaluate
        def force_pass(args,stage,*a,**kw):
            real(args,stage,*a,**kw)
            return [runner.AUG] if stage in ("screen","confirmed") else []
        with patch.object(runner,"evaluate",side_effect=force_pass):
            report=runner.run(args)
            assert report["chosen"]==runner.AUG and report["first_level_snapshots"]==12
            assert report["second_level_jobs"]==3+3+6+3 and report["new_q2_q3_metrics"] is False
            assert all(n not in ("2024-04-01","2024-07-01") for n in
                       pd.read_csv(args.out/"screen-quarter-metrics.csv").forecast_period)
            args.resume=True
            with patch.object(e,"fit_heads",side_effect=AssertionError("Unplanned event head refit")),patch.object(
                    runner.primary,"fit_booster",side_effect=AssertionError("Unplanned official refit")):
                assert runner.run(args)==report
        # Detect damaged first-stage prediction artifacts before evaluation.
        directory=next((args.out/"heads").glob("*/*/complete.json")).parent
        with (directory/"probabilities.npy").open("ab") as f: f.write(b"x")
        target=runner.primary.read_json(directory/"complete.json")["identity"]["forecast_period"]
        query=pd.concat([cache["train"].drop(columns="target"),cache["test"]],ignore_index=True)
        expect_error(lambda:runner.score_snapshot(args,report["signature"],h,query.loc[query.forecast_period.eq(target)]))


if __name__=="__main__":
    for name,test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: PASS",flush=True)
