"""Longitudinal archive identity, availability, OOB, auxiliary target and full lifecycle."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import sys

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import run_hw4_event_library as runner
from hw4_inspections import event_learning as event, event_library as lib
from test_hw4_event_learning import historical
from test_hw4_final_research import fixture, expect_error


def test_event_library_earliest_admissible_per_key_and_future_release_exclusion():
    sites=[f"site{i}" for i in range(25)]
    heldout=event.site_fold(sites[0])
    releases=[]
    for q in ["2023-07-01","2023-10-01","2024-01-01"]:
        cutoff=event.previous_cutoff(q)
        hist=event.snapshot(historical(q,sites),q,cutoff)
        release,query=lib.make_release(hist,q,cutoff,sites)
        releases.append(release)
    q="2023-10-01";cutoff=event.previous_cutoff(q)
    x,y,audit=lib.asof_library(releases[:2],q,cutoff,heldout)
    xf,yf,af=lib.asof_library(releases,q,cutoff,heldout)
    pd.testing.assert_frame_equal(x,xf);pd.testing.assert_frame_equal(y,yf)
    assert af==audit and audit["discarded_later_copies"]>0
    assert not y.fold.eq(heldout).any() and not y.duplicated(["site_id","quarter_start"]).any()
    assert y.source_period.max()==pd.Timestamp(q)
    # Modify only the later copy: the earlier features, date and target must win.
    first=releases[0]
    second=releases[1]
    pair=set(zip(first.outcomes.site_id,first.outcomes.quarter_start))
    match=next(i for i,row in second.outcomes.iterrows() if row.fold!=heldout and (row.site_id,row.quarter_start) in pair)
    second.outcomes.loc[match,"recorded_load"]=44
    second.features.loc[match,"visit_n"]=999
    nx,ny,newaudit=lib.asof_library(releases,q,cutoff,heldout)
    pd.testing.assert_frame_equal(nx,x)
    pd.testing.assert_frame_equal(ny,y)
    assert newaudit["keys_with_conflicting_later_outcomes"]>0
    # If the earlier issuance is inadmissible for THIS heldout partition,
    # the earliest still-admissible later copy is the canonical example.
    first.coverage[heldout]=(pd.Timestamp("2023-07-01"),pd.Timestamp("2023-07-08"))
    bx,by,_=lib.asof_library(releases,q,cutoff,heldout)
    assert 999 in bx.visit_n.values and (by.source_period==pd.Timestamp(q)).all()


def test_event_library_load_is_only_a_supervised_outcome():
    q="2024-01-01";cutoff=event.previous_cutoff(q)
    sites=[f"site{i}" for i in range(20)]
    h=historical(q,sites)
    release,xq=lib.make_release(event.snapshot(h,q,cutoff),q,cutoff,sites)
    row=release.outcomes.iloc[0]
    altered=h.copy()
    mask=altered.site_id.eq(row.site_id)&altered.inspection_week.eq(row.label_week.strftime("%Y-%m-%d"))
    altered.loc[mask,"violation_codes"]="39 40 41 42 43 44"
    changed,cq=lib.make_release(event.snapshot(altered,q,cutoff),q,cutoff,sites)
    pd.testing.assert_series_equal(release.features.iloc[0],changed.features.iloc[0])
    assert changed.outcomes.recorded_load.iloc[0]==6 and release.outcomes.recorded_load.iloc[0]!=6
    assert "recorded_load" not in release.features and "label_week" not in xq
    k=event.site_fold(sites[0]);x,y,_=lib.asof_library([release],q,cutoff,k)
    bundle=lib.fit_library(x,y,1,True)
    assert lib.predict_library(bundle,xq.head()).shape==(5,3)


def test_event_library_complete_forced_load_export_and_resume_no_fits():
    with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
        root=Path(temp)
        args=fixture(root)
        original=joblib.load(args.feature_cache)
        for frame,name in [(original["train"],"train.csv"),(original["test"],"test_student.csv")]:
            frame["history_before"]=[event.previous_cutoff(q).strftime("%Y-%m-%d") for q in frame.forecast_period]
            frame.to_csv(args.raw/name,index=False)
        joblib.dump(original,args.feature_cache)
        sites=original["train"].site_id.unique().tolist()
        qlist=pd.concat([original["train"].forecast_period,original["test"].forecast_period]).unique()
        pd.concat([historical(q,sites) for q in qlist]).to_csv(args.raw/"inspection_history.csv",index=False)
        p=args.feature_cache.with_name("features-meta.json");proof=runner.primary.read_json(p)
        proof.update(cache_sha256=runner.primary.sha(args.feature_cache),
                     raw_sha256={k:runner.primary.sha(args.raw/k) for k in runner.primary.RAW_FILES})
        runner.primary.write_json(p,proof)
        runner.window.run(args)
        args.window_run=args.out
        args.out=root/"library";args.out.mkdir()
        real=runner.window.evaluate
        def admit(args,stage,*a,**kw):
            real(args,stage,*a,**kw)
            return [runner.LOAD+"__blend75"] if stage in ("screen","confirmed") else []
        with patch.object(runner.window,"evaluate",side_effect=admit):
            result=runner.run(args)
            assert result["chosen"]==runner.LOAD+"__blend75" and result["issued_snapshots"]==12
            assert result["second_stage_jobs"]==6+6+3 and result["new_q2_q3_metrics"] is False
            assert len(result["files"])==1
            args.resume=True
            with patch.object(lib,"fit_library",side_effect=AssertionError("Unexpected event model refit")),patch.object(
                    runner.primary,"fit_booster",side_effect=AssertionError("Unexpected official model refit")):
                assert runner.run(args)==result
        # Wrong original OOF must be refused without training any model.
        with (args.window_run/"screen-oof.csv").open("ab") as h: h.write(b"x")
        expect_error(lambda:runner.v1_proof(args))


if __name__=="__main__":
    for name,test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: PASS",flush=True)
