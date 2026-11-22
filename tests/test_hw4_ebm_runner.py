"""Synthetic-only EBM algorithm selection, full guard and resume checks."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import run_hw4_ebm_search as runner
from test_hw4_final_research import fixture as parent_fixture,expect_error
from test_hw4_ebm_search import fixture as full_fixture,RUNTIME


def fixture(root,perfect=False):
    args=parent_fixture(root)
    cached=joblib.load(args.feature_cache);template,_,cats,_=full_fixture()
    n=len(cached["train"])+len(cached["test"])
    base=pd.concat([template.iloc[[1]]]*n,ignore_index=True)
    base["history_violation_mean"]=np.tile(np.arange(16,dtype=float),12)
    base["last_routine_violation_count"]=np.tile(np.arange(16,dtype=float),12)
    all_periods=pd.concat([cached["train"].forecast_period,cached["test"].forecast_period],ignore_index=True)
    base["forecast_quarter"]=pd.to_datetime(all_periods).dt.quarter.to_numpy()
    views={"base":base}
    for view,columns in [("transition",runner.primary.TRANSITION_COLUMNS),("seasonal",runner.primary.SEASONAL_COLUMNS)]:
        views[view]=base.assign(**{name:.1 for name in columns})
    cached.update(views=views,cats=cats);joblib.dump(cached,args.feature_cache)
    meta=runner.primary.read_json(args.feature_cache.with_name("features-meta.json"));meta["cache_sha256"]=runner.primary.sha(args.feature_cache)
    runner.primary.write_json(args.feature_cache.with_name("features-meta.json"),meta)
    old=runner.primary.read_csv(args.baseline_cv)
    old["additive_logit__blend75"]=old.target if perfect else .5
    args.additive_oof=root/"previous07.csv";old.to_csv(args.additive_oof,index=False)
    args.additive_test=args.context_test
    old["algorithm_raw"]=.5;args.gam_oof=root/"gam.csv";old.to_csv(args.gam_oof,index=False)
    args.runtime=RUNTIME;args.threads=2
    return args


def test_ebm_nested_selection_uses_inner_and_one_aggregated_full_recipe():
    recipes=deepcopy(runner.EBM_RECIPES)
    records={name:{q:dict(ap=.2,logloss=.6,inner_quarter="past") for q in runner.primary.QUARTERS} for name in recipes}
    first,later=list(recipes)[:2]
    for q,a,b in zip(runner.primary.QUARTERS,[.9,.1,.1],[.5,.6,.6]):
        records[first][q]["ap"]=a;records[later][q]["ap"]=b
    choices=runner.select_inner(records,recipes)
    assert choices["outer"][runner.primary.QUARTERS[0]]["recipe"]==first
    assert choices["outer"][runner.primary.QUARTERS[1]]["recipe"]==later
    assert choices["final"]["recipe"]==later
    before=deepcopy(choices)
    for table in records.values():
        for record in table.values():record["outer_ap"]=1
    assert runner.select_inner(records,recipes)==before


def test_ebm_predeclared_GAM_domination_guard_uses_both_variants_and_all_quarters():
    rows=pd.DataFrame([dict(forecast_period=q,target=y) for q in runner.primary.QUARTERS for y in [1,0,0,1]])
    raw=np.tile([.8,.2,.7,.6],3);anchor=np.tile([.6,.3,.7,.4],3);gam=np.tile([.9,.1,.2,.8],3)
    result=runner.domination(rows,raw,anchor,gam)
    assert result["full_stop"] and all(v["clearly_dominated"] for v in result["variants"].values())
    gam[:4]=raw[:4]
    assert not runner.domination(rows,raw,anchor,gam)["full_stop"]
    assert not runner.domination(rows,None,anchor,gam)["evaluated"]


def test_ebm_native_tiny_pipeline_all24inner_before_outer_gates_full_resume():
    with TemporaryDirectory() as temporary,redirect_stdout(StringIO()):
        args=fixture(Path(temporary));original=runner.fit_ebm;calls=[]
        def fitting(*a,**kw):
            assert (args.out/"freeze.json").exists()
            if len(calls)<24:
                assert not list((args.out/"models").glob("*/complete.json"))
            calls.append(len(a[1]));return original(*a,**kw)
        with patch.object(runner,"fit_ebm",side_effect=fitting):result=runner.run(args)
        assert result["inner_fits_total"]==24 and result["outer_full_fits_total"]==12
        assert result["actual_model_fits_total"]==len(calls)==36
        assert result["screen_eligible"] and result["confirmed_eligible"]
        assert not result["full_stopped_by_GAM_domination"] and len(result["files"])==1
        args.resume=True
        with patch.object(runner,"fit_ebm",side_effect=AssertionError("Resume refit")):resumed=runner.run(args)
        assert resumed["new_inner_fits_this_invocation"]==resumed["new_outer_full_fits_this_invocation"]==0
        assert resumed["files"]==result["files"]
        path=next((args.out/"inner").glob("*/scores.json"));path.write_text(path.read_text()+" ")
        expect_error(lambda:runner.run(args))


def test_ebm_failed07_gate_stops_without_confirmation_or_full():
    with TemporaryDirectory() as temporary,redirect_stdout(StringIO()):
        result=runner.run(fixture(Path(temporary),perfect=True))
        assert result["inner_fits_total"]==24 and result["outer_full_fits_total"]==3
        assert not result["screen_eligible"] and result["files"]==[]
        assert result["actual_model_fits_total"]==27


if __name__=="__main__":
    for name,test in list(globals().items()):
        if name.startswith("test_"):test();print(f"{name}: PASS",flush=True)
