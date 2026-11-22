"""Synthetic-only four-recipe greedy/canonical algorithm selection and resume."""
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
import run_hw4_ebm_canonical as runner
from test_hw4_final_research import fixture as parent_fixture,expect_error
from test_hw4_ebm_search import fixture as full_fixture,RUNTIME


from test_hw4_ebm_runner import fixture

def test_ebm_canonical_nested_selection_uses_inner_and_one_aggregated_full_recipe():
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


def test_ebm_canonical_predeclared_GAM_domination_guard_uses_both_variants_and_all_quarters():
    rows=pd.DataFrame([dict(forecast_period=q,target=y) for q in runner.primary.QUARTERS for y in [1,0,0,1]])
    raw=np.tile([.8,.2,.7,.6],3);anchor=np.tile([.6,.3,.7,.4],3);gam=np.tile([.9,.1,.2,.8],3)
    result=runner.domination(rows,raw,anchor,gam)
    assert result["full_stop"] and all(v["clearly_dominated"] for v in result["variants"].values())
    gam[:4]=raw[:4]
    assert not runner.domination(rows,raw,anchor,gam)["full_stop"]
    assert not runner.domination(rows,None,anchor,gam)["evaluated"]


def test_ebm_canonical_native_tiny_pipeline_all12inner_before_outer_gates_full_resume():
    with TemporaryDirectory() as temporary,redirect_stdout(StringIO()):
        args=fixture(Path(temporary));original=runner.fit_ebm;calls=[]
        def fitting(*a,**kw):
            assert (args.out/"freeze.json").exists()
            if len(calls)<12:
                assert not list((args.out/"models").glob("*/complete.json"))
            calls.append(len(a[1]));return original(*a,**kw)
        with patch.object(runner,"fit_ebm",side_effect=fitting):result=runner.run(args)
        assert result["inner_fits_total"]==12 and result["outer_full_fits_total"]==12
        assert result["actual_model_fits_total"]==len(calls)==24
        assert result["screen_eligible"] and result["confirmed_eligible"]
        assert not result["full_stopped_by_GAM_domination"] and len(result["files"])==1
        args.resume=True
        with patch.object(runner,"fit_ebm",side_effect=AssertionError("Resume refit")):resumed=runner.run(args)
        assert resumed["new_inner_fits_this_invocation"]==resumed["new_outer_full_fits_this_invocation"]==0
        assert resumed["files"]==result["files"]
        path=next((args.out/"inner").glob("*/scores.json"));path.write_text(path.read_text()+" ")
        expect_error(lambda:runner.run(args))


def test_ebm_canonical_failed07_gate_stops_without_confirmation_or_full():
    with TemporaryDirectory() as temporary,redirect_stdout(StringIO()):
        result=runner.run(fixture(Path(temporary),perfect=True))
        assert result["inner_fits_total"]==12 and result["outer_full_fits_total"]==3
        assert not result["screen_eligible"] and result["files"]==[]
        assert result["actual_model_fits_total"]==15


if __name__=="__main__":
    for name,test in list(globals().items()):
        if name.startswith("test_"):test();print(f"{name}: PASS",flush=True)
