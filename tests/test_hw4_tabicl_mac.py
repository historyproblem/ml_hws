"""Own22 view, explicit resource evidence and strict forward-only Mac behavior."""
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import sys

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import run_hw4_tabicl_mac as mac


def test_tabicl_mac_exact22_native_categories_no_peers():
    names=[*mac.COMPACT_BASE_COLUMNS,"ctx_area_extra","row_id"]
    x=pd.DataFrame(np.zeros((3,len(names))),columns=names)
    x["risk_level"]=["low","medium",None]
    x["forecast_quarter"]=[1,2,4]
    x["history_violation_mean"]=[1.,np.inf,0.]
    v=mac.view(x)
    assert list(v)==list(mac.FIELDS) and len(mac.FIELDS)==22
    assert not any(c.startswith("ctx_") or c in {"forecast_year","cutoff_gap_days","row_id"} for c in v)
    assert v.forecast_quarter.tolist()==["1","2","4"] and v.risk_level.iloc[2]=="<none>"
    assert np.isnan(v.history_violation_mean.iloc[1]) and np.isinf(x.history_violation_mean.iloc[1])


def test_tabicl_mac_historical_job_never_uses_current_targets_or_returns_metric():
    class Model:
        def __init__(self,**settings):
            assert settings["n_estimators"]==4 and settings["device"]=="cpu" and settings["use_amp"] is False
            self.n_estimators=4
        def fit(self,x,y):
            self.p=float(np.mean(y));self.classes_=[0,1];self.model_=torch.nn.Linear(1,1)
        def predict_proba(self,x):
            return np.tile([1-self.p,self.p],(len(x),1))
    data=pd.DataFrame({"row_id":["a","b","c","d","e","f"],"forecast_period":["2023-01-01"]*4+["2024-04-01"]*2,
                       "target":[0,1,0,0,1,1]})
    test=pd.DataFrame({"row_id":["test"]});x=pd.DataFrame({"load":np.arange(7)})
    plan={"signature":"synthetic","context_cap":10000}
    with TemporaryDirectory() as temp,redirect_stdout(StringIO()),patch.dict(sys.modules,{"tabicl":SimpleNamespace(TabICLClassifier=Model)}):
        args=Namespace(out=Path(temp),resume=False,checkpoint="fixture")
        p,r=mac.forecast_job(args,plan,"2024-04-01",data,test,x)
        assert r["identity"]["train_rows"]==4 and np.array_equal(p,np.full(2,.25)) and "ap" not in r
        args.resume=True
        modified=data.copy();modified.loc[4:,"target"]=0
        p2,_=mac.forecast_job(args,plan,"2024-04-01",modified,test,x)
        assert np.array_equal(p2,p)


if __name__=="__main__":
    for n,f in list(globals().items()):
        if n.startswith("test_"):
            f();print(n,"PASS")
