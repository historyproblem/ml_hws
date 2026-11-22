"""TabICLv2 full-context producer invariants, independent of a V100."""
from argparse import Namespace
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import run_hw4_tabicl_v100 as icl


def test_tabicl_keeps_full_prefix_without_using_targets_or_capping_at_10000():
    periods=["2023-01-01"]*7000+["2023-04-01"]*6000+["2023-07-01"]*3000
    t=pd.DataFrame({"forecast_period":periods,"row_id":[f"r{i:05d}" for i in range(len(periods))]})
    t=t.sample(frac=1,random_state=17).reset_index(drop=True)
    rows=t.loc[icl.prefix(t,"2023-07-01")]
    assert len(rows)==13000 and rows.forecast_period.max()=="2023-04-01"
    assert list(rows.row_id)==list(rows.sort_values(["forecast_period","row_id"]).row_id)
    assert len(icl.prefix(t,"full"))==len(t)
    assert len(icl.prefix(t,"2024-07-01"))==len(t)


def test_tabicl_uses_local_checkpoint_fp32_single_member_batches():
    called={}
    def classifier(**kwargs):
        called.update(kwargs)
        return "local"
    with patch.dict(sys.modules,{"tabicl":SimpleNamespace(TabICLClassifier=classifier)}):
        assert icl.model(Namespace(checkpoint=Path("local.ckpt"),device="cpu",threads=2))=="local"
    assert called["n_estimators"]==4 and called["batch_size"]==1 and called["n_jobs"]==2
    assert called["kv_cache"] is False and called["allow_auto_download"] is False
    assert called["use_amp"] is False and called["use_fa3"] is False


def test_tabicl_current_recent_labels_are_excluded_and_saved_forecasts_reused():
    class Tiny:
        n_estimators=4
        def fit(self,x,y):
            self.mean=float(np.mean(y))
            self.classes_=[0,1]
        def predict_proba(self,x):
            scores=np.full(len(x),self.mean)
            return np.column_stack([1-scores,scores])
    period="2024-04-01"
    train=pd.DataFrame({"row_id":["z","a","b","hold0","hold1"],"site_id":["x","y","z","x","y"],
        "forecast_period":["2023-01-01","2023-04-01","2023-07-01",period,period],"target":[1,0,0,1,1]})
    test=pd.DataFrame({"row_id":["test"]})
    frame=pd.DataFrame({"x":np.arange(6)})
    with TemporaryDirectory() as temp,redirect_stdout(io.StringIO()),patch.object(icl,"model",side_effect=lambda args:Tiny()):
        args=Namespace(out=Path(temp),device="cpu",resume=False)
        p,record=icl.job(args,"sig",icl.CASES[1],period,train,test,frame,[])
        assert np.array_equal(p,np.full(2,1/3)) and record["identity"]["train_rows"]==3
        assert "ap" not in record and record["repeated_forward_exact"] is None
        args.resume=True
        with patch.object(icl,"model",side_effect=AssertionError("New fit in resume")):
            changed=train.copy()
            changed.loc[changed.forecast_period.eq(period),"target"]=0
            second,_=icl.job(args,"sig",icl.CASES[1],period,changed,test,frame,[])
            np.testing.assert_array_equal(second,p)


if __name__=="__main__":
    for name,test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: PASS",flush=True)
