"""Platform-independent TabPFN V2 producer contract checks; no pretrained download."""
import io
from pathlib import Path
import sys
import tarfile
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import run_hw4_tabpfn_v100 as pfn
from hw4_inspections.classical_features import COMPACT_BASE_COLUMNS


def rejects(call):
    try: call()
    except ValueError: return
    raise AssertionError("Expected a rejected input")


def test_tabpfn_tail_cap_causality_and_order_invariance():
    train=pd.DataFrame({"forecast_period":["2023-04-01","2023-10-01","2023-04-01","2023-01-01","2023-04-01"],
                        "row_id":["r03","r99","r01","r00","r02"],"target":[1,0,0,1,1]})
    selected=lambda df,q:df.loc[pfn.recent_indices(df,q,2),"row_id"].tolist()
    assert selected(train,"2023-07-01")==["r02","r03"]
    shuffled=train.sample(frac=1,random_state=17).reset_index(drop=True).assign(target=lambda d:1-d.target)
    assert selected(shuffled,"2023-07-01")==["r02","r03"]
    assert selected(train,"full")==["r03","r99"]
    assert selected(train,"2024-04-01")==["r03","r99"]
    rejects(lambda:pfn.recent_indices(train,"2022-12-10",2))


def test_tabpfn_views_keep_genuine_categories_and_native_missing():
    names=list(COMPACT_BASE_COLUMNS)+[f"ctx_{i}" for i in range(72)]+[f"other_{i}" for i in range(250)]
    x=pd.DataFrame(np.zeros((3,len(names))),columns=names)
    x["forecast_quarter"]=[2,3,4]
    x["facility_type"]=["cafe",None,"shop"]
    x.loc[1,"history_violation_mean"]=np.inf
    cats=["facility_type","risk_level","area_id",*[f"other_{i}" for i in range(7)]]
    result=pfn.views(x,cats)
    small,cs=result["pfn_v2_compact94"];large,cl=result["pfn_v2_full346"]
    assert small.shape==(3,94) and large.shape==(3,346)
    assert small.forecast_quarter.tolist()==["2","3","4"]
    assert large.forecast_quarter.tolist()==[2.,3.,4.]
    assert small.facility_type.tolist()==["cafe","<none>","shop"]
    assert pd.isna(small.history_violation_mean.iloc[1])
    assert list(small.columns[cs])==["facility_type","risk_level","area_id","forecast_quarter"]
    assert list(large.columns[cl])==cats and np.isinf(x.history_violation_mean.iloc[1])


def test_tabpfn_cuda_oom_only_changes_query_chunking():
    class Sized:
        def __init__(self): self.args=[]
        def predict_proba(self,x):
            self.args.append(len(x))
            if len(x)>128: raise torch.cuda.OutOfMemoryError("synthetic large query batch")
            p=x.v.to_numpy(dtype=float)/900
            return np.column_stack([1-p,p])
    model=Sized();frame=pd.DataFrame({"v":np.arange(600)})
    with patch.object(torch.cuda,"empty_cache"):
        values,chunk,attempts=pfn.predict_with_memory_policy(model,frame,"cuda")
    assert attempts==[600,512] and chunk==128
    np.testing.assert_array_equal(values,frame.v/900)
    assert len(model.args)==2+5


def test_tabpfn_only_seven_regular_return_members():
    with TemporaryDirectory() as temp:
        root=Path(temp)
        for name in pfn.RETURN_FILES: (root/name).write_text(name+"\n")
        (root/"secret-local-input.csv").write_text("private fixture")
        path=pfn.pack(root)
        with tarfile.open(path,"r:gz") as archive:
            assert archive.getnames()==list(pfn.RETURN_FILES)
            assert len(archive.getmembers())==7 and all(m.isreg() and m.linkname=="" for m in archive)
            assert archive.extractfile("recent_predictions.csv").read()==b"recent_predictions.csv\n"


if __name__=="__main__":
    for name,test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: PASS",flush=True)
