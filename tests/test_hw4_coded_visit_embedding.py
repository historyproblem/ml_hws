"""Fold exclusion, cutoff, class-label isolation, exact base-prefix and persisted inference."""
from io import BytesIO
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"tests")]
from hw4_inspections import coded_visit_embedding as code
from hw4_inspections.additive_search import fit_recipe,recipe_by_id
from test_hw4_additive_logit import _frame


def fixture():
    names=[f"site-{i}" for i in range(100)]
    h=[]
    for i,s in enumerate(names):
        for k,t in enumerate(["2023-02-06","2023-04-10","2023-08-07","2023-12-04"]):
            h.append(("2024-01-01",s,t,"routine","fail" if (i+k)%3==0 else "conditional" if i%3==0 else "pass",
                      "01 1 4" if (i+k)%3==0 else "1 3"))
    rows=pd.DataFrame({"row_id":range(101),"site_id":[*names,"absent"],"forecast_period":"2024-01-01",
                       "history_before":"2023-12-25","target":0})
    return rows,pd.DataFrame(h,columns=code.HIST_FIELDS)


def test_code_teacher_inputs_opaque_unique_and_correct_scale():
    x=code.code_matrix([(1,2,44),(1,),()])
    assert x.shape==(3,45) and x[0,0]==x[0,1]==x[0,43]==1.
    assert x[0,:44].sum()==3 and x[0,-1]==np.log1p(3)/np.log(45.)
    assert x[2].sum()==0


def test_code_teacher_no_focal_result_or_future_leak_and_unknowns_score():
    rows,h=fixture()
    base,models,audit=code.embed(rows,h,auxiliary=True)
    assert len(models)==5 and audit["auxiliary"]["n"]==len(h)
    assert base.iloc[-1].isna().all() and base.iloc[:-1].notna().all().all()
    fq=rows.site_id.iloc[0]; k=code.site_fold(fq)
    altered=h.copy()
    selected=altered.site_id.map(code.site_fold).eq(k)
    altered.loc[selected,"result"]="unknown"  # all focal-bucket outcomes disappear
    altered=pd.concat([altered,h.iloc[[0]].assign(inspection_week="2023-12-25",violation_codes="44",result="fail"),
                        h.iloc[[0]].assign(forecast_period="2024-04-01",violation_codes="44")],ignore_index=True)
    other,nmodels,_=code.embed(rows.assign(target=1),altered)
    targetrows=rows.site_id.map(code.site_fold).eq(k)
    np.testing.assert_array_equal(base[targetrows],other[targetrows])
    np.testing.assert_array_equal(models[k].coef_,nmodels[k].coef_)
    # All four of the queried site's visits, including the most recent unknown,
    # should contribute in the correct chronological order and with no fallback.
    vh=code.routine_snapshot(altered,"2024-01-01","2023-12-25")
    own=vh[vh.site_id.eq(fq)]
    raw=nmodels[k].decision_function(code.code_matrix(own.tokens))
    np.testing.assert_allclose(other.iloc[0],[raw[-1],raw[-3:].mean(),raw.mean()],rtol=0,atol=1e-14)
    data=BytesIO();joblib.dump(nmodels,data);data.seek(0)
    np.testing.assert_array_equal(other,code.apply_heads(rows,altered,joblib.load(data)))


def test_joint_transform_and_parameters_exact_to_g0_part():
    frame=_frame();y=(np.arange(40)%3==0).astype(int)
    original=fit_recipe(frame,y,recipe_by_id(code.G0))
    raw=pd.DataFrame(np.arange(120).reshape(40,3)/71.,columns=code.FEATURES);raw.iloc[3,:]=np.nan
    augmented=code.fit_augmented(original,frame,raw,y)
    base=original["transformer"].transform(frame)
    joint=np.column_stack([augmented["base"].transform(frame),augmented["extra"].transform(raw)])
    np.testing.assert_array_equal(base,joint[:,:base.shape[1]])
    assert joint.shape[1]==base.shape[1]+6 and joint[3,-3:].tolist()==[1.,1.,1.]
    assert augmented["model"].C==.01 and augmented["model"].max_iter==5000
    med=augmented["extra"].medians_.copy()
    qq=raw.iloc[:2].copy();qq.iloc[0,:]=100.
    p=code.predict(augmented,frame.iloc[:2],qq)
    np.testing.assert_array_equal(med,augmented["extra"].medians_)
    stream=BytesIO();joblib.dump(augmented,stream);stream.seek(0)
    np.testing.assert_array_equal(p,code.predict(joblib.load(stream),frame.iloc[:2],qq))


if __name__=="__main__":
    for name,test in list(globals().items()):
        if name.startswith("test_"):
            test();print(name,"PASS",flush=True)
