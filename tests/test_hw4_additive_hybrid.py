"""Tiny interaction/recency integration and explicit penalty-scale checks."""
from io import BytesIO

import joblib
import numpy as np
import pandas as pd

from hw4_inspections.additive_hybrid import RECIPES,choose_final,choose_inner,fit_recipe,predict_recipe,recipe_by_id
from hw4_inspections.classical_features import recency_weights
from test_hw4_additive_logit import _frame


def test_hybrid12_meanone_weights_keep_penalty_scale_and_reduce_ess():
    assert len(RECIPES)==12
    periods=pd.Series(["2022-01-01","2023-01-01"]*20)
    weights={h:recency_weights(periods,"2023-07-01",half_life_quarters=h) for h in [2,4,8]}
    ess={h:w.sum()**2/np.square(w).sum() for h,w in weights.items()}
    assert ess[2]<ess[4]<ess[8]
    for w in weights.values():
        np.testing.assert_allclose(w.mean(),1,atol=1e-15)
        for c in [.003,.01,.03,.1]:np.testing.assert_allclose(1/(c*w.sum()),1/(c*len(w)),rtol=1e-15)


def test_hybrid_fit_reuses_fixed_interaction_view_and_reload_exact():
    frame=_frame();y=np.array([0,1]*20);periods=pd.Series(["2022-01-01","2023-01-01"]*20)
    bundle=fit_recipe(frame,y,recipe_by_id("hybrid_H4_C0.01"),periods=periods,reference="2023-07-01")
    assert bundle['transformer'].view=='core_interactions' and bundle['threads']==2
    assert abs(bundle['weight_summary']['l2_scale_ratio']-1)<1e-12
    p=predict_recipe(bundle,frame);stream=BytesIO();joblib.dump(bundle,stream);stream.seek(0)
    np.testing.assert_array_equal(p,predict_recipe(joblib.load(stream),frame))


def test_hybrid_ties_prefer_stronger_l2_longer_history_and_majority():
    rows=[dict(recipe=r['id'],status='complete',ap=.3) for r in RECIPES]
    assert choose_inner(rows)=='hybrid_H8_C0.003'
    all_rows=[dict(r,quarter=q) for q in range(3) for r in rows]
    assert choose_final(['hybrid_H4_C0.01','hybrid_H2_C0.003','hybrid_H4_C0.01'],all_rows)=='hybrid_H4_C0.01'
