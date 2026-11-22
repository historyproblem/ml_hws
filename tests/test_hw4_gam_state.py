"""Small state-only augmentation, exact base prefix, bounded past features."""
from io import BytesIO

import joblib
import numpy as np

from hw4_inspections.additive_search import SearchTransform
from hw4_inspections.gam_state import CATEGORICAL,NUMERIC,RECIPES,StateTransform,choose_inner,fit_recipe,predict_recipe,recipe_by_id
from test_hw4_additive_logit import _frame


def fixture():
    f=_frame();n=len(f)
    f['history_fail_rate']=np.linspace(0,1,n);f['recent365_fail_rate']=np.linspace(.1,.9,n)
    f['history_n_visits']=np.arange(n)%8;f['last_routine_age_days']=np.arange(n)*7+7
    f['last_routine_result']=['pass','conditional','fail','<none>']*(n//4)
    f['lag1_result']=['pass','fail']*(n//2);f['lag1_inspection_type']=['routine','complaint']*(n//2)
    return f


def test_state_exact_unchanged_E_prefix_small_fields_and_declaredproducts():
    f=fixture();base=SearchTransform('core_interactions').fit(f).transform(f)
    for view in ['state','state_load_outcome']:
        transform=StateTransform(view).fit(f);matrix=transform.transform(f)
        np.testing.assert_array_equal(base,matrix[:,:transform.base_size_])
        assert len(NUMERIC)==4 and len(CATEGORICAL)==3 and len(RECIPES)==8
        if view=='state_load_outcome':
            assert len(transform.burden_indices_)<=5
            assert sum(c.startswith('stateLoadOutcome::') for c in transform.feature_names_)<=20


def test_state_stats_vocab_and_products_fit_only_tailbounded_missing_safe():
    f=fixture();transform=StateTransform('state_load_outcome').fit(f);stats={k:dict(v) for k,v in transform.stats_.items()}
    q=f.iloc[:3].copy();q['last_routine_age_days']=[-1e8,1e8,np.nan];q['last_routine_result']='unseen';q['lag1_result']='unknown';q['lag1_inspection_type']='newtype'
    value=transform.transform(q);assert np.isfinite(value).all() and transform.stats_==stats
    clipped=q.copy();clipped['last_routine_age_days']=clipped.last_routine_age_days.clip(stats['last_routine_age_days']['minimum'],stats['last_routine_age_days']['maximum'])
    np.testing.assert_array_equal(value,transform.transform(clipped))
    assert 'unseen' not in transform.outcome_encoder_.categories_[0]


def test_state_logit_reload_and_tie_rule():
    f=fixture();y=np.array([0,1]*20);bundle=fit_recipe(f,y,recipe_by_id('state_load_outcome_C0.01'))
    p=predict_recipe(bundle,f);stream=BytesIO();joblib.dump(bundle,stream);stream.seek(0)
    np.testing.assert_array_equal(p,predict_recipe(joblib.load(stream),f))
    assert choose_inner([dict(recipe=r['id'],status='complete',ap=.3) for r in RECIPES])=='state_C0.003'
