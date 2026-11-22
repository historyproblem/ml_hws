"""Tiny official-baseline semantics and genuine zero-correction controls."""
from io import BytesIO
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from catboost import Pool
from scipy.special import expit

from hw4_inspections.gam_residual import choose_count,compose_probability,fit_residual,predict_residual,residual_raw


def fixture():
    rng=np.random.default_rng(42)
    frame=pd.DataFrame({'x':rng.normal(size=64),'category':['a','b']*32})
    target=(frame.x.to_numpy()+rng.normal(size=64)>.1).astype(int)
    baseline=.2+.5*frame.x.to_numpy()
    return frame,target,baseline


def test_residual_native_raw_baseline_adds_once_and_reload_exact():
    x,y,b=fixture();bundle=fit_residual(x,y,['category'],b,iterations=5)
    no=residual_raw(bundle,x)
    pool=Pool(x,cat_features=['category'],baseline=b)
    native=bundle['model'].predict(pool,prediction_type='RawFormulaVal',thread_count=2)
    np.testing.assert_array_equal(native,no+pool.get_baseline()[:,0])
    np.testing.assert_array_equal(predict_residual(bundle,x,b),expit(b+no))
    stream=BytesIO();joblib.dump(bundle,stream);stream.seek(0)
    np.testing.assert_array_equal(no,residual_raw(joblib.load(stream),x))
    assert bundle['actual_params']['loss_function']=='Logloss' and bundle['original_binary_targets']
    assert bundle['actual_params']['boost_from_average'] is False


def test_residual_zero_means_exact_teacher_and_no_native_fit_or_predict():
    x,y,b=fixture()
    with patch('hw4_inspections.gam_residual.CatBoostClassifier.fit',side_effect=AssertionError('Zero branch fit')):
        zero=fit_residual(x,y,['category'],b,iterations=0)
    np.testing.assert_array_equal(predict_residual(zero,x,b),expit(b))
    assert not zero['native_model_fit'] and zero['model'] is None
    nonzero=fit_residual(x,y,['category'],b,iterations=5)
    with patch.object(nonzero['model'],'predict',side_effect=AssertionError('ntree_end0 would mean all trees')):
        np.testing.assert_array_equal(residual_raw(nonzero,x,trees=0),np.zeros(len(x)))


def test_residual_rejects_probabilities_as_labels_and_bad_schema_baseline():
    x,y,b=fixture()
    for label,baseline in [(y*.5,b),(y,b[:-1]),(y,np.full(len(y),np.nan))]:
        try:fit_residual(x,label,['category'],baseline,iterations=0);raise AssertionError('Invalid target/baseline accepted')
        except ValueError:pass
    for name in ['target','row_id','site_id','forecast_period','history_before']:
        try:fit_residual(x.assign(**{name:0}),y,['category'],b,iterations=0);raise AssertionError('Raw identifier accepted')
        except ValueError:pass


def test_residual_inner_zero_wins_a_perfect_teacher_tie():
    x,y,_=fixture();baseline=np.where(y==1,10.,-10.)
    bundle=fit_residual(x,y,['category'],baseline,iterations=400)
    chosen,curve=choose_count(bundle,x,baseline,y)
    assert chosen==0 and [row['trees'] for row in curve]==[0,50,100,200,400]
    np.testing.assert_allclose([row['ap'] for row in curve],1,atol=1e-14,rtol=0)
