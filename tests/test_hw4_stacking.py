"""Analytic gradient, positive bounds, monotonic boundary and exact replay."""
from io import BytesIO

import joblib
import numpy as np
from scipy.special import expit

from hw4_inspections.stacking import FEATURES,fit_stacker,meta_features,objective_gradient,predict_stacker


def test_stacking_analytic_gradient_matches_finite_difference():
    rng=np.random.default_rng(42);x=rng.normal(size=(80,2));y=np.tile([0,1],40);theta=np.array([.7,.3,-.4])
    _,gradient=objective_gradient(theta,x,y);finite=[]
    for j in range(3):
        delta=np.zeros(3);delta[j]=1e-6
        finite.append((objective_gradient(theta+delta,x,y)[0]-objective_gradient(theta-delta,x,y)[0])/2e-6)
    np.testing.assert_allclose(gradient,finite,rtol=1e-6,atol=1e-8)


def test_stacking_convex_fit_nonnegative_deterministic_and_reload_exact():
    rng=np.random.default_rng(42);cb=expit(rng.normal(size=200));gam=rng.normal(size=200);x=meta_features(cb,gam)
    y=(rng.random(200)<expit(x@np.array([.7,.9])-.6)).astype(int)
    first=fit_stacker(cb,gam,y);second=fit_stacker(cb,gam,y)
    assert (first['weights']>=0).all() and first['features']==list(FEATURES) and first['solver_success']
    np.testing.assert_array_equal(first['weights'],second['weights'])
    p=predict_stacker(first,cb,gam);stream=BytesIO();joblib.dump(first,stream);stream.seek(0)
    np.testing.assert_array_equal(p,predict_stacker(joblib.load(stream),cb,gam))
    assert np.isfinite(p).all() and np.isfinite(meta_features([0,1],[-2,2])).all()


def test_stacking_one_zero_weight_has_only_other_input_ordering():
    gam=np.linspace(-3,3,40);cb=np.linspace(.99,.01,40)
    bundle=dict(weights=np.array([0.,1.4]),intercept=-.7,features=list(FEATURES))
    p=predict_stacker(bundle,cb,gam)
    np.testing.assert_array_equal(np.argsort(p),np.argsort(gam))
    np.testing.assert_array_equal(p,predict_stacker(bundle,cb[::-1],gam))


def test_stacking_rejects_extra_features_targets_and_bad_probability_domain():
    for cb,gam in [(np.zeros((4,2)),np.zeros(4)),(np.array([-1.,2.]),np.zeros(2)),(np.ones(4),np.zeros(3))]:
        try:meta_features(cb,gam);raise AssertionError('Invalid inputs accepted')
        except ValueError:pass
    try:fit_stacker(np.full(4,.5),np.zeros(4),np.ones(4));raise AssertionError('One-class labels accepted')
    except ValueError:pass
