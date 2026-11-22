"""One fixed convex nonnegative two-logit prequential calibration model."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits


LAMBDA = .001
EPSILON = 1e-6
FEATURES = ("context04_mean3_clipped_logit", "fixed_interaction_GAM_raw_logit")
OPTIMIZER = dict(method="L-BFGS-B", gtol=1e-9, ftol=1e-15, maxiter=2000)
REFERENCE_SEEDS = (42, 17, 2026)


def fit_reference(frame,target,cats,seed,iterations=500):
    """Only the exact original reference family; no tree-prefix/recipe search."""
    from copy import deepcopy
    from .classical_models import SPECS,fit_model
    if seed not in REFERENCE_SEEDS:raise ValueError('Fixed reference seeds only')
    spec=deepcopy(SPECS['cb_reference'])
    # Explicit values equal the real-data CPU defaults of the original family;
    # pinning avoids the special bootstrap default for tiny synthetic datasets.
    spec['params'].update(boosting_type='Plain',bootstrap_type='MVS',subsample=.8)
    return fit_model(spec,frame,target,cats,seed,2,iterations=iterations)


def predict_reference(bundle,frame):
    from .classical_models import _transform
    if bundle['family']!='catboost':raise ValueError('Original CatBoost reference required')
    p=bundle['model'].predict_proba(_transform(bundle,frame),thread_count=2)[:,1]
    if p.dtype!=np.float64 or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise FloatingPointError('Invalid reference probabilities')
    return p


def meta_features(cb_probability,gam_raw):
    cb=np.asarray(cb_probability,dtype=np.float64);gam=np.asarray(gam_raw,dtype=np.float64)
    if cb.ndim!=1 or gam.shape!=cb.shape or not np.isfinite(cb).all() or not np.isfinite(gam).all():
        raise ValueError('Aligned finite CB probabilities and GAM raw logits required')
    if ((cb<0)|(cb>1)).any():raise ValueError('CB inputs must be probabilities, not raw logits')
    return np.column_stack([logit(np.clip(cb,EPSILON,1-EPSILON)),gam])


def objective_gradient(theta,features,target):
    weights=theta[:2];raw=features@weights+theta[2]
    residual=expit(raw)-target
    value=float(np.mean(np.logaddexp(0,raw)-target*raw)+LAMBDA/2*np.dot(weights,weights))
    gradient=np.r_[features.T@residual/len(target)+LAMBDA*weights,residual.mean()]
    return value,gradient


def fit_stacker(cb_probability,gam_raw,target):
    features=meta_features(cb_probability,gam_raw);y=np.asarray(target)
    if not len(features) or y.shape!=(len(features),) or not np.isfinite(y).all() or set(np.unique(y))!={0,1}:
        raise ValueError('Aligned binary meta labels with both classes required')
    initial=np.array([.5,.5,float(logit(y.mean()))])
    with threadpool_limits(limits=2):
        result=minimize(objective_gradient,initial,args=(features,y),jac=True,
                        method=OPTIMIZER['method'],bounds=[(0,None),(0,None),(None,None)],
                        options={k:v for k,v in OPTIMIZER.items() if k!='method'})
    if not result.success or not np.isfinite(result.x).all() or (result.x[:2]<0).any():
        raise RuntimeError(f'Fixed convex stacker did not converge: {result.message}')
    value,gradient=objective_gradient(result.x,features,y)
    projected=gradient.copy()
    for j in range(2):
        if result.x[j]==0 and projected[j]>0:projected[j]=0
    return dict(weights=result.x[:2].copy(),intercept=float(result.x[2]),
        features=list(FEATURES),lambda_=LAMBDA,epsilon=EPSILON,optimizer=dict(OPTIMIZER),
        bounds=['nonnegative','nonnegative','unbounded'],initial=initial.tolist(),
        objective=value,gradient=gradient.tolist(),projected_gradient_inf=float(np.max(abs(projected))),
        solver_success=bool(result.success),solver_message=str(result.message),iterations=int(result.nit),
        training_rows=len(y),active_inputs=int((result.x[:2]>1e-12).sum()),
        class_weights=None,sample_weights=None,raw_features_used=False,threads=2)


def predict_stacker(bundle,cb_probability,gam_raw):
    features=meta_features(cb_probability,gam_raw)
    if bundle['features']!=list(FEATURES) or np.asarray(bundle['weights']).shape!=(2,) or (np.asarray(bundle['weights'])<0).any():
        raise ValueError('Not the frozen nonnegative2logit estimator')
    with threadpool_limits(limits=2):p=expit(features@bundle['weights']+bundle['intercept'])
    if p.dtype!=np.float64 or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise FloatingPointError('Invalid calibrated stacker probabilities')
    return p
