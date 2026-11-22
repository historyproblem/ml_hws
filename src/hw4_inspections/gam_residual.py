"""Binary CatBoost corrections on causal GAM raw-logit baselines.

Training labels remain the original binary labels. The caller must supply
strictly past-fitted OOF teacher logits for every meta-training row. Prediction
adds teacher raw logits to CB RawFormulaVal from a dataframe WITHOUT baseline.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.special import expit
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from .classical_features import FORBIDDEN_COLUMNS
from .features import MISSING


TEACHER_RECIPE = "core_interactions_C0.01"
GRID = (0, 50, 100, 200, 400)
PARAMS = dict(depth=3, learning_rate=.03, l2_leaf_reg=20., rsm=.8,
              bootstrap_type="Bernoulli", subsample=.8, boost_from_average=False,
              loss_function="Logloss", task_type="CPU", thread_count=2,
              verbose=False, allow_writing_files=False)


def _raw_vector(value, n):
    value=np.asarray(value,dtype=np.float64)
    if value.shape!=(n,) or not np.isfinite(value).all():
        raise ValueError("Finite row-aligned raw logits required")
    return value


def _frame(frame,cats):
    if not isinstance(frame,pd.DataFrame) or not frame.columns.is_unique or FORBIDDEN_COLUMNS.intersection(frame):
        raise ValueError("Only unique causal feature columns, no raw identifiers/targets")
    if len(cats)!=len(set(cats)) or not set(cats)<=set(frame):raise ValueError("Invalid categorical names")
    result=frame.copy()
    for name in cats:result[name]=result[name].astype(object).where(result[name].notna(),MISSING).astype(str)
    numeric=[name for name in frame if name not in cats]
    result[numeric]=result[numeric].replace([np.inf,-np.inf],np.nan)
    return result


def teacher_raw(bundle,frame):
    """Direct fixed GAM decision function, without clipped-probability logits."""
    with threadpool_limits(limits=2):
        values=bundle['model'].decision_function(bundle['transformer'].transform(frame))
    return _raw_vector(values,len(frame))


def compose_probability(teacher_logits,correction):
    baseline=np.asarray(teacher_logits,dtype=np.float64)
    if baseline.ndim!=1:raise ValueError("One-dimensional teacher logits required")
    baseline=_raw_vector(baseline,len(baseline));correction=_raw_vector(correction,len(baseline))
    return expit(baseline+correction)


def fit_residual(frame,y,cats,baseline_logits,*,iterations=400,seed=42):
    if isinstance(iterations,bool) or not isinstance(iterations,Integral) or not 0<=iterations<=400:
        raise ValueError("Residual tree count must be integer0..400")
    baseline=_raw_vector(baseline_logits,len(frame));target=np.asarray(y)
    if target.shape!=(len(frame),) or not np.isfinite(target).all() or set(np.unique(target))!={0,1}:
        raise ValueError("Original aligned binary labels with both classes required")
    features=_frame(frame,cats)
    bundle=dict(model=None,iterations=int(iterations),seed=int(seed),features=list(frame),cats=list(cats),
                params={**PARAMS,'random_seed':int(seed),'iterations':int(iterations)},
                original_binary_targets=True,init_model=None,training_rows=len(frame),
                baseline_role='Caller-verified strictpast OOF GAM raw logits, never in-sample teacher predictions')
    if iterations==0:
        bundle['native_model_fit']=False
        return bundle
    pool=Pool(features,label=target,cat_features=list(cats),baseline=baseline)
    native=pool.get_baseline()[:,0]
    model=CatBoostClassifier(**bundle['params'])
    model.fit(pool)
    bundle.update(model=model,native_model_fit=True,
                  native_baseline_dtype=str(native.dtype),
                  native_baseline_max_cast_error=float(np.max(np.abs(native-baseline))),
                  actual_params=model.get_all_params())
    if model.get_all_params()['boost_from_average'] is not False:
        raise AssertionError('Unexpected class-prior initialization on top of baseline')
    return bundle


def residual_raw(bundle,frame,trees=None):
    if list(frame)!=bundle['features']:raise ValueError('Residual prediction schema/order differs')
    count=bundle['iterations'] if trees is None else trees
    if isinstance(count,bool) or not isinstance(count,Integral) or not 0<=count<=bundle['iterations']:
        raise ValueError('Invalid tree prefix')
    if count==0:
        # Native ntree_end=0 means ALL trees, so zero is an explicit sentinel.
        return np.zeros(len(frame),dtype=np.float64)
    if bundle['model'] is None:raise ValueError('Missing nonzero residual model')
    value=bundle['model'].predict(_frame(frame,bundle['cats']),prediction_type='RawFormulaVal',
                                  ntree_start=0,ntree_end=int(count),thread_count=2)
    return _raw_vector(value,len(frame))


def predict_residual(bundle,frame,teacher_logits,trees=None):
    return compose_probability(teacher_logits,residual_raw(bundle,frame,trees))


def choose_count(bundle,frame,teacher_logits,target):
    if bundle['iterations']!=400:raise ValueError('Selection requires the one fixed400-tree inner fit')
    values=np.asarray(target)
    if values.shape!=(len(frame),) or not np.isin(values,[0,1]).all():raise ValueError('Aligned inner binary labels required')
    curve=[dict(trees=count,ap=float(average_precision_score(values,predict_residual(bundle,frame,teacher_logits,count)))) for count in GRID]
    return max(curve,key=lambda r:(r['ap'],-r['trees']))['trees'],curve
