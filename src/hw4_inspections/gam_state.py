"""Small past-outcome/cadence memory appended to the unchanged winning E view."""
from __future__ import annotations

from collections import Counter
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from threadpoolctl import threadpool_limits

from .additive_logit import AdditiveLogitTransform
from .additive_search import CLASSIFIER,SearchTransform,_categorical


NUMERIC=('history_fail_rate','recent365_fail_rate','history_n_visits','last_routine_age_days')
CATEGORICAL=('lag1_result','last_routine_result','lag1_inspection_type')
LOG_FIELDS=frozenset({'history_n_visits','last_routine_age_days'})
RECIPES=tuple(dict(id=f'{view}_C{c:g}',view=view,C=c,complexity_order=rank)
              for view,rank in [('state',0),('state_load_outcome',1)] for c in [.003,.01,.03,.1])


def recipe_by_id(name):
    for recipe in RECIPES:
        if recipe['id']==name:return dict(recipe)
    raise ValueError(f'Unknown frozen state recipe {name}')


class StateTransform:
    def __init__(self,view):
        if view not in {'state','state_load_outcome'}:raise ValueError(view)
        self.view=view

    def _numeric(self,frame):
        output,missing=[],[]
        for name in NUMERIC:
            raw=AdditiveLogitTransform._raw_numeric(frame,name);missing.append(np.isnan(raw).astype(float));state=self.stats_[name]
            value=np.clip(np.where(np.isnan(raw),state['median'],raw),state['minimum'],state['maximum'])
            if name in LOG_FIELDS:value=np.log1p(value)
            output.append(value)
        return np.column_stack(output),np.column_stack(missing)

    def _products(self,frame,base):
        outcomes=self.outcome_encoder_.transform(_categorical(frame,['last_routine_result']))
        return (base[:,self.burden_indices_,None]*outcomes[:,None,:]).reshape(len(frame),-1)

    def fit(self,frame):
        missing=set([*NUMERIC,*CATEGORICAL])-set(frame)
        if missing:raise ValueError(f'Missing semantic state fields:{sorted(missing)}')
        self.base_=SearchTransform('core_interactions').fit(frame)
        self.base_size_=len(self.base_.feature_names_)
        self.stats_={}
        for name in NUMERIC:
            raw=AdditiveLogitTransform._raw_numeric(frame,name);observed=raw[np.isfinite(raw)]
            self.stats_[name]=dict(minimum=float(observed.min()) if len(observed) else 0.,maximum=float(observed.max()) if len(observed) else 0.,median=float(np.median(observed)) if len(observed) else 0.)
            if name in LOG_FIELDS and self.stats_[name]['minimum']<0:raise ValueError('Negative count/age in historical state')
        numeric,_=self._numeric(frame);self.scaler_=StandardScaler().fit(numeric)
        self.encoder_=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(_categorical(frame,CATEGORICAL))
        names=[*self.base_.feature_names_,*[f'state::{c}' for c in NUMERIC],
               *[f'state::{c}__missing' for c in NUMERIC],
               *[f'stateOHE::{c}' for c in self.encoder_.get_feature_names_out(CATEGORICAL)]]
        self.product_scaler_=None
        if self.view=='state_load_outcome':
            self.burden_indices_=[i for i,c in enumerate(self.base_.feature_names_) if c.startswith('last_routine_violation_count__spline')]
            if not self.burden_indices_:
                self.burden_indices_=[self.base_.feature_names_.index('last_routine_violation_count')]
            self.outcome_encoder_=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(_categorical(frame,['last_routine_result']))
            products=self._products(frame,self.base_.transform(frame));self.product_scaler_=StandardScaler().fit(products)
            for i in self.burden_indices_:
                for outcome in self.outcome_encoder_.categories_[0]:names.append(f'stateLoadOutcome::{self.base_.feature_names_[i]}::{outcome}')
        if len(names)!=len(set(names)):raise AssertionError('Duplicate state feature names')
        self.feature_names_=names
        return self

    def transform(self,frame):
        if not hasattr(self,'base_'):raise ValueError('StateTransform must be fitted')
        base=self.base_.transform(frame)
        if not len(frame):return np.empty((0,len(self.feature_names_)))
        numeric,missing=self._numeric(frame)
        blocks=[base,self.scaler_.transform(numeric),missing,self.encoder_.transform(_categorical(frame,CATEGORICAL))]
        if self.product_scaler_ is not None:blocks.append(self.product_scaler_.transform(self._products(frame,base)))
        result=np.concatenate(blocks,axis=1)
        if not np.isfinite(result).all() or result.shape[1]!=len(self.feature_names_):raise ValueError('Invalid state design matrix')
        return result


def fit_recipe(frame,target,recipe):
    recipe=recipe_by_id(recipe['id']);y=np.asarray(target)
    if y.shape!=(len(frame),) or not np.isfinite(y).all() or set(np.unique(y))!={0,1}:raise ValueError('Aligned binary labels required')
    transform=StateTransform(recipe['view']).fit(frame);model=LogisticRegression(C=recipe['C'],**CLASSIFIER)
    with threadpool_limits(limits=2),warnings.catch_warnings():
        warnings.filterwarnings('error',category=ConvergenceWarning);model.fit(transform.transform(frame),y)
    return dict(transformer=transform,model=model,recipe=recipe,features=transform.feature_names_,
                iterations=int(model.n_iter_.max()),threads=2,classifier={'C':recipe['C'],**CLASSIFIER})


def predict_recipe(bundle,frame):
    if not len(frame):return np.empty(0,dtype=np.float64)
    with threadpool_limits(limits=2):p=bundle['model'].predict_proba(bundle['transformer'].transform(frame))[:,1]
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise FloatingPointError('Invalid state probability')
    return p


def choose_inner(records):
    valid=[r for r in records if r['status']=='complete' and np.isfinite(r['ap'])]
    if not valid:raise RuntimeError('All state recipes invalid; no quarter omitted')
    return min(valid,key=lambda r:(-r['ap'],recipe_by_id(r['recipe'])['C'],recipe_by_id(r['recipe'])['complexity_order'],r['recipe']))['recipe']


def choose_final(winners,records):
    votes=Counter(winners);valid={}
    for recipe in RECIPES:
        rows=[r for r in records if r['recipe']==recipe['id']]
        if len(rows)==3 and all(r['status']=='complete' for r in rows):valid[recipe['id']]=float(np.mean([r['ap'] for r in rows]))
    if not valid:raise RuntimeError('No state recipe numerically valid all3inner quarters')
    return min(valid,key=lambda name:(-votes[name],-valid[name],recipe_by_id(name)['C'],recipe_by_id(name)['complexity_order'],name))
