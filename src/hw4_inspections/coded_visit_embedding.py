"""Learn a single out-of-site score of a past routine's recorded codes."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .additive_search import CLASSIFIER
from .event_learning import FOLDS, site_fold
from .features import RESULTS, _codes

FEATURES = ("last_routine_code_logit", "last3_routine_code_logit_mean", "all_routine_code_logit_mean")
HIST_FIELDS = ("forecast_period","site_id","inspection_week","inspection_type","result","violation_codes")
TEACHER = dict(C=.1, penalty="l2", solver="lbfgs", max_iter=1000, tol=1e-7, class_weight=None, fit_intercept=True)
G0 = "core_interactions_C0.01"


def routine_snapshot(history, period, cutoff):
    dt=pd.Timestamp(cutoff)
    if dt > pd.Timestamp(period):
        raise ValueError("Cutoff follows the forecast period start")
    h=history.loc[history.forecast_period.eq(period)&history.inspection_type.eq("routine"),list(HIST_FIELDS)].copy()
    h["week"]=pd.to_datetime(h.inspection_week,errors="raise")
    if h.week.isna().any() or h.site_id.isna().any(): raise ValueError("Missing routine week or site")
    h=h.loc[h.week.lt(dt)].copy()
    h["tokens"]=h.violation_codes.fillna("").astype(str).map(_codes)
    h["result"]=h.result.fillna("").astype(str)
    h=h[["site_id","week","result","tokens"]].drop_duplicates()
    if h.duplicated(["site_id","week"]).any():
        raise ValueError("Conflicting records for one routine site/week")
    h=h.sort_values(["week","site_id"],kind="stable").reset_index(drop=True)
    h["bucket"]=h.site_id.map(site_fold)
    return h


def code_matrix(tokens):
    values=np.zeros((len(tokens),45),dtype=np.float64)
    for i,ids in enumerate(tokens):
        if ids:
            values[i,np.asarray(ids,dtype=int)-1]=1.
    values[:,-1]=np.log1p(values[:,:44].sum(axis=1))/np.log(45.)
    return values


def fit_teacher(matrix,y):
    if set(np.unique(y))!={0,1}: raise ValueError("Code teacher requires both historical outcome classes")
    m=LogisticRegression(**TEACHER)
    with threadpool_limits(limits=2),warnings.catch_warnings():
        warnings.filterwarnings("error",category=ConvergenceWarning)
        m.fit(matrix,np.asarray(y,dtype=int))
    return m


def aggregate(rows,h,score):
    predicted=pd.DataFrame({"site_id":h.site_id,"score":score})
    groups=predicted.groupby("site_id",sort=False)
    values=pd.concat([groups.score.last(),groups.tail(3).groupby("site_id",sort=False).score.mean(),groups.score.mean()],axis=1)
    values.columns=list(FEATURES)
    return values.reindex(rows.site_id).set_axis(rows.index)



def apply_heads(rows,history,models):
    h=routine_snapshot(history,rows.forecast_period.iloc[0],rows.history_before.iloc[0])
    x=code_matrix(h.tokens)
    score=np.full(len(h),np.nan)
    for k,model in models.items():
        keep=h.bucket.eq(k).to_numpy()
        if keep.any(): score[keep]=model.decision_function(x[keep])
    if not np.isfinite(score).all(): raise ValueError("Saved code heads do not cover all visits")
    return aggregate(rows,h,score)


def embed(rows,history,*,auxiliary=False):
    """Keep all focal visits out of its teacher's learned code coefficients."""
    if rows.forecast_period.nunique()!=1 or rows.history_before.nunique()!=1 or not rows.site_id.is_unique:
        raise ValueError("Expected one snapshot/cutoff and unique query sites")
    period=rows.forecast_period.iloc[0]; cutoff=rows.history_before.iloc[0]
    h=routine_snapshot(history,period,cutoff)
    x=code_matrix(h.tokens)
    valid=h.result.isin(RESULTS).to_numpy()
    y=h.result.eq("fail").to_numpy(dtype=int)
    fold=h.bucket.to_numpy(dtype=int)
    if h.empty: raise ValueError("No eligible historical routines")
    # Each actual historical visit gets the output of the model which excludes
    # its whole site's SHA256 bucket. No official row fields enter a fit.
    score=np.full(len(h),np.nan);count_score=np.full(len(h),np.nan)
    models={}; evidence=[]
    for k in range(FOLDS):
        training=valid & (fold!=k);evaluation=fold==k
        if not evaluation.any(): continue
        own=set(h.loc[evaluation,"site_id"]); others=set(h.loc[training,"site_id"])
        if own&others: raise AssertionError("Historical code teacher contains a focal site")
        model=fit_teacher(x[training],y[training])
        score[evaluation]=model.decision_function(x[evaluation])
        models[k]=model
        rec={"fold":k,"known_training_visits":int(training.sum()),"training_sites":len(others),
             "heldout_historical_visits":int(evaluation.sum()),"heldout_historical_sites":len(own),
             "unknown_outcomes_excluded_from_fit":int(((~valid)&(fold!=k)).sum()),
             "n_iter":int(model.n_iter_.max()),"training_max_week":h.loc[training,"week"].max().strftime("%Y-%m-%d")}
        if auxiliary:
            comparator=fit_teacher(x[training,-1:],y[training])
            count_score[evaluation]=comparator.decision_function(x[evaluation,-1:])
        evidence.append(rec)
    if not np.isfinite(score).all(): raise ValueError("Missing historical OOB code score")
    out=aggregate(rows,h,score)
    audit={"forecast_period":period,"cutoff":cutoff,"known_training_outcomes_in_snapshot":int(valid.sum()),
           "unique_routine_visits":len(h),"routine_sites":h.site_id.nunique(),"query_rows":len(rows),
           "query_with_any_routine":int(out[FEATURES[0]].notna().sum()),"buckets":evidence}
    if auxiliary:
        if period!="2024-01-01": raise ValueError("The single auxiliary OOB diagnostic is restricted to 2024Q1")
        audit["auxiliary"]={"target":"failure of the same historical routine, all endpoints before forecast cutoff",
            "n":int(valid.sum()),"positive":int(y[valid].sum()),"code_and_count_ap":float(average_precision_score(y[valid],score[valid])),
            "count_only_ap":float(average_precision_score(y[valid],count_score[valid])),
            "used_for_stopping_or_selection":False}
    return out,models,audit


class ExtraScores:
    """Fit medians and standardization only on official training rows."""
    def fit(self,scores):
        if list(scores)!=list(FEATURES): raise ValueError("Unexpected learned score schema")
        raw=scores.to_numpy(dtype=float)
        if np.isinf(raw).any(): raise ValueError("Infinite score")
        self.medians_=np.array([np.nanmedian(c) if np.isfinite(c).any() else 0. for c in raw.T])
        self.scaler_=StandardScaler().fit(np.where(np.isnan(raw),self.medians_,raw))
        return self

    def transform(self,scores):
        if list(scores)!=list(FEATURES): raise ValueError("Unexpected learned score schema")
        raw=scores.to_numpy(dtype=float)
        if np.isinf(raw).any(): raise ValueError("Infinite score")
        mask=np.isnan(raw)
        return np.column_stack([self.scaler_.transform(np.where(mask,self.medians_,raw)),mask.astype(float)])


def check_base(bundle):
    if (bundle["recipe"]["id"]!=G0 or bundle["recipe"]["view"]!="core_interactions" or
            bundle["classifier"]!={"C":.01,**CLASSIFIER} or bundle.get("sample_weights") is not None):
        raise ValueError("Expected the exact unweighted G0 teacher bundle")


def fit_augmented(bundle,base,scores,y):
    check_base(bundle)
    mat=bundle["transformer"].transform(base)
    ext=ExtraScores().fit(scores)
    joint=np.column_stack([mat,ext.transform(scores)])
    if not np.array_equal(joint[:,:mat.shape[1]],mat):
        raise AssertionError("G0 numerical prefix changed")
    m=LogisticRegression(C=.01,**CLASSIFIER)
    with threadpool_limits(limits=2),warnings.catch_warnings():
        warnings.filterwarnings("error",category=ConvergenceWarning)
        m.fit(joint,np.asarray(y,dtype=int))
    return {"base":bundle["transformer"],"extra":ext,"model":m,"base_columns":mat.shape[1],
            "feature_names":[*bundle["feature_names"],*FEATURES,*[f"{c}_missing" for c in FEATURES]],
            "base_prefix_exact":True,"n_iter":int(m.n_iter_.max())}


def predict(bundle,base,scores):
    matrix=bundle["base"].transform(base)
    if matrix.shape[1]!=bundle["base_columns"]: raise ValueError("G0 numerical prefix shape changed")
    joint=np.column_stack([matrix,bundle["extra"].transform(scores)])
    with threadpool_limits(limits=2):
        p=np.asarray(bundle["model"].predict_proba(joint)[:,1],dtype=np.float64)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any(): raise ValueError("Invalid augmented GAM probability")
    return p
