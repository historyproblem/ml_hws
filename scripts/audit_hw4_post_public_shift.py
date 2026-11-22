"""Label-free domain diagnostics after Public feedback, no target AP scoring.

The binary label below denotes test membership, never inspection failure.
Site-group cross-fitting prevents overlapping sites from trivially memorizing
membership. Explicit time fields are omitted; context is a diagnostic clock.
"""
from pathlib import Path
import hashlib
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hw4_inspections.classical_features import COMPACT_BASE_COLUMNS
from hw4_inspections.classical_models import _clean

MAIN=Path('/Users/khlystov/Documents/ChatGPT/ML ДЗ')
OUT=MAIN/'analysis/hw4-post-public-20261122/shift-audit'
CACHE=MAIN/'analysis/hw4-deep-research-20261122/features.joblib'


def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    from catboost import CatBoostClassifier
    start=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=True)
    cache=joblib.load(CACHE)
    # Do not read competition labels, even for counts or descriptive slices.
    train=cache['train'].drop(columns='target');test=cache['test'];x=cache['views']['base'];cats=cache['cats']
    meta=pd.concat([train,test],ignore_index=True)
    assert 'target'not in meta and len(x)==len(meta)
    expected=json.loads(CACHE.with_name('features-meta.json').read_text())
    assert sha(CACHE)==expected['cache_sha256']
    drops={'forecast_year','forecast_quarter','cutoff_gap_days'}
    views=dict(static3=['facility_type','risk_level','area_id'],
       own21=[c for c in x if c in COMPACT_BASE_COLUMNS and c not in drops],
       own271=[c for c in x if not c.startswith('ctx_')and c not in drops],
       compact93=[c for c in x if (c in COMPACT_BASE_COLUMNS or c.startswith('ctx_'))and c not in drops])
    assert {n:len(c)for n,c in views.items()}==dict(static3=3,own21=21,own271=271,compact93=93)
    domain=np.r_[np.zeros(len(train),dtype=int),np.ones(len(test),dtype=int)]
    rows=[];importance=[];predictions=[];fold_log=[]
    cohorts=dict(all=np.arange(len(meta)),same_season=np.r_[np.flatnonzero(train.forecast_period.isin(['2022-10-01','2023-10-01'])),np.arange(len(train),len(meta))])
    for cohort,indices in cohorts.items():
        for view,columns in views.items():
            if cohort=='same_season'and view=='compact93':continue
            label=domain[indices];sites=meta.site_id.iloc[indices].to_numpy();frame=_clean(x.iloc[indices][columns], [c for c in cats if c in columns])
            categorical=[c for c in cats if c in columns]
            probability=np.empty(len(indices));fold_import=[]
            splitter=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=20261122)
            for k,(fit,valid)in enumerate(splitter.split(frame,label,groups=sites)):
                assert not set(sites[fit])&set(sites[valid])
                ratio=float((label[fit]==0).sum()/(label[fit]==1).sum())
                model=CatBoostClassifier(iterations=200,depth=3,learning_rate=.05,l2_leaf_reg=10.,
                    loss_function='Logloss',thread_count=2,random_seed=42,class_weights=[1.,ratio],
                    task_type='CPU',verbose=False,allow_writing_files=False)
                model.fit(frame.iloc[fit],label[fit],cat_features=categorical)
                p=model.predict_proba(frame.iloc[valid],thread_count=2)[:,1]
                probability[valid]=p
                fold_import.append(model.get_feature_importance())
                fold_log.append(dict(cohort=cohort,view=view,fold=k,train_n=len(fit),valid_n=len(valid),site_disjoint=True,
                    source_train_n=int((label[fit]==0).sum()),target_train_n=int((label[fit]==1).sum()),class_weight_ratio=ratio))
            auc=float(roc_auc_score(label,probability))
            source=probability[label==0]
            w=np.clip(source/np.maximum(1-source,1e-12),.25,4.);w/=w.mean()
            n=len(source)
            rows.append(dict(cohort=cohort,view=view,n=len(label),source_n=n,target_n=int(label.sum()),
                domain_oof_roc_auc=auc,weights_ESS=float(w.sum()**2/np.square(w).sum()),ESS_fraction=float(w.sum()**2/np.square(w).sum()/n),
                weight_min=float(w.min()),weight_q25=float(np.quantile(w,.25)),weight_median=float(np.median(w)),
                weight_q75=float(np.quantile(w,.75)),weight_max=float(w.max()),
                mean_domain_probability_source=float(source.mean()),mean_domain_probability_target=float(probability[label==1].mean())))
            for name,value in zip(columns,np.mean(fold_import,axis=0)):
                importance.append(dict(cohort=cohort,view=view,feature=name,domain_importance=float(value)))
            predictions.append(pd.DataFrame(dict(row_id=meta.row_id.iloc[indices].to_numpy(),site_id=sites,
                forecast_period=meta.forecast_period.iloc[indices].to_numpy(),domain=label,p_domain=probability,
                cohort=cohort,view=view)))
            print(json.dumps(dict(event='domain_complete',**rows[-1])),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'domain-summary.csv',index=False)
    pd.DataFrame(importance).to_csv(OUT/'domain-importance.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(OUT/'domain-oof.csv',index=False)
    pd.DataFrame(fold_log).to_csv(OUT/'domain-folds.csv',index=False)
    diagnostics=[]
    for view in ['static3','own21','own271']:
        for name in views[view]:
            a=x[name].iloc[:len(train)];b=x[name].iloc[len(train):]
            if name in cats:
                pa=a.fillna('<missing>').value_counts(normalize=True);pb=b.fillna('<missing>').value_counts(normalize=True)
                keys=pa.index.union(pb.index)
                distance=float(np.abs(pa.reindex(keys,fill_value=0)-pb.reindex(keys,fill_value=0)).sum()/2)
                diagnostics.append(dict(view=view,feature=name,kind='categorical_TV',distance=distance))
            else:
                av=pd.to_numeric(a,errors='raise').to_numpy(float);bv=pd.to_numeric(b,errors='raise').to_numpy(float)
                valid_a=av[np.isfinite(av)];valid_b=bv[np.isfinite(bv)]
                distance=float(ks_2samp(valid_a,valid_b).statistic)if len(valid_a)and len(valid_b)else None
                diagnostics.append(dict(view=view,feature=name,kind='numeric_KS',distance=distance,
                    source_missing=float((~np.isfinite(av)).mean()),target_missing=float((~np.isfinite(bv)).mean())))
    pd.DataFrame(diagnostics).to_csv(OUT/'univariate-shift.csv',index=False)
    receipt=dict(status='COMPLETE',source_sha256=sha(Path(__file__)),plan_sha256=sha(OUT/'plan.json'),feature_cache_sha256=sha(CACHE),
        task_labels_used=False,task_AP_computed=0,new_2024q2_q3_task_metrics=0,kaggle_uploads=0,
        actual_domain_models=28,group='site_id',model_threads=2,seconds=time.monotonic()-start,
        caveat='Domain AUROC measures covariates and membership, not failure-prediction quality. Full-view context may merely identify the snapshot.' )
    (OUT/'completion.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
