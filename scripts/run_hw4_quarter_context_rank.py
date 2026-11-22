"""Two fixed rank-context CatBoost ablations after Public feedback.

Old3 task AP only; no recent diagnostic or Public upload. Rank normalization
uses unlabeled per-quarter query cohorts. The experiment is exploratory.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from hw4_inspections.quarter_context_rank import quarter_rank_context,columns_for_view
from hw4_inspections.classical_models import fit_model,predict_model
from hw4_inspections.next_round import QUARTERS,REQUIRED,evaluate_candidates

MAIN=Path('/Users/khlystov/Documents/ChatGPT/ML ДЗ')
BASE=MAIN/'analysis/hw4-post-public-20261122'
OUT=BASE/'quarter-rank-context'
CACHE=MAIN/'analysis/hw4-deep-research-20261122/features.joblib'
SEEDS=(42,17,2026)
SPEC=dict(family='catboost',feature_set='context',add_site=False,recency=False,max_iterations=500,tune_iterations=False,
 params=dict(depth=5,learning_rate=.04,l2_leaf_reg=10.,boosting_type='Plain',bootstrap_type='MVS',subsample=.8,rsm=1.,border_count=254))


def sha(p):
    with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def csv(p):return pd.read_csv(p,float_precision='round_trip')
def write(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()


def job(signature,name,q,seed,train,frame,cats):
    full=q=='full';fit=np.arange(len(train))if full else np.flatnonzero(train.forecast_period.lt(q))
    valid=np.arange(len(train),len(frame))if full else np.flatnonzero(train.forecast_period.eq(q))
    identity=dict(signature=signature,case=name,quarter=q,seed=seed,train_rows=len(fit),predict_rows=len(valid),features=list(frame))
    path=OUT/'jobs'/f'{name}__{q}__s{seed}'
    if path.exists():
        marker=read(path/'complete.json')
        if marker['identity']!=identity or not marker['reload_exact']or any(sha(path/n)!=h for n,h in marker['sha256'].items()):raise ValueError('Changed rank-context checkpoint')
        return np.load(path/'prediction.npy',allow_pickle=False)
    print(json.dumps(dict(event='fit_start',**{k:v for k,v in identity.items()if k!='features'})),flush=True)
    bundle=fit_model(SPEC,frame.iloc[fit],train.target.iloc[fit],cats,seed,2,iterations=500)
    p=predict_model(bundle,frame.iloc[valid]);path.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=path.parent,prefix='.pending-')as tmp:
        d=Path(tmp)/'job';d.mkdir();joblib.dump(bundle,d/'model.joblib',compress=1)
        if not np.array_equal(p,predict_model(joblib.load(d/'model.joblib'),frame.iloc[valid])):raise ValueError('Rank-model reload differs')
        np.save(d/'prediction.npy',p,allow_pickle=False)
        write(d/'complete.json',dict(identity=identity,reload_exact=True,new_recent_AP=0,
            sha256={n:sha(d/n)for n in ['model.joblib','prediction.npy']}))
        os.rename(d,path)
    return p


def assess(stage,meta,context,vectors):
    result=evaluate_candidates(meta,{'context_3seed':context,**vectors},reference='context_3seed',bootstrap=2000,seed=20261122)
    s=result['summary'].copy();s['eligible'] &= s.worst_quarter_delta.ge(-.005)
    s.to_csv(OUT/f'{stage}-summary.csv',index=False);result['quarter_scores'].to_csv(OUT/f'{stage}-quarters.csv',index=False)
    pd.concat([meta,pd.DataFrame({'context_3seed':context,**vectors})],axis=1).to_csv(OUT/f'{stage}-oof.csv',index=False)
    return s


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cache=joblib.load(CACHE);train,test,sample,base,cats=cache['train'],cache['test'],cache['sample'],cache['views']['base'],cache['cats']
    meta_rows=pd.concat([train.drop(columns='target'),test],ignore_index=True)
    rank=quarter_rank_context(base,meta_rows.forecast_period)
    views={v:rank[columns_for_view(rank,v)]for v in ['compact94','full346']}
    original94=base[columns_for_view(base,'compact94')]
    anchor_path=MAIN/'analysis/hw4-next-round-20261121/baseline_oof.csv'
    matched_path=MAIN/'analysis/hw4-six-hour-20261122/events/matched42-oof.csv'
    plan=dict(format='hw4-post-public-fixed-rank-context-1',views={k:list(v)for k,v in views.items()},
      source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),**{n:sha(ROOT/'src/hw4_inspections'/n)for n in ['quarter_context_rank.py','classical_models.py','classical_features.py','next_round.py','classical_analysis.py']}},
      feature_cache_sha256=sha(CACHE),baseline_mean3_sha256=sha(anchor_path),matched_seed42_sha256=sha(matched_path),
      raw_sha256={n:sha(MAIN/'data/hw4/raw'/n)for n in ['train.csv','test_student.csv','inspection_history.csv','sample_submission.csv']},
      spec=SPEC,grouping='Existing unlabeled query rows within exact forecast_period. Average percentile rank; NaN unchanged. No pooled history, no target or rawID model feature.',
      calendar='Legacy derived fields are retained exactly in each fixed view. Compact94 omits year/gap by its definition; full346 retains original3 calendar fields.',
      quarters=list(QUARTERS),screen_seed=42,confirm_seeds=[17,2026],
      controls='Rawfull346 seed42 existing matched-eventOOF; rawcompact94 exactly matched new3fits (rawcompact96 is NOT identical). No control selected as a new candidate.',
      gate='Each ranked raw/fixed75%context+.25rank probability:mean old delta>0,>=2nonnegative quarters,worst>=-.005; then samecase3seedconfirmationandgate. Fullonlypassed.',
      full='All3 fixed seeds,exact500iterations; at most one best eligible variant per passing view.',
      new_recent_AP=0,kaggle_uploads=0,interpretation='Exploratory after Public feedback; old development reused. Domain separation is not evidence of causal failure or future gain.')
    plan['signature']=digest(plan);sig=plan['signature']
    frozen=OUT/'freeze.json'
    if frozen.exists():
        if read(frozen)!=plan:raise ValueError('Changed post-Public rank-context recipe/source')
    else:write(frozen,plan)
    meta=pd.concat([train.loc[train.forecast_period.eq(q),list(REQUIRED)]for q in QUARTERS],ignore_index=True)
    old=csv(anchor_path).set_index('row_id').loc[meta.row_id].reset_index();pd.testing.assert_frame_equal(meta,old[list(REQUIRED)],check_dtype=False)
    context=old.prediction.to_numpy();matched=csv(matched_path).set_index('row_id').loc[meta.row_id].reset_index()
    pd.testing.assert_frame_equal(meta,matched[list(REQUIRED)],check_dtype=False)
    values={};ranked={}
    for name,frame in views.items():
        vectors=[job(sig,name,q,42,train,frame,[c for c in cats if c in frame])for q in QUARTERS]
        ranked[name]=np.concatenate(vectors)
        values[name+'__raw']=ranked[name];values[name+'__blend75']=.75*context+.25*ranked[name]
    controls=np.concatenate([job(sig,'rawcompact94_control',q,42,train,original94,[c for c in cats if c in original94])for q in QUARTERS])
    pd.concat([meta,pd.DataFrame(dict(rawfull346_seed42=matched.matched_context_seed42.to_numpy(),rawcompact94_seed42=controls,**ranked))],axis=1).to_csv(OUT/'matched-controls-oof.csv',index=False)
    screen=assess('screen',meta,context,values)
    files=[]
    for name,frame in views.items():
        if not screen.loc[screen.model.str.startswith(name+'__'),'eligible'].any():continue
        all_seeds=[ranked[name]]
        for seed in [17,2026]:all_seeds.append(np.concatenate([job(sig,name,q,seed,train,frame,[c for c in cats if c in frame])for q in QUARTERS]))
        mean=np.mean(all_seeds,axis=0)
        confirmed=assess('confirmed-'+name,meta,context,{name+'__raw':mean,name+'__blend75':.75*context+.25*mean})
        eligible=confirmed.loc[confirmed.eligible].sort_values(['mean_ap','model'],ascending=[False,True],kind='stable')
        if not len(eligible):continue
        choice=eligible.iloc[0];full=np.mean([job(sig,name,'full',s,train,frame,[c for c in cats if c in frame])for s in SEEDS],axis=0)
        if choice.model.endswith('__blend75'):
            anchor=csv(MAIN/'outputs/hw4/final-20261120/submission_04_context_cb.csv').set_index('row_id').loc[test.row_id,'target'].to_numpy()
            full=.75*anchor+.25*full
        output=sample[['row_id']].copy();output['target']=pd.Series(full,index=test.row_id).loc[sample.row_id].to_numpy()
        path=OUT/f'submission_rankctx_{choice.model}.csv';output.to_csv(path,index=False)
        if not np.isfinite(output.target).all()or not output.target.between(0,1).all():raise ValueError('Invalid fullrank probabilities')
        files.append(dict(file=path.name,sha256=sha(path),rows=len(output),old_mean_ap=float(choice.mean_ap),public_ap=None))
    result=dict(complete=True,signature=sig,source_jobs=len(list((OUT/'jobs').glob('*/complete.json'))),files=files,
      new_recent_AP=0,kaggle_uploads=0,post_public_exploratory=True)
    write(OUT/'manifest.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
