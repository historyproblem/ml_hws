"""One fixed3:1 same-season sample-weighting ablation of current reference CB.

Exploratory after Public feedback. All three old quarters are always reported;
the target-like oldQ4 must additionally be nonnegative before confirmation/full.
No recent-quarter AP, parameter search, new blend weights or Kaggle upload.
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
from hw4_inspections.classical_models import fit_model,predict_model
from hw4_inspections.next_round import QUARTERS,REQUIRED,evaluate_candidates
from run_hw4_quarter_context_rank import SPEC,sha,read,write,digest,csv

MAIN=Path('/Users/khlystov/Documents/ChatGPT/ML ДЗ')
OUT=MAIN/'analysis/hw4-post-public-20261122/same-season'
CACHE=MAIN/'analysis/hw4-deep-research-20261122/features.joblib'
SEEDS=(42,17,2026)


def weights(periods,q):
    p=pd.PeriodIndex(periods,freq='Q');target=pd.Period(q,freq='Q')
    if len(p)==0 or np.any(p.asi8>=target.ordinal):raise ValueError('Only past training quarters may enter weights')
    w=np.where(p.quarter==target.quarter,3.,1.)
    return w/w.mean()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cache=joblib.load(CACHE);train,test,sample,frame,cats=cache['train'],cache['test'],cache['sample'],cache['views']['base'],cache['cats']
    assert set(test.forecast_period)=={'2024-10-01'} and len(frame.columns)==346
    anchor_path=MAIN/'analysis/hw4-next-round-20261121/baseline_oof.csv'
    plan=dict(format='hw4-post-public-single-same-season-1',spec=SPEC,features=list(frame),cats=cats,
      source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
        'scripts/run_hw4_quarter_context_rank.py':sha(ROOT/'scripts/run_hw4_quarter_context_rank.py'),
        **{n:sha(ROOT/'src/hw4_inspections'/n)for n in ['classical_models.py','next_round.py','classical_analysis.py']}},
      cache_sha256=sha(CACHE),anchor_sha256=sha(anchor_path),raw_sha256={n:sha(MAIN/'data/hw4/raw'/n)for n in ['train.csv','test_student.csv','inspection_history.csv','sample_submission.csv']},
      sample_weights='Exactly3 for past rows whose forecast quarter-of-year matches the query quarter;1 for others; normalize arithmetic mean1. Dates only, no target. Full query is2024Q4.',
      quarters=list(QUARTERS),seed42_screen=True,confirmation_seeds=[17,2026],
      variants=['raw','.75*incumbentC+.25*seasonCB'],
      gate='Old mean delta>0,>=2nonnegative quarters,worst>=-.005 AND old2023Q4 delta>=0; same rule again after all3 seeds. Do not drop other quarters.',
      full='Only if stronger gate passes; same500iterations/spec, all3seeds, at mostonehighesteligiblefixedvariant.',
      prior_difference='NOT half-life/recency weighting; no calendar capacity/view changes; one fixed3:1 amplitude, no grid.',
      interpretation='Exploratory postPublic olddev test; nominal intervals, no freshholdout claim.',new_recent_AP=0,kaggle_uploads=0)
    plan['signature']=digest(plan);sig=plan['signature']
    if (OUT/'freeze.json').exists():
        if read(OUT/'freeze.json')!=plan:raise ValueError('Changed fixed same-season experiment')
    else:write(OUT/'freeze.json',plan)
    def job(q,seed):
        full=q=='full';reference='2024-10-01'if full else q
        fit=np.arange(len(train))if full else np.flatnonzero(train.forecast_period.lt(q))
        valid=np.arange(len(train),len(frame))if full else np.flatnonzero(train.forecast_period.eq(q))
        w=weights(train.forecast_period.iloc[fit],reference)
        identity=dict(signature=sig,period=q,seed=seed,train_rows=len(fit),predict_rows=len(valid),
          weights_sha256=hashlib.sha256(w.tobytes()).hexdigest())
        path=OUT/'jobs'/f'{q}__s{seed}'
        if path.exists():
            m=read(path/'complete.json')
            if m['identity']!=identity or not m['reload_exact']or any(sha(path/n)!=h for n,h in m['sha256'].items()):raise ValueError('Changed same-season checkpoint')
            return np.load(path/'prediction.npy',allow_pickle=False)
        print(json.dumps(dict(event='fit_start',**identity)),flush=True)
        bundle=fit_model(SPEC,frame.iloc[fit],train.target.iloc[fit],cats,seed,2,iterations=500,sample_weight=w)
        p=predict_model(bundle,frame.iloc[valid]);path.parent.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=path.parent,prefix='.pending-')as temporary:
            d=Path(temporary)/'job';d.mkdir();joblib.dump(bundle,d/'model.joblib',compress=1)
            if not np.array_equal(p,predict_model(joblib.load(d/'model.joblib'),frame.iloc[valid])):raise ValueError('Same-season reload differs')
            np.save(d/'prediction.npy',p,allow_pickle=False)
            write(d/'complete.json',dict(identity=identity,reload_exact=True,metrics_recent=0,
              weights=dict(mean=float(w.mean()),sum=float(w.sum()),min=float(w.min()),max=float(w.max()),ESS=float(w.sum()**2/np.square(w).sum())),
              sha256={n:sha(d/n)for n in ['model.joblib','prediction.npy']}))
            os.rename(d,path)
        return p
    meta=pd.concat([train.loc[train.forecast_period.eq(q),list(REQUIRED)]for q in QUARTERS],ignore_index=True)
    old=csv(anchor_path).set_index('row_id').loc[meta.row_id].reset_index()
    pd.testing.assert_frame_equal(meta,old[list(REQUIRED)],check_dtype=False)
    anchor=old.prediction.to_numpy()
    def assess(stage,p):
        vs=dict(context_3seed=anchor,same_season_raw=p,same_season_blend75=.75*anchor+.25*p)
        r=evaluate_candidates(meta,vs,reference='context_3seed',bootstrap=2000,seed=20261122)
        s=r['summary'].copy();s['q4_nonnegative']=[bool(r['quarter_scores'].loc[(r['quarter_scores'].model==name)&(r['quarter_scores'].forecast_period=='2023-10-01'),'delta_vs_reference'].iloc[0]>=0)for name in s.model]
        s['eligible'] &= s.worst_quarter_delta.ge(-.005)&s.q4_nonnegative
        s.to_csv(OUT/f'{stage}-summary.csv',index=False);r['quarter_scores'].to_csv(OUT/f'{stage}-quarters.csv',index=False)
        pd.concat([meta,pd.DataFrame(vs)],axis=1).to_csv(OUT/f'{stage}-oof.csv',index=False)
        return s
    first=np.concatenate([job(q,42)for q in QUARTERS]);screen=assess('screen',first);files=[]
    if screen.eligible.any():
        all3=[first,*[np.concatenate([job(q,s)for q in QUARTERS])for s in [17,2026]]]
        confirmed=assess('confirmed',np.mean(all3,axis=0))
        good=confirmed.loc[confirmed.eligible].sort_values(['mean_ap','model'],ascending=[False,True],kind='stable')
        if len(good):
            chosen=good.iloc[0];full=np.mean([job('full',s)for s in SEEDS],axis=0)
            if chosen.model=='same_season_blend75':
                context=csv(MAIN/'outputs/hw4/final-20261120/submission_04_context_cb.csv').set_index('row_id').loc[test.row_id,'target'].to_numpy()
                full=.75*context+.25*full
            output=sample[['row_id']].copy();output['target']=pd.Series(full,index=test.row_id).loc[sample.row_id].to_numpy()
            path=OUT/f'submission_{chosen.model}.csv';output.to_csv(path,index=False)
            files.append(dict(file=path.name,sha256=sha(path),rows=len(output),old_mean_ap=float(chosen.mean_ap),public_ap=None))
    result=dict(complete=True,signature=sig,source_jobs=len(list((OUT/'jobs').glob('*/complete.json'))),files=files,
      new_recent_AP=0,kaggle_uploads=0,post_public_exploratory=True)
    write(OUT/'manifest.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
