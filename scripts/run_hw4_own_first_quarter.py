"""One fixed target-aligned own-history feature block, old3 validation only."""
from pathlib import Path
import json
import os
import sys
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from hw4_inspections.own_first_quarter import FEATURES,build_own_first_quarter
from hw4_inspections.classical_models import fit_model,predict_model
from hw4_inspections.next_round import QUARTERS,REQUIRED,evaluate_candidates
from run_hw4_quarter_context_rank import SPEC,sha,read,write,digest,csv

MAIN=Path('/Users/khlystov/Documents/ChatGPT/ML ДЗ')
OUT=MAIN/'analysis/hw4-post-public-20261122/own-first-quarter'
CACHE=MAIN/'analysis/hw4-deep-research-20261122/features.joblib'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cache=joblib.load(CACHE);train,test,sample,base,cats=cache['train'],cache['test'],cache['sample'],cache['views']['base'],cache['cats']
    rows=pd.concat([train.drop(columns='target'),test],ignore_index=True)
    h=pd.read_csv(MAIN/'data/hw4/raw/inspection_history.csv')
    extra=build_own_first_quarter(rows,h)
    if list(extra)!=list(FEATURES)or len(extra)!=len(base)or len(set(FEATURES)&set(base)):raise ValueError('Wrong own-first-quarter view')
    feature_file=OUT/'features-own-firstq.joblib'
    if feature_file.exists():pd.testing.assert_frame_equal(extra,joblib.load(feature_file),check_exact=True)
    else:joblib.dump(extra,feature_file,compress=1)
    frame=pd.concat([base,extra],axis=1)
    assert len(frame.columns)==357 and set(frame.columns).isdisjoint({'row_id','site_id','forecast_period','history_before','target'})
    anchor_path=MAIN/'analysis/hw4-next-round-20261121/baseline_oof.csv'
    plan=dict(format='hw4-post-public-own-first-quarter-1',spec=SPEC,features=list(frame),new_features=list(FEATURES),
      source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
        'scripts/run_hw4_quarter_context_rank.py':sha(ROOT/'scripts/run_hw4_quarter_context_rank.py'),
        **{n:sha(ROOT/'src/hw4_inspections'/n)for n in ['own_first_quarter.py','seasonal_context.py','classical_models.py','next_round.py','classical_analysis.py']}},
      cache_sha256=sha(CACHE),own_feature_matrix_sha256=sha(feature_file),anchor_sha256=sha(anchor_path),
      raw_sha256={n:sha(MAIN/'data/hw4/raw'/n)for n in ['train.csv','test_student.csv','inspection_history.csv','sample_submission.csv']},
      feature_rules='Exact forecast snapshot;week<rowcutoff;fullyendedquarterend<=cutoff;quarterstartMonday>=globalearliestobservedweek. Existing_first_routines excludes unknown/conflicting/boundary-earliest cells without later substitution. Only ownsite results/load; zero support counts and NaNmeans. Observed envelope is not completeness proof.',
      distinction='Original274 has allRoutinecount/failrate/lastRoutine but no completed-calendar-first cohorts,last2/4routine means or same-season OWNsummary. Previousseasonal12 was PEERcontext,notthis.',
      seeds=[42,17,2026],quarters=list(QUARTERS),variants=['raw','.75C+.25ownfirstq'],
      gate='Oldmean>C,>=2nonnegativequarters,worst>=-.005 ANDold2023Q4delta>=0 beforeconfirmationandagainbeforefull. No exclusionofotherquarters.',
      full='Onlypassedvariant;all3seeds,500fixedtrees,nohyperparameterorweightsearch.',
      new_recent_AP=0,kaggle_uploads=0,post_public_exploratory=True)
    plan['signature']=digest(plan);sig=plan['signature']
    if (OUT/'freeze.json').exists():
        if read(OUT/'freeze.json')!=plan:raise ValueError('Changed own-first-quarter feature hypothesis')
    else:write(OUT/'freeze.json',plan)
    # Availability statistics only; no official current-row target is inspected.
    coverage=[]
    for q,idx in rows.groupby('forecast_period',sort=True).groups.items():
        e=extra.loc[idx]
        coverage.append(dict(forecast_period=q,n=len(e),has_completed_first=float(e.own_firstq_n.gt(0).mean()),
          has_same_season=float(e.own_firstq_same_season_n.gt(0).mean()),mean_past_first_support=float(e.own_firstq_n.mean())))
    pd.DataFrame(coverage).to_csv(OUT/'feature-coverage.csv',index=False)
    def job(q,seed):
        full=q=='full';fit=np.arange(len(train))if full else np.flatnonzero(train.forecast_period.lt(q))
        valid=np.arange(len(train),len(frame))if full else np.flatnonzero(train.forecast_period.eq(q))
        identity=dict(signature=sig,period=q,seed=seed,train_rows=len(fit),predict_rows=len(valid))
        path=OUT/'jobs'/f'{q}__s{seed}'
        if path.exists():
            m=read(path/'complete.json')
            if m['identity']!=identity or not m['reload_exact']or any(sha(path/n)!=v for n,v in m['sha256'].items()):raise ValueError('Changed ownfirst checkpoint')
            return np.load(path/'prediction.npy',allow_pickle=False)
        print(json.dumps(dict(event='fit_start',**identity)),flush=True)
        bundle=fit_model(SPEC,frame.iloc[fit],train.target.iloc[fit],cats,seed,2,iterations=500)
        p=predict_model(bundle,frame.iloc[valid]);path.parent.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=path.parent,prefix='.pending-')as temporary:
            d=Path(temporary)/'job';d.mkdir();joblib.dump(bundle,d/'model.joblib',compress=1)
            if not np.array_equal(p,predict_model(joblib.load(d/'model.joblib'),frame.iloc[valid])):raise ValueError('Ownfirst reload differs')
            np.save(d/'prediction.npy',p,allow_pickle=False)
            write(d/'complete.json',dict(identity=identity,reload_exact=True,new_recent_AP=0,
              sha256={n:sha(d/n)for n in ['model.joblib','prediction.npy']}));os.rename(d,path)
        return p
    meta=pd.concat([train.loc[train.forecast_period.eq(q),list(REQUIRED)]for q in QUARTERS],ignore_index=True)
    old=csv(anchor_path).set_index('row_id').loc[meta.row_id].reset_index()
    pd.testing.assert_frame_equal(meta,old[list(REQUIRED)],check_dtype=False);anchor=old.prediction.to_numpy()
    def assess(stage,p):
        values=dict(context_3seed=anchor,own_firstq_raw=p,own_firstq_blend75=.75*anchor+.25*p)
        r=evaluate_candidates(meta,values,reference='context_3seed',bootstrap=2000,seed=20261122)
        s=r['summary'].copy();s['q4_nonnegative']=[bool(r['quarter_scores'].loc[(r['quarter_scores'].model==n)&(r['quarter_scores'].forecast_period=='2023-10-01'),'delta_vs_reference'].iloc[0]>=0)for n in s.model]
        s['eligible'] &= s.worst_quarter_delta.ge(-.005)&s.q4_nonnegative
        s.to_csv(OUT/f'{stage}-summary.csv',index=False);r['quarter_scores'].to_csv(OUT/f'{stage}-quarters.csv',index=False)
        pd.concat([meta,pd.DataFrame(values)],axis=1).to_csv(OUT/f'{stage}-oof.csv',index=False)
        return s
    first=np.concatenate([job(q,42)for q in QUARTERS]);screen=assess('screen',first);files=[]
    if screen.eligible.any():
        all3=[first,*[np.concatenate([job(q,s)for q in QUARTERS])for s in [17,2026]]]
        confirmed=assess('confirmed',np.mean(all3,axis=0));good=confirmed.loc[confirmed.eligible].sort_values(['mean_ap','model'],ascending=[False,True],kind='stable')
        if len(good):
            chosen=good.iloc[0];full=np.mean([job('full',s)for s in [42,17,2026]],axis=0)
            if chosen.model=='own_firstq_blend75':
                context=csv(MAIN/'outputs/hw4/final-20261120/submission_04_context_cb.csv').set_index('row_id').loc[test.row_id,'target'].to_numpy()
                full=.75*context+.25*full
            output=sample[['row_id']].copy();output['target']=pd.Series(full,index=test.row_id).loc[sample.row_id].to_numpy()
            path=OUT/f'submission_{chosen.model}.csv';output.to_csv(path,index=False)
            files.append(dict(file=path.name,sha256=sha(path),rows=len(output),old_mean_ap=float(chosen.mean_ap),public_ap=None))
    result=dict(complete=True,signature=sig,source_jobs=len(list((OUT/'jobs').glob('*/complete.json'))),files=files,new_recent_AP=0,kaggle_uploads=0,post_public_exploratory=True)
    write(OUT/'manifest.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
