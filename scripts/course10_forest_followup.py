"""Small, explicit follow-up of one course9 RandomForest; no prediction blending."""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from spend_baselines.cli import write_submission
from spend_baselines.data import load_dataset
from spend_baselines.suite_data import FEATURES,CachedData,atomic_json,digest,split_indices
from spend_baselines.suite_models import Preprocessor
from spend_behavior.report import comparison
from sklearn.ensemble import RandomForestRegressor

SEASON_COLUMNS=['year_spend30','year_days30','year_spend90','year_days90',
                'year_spend_change90','year_days_change90','year_spend_ratio30_90',
                'year_coverage30','year_coverage90']+[f'month_{m}' for m in range(1,13)]


def seasonal_features(objects,history,history_start):
    """Every used purchase day is strictly before the individual row cutoff."""
    result=np.full((len(objects),len(SEASON_COLUMNS)),np.nan,dtype=np.float32)
    coverage_start=np.datetime64(pd.Timestamp(history_start).date(),'D').astype(int)
    for i,row in enumerate(objects.itertuples()):
        cutoff=pd.Timestamp(row.as_of_date)
        t=np.datetime64(cutoff.date(),'D').astype(int)
        previous=cutoff-pd.DateOffset(years=1)
        ya=np.datetime64(previous.date(),'D').astype(int)
        h=history.customers.get(str(row.customer_id))
        dates,cumulative=(h[0],h[2]) if h is not None else (np.array([],int),np.array([0.]))
        def window(left,right):
            if left<coverage_start or right>t:return np.nan,np.nan
            lo,hi=np.searchsorted(dates,[left,right],side='left')
            return float(cumulative[hi]-cumulative[lo]),float(hi-lo)
        y30,d30=window(ya,ya+30);y90,d90=window(ya-90,ya);recent,days=window(t-90,t)
        result[i,:9]=[np.log1p(y30),np.log1p(d30),np.log1p(y90/3),np.log1p(d90/3),
            np.log1p(recent/3)-np.log1p(y90/3),np.log1p(days/3)-np.log1p(d90/3),
            np.log1p(y30)-np.log1p(y90/3),float(ya>=coverage_start and ya+30<=t),
            float(ya-90>=coverage_start and ya<=t)]
        result[i,9:]=np.eye(12,dtype=np.float32)[cutoff.month-1]
    return result


def make_matrix(cache,profile,season):
    ix=np.arange(len(cache.objects));x,_=cache.matrix(profile['features'],ix);names=list(FEATURES[profile['features']])
    if profile['season']:x=np.c_[x,season];names=names+SEASON_COLUMNS
    if np.isinf(x).any():raise ValueError('Infinite features')
    return x,names


def fit_predict(objects,x,profile,base,cutoff,vi,threads):
    tr=split_indices(objects,'D0',cutoff)
    assert objects.iloc[tr].target_end.max()<=pd.Timestamp(cutoff)
    w=np.ones(len(tr))
    if profile['half_life_days']:
        age=(pd.Timestamp(cutoff)-objects.iloc[tr].target_end).dt.total_seconds().to_numpy()/86400
        w=np.exp2(-age/profile['half_life_days'])
    start=time.monotonic()
    pre=Preprocessor().fit(x[tr],w)
    model=RandomForestRegressor(**{**base,**profile['overrides']},random_state=42,n_jobs=threads)
    model.fit(pre.transform(x[tr]),np.log1p(objects.iloc[tr].spend_30d.to_numpy(float)),sample_weight=w)
    fit_seconds=time.monotonic()-start;start=time.monotonic()
    z=np.maximum(model.predict(pre.transform(x[vi])),0)
    if not np.isfinite(z).all():raise ValueError('Invalid predictions')
    return z,pre,model,{'fit_seconds':fit_seconds,'inference_seconds':time.monotonic()-start,
        'n_train':len(tr),'max_target_end':str(objects.iloc[tr].target_end.max())}


def trial(args,cache,profile,cfg,x,date):
    directory=args.out/'jobs'/profile['name']/date;directory.mkdir(parents=True,exist_ok=True)
    marker=directory/'result.json'
    if marker.exists():
        r=json.loads(marker.read_text());assert digest(directory/'predictions.csv.gz')==r['sha256'];return
    # Reuse the exact local submitted-profile validation, no repeat fitting.
    original=Path('outputs/course10-final-forest/jobs/forest_features_third')/date/'42'
    if profile['name']=='rf_current' and (original/'result.json').exists():
        old=json.loads((original/'result.json').read_text())
        assert old['job']['settings']=={**cfg['base'],**profile['overrides']} and old['job']['seed']==42
        assert old['sha256']['predictions.csv.gz']==digest(original/'predictions.csv.gz')
        a=pd.read_csv(original/'predictions.csv.gz',dtype={'Id':str,'customer_id':str})
        metadata={'fit_seconds':old['train_seconds'],'inference_seconds':old['inference_seconds'],
                  'n_train':old['n_train'],'max_target_end':old['max_target_end_train'],'reused':True}
    else:
        vi=np.flatnonzero((cache.objects.official&cache.objects.as_of_date.eq(date)).to_numpy())
        z,_,_,metadata=fit_predict(cache.objects,x,profile,cfg['base'],date,vi,args.threads)
        a=cache.objects.iloc[vi][['Id','customer_id']].copy().reset_index(drop=True)
        a['z']=z;a['target_z']=np.log1p(cache.objects.iloc[vi].spend_30d.to_numpy(float))
    a['profile']=profile['name'];a['date']=date;a.to_csv(directory/'predictions.csv.gz',index=False)
    score=float(np.sqrt(np.mean((a.z-a.target_z)**2)))
    atomic_json(marker,{'profile':profile,'settings':{**cfg['base'],**profile['overrides']},
         'date':date,'RMSLE':score,**metadata,'sha256':digest(directory/'predictions.csv.gz')})
    print(f"{profile['name']} {date}: {score:.6f}",flush=True)


def frames(args,name,dates):
    return pd.concat([pd.read_csv(args.out/'jobs'/name/d/'predictions.csv.gz',dtype={'Id':str,'customer_id':str})
                     for d in dates],ignore_index=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/course10_forest_followup.json'))
    p.add_argument('--cache',type=Path,default=Path('outputs/behavior-cpu-screen/base_cache'))
    p.add_argument('--data',type=Path,default=Path('data/raw'))
    p.add_argument('--out',type=Path,default=Path('outputs/course10-forest-followup'))
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--refit',help='Explicit profile; no automatic submission or replacement')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(args.config.read_text())
    assert cfg['model']=='RandomForestRegressor' and cfg['dataset']=='D0' and cfg['seed']==42
    assert cfg['task']=='direct' and cfg['max_seminar']==10
    cache=CachedData(args.cache);profiles={x['name']:x for x in cfg['profiles']}
    identity={'config':cfg,'cache_sha256':digest(args.cache/'ready.json'),'source_sha256':digest(Path(__file__)),
              'input_sha256':{n:digest(args.data/n) for n in ['train.csv','test.csv','transactions.csv','sample_submission.csv']}}
    manifest=args.out/'manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())==identity,'Use a new output directory'
    else:atomic_json(manifest,identity)
    with threadpool_limits(limits=args.threads):
        season_path=args.out/'season.npy'
        if season_path.exists():season=np.load(season_path,allow_pickle=False)
        else:
            train,test,history,checks=load_dataset(args.data)
            start=min(v[0][0] for v in history.customers.values())
            season=seasonal_features(cache.objects,history,pd.Timestamp('1970-01-01')+pd.Timedelta(days=int(start)))
            np.save(season_path,season);atomic_json(args.out/'data_checks.json',checks)
        if args.refit:
            profile=profiles[args.refit];x,names=make_matrix(cache,profile,season)
            vi=np.flatnonzero(cache.objects.is_test.to_numpy());cutoff=cache.objects.iloc[vi].as_of_date.min()
            z,pre,model,meta=fit_predict(cache.objects,x,profile,cfg['base'],cutoff,vi,args.threads)
            directory=args.out/'refit'/profile['name'];directory.mkdir(parents=True,exist_ok=True)
            joblib.dump({'preprocessor':pre,'model':model},directory/'model.joblib',compress=3)
            test=cache.objects.iloc[vi].copy();sample=pd.read_csv(args.data/'sample_submission.csv',dtype={'Id':str})
            write_submission(test,z,directory/'submission.csv',sample_ids=sample.Id)
            np.savez_compressed(directory/'predictions.npz',Id=test.Id.to_numpy(str),customer_id=test.customer_id.to_numpy(str),z=z)
            atomic_json(directory/'result.json',{'profile':profile,'settings':{**cfg['base'],**profile['overrides']},
                'feature_names':names,'seed':42,'single_predictor':True,'task':'direct','max_seminar':10,
                'dataset':'D0','cutoff':str(cutoff),'Kaggle_uploaded':False,**meta,
                'sha256':{v.name:digest(v) for v in directory.iterdir() if v.name!='result.json'}})
            print('Ready:',directory/'submission.csv',flush=True);return
        for profile in profiles.values():
            x,_=make_matrix(cache,profile,season)
            for date in cfg['tuning_dates']:trial(args,cache,profile,cfg,x,date)
        ranking=[]
        for name in profiles:
            a=frames(args,name,cfg['tuning_dates']);ranking.append({'profile':name,'RMSLE':float(np.sqrt(np.mean((a.z-a.target_z)**2)))})
        ranking=sorted(ranking,key=lambda r:(r['RMSLE'],r['profile']))
        chosen=list(dict.fromkeys(['rf_current']+[r['profile'] for r in ranking[:3]]))
        atomic_json(args.out/'early_selection.json',{'ranking':ranking,'confirmation_profiles':chosen,'uses_confirmation':False})
        for name in chosen:
            profile=profiles[name];x,_=make_matrix(cache,profile,season)
            for date in cfg['confirmation_dates']:trial(args,cache,profile,cfg,x,date)
        rows=[];perdate=[]
        base=frames(args,'rf_current',cfg['confirmation_dates']).set_index('Id')
        for name in chosen:
            a=frames(args,name,cfg['confirmation_dates']);a['reference_z']=base.loc[a.Id,'z'].to_numpy()
            rows.append({'profile':name,**comparison(a,2000)})
            for date,g in a.groupby('date'):perdate.append({'profile':name,'date':date,**comparison(g,2000)})
        pd.DataFrame(ranking).to_csv(args.out/'early_scores.csv',index=False)
        pd.DataFrame(rows).to_csv(args.out/'confirmation_scores.csv',index=False)
        pd.DataFrame(perdate).to_csv(args.out/'confirmation_dates.csv',index=False)
        print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':main()
