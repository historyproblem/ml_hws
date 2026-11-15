"""Small causal history-feature study with Ridge/RandomForest, no new architectures."""
import argparse,json,runpy,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from spend_baselines.data import load_dataset
from spend_baselines.cli import write_submission
from spend_baselines.suite_data import FEATURES,CachedData,atomic_json,digest
from spend_baselines.suite_models import Preprocessor
from spend_behavior.report import comparison
from spend_course10.history_features import HISTORY_COLUMNS,LAG_COLUMNS,monthly_history_features


def frozen(path,value):
    if path.exists() and json.loads(path.read_text())!=value:raise ValueError(f'Frozen definition changed: {path}')
    atomic_json(path,value)


def prepare(args,cfg):
    base=CachedData(args.cache)
    ix=np.flatnonzero((base.objects.official|base.objects.is_test).to_numpy())
    objects=base.objects.iloc[ix].reset_index(drop=True)
    inputs={n:digest(args.data/n) for n in ['train.csv','test.csv','transactions.csv','sample_submission.csv']}
    helpers=runpy.run_path('scripts/course10_forest_followup.py',run_name='history_feature_helpers')
    sources=['scripts/run_monthly_history.py','src/spend_course10/history_features.py',
             'scripts/course10_forest_followup.py','src/spend_baselines/data.py','src/spend_baselines/suite_models.py']
    identity={'config':cfg,'input_sha256':inputs,'cache_sha256':digest(args.cache/'ready.json'),
              'source_sha256':{n:digest(n) for n in sources}}
    frozen(args.out/'manifest.json',identity)
    ready=args.out/'features.ready.json';payload=args.out/'history_features.npz'
    if ready.exists():
        r=json.loads(ready.read_text());assert r['identity']==identity and r['sha256']==digest(payload)
        with np.load(payload,allow_pickle=False) as a:
            assert np.array_equal(a['Id'],objects.Id.to_numpy(str));monthly,season=a['monthly'],a['season']
    else:
        _,_,history,checks=load_dataset(args.data)
        start=min(v[0][0] for v in history.customers.values())
        start=pd.Timestamp('1970-01-01')+pd.Timedelta(days=int(start))
        monthly=monthly_history_features(objects,history,start)
        season=helpers['seasonal_features'](objects,history,start)
        np.savez_compressed(payload,Id=objects.Id.to_numpy(str),monthly=monthly,season=season)
        atomic_json(ready,{'identity':identity,'sha256':digest(payload),'target_checks':checks,
                         'monthly_columns':HISTORY_COLUMNS,'season_columns':helpers['SEASON_COLUMNS']})
    matrices={}
    for p in cfg['profiles']:
        x=base.matrix(p['base'],ix)[0];names=list(FEATURES[p['base']])
        if p['season']:x=np.c_[x,season];names+=helpers['SEASON_COLUMNS']
        if p['history']!='none':
            cols=np.arange(len(HISTORY_COLUMNS)) if p['history']=='full' else np.arange(len(LAG_COLUMNS),len(HISTORY_COLUMNS))
            x=np.c_[x,monthly[:,cols]];names+=[HISTORY_COLUMNS[k] for k in cols]
        assert x.shape==(len(objects),len(names)) and not np.isinf(x).any()
        matrices[p['name']]=(x,names)
    return objects,matrices


def fit(objects,x,profile,cfg,cutoff,vi,threads):
    tr=np.flatnonzero((objects.official&objects.target_end.le(pd.Timestamp(cutoff))).to_numpy())
    assert objects.iloc[tr].target_end.max()<=pd.Timestamp(cutoff)
    start=time.monotonic();w=np.ones(len(tr));pre=Preprocessor().fit(x[tr],w)
    if profile['model']=='forest':model=RandomForestRegressor(**cfg['forest'],random_state=cfg['seed'],n_jobs=threads)
    else:model=Ridge(alpha=profile['alpha'])
    model.fit(pre.transform(x[tr]),np.log1p(objects.iloc[tr].spend_30d.to_numpy(float)),sample_weight=w)
    elapsed=time.monotonic()-start;start=time.monotonic()
    z=np.maximum(np.asarray(model.predict(pre.transform(x[vi])),dtype=float),0)
    assert np.isfinite(z).all()
    return z,pre,model,{'n_train':len(tr),'max_target_end':str(objects.iloc[tr].target_end.max()),
                          'fit_seconds':elapsed,'inference_seconds':time.monotonic()-start}


def trial(args,cfg,objects,matrices,profile,date):
    d=args.out/'jobs'/profile['name']/date;d.mkdir(parents=True,exist_ok=True)
    marker=d/'result.json';path=d/'predictions.csv.gz'
    if marker.exists():
        r=json.loads(marker.read_text());assert r['profile']==profile and r['sha256']==digest(path);return
    vi=np.flatnonzero((objects.official&objects.as_of_date.eq(date)).to_numpy())
    if profile['name']=='season_control':
        old=args.control/'jobs/rf_season'/date;r=json.loads((old/'result.json').read_text())
        m=json.loads((args.control/'manifest.json').read_text());own=json.loads((args.out/'manifest.json').read_text())
        assert m['cache_sha256']==own['cache_sha256'] and m['input_sha256']==own['input_sha256']
        assert r['settings']==cfg['forest'] and r['sha256']==digest(old/'predictions.csv.gz')
        a=pd.read_csv(old/'predictions.csv.gz',dtype={'Id':str,'customer_id':str})
        metadata={'n_train':r['n_train'],'max_target_end':r['max_target_end'],'fit_seconds':r['fit_seconds'],
                  'inference_seconds':r['inference_seconds'],'reused':True}
    else:
        z,_,_,metadata=fit(objects,matrices[profile['name']][0],profile,cfg,date,vi,args.threads)
        a=objects.iloc[vi][['Id','customer_id']].copy().reset_index(drop=True)
        a['z']=z;a['target_z']=np.log1p(objects.iloc[vi].spend_30d.to_numpy(float))
    assert set(a.Id)==set(objects.iloc[vi].Id.astype(str))
    a['date']=date;a['profile']=profile['name'];a.to_csv(path,index=False)
    score=float(np.sqrt(np.mean((a.z-a.target_z)**2)))
    atomic_json(marker,{'profile':profile,'date':date,'RMSLE':score,**metadata,'sha256':digest(path)})
    print(f"{profile['name']} {date}: {score:.6f}",flush=True)


def frame(args,name,dates):
    return pd.concat([pd.read_csv(args.out/'jobs'/name/d/'predictions.csv.gz',dtype={'Id':str,'customer_id':str})
                      for d in dates],ignore_index=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/course10_monthly_history.json'))
    p.add_argument('--cache',type=Path,default=Path('outputs/behavior-cpu-screen/base_cache'))
    p.add_argument('--data',type=Path,default=Path('data/raw'))
    p.add_argument('--control',type=Path,default=Path('outputs/course10-forest-followup'))
    p.add_argument('--out',type=Path,default=Path('outputs/course10-monthly-history'))
    p.add_argument('--threads',type=int,default=2);p.add_argument('--refit')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);cfg=json.loads(args.config.read_text())
    assert cfg['dataset']=='D0' and cfg['task']=='direct' and cfg['seed']==42 and cfg['max_seminar']==10
    with threadpool_limits(limits=args.threads):
        objects,matrices=prepare(args,cfg);profiles={x['name']:x for x in cfg['profiles']}
        if args.refit:
            p=profiles[args.refit];x,names=matrices[args.refit]
            vi=np.flatnonzero(objects.is_test.to_numpy());cutoff=objects.iloc[vi].as_of_date.min()
            z,pre,model,meta=fit(objects,x,p,cfg,cutoff,vi,args.threads)
            d=args.out/'refit'/args.refit;d.mkdir(parents=True,exist_ok=True)
            sample=pd.read_csv(args.data/'sample_submission.csv',dtype={'Id':str})
            write_submission(objects.iloc[vi],z,d/'submission.csv',sample_ids=sample.Id)
            np.savez_compressed(d/'predictions.npz',Id=objects.iloc[vi].Id.to_numpy(str),customer_id=objects.iloc[vi].customer_id.to_numpy(str),z=z)
            joblib.dump({'preprocessor':pre,'model':model},d/'model.joblib',compress=3)
            atomic_json(d/'result.json',{'profile':p,'settings':cfg['forest'] if p['model']=='forest' else {'alpha':p['alpha']},
                'seed':42,'task':'direct','single_predictor':True,'max_seminar':10,'dataset':'D0','cutoff':str(cutoff),
                'feature_names':names,**meta,'Kaggle_uploaded':False,
                'sha256':{q.name:digest(q) for q in d.iterdir() if q.name!='result.json'}})
            print('READY',d/'submission.csv',flush=True);return
        for p in profiles.values():
            for date in cfg['tuning_dates']:trial(args,cfg,objects,matrices,p,date)
        ranking=[]
        for name,p in profiles.items():
            a=frame(args,name,cfg['tuning_dates']);ranking.append({'profile':name,'model':p['model'],'RMSLE':float(np.sqrt(np.mean((a.z-a.target_z)**2)))})
        ranking=sorted(ranking,key=lambda r:(r['RMSLE'],r['profile']))
        topforest=[r['profile'] for r in ranking if r['model']=='forest' and r['profile']!='season_control'][:2]
        topridge=[r['profile'] for r in ranking if r['model']=='ridge'][:1]
        chosen=['season_control']+topforest+topridge
        frozen(args.out/'selection.json',{'ranking':ranking,'confirmation_profiles':chosen,'uses_confirmation':False})
        pd.DataFrame(ranking).to_csv(args.out/'early_scores.csv',index=False)
        for name in chosen:
            for date in cfg['confirmation_dates']:trial(args,cfg,objects,matrices,profiles[name],date)
        late=[];dates=[];pooled=[]
        for name in chosen:
            a=frame(args,name,cfg['tuning_dates']+cfg['confirmation_dates'])
            b=frame(args,'season_control',cfg['tuning_dates']+cfg['confirmation_dates']).set_index('Id').loc[a.Id]
            a['reference_z']=b.z.to_numpy()
            late.append({'profile':name,**comparison(a[a.date.isin(cfg['confirmation_dates'])],2000)})
            pooled.append({'profile':name,**comparison(a,2000)})
            for date,g in a.groupby('date'):dates.append({'profile':name,'date':date,**comparison(g,2000)})
        pd.DataFrame(late).to_csv(args.out/'confirmation_scores.csv',index=False)
        pd.DataFrame(pooled).to_csv(args.out/'pooled5_scores.csv',index=False)
        pd.DataFrame(dates).to_csv(args.out/'per_date_scores.csv',index=False)
        print(pd.DataFrame(late).to_string(index=False),flush=True)


if __name__=='__main__':main()
