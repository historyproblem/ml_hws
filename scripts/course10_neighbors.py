"""Course4 kNN regression control; one neighbor query shared by fixed settings."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
from spend_baselines.suite_data import CachedData,atomic_json,digest,split_indices
from spend_baselines.suite_models import Preprocessor
from spend_behavior.report import comparison


def neighbor_average(distance,index,target,k,weights):
    distance,index=distance[:,:k],index[:,:k]
    values=target[index]
    if weights=='uniform':return values.mean(axis=1)
    w=1/np.maximum(distance,1e-30)
    zero=distance==0
    exact=zero.any(axis=1)
    w[exact]=zero[exact]
    return (values*w).sum(axis=1)/w.sum(axis=1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/course10_neighbors.json'))
    p.add_argument('--cache',type=Path,default=Path('outputs/behavior-cpu-screen/base_cache'))
    p.add_argument('--out',type=Path,default=Path('outputs/course10-neighbors'))
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(args.config.read_text());assert cfg['model']=='KNeighborsRegressor' and cfg['seminar']==4
    assert cfg['dataset']=='D0' and cfg['features']=='F0'
    c=CachedData(args.cache);obj=c.objects;x=c.matrix('F0',np.arange(len(obj)))[0]
    identity={'config':cfg,'cache_sha256':digest(args.cache/'ready.json'),'source_sha256':digest(Path(__file__))}
    path=args.out/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==identity
    else:atomic_json(path,identity)
    names=[f'knn_{k}_{w}' for k in cfg['n_neighbors'] for w in cfg['weights']]
    results={}
    def evaluate(date,chosen):
        tr=split_indices(obj,'D0',date);vi=np.flatnonzero((obj.official&obj.as_of_date.eq(date)).to_numpy())
        assert obj.iloc[tr].target_end.max()<=pd.Timestamp(date)
        with threadpool_limits(limits=2):
            start=time.monotonic();pre=Preprocessor().fit(x[tr],np.ones(len(tr)))
            y=np.log1p(obj.iloc[tr].spend_30d.to_numpy(float));train=pre.transform(x[tr]);test=pre.transform(x[vi])
            query=NearestNeighbors(n_neighbors=max(cfg['n_neighbors']),algorithm='brute',metric='euclidean',n_jobs=2).fit(train)
            distance,index=query.kneighbors(test);elapsed=time.monotonic()-start
        for k in cfg['n_neighbors']:
            for w in cfg['weights']:
                name=f'knn_{k}_{w}'
                if name not in chosen:continue
                z=neighbor_average(distance,index,y,k,w)
                assert np.isfinite(z).all() and (z>=0).all()
                a=obj.iloc[vi][['Id','customer_id']].copy().reset_index(drop=True)
                a['date'],a['profile'],a['z'],a['target_z']=date,name,z,np.log1p(obj.iloc[vi].spend_30d.to_numpy(float))
                d=args.out/'jobs'/name/date;d.mkdir(parents=True,exist_ok=True);a.to_csv(d/'predictions.csv.gz',index=False)
                score=float(np.sqrt(np.mean((a.z-a.target_z)**2)))
                atomic_json(d/'result.json',{'k':k,'weights':w,'RMSLE':score,'n_train':len(tr),
                    'max_target_end':str(obj.iloc[tr].target_end.max()),'shared_query_seconds':elapsed,'sha256':digest(d/'predictions.csv.gz')})
                results[name,date]=a;print(name,date,f'{score:.6f}',flush=True)
    for d in cfg['tuning_dates']:evaluate(d,names)
    ranking=[]
    for n in names:
        a=pd.concat([results[n,d] for d in cfg['tuning_dates']],ignore_index=True)
        ranking.append({'profile':n,'RMSLE':float(np.sqrt(np.mean((a.z-a.target_z)**2)))})
    ranking=sorted(ranking,key=lambda x:(x['RMSLE'],x['profile']))
    atomic_json(args.out/'early_selection.json',{'ranking':ranking,'winner':ranking[0]['profile'],'uses_confirmation':False})
    for d in cfg['confirmation_dates']:evaluate(d,[ranking[0]['profile']])
    a=pd.concat([results[ranking[0]['profile'],d] for d in cfg['confirmation_dates']],ignore_index=True)
    base=pd.concat([pd.read_csv(Path('outputs/course10-final-forest/jobs/forest_features_third')/d/'42/predictions.csv.gz',dtype={'Id':str,'customer_id':str}) for d in cfg['confirmation_dates']],ignore_index=True).set_index('Id').loc[a.Id]
    a['reference_z']=base.z.to_numpy()
    report={'winner':ranking[0]['profile'],'early_ranking':ranking,'confirmation':comparison(a,2000),
        'per_date':{date:comparison(g,2000) for date,g in a.groupby('date')},'Kaggle_uploaded':False}
    atomic_json(args.out/'report.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
