"""Temporal snapshot augmentation for the same stable, ordinary Ridge model."""
import argparse,json,time,runpy
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from spend_baselines.data import load_dataset
from spend_baselines.cli import write_submission
from spend_baselines.suite_data import CachedData,atomic_json,digest,row_weights,split_indices
from spend_baselines.suite_models import Preprocessor
from spend_behavior.report import comparison
from spend_course10.history_features import monthly_history_features,HISTORY_COLUMNS
from run_monthly_history import frozen


def weights(objects,tr,p,cutoff):
    if p['weights']=='customer_month':w=row_weights(objects,tr,p['dataset'])
    else:
        n=int((objects.official&objects.target_end.le(cutoff)).sum())
        w=np.full(len(tr),n/len(tr))
    if p['half_life']:
        age=(pd.Timestamp(cutoff)-objects.iloc[tr].target_end).dt.total_seconds().to_numpy()/86400
        w*=np.exp2(-age/p['half_life'])
    assert np.isfinite(w).all() and (w>0).all()
    return w


def fit(objects,x,p,cfg,date,vi,threads):
    tr=split_indices(objects,p['dataset'],date);w=weights(objects,tr,p,date)
    assert objects.iloc[tr].target_end.max()<=pd.Timestamp(date)
    y=np.log1p(objects.iloc[tr].spend_30d.to_numpy(float));assert np.isfinite(y).all()
    start=time.monotonic();pre=Preprocessor().fit(x[tr],w)
    model=Ridge(alpha=cfg['alpha']).fit(pre.transform(x[tr]).astype(np.float64),y,sample_weight=w)
    elapsed=time.monotonic()-start;start=time.monotonic()
    z=np.maximum(model.predict(pre.transform(x[vi]).astype(np.float64)),0);assert np.isfinite(z).all()
    return z,pre,model,{'n_train':len(tr),'weight_sum':float(w.sum()),'max_target_end':str(objects.iloc[tr].target_end.max()),
        'fit_seconds':elapsed,'inference_seconds':time.monotonic()-start,'precision':'float64'}


def frame(args,name,dates):
    return pd.concat([pd.read_csv(args.out/'jobs'/name/d/'predictions.csv.gz',dtype={'Id':str,'customer_id':str}) for d in dates],ignore_index=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/course10_snapshot_history.json'))
    parser.add_argument('--cache',type=Path,default=Path('outputs/behavior-cpu-screen/base_cache'))
    parser.add_argument('--data',type=Path,default=Path('data/raw'));parser.add_argument('--out',type=Path,default=Path('outputs/course10-snapshot-history'))
    parser.add_argument('--threads',type=int,default=2);parser.add_argument('--refit')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True);cfg=json.loads(args.config.read_text())
    assert cfg['seed']==42 and cfg['task']=='direct' and cfg['max_seminar']==10
    c=CachedData(args.cache);obj=c.objects
    identity={'config':cfg,'cache_sha256':digest(args.cache/'ready.json'),
        'input_sha256':{n:digest(args.data/n) for n in ['train.csv','test.csv','transactions.csv','sample_submission.csv']},
        'source_sha256':{n:digest(n) for n in ['scripts/run_snapshot_history.py','src/spend_course10/history_features.py','scripts/course10_forest_followup.py','src/spend_baselines/suite_data.py']}}
    frozen(args.out/'manifest.json',identity)
    with threadpool_limits(limits=args.threads):
        ready=args.out/'features.ready.json';payload=args.out/'features.npy'
        if ready.exists():
            state=json.loads(ready.read_text());assert state['identity']==identity and state['sha256']==digest(payload)
            x=np.load(payload,allow_pickle=False);checks=state['checks']
        else:
            _,_,history,checks=load_dataset(args.data)
            known=obj[~obj.is_test].copy();reconstructed=history.reconstruct_targets(known)
            target_error=float(np.max(np.abs(reconstructed-known.spend_30d.to_numpy(float))))
            assert target_error<.011
            checks['expanded_targets_checked']=len(known);checks['expanded_target_max_error']=target_error
            helper=runpy.run_path('scripts/course10_forest_followup.py',run_name='snapshot_season_helpers')
            start=min(v[0][0] for v in history.customers.values());start=pd.Timestamp('1970-01-01')+pd.Timedelta(days=int(start))
            x=np.c_[c.matrix('F0',np.arange(len(obj)))[0],helper['seasonal_features'](obj,history,start),monthly_history_features(obj,history,start)]
            np.save(payload,x);atomic_json(ready,{'identity':identity,'sha256':digest(payload),'checks':checks})
        assert x.shape==(len(obj),83) and not np.isinf(x).any()
        window=x.copy()
        for date,group in obj.groupby('as_of_date',sort=False):
            shares=np.bincount(pd.date_range(date,periods=30).month,minlength=13)[1:]/30.
            window[group.index,24:36]=shares.astype(np.float32)
        matrices={'asof_month':x,'target_window':window};profiles={p['name']:p for p in cfg['profiles']}
        def execute(p,date):
            d=args.out/'jobs'/p['name']/date;d.mkdir(parents=True,exist_ok=True);marker=d/'result.json';path=d/'predictions.csv.gz'
            if marker.exists():
                old=json.loads(marker.read_text());assert old['profile']==p and old['sha256']==digest(path);return
            vi=np.flatnonzero((obj.official&obj.as_of_date.eq(date)).to_numpy())
            z,_,_,meta=fit(obj,matrices[p['calendar']],p,cfg,date,vi,args.threads)
            a=obj.iloc[vi][['Id','customer_id']].copy().reset_index(drop=True);a['date'],a['z'],a['target_z']=date,z,np.log1p(obj.iloc[vi].spend_30d.to_numpy(float))
            a.to_csv(path,index=False);score=float(np.sqrt(np.mean((a.z-a.target_z)**2)))
            atomic_json(marker,{'profile':p,'date':date,'RMSLE':score,**meta,'sha256':digest(path)})
            print(p['name'],date,f'{score:.6f}','train',meta['n_train'],'weight',round(meta['weight_sum'],1),flush=True)
        if args.refit:
            p=profiles[args.refit];vi=np.flatnonzero(obj.is_test.to_numpy());date=obj.iloc[vi].as_of_date.min()
            z,pre,model,meta=fit(obj,matrices[p['calendar']],p,cfg,date,vi,args.threads)
            d=args.out/'refit'/args.refit;d.mkdir(parents=True,exist_ok=True)
            joblib.dump({'preprocessor':pre,'model':model},d/'model.joblib',compress=3)
            sample=pd.read_csv(args.data/'sample_submission.csv',dtype={'Id':str});write_submission(obj.iloc[vi],z,d/'submission.csv',sample_ids=sample.Id)
            np.savez_compressed(d/'inputs_predictions.npz',Id=obj.iloc[vi].Id.to_numpy(str),customer_id=obj.iloc[vi].customer_id.to_numpy(str),X=matrices[p['calendar']][vi],z=z)
            atomic_json(d/'result.json',{'profile':p,'alpha':cfg['alpha'],'seed':42,'task':'direct','one_predictor':True,**meta,
                'Kaggle_uploaded':False,'sha256':{q.name:digest(q) for q in d.iterdir() if q.name!='result.json'}})
            print('READY',d/'submission.csv',flush=True);return
        for p in profiles.values():
            for date in cfg['tuning_dates']:execute(p,date)
        ranking=[]
        for name,p in profiles.items():
            a=frame(args,name,cfg['tuning_dates']);ranking.append({'profile':name,'control':p['control'],'RMSLE':float(np.sqrt(np.mean((a.z-a.target_z)**2)))})
        ranking=sorted(ranking,key=lambda r:(r['RMSLE'],r['profile']));chosen=['d0_control']+[r['profile'] for r in ranking if not r['control']][:2]
        frozen(args.out/'selection.json',{'ranking':ranking,'confirmation_profiles':chosen,'uses_confirmation':False})
        pd.DataFrame(ranking).to_csv(args.out/'early_scores.csv',index=False)
        for name in chosen:
            for date in cfg['confirmation_dates']:execute(profiles[name],date)
        rows=[];dates=[]
        for name in chosen:
            a=frame(args,name,cfg['confirmation_dates']);b=frame(args,'d0_control',cfg['confirmation_dates']).set_index('Id').loc[a.Id];a['reference_z']=b.z.to_numpy()
            rows.append({'profile':name,**comparison(a,2000)})
            for date,g in a.groupby('date'):dates.append({'profile':name,'date':date,'RMSLE':float(np.sqrt(np.mean((g.z-g.target_z)**2)))})
        pd.DataFrame(rows).to_csv(args.out/'confirmation_scores.csv',index=False);pd.DataFrame(dates).to_csv(args.out/'per_date_scores.csv',index=False)
        print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':main()
